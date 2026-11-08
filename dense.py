"""稠密重建：相邻帧对 SGBM 立体匹配 + 多对点云融合。"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
from frame_cache import get_image
import numpy as np
from scipy.spatial import cKDTree

from poses import CameraIntrinsics, CameraPose
from exporter import write_ply

logger = logging.getLogger(__name__)


# ---- 异常 ----
class DenseAbortedError(RuntimeError):
    """用户中断；已算帧对已落盘，下次可续跑。"""
    pass


# ---- 配置与结果 ----
CACHE_VERSION = 3  # v3: 加入畸变模型，视差范围按深度计算


@dataclass
class DenseConfig:
    downscale: int = 2              # 降采样倍数，越大越快、精度越低
    max_pairs: int = 60             # 帧对数量上限
    num_disparities: int = 128      # 视差范围，必须为 16 的倍数
    block_size: int = 5             # 块大小，奇数 3~11
    min_baseline_ratio: float = 0.005   # 基线 / 场景尺度 下限
    max_baseline_ratio: float = 0.5     # 基线 / 场景尺度 上限
    min_depth_ratio: float = 0.02       # 最小深度 / 场景尺度
    max_depth_ratio: float = 5.0        # 最大深度 / 场景尺度
    voxel_ratio: float = 0.005          # 体素大小 / 场景尺度
    outlier_k: int = 8                  # 统计滤波近邻数
    outlier_std: float = 2.0            # 统计滤波标准差倍数
    max_points: int = 2_000_000         # 点数上限，超出随机采样
    min_disparity: float = 1.0          # 视差有效下限（像素）
    fusion_min_views: int = 1           # 体素融合所需最少帧对数（>1 去掉单对产生的点）
    enable_cache: bool = True           # 帧对级缓存
    cache_on_success: bool = False      # 正常完成也写缓存，默认只写中断


@dataclass
class DenseResult:
    output_path: str
    num_points: int
    num_pairs_total: int
    num_pairs_used: int
    num_pairs_failed: int
    num_pairs_cached: int
    scene_scale: float
    voxel_size: float
    elapsed_sec: float


# ---- 主入口 ----
def dense_reconstruct(
    frame_paths: List[str],
    intrinsics: CameraIntrinsics,
    poses: List[Optional[CameraPose]],
    output_path: str,
    *,
    keyframes: Optional[List[int]] = None,
    config: Optional[DenseConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cache_dir: Optional[str] = None,
    stop_event=None,
) -> DenseResult:
    """输出带 RGB 的 PLY 点云。"""
    t_start = time.time()
    cfg = config or DenseConfig()

    # ---- 位姿有效性 ----
    valid_idx = [i for i, p in enumerate(poses) if p is not None]
    if len(valid_idx) < 2:
        raise RuntimeError("稠密重建需要至少 2 个有效位姿。")

    # ---- 缓存开关 ----
    use_cache = bool(cfg.enable_cache and cache_dir is not None)
    cache_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    if use_cache:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        meta_path = cache_path / "meta.json"

    # ---- 场景尺度 ----
    scene_scale = _estimate_scene_scale(poses)
    voxel_size = max(scene_scale * cfg.voxel_ratio, 1e-4)
    min_depth = max(scene_scale * cfg.min_depth_ratio, 1e-3)
    max_depth = scene_scale * cfg.max_depth_ratio
    min_baseline = scene_scale * cfg.min_baseline_ratio
    max_baseline = scene_scale * cfg.max_baseline_ratio

    logger.info(
        "[dense] 场景尺度=%.4f, 体素=%.4f, 深度=[%.3f, %.3f], 基线=[%.3f, %.3f]",
        scene_scale, voxel_size, min_depth, max_depth, min_baseline, max_baseline,
    )

    # ---- 缓存指纹 ----
    # 含内参 / 位姿哈希，SfM 结果变化时缓存自动作废
    intr_poses_hash = _intr_poses_hash(intrinsics, poses)

    if use_cache:
        fingerprint = _cache_fingerprint(frame_paths, cfg, scene_scale,
                                         intr_poses_hash)
        if _load_and_check_meta(meta_path, fingerprint):
            n_existing = len(list(cache_path.glob("pair_*.npz")))
            logger.info("[dense] 缓存命中：参数指纹一致，已存在 %d 个帧对文件", n_existing)
        else:
            _clear_cache(cache_path)
            meta_path.write_text(
                json.dumps(fingerprint, indent=2, ensure_ascii=False)
            )
            logger.info("[dense] 参数指纹不匹配或缓存缺失，已重置缓存目录")

    # ---- 选帧对 ----
    pairs = _select_pairs(keyframes, len(frame_paths), cfg.max_pairs)
    logger.info("[dense] 帧对: %d 个（关键帧 %s）",
                len(pairs), "有" if keyframes else "无")
    if not pairs:
        raise RuntimeError("没有可用的帧对。")

    # ---- 主循环 ----
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    used = 0
    failed = 0
    cached_hits = 0

    pending_writes: List[Tuple[Path, Optional[np.ndarray], Optional[np.ndarray]]] = []

    def _flush_pending() -> None:
        if not pending_writes:
            return
        n = len(pending_writes)
        for p, xyz_p, rgb_p in pending_writes:
            _save_pair_cache(p, xyz_p, rgb_p)
        pending_writes.clear()
        logger.info("[dense] 已保存 %d 个帧对到缓存: %s", n, cache_path)

    try:
        for i, (il, ir) in enumerate(pairs):
            # ---- 中断检查 ----
            if stop_event is not None and stop_event.is_set():
                logger.info("[dense] 检测到停止信号，保存进度后退出")
                _flush_pending()
                raise DenseAbortedError("稠密重建被用户中断")

            pair_file = (cache_path / _pair_filename(il, ir)) if use_cache else None

            pl, pr = poses[il], poses[ir]
            if pl is None or pr is None:
                failed += 1
                _emit_progress(progress_callback, i, len(pairs),
                               f"帧对 {il} → {ir}（位姿缺失）")
                continue

            baseline = float(np.linalg.norm(pr.center - pl.center))
            if baseline < min_baseline or baseline > max_baseline:
                logger.debug("[dense] 跳过帧对 (%d,%d)：基线 %.4f 越界", il, ir, baseline)
                failed += 1
                if pair_file is not None:
                    pending_writes.append((pair_file, None, None))
                _emit_progress(progress_callback, i, len(pairs),
                               f"帧对 {il} → {ir}（基线越界）")
                continue

            # ---- 先查缓存 ----
            if pair_file is not None:
                xyz_c, rgb_c, state = _load_pair_cache(pair_file)
                if state == "hit":
                    all_xyz.append(xyz_c)
                    all_rgb.append(rgb_c)
                    used += 1
                    cached_hits += 1
                    _emit_progress(progress_callback, i, len(pairs),
                                   f"帧对 {il} → {ir}（缓存命中）")
                    continue
                elif state == "empty":
                    failed += 1
                    _emit_progress(progress_callback, i, len(pairs),
                                   f"帧对 {il} → {ir}（缓存标记失败）")
                    continue

            # ---- 实际计算 ----
            _emit_progress(progress_callback, i, len(pairs),
                           f"帧对 {il} → {ir}（计算中…）")

            try:
                out = _process_pair(
                    frame_paths[il], frame_paths[ir],
                    pl, pr, intrinsics, cfg,
                    min_depth=min_depth, max_depth=max_depth,
                )
            except Exception as e:
                logger.warning("[dense] 帧对 (%d,%d) 失败: %s", il, ir, e)
                failed += 1
                if pair_file is not None:
                    pending_writes.append((pair_file, None, None))
                continue

            if out is None or len(out[0]) == 0:
                failed += 1
                if pair_file is not None:
                    pending_writes.append((pair_file, None, None))
                continue

            xyz, rgb = out
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            used += 1

            if pair_file is not None:
                pending_writes.append((pair_file, xyz, rgb))
    except BaseException:
        try:
            _flush_pending()
        except Exception as fe:
            logger.warning("[dense] 异常路径 flush 失败: %s", fe)
        raise

    # ---- 正常完成 ----
    if progress_callback is not None:
        progress_callback(len(pairs), len(pairs), "合并中")

    if use_cache:
        if cfg.cache_on_success:
            _flush_pending()
        else:
            if pending_writes:
                logger.info(
                    "[dense] 正常完成，丢弃 %d 个待写帧对（cache_on_success=False）",
                    len(pending_writes),
                )
            pending_writes.clear()

    if not all_xyz:
        raise RuntimeError("稠密重建失败：没有产生任何点。")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    logger.info(
        "[dense] 原始点数: %d（成功 %d / %d 对，缓存命中 %d，失败 %d）",
        len(xyz), used, len(pairs), cached_hits, failed,
    )

    # ---- 体素降采样 + 帧对融合 ----
    _src = np.concatenate([np.full(len(a), i, dtype=np.int64)
                           for i, a in enumerate(all_xyz)]) if all_xyz else None
    xyz, rgb = _voxel_downsample(xyz, rgb, voxel_size,
                                 min_views=cfg.fusion_min_views, src=_src)
    logger.info("[dense] 体素降采样后: %d 点", len(xyz))

    # ---- 统计离群点过滤 ----
    if len(xyz) > cfg.outlier_k + 1:
        keep = _statistical_outlier_filter(xyz, cfg.outlier_k, cfg.outlier_std)
        n_before = len(xyz)
        xyz, rgb = xyz[keep], rgb[keep]
        logger.info("[dense] 离群点过滤: %d → %d", n_before, len(xyz))

    # ---- 点数上限 ----
    if len(xyz) > cfg.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), cfg.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        logger.info("[dense] 截断到上限 %d 点", cfg.max_points)

    # ---- 写 PLY ----
    write_ply(output_path, xyz, rgb)

    elapsed = time.time() - t_start
    logger.info("[dense] 完成: %s（%d 点，耗时 %.1fs）",
                output_path, len(xyz), elapsed)

    return DenseResult(
        output_path=output_path,
        num_points=int(len(xyz)),
        num_pairs_total=len(pairs),
        num_pairs_used=used,
        num_pairs_failed=failed,
        num_pairs_cached=cached_hits,
        scene_scale=float(scene_scale),
        voxel_size=float(voxel_size),
        elapsed_sec=elapsed,
    )


# ---- 帧对缓存 ----
def _pair_filename(il: int, ir: int) -> str:
    return f"pair_{int(il):04d}_{int(ir):04d}.npz"


def _intr_poses_hash(intrinsics: CameraIntrinsics,
                     poses: List[Optional[CameraPose]]) -> str:
    """内参 + 位姿哈希；位姿抽样到最多 30 个，保证任意序列长度耗时恒定。"""
    h = hashlib.md5()
    K = np.asarray(intrinsics.K, dtype=np.float32)
    h.update(K.tobytes())
    # 畸变模型直接影响稠密结果，必须进指纹
    if intrinsics.dist is not None:
        h.update(np.asarray(intrinsics.dist, dtype=np.float64).tobytes())

    n = len(poses)
    step = max(1, n // 30)
    for i in range(0, n, step):
        p = poses[i]
        if p is None:
            h.update(b"none")
        else:
            R = np.asarray(p.R, dtype=np.float32)
            t = np.asarray(p.t, dtype=np.float32).reshape(3)
            h.update(R.tobytes())
            h.update(t.tobytes())
    h.update(f"n={n}".encode())
    return h.hexdigest()


def _cache_fingerprint(
    frame_paths: List[str],
    cfg: DenseConfig,
    scene_scale: float,
    intr_poses_hash: str,
) -> dict:
    """影响输出几何的参数变化都会使缓存失效。"""
    nd = max(16, (int(cfg.num_disparities) // 16) * 16)
    bs = int(cfg.block_size)
    if bs % 2 == 0:
        bs += 1
    bs = max(3, min(bs, 21))

    try:
        first_mtime = int(os.path.getmtime(frame_paths[0]))
    except OSError:
        first_mtime = 0
    try:
        last_mtime = int(os.path.getmtime(frame_paths[-1]))
    except OSError:
        last_mtime = 0

    return {
        "version": CACHE_VERSION,
        "downscale": int(cfg.downscale),
        "num_disparities": nd,
        "block_size": bs,
        "min_disparity": round(float(cfg.min_disparity), 6),
        "scene_scale": round(float(scene_scale), 6),
        "n_frames": len(frame_paths),
        "first_frame_mtime": first_mtime,
        "last_frame_mtime": last_mtime,
        "intr_poses_hash": intr_poses_hash,
    }


def _load_and_check_meta(meta_path: Path, expected: dict) -> bool:
    """指纹一致才返回 True。"""
    if not meta_path.exists():
        return False
    try:
        old = json.loads(meta_path.read_text())
    except Exception:
        return False
    if not isinstance(old, dict):
        return False
    return old == expected


def _clear_cache(cache_path: Path) -> None:
    removed = 0
    for f in cache_path.glob("pair_*.npz"):
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            logger.warning("[dense] 删除缓存文件失败 %s: %s", f, e)
    if removed > 0:
        logger.info("[dense] 已清除 %d 个旧缓存文件", removed)


def _load_pair_cache(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if not path.exists():
        return None, None, "miss"
    try:
        with np.load(path) as data:
            xyz = np.asarray(data["xyz"])
            rgb = np.asarray(data["rgb"])
    except Exception as e:
        logger.debug("[dense] 读取缓存失败 %s: %s", path, e)
        # 删除损坏缓存，避免下次重复失败
        try:
            path.unlink()
        except OSError:
            pass
        return None, None, "miss"

    if xyz.shape[0] == 0:
        return None, None, "empty"
    if rgb.shape[0] != xyz.shape[0]:
        return None, None, "miss"
    return xyz.astype(np.float32), rgb.astype(np.uint8), "hit"


def _save_pair_cache(
    path: Path,
    xyz: Optional[np.ndarray],
    rgb: Optional[np.ndarray],
) -> None:
    """失败时写空数组，标记「已尝试且失败」以免重算。"""
    if xyz is None or rgb is None or len(xyz) == 0:
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
    else:
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.uint8)

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        np.savez(tmp, xyz=xyz, rgb=rgb)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("[dense] 写入缓存失败 %s: %s", path, e)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _emit_progress(cb, i: int, total: int, msg: str) -> None:
    if cb is not None:
        try:
            cb(i, total, msg)
        except Exception as e:
            logger.debug("[dense] progress_callback 异常: %s", e)


# ---- 帧对处理 ----
def _thin_prism_undistort_map(K: np.ndarray, params: np.ndarray,
                              w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """THIN_PRISM_FISHEYE 反向映射：针孔 → 等距鱼眼 → 径向/切向/薄棱镜 → 像素，
    按 thin_prism.h 正向公式生成 remap 网格（OpenCV 无此模型）。"""
    fx, fy, cx, cy = (float(params[0]), float(params[1]),
                      float(params[2]), float(params[3]))
    k1, k2, p1, p2, k3, k4, sx1, sy1 = (float(v) for v in params[4:12])

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    u = (xs - float(K[0, 2])) / float(K[0, 0])
    v = (ys - float(K[1, 2])) / float(K[1, 1])

    theta = np.sqrt(u * u + v * v)
    scale = np.ones_like(theta)
    nz = theta > 1e-12
    scale[nz] = np.arctan(theta[nz]) / theta[nz]
    uu = u * scale
    vv = v * scale

    r2 = uu * uu + vv * vv
    radial = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    du = (uu * radial + 2.0 * p1 * uu * vv + p2 * (r2 + 2.0 * uu * uu)
          + sx1 * r2)
    dv = (vv * radial + 2.0 * p2 * uu * vv + p1 * (r2 + 2.0 * vv * vv)
          + sy1 * r2)

    map_x = (fx * (uu + du) + cx).astype(np.float32)
    map_y = (fy * (vv + dv) + cy).astype(np.float32)
    return map_x, map_y


def _process_pair(
    path_l: str,
    path_r: str,
    pose_l: CameraPose,
    pose_r: CameraPose,
    intrinsics: CameraIntrinsics,
    cfg: DenseConfig,
    *,
    min_depth: float,
    max_depth: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """单帧对：立体校正 → SGBM → 反投影到世界系。"""
    # ---- 读图 + 降采样 ----
    img_l = get_image(path_l)
    img_r = get_image(path_r)
    if img_l is None or img_r is None:
        return None

    ds = max(1, int(cfg.downscale))
    if ds > 1:
        h, w = img_l.shape[:2]
        new_size = (max(2, w // ds), max(2, h // ds))
        img_l = cv2.resize(img_l, new_size, interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(img_r, new_size, interpolation=cv2.INTER_AREA)

    h, w = img_l.shape[:2]

    # ---- 缩放后的内参 ----
    K = np.asarray(intrinsics.K, dtype=np.float64).copy()
    K[0, 0] /= ds
    K[1, 1] /= ds
    K[0, 2] /= ds
    K[1, 2] /= ds

    # ---- 相对位姿 ----
    R_rel = pose_r.R @ pose_l.R.T
    t_rel = (pose_r.t - R_rel @ pose_l.t).reshape(3, 1)

    # ---- 去畸变：SfM 位姿基于反畸变点 ----
    dist = (None if intrinsics.dist is None
            else np.asarray(intrinsics.dist, dtype=np.float64).ravel())
    if dist is not None:
        if dist.size == 12:
            # THIN_PRISM_FISHEYE：OpenCV 无此模型，用自定义反向映射
            p = dist.copy()
            p[0] /= ds
            p[1] /= ds
            p[2] /= ds
            p[3] /= ds
            map_x, map_y = _thin_prism_undistort_map(K, p, w, h)
        else:
            map_x, map_y = cv2.initUndistortRectifyMap(
                K, dist, None, K, (w, h), cv2.CV_32FC1)
        img_l = cv2.remap(img_l, map_x, map_y, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, map_x, map_y, cv2.INTER_LINEAR)

    # ---- 基线滚转 ----
    # 相对位移以竖直为主时（如 ETH3D relief 的俯仰拍摄），stereoRectify 不会把
    # 基线转平，SGBM 水平视差搜索失效；故先绕光轴转 90°，位姿与内参同转。
    rolled = False
    if abs(float(t_rel[1, 0])) > abs(float(t_rel[0, 0])):
        rolled = True
        fx0, fy0, cx0, cy0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w_old, h_old = w, h
        if float(t_rel[1, 0]) > 0:
            # 使 t_x' = -t_y < 0（右相机在右，正视差）：Rz(+90)，顺时针
            R3 = np.array([[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
            img_l = cv2.rotate(img_l, cv2.ROTATE_90_CLOCKWISE)
            img_r = cv2.rotate(img_r, cv2.ROTATE_90_CLOCKWISE)
            K = np.array([[fy0, 0.0, (h_old - 1) - cy0],
                          [0.0, fx0, cx0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            # 使 t_x' = t_y < 0：Rz(-90)，逆时针
            R3 = np.array([[0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
            img_l = cv2.rotate(img_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_r = cv2.rotate(img_r, cv2.ROTATE_90_COUNTERCLOCKWISE)
            K = np.array([[fy0, 0.0, cy0],
                          [0.0, fx0, (w_old - 1) - cx0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        h, w = img_l.shape[:2]
        R_rel = R3 @ R_rel @ R3.T
        t_rel = (R3 @ t_rel.reshape(3)).reshape(3, 1)

    # ---- 立体校正 ----
    R1_r, R2_r, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, None, K, None, (w, h), R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        K, None, R1_r, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K, None, R2_r, P2, (w, h), cv2.CV_32FC1)

    rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

    # 直接对校正图取灰度，省一遍原图转换
    gray_l_r = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # ---- 视差范围：按深度换算，不写死 0/128 ----
    # 校正后 d = fx_rect * baseline / Z
    fx_rect = float(P1[0, 0])
    baseline = abs(float(P2[0, 3]) / fx_rect) if abs(fx_rect) > 1e-9 else 0.0
    if baseline > 1e-9:
        d_far = fx_rect * baseline / max(max_depth, 1e-9)
        d_near = fx_rect * baseline / max(min_depth, 1e-9)
        min_disp = int(max(0, np.floor(d_far) - 2))
        need = int(np.ceil(d_near - min_disp))
        num_disp = int(min(256, max(16, np.ceil(max(need, 16) / 16.0) * 16)))
    else:
        min_disp, num_disp = 0, int(cfg.num_disparities)

    # ---- SGBM 视差 ----
    disp = _compute_disparity(
        gray_l_r, gray_r_r, num_disp, cfg.block_size, min_disp,
    )
    if disp is None:
        return None

    # ---- 反投影到校正左相机系 ----
    pts3d_rect = cv2.reprojectImageTo3D(disp, Q)
    Z_rect = pts3d_rect[..., 2]

    valid = (
        np.isfinite(pts3d_rect).all(axis=2)
        & (disp >= max(float(cfg.min_disparity), float(min_disp)))
        & (Z_rect > min_depth)
        & (Z_rect < max_depth)
    )
    if not np.any(valid):
        return None

    # ---- 校正左系 → 原左系 → 世界系（仅有效像素）----
    ys, xs = np.nonzero(valid)
    pts_valid = pts3d_rect[ys, xs].astype(np.float64)

    R_L = np.asarray(pose_l.R, dtype=np.float64)
    t_L = pose_l.t.reshape(3).astype(np.float64)

    pts_cam_l = pts_valid @ R1_r              # 校正系 → 旋转后的左相机系
    if rolled:
        pts_cam_l = pts_cam_l @ R3            # 旋转系 → 原左相机系
    pts_world = (pts_cam_l - t_L) @ R_L       # 原左相机系 → 世界系
    xyz = pts_world.astype(np.float32)

    rgb = rect_l[ys, xs][:, ::-1].astype(np.uint8)   # BGR → RGB

    return xyz, rgb


# SGBM 实例缓存：参数相同则复用
_SGBM_CACHE: Dict[Tuple[int, int, int], "cv2.StereoSGBM"] = {}


def _get_sgbm(num_disparities: int, block_size: int,
              min_disparity: int = 0) -> "cv2.StereoSGBM":
    nd = max(16, (int(num_disparities) // 16) * 16)
    md = int(max(0, min_disparity))
    bs = int(block_size)
    if bs % 2 == 0:
        bs += 1
    bs = max(3, min(bs, 21))
    key = (nd, md, bs)
    sgbm = _SGBM_CACHE.get(key)
    if sgbm is None:
        sgbm = cv2.StereoSGBM_create(
            minDisparity=md,
            numDisparities=nd,
            blockSize=bs,
            P1=8 * bs * bs,
            P2=32 * bs * bs,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        _SGBM_CACHE[key] = sgbm
    return sgbm


def _compute_disparity(
    gray_l: np.ndarray,
    gray_r: np.ndarray,
    num_disparities: int,
    block_size: int,
    min_disparity: int = 0,
) -> Optional[np.ndarray]:
    """返回 float32 像素视差，含 minDisparity 偏移。"""
    try:
        stereo = _get_sgbm(num_disparities, block_size, min_disparity)
        disp16 = stereo.compute(gray_l, gray_r)
    except cv2.error as e:
        logger.warning("[dense] SGBM 失败: %s", e)
        return None

    disp = disp16.astype(np.float32) / 16.0
    return disp


# ---- 点云后处理 ----
def _estimate_scene_scale(poses: List[Optional[CameraPose]]) -> float:
    """包围盒对角线；不足 2 个位姿时返回 1.0。"""
    centers = np.array([p.center for p in poses if p is not None],
                       dtype=np.float64)
    if len(centers) < 2:
        return 1.0
    diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    return max(diag, 1e-6)


def _select_pairs(
    keyframes: Optional[List[int]],
    n_frames: int,
    max_pairs: int,
) -> List[Tuple[int, int]]:
    """优先关键帧相邻对，否则全部相邻对；超量均匀截断。"""
    if keyframes and len(keyframes) >= 2:
        kf = sorted(set(int(k) for k in keyframes if 0 <= k < n_frames))
        pairs = [(kf[i], kf[i + 1]) for i in range(len(kf) - 1)]
    else:
        pairs = [(i, i + 1) for i in range(n_frames - 1)]

    if len(pairs) > max_pairs:
        idx = np.round(np.linspace(0, len(pairs) - 1, max_pairs)).astype(int)
        idx = np.unique(idx)
        pairs = [pairs[i] for i in idx]
    return pairs


def _voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
    min_views: int = 1,
    src: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """每个体素取 xyz / rgb 均值。

    min_views > 1 时再按「命中该体素的不同帧对数」过滤一遍：同一个体素被多个
    帧对独立重建出来才算可信，能去掉单个帧对产生的重复点与边缘噪声。
    """
    if len(xyz) == 0:
        return xyz, rgb

    keys = np.floor(xyz / voxel_size).astype(np.int64)
    keys -= keys.min(axis=0)
    shape = keys.max(axis=0) + 1
    linear = (keys[:, 0] * shape[1] + keys[:, 1]) * shape[2] + keys[:, 2]

    uniq, inv = np.unique(linear, return_inverse=True)
    n = len(uniq)
    counts = np.bincount(inv, minlength=n).astype(np.float64)

    out_xyz = np.empty((n, 3), dtype=np.float64)
    out_rgb = np.empty((n, 3), dtype=np.float64)
    for c in range(3):
        out_xyz[:, c] = np.bincount(inv, weights=xyz[:, c].astype(np.float64),
                                    minlength=n) / counts
        out_rgb[:, c] = np.bincount(inv, weights=rgb[:, c].astype(np.float64),
                                    minlength=n) / counts

    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

    if min_views > 1 and src is not None and len(src) == len(inv):
        s = np.asarray(src, dtype=np.int64)
        span = int(s.max()) + 1
        # 每个（体素, 帧对）组合只算一次
        uniq_pair = np.unique(inv.astype(np.int64) * span + s)
        vox_of_pair = uniq_pair // span
        vox_ids, views = np.unique(vox_of_pair, return_counts=True)
        keep = np.zeros(n, dtype=bool)
        keep[vox_ids[views >= min_views]] = True
        if not np.any(keep):
            logger.warning("[dense] 没有体素满足 fusion_min_views=%d，跳过该过滤",
                           min_views)
        else:
            out_xyz = out_xyz[keep]
            out_rgb = out_rgb[keep]

    return out_xyz.astype(np.float32), out_rgb


def _statistical_outlier_filter(
    xyz: np.ndarray,
    k: int,
    std_ratio: float,
) -> np.ndarray:
    if len(xyz) < k + 1:
        return np.ones(len(xyz), dtype=bool)
    try:
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=k + 1, workers=-1)
    except Exception as e:
        logger.warning("[dense] KDTree 查询失败: %s，跳过过滤", e)
        return np.ones(len(xyz), dtype=bool)

    mean_dists = dists[:, 1:].mean(axis=1)
    mu = float(mean_dists.mean())
    sigma = float(mean_dists.std())
    threshold = mu + std_ratio * sigma
    return mean_dists < threshold