"""
稠密重建模块：基于相邻帧对的双目立体匹配（SGBM）+ 多对融合。

输入：SfM 输出的 frame_paths + 内参 + 位姿
输出：稠密点云 PLY（带 RGB）

流程：
  1. 估计场景尺度（相机中心分布），据此设定深度范围 / 体素大小
  2. 选帧对（关键帧相邻对优先，否则全部相邻对，均匀截断到 max_pairs）
  3. 每对：立体校正（stereoRectify + remap）→ SGBM 视差 → 反投影 → 世界坐标
  4. 深度范围 + 基线比例过滤
  5. 所有对点云合并
  6. 体素降采样 + 统计离群点过滤
  7. 写带颜色的二进制 PLY

帧对级缓存（延后写盘版）：
  · 主循环把已算完的帧对结果累积在内存，不实时落盘。
  · 正常完成时不写缓存（避免无谓的磁盘写入）。
  · 仅在「用户主动中断」或「配置要求」时，一次性批量写盘。
  · 下次启动时读 cache_dir/meta.json 校验参数指纹，一致则复用已落盘的帧对。

纯 OpenCV + SciPy + NumPy，无外部 SfM/MVS 依赖。
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from poses import CameraIntrinsics, CameraPose
from exporter import write_ply

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class DenseAbortedError(RuntimeError):
    """稠密重建被用户中断（已算帧对已落盘，可下次续跑）。"""
    pass


# ---------------------------------------------------------------------------
# 配置与结果
# ---------------------------------------------------------------------------

CACHE_VERSION = 1


@dataclass
class DenseConfig:
    """稠密重建参数。"""
    downscale: int = 2              # 帧降采样倍数（越大越快，精度越低）
    max_pairs: int = 60             # 最多处理的帧对数量
    num_disparities: int = 128      # SGBM 视差范围，必须为 16 的倍数
    block_size: int = 5             # SGBM 块大小，奇数 3~11
    min_baseline_ratio: float = 0.005   # 基线 / 场景尺度 下限
    max_baseline_ratio: float = 0.5     # 基线 / 场景尺度 上限
    min_depth_ratio: float = 0.02       # 最小深度 / 场景尺度
    max_depth_ratio: float = 5.0        # 最大深度 / 场景尺度
    voxel_ratio: float = 0.005          # 体素大小 / 场景尺度
    outlier_k: int = 8                  # 统计滤波近邻数
    outlier_std: float = 2.0            # 统计滤波标准差倍数
    max_points: int = 2_000_000         # 输出点数上限（超出则随机采样）
    min_disparity: float = 1.0          # 视差有效下限（像素）
    enable_cache: bool = True           # 启用帧对级缓存
    cache_on_success: bool = False      # 正常完成时也写缓存（默认只在中断时写）


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


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

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
    """从 SfM 结果做稠密重建，输出带 RGB 的点云 PLY。

    参数：
        frame_paths: 帧图像路径列表，与 poses 一一对应。
        intrinsics: 相机内参。
        poses: 相机位姿列表，None 表示该帧位姿缺失。
        output_path: 输出 PLY 路径。
        keyframes: 关键帧索引列表（可选，优先用于选帧对）。
        config: 稠密重建参数，None 用默认。
        progress_callback: (已完成对数, 总对数, 描述) 回调，可选。
        cache_dir: 帧对缓存目录；为 None 或 config.enable_cache=False 时不启用。
        stop_event: 可选的中断信号（threading.Event 或兼容对象）。
                    置位后：把内存里已算完的帧对批量落盘，抛 DenseAbortedError。

    返回：
        DenseResult，含统计信息。

    异常：
        DenseAbortedError：用户中断。已算帧对已写入缓存。
        RuntimeError：无法产生任何点，或有效位姿不足。
    """
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
    if use_cache:
        fingerprint = _cache_fingerprint(frame_paths, cfg, scene_scale)
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

    # 内存中累积的待写缓存：[(pair_path, xyz, rgb)]
    # xyz / rgb 为 None 表示该对失败（写空数组作标记）
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
        # 任何异常（含 DenseAbortedError、KeyboardInterrupt）：
        # 已算的帧对都要保住，flush 后再往上抛
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
            # 默认不写缓存，直接丢弃（省磁盘写入）
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

    # ---- 体素降采样 ----
    xyz, rgb = _voxel_downsample(xyz, rgb, voxel_size)
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


# ---------------------------------------------------------------------------
# 帧对缓存
# ---------------------------------------------------------------------------

def _pair_filename(il: int, ir: int) -> str:
    return f"pair_{int(il):04d}_{int(ir):04d}.npz"


def _cache_fingerprint(
    frame_paths: List[str],
    cfg: DenseConfig,
    scene_scale: float,
) -> dict:
    """生成参数指纹；任意影响输出几何的项变化都会使缓存失效。"""
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
    }


def _load_and_check_meta(meta_path: Path, expected: dict) -> bool:
    """读 meta.json 与期望指纹比对；一致返回 True。"""
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
    """删除缓存目录下所有 pair_*.npz 文件。"""
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
    """读帧对缓存。

    返回 (xyz, rgb, state)：
        state = 'hit'   → 读到有效点云
        state = 'empty' → 该对曾算过但失败（空数组）
        state = 'miss'  → 文件不存在或损坏，应重算
    """
    if not path.exists():
        return None, None, "miss"
    try:
        with np.load(path) as data:
            xyz = np.asarray(data["xyz"])
            rgb = np.asarray(data["rgb"])
    except Exception as e:
        logger.debug("[dense] 读取缓存失败 %s: %s", path, e)
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
    """写帧对缓存。失败时写空数组以标记「已尝试且失败」。

    先写 .tmp 再原子替换，避免中断留下半截 npz。
    """
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


# ---------------------------------------------------------------------------
# 帧对处理：立体校正 + SGBM + 反投影
# ---------------------------------------------------------------------------

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
    """处理一对图像：立体校正 → SGBM → 反投影到世界坐标。

    返回 (xyz_world (N,3) float32, rgb (N,3) uint8) 或 None。
    """
    # ---- 读图 + 降采样 ----
    img_l = cv2.imread(path_l, cv2.IMREAD_COLOR)
    img_r = cv2.imread(path_r, cv2.IMREAD_COLOR)
    if img_l is None or img_r is None:
        return None

    ds = max(1, int(cfg.downscale))
    if ds > 1:
        h, w = img_l.shape[:2]
        new_size = (max(2, w // ds), max(2, h // ds))
        img_l = cv2.resize(img_l, new_size, interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(img_r, new_size, interpolation=cv2.INTER_AREA)

    h, w = img_l.shape[:2]
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # ---- 缩放后的内参 ----
    K = np.asarray(intrinsics.K, dtype=np.float64).copy()
    K[0, 0] /= ds
    K[1, 1] /= ds
    K[0, 2] /= ds
    K[1, 2] /= ds

    # ---- 相对位姿 ----
    R_rel = pose_r.R @ pose_l.R.T
    t_rel = (pose_r.t - R_rel @ pose_l.t).reshape(3, 1)

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
    gray_l_r = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # ---- SGBM 视差 ----
    disp = _compute_disparity(
        gray_l_r, gray_r_r, cfg.num_disparities, cfg.block_size,
    )
    if disp is None:
        return None

    # ---- 反投影到校正左相机系 ----
    pts3d_rect = cv2.reprojectImageTo3D(disp, Q)
    Z_rect = pts3d_rect[..., 2]

    valid = (
        np.isfinite(pts3d_rect).all(axis=2)
        & (disp >= cfg.min_disparity)
        & (Z_rect > min_depth)
        & (Z_rect < max_depth)
    )
    if not np.any(valid):
        return None

    # ---- 校正左系 → 原左系 → 世界系（只对有效像素做变换）----
    ys, xs = np.nonzero(valid)
    pts_valid = pts3d_rect[ys, xs].astype(np.float64)  # (M, 3)

    R_L = np.asarray(pose_l.R, dtype=np.float64)
    t_L = pose_l.t.reshape(3).astype(np.float64)

    pts_cam_l = pts_valid @ R1_r              # 校正系 → 原左系
    pts_world = (pts_cam_l - t_L) @ R_L       # 原左系 → 世界系
    xyz = pts_world.astype(np.float32)

    rgb = rect_l[ys, xs][:, ::-1].astype(np.uint8)   # BGR → RGB

    return xyz, rgb


def _compute_disparity(
    gray_l: np.ndarray,
    gray_r: np.ndarray,
    num_disparities: int,
    block_size: int,
) -> Optional[np.ndarray]:
    """SGBM 视差计算，返回 float32（像素单位）。"""
    nd = max(16, (int(num_disparities) // 16) * 16)
    bs = int(block_size)
    if bs % 2 == 0:
        bs += 1
    bs = max(3, min(bs, 21))

    try:
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
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
        disp16 = stereo.compute(gray_l, gray_r)
    except cv2.error as e:
        logger.warning("[dense] SGBM 失败: %s", e)
        return None

    disp = disp16.astype(np.float32) / 16.0
    return disp


# ---------------------------------------------------------------------------
# 点云后处理
# ---------------------------------------------------------------------------

def _estimate_scene_scale(poses: List[Optional[CameraPose]]) -> float:
    """用相机中心分布的包围盒对角线估计场景尺度。"""
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
    """选帧对：优先关键帧相邻对，否则全部相邻对，均匀截断。"""
    if keyframes and len(keyframes) >= 2:
        kf = sorted(set(int(k) for k in keyframes if 0 <= k < n_frames))
        pairs = [(kf[i], kf[i + 1]) for i in range(len(kf) - 1)]
    else:
        pairs = [(i, i + 1) for i in range(n_frames - 1)]

    if len(pairs) > max_pairs:
        # 用 round 而非截断，避免间距 < 1 时取到重复索引
        idx = np.round(np.linspace(0, len(pairs) - 1, max_pairs)).astype(int)
        idx = np.unique(idx)
        pairs = [pairs[i] for i in idx]
    return pairs


def _voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """体素降采样：每个体素内取 xyz / rgb 均值。"""
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
    return out_xyz.astype(np.float32), out_rgb


def _statistical_outlier_filter(
    xyz: np.ndarray,
    k: int,
    std_ratio: float,
) -> np.ndarray:
    """统计离群点过滤：剔除近邻平均距离过大的点。"""
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