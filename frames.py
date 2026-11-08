"""从视频或图片集抽帧，支持均匀采样与两阶段智能采样。"""

import logging
import os
import re
import tempfile
import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 图片集输入支持的扩展名
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
}


# ---- 配置 ----
DEFAULT_OPTICAL_FLOW_METHOD = "farneback"  # 或 "lk"
LK_WINDOW_SIZE = (15, 15)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

# 打分采样上限，控制耗时
MAX_SCORE_SAMPLES = 100

# 打分缩放后的最大边长，保证跨视频分数可比
SCORE_MAX_SIDE = 480

# 门控分位数：剔除分数最低的该比例帧
SHARPNESS_QUANTILE = 0.05
FLOW_QUANTILE = 0.05

# 门控相对下限：阈值不低于中位数的该比例
SHARPNESS_MEDIAN_RATIO = 0.08
FLOW_MEDIAN_RATIO = 0.08

# 时间覆盖兜底：每箱至少保留 min_per_bin 个通过帧
TIME_COVERAGE_BINS = 20
TIME_COVERAGE_MIN_PER_BIN = 12

# 最终选择的分层段数，各段均分名额，避免低权重段被跳过
SELECT_BINS = 20

# 两阶段粗提取帧数
COARSE_FRAMES = 40


# ---- 公共入口 ----
def extract_frames(
    video_path: str,
    output_dir: str,
    *,
    fps: float = 15.0,
    scale: float = 0.5,
    min_frames: int = 30,
    max_frames: int = 200,
    smart_sampling: bool = False,
    two_stage: bool = False,
    poses_output_dir: Optional[str] = None,
    optical_flow_method: str = DEFAULT_OPTICAL_FLOW_METHOD,
    feature_type: str = "orb",
) -> List[str]:
    strategy = "two_stage" if two_stage else ("smart" if smart_sampling else "uniform")
    logger.info(
        "extract_frames 开始: strategy=%s, video=%s, output_dir=%s, "
        "fps=%.2f, scale=%.2f, frames=[%d, %d], flow=%s, feature=%s",
        strategy, video_path, output_dir,
        fps, scale, min_frames, max_frames,
        optical_flow_method, feature_type,
    )

    # 避免旧帧残留影响后续判断
    _clear_old_frames(output_dir)

    if two_stage:
        result = _two_stage_extract(
            video_path, output_dir, fps, scale,
            min_frames, max_frames,
            optical_flow_method, feature_type,
        )
    elif smart_sampling:
        result = _smart_extract(
            video_path, output_dir, fps, scale,
            min_frames, max_frames, optical_flow_method,
        )
    else:
        result = _uniform_extract(
            video_path, output_dir, fps, scale, min_frames, max_frames,
        )

    logger.info(
        "extract_frames 完成: strategy=%s, 保存 %d 帧 -> %s",
        strategy, len(result), output_dir,
    )
    return result


def _clear_old_frames(output_dir: str) -> None:
    if not os.path.isdir(output_dir):
        return
    n = 0
    for name in os.listdir(output_dir):
        if name.startswith("frame_") and name.endswith(".png"):
            try:
                os.unlink(os.path.join(output_dir, name))
                n += 1
            except OSError:
                pass
    if n > 0:
        logger.info("已清理 %d 个旧帧文件", n)


# ---- 图片集输入 ----
def _natural_key(path: str):
    parts = re.split(r"(\d+)", os.path.normcase(path))
    return [(1, int(p)) if p.isdigit() else (0, p) for p in parts]


def list_images(image_dir: str, *, recursive: bool = True) -> List[str]:
    """顶层无图片时递归查找，如 ETH3D 的 <scene>/images/dslr_images。"""
    root = os.path.abspath(image_dir)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"图片目录不存在: {image_dir}")

    def _scan(d: str, recurse: bool) -> List[str]:
        out: List[str] = []
        if recurse:
            for dirpath, dirnames, filenames in os.walk(d):
                dirnames.sort()
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                        out.append(os.path.join(dirpath, fn))
        else:
            try:
                names = os.listdir(d)
            except OSError as e:
                raise NotADirectoryError(f"无法读取图片目录: {image_dir}") from e
            for fn in names:
                p = os.path.join(d, fn)
                if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                    out.append(p)
        return out

    files = _scan(root, False)
    if files:
        files.sort(key=_natural_key)
        return files

    if not recursive:
        return []
    logger.info("顶层未找到图片，递归扫描子目录: %s", root)
    files = _scan(root, True)
    files.sort(key=lambda p: _natural_key(os.path.relpath(p, root)))
    return files


def exif_focal_ratio(image_path: str) -> Optional[float]:
    """从 EXIF 的 35mm 等效焦距推「fx / 图像宽度」，读不到返回 None。

    35mm 等效焦距按对角线定义，所以
        fx / width = f35 / 43.27 * sqrt(1 + (h/w)^2)
    与图像分辨率无关——缩放后的帧可以直接用同一个比值乘当前宽度。
    """
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            ex = im.getexif()
            # FocalLengthIn35mmFilm 在 Exif 子 IFD 里，部分写入器会放到顶层
            f35 = ex.get(41989)
            if not f35:
                f35 = ex.get_ifd(0x8769).get(41989)
            w, h = im.size
    except Exception:
        return None
    try:
        f35 = float(f35)
    except (TypeError, ValueError):
        return None
    if f35 <= 0 or w <= 0 or h <= 0:
        return None
    return f35 / 43.26661530556787 * (1.0 + (h / w) ** 2) ** 0.5


def import_images(
    image_dir: str,
    output_dir: str,
    *,
    scale: float = 1.0,
    max_frames: int = 0,
    recursive: bool = True,
) -> List[str]:
    """scale=1 时直接引用原图，避免大图片集占双份磁盘。"""
    all_files = list_images(image_dir, recursive=recursive)
    if not all_files:
        raise FileNotFoundError(f"图片目录中没有可用图片: {image_dir}")

    files = all_files
    if max_frames and max_frames > 0 and len(files) > max_frames:
        idx = np.unique(np.linspace(0, len(files) - 1, max_frames, dtype=int))
        files = [files[i] for i in idx]
        logger.info("图片集共 %d 张，按 max_frames 均匀抽取 %d 张",
                    len(all_files), len(files))

    if scale >= 0.999:
        logger.info("图片集共 %d 张，直接引用原图（scale=1.0，不复制）",
                    len(files))
        return files

    os.makedirs(output_dir, exist_ok=True)
    _clear_old_frames(output_dir)

    paths: List[str] = []
    skipped = 0
    out_size = None
    for i, src in enumerate(files):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            logger.warning("跳过无法解码的图片: %s", src)
            continue
        h, w = img.shape[:2]
        nw, nh = _compute_resized_size(w, h, scale, log=False)
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        if not cv2.imwrite(out_path, img):
            skipped += 1
            logger.warning("写入缩放帧失败: %s", out_path)
            continue
        if out_size is None:
            out_size = (nw, nh)
            logger.info("缩放尺寸: %dx%d (scale=%.2f) -> %dx%d",
                        w, h, scale, nw, nh)
        paths.append(out_path)

    logger.info(
        "图片集导入完成: %d 张 → %s（缩放 %.2f，输出 %s，跳过 %d 张）",
        len(paths), output_dir, scale,
        f"{out_size[0]}x{out_size[1]}" if out_size else "-", skipped,
    )
    return paths



# ---- 均匀采样 ----
def _uniform_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("无法打开视频: %s", video_path)
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = _compute_resized_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            scale,
        )
        num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)

        logger.info(
            "均匀采样: 总帧数=%d, 原始FPS=%.2f, 目标FPS=%.2f, 输出尺寸=%dx%d, 目标帧数=%d",
            total, orig_fps, fps, w, h, num_frames,
        )
        logger.info(
            "均匀采样索引范围: 首=%d, 末=%d, 去重后=%d",
            int(indices[0]), int(indices[-1]), int(np.unique(indices).size),
        )

        os.makedirs(output_dir, exist_ok=True)
        paths, _ = _extract_indices(cap, indices, output_dir, w, h)
        logger.info("均匀采样完成: 计划 %d 帧, 实际保存 %d 帧", num_frames, len(paths))
        return paths
    finally:
        cap.release()


# ---- 单阶段智能采样 ----
def _smart_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
    flow_method: str,
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("无法打开视频: %s", video_path)
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = _compute_resized_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            scale,
        )
        logger.info(
            "单阶段智能采样: 总帧数=%d, 原始FPS=%.2f, 目标FPS=%.2f, 输出尺寸=%dx%d, flow=%s",
            total, orig_fps, fps, w, h, flow_method,
        )
        return _smart_extract_from_cap(
            cap, output_dir, total, orig_fps, fps,
            min_frames, max_frames, w, h, flow_method,
        )
    finally:
        cap.release()


def _smart_extract_from_cap(
    cap: cv2.VideoCapture,
    output_dir: str,
    total: int,
    orig_fps: float,
    fps: float,
    min_frames: int,
    max_frames: int,
    w: int,
    h: int,
    flow_method: str,
) -> List[str]:
    """打分采样、门控、时间覆盖兜底后分层加权选择；通过帧不足时回退均匀采样。"""
    sample_indices = _make_sample_indices(total)
    logger.info("打分采样点: %d 个（上限 %d），首=%d, 末=%d",
                len(sample_indices), MAX_SCORE_SAMPLES,
                int(sample_indices[0]) if len(sample_indices) else -1,
                int(sample_indices[-1]) if len(sample_indices) else -1)

    sharpness, flow = _compute_gating_scores(cap, sample_indices, flow_method)
    sharp_full = _interp_to_full(sample_indices, sharpness, total)
    flow_full = _interp_to_full(sample_indices, flow, total)

    num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
    valid = _gating_mask(sharp_full, flow_full)
    n_gated = int(valid.sum())
    valid = _enforce_time_coverage(valid, sharp_full, flow_full, total)
    n_valid = int(valid.sum())

    logger.info(
        "门控结果: 通过=%d/%d (%.1f%%), 时间覆盖兜底后=%d (%.1f%%), 目标帧数=%d",
        n_gated, total, 100.0 * n_gated / max(1, total),
        n_valid, 100.0 * n_valid / max(1, total),
        num_frames,
    )

    if n_valid < min_frames:
        logger.warning(
            "通过门控的有效帧 %d < min_frames %d，回退到均匀采样。",
            n_valid, min_frames,
        )
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        target = min(num_frames, n_valid)
        # 段内按 清晰度 × 光流 加权
        score = sharp_full * flow_full
        weights = score * valid.astype(np.float64)
        if weights.sum() <= 1e-9:
            weights = valid.astype(np.float64)
        indices = _stratified_select(weights, valid, target, n_bins=SELECT_BINS)
        logger.info("时间分层选择: n_bins=%d, 目标=%d, 实际选中=%d",
                    SELECT_BINS, target, len(indices))

    os.makedirs(output_dir, exist_ok=True)
    paths, _ = _extract_indices(cap, indices, output_dir, w, h)
    logger.info("单阶段智能采样完成: 计划 %d 帧, 实际保存 %d 帧", len(indices), len(paths))
    return paths


# ---- 两阶段智能采样 ----
def _two_stage_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
    flow_method: str,
    feature_type: str,
) -> List[str]:
    from poses import estimate_poses  # 延迟导入，避免循环依赖

    logger.info("两阶段智能采样: 开始, video=%s, feature=%s, flow=%s",
                video_path, feature_type, flow_method)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("无法打开视频: %s", video_path)
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = _compute_resized_size(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        scale,
    )

    logger.info(
        "两阶段: 总帧数=%d, 原始FPS=%.2f, 目标FPS=%.2f, 输出尺寸=%dx%d",
        total, orig_fps, fps, w, h,
    )

    coarse_dir = tempfile.mkdtemp(prefix="coarse_")
    logger.info("粗提取临时目录: %s", coarse_dir)

    try:
        # ---- 阶段 1：粗提取 ----
        coarse_raw = np.linspace(0, total - 1, min(COARSE_FRAMES, total), dtype=int)
        coarse_paths, coarse_indices = _extract_indices(cap, coarse_raw, coarse_dir, w, h)
        logger.info("阶段 1: 粗提取 %d 帧（计划 %d 帧）",
                    len(coarse_paths), len(coarse_raw))

        if len(coarse_paths) < 2:
            logger.warning("粗提取帧数不足（%d < 2），回退到单阶段智能采样。",
                           len(coarse_paths))
            return _smart_extract_from_cap(
                cap, output_dir, total, orig_fps, fps,
                min_frames, max_frames, w, h, flow_method,
            )

        try:
            # 粗位姿只用于视差加权，禁掉回环/PGO
            coarse_result = estimate_poses(
            coarse_paths, feature_type,
            enable_loop=False,
            enable_pgo=False,
            enable_global_matching=False,        # 粗位姿用不到
            )
            coarse_poses = coarse_result.poses
        except Exception as e:
            logger.warning("粗位姿估计失败: %s。回退到单阶段智能采样。", e)
            return _smart_extract_from_cap(
                cap, output_dir, total, orig_fps, fps,
                min_frames, max_frames, w, h, flow_method,
            )

        n_pose_ok = sum(1 for p in coarse_poses if p is not None)
        logger.info("阶段 1: 粗位姿估计成功 %d/%d 帧",
                    n_pose_ok, len(coarse_poses))

        # ---- 阶段 2：打分与选择 ----
        sample_indices = _make_sample_indices(total)
        logger.info("阶段 2: 打分采样点 %d 个", len(sample_indices))

        sharpness, flow = _compute_gating_scores(cap, sample_indices, flow_method)
        sharp_full = _interp_to_full(sample_indices, sharpness, total)
        flow_full = _interp_to_full(sample_indices, flow, total)
        parallax_full = _compute_parallax_scores(total, coarse_indices, coarse_poses)

        num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
        valid = _gating_mask(sharp_full, flow_full)
        n_gated = int(valid.sum())
        valid = _enforce_time_coverage(valid, sharp_full, flow_full, total)
        n_valid = int(valid.sum())

        logger.info(
            "阶段 2: 门控通过=%d/%d (%.1f%%), 时间覆盖兜底后=%d (%.1f%%), 目标帧数=%d",
            n_gated, total, 100.0 * n_gated / max(1, total),
            n_valid, 100.0 * n_valid / max(1, total),
            num_frames,
        )

        if n_valid < min_frames:
            logger.warning(
                "通过门控的有效帧 %d < min_frames %d，回退到均匀采样。",
                n_valid, min_frames,
            )
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
        else:
            # 清晰度 × 光流 × (0.3 + 0.7×视差)：低视差帧保留基础权重，
            # 占比低但不会被完全跳过
            score = sharp_full * flow_full
            weight_parallax = 0.3 + 0.7 * parallax_full
            weights = score * weight_parallax * valid.astype(np.float64)
            if weights.sum() <= 1e-9:
                logger.warning("加权分数全为 0，退化为均匀权重。")
                weights = valid.astype(np.float64)
            target = min(num_frames, n_valid)
            indices = _stratified_select(weights, valid, target, n_bins=SELECT_BINS)
            logger.info("阶段 2: 分层选择 n_bins=%d, 目标=%d, 实际选中=%d",
                        SELECT_BINS, target, len(indices))

        os.makedirs(output_dir, exist_ok=True)
        paths, _ = _extract_indices(cap, indices, output_dir, w, h)
        logger.info("两阶段智能采样完成: 计划 %d 帧, 实际保存 %d 帧",
                    len(indices), len(paths))
        return paths
    finally:
        cap.release()
        shutil.rmtree(coarse_dir, ignore_errors=True)
        logger.info("已清理粗提取临时目录: %s", coarse_dir)


# ---- 时间分层选择 ----
def _stratified_select(
    weights: np.ndarray,
    valid: np.ndarray,
    num_frames: int,
    n_bins: int = SELECT_BINS,
) -> np.ndarray:
    n = len(weights)
    if n == 0 or num_frames <= 0:
        logger.info("分层选择跳过: n=%d, num_frames=%d", n, num_frames)
        return np.array([], dtype=int)
    num_frames = min(num_frames, int(valid.sum()))
    if num_frames <= 0:
        logger.info("分层选择跳过: 有效帧为 0")
        return np.array([], dtype=int)

    n_bins = max(1, min(n_bins, num_frames))

    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    per_bin = num_frames // n_bins
    extra = num_frames % n_bins

    selected: List[int] = []
    empty_bins = 0
    zero_weight_bins = 0
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        k = per_bin + (1 if b < extra else 0)
        k = min(k, hi - lo)
        if k <= 0:
            continue

        seg_valid = valid[lo:hi]
        if not seg_valid.any():
            empty_bins += 1
            continue

        seg_w = weights[lo:hi].copy()
        seg_w[~seg_valid] = 0.0

        if seg_w.sum() <= 1e-9:
            zero_weight_bins += 1
            valid_local = np.where(seg_valid)[0]
            take = min(k, len(valid_local))
            idx_local = np.round(
                np.linspace(0, len(valid_local) - 1, take)
            ).astype(int)
            seg_indices = valid_local[idx_local]
        else:
            seg_indices = _deterministic_select(seg_w, k)

        selected.extend((seg_indices + lo).tolist())

    result = np.asarray(sorted(set(selected)), dtype=int)
    logger.info(
        "分层选择: n_bins=%d, 目标=%d, 选中=%d, 空段=%d, 零权重段=%d",
        n_bins, num_frames, len(result), empty_bins, zero_weight_bins,
    )
    return result


# ---- 清晰度与光流打分 ----
def _compute_gating_scores(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    flow_method: str,
    score_max_side: int = SCORE_MAX_SIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """单次遍历同时算清晰度与光流，避免重复读帧。"""
    n = len(indices)
    sharpness = np.zeros(n, dtype=np.float64)
    flow = np.zeros(n, dtype=np.float64)
    if n == 0:
        logger.info("打分跳过: 采样点为空")
        return sharpness, flow

    method = flow_method.lower()
    prev_gray = None
    prev_idx = None
    read_fail = 0

    idx_arr = np.asarray(indices, dtype=np.int64)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_arr[0]))
    cur_frame = int(idx_arr[0])

    for k in range(n):
        target = int(idx_arr[k])
        if target < cur_frame:
            read_fail += 1
            logger.info("打分跳过（索引倒退）: 帧索引=%d", target)
            prev_gray = None
            prev_idx = None
            continue

        reached = True
        while cur_frame < target:
            ok, _ = cap.read()
            if not ok:
                reached = False
                break
            cur_frame += 1
        if not reached:
            read_fail += 1
            logger.info("打分推进失败: 帧索引=%d", target)
            prev_gray = None
            prev_idx = None
            continue

        ok, bgr = cap.read()
        if not ok:
            read_fail += 1
            logger.info("打分读取失败: 帧索引=%d", target)
            prev_gray = None
            prev_idx = None
            continue
        cur_frame += 1

        h0, w0 = bgr.shape[:2]
        s = score_max_side / max(h0, w0)
        if s < 1.0:
            bgr = cv2.resize(
                bgr,
                (max(2, int(w0 * s)), max(2, int(h0 * s))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        sharpness[k] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if prev_gray is not None:
            if method == "lk":
                mag = _lk_flow_magnitude(prev_gray, gray)
            else:
                f = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, **FARNEBACK_PARAMS,
                )
                mag = float(np.mean(np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)))
            gap = max(1, target - prev_idx) if prev_idx is not None else 1
            flow[k] = mag / gap
        prev_gray = gray
        prev_idx = target

    if n >= 2 and flow[0] == 0.0 and flow[1] > 0.0:
        flow[0] = flow[1]

    logger.info(
        "打分完成: 采样=%d, 读取失败=%d, 清晰度[中位=%.3f, 最大=%.3f], "
        "光流[中位=%.4f, 最大=%.4f]",
        n, read_fail,
        float(np.median(sharpness)), float(sharpness.max()) if n else 0.0,
        float(np.median(flow)), float(flow.max()) if n else 0.0,
    )
    return sharpness, flow


def _gating_mask(sharp_full: np.ndarray, flow_full: np.ndarray) -> np.ndarray:
    sharp_q = float(np.quantile(sharp_full, SHARPNESS_QUANTILE))
    flow_q = float(np.quantile(flow_full, FLOW_QUANTILE))
    sharp_med = float(np.median(sharp_full))
    flow_med = float(np.median(flow_full))

    sharp_thresh = max(sharp_q, SHARPNESS_MEDIAN_RATIO * sharp_med)
    flow_thresh = max(flow_q, FLOW_MEDIAN_RATIO * flow_med)
    mask = (sharp_full >= sharp_thresh) & (flow_full >= flow_thresh)

    logger.info(
        "门控阈值: sharpness>=%.4f (q=%.4f, med=%.4f), "
        "flow>=%.4f (q=%.4f, med=%.4f)",
        sharp_thresh, sharp_q, sharp_med,
        flow_thresh, flow_q, flow_med,
    )
    return mask


def _enforce_time_coverage(
    valid: np.ndarray,
    sharp_full: np.ndarray,
    flow_full: np.ndarray,
    total: int,
    n_bins: int = TIME_COVERAGE_BINS,
    min_per_bin: int = TIME_COVERAGE_MIN_PER_BIN,
) -> np.ndarray:
    valid = valid.copy()
    if total <= 0:
        return valid
    edges = np.linspace(0, total, n_bins + 1, dtype=int)
    touched = 0
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        if int(valid[lo:hi].sum()) >= min_per_bin:
            continue
        seg_score = sharp_full[lo:hi] * flow_full[lo:hi]
        k = min(min_per_bin, hi - lo)
        top = np.argsort(seg_score)[-k:] + lo
        valid[top] = True
        touched += 1

    if touched > 0:
        logger.info("时间覆盖兜底: 共补足 %d 个分箱（每箱至少 %d 帧）",
                    touched, min_per_bin)
    else:
        logger.info("时间覆盖兜底: 无需补足")
    return valid


def _lk_flow_magnitude(prev: np.ndarray, curr: np.ndarray) -> float:
    h, w = prev.shape
    step = 16
    y_coords = np.arange(0, h, step, dtype=np.float32)
    x_coords = np.arange(0, w, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, pts, None,
        winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA,
    )
    if status is None:
        logger.info("LK 光流: status 为 None，返回 0")
        return 0.0
    valid = status.ravel() == 1
    if not valid.any():
        logger.info("LK 光流: 无有效跟踪点，返回 0")
        return 0.0
    disp = (next_pts[valid] - pts[valid]).reshape(-1, 2)
    return float(np.mean(np.sqrt(np.sum(disp ** 2, axis=1))))


# ---- 视差打分 ----
def _compute_parallax_scores(
    total: int,
    coarse_indices: np.ndarray,
    coarse_poses: List,
) -> np.ndarray:
    """用相机中心距离（真实基线）作视差代理，避免 t 向量受旋转影响；另加旋转角小幅补偿。"""
    valid_pairs = [(i, p) for i, p in enumerate(coarse_poses) if p is not None]
    if len(valid_pairs) < 2:
        logger.warning("有效粗位姿不足（%d < 2），视差分数置零。", len(valid_pairs))
        return np.zeros(total, dtype=np.float64)

    coarse_scores = np.zeros(len(coarse_indices), dtype=np.float64)
    for k in range(1, len(valid_pairs)):
        i1, p1 = valid_pairs[k - 1]
        i2, p2 = valid_pairs[k]
        baseline = float(np.linalg.norm(p2.center - p1.center))
        R_rel = np.asarray(p2.R) @ np.asarray(p1.R).T
        angle = _rotation_angle(R_rel)
        coarse_scores[i2] = baseline + 0.1 * np.radians(angle)

    parallax = np.interp(np.arange(total), coarse_indices, coarse_scores)
    parallax = _gaussian_smooth(parallax, sigma=2.0)
    if parallax.max() > 1e-9:
        parallax = parallax / parallax.max()

    logger.info(
        "视差分数: 有效位姿对=%d, 归一化后[中位=%.4f, 最大=%.4f]",
        len(valid_pairs) - 1,
        float(np.median(parallax)), float(parallax.max()) if parallax.size else 0.0,
    )
    return parallax


def _rotation_angle(R: np.ndarray) -> float:
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv))


# ---- 采样辅助 ----
def _make_sample_indices(total: int) -> np.ndarray:
    step = _sample_step(total)
    indices = np.arange(0, total, step, dtype=int)
    if indices.size == 0:
        indices = np.array([0], dtype=int)
    if indices[-1] != total - 1:
        indices = np.append(indices, total - 1)
    logger.info("构造采样索引: total=%d, step=%d, n=%d", total, step, len(indices))
    return indices


def _sample_step(total: int) -> int:
    return max(1, total // MAX_SCORE_SAMPLES)


def _interp_to_full(sample_indices: np.ndarray, sample_scores: np.ndarray, total: int) -> np.ndarray:
    sample_indices = np.asarray(sample_indices)
    sample_scores = np.asarray(sample_scores, dtype=np.float64)
    if sample_indices.size == 0:
        return np.zeros(total, dtype=np.float64)
    if sample_indices.size == 1:
        return np.full(total, float(sample_scores[0]), dtype=np.float64)
    return np.interp(np.arange(total), sample_indices, sample_scores)


def _gaussian_smooth(scores: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """平滑 1D 分数，避免孤立峰值主导分配。"""
    if len(scores) < 3 or sigma <= 0:
        return scores
    size = int(4 * sigma + 1) | 1  # 保证奇数
    kernel = cv2.getGaussianKernel(size, sigma).reshape(-1)
    pad = size // 2
    padded = np.pad(scores, pad, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(scores)]


def _deterministic_select(weights: np.ndarray, num_frames: int) -> np.ndarray:
    """确定性选择，保证同输入同结果。"""
    n = len(weights)
    num_frames = min(num_frames, n)
    if num_frames >= n:
        return np.arange(n)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = w.sum()
    if total <= 0.0:
        return np.linspace(0, n - 1, num_frames, dtype=int)
    cum = np.cumsum(w) / total
    targets = (np.arange(num_frames) + 0.5) / num_frames
    indices = np.searchsorted(cum, targets)
    indices = np.clip(indices, 0, n - 1)
    return np.unique(indices)


def _target_frame_count(
    total: int,
    orig_fps: float,
    target_fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    raw = int(total * target_fps / orig_fps)
    n = min(max(raw, min_frames), max_frames)
    result = max(1, min(n, total))
    logger.info(
        "目标帧数计算: total=%d, orig_fps=%.2f, target_fps=%.2f, "
        "raw=%d, clamp[%d, %d] -> %d",
        total, orig_fps, target_fps, raw, min_frames, max_frames, result,
    )
    return result


def _compute_resized_size(orig_w: int, orig_h: int, scale: float,
                          log: bool = True) -> Tuple[int, int]:
    """宽高取偶数，避免后续奇数尺寸的边界问题。"""
    w = int(orig_w * scale)
    h = int(orig_h * scale)
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    result = (max(w, 2), max(h, 2))
    if log:
        logger.info("缩放尺寸: %dx%d (scale=%.2f) -> %dx%d",
                    orig_w, orig_h, scale, result[0], result[1])
    return result


def _extract_indices(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    output_dir: str,
    w: int,
    h: int,
) -> Tuple[List[str], np.ndarray]:
    paths: List[str] = []
    used: List[int] = []
    read_fail = 0
    idx_arr = np.asarray(indices, dtype=np.int64)
    if idx_arr.size == 0:
        return paths, np.asarray(used, dtype=int)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_arr[0]))
    cur_frame = int(idx_arr[0])

    for i, target in enumerate(idx_arr):
        target = int(target)
        if target < cur_frame:
            read_fail += 1
            logger.info("抽帧跳过（索引倒退）: 输出序号=%d, 帧索引=%d", i, target)
            continue

        reached = True
        while cur_frame < target:
            ok, _ = cap.read()
            if not ok:
                reached = False
                break
            cur_frame += 1
        if not reached:
            read_fail += 1
            logger.info("抽帧推进失败: 输出序号=%d, 帧索引=%d", i, target)
            continue

        ok, bgr = cap.read()
        if not ok:
            read_fail += 1
            logger.info("抽帧读取失败: 输出序号=%d, 帧索引=%d", i, target)
            continue
        cur_frame += 1

        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        fname = f"frame_{i:04d}.png"
        fpath = os.path.join(output_dir, fname)
        if not cv2.imwrite(fpath, resized):
            logger.error("无法写入帧图像: %s", fpath)
            raise OSError(f"无法写入帧图像: {fpath}")
        paths.append(os.path.abspath(fpath))
        used.append(target)

    if read_fail > 0:
        logger.warning("抽帧: 请求 %d 帧, 读取失败 %d 帧, 成功 %d 帧",
                       len(indices), read_fail, len(paths))
    else:
        logger.info("抽帧: 请求 %d 帧, 全部成功", len(indices))
    return paths, np.asarray(used, dtype=int)