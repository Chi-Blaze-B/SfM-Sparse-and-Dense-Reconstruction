"""frame_meta.json 读写与参数校验。"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

# 与抽帧结果强相关，改动会使下游（位姿 / 点云）缓存失效
FIELDS = (
    "source_type",   # "video" | "images"，用于识别输入方式切换
    "video",
    "image_dir",
    "scale",
    "fps",
    "sampling_mode",
    "feature_type",
    "min_frames",
    "max_frames",
)

# 需要按绝对路径比较的字段
_PATH_FIELDS = ("video", "image_dir")


def write_meta(path: Path, cfg: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False))


def check_meta(meta_path: Path, cfg: Dict[str, Any]) -> List[str]:
    """返回不一致项；空列表表示一致或无法判定。"""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return []
    try:
        old = json.loads(meta_path.read_text())
    except Exception:
        return []
    if not isinstance(old, dict):
        return []

    reasons: List[str] = []
    for key in FIELDS:
        o = old.get(key)
        n = cfg.get(key)
        if o is None or n is None:
            continue
        if key in _PATH_FIELDS:
            if os.path.abspath(str(o)) != os.path.abspath(str(n)):
                reasons.append(
                    f"{key} {os.path.basename(str(o))} -> "
                    f"{os.path.basename(str(n))}"
                )
        elif isinstance(n, float):
            try:
                if abs(float(o) - float(n)) >= 1e-6:
                    reasons.append(f"{key} {o} → {n}")
            except (TypeError, ValueError):
                reasons.append(f"{key} {o} → {n}")
        else:
            if o != n:
                reasons.append(f"{key} {o} → {n}")
    return reasons


def can_reuse_frames(workdir: Path) -> bool:
    """校验 frame_paths.txt 中每个路径仍存在，兼容视频与图片集（scale=1）两种模式。"""
    workdir = Path(workdir)
    frame_paths_file = workdir / "frame_paths.txt"
    if not frame_paths_file.exists():
        return False
    try:
        paths = [p.strip() for p in frame_paths_file.read_text().splitlines()
                 if p.strip()]
    except OSError:
        return False
    if not paths:
        return False
    return all(os.path.isfile(p) for p in paths)


def invalidate_downstream(workdir: Path) -> List[str]:
    """帧变化会使稠密结果不可信，一并清理。"""
    removed: List[str] = []
    workdir = Path(workdir)
    for name in ("intrinsics.npy", "poses.npy", "sparse_points.npy",
                 "sfm_meta.json"):
        p = workdir / name
        try:
            if p.exists():
                p.unlink()
                removed.append(name)
        except OSError:
            pass

    # 帧变化会影响 SGBM 输入
    dense_pairs = workdir / "dense_pairs"
    if dense_pairs.exists() and dense_pairs.is_dir():
        n = 0
        for f in dense_pairs.glob("pair_*.npz"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        meta = dense_pairs / "meta.json"
        try:
            if meta.exists():
                meta.unlink()
        except OSError:
            pass
        if n > 0:
            removed.append(f"dense_pairs/（{n} 个帧对）")
    return removed