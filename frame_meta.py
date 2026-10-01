"""frame_meta.json 读写与参数校验。

cli / gui 共用，避免两处逻辑漂移。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

# 与抽帧结果强相关的字段；改动会使下游（位姿 / 点云）缓存失效
FIELDS = (
    "video",
    "scale",
    "fps",
    "sampling_mode",
    "feature_type",
    "min_frames",
    "max_frames",
)


def write_meta(path: Path, cfg: Dict[str, Any]) -> None:
    """写 frame_meta.json。"""
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False))


def check_meta(meta_path: Path, cfg: Dict[str, Any]) -> List[str]:
    """比对 meta.json 与当前配置。

    返回不一致项的人类可读描述；空列表表示一致或无法判定。
    """
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
        if key == "video":
            if os.path.abspath(str(o)) != os.path.abspath(str(n)):
                reasons.append(
                    f"video {os.path.basename(str(o))} → "
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


def invalidate_downstream(workdir: Path) -> List[str]:
    """删除因抽帧参数变化而失效的下游缓存。

    返回被删除的文件名列表。
    """
    removed: List[str] = []
    for name in ("intrinsics.npy", "poses.npy", "sparse_points.npy"):
        p = Path(workdir) / name
        try:
            if p.exists():
                p.unlink()
                removed.append(name)
        except OSError:
            pass
    return removed