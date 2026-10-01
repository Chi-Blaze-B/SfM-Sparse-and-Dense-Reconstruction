"""SfM / 稠密重建结果导出工具。

集中管理：
  · 二进制 PLY（可选带 RGB）
  · cameras.txt（位姿 + 四元数）
  · intrinsics.json（内参）
以及两个纯函数辅助：旋转矩阵→四元数、从 3×3 K 还原 CameraIntrinsics。

只依赖 poses 的数据类，不反向依赖 gui / cli / dense。
"""

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from poses import CameraIntrinsics, CameraPose


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

def rot_to_quat(R: np.ndarray):
    """旋转矩阵 → (qx, qy, qz, qw)。"""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def intr_from_K(K: np.ndarray) -> CameraIntrinsics:
    """从 3×3 内参矩阵还原 CameraIntrinsics（位姿走缓存时用）。"""
    K = np.asarray(K, dtype=np.float64)
    return CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
    )


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

_PLY_DTYPE_RGB = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])


def _ply_header(n: int, *, with_color: bool) -> str:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if with_color:
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def write_ply(path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """写二进制 little-endian PLY。

    rgb=None 时只写 x y z；否则写 x y z + red green blue。
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz 形状应为 (N,3)，实际 {xyz.shape}")
    n = int(xyz.shape[0])

    use_rgb = rgb is not None
    if use_rgb:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 2 or rgb.shape != (n, 3):
            raise ValueError(
                f"rgb 形状应与 xyz 匹配 (N,3)，实际 {rgb.shape}")

    header = _ply_header(n, with_color=use_rgb)
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if n == 0:
            return
        if use_rgb:
            arr = np.empty(n, dtype=_PLY_DTYPE_RGB)
            arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(arr.tobytes())
        else:
            f.write(xyz.tobytes())


# ---------------------------------------------------------------------------
# 相机位姿
# ---------------------------------------------------------------------------

def write_cameras_txt(path, poses: Sequence[Optional[CameraPose]]) -> None:
    """写 cameras.txt：frame_index tx ty tz qx qy qz qw。"""
    lines = [
        "# frame_index tx ty tz qx qy qz qw",
        "# 约定: X_cam = R @ X_world + t  (OpenCV 相机系)",
    ]
    for i, p in enumerate(poses):
        if p is None:
            continue
        t = p.t.reshape(3)
        q = rot_to_quat(p.R)
        lines.append(
            f"{i} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 内参
# ---------------------------------------------------------------------------

def write_intrinsics_json(path, K: np.ndarray, w: int, h: int) -> None:
    K = np.asarray(K, dtype=np.float64)
    data = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "image_width": int(w),
        "image_height": int(h),
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))