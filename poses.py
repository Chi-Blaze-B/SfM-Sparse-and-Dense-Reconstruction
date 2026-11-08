"""
增量式运动恢复结构：相机位姿估算与稀疏重建
"""

import gc
import bisect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import psutil
import pyceres
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import MiniBatchKMeans

from frame_cache import get_image

logger = logging.getLogger(__name__)


# ---- 可选 Cython 加速 ----
# 发布包自带 sfm_accel.pyd（无 OpenMP、自包含）；add_dll_directory 仅供自行重编译开
# OpenMP 时查找依赖，失败一律回退纯 Python，结果一致。
sfm_accel = None
_ACCEL_DLL_HANDLE = None
if os.environ.get("SFM_ACCEL", "1") != "0":
    try:
        if hasattr(os, "add_dll_directory"):
            _ACCEL_DLL_HANDLE = os.add_dll_directory(
                os.path.dirname(os.path.abspath(__file__)))
        import sfm_accel
    except Exception as _accel_exc:      # noqa: BLE001 - 任何失败都回退
        sfm_accel = None
        logger.debug("sfm_accel 不可用，使用纯 Python 实现: %s", _accel_exc)

# BA 残差内核：首次调用时如果参数布局不符就永久退回纯 Python，避免每次异常开销。
# SFM_ACCEL_COST=0 只关这一个内核（其余 Cython 加速保留），用于测速对照。
_ACCEL_COST_OK = bool(sfm_accel is not None
                      and hasattr(sfm_accel, "reproj_cost_eval")
                      and os.environ.get("SFM_ACCEL_COST", "1") != "0")


# ---- 1. 模块级常量与状态 ----
class FrameStatus:
    UNPROCESSED = "unprocessed"
    TRACKED = "tracked"
    KEYFRAME = "keyframe"
    LOST = "lost"
    RELOCATED = "relocated"
    INVALID = "invalid"


MATCH_RATIO = 0.75
MATCH_RATIO_ROOTSIFT = 0.92
MATCH_DIST_ORB = 90
# OpenCV 原始 SIFT 描述子未归一化，行范数约 512，阈值必须在数百量级；
# 1.5 会滤掉全部原始 SIFT 匹配，只留 RootSIFT。
MATCH_DIST_SIFT = 350.0
MATCH_DIST_ROOTSIFT = 1.5
MATCH_DIST_UPDATE_ORB = 35
MATCH_DIST_UPDATE_SIFT = 150.0
MATCH_DIST_UPDATE_ROOTSIFT = 0.5

KF_ANGLE_DEG = 5.0
KF_TRANS_RATIO = 0.15
KF_COVIS_RATIO = 0.60
KF_CULL_WINDOW = 10
KF_CULL_COVIS = 0.80
MIN_KF_FRAME_GAP = 2
# 地图点剔除间隔；短序列里设得过大等于从不执行
PRUNE_EVERY_FRAMES = 40

MAX_REPROJ_ERROR = 4.0
MIN_OBSERVATIONS = 2
MULTIVIEW_MIN_OBS = 3
MULTIVIEW_ACCEPT_IMPROVE = 0.90
# 多视图三角化接受一个点所需的重投影内点占比（2 观测点时仍要求全内点）
MULTIVIEW_MIN_INLIER_RATIO = 0.6
ADD_OBS_ERR_SCALE = 3.0
MAX_OBS_PER_POINT = 10
# 补观测 / 补 track 的重投影几何门槛（像素）：只按投影半径 + 描述子距离会把约 1/4
# 的错误绑定放进来，收紧后只接受重投影对得上的关键点。
TRACK_GATE_PX = float(os.environ.get("SFM_TRACK_GATE", "3.0"))

BASELINE_MIN_POS_RATIO = 0.5
BASELINE_MIN_ANGLE_DEG = 2.0

BA_MAX_POINTS = 8000
BA_MAX_OBS = 20000
BA_LOCAL_TRIGGER_EVERY = 1
BA_EM_ITERS = 3                     # EM 外点迭代次数
# num_threads 只对 C++ 代价函数有效：Python 的 Evaluate 受 GIL 串行化，
# 多线程反而更慢（18000 观测实测），故固定 1。
SOLVER_NUM_THREADS = max(1, int(os.environ.get("SFM_BA_THREADS", "1")))
BA_EM_MIN_WEIGHT = 0.01             # 内点概率下限
BA_EM_REASSIGN_RATIO = 0.5          # 重关联时误差需低于旧值此比例
# 局部 BA 焦距软先验宽度（相对 fx）：窗口覆盖不足全图，用极强先验近似固定，
# 同时避开共享常量块的崩溃。
LOCAL_FOCAL_PRIOR_RATIO = 0.002

FILTER_MEDIAN_FACTOR = 3.0
FILTER_REPROJ_FACTOR = 2.0

MERGE_MAX_DESC_DIST_ORB = 20
MERGE_MAX_DESC_DIST_SIFT = 100.0
MERGE_MAX_DESC_DIST_ROOTSIFT = 0.5
MERGE_MAX_XYZ_DIST_FACTOR = 0.02

LOOP_MIN_SCORE = 0.15
LOOP_RELOC_SCORE = 0.05
LOOP_SKIP_RECENT = 15
LOOP_MIN_SEQ_CONSISTENCY = 2
LOOP_CANDIDATE_NMS_GAP = 5
LOOP_SEQ_CHECK_K = 2
LOOP_SEQ_CHECK_THRESH = 0.3
LOOP_MIN_INLIER_RATIO = 0.20

PGO_CAUCHY_DELTA = 0.15
PGO_SEQ_WEIGHT = 1.0
PGO_LOOP_WEIGHT = 1.5
PGO_MAX_DIVERGENCE_RATIO = 0.3
PGO_OUTLIER_REL_POSE_THRESH = 0.5

GLOBAL_MATCH_TOP_K = 10
GLOBAL_MATCH_SKIP_RECENT = 5
GLOBAL_MATCH_MIN_INLIERS = 12
GLOBAL_MATCH_TRIGGER_EVERY_KF = 1
GLOBAL_BA_EVERY_KF = 20

# ---- 无标定数据的径向畸变自标定 ----
# k1 与焦距强相关，给一个宽松的弱先验只用于防止发散（真值量级约 0.02）。
K1_PRIOR_SIGMA = 0.5
# 自标定出的 |k1| 小于这个值时不值得再跑一遍去畸变流程
SELF_CALIB_MIN_K1 = 0.004
# 帧数太少时 BA 估出的 k1 是噪声，不足以触发第二遍去畸变
SELF_CALIB_MIN_FRAMES = 10
SELF_CALIB_MIN_KF = 5
# 第三遍：按残余 k1 合成 k1_total 再重跑。实测能减小中心/环形误差，但会放大相邻帧
# 相对旋转、累积成约 12° 全局漂移且耗时 +22%，默认关闭。
SELF_CALIB_ITER = False
SELF_CALIB_RESIDUAL_MIN = 0.003
K1_RESIDUAL_PRIOR_SIGMA = 0.1

# ---- 位姿离群检测 ----
# 与邻居的相对旋转偏差超过这个角度就判定该帧不可信（两视图 E 矩阵独立估计）
POSE_OUTLIER_DEG = 12.0
# 旋转一致性边权重（残差单位为弧度）；设为 0 可关闭
ROT_EDGE_WEIGHT = 100.0


_warned_keys: Set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(message)


# ---- 2. 内存与自适应配置 ----
class MemoryGuard:
    def __init__(self, high=0.85, low=0.65):
        self.high = high
        self.low = low
        self.max_workers = max(1, psutil.cpu_count(logical=False) or 4)
        self.current = self.max_workers

    def update(self) -> int:
        pct = psutil.virtual_memory().percent / 100.0
        if pct > self.high and self.current > 1:
            self.current = max(1, self.current - 1)
            gc.collect()
            logger.warning("[mem] %.1f%% -> workers=%d", pct * 100, self.current)
        elif pct < self.low and self.current < self.max_workers:
            self.current += 1
        return self.current


_MEM_GUARD = MemoryGuard()


@dataclass
class AutoConfig:
    image_size: float
    feature_count: int
    min_features: int
    reproj_thresh: float
    ransac_thresh: float
    triang_thresh: float
    guided_radius_px: float
    init_search_ratio: float
    init_candidate_window: int
    init_max_pairs: int
    init_min_inliers: int
    init_min_parallax_deg: float
    pnp_window: int
    pnp_min_inliers: int
    pnp_ransac_iters: int
    multiview_trigger_every_kf: int
    ba_local_covis_neighbors: int
    ba_min_obs: int
    ba_local_iters: int
    ba_local_refinements: int
    ba_global_iters: int
    ba_global_image_growth: float
    ba_global_point_growth: float
    loop_vocab_size: int
    loop_min_inliers: int
    loop_max_candidates: int
    loop_min_gap: int
    reloc_min_matches: int
    reloc_min_inliers: int
    reloc_max_candidates: int
    pgo_iters: int
    complete_max_transitivity: int
    continue_max_angle_deg: float
    filter_min_tri_angle_deg: float


def derive_auto_config(n_frames, img_shape, feature_type, has_intrinsics, has_distortion) -> AutoConfig:
    h, w = img_shape
    image_size = float(max(h, w))
    n_pixels = float(h * w)
    base_feat = int(np.clip(np.sqrt(n_pixels) * 1.5, 2000, 12000))
    ft = feature_type.lower()
    feature_count = int(base_feat * 1.5) if ft == "orb" else base_feat
    min_features = max(80, int(feature_count * 0.01))
    # ORB 的关键点定位误差比 SIFT 大，PnP / 三角化阈值需要更宽
    if ft == "orb":
        reproj_thresh = float(np.clip(image_size * 0.0020, 1.0, 4.0))
    else:
        reproj_thresh = float(np.clip(image_size * 0.0010, 0.5, 2.0))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh
    guided_radius_px = float(np.clip(image_size * 0.008, 5.0, 20.0))

    if n_frames <= 20:
        init_search_ratio = 1.0
        init_candidate_window = max(1, n_frames - 1)
    elif n_frames <= 50:
        # 图片集（非视频）顺序不可靠，候选对放宽到全部两两组合
        init_search_ratio = 1.0
        init_candidate_window = n_frames
    elif n_frames <= 200:
        init_search_ratio = 0.06
        init_candidate_window = 15
    else:
        init_search_ratio = 0.03
        init_candidate_window = 15

    if n_frames <= 50:
        init_max_pairs = n_frames * (n_frames - 1) // 2
    else:
        init_max_pairs = int(np.clip(n_frames * 3, 100, 500))
    init_min_inliers = int(np.clip(image_size * 0.022, 60, 120))
    init_min_parallax_deg = 8.0
    if n_frames <= 50:
        pnp_window = 40
    elif n_frames <= 200:
        pnp_window = 30
    else:
        pnp_window = 20
    pnp_min_inliers = int(np.clip(image_size * 0.008, 12, 25))
    pnp_ransac_iters = int(np.clip(100 + image_size * 0.05, 100, 300))
    if n_frames <= 100:
        multiview_trigger_every_kf = 5
    elif n_frames <= 500:
        multiview_trigger_every_kf = 10
    else:
        multiview_trigger_every_kf = 15
    ba_local_covis_neighbors = int(np.clip(12 - np.log10(max(n_frames, 10)) * 2, 6, 12))
    ba_min_obs = int(np.clip(n_frames * 2, 50, 150))
    ba_local_iters = 25
    ba_local_refinements = 2
    ba_global_iters = int(np.clip(40 + np.log2(max(n_frames, 2)) * 3, 40, 80))
    ba_global_image_growth = 1.10
    ba_global_point_growth = 1.10
    loop_vocab_size = int(np.clip(np.sqrt(n_frames) * 40, 400, 1500))
    loop_min_inliers = int(np.clip(image_size * 0.006, 12, 25))
    loop_max_candidates = 10
    loop_min_gap = max(15, n_frames // 8)
    reloc_min_matches = int(np.clip(image_size * 0.030, 30, 80))
    reloc_min_inliers = int(np.clip(image_size * 0.015, 15, 40))
    reloc_max_candidates = 8
    pgo_iters = 80
    complete_max_transitivity = 5
    continue_max_angle_deg = 2.0
    filter_min_tri_angle_deg = 2.0
    return AutoConfig(
        image_size=image_size,
        feature_count=feature_count,
        min_features=min_features,
        reproj_thresh=reproj_thresh,
        ransac_thresh=ransac_thresh,
        triang_thresh=triang_thresh,
        guided_radius_px=guided_radius_px,
        init_search_ratio=init_search_ratio,
        init_candidate_window=init_candidate_window,
        init_max_pairs=init_max_pairs,
        init_min_inliers=init_min_inliers,
        init_min_parallax_deg=init_min_parallax_deg,
        pnp_window=pnp_window,
        pnp_min_inliers=pnp_min_inliers,
        pnp_ransac_iters=pnp_ransac_iters,
        multiview_trigger_every_kf=multiview_trigger_every_kf,
        ba_local_covis_neighbors=ba_local_covis_neighbors,
        ba_min_obs=ba_min_obs,
        ba_local_iters=ba_local_iters,
        ba_local_refinements=ba_local_refinements,
        ba_global_iters=ba_global_iters,
        ba_global_image_growth=ba_global_image_growth,
        ba_global_point_growth=ba_global_point_growth,
        loop_vocab_size=loop_vocab_size,
        loop_min_inliers=loop_min_inliers,
        loop_max_candidates=loop_max_candidates,
        loop_min_gap=loop_min_gap,
        reloc_min_matches=reloc_min_matches,
        reloc_min_inliers=reloc_min_inliers,
        reloc_max_candidates=reloc_max_candidates,
        pgo_iters=pgo_iters,
        complete_max_transitivity=complete_max_transitivity,
        continue_max_angle_deg=continue_max_angle_deg,
        filter_min_tri_angle_deg=filter_min_tri_angle_deg,
    )


_RUNTIME_CFG: Optional[AutoConfig] = None


def _cfg() -> AutoConfig:
    if _RUNTIME_CFG is None:
        return derive_auto_config(100, (1080, 1920), "sift", False, False)
    return _RUNTIME_CFG


# ---- 3. 数据结构 ----
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[np.ndarray] = None

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)

    @property
    def K64(self) -> np.ndarray:
        return self.K.astype(np.float64)


@dataclass
class CameraPose:
    R: np.ndarray
    t: np.ndarray

    @property
    def RT(self) -> np.ndarray:
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = self.R
        RT[:3, 3] = self.t.flatten()
        return RT

    @property
    def center(self) -> np.ndarray:
        return (-self.R.T @ self.t).flatten()

    def copy(self) -> "CameraPose":
        return CameraPose(self.R.copy(), self.t.copy())


@dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray


@dataclass
class Observation:
    frame_idx: int
    kp_idx: int
    u: float
    v: float
    color: Optional[np.ndarray] = None


@dataclass
class MapPoint:
    idx: int
    xyz: np.ndarray
    desc: np.ndarray
    obs: List[Observation] = field(default_factory=list)
    obs_set: Set[Tuple[int, int]] = field(default_factory=set)
    desc_age: int = 0

    @property
    def obs_count(self) -> int:
        return len(self.obs)

    def add_observation(self, obs: Observation) -> bool:
        key = (obs.frame_idx, obs.kp_idx)
        if key in self.obs_set:
            return False
        self.obs_set.add(key)
        self.obs.append(obs)
        return True


@dataclass
class Frame:
    idx: int
    path: str
    kps: List[cv2.KeyPoint] = field(default_factory=list)
    desc: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    pose: Optional[CameraPose] = None
    status: str = FrameStatus.UNPROCESSED
    feat_map: Optional[np.ndarray] = None
    track_quality: float = 0.0
    is_keyframe: bool = False
    kdtree: Optional[cKDTree] = None

    def __post_init__(self):
        if self.feat_map is None:
            self.feat_map = np.full(len(self.kps), -1, dtype=np.int64)
        self._build_kdtree()

    def _build_kdtree(self):
        if len(self.kps) > 0:
            pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            try:
                self.kdtree = cKDTree(pts)
            except Exception:
                self.kdtree = None


@dataclass
class SfMResult:
    intrinsics: CameraIntrinsics
    poses: List[Optional[CameraPose]]
    xyz: np.ndarray
    rgb: np.ndarray
    obs_per_point: List[List[Tuple[int, int, float, float]]]
    keyframes: List[int]
    feature_type: str
    loop_closures: List[Tuple[int, int]] = field(default_factory=list)
    frame_status: Optional[List[str]] = None


# ---- 4. 几何与数学工具 ----
def _build_K(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _registered_ba_frames(frames, map_points) -> List[int]:
    """参与全局 BA 的帧：不限于关键帧，PnP 注册的 TRACKED / RELOCATED 也要精修。

    无观测的帧参数块不受约束，按 obs 过滤掉。
    """
    seen = set()
    for mp in map_points:
        for o in mp.obs:
            seen.add(o.frame_idx)
    return [k for k in sorted(frames) if frames[k].pose is not None and k in seen]


def _rotation_angle_deg(R):
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv) * 180.0 / np.pi)


def _relative_pose(pose_i, pose_j):
    R = pose_j.R @ pose_i.R.T
    t = pose_j.t - R @ pose_i.t
    return CameraPose(R, t)


def _compose_pose(pose_i, T_ij):
    """pose_i ∘ T_ij：把帧 i 的世界位姿与 i→j 的相对位姿合成为帧 j 的世界位姿。"""
    R_ij = np.asarray(T_ij.R, dtype=np.float64)
    t_ij = np.asarray(T_ij.t, dtype=np.float64).reshape(3, 1)
    R = R_ij @ np.asarray(pose_i.R, dtype=np.float64)
    t = R_ij @ np.asarray(pose_i.t, dtype=np.float64).reshape(3, 1) + t_ij
    return CameraPose(R, t)


def _tri_angle_mode() -> str:
    """三角化角判据固定为 `max`：只需存在一对相机过阈值。

    `min` 要求所有相机对都过，会误删「一对宽基线 + 一对窄基线」的正常点。
    """
    return "max"


def _compute_tri_angle(X, cam_centers, mode="min"):
    """相机对三角化角的极值，语义由 `mode` 决定（见 `_tri_angle_mode`）。"""
    n = len(cam_centers)
    if n < 2:
        return 0.0
    rays = X[None, :] - cam_centers
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays / np.maximum(norms, 1e-9)
    cos_mat = rays @ rays.T
    tri_angle = 180.0 if mode == "min" else 0.0
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.clip(cos_mat[i, j], -1.0, 1.0))
            a = float(np.degrees(np.arccos(c)))
            if mode == "min":
                if a < tri_angle:
                    tri_angle = a
            elif a > tri_angle:
                tri_angle = a
    if mode == "min":
        return tri_angle if tri_angle < 180.0 else 0.0
    return tri_angle


def _check_obs_angle(pt, new_frame_idx, frames, min_angle_deg):
    if new_frame_idx not in frames:
        return False
    if frames[new_frame_idx].pose is None:
        return False
    cam_centers = [frames[new_frame_idx].pose.center]
    for o in pt.obs:
        f = frames.get(o.frame_idx)
        if f is not None and f.pose is not None:
            cam_centers.append(f.pose.center)
    if len(cam_centers) < 2:
        return True
    return _compute_tri_angle(np.asarray(pt.xyz, dtype=np.float64),
                              np.asarray(cam_centers, dtype=np.float64),
                              _tri_angle_mode()) >= min_angle_deg


def estimate_sim3(src, dst):
    if len(src) < 3 or len(src) != len(dst):
        return None
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float((src_c ** 2).sum()) / len(src)
    if var_src < 1e-12:
        return None
    cov = (dst_c.T @ src_c) / len(src)
    try:
        U, S, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1.0
    R = U @ np.diag(d) @ Vt
    s = float((S * d).sum()) / var_src
    if s < 1e-9:
        return None
    t = mu_dst - s * (R @ mu_src)
    return Sim3(s, R, t)


# ---- 5. 描述子与特征提取 ----
def get_metric(desc):
    if desc is None or len(desc) == 0:
        return cv2.NORM_HAMMING, MATCH_DIST_ORB, MATCH_DIST_UPDATE_ORB
    if desc.dtype != np.uint8:
        if _is_rootsift_like(desc):
            return cv2.NORM_L2, MATCH_DIST_ROOTSIFT, MATCH_DIST_UPDATE_ROOTSIFT
        return cv2.NORM_L2, MATCH_DIST_SIFT, MATCH_DIST_UPDATE_SIFT
    return cv2.NORM_HAMMING, MATCH_DIST_ORB, MATCH_DIST_UPDATE_ORB


_ROOTSIFT_CACHE = {}
_ROOTSIFT_CACHE_BYTES = [0]

# 配对缓存（SFM_PAIR_CACHE，默认关）：值持有描述子强引用（键用 id()），上限 512 对 ≈ 100 MB。
_PAIR_CACHE_ON = (os.environ.get("SFM_PAIR_CACHE", "0").strip() not in ("", "0"))
_PAIR_CACHE = {}
_PAIR_CACHE_MAX = 512
_PAIR_CACHE_STATS = {"hit": 0, "miss": 0}
# 重定位帧的引导匹配（SFM_GUIDED_RELOC，默认关）
_GUIDED_RELOC_ON = (os.environ.get("SFM_GUIDED_RELOC", "0").strip()
                    not in ("", "0"))


def _is_rootsift_like(desc):
    if desc is None or len(desc) == 0:
        return False
    if desc.dtype == np.uint8:
        return False
    # 同一描述子数组在匹配 / EM 重关联里反复查询，norm 只算一次。
    # 键用 id + 强引用防复用；只缓存小数组并设总字节上限，避免留住临时大数组。
    key = id(desc)
    hit = _ROOTSIFT_CACHE.get(key)
    if hit is not None and hit[0] is desc:
        return hit[1]
    try:
        res = abs(float(np.linalg.norm(desc[0])) - 1.0) < 0.1
    except Exception:
        res = False
    nb = int(desc.nbytes)
    if nb <= 1 << 20:
        if _ROOTSIFT_CACHE_BYTES[0] + nb > (64 << 20):
            _ROOTSIFT_CACHE.clear()
            _ROOTSIFT_CACHE_BYTES[0] = 0
        _ROOTSIFT_CACHE[key] = (desc, res)
        _ROOTSIFT_CACHE_BYTES[0] += nb
    return res


def _update_dist_for(desc):
    if desc is None or len(desc) == 0:
        return MATCH_DIST_UPDATE_ORB
    if desc.dtype != np.uint8:
        if _is_rootsift_like(desc):
            return MATCH_DIST_UPDATE_ROOTSIFT
        return MATCH_DIST_UPDATE_SIFT
    return MATCH_DIST_UPDATE_ORB


def match_descriptors(desc1, desc2, norm_type, max_dist, ratio=None):
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    if ratio is None:
        if _is_rootsift_like(desc1):
            ratio = MATCH_RATIO_ROOTSIFT
        else:
            ratio = MATCH_RATIO
    if len(desc2) < 2:
        bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)
        return [m for m in bf_cross.match(desc1, desc2) if m.distance < max_dist]
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            if m.distance < max_dist:
                good.append(m)
    return good


def _match_ratio_absolute(desc1, desc2, norm_type, max_dist):
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return match_descriptors(desc1, desc2, norm_type, max_dist)
    if _is_rootsift_like(desc1):
        ratio = MATCH_RATIO_ROOTSIFT
    else:
        ratio = MATCH_RATIO
    # 同一对描述子会在初始对搜索、相对位姿、旋转边、回环处反复匹配。键带 id()，
    # 值必须持有两个数组的强引用，否则 id 被回收复用会命中错误结果。
    key = None
    if _PAIR_CACHE_ON:
        key = (id(desc1), id(desc2), int(norm_type), float(max_dist), float(ratio))
        hit = _PAIR_CACHE.get(key)
        if hit is not None:
            _PAIR_CACHE_STATS["hit"] += 1
            return [cv2.DMatch(q, t, d) for q, t, d in hit[2]]
        _PAIR_CACHE_STATS["miss"] += 1
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    m12 = bf.knnMatch(desc1, desc2, k=2)
    m21 = bf.knnMatch(desc2, desc1, k=2)
    if sfm_accel is not None and hasattr(sfm_accel, "match_ratio_filter"):
        # 只把比值筛选 + 双向一致性搬进 Cython；knnMatch 本身仍在 OpenCV 里
        pairs = sfm_accel.match_ratio_filter(m12, m21, ratio, float(max_dist))
        out = [cv2.DMatch(q, t, 0.0) for q, t in pairs]
        _pair_cache_store(key, desc1, desc2, out)
        return out
    keep12 = {}
    for pair in m12:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            if m.distance < max_dist:
                keep12[m.queryIdx] = m.trainIdx
    keep21 = {}
    for pair in m21:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            if m.distance < max_dist:
                keep21[m.queryIdx] = m.trainIdx
    out = [cv2.DMatch(q, t, 0.0) for q, t in keep12.items() if keep21.get(t) == q]
    _pair_cache_store(key, desc1, desc2, out)
    return out


def _pair_cache_store(key, desc1, desc2, matches):
    """写入配对缓存：值里持描述子强引用，防止 id() 复用。"""
    if key is None:
        return
    if len(_PAIR_CACHE) >= _PAIR_CACHE_MAX:
        _PAIR_CACHE.clear()
    _PAIR_CACHE[key] = (desc1, desc2,
                        tuple((int(m.queryIdx), int(m.trainIdx), float(m.distance))
                              for m in matches))


def _sample_colors(bgr, kps):
    if not kps:
        return np.zeros((0, 3), dtype=np.uint8)
    h, w = bgr.shape[:2]
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    return bgr[ys, xs][:, ::-1].copy()


def _undistort_thin_prism(pts, params):
    """THIN_PRISM_FISHEYE 反畸变（Newton + 信赖域），params 为 12 个全参数。

    除畸变外还有等距鱼眼映射层 u = uu*sin(θ)/(θ·cosθ)，少这层 BA 会靠缩焦距补偿。
    """
    p = np.asarray(params, dtype=np.float64).ravel()
    if p.size < 12:
        return pts
    fx, fy, cx, cy = p[:4]
    k1, k2, p1, p2, k3, k4, sx1, sy1 = p[4:12]
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x0 = (pts[:, 0] - cx) / fx
    y0 = (pts[:, 1] - cy) / fy
    x = x0.copy()
    y = y0.copy()
    for _ in range(100):
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        r8 = r6 * r2
        radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8
        d_radial = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6
        dx = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2) + sx1 * r2
        dy = y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2) + sy1 * r2
        # J = I + d(dx,dy)/d(x,y)
        a = 1.0 + radial + 2.0 * x2 * d_radial + 2.0 * p1 * y + 6.0 * p2 * x + 2.0 * sx1 * x
        b = 2.0 * xy * d_radial + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * sx1 * y
        c = 2.0 * xy * d_radial + 2.0 * p2 * y + 2.0 * p1 * x + 2.0 * sy1 * x
        d = 1.0 + radial + 2.0 * y2 * d_radial + 2.0 * p2 * x + 6.0 * p1 * y + 2.0 * sy1 * y
        bx = x + dx - x0
        by = y + dy - y0
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)
        step_x = (d * bx - b * by) / det
        step_y = (-c * bx + a * by) / det
        radius_sq = np.maximum((x * x + y * y) * 0.01, 0.01)
        norm_sq = step_x * step_x + step_y * step_y
        scale = np.where(norm_sq > radius_sq,
                         np.sqrt(radius_sq / np.maximum(norm_sq, 1e-30)), 1.0)
        step_x *= scale
        step_y *= scale
        x -= step_x
        y -= step_y
        if norm_sq.size and float(np.max(norm_sq)) < 1e-10:
            break
    # 鱼眼 -> 透视：u = uu * sin(θ)/(θ·cosθ)
    theta = np.sqrt(x * x + y * y)
    theta_cos = theta * np.cos(theta)
    fisheye_scale = np.ones_like(theta)
    ok = theta_cos > 1e-12
    fisheye_scale[ok] = np.sin(theta[ok]) / theta_cos[ok]
    x = x * fisheye_scale
    y = y * fisheye_scale
    out = np.stack([fx * x + cx, fy * y + cy], axis=1)
    bad = ~np.isfinite(out).all(axis=1)
    if np.any(bad):
        out[bad] = pts[bad]  # 极端角点不收敛时退回原坐标，保持描述子对齐
    return out


def _undistort_kps(kps, K, dist):
    if dist is None or not kps:
        return kps
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    K64 = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    undist = None
    # 12 个参数且前 4 个与 K 一致 -> 视为 THIN_PRISM_FISHEYE 全参数
    if (d.size == 12
            and abs(d[0] - K64[0, 0]) <= 1e-3 * max(1.0, abs(K64[0, 0]))
            and abs(d[1] - K64[1, 1]) <= 1e-3 * max(1.0, abs(K64[1, 1]))
            and abs(d[2] - K64[0, 2]) <= 1.0
            and abs(d[3] - K64[1, 2]) <= 1.0):
        undist = _undistort_thin_prism(pts, d)
    else:
        try:
            undist = cv2.undistortPoints(
                pts.reshape(-1, 1, 2), K64, d.reshape(-1, 1), P=K64
            ).reshape(-1, 2)
        except cv2.error as e:
            _warn_once("undistort_fail", f"undistortPoints 失败 ({e})")
            return kps
    return [cv2.KeyPoint(float(x), float(y), kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp, (x, y) in zip(kps, undist)]


def _update_desc(pt, new_desc):
    old = pt.desc
    if old is None:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
        return
    if new_desc.dtype != old.dtype:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
        return
    up_dist = _update_dist_for(new_desc)
    if old.dtype == np.uint8:
        d = int(cv2.norm(old, new_desc, cv2.NORM_HAMMING))
    else:
        d = float(cv2.norm(old, new_desc, cv2.NORM_L2))
    if pt.obs_count < 3 or pt.desc_age > 5 or d >= up_dist * 1.2:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
    else:
        pt.desc_age += 1


def _aggregate_colors(points):
    out = np.full((len(points), 3), 128, dtype=np.uint8)
    for j, pt in enumerate(points):
        colors = [o.color for o in pt.obs if o.color is not None]
        if not colors:
            continue
        arr = np.stack(colors, axis=0).astype(np.float32)
        out[j] = np.clip(np.median(arr, axis=0), 0, 255).astype(np.uint8)
    return out


def _aggregate_observations(points):
    return [[(o.frame_idx, o.kp_idx, o.u, o.v) for o in pt.obs] for pt in points]


_DETECTOR = None


def _init_worker(feature_type, feature_count, disable_cache=True):
    global _DETECTOR
    cv2.setNumThreads(1)
    if disable_cache:
        # 子进程每张图只读一次，各自建一份 LRU 缓存纯属浪费内存；
        # 父进程的缓存也跨不过 spawn，这里直接关掉。
        from frame_cache import configure as _configure_cache
        _configure_cache(max_bytes=0)
    ft = feature_type.lower()
    if ft in ("sift", "rootsift"):
        try:
            _DETECTOR = cv2.SIFT_create(nfeatures=feature_count, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)
            return
        except cv2.error:
            ft = "orb"
    _DETECTOR = cv2.ORB_create(nfeatures=int(feature_count * 1.5), scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)


def _pack_kps(kps):
    """cv2.KeyPoint 不能被 joblib/loky 跨进程序列化，拆成 numpy 数组传输。"""
    n = len(kps) if kps is not None else 0
    if n == 0:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int32), np.zeros(0, np.int32))
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    sizes = np.array([kp.size for kp in kps], dtype=np.float32)
    angles = np.array([kp.angle for kp in kps], dtype=np.float32)
    responses = np.array([kp.response for kp in kps], dtype=np.float32)
    octaves = np.array([kp.octave for kp in kps], dtype=np.int32)
    class_ids = np.array([kp.class_id for kp in kps], dtype=np.int32)
    return pts, sizes, angles, responses, octaves, class_ids


def _unpack_kps(packed):
    pts, sizes, angles, responses, octaves, class_ids = packed
    return [cv2.KeyPoint(float(x), float(y), float(s), float(a), float(r),
                         int(o), int(c))
            for (x, y), s, a, r, o, c in
            zip(pts, sizes, angles, responses, octaves, class_ids)]


def _extract_one(i, p, feature_type, K_und, dist_coeffs):
    global _DETECTOR
    img = get_image(p)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {p}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = _DETECTOR.detectAndCompute(gray, None)
    kp = kp if kp is not None else []
    if K_und is not None and dist_coeffs is not None:
        kp = _undistort_kps(kp, K_und, dist_coeffs)
    colors = _sample_colors(img, kp)
    is_sift = feature_type in ("sift", "rootsift")
    if is_sift:
        empty_desc = np.zeros((0, 128), dtype=np.float32)
    else:
        empty_desc = np.zeros((0, 32), dtype=np.uint8)
    if desc is None:
        desc = empty_desc
    if feature_type == "rootsift":
        d = desc.astype(np.float32)
        d /= np.sum(np.abs(d), axis=1, keepdims=True) + 1e-8
        np.sqrt(d, out=d)
        desc = d
    shape = img.shape[:2]
    packed_kps = _pack_kps(kp)
    del img, gray, kp
    return i, p, packed_kps, desc, colors, shape


def _make_detector(feature_type, feature_count):
    ft = feature_type.lower()
    if ft in ("sift", "rootsift"):
        try:
            det = cv2.SIFT_create(nfeatures=feature_count, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)
            return det, ft
        except cv2.error:
            logger.warning("SIFT 不可用, 回退到 ORB。")
            ft = "orb"
    det = cv2.ORB_create(nfeatures=int(feature_count * 1.5), scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)
    return det, ft


def _auto_feature_workers(h: int, w: int, feature_type: str) -> int:
    """按图像尺寸与特征类型估算并行度：SIFT ~230 MB/MP、ORB ~20 MB/MP，另加 60 MB。

    7MP 图 SIFT 单进程峰值约 1.6 GB，按物理核数开满会 OOM。
    """
    try:
        avail_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
    except Exception:
        avail_mb = 2048.0
    mp = max(h * w / 1e6, 0.05)
    per_mp = 230.0 if feature_type in ("sift", "rootsift") else 20.0
    per_worker = mp * per_mp + 60.0
    n = int(max(1, (avail_mb * 0.5) / per_worker))
    cap = max(1, psutil.cpu_count(logical=False) or 4)
    return max(1, min(n, cap))


def extract_features(paths, feature_type, dist_coeffs=None, camera_matrix=None, max_workers=None):
    cfg = _cfg()
    _, feature_type = _make_detector(feature_type, cfg.feature_count)
    K_und = None
    if camera_matrix is not None and dist_coeffs is not None:
        K_und = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    n = len(paths)
    if n == 0:
        return [], None, feature_type
    workers = max_workers or _MEM_GUARD.update()
    # n_jobs=1 时 joblib 不跑 initializer，父进程 _DETECTOR 仍为 None，需先自行初始化；
    # 父进程的帧缓存还要给稠密重建用，不能跟着关。
    if workers <= 1:
        _init_worker(feature_type, cfg.feature_count, disable_cache=False)
    results = Parallel(
        n_jobs=workers,
        backend="loky",
        initializer=_init_worker,
        initargs=(feature_type, cfg.feature_count),
    )(delayed(_extract_one)(i, p, feature_type, K_und, dist_coeffs) for i, p in enumerate(paths))
    frames = []
    img_shape = results[0][5] if results else None
    for i, p, packed_kps, desc, colors, shape in results:
        frames.append(Frame(idx=i, path=p, kps=_unpack_kps(packed_kps),
                            desc=desc, colors=colors))
    return frames, img_shape, feature_type


# ---- 6. 共视图与代价函数 ----
class CovisibilityGraph:
    def __init__(self):
        self.graph: Dict[int, Dict[int, int]] = {}
        self.covis_pts: Dict[int, int] = {}

    def add_keyframe(self, kf_idx: int) -> None:
        self.graph.setdefault(kf_idx, {})

    def remove_keyframe(self, kf_idx: int) -> None:
        self.graph.pop(kf_idx, None)
        self.covis_pts.pop(kf_idx, None)
        for nbrs in self.graph.values():
            nbrs.pop(kf_idx, None)

    def update_from_frame(self, kf_idx, map_points, frames):
        if kf_idx not in self.graph:
            self.graph[kf_idx] = {}
        fm = frames[kf_idx].feat_map
        if fm is None:
            return
        self.graph[kf_idx] = {}
        pts = {int(x) for x in fm if x >= 0}
        self.covis_pts[kf_idx] = len(pts)
        if not pts:
            return
        counts = {}
        for pt_idx in pts:
            if pt_idx >= len(map_points):
                continue
            for o in map_points[pt_idx].obs:
                if o.frame_idx != kf_idx:
                    counts[o.frame_idx] = counts.get(o.frame_idx, 0) + 1
        for other, c in counts.items():
            if other not in frames:
                continue
            if not frames[other].is_keyframe:
                continue
            if other not in self.graph:
                self.graph[other] = {}
            self.graph[kf_idx][other] = c
            self.graph[other][kf_idx] = c

    def get_neighbors(self, seed_kf: int, n: int = 20) -> List[int]:
        if seed_kf not in self.graph:
            return [seed_kf]
        items = sorted(self.graph[seed_kf].items(), key=lambda kv: -kv[1])
        return [seed_kf] + [kf for kf, _ in items[:n]]

    def get_ranked_by_covis(self, exclude_kf: int) -> List[Tuple[int, int]]:
        return sorted(
            [(k, v) for k, v in self.covis_pts.items() if k != exclude_kf],
            key=lambda x: -x[1],
        )


def _quat_to_R(q) -> np.ndarray:
    """xyzw 顺序，内部归一化。"""
    if sfm_accel is not None and hasattr(sfm_accel, "quat_to_R"):
        return sfm_accel.quat_to_R(
            np.ascontiguousarray(q, dtype=np.float64).reshape(4))
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = np.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _reproj_residual(q, t, f, cx, cy, X, k1=0.0) -> np.ndarray:
    R = _quat_to_R(q)
    pc = R @ np.asarray(X, np.float64).reshape(3) + np.asarray(t, np.float64).reshape(3)
    zc = pc[2] if abs(pc[2]) > 1e-9 else 1e-9
    x = pc[0] / zc
    y = pc[1] / zc
    # SIMPLE_RADIAL：像素 = f * 归一化坐标 * (1 + k1 r²) + 主点
    rad = 1.0 + float(k1) * (x * x + y * y)
    return np.array([f * x * rad + cx, f * y * rad + cy], dtype=np.float64)


class _ReprojCost(pyceres.CostFunction):
    """重投影误差，参数块 [q(4), t(3), f(1), cx(1), cy(1), k1(1), X(3)]，q 为 xyzw。

    相机模型为 SIMPLE_RADIAL：u = f·x·(1 + k1·r²) + cx。k1=0 时退化为针孔。
    pyceres 2.x 无 AutoDiffCostFunction，只能在 Evaluate 里同时给残差和解析雅可比。
    """

    def __init__(self, u, v, weight=1.0):
        super().__init__()
        self.u = float(u)
        self.v = float(v)
        self.w = float(np.sqrt(max(weight, 1e-12)))
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([4, 3, 1, 1, 1, 1, 3])

    def Evaluate(self, params, residuals, jacobians):
        # Cython 内核与下面的纯 Python 实现逐行同构（不开 fast-math），
        # SFM_ACCEL=0 或 pyd 缺失时自动走回退路径。
        global _ACCEL_COST_OK
        if _ACCEL_COST_OK:
            try:
                q_a, t_a, f_a, cx_a, cy_a, k1_a, X_a = params
                _f = float(f_a[0])
                _cx = float(cx_a[0])
                _cy = float(cy_a[0])
                _k1 = float(k1_a[0])
                if jacobians is None:
                    sfm_accel.reproj_cost_residual(
                        self.u, self.v, self.w, q_a, t_a,
                        _f, _cx, _cy, _k1, X_a, residuals)
                else:
                    sfm_accel.reproj_cost_eval(
                        self.u, self.v, self.w, q_a, t_a,
                        _f, _cx, _cy, _k1, X_a, residuals,
                        jacobians[0], jacobians[1], jacobians[2],
                        jacobians[3], jacobians[4], jacobians[5],
                        jacobians[6])
                return True
            except Exception as _e:  # noqa: BLE001 - 回退纯 Python
                _ACCEL_COST_OK = False
                logger.warning("[accel] BA 残差内核回退纯 Python: %s", _e)
        # 纯标量实现：Ceres 每步回调进 Python，小 numpy 数组开销显著；
        # 结果与原矩阵写法逐位一致。
        q_a, t_a, f_a, cx_a, cy_a, k1_a, X_a = params
        f = float(f_a[0])
        cx = float(cx_a[0])
        cy = float(cy_a[0])
        k1 = float(k1_a[0])
        x = float(q_a[0]); y = float(q_a[1]); z = float(q_a[2]); qw = float(q_a[3])
        n = (x * x + y * y + z * z + qw * qw) ** 0.5 + 1e-12
        x /= n; y /= n; z /= n; qw /= n
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        xw = x * qw; yw = y * qw; zw = z * qw
        r00 = 1 - 2 * (yy + zz); r01 = 2 * (xy - zw); r02 = 2 * (xz + yw)
        r10 = 2 * (xy + zw);     r11 = 1 - 2 * (xx + zz); r12 = 2 * (yz - xw)
        r20 = 2 * (xz - yw);     r21 = 2 * (yz + xw);     r22 = 1 - 2 * (xx + yy)
        X0 = float(X_a[0]); X1 = float(X_a[1]); X2 = float(X_a[2])
        t0 = float(t_a[0]); t1 = float(t_a[1]); t2 = float(t_a[2])
        pc0 = r00 * X0 + r01 * X1 + r02 * X2 + t0
        pc1 = r10 * X0 + r11 * X1 + r12 * X2 + t1
        pc2 = r20 * X0 + r21 * X1 + r22 * X2 + t2
        zc = pc2 if abs(pc2) > 1e-9 else 1e-9
        inv_z = 1.0 / zc
        xn = pc0 * inv_z
        yn = pc1 * inv_z
        r2 = xn * xn + yn * yn
        rad = 1.0 + k1 * r2
        s = self.w
        residuals[0] = s * (f * xn * rad + cx - self.u)
        residuals[1] = s * (f * yn * rad + cy - self.v)
        if jacobians is None:
            return True

        # d(u,v)/dXc 的两行系数（SIMPLE_RADIAL 下比针孔多一项 2·k1·x²）
        dr2 = 2.0 * k1
        a = f * (rad + dr2 * xn * xn) * inv_z
        b = -f * xn * (rad + dr2 * r2) * inv_z
        c = f * (rad + dr2 * yn * yn) * inv_z
        d = -f * yn * (rad + dr2 * r2) * inv_z
        # d(RX)/dq（3x4，q 已归一化）：直接展开旋转矩阵，避开归一化约束下的求导错误；
        # 径向分量由 manifold 投影掉。
        ux = 2 * (y * X1 + z * X2)
        uy = 2 * (-2 * y * X0 + x * X1 + qw * X2)
        uz = 2 * (-2 * z * X0 - qw * X1 + x * X2)
        uw = 2 * (-z * X1 + y * X2)
        vx = 2 * (y * X0 - 2 * x * X1 - qw * X2)
        vy = 2 * (x * X0 + z * X2)
        vz = 2 * (qw * X0 - 2 * z * X1 + y * X2)
        vw = 2 * (z * X0 - x * X2)
        wx = 2 * (z * X0 + qw * X1 - 2 * x * X2)
        wy = 2 * (-qw * X0 + z * X1 - 2 * y * X2)
        wz = 2 * (x * X0 + y * X1)
        ww = 2 * (-y * X0 + x * X1)

        Jq = jacobians[0]
        Jq[0] = s * (a * ux + b * wx); Jq[1] = s * (a * uy + b * wy)
        Jq[2] = s * (a * uz + b * wz); Jq[3] = s * (a * uw + b * ww)
        Jq[4] = s * (c * vx + d * wx); Jq[5] = s * (c * vy + d * wy)
        Jq[6] = s * (c * vz + d * wz); Jq[7] = s * (c * vw + d * ww)

        Jt = jacobians[1]
        Jt[0] = s * a; Jt[1] = 0.0; Jt[2] = s * b
        Jt[3] = 0.0; Jt[4] = s * c; Jt[5] = s * d

        Jf = jacobians[2]
        Jf[0] = s * xn * rad; Jf[1] = s * yn * rad

        Jcx = jacobians[3]
        Jcx[0] = s; Jcx[1] = 0.0

        Jcy = jacobians[4]
        Jcy[0] = 0.0; Jcy[1] = s

        Jk = jacobians[5]
        Jk[0] = s * f * xn * r2
        Jk[1] = s * f * yn * r2

        JX = jacobians[6]
        JX[0] = s * (a * r00 + b * r20)
        JX[1] = s * (a * r01 + b * r21)
        JX[2] = s * (a * r02 + b * r22)
        JX[3] = s * (c * r10 + d * r20)
        JX[4] = s * (c * r11 + d * r21)
        JX[5] = s * (c * r12 + d * r22)
        return True


class _RotCost(pyceres.CostFunction):
    """只用两视图 E 矩阵约束相对旋转：r = Log(R_meas^T · R_j R_i^T)。

    这是**独立于地图**的证据：回环边 / PGO 的测量都打在地图点上，只能让模型自洽，
    纠正不了累积旋转漂移；E 矩阵只依赖两张图像本身。雅可比用中心差分。
    """

    def __init__(self, R_meas, weight=1.0):
        super().__init__()
        self.R_meas = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
        self.w = float(np.sqrt(max(weight, 1e-12)))
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4, 4])

    def _residual(self, qa, qb):
        Ri = _quat_to_R(qa)
        Rj = _quat_to_R(qb)
        R_err = self.R_meas.T @ (Rj @ Ri.T)
        rv, _ = cv2.Rodrigues(R_err)
        return rv.ravel()

    def Evaluate(self, params, residuals, jacobians):
        qa = np.asarray(params[0], dtype=np.float64).reshape(4).copy()
        qb = np.asarray(params[1], dtype=np.float64).reshape(4).copy()
        r = self._residual(qa, qb)
        residuals[:] = self.w * r
        if jacobians is None:
            return True
        eps = 1e-6
        Ja = np.empty((3, 4), dtype=np.float64)
        Jb = np.empty((3, 4), dtype=np.float64)
        for k in range(4):
            old = qa[k]; qa[k] = old + eps; rp = self._residual(qa, qb)
            qa[k] = old - eps; rm = self._residual(qa, qb); qa[k] = old
            Ja[:, k] = (rp - rm) / (2.0 * eps)
            old = qb[k]; qb[k] = old + eps; rp = self._residual(qa, qb)
            qb[k] = old - eps; rm = self._residual(qa, qb); qb[k] = old
            Jb[:, k] = (rp - rm) / (2.0 * eps)
        for _bi, _J in ((0, Ja), (1, Jb)):
            _arr = jacobians[_bi]
            _vals = (self.w * _J).ravel()
            if getattr(_arr, "ndim", 1) == 1:
                _arr[:] = _vals
            else:
                _arr[:, :] = _vals.reshape(_arr.shape)
        return True


class _RelPoseCost(pyceres.CostFunction):
    """回环端点的六维相对位姿残差：r = [Log(R_measᵀR_jR_iᵀ), (t_meas − t_pred)/scale]。

    与 _RotCost 同思路，但把平移也纳入；平移按场景尺度归一化，与旋转弧度同量级。
    只约束端点位姿，不碰地图点，避免重三角化引入重复观测。雅可比用中心差分
    （回环边数量少，Python 侧代价可接受）。
    """

    def __init__(self, R_meas, t_meas, weight=1.0, t_scale=1.0):
        super().__init__()
        self.R_meas = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
        self.t_meas = np.asarray(t_meas, dtype=np.float64).reshape(3, 1)
        self.w = float(np.sqrt(max(weight, 1e-12)))
        self.scale = max(float(t_scale), 1e-9)
        self.set_num_residuals(6)
        self.set_parameter_block_sizes([4, 3, 4, 3])

    def _residual(self, qa, ta, qb, tb):
        Ri = _quat_to_R(qa)
        Rj = _quat_to_R(qb)
        R_err = self.R_meas.T @ (Rj @ Ri.T)
        rv, _ = cv2.Rodrigues(R_err)
        t_pred = tb - Rj @ Ri.T @ ta
        return np.concatenate([rv.ravel(),
                               ((self.t_meas - t_pred) / self.scale).ravel()])

    def Evaluate(self, params, residuals, jacobians):
        qa = np.asarray(params[0], dtype=np.float64).reshape(4).copy()
        ta = np.asarray(params[1], dtype=np.float64).reshape(3, 1).copy()
        qb = np.asarray(params[2], dtype=np.float64).reshape(4).copy()
        tb = np.asarray(params[3], dtype=np.float64).reshape(3, 1).copy()
        r = self._residual(qa, ta, qb, tb)
        residuals[:] = self.w * r
        if jacobians is None:
            return True
        eps = 1e-6
        for bi, n in ((0, 4), (1, 3), (2, 4), (3, 3)):
            arr = (qa, ta, qb, tb)[bi].reshape(-1)
            J = np.empty((6, n), dtype=np.float64)
            for k in range(n):
                old = float(arr[k])
                arr[k] = old + eps
                rp = self._residual(qa, ta, qb, tb)
                arr[k] = old - eps
                rm = self._residual(qa, ta, qb, tb)
                arr[k] = old
                J[:, k] = (rp - rm) / (2.0 * eps)
            _arr = jacobians[bi]
            _vals = (self.w * J).ravel()
            if getattr(_arr, "ndim", 1) == 1:
                _arr[:] = _vals
            else:
                _arr[:, :] = _vals.reshape(_arr.shape)
        return True


def _collect_rotation_edges(frames, keyframes, norm_type, match_dist, K,
                            ransac_thresh, max_gap=6):
    """相邻关键帧的两视图相对旋转（x_b = R x_a + t），用于独立约束旋转。"""
    ids = sorted(k for k in keyframes
                 if k in frames and frames[k].pose is not None)
    edges = []
    for a, b in zip(ids[:-1], ids[1:]):
        if b - a > max_gap:
            continue
        R_rel = _rel_rotation_2view(frames[a], frames[b], norm_type,
                                    match_dist, K, ransac_thresh)
        if R_rel is not None:
            edges.append((a, b, R_rel))
    return edges


class _ScalarPriorCost(pyceres.CostFunction):
    """一维软先验 w*(x - x0)，用于固定共享内参。

    pyceres 2.6 对共享参数块 set_parameter_block_constant 会偶发崩溃。
    """

    def __init__(self, prior: float, weight: float):
        super().__init__()
        self.prior = float(prior)
        self.w = float(np.sqrt(max(weight, 1e-12)))
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, params, residuals, jacobians):
        x = float(params[0][0])
        residuals[0] = self.w * (x - self.prior)
        if jacobians is not None:
            jacobians[0][0] = self.w
        return True


class _Sim3PGOEdge(pyceres.CostFunction):
    """Sim(3) 位姿图残差，参数块 [q_i, t_i, s_i, q_j, t_j, s_j]，7 维残差，q 为 xyzw。

    雅可比用中心差分：位姿图规模小，Python 侧代价可接受。
    """

    _BLOCK_SIZES = (4, 3, 1, 4, 3, 1)

    def __init__(self, T_meas, s_meas=1.0, weight=1.0):
        super().__init__()
        self.T_meas_inv = np.linalg.inv(np.asarray(T_meas, dtype=np.float64))
        self.s_meas = max(float(s_meas), 1e-12)
        self.w = float(np.sqrt(max(weight, 1e-12)))
        self.set_num_residuals(7)
        self.set_parameter_block_sizes(list(self._BLOCK_SIZES))

    def _residual(self, params):
        qi, ti, si, qj, tj, sj = params
        Ri = _quat_to_R(qi)
        Rj = _quat_to_R(qj)
        si = max(float(np.asarray(si).reshape(-1)[0]), 1e-12)
        sj = max(float(np.asarray(sj).reshape(-1)[0]), 1e-12)
        ti = np.asarray(ti, dtype=np.float64).reshape(3)
        tj = np.asarray(tj, dtype=np.float64).reshape(3)
        R_pred = Rj @ Ri.T
        t_pred = tj - (sj / si) * (R_pred @ ti)
        s_pred = sj / si
        T_pred = np.eye(4)
        T_pred[:3, :3] = R_pred
        T_pred[:3, 3] = t_pred
        T_err = self.T_meas_inv @ T_pred
        rv_err, _ = cv2.Rodrigues(T_err[:3, :3])
        t_err = T_err[:3, 3]
        s_err = np.log(max(s_pred / self.s_meas, 1e-6))
        return np.concatenate([t_err, rv_err.ravel(), [s_err]])

    def Evaluate(self, params, residuals, jacobians):
        base = [np.array(p, dtype=np.float64, copy=True).reshape(-1) for p in params]
        r = self._residual(base)
        residuals[:] = self.w * r
        if jacobians is None:
            return True
        eps = 1e-6
        for bi, sz in enumerate(self._BLOCK_SIZES):
            J = np.empty((7, sz), dtype=np.float64)
            for j in range(sz):
                old = base[bi][j]
                base[bi][j] = old + eps
                rp = self._residual(base)
                base[bi][j] = old - eps
                rm = self._residual(base)
                base[bi][j] = old
                J[:, j] = (rp - rm) / (2.0 * eps)
            jacobians[bi][:] = (self.w * J).ravel()
        return True


# ---- 7. 几何验证与 BA ----
def _verify_matches_fh(pts1, pts2, thresh, min_inliers):
    if len(pts1) < max(8, min_inliers):
        return None, None, None
    pts1c = pts1.astype(np.float64).reshape(-1, 1, 2)
    pts2c = pts2.astype(np.float64).reshape(-1, 1, 2)
    H = mask_h = None
    try:
        H, mask_h = cv2.findHomography(pts1c, pts2c, cv2.RANSAC, thresh, maxIters=2000, confidence=0.999)
    except cv2.error:
        pass
    n_h = int(mask_h.sum()) if mask_h is not None else 0
    F = mask_f = None
    try:
        F, mask_f = cv2.findFundamentalMat(pts1c, pts2c, cv2.FM_RANSAC, thresh, 0.999, maxIters=2000)
    except cv2.error:
        pass
    n_f = int(mask_f.sum()) if mask_f is not None else 0
    score_h = n_h - 4
    score_f = n_f - 7
    best_mask = best_model = best_type = None
    best_count = 0
    if H is not None and mask_h is not None:
        if n_h >= min_inliers and score_h >= score_f:
            best_mask = mask_h.ravel().astype(bool)
            best_count = n_h
            best_model = H
            best_type = "H"
    if F is not None and mask_f is not None:
        if n_f >= min_inliers:
            if score_f > score_h or best_type is None:
                m = mask_f.ravel().astype(bool)
                if int(m.sum()) > best_count:
                    best_mask = m
                    best_count = int(m.sum())
                    best_model = F
                    best_type = "F"
    if best_mask is None or best_count < min_inliers:
        return None, None, None
    return best_mask, best_type, best_model


def _find_essential(pts1, pts2, K, threshold):
    K32 = K.astype(np.float32)
    try:
        return cv2.findEssentialMat(pts1, pts2, cameraMatrix=K32, method=cv2.RANSAC, prob=0.999, threshold=threshold)
    except (cv2.error, TypeError):
        pass
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=threshold)


def _recover_pose(E, pts1, pts2, K, mask=None):
    K32 = K.astype(np.float32)
    try:
        return cv2.recoverPose(E, pts1, pts2, cameraMatrix=K32)
    except (cv2.error, TypeError):
        pass
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))


def _check_baseline_sufficient(pts_prev, pts_curr, R_rel, t_rel, K, inlier_mask,
                               min_pos_ratio=BASELINE_MIN_POS_RATIO, min_angle_deg=BASELINE_MIN_ANGLE_DEG):
    inlier_ids = np.where(inlier_mask)[0]
    if len(inlier_ids) < 8:
        return False, 0.0
    t_norm = float(np.linalg.norm(t_rel))
    if t_norm < 1e-12:
        return False, 0.0
    t_unit = (t_rel / t_norm).reshape(3)
    RT1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    RT2 = np.hstack([np.asarray(R_rel), t_unit.reshape(3, 1)])
    P1 = np.asarray(K) @ RT1
    P2 = np.asarray(K) @ RT2
    pp = pts_prev[inlier_ids].T.astype(np.float64)
    pc = pts_curr[inlier_ids].T.astype(np.float64)
    pts4d = cv2.triangulatePoints(P1, P2, pp, pc)
    w = pts4d[3]
    valid_w = np.abs(w) > 1e-12
    if int(valid_w.sum()) < 5:
        return False, 0.0
    pts3d = (pts4d[:3] / np.where(valid_w, w, 1.0)).T
    cam_prev = np.zeros(3)
    cam_curr = (-np.asarray(R_rel).T @ t_unit).reshape(3)
    d1 = pts3d[:, 2]
    d2 = (pts3d - cam_curr) @ np.asarray(R_rel)[2]
    pos = valid_w & (d1 > 0) & (d2 > 0)
    if int(pos.sum()) < 5:
        return False, 0.0
    if float(pos.mean()) < min_pos_ratio:
        return False, 0.0
    v1 = pts3d[pos] - cam_prev
    v2 = pts3d[pos] - cam_curr
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    good = (n1 > 1e-12) & (n2 > 1e-12)
    if int(good.sum()) < 5:
        return False, 0.0
    cos_a = np.sum(v1[good] * v2[good], axis=1) / (n1[good] * n2[good])
    angles = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    med = float(np.median(angles))
    return med >= min_angle_deg, med


def _triangulate_multiview_batch(pts_obs_map, frames, K, min_angle, max_reproj, workers):
    """多视图三角化（numpy 批量高斯-牛顿）：位姿固定，每点 3 个未知数且相互独立。

    Huber 阈值即 max_reproj；批量向量化避免逐点 Ceres 回调的 Python 开销。
    """
    if not pts_obs_map:
        return {}

    angle_mode = _tri_angle_mode()

    # ---- 展平成数组（观测按点分块排列） ----
    pid_list = []
    obs_pt = []
    obs_f = []
    uv_rows = []
    for pid, obs_list in pts_obs_map.items():
        if len(obs_list) < 2:
            continue
        bad = False
        for f_idx, _, _ in obs_list:
            f = frames.get(f_idx)
            if f is None or f.pose is None:
                bad = True
                break
        if bad:
            continue
        p = len(pid_list)
        pid_list.append(pid)
        for f_idx, u, v in obs_list:
            obs_pt.append(p)
            obs_f.append(int(f_idx))
            uv_rows.append((u, v))
    if not pid_list:
        return {}

    obs_pt = np.asarray(obs_pt, dtype=np.int64)
    n_pts = len(pid_list)
    n_obs = len(obs_pt)
    uv = np.asarray(uv_rows, dtype=np.float64)

    uniq_f, cam_idx = np.unique(np.asarray(obs_f, dtype=np.int64), return_inverse=True)
    m = len(uniq_f)
    R = np.empty((m, 3, 3), dtype=np.float64)
    t = np.empty((m, 3), dtype=np.float64)
    C = np.empty((m, 3), dtype=np.float64)
    for i, f_idx in enumerate(uniq_f):
        pose = frames[int(f_idx)].pose
        R[i] = pose.R
        t[i] = np.asarray(pose.t, dtype=np.float64).reshape(3)
        C[i] = np.asarray(pose.center, dtype=np.float64).reshape(3)
    Rc = R[cam_idx]
    tc = t[cam_idx]
    Cc = C[cam_idx]

    K64 = np.asarray(K, dtype=np.float64)
    fx, fy = float(K64[0, 0]), float(K64[1, 1])
    cx, cy = float(K64[0, 2]), float(K64[1, 2])
    thr = float(max_reproj)

    starts = np.searchsorted(obs_pt, np.arange(n_pts), side="left")
    counts = np.diff(np.append(starts, n_obs))

    # ---- 初值：前两个观测做 DLT（按相机对分组，批量调 OpenCV） ----
    ia = starts
    ib = starts + 1
    pair_key = cam_idx[ia].astype(np.int64) * m + cam_idx[ib]
    X = np.zeros((n_pts, 3), dtype=np.float64)
    usable = np.zeros(n_pts, dtype=bool)
    order = np.argsort(pair_key, kind="stable")
    bounds = np.flatnonzero(np.diff(pair_key[order])) + 1
    for g in np.split(order, bounds):
        ca = int(cam_idx[ia[g[0]]])
        cb = int(cam_idx[ib[g[0]]])
        P1 = K64 @ np.hstack([R[ca], t[ca][:, None]])
        P2 = K64 @ np.hstack([R[cb], t[cb][:, None]])
        pts4d = cv2.triangulatePoints(
            P1, P2,
            uv[ia[g]].T.reshape(2, -1).astype(np.float64),
            uv[ib[g]].T.reshape(2, -1).astype(np.float64))
        w = pts4d[3]
        ok = np.abs(w) > 1e-12
        if not np.any(ok):
            continue
        Xi = (pts4d[:3, ok] / w[ok]).T
        fin = np.isfinite(Xi).all(axis=1)
        sel = g[ok][fin]
        X[sel] = Xi[fin]
        usable[sel] = True
    if not np.any(usable):
        return {}

    # ---- 每个点独立做 LM（Huber），全向量化 ----
    def _project(Xp):
        RX = np.einsum("nij,nj->ni", Rc, Xp[obs_pt])
        pc = RX + tc
        z = pc[:, 2]
        iz = 1.0 / np.where(np.abs(z) > 1e-9, z, 1e-9)
        du = fx * pc[:, 0] * iz + cx - uv[:, 0]
        dv = fy * pc[:, 1] * iz + cy - uv[:, 1]
        return np.stack([du, dv], axis=1), pc, iz

    def _point_cost(r_):
        s = np.einsum("ni,ni->n", r_, r_)
        rho = np.where(s <= thr * thr, s,
                       2.0 * thr * np.sqrt(np.maximum(s, 1e-30)) - thr * thr)
        return np.add.reduceat(rho, starts)

    def _lm_iters(X_cur, lam_cur, n_iter, active=None):
        """向量化 LM。给了 active 就只用这些观测拟合（稳健重拟合阶段用）。"""
        for _ in range(n_iter):
            r, pc, iz = _project(X_cur)
            cost = _point_cost(r)
            rn = np.sqrt(np.einsum("ni,ni->n", r, r))
            w = np.where(rn <= thr, 1.0, thr / np.maximum(rn, 1e-12))
            if active is not None:
                w = w * active
            Jp = np.zeros((n_obs, 2, 3), dtype=np.float64)
            Jp[:, 0, 0] = fx * iz
            Jp[:, 0, 2] = -fx * pc[:, 0] * iz * iz
            Jp[:, 1, 1] = fy * iz
            Jp[:, 1, 2] = -fy * pc[:, 1] * iz * iz
            J = np.einsum("nij,njk->nik", Jp, Rc)
            HJ = np.einsum("n,nia,nib->nab", w, J, J)
            bJ = np.einsum("n,nia,ni->na", w, J, r)
            H = np.add.reduceat(HJ, starts, axis=0)
            b = np.add.reduceat(bJ, starts, axis=0)
            dg = np.einsum("nii->ni", H)
            A = H + (lam_cur[:, None] * (dg + 1e-9))[:, :, None] * np.eye(3)[None]
            try:
                step = np.linalg.solve(A, -b[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = np.einsum("nij,nj->ni", np.linalg.pinv(A), -b)
            step = np.where(np.isfinite(step), step, 0.0)
            X_new = X_cur + step
            r_new, _, _ = _project(X_new)
            cost_new = _point_cost(r_new)
            acc = np.isfinite(cost_new) & (cost_new < cost)
            X_cur[acc] = X_new[acc]
            lam_cur = np.where(acc, np.maximum(lam_cur * 0.3, 1e-8),
                               np.minimum(lam_cur * 10.0, 1e8))
        return X_cur, lam_cur

    def _obs_mask(X_cur):
        r, pc, _ = _project(X_cur)
        rn = np.sqrt(np.einsum("ni,ni->n", r, r))
        return np.isfinite(rn) & (pc[:, 2] > 1e-6) & (rn <= thr)

    lam = np.full(n_pts, 1e-4, dtype=np.float64)
    X, lam = _lm_iters(X, lam, 8)   # 8 次 LM 迭代

    # ---- 稳健校验（RANSAC 式）：只要求内点占比够，不要求每条观测都在阈值内 ----
    # 遮挡一条就整点丢弃太严；内点足够时只用内点重拟合一次。
    ok_obs = _obs_mask(X)
    inlier_cnt = np.add.reduceat(ok_obs.astype(np.int64), starts)
    need = np.maximum(2, np.ceil(MULTIVIEW_MIN_INLIER_RATIO * counts).astype(np.int64))
    robust_ok = inlier_cnt >= need
    if np.any(robust_ok & (inlier_cnt < counts)):
        X, lam = _lm_iters(X, lam, 4, active=ok_obs.astype(np.float64))
        ok_obs = _obs_mask(X)
        inlier_cnt = np.add.reduceat(ok_obs.astype(np.int64), starts)
        robust_ok = inlier_cnt >= need
    valid = usable & robust_ok & np.isfinite(X).all(axis=1)

    kmax = int(counts.max())
    if kmax >= 2:
        if kmax <= 64:
            slot = np.arange(n_obs) - np.repeat(starts, counts)
            pt_of_obs = np.repeat(np.arange(n_pts), counts)
            padX = np.zeros((n_pts, kmax, 3), dtype=np.float64)
            padC = np.zeros((n_pts, kmax, 3), dtype=np.float64)
            mask = np.zeros((n_pts, kmax), dtype=bool)
            padX[pt_of_obs, slot] = X[obs_pt]
            padC[pt_of_obs, slot] = Cc
            mask[pt_of_obs, slot] = True
            rays = padX - padC
            nrm = np.linalg.norm(rays, axis=2, keepdims=True)
            rays = rays / np.maximum(nrm, 1e-9)
            cos = np.einsum("nki,nli->nkl", rays, rays)
            ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            ang = np.where(mask[:, :, None] & mask[:, None, :], ang, 1e9)
            iu = np.triu_indices(kmax, 1)
            pair_ang = ang[:, iu[0], iu[1]]
            if angle_mode == "max":
                # 无效对在 min 语义下是 1e9；取 max 前必须滤掉，否则"无穷角"会直接放行
                pair_ang = np.where(pair_ang >= 1e9, -1.0, pair_ang)
                agg_ang = pair_ang.max(axis=1)
            else:
                agg_ang = pair_ang.min(axis=1)
            valid &= (counts < 2) | (agg_ang >= min_angle)

        else:
            for p, pid in enumerate(pid_list):
                if not valid[p]:
                    continue
                cams = [C[cam_idx[j]] for j in range(starts[p], starts[p] + counts[p])]
                if len(cams) < 2:
                    continue
                ang = _compute_tri_angle(X[p], np.asarray(cams, dtype=np.float64),
                                         angle_mode)
                if ang < min_angle:
                    valid[p] = False

    result = {}
    for p, pid in enumerate(pid_list):
        if not usable[p]:
            continue
        result[pid] = (X[p].copy(), bool(valid[p]))
    return result


# ---- EM 工具：E 步与加权 BA ----
def _em_inlier_weights(residuals: np.ndarray) -> np.ndarray:
    """内点/外点混合模型的 EM E 步。"""
    d2 = np.sum(residuals ** 2, axis=1)
    n = len(d2)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    # 二维残差：E[d^2] = 2 * sigma^2 -> sigma^2 ≈ median(d^2) / 2
    sigma2 = max(float(np.median(d2)) / 2.0, 1e-6)
    pi_in = 0.7
    w = np.ones(n, dtype=np.float64)
    d2_max = max(float(d2.max()), 1.0)
    for _ in range(3):
        lik_in = pi_in * np.exp(-d2 / (2.0 * sigma2))
        lik_out = (1.0 - pi_in) / d2_max
        w = lik_in / (lik_in + lik_out + 1e-30)
        pi_in = max(float(w.mean()), 1e-3)
        sigma2 = max(float(np.sum(w * d2) / max(np.sum(w), 1e-6)), 1e-6)
    return np.clip(w, BA_EM_MIN_WEIGHT, 1.0)


def _skew_batch(a):
    S = np.zeros((len(a), 3, 3), dtype=np.float64)
    S[:, 0, 1] = -a[:, 2]; S[:, 0, 2] = a[:, 1]
    S[:, 1, 0] = a[:, 2];  S[:, 1, 2] = -a[:, 0]
    S[:, 2, 0] = -a[:, 1]; S[:, 2, 1] = a[:, 0]
    return S


def _rodrigues_batch(w):
    th = np.linalg.norm(w, axis=1)
    K = _skew_batch(w / np.maximum(th, 1e-12)[:, None])
    s = np.sin(th)[:, None, None]
    c = np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1.0 - c) * (K @ K)


def _acc_rows(idx, vals, K):
    """按 idx 把 (n, m) 累加成 (K, m)；bincount 比 np.add.at 快一个数量级。"""
    out = np.empty((K, vals.shape[1]), dtype=np.float64)
    for j in range(vals.shape[1]):
        out[:, j] = np.bincount(idx, weights=vals[:, j], minlength=K)
    return out


def _ba_solve_numpy(obs_cam, obs_pt, uv, weights, R, t, X, f, cx, cy,
                    opt_f, opt_cx, opt_cy, const_cam, free_pt,
                    max_iter, ftol, focal_prior=None, focal_prior_sigma=0.0):
    """向量化 LM 光束法平差（点块 Schur 消元）：相机 [dω, dτ] 左乘扰动，点 X，内参 [f, cx, cy]。

    pyceres 的 Python CostFunction 在两万级观测时几乎就是全部耗时，这里用 BLAS 一次算完。
    """
    n_obs = len(obs_cam)
    n_cam = len(R)
    n_pt = len(X)
    N = 6 * n_cam + 3
    if n_obs == 0:
        return R, t, X, f, cx, cy, {"success": False, "iters": 0, "cost": 0.0, "message": "no obs"}

    free_cam = ~np.asarray(const_cam, dtype=bool)
    free_pt = np.asarray(free_pt, dtype=bool).copy()
    free_obs = free_pt[obs_pt]
    sw = np.sqrt(np.asarray(weights, dtype=np.float64))

    # 观测按点排序：点块聚合与点对枚举都依赖这个顺序
    order = np.argsort(obs_pt, kind="stable")
    pt_sorted = obs_pt[order]
    starts = np.searchsorted(pt_sorted, np.arange(n_pt), side="left")
    counts = np.diff(np.append(starts, n_obs))

    # Schur 修正需要同一批点的所有观测两两组合
    sel_free = np.arange(n_obs)[free_obs[order]]
    pair1 = [order[sel_free]]
    pair2 = [order[sel_free]]
    for k in np.unique(counts):
        if k < 2:
            continue
        g = np.nonzero(counts == k)[0]
        g = g[free_pt[g]]
        if g.size == 0:
            continue
        a, b = np.triu_indices(k, 1)
        pair1.append(order[(starts[g][:, None] + a[None, :]).ravel()])
        pair2.append(order[(starts[g][:, None] + b[None, :]).ravel()])
    p1 = np.concatenate(pair1)
    p2 = np.concatenate(pair2)

    dof = np.concatenate([np.repeat(free_cam[:, None], 6, axis=1).ravel(),
                          np.array([bool(opt_f), bool(opt_cx), bool(opt_cy)])])
    ii6 = (6 * np.arange(n_cam)[:, None] + np.arange(6)[None, :]).ravel()
    ii3 = 6 * n_cam + np.arange(3)
    has_prior = focal_prior is not None and focal_prior_sigma > 0
    w_prior = 1.0 / focal_prior_sigma ** 2 if has_prior else 0.0

    Rc_all = R[obs_cam]

    def evaluate(R_, t_, X_, f_, cx_, cy_, need_jac):
        RX = np.einsum("nij,nj->ni", R_[obs_cam], X_[obs_pt])
        pc = RX + t_[obs_cam]
        z = pc[:, 2]
        iz = 1.0 / np.where(np.abs(z) > 1e-9, z, 1e-9)
        du = f_ * pc[:, 0] * iz + cx_ - uv[:, 0]
        dv = f_ * pc[:, 1] * iz + cy_ - uv[:, 1]
        r = np.stack([du, dv], axis=1) * sw[:, None]
        if not need_jac:
            return r, None, None, None
        Jp = np.zeros((n_obs, 2, 3), dtype=np.float64)
        Jp[:, 0, 0] = f_ * iz
        Jp[:, 0, 2] = -f_ * pc[:, 0] * iz * iz
        Jp[:, 1, 1] = f_ * iz
        Jp[:, 1, 2] = -f_ * pc[:, 1] * iz * iz
        w3 = sw[:, None, None]
        Jx = np.einsum("nij,njk->nik", Jp, R_[obs_cam]) * w3
        Jw = -np.einsum("nij,njk->nik", Jp, _skew_batch(RX)) * w3
        Jc = np.concatenate([Jw, Jp * w3], axis=2)
        Ji = np.zeros((n_obs, 2, 3), dtype=np.float64)
        Ji[:, 0, 0] = pc[:, 0] * iz
        Ji[:, 0, 1] = 1.0
        Ji[:, 1, 0] = pc[:, 1] * iz
        Ji[:, 1, 2] = 1.0
        Ji *= w3
        return r, Jc, Ji, Jx

    def total_cost(r_, f_):
        c = 0.5 * float(np.einsum("ni,ni->", r_, r_))
        if has_prior:
            c += 0.5 * ((f_ - focal_prior) / focal_prior_sigma) ** 2
        return c

    R = R.copy(); t = t.copy(); X = X.copy()
    lam = 1e-4
    r, Jc, Ji, Jx = evaluate(R, t, X, f, cx, cy, True)
    cost = total_cost(r, f)
    cost0 = cost
    n_acc = 0
    n_rej = 0
    iters = 0
    for _ in range(max_iter):
        iters += 1
        H = np.zeros((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        # ---- 相机 / 内参块（所有观测都贡献） ----
        Hc6 = _acc_rows(obs_cam, np.einsum("nia,nib->nab", Jc, Jc).reshape(n_obs, 36), n_cam)
        blk_rows = 6 * np.arange(n_cam)[:, None, None] + np.arange(6)[None, :, None]
        blk_cols = 6 * np.arange(n_cam)[:, None, None] + np.arange(6)[None, None, :]
        H[blk_rows, blk_cols] += Hc6.reshape(n_cam, 6, 6)
        hci = _acc_rows(obs_cam, np.einsum("nia,nib->nab", Jc, Ji).reshape(n_obs, 18), n_cam)
        hci = hci.reshape(6 * n_cam, 3)
        H[np.ix_(ii6, ii3)] += hci
        H[np.ix_(ii3, ii6)] += hci.T
        H[np.ix_(ii3, ii3)] += np.einsum("nia,nib->ab", Ji, Ji)
        b[ii6] += _acc_rows(obs_cam, np.einsum("nia,ni->na", Jc, r), n_cam).ravel()
        b[ii3] += np.einsum("nia,ni->a", Ji, r)
        if has_prior:
            H[ii3[0], ii3[0]] += w_prior
            b[ii3[0]] += w_prior * (f - focal_prior)

        # ---- 点块 ----
        Hxx = _acc_rows(obs_pt, np.einsum("nia,nib->nab", Jx, Jx).reshape(n_obs, 9), n_pt)
        Hxx = Hxx.reshape(n_pt, 3, 3)
        bx = _acc_rows(obs_pt, np.einsum("nia,ni->na", Jx, r), n_pt)
        dgp = np.einsum("nii->ni", Hxx)
        D = Hxx + (lam * (dgp + 1e-12))[:, :, None] * np.eye(3)[None]
        D[~free_pt] = np.eye(3)[None]
        try:
            Dinv = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            Dinv = np.linalg.pinv(D)

        # ---- Schur 消元 ----
        Jc_flat = Jc.reshape(n_obs, 2, 6)
        Ji_flat = Ji.reshape(n_obs, 2, 3)
        J9 = np.concatenate([Jc_flat, Ji_flat], axis=2)                     # (n, 2, 9)
        W = np.matmul(np.matmul(Jx[p1], Dinv[obs_pt[p1]]), Jx[p2].transpose(0, 2, 1))
        ci, cj = obs_cam[p1], obs_cam[p2]
        blk9 = np.matmul(np.matmul(J9[p1].transpose(0, 2, 1), W), J9[p2])   # (M, 9, 9)
        key = ci.astype(np.int64) * n_cam + cj
        S9 = _acc_rows(key, blk9.reshape(-1, 81), n_cam * n_cam).reshape(n_cam, n_cam, 9, 9)
        # 只枚举了 i<=j 的点对，镜像块（转置）要补上；i==j 的块本身对称
        off = p1 != p2
        if np.any(off):
            key_rev = cj[off].astype(np.int64) * n_cam + ci[off]
            S9 += _acc_rows(key_rev, blk9[off].transpose(0, 2, 1).reshape(-1, 81),
                            n_cam * n_cam).reshape(n_cam, n_cam, 9, 9)
        # 内参列在所有相机上指向同一组全局下标，不能用花式索引散播（重复下标
        # 时 -= 只保留最后一个值），只能分块累加。
        s_rows = 6 * np.arange(n_cam)[:, None, None, None] + np.arange(6)[None, None, :, None]
        s_cols = 6 * np.arange(n_cam)[None, :, None, None] + np.arange(6)[None, None, None, :]
        H[s_rows, s_cols] -= S9[:, :, :6, :6]
        H[np.ix_(ii6, ii3)] -= S9[:, :, :6, 6:9].sum(1).reshape(6 * n_cam, 3)
        H[np.ix_(ii3, ii6)] -= (S9[:, :, 6:9, :6].sum(0)
                                .reshape(n_cam, 3, 6).transpose(1, 0, 2).reshape(3, 6 * n_cam))
        H[np.ix_(ii3, ii3)] -= S9[:, :, 6:9, 6:9].sum(axis=(0, 1))

        v = np.einsum("nij,nj->ni", np.matmul(Jx, Dinv[obs_pt]), bx[obs_pt])
        v[~free_obs] = 0.0
        b9 = _acc_rows(obs_cam, np.einsum("nia,ni->na", J9, v), n_cam)      # (C, 9)
        b[ii6] -= b9[:, :6].ravel()
        b[ii3] -= b9[:, 6:].sum(0)

        # ---- 阻尼 + 固定自由度 ----
        diagH = np.diag(H).copy()
        H[np.diag_indices(N)] += lam * (diagH + 1e-12)
        H[~dof, :] = 0.0
        H[:, ~dof] = 0.0
        H[np.arange(N), np.arange(N)] = np.where(dof, np.diag(H), 1.0)
        b[~dof] = 0.0
        try:
            dc = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            dc = np.linalg.lstsq(H, -b, rcond=None)[0]
        if not np.isfinite(dc).all():
            lam *= 10.0
            continue
        # ---- 回代点位移：dp = D^-1 (bx - B^T dc) ----
        dc_cam = dc[(6 * obs_cam[:, None] + np.arange(6)[None, :]).ravel()].reshape(n_obs, 6)
        dc_intr = np.broadcast_to(dc[ii3], (n_obs, 3))
        Bt = np.concatenate([Jc_flat, Ji_flat], axis=2)
        proj = np.einsum("nia,na->ni", Bt[:, :, :6], dc_cam) + np.einsum("nia,na->ni", Bt[:, :, 6:], dc_intr)
        acc = _acc_rows(obs_pt, np.einsum("nia,na->ni", Jx.transpose(0, 2, 1), proj), n_pt)
        dp = -np.einsum("nij,nj->ni", Dinv, bx + acc)
        dp[~free_pt] = 0.0

        dw = dc[(6 * np.arange(n_cam)[:, None] + np.arange(3)[None, :]).ravel()].reshape(n_cam, 3)
        dt = dc[(6 * np.arange(n_cam)[:, None] + 3 + np.arange(3)[None, :]).ravel()].reshape(n_cam, 3)
        R_new = np.matmul(_rodrigues_batch(dw), R)
        t_new = t + dt
        X_new = X + dp
        f_new = f + dc[ii3[0]]
        cx_new = cx + dc[ii3[1]]
        cy_new = cy + dc[ii3[2]]
        r_new, _, _, _ = evaluate(R_new, t_new, X_new, f_new, cx_new, cy_new, False)
        cost_new = total_cost(r_new, f_new)
        # Nielsen 的增益比更新：比“接受就乘 0.3”收敛快得多
        pred = 0.5 * float(dc @ (H @ dc))
        rho = (cost - cost_new) / pred if pred > 1e-30 else -1.0
        if np.isfinite(cost_new) and rho > 0.0:
            rel = (cost - cost_new) / max(cost, 1e-12)
            R, t, X, f, cx, cy = R_new, t_new, X_new, f_new, cx_new, cy_new
            cost = cost_new
            n_acc += 1
            if rho > 0.75:
                lam = max(lam / 3.0, 1e-12)
            elif rho < 0.25:
                lam = min(lam * 2.0, 1e12)
            r, Jc, Ji, Jx = evaluate(R, t, X, f, cx, cy, True)
            if rel < ftol:
                break
        else:
            n_rej += 1
            lam = min(lam * 4.0, 1e12)

    return R, t, X, f, cx, cy, {"success": True, "iters": iters, "cost": cost,
                                "cost0": cost0, "acc": n_acc, "rej": n_rej, "lam": lam}


def _run_ba_numpy(obs_list, all_pt_ids, keyframe_ids, frames, map_points,
                  focal, fy, cx, cy, gauge_a, gauge_b, fixed_kfs, fixed_pts,
                  optimize_points, optimize_principal, fix_focal,
                  focal_prior, focal_prior_sigma, max_iter, em_iters,
                  prev_obs_weights, _state, is_global):
    """bundle_adjustment 的 numpy 后端，语义与 Ceres 版一致。"""
    kf_list = list(keyframe_ids)
    kf_pos = {k: i for i, k in enumerate(kf_list)}
    pt_list = list(all_pt_ids)
    pt_pos = {p: i for i, p in enumerate(pt_list)}
    n_obs = len(obs_list)
    oc = np.fromiter((kf_pos[f] for f, _, _, _ in obs_list), dtype=np.int64, count=n_obs)
    op = np.fromiter((pt_pos[p] for _, p, _, _ in obs_list), dtype=np.int64, count=n_obs)
    uvs = np.array([[u, v] for _, _, u, v in obs_list], dtype=np.float64)

    cnt = np.bincount(op, minlength=len(pt_list))
    free_pt = cnt >= 2
    if not optimize_points:
        free_pt[:] = False
    if fixed_pts:
        for p in fixed_pts:
            if p in pt_pos:
                free_pt[pt_pos[p]] = False

    R_arr = np.array([np.asarray(frames[k].pose.R, dtype=np.float64) for k in kf_list])
    t_arr = np.array([np.asarray(frames[k].pose.t, dtype=np.float64).reshape(3) for k in kf_list])
    X_arr = np.array([np.asarray(map_points[p].xyz, dtype=np.float64).reshape(3) for p in pt_list])

    const_cam = np.zeros(len(kf_list), dtype=bool)
    for k in (gauge_a, gauge_b):
        if k in kf_pos:
            const_cam[kf_pos[k]] = True
    if fixed_kfs:
        for k in fixed_kfs:
            if k in kf_pos:
                const_cam[kf_pos[k]] = True

    if prev_obs_weights is not None and len(prev_obs_weights) == n_obs:
        w = np.ascontiguousarray(np.asarray(prev_obs_weights, dtype=np.float64).copy())
    else:
        w = np.ones(n_obs, dtype=np.float64)

    use_prior = focal_prior is not None and focal_prior_sigma > 0
    f_now, cx_now, cy_now = float(focal), float(cx), float(cy)
    total = max(1, int(em_iters))
    info = {"cost": 0.0, "iters": 0}
    for em in range(total):
        R_arr, t_arr, X_arr, f_now, cx_now, cy_now, info = _ba_solve_numpy(
            oc, op, uvs, w, R_arr, t_arr, X_arr, f_now, cx_now, cy_now,
            not fix_focal, bool(optimize_principal), bool(optimize_principal),
            const_cam, free_pt, max_iter, 1e-6,
            focal_prior=(float(focal_prior) if use_prior else None),
            focal_prior_sigma=(float(focal_prior_sigma) if use_prior else 0.0))
        (logger.info if is_global else logger.debug)(
            "[BA] %s frames=%d pts=%d obs=%d em=%d/%d cost %.2f->%.2f iters=%d acc=%d rej=%d lam=%.1e",
            "global" if is_global else "local", len(kf_list), len(pt_list),
            n_obs, em + 1, total, info.get("cost0", 0.0), info["cost"], info["iters"],
            info.get("acc", 0), info.get("rej", 0), info.get("lam", 0.0))
        if total <= 1 or em == total - 1:
            break
        pc = np.einsum("nij,nj->ni", R_arr[oc], X_arr[op]) + t_arr[oc]
        iz = 1.0 / np.where(np.abs(pc[:, 2]) > 1e-9, pc[:, 2], 1e-9)
        residuals = np.stack([f_now * pc[:, 0] * iz + cx_now - uvs[:, 0],
                              f_now * pc[:, 1] * iz + cy_now - uvs[:, 1]], axis=1)
        new_w = _em_inlier_weights(residuals)
        if np.max(np.abs(new_w - w)) < 0.02:
            break
        w = new_w

    for i, k in enumerate(kf_list):
        frames[k].pose = CameraPose(R_arr[i], t_arr[i].reshape(3, 1))
    for i, p in enumerate(pt_list):
        map_points[p].xyz = X_arr[i].astype(np.float32)
    if _state is not None:
        _state["success"] = True
        _state["obs_weights"] = w
    _fy_ratio = (float(fy) / float(focal)) if abs(float(focal)) > 1e-12 else 1.0
    return f_now, f_now * _fy_ratio, cx_now, cy_now


def bundle_adjustment(keyframe_ids, map_points, frames, focal, fy, cx, cy,
                      optimize_points=True, reproj_thresh=1.0, image_size=1920,
                      max_iter=None, is_global=False, focal_init=None,
                      optimize_principal=False, fixed_kfs: Optional[Set[int]] = None,
                      fixed_pts: Optional[Set[int]] = None,
                      _state: Optional[Dict[str, object]] = None,
                      em_iters: int = 1,
                      prev_obs_weights: Optional[np.ndarray] = None,
                      fix_focal: bool = False,
                      focal_prior: Optional[float] = None,
                      focal_prior_sigma: float = 0.0,
                      k1: float = 0.0,
                      optimize_k1: bool = False,
                      k1_prior_sigma: float = 0.0,
                      extra_obs: Optional[List[Tuple[int, int, float, float]]] = None,
                      rot_edges: Optional[List[Tuple[int, int, np.ndarray]]] = None,
                      rot_weight: float = ROT_EDGE_WEIGHT,
                      loop_edges: Optional[List[Tuple[int, int, CameraPose]]] = None,
                      loop_weight: float = 0.0,
                      loop_t_scale: float = 1.0,
                      ) -> Tuple[float, float, float, float, float]:
    """Ceres 光束法平差；em_iters>1 时启用 EM 外点加权。

    prev_obs_weights 长度与 obs_list 一致时作为 warm start，_state 回写 obs_weights。
    未标定数据可把 k1 作为共享参数一起优化（SIMPLE_RADIAL），返回 5 元组。
    extra_obs 是 (帧, 点, u, v)：窗口外帧对窗口内地图点的观测，位姿固定、只参与残差。
    """
    cfg = _cfg()
    if _state is not None:
        _state["success"] = False
    if max_iter is None:
        if is_global:
            max_iter = cfg.ba_global_iters
        else:
            max_iter = cfg.ba_local_iters
    if focal_init is None:
        focal_init = focal

    valid_kfs = [f for f in keyframe_ids if f in frames and frames[f].pose is not None]
    if len(valid_kfs) < 2:
        return focal, fy, cx, cy, k1
    keyframe_ids = valid_kfs

    obs_list = []
    for f_idx in keyframe_ids:
        fr = frames[f_idx]
        fm = fr.feat_map
        if fm is None:
            continue
        for kp_idx, pt_idx in enumerate(fm):
            pt_idx = int(pt_idx)
            if 0 <= pt_idx < len(map_points):
                u, v = fr.kps[kp_idx].pt
                obs_list.append((f_idx, pt_idx, float(u), float(v)))
    # 窗口外帧对窗口内地图点的观测：位姿固定，但残差参与，给窗口边界上的点外部约束
    extra_frames = set()
    if extra_obs:
        for f_idx, pt_idx, u, v in extra_obs:
            fr = frames.get(f_idx)
            if fr is None or fr.pose is None:
                continue
            if not (0 <= pt_idx < len(map_points)):
                continue
            obs_list.append((int(f_idx), int(pt_idx), float(u), float(v)))
            extra_frames.add(int(f_idx))
    if len(obs_list) < cfg.ba_min_obs:
        if _state is not None:
            _state["success"] = True
        return focal, fy, cx, cy, k1

    if len(obs_list) > BA_MAX_OBS:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(obs_list), BA_MAX_OBS, replace=False)
        obs_list = [obs_list[i] for i in sel]

    pt_kf_count = {}
    for f, pt, _, _ in obs_list:
        pt_kf_count.setdefault(pt, set()).add(f)
    eligible = {pt for pt, kfs in pt_kf_count.items() if len(kfs) >= 2}
    obs_list = [o for o in obs_list if o[1] in eligible]
    if not obs_list:
        if _state is not None:
            _state["success"] = True
        return focal, fy, cx, cy, k1

    all_pt_ids = list(eligible)
    if len(all_pt_ids) > BA_MAX_POINTS:
        cnt = {}
        for _, pid, _, _ in obs_list:
            cnt[pid] = cnt.get(pid, 0) + 1
        all_pt_ids = sorted(all_pt_ids, key=lambda x: cnt[x], reverse=True)[:BA_MAX_POINTS]
        pset = set(all_pt_ids)
        obs_list = [o for o in obs_list if o[1] in pset]

    n_obs = len(obs_list)
    if n_obs < 10:
        return focal, fy, cx, cy, k1

    obs_count = {}
    for f, _, _, _ in obs_list:
        obs_count[f] = obs_count.get(f, 0) + 1
    sorted_kfs = sorted(obs_count.keys(), key=lambda x: -obs_count[x])
    if fixed_kfs:
        uf = [k for k in sorted(fixed_kfs) if k in obs_count]
        if len(uf) >= 1:
            gauge_a = uf[0]
            gauge_b = uf[1] if len(uf) >= 2 else sorted_kfs[0]
        else:
            gauge_a = sorted_kfs[0]
            gauge_b = sorted_kfs[1]
    elif is_global:
        srt = sorted(obs_count.keys())
        gauge_a, gauge_b = srt[0], srt[-1]
    else:
        gauge_a, gauge_b = sorted_kfs[0], sorted_kfs[1]

    # 全局 BA 的 7-DOF Sim3 gauge：固定 gauge_a 的 6 维 + gauge_b 沿基线主轴的一维平移。
    # 全固定两部相机会多钉 5 个约束，位姿有微小不一致时会把全局 BA 钉死并在内部攒应力。
    _gauge_7dof = bool(is_global and not fixed_kfs)
    _gauge_axis = -1
    if _gauge_7dof:
        _pose_a = frames[gauge_a].pose if gauge_a in frames else None
        _pose_b = frames[gauge_b].pose if gauge_b in frames else None
        if _pose_a is None or _pose_b is None or gauge_a == gauge_b:
            _gauge_7dof = False
        else:
            _base_vec = (np.asarray(_pose_b.center, dtype=np.float64).reshape(3)
                         - np.asarray(_pose_a.center, dtype=np.float64).reshape(3))
            _gauge_axis = int(np.argmax(np.abs(_base_vec)))
            logger.debug("[gauge] 7-DOF：a=%d 全固定，b=%d 只固定平移轴 %d（基线 %s）",
                         gauge_a, gauge_b, _gauge_axis,
                         np.array2string(_base_vec, precision=3))

    # numpy 后端：观测/参数打包成数组，用向量化 LM 求解（Ceres 后端保留作对照）。
    # k1 自标定 / 窗口外观测只在 Ceres 实现，numpy 后端静默忽略会产生两套模型，故明确回退。
    if os.environ.get("SFM_BA_BACKEND", "ceres") == "numpy":
        if optimize_k1:
            _warn_once("ba_numpy_k1",
                       "[BA] numpy 后端不支持 k1 自标定，本次改用 Ceres")
        elif extra_obs:
            _warn_once("ba_numpy_extra",
                       "[BA] numpy 后端不支持窗口外观测，本次改用 Ceres")
        else:
            f_n, fy_n, cx_n, cy_n = _run_ba_numpy(
                obs_list, all_pt_ids, keyframe_ids, frames, map_points,
                focal, fy, cx, cy, gauge_a, gauge_b, fixed_kfs, fixed_pts,
                optimize_points, optimize_principal, fix_focal,
                focal_prior, focal_prior_sigma, max_iter, em_iters,
                prev_obs_weights, _state, is_global)
            return f_n, fy_n, cx_n, cy_n, k1

    cam_q = {}
    cam_t = {}
    _cam_ids = list(dict.fromkeys(list(keyframe_ids) + sorted(extra_frames)))
    for k in _cam_ids:
        pose = frames[k].pose
        cam_q[k] = np.ascontiguousarray(Rot.from_matrix(pose.R).as_quat(), dtype=np.float64)
        cam_t[k] = np.ascontiguousarray(pose.t.flatten(), dtype=np.float64)
    # 同一物理相机：内参在所有帧之间共享；fy 固定不优化
    intr_f = np.array([float(focal)], dtype=np.float64)
    intr_cx = np.array([float(cx)], dtype=np.float64)
    intr_cy = np.array([float(cy)], dtype=np.float64)
    intr_k1 = np.array([float(k1)], dtype=np.float64)
    pt_params = {pid: np.ascontiguousarray(map_points[pid].xyz.astype(np.float64).copy()) for pid in all_pt_ids}

    # EM 迭代：E 步求内点概率 -> M 步加权最小二乘
    if (prev_obs_weights is not None
            and len(prev_obs_weights) == n_obs):
        obs_weights = np.ascontiguousarray(
            np.asarray(prev_obs_weights, dtype=np.float64).copy()
        )
    else:
        obs_weights = np.ones(n_obs, dtype=np.float64)
    total_iters = max(1, int(em_iters))
    last_success = False

    for em_iter in range(total_iters):
        # 本轮开始前的快照：Ceres 原地修改参数块，代价反而上升时用它在最后回滚
        _prev_q = {k: v.copy() for k, v in cam_q.items()}
        _prev_t = {k: v.copy() for k, v in cam_t.items()}
        _prev_pt = {pid: v.copy() for pid, v in pt_params.items()}
        _prev_intr = (float(intr_f[0]), float(intr_cx[0]),
                      float(intr_cy[0]), float(intr_k1[0]))
        problem = pyceres.Problem()
        # 鲁棒核（SFM_BA_ROBUST=1）：局部 SOFT_L1、全局 Huber，尺度 1.0（残差即像素误差）。
        # 无核时单个大残差会按二次代价把整条 track 和共享内参一起拉偏。
        _robust = None
        if os.environ.get("SFM_BA_ROBUST", "0").strip() not in ("", "0"):
            _robust = (pyceres.HuberLoss(1.0) if is_global
                       else pyceres.SoftLOneLoss(1.0))
        # 加权残差 = sqrt(w) * r，等价于加权最小二乘；无需额外鲁棒核
        for i, (f, pid, u, v) in enumerate(obs_list):
            cost = _ReprojCost(u, v, obs_weights[i])
            problem.add_residual_block(
                cost, _robust,
                [cam_q[f], cam_t[f], intr_f, intr_cx, intr_cy, intr_k1, pt_params[pid]])

        for k in keyframe_ids:
            if problem.has_parameter_block(cam_q[k]):
                problem.set_manifold(cam_q[k], pyceres.EigenQuaternionManifold())

        # 独立于地图的旋转约束（两视图 E 矩阵），只加在显式传入的边上
        if rot_edges:
            _n_rot = 0
            for a, b, R_meas in rot_edges:
                if a not in cam_q or b not in cam_q:
                    continue
                if not (problem.has_parameter_block(cam_q[a])
                        and problem.has_parameter_block(cam_q[b])):
                    continue
                problem.add_residual_block(
                    _RotCost(R_meas, rot_weight), None,
                    [cam_q[a], cam_q[b]])
                _n_rot += 1
            if em_iter == 0 and _n_rot < len(rot_edges):
                logger.debug("[BA] 旋转边 %d/%d（部分帧无观测块被跳过）",
                             _n_rot, len(rot_edges))

        # 回环端点的相对位姿约束：只加显式传入的边，默认空。
        if loop_edges:
            _n_loop = 0
            for a, b, T_meas in loop_edges:
                if a not in cam_q or b not in cam_q:
                    continue
                if a not in cam_t or b not in cam_t:
                    continue
                if not (problem.has_parameter_block(cam_q[a])
                        and problem.has_parameter_block(cam_q[b])
                        and problem.has_parameter_block(cam_t[a])
                        and problem.has_parameter_block(cam_t[b])):
                    continue
                problem.add_residual_block(
                    _RelPoseCost(T_meas.R, T_meas.t, loop_weight, loop_t_scale),
                    None, [cam_q[a], cam_t[a], cam_q[b], cam_t[b]])
                _n_loop += 1
            if em_iter == 0:
                logger.info("[BA] 回环位姿边 %d/%d（权重 %.0f，尺度 %.3f）",
                            _n_loop, len(loop_edges), loop_weight, loop_t_scale)

        const_kfs = {gauge_a}
        if gauge_b in cam_q:
            if (_gauge_7dof and _gauge_axis >= 0
                    and problem.has_parameter_block(cam_t[gauge_b])):
                # 只钉住基线主轴方向的一维平移（定尺度），旋转与另两维平移交给数据
                problem.set_manifold(
                    cam_t[gauge_b], pyceres.SubsetManifold(3, [_gauge_axis]))
            else:
                const_kfs.add(gauge_b)
        if fixed_kfs:
            const_kfs |= {k for k in fixed_kfs if k in cam_q}
        const_kfs |= {k for k in extra_frames if k in cam_q}
        for k in const_kfs:
            # 纳入了窗口外共视帧之后，某些固定帧可能一条有效观测都没有，
            # 它的参数块就不会被加进 problem，直接设常量会触发 Ceres 的断言。
            if not problem.has_parameter_block(cam_q[k]):
                continue
            problem.set_parameter_block_constant(cam_q[k])
            problem.set_parameter_block_constant(cam_t[k])
        if not optimize_principal:
            problem.set_parameter_block_constant(intr_cx)
            problem.set_parameter_block_constant(intr_cy)
        if fix_focal:
            # 已提供标定内参时固定焦距，避免 BA 把它拉偏造成非刚性形变
            problem.set_parameter_block_constant(intr_f)
        elif focal_prior is not None and focal_prior_sigma > 0:
            # 软先验：局部 BA 里把焦距钉在当前值附近，避免共享常量块的崩溃
            _w = 1.0 / (float(focal_prior_sigma) ** 2)
            problem.add_residual_block(
                _ScalarPriorCost(float(focal_prior), _w), None, [intr_f])
        # k1 与焦距强相关：不优化时固定当前值；优化时加一个弱先验拉向 0，
        # 避免少量广角观测把畸变项吃掉。
        if not optimize_k1:
            problem.set_parameter_block_constant(intr_k1)
        elif k1_prior_sigma > 0:
            _wk = 1.0 / (float(k1_prior_sigma) ** 2)
            problem.add_residual_block(
                _ScalarPriorCost(0.0, _wk), None, [intr_k1])
        if not optimize_points or fixed_pts:
            for pid in all_pt_ids:
                if not optimize_points:
                    problem.set_parameter_block_constant(pt_params[pid])
                elif fixed_pts and pid in fixed_pts:
                    problem.set_parameter_block_constant(pt_params[pid])

        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
        options.max_num_iterations = max_iter
        options.num_threads = SOLVER_NUM_THREADS
        options.function_tolerance = 1e-6
        options.gradient_tolerance = 1e-6
        options.parameter_tolerance = 1e-6
        summary = pyceres.SolverSummary()
        try:
            pyceres.solve(options, problem, summary)
        except Exception as e:
            logger.warning("[BA] pyceres 失败: %s", e)
            if em_iter == 0:
                return focal, fy, cx, cy, k1
            break
        if not summary.IsSolutionUsable():
            logger.warning("[BA] 求解器未正常终止: %s", summary.message)
            if em_iter == 0:
                return focal, fy, cx, cy, k1
            break
        # 最终 BA 回滚：若本轮迭代把代价弄大了，恢复到本轮开始前的参数
        if (summary.final_cost > summary.initial_cost * 1.0001 + 1e-9):
            for k in keyframe_ids:
                cam_q[k][:] = _prev_q[k]
                cam_t[k][:] = _prev_t[k]
            for pid in all_pt_ids:
                pt_params[pid][:] = _prev_pt[pid]
            intr_f[0], intr_cx[0], intr_cy[0], intr_k1[0] = _prev_intr
            logger.warning("[BA] 代价上升 %.3f -> %.3f，回滚本轮（%s）",
                           summary.initial_cost, summary.final_cost,
                           "global" if is_global else "local")
            break
        last_success = True
        (logger.info if is_global else logger.debug)(
            "[BA] %s frames=%d pts=%d obs=%d em=%d/%d cost %.2f->%.2f iters=%d %s",
            "global" if is_global else "local", len(keyframe_ids),
            len(all_pt_ids), n_obs, em_iter + 1, total_iters,
            summary.initial_cost, summary.final_cost,
            summary.num_successful_steps, summary.message[:48])

        if total_iters <= 1 or em_iter == total_iters - 1:
            break

        residuals = np.zeros((n_obs, 2), dtype=np.float64)
        for i, (f, pid, u, v) in enumerate(obs_list):
            pred = _reproj_residual(cam_q[f], cam_t[f], intr_f[0],
                                    intr_cx[0], intr_cy[0], pt_params[pid],
                                    intr_k1[0])
            residuals[i, 0] = pred[0] - u
            residuals[i, 1] = pred[1] - v
        new_weights = _em_inlier_weights(residuals)
        if np.max(np.abs(new_weights - obs_weights)) < 0.02:
            break
        obs_weights = new_weights

    if not last_success:
        return focal, fy, cx, cy, k1

    for k in keyframe_ids:
        R = Rot.from_quat(cam_q[k]).as_matrix()
        frames[k].pose = CameraPose(R, cam_t[k].reshape(3, 1))
    for pid in all_pt_ids:
        map_points[pid].xyz = pt_params[pid].astype(np.float32)

    focal_new = float(intr_f[0])
    cx_new = float(intr_cx[0])
    cy_new = float(intr_cy[0])
    k1_new = float(intr_k1[0])
    # BA 只优化一个等向焦距 f；fy 必须按进入 BA 前的比例跟着更新，
    # 否则 fx 收敛后 fy 仍停在初值，后续 PnP / 过滤会用各向异性的 K。
    _fy_ratio = (float(fy) / float(focal)) if abs(float(focal)) > 1e-12 else 1.0
    fy_new = focal_new * _fy_ratio
    if _state is not None:
        _state["success"] = True
        _state["obs_weights"] = obs_weights
        _state["k1"] = k1_new
    return focal_new, fy_new, cx_new, cy_new, k1_new


def _pose_to_arr8(pose: CameraPose, s=1.0) -> np.ndarray:
    q = Rot.from_matrix(pose.R).as_quat()
    return np.ascontiguousarray(np.array([s, *pose.t.flatten(), *q], dtype=np.float64))


def _arr8_to_pose(arr: np.ndarray) -> CameraPose:
    R = Rot.from_quat(arr[4:8]).as_matrix()
    return CameraPose(R, arr[1:4].reshape(3, 1))


def pose_graph_optimize(keyframes, poses, edges, iters=None,
                        cauchy_delta=PGO_CAUCHY_DELTA, edge_weights=None):
    if iters is None:
        iters = _cfg().pgo_iters
    if not keyframes or len(edges) < 2:
        return poses
    n = len(keyframes)
    if n < 2:
        return poses

    problem = pyceres.Problem()
    loss = pyceres.CauchyLoss(cauchy_delta)
    pose_params = {}
    for kf in keyframes:
        if kf not in poses:
            continue
        if poses[kf] is None:
            continue
        pose_params[kf] = _pose_to_arr8(poses[kf])

    if len(pose_params) < 2:
        return poses

    valid_edges = []
    for idx_e, (a, b, T) in enumerate(edges):
        if a in pose_params and b in pose_params:
            if edge_weights is None:
                w = 1.0
            else:
                w = float(edge_weights.get(idx_e, 1.0))
            valid_edges.append((a, b, T, w))
    if len(valid_edges) < 2:
        return poses

    # 节点拆成三个参数块：q(4, 带四元数 manifold) / t(3) / s(1)
    s_blocks, t_blocks, q_blocks = {}, {}, {}
    for kf, arr in pose_params.items():
        s_blocks[kf] = np.ascontiguousarray(arr[0:1], dtype=np.float64)
        t_blocks[kf] = np.ascontiguousarray(arr[1:4], dtype=np.float64)
        q_blocks[kf] = np.ascontiguousarray(arr[4:8], dtype=np.float64)

    for a, b, T, w in valid_edges:
        T_meas = np.eye(4)
        T_meas[:3, :3] = T.R
        T_meas[:3, 3] = T.t.flatten()
        cost = _Sim3PGOEdge(T_meas, weight=w)
        problem.add_residual_block(
            cost, loss,
            [q_blocks[a], t_blocks[a], s_blocks[a],
             q_blocks[b], t_blocks[b], s_blocks[b]])

    fixed_kf = keyframes[0]
    if fixed_kf in s_blocks and problem.has_parameter_block(s_blocks[fixed_kf]):
        problem.set_parameter_block_constant(q_blocks[fixed_kf])
        problem.set_parameter_block_constant(t_blocks[fixed_kf])
        problem.set_parameter_block_constant(s_blocks[fixed_kf])

    for kf in pose_params:
        if problem.has_parameter_block(q_blocks[kf]):
            problem.set_manifold(q_blocks[kf], pyceres.EigenQuaternionManifold())

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = iters
    options.num_threads = SOLVER_NUM_THREADS
    options.function_tolerance = 1e-6
    options.gradient_tolerance = 1e-6
    options.parameter_tolerance = 1e-6
    summary = pyceres.SolverSummary()
    try:
        pyceres.solve(options, problem, summary)
    except Exception as e:
        logger.warning("[PGO] pyceres 失败: %s", e)
        return poses

    if not summary.IsSolutionUsable():
        logger.warning("[PGO] 求解器未正常终止, 返回原始位姿: %s", summary.message)
        return poses

    logger.info("[PGO] nodes=%d edges=%d cost %.4f->%.4f (%s)",
                len(pose_params), len(valid_edges), summary.initial_cost,
                summary.final_cost, summary.message[:48])
    new_poses = dict(poses)
    for kf in pose_params:
        if not problem.has_parameter_block(s_blocks[kf]):
            continue
        arr = np.concatenate([[float(s_blocks[kf][0])], t_blocks[kf], q_blocks[kf]])
        new_poses[kf] = _arr8_to_pose(arr)
    return new_poses


def _pgo_edge_error(poses, edge):
    """一条 PGO 边在当前解下的 (旋转°, 平移) 残差。"""
    a, b, T_meas = edge
    pa, pb = poses.get(a), poses.get(b)
    if pa is None or pb is None:
        return None
    T_cur = _relative_pose(pa, pb)
    T_meas_mat = np.eye(4)
    T_meas_mat[:3, :3] = T_meas.R
    T_meas_mat[:3, 3] = T_meas.t.flatten()
    T_cur_mat = np.eye(4)
    T_cur_mat[:3, :3] = T_cur.R
    T_cur_mat[:3, 3] = T_cur.t.flatten()
    T_err = T_meas_mat @ np.linalg.inv(T_cur_mat)
    rv, _ = cv2.Rodrigues(T_err[:3, :3])
    return float(np.degrees(np.linalg.norm(rv))), float(np.linalg.norm(T_err[:3, 3]))


def _pgo_optimize_filtered(keyframes, poses_dict, edges, weights, scale_pgo,
                           n_seq, max_rot_deg=10.0, max_trans_ratio=0.05,
                           max_rounds=3):
    """逐边淘汰的 PGO：解完按回环边残差剔边，再从原始位姿重解，最多 max_rounds 轮。

    旋转 > max_rot_deg 或平移 > max_trans_ratio×尺度即剔除；前 n_seq 条序列边
    始终保留（轨迹骨架）。返回 (new_poses, kept_indices)，解不出返回 (None, [])。
    """
    active = list(range(len(edges)))
    new_poses = None
    for rnd in range(max_rounds):
        if len(active) < 3:
            logger.warning("[PGO] 逐边过滤后只剩 %d 条边，放弃 PGO", len(active))
            return None, []
        sub_edges = [edges[i] for i in active]
        sub_w = {j: weights[i] for j, i in enumerate(active)}
        new_poses = pose_graph_optimize(keyframes, poses_dict, sub_edges,
                                        edge_weights=sub_w)
        bad = set()
        for i in active:
            if i < n_seq:
                continue
            err = _pgo_edge_error(new_poses, edges[i])
            if err is None:
                continue
            rot_deg, trans = err
            if rot_deg > max_rot_deg or trans > max_trans_ratio * scale_pgo:
                bad.add(i)
        if not bad:
            logger.info("[PGO] 逐边过滤 %d 轮收敛：保留 %d/%d 条边（回环 %d/%d）",
                        rnd + 1, len(active), len(edges),
                        sum(1 for i in active if i >= n_seq), len(edges) - n_seq)
            return new_poses, active
        logger.info("[PGO] 第 %d 轮剔除 %d 条回环边（旋转 >%.0f° 或平移 >%.2f×尺度）",
                    rnd + 1, len(bad), max_rot_deg, max_trans_ratio)
        active = [i for i in active if i not in bad]
    return new_poses, active



# ---- 8. 点云管理 ----
def _compute_point_errors(map_points, poses_by_frame, focal, fy, cx, cy):
    N = len(map_points)
    mean_err = np.full(N, np.inf, dtype=np.float64)
    valid_count = np.zeros(N, dtype=np.int32)
    neg_ratio = np.zeros(N, dtype=np.float64)
    if N == 0:
        return mean_err, valid_count, neg_ratio
    xyz_all = np.full((N, 3), np.nan, dtype=np.float64)
    for i, pt in enumerate(map_points):
        if pt.xyz.size == 3:
            xyz_all[i] = pt.xyz
    invalid_pts = ~np.isfinite(xyz_all).all(axis=1)
    neg_ratio[invalid_pts] = 1.0
    flat = [
        (i, o.frame_idx, o.u, o.v)
        for i, pt in enumerate(map_points)
        if not invalid_pts[i]
        for o in pt.obs
    ]
    if not flat:
        return mean_err, valid_count, neg_ratio
    arr = np.asarray(flat, dtype=np.float64)
    pt_ids = arr[:, 0].astype(np.int64)
    f_ids = arr[:, 1].astype(np.int64)
    us = arr[:, 2]
    vs = arr[:, 3]
    max_f = int(f_ids.max()) + 1 if len(f_ids) else 0
    R_all = np.zeros((max_f, 3, 3))
    t_all = np.zeros((max_f, 3))
    pose_ok = np.zeros(max_f, dtype=bool)
    for f, pose in poses_by_frame.items():
        if f < max_f and pose is not None:
            R_all[f] = pose.R
            t_all[f] = pose.t.flatten()
            pose_ok[f] = True
    keep = (~invalid_pts[pt_ids]) & pose_ok[f_ids]
    if not np.any(keep):
        return mean_err, valid_count, neg_ratio
    pt_ids = pt_ids[keep]
    f_ids = f_ids[keep]
    us = us[keep]
    vs = vs[keep]
    xyz_o = xyz_all[pt_ids]
    R_o = R_all[f_ids]
    t_o = t_all[f_ids]
    pt_cam = np.einsum("nij,nj->ni", R_o, xyz_o) + t_o
    depth = pt_cam[:, 2]
    valid_depth = depth > 0
    depth_safe = np.where(valid_depth, depth, 1.0)
    pu = focal * (pt_cam[:, 0] / depth_safe) + cx
    pv = fy * (pt_cam[:, 1] / depth_safe) + cy
    err = np.sqrt((pu - us) ** 2 + (pv - vs) ** 2)
    err[~valid_depth] = 0.0
    total = np.bincount(pt_ids, minlength=N)
    vd = pt_ids[valid_depth]
    neg = pt_ids[~valid_depth]
    valid_count[:] = np.bincount(vd, minlength=N)
    err_sum = np.bincount(vd, weights=err[valid_depth], minlength=N)
    neg_count = np.bincount(neg, minlength=N)
    has_valid = valid_count > 0
    mean_err[has_valid] = err_sum[has_valid] / valid_count[has_valid]
    has_obs = total > 0
    neg_ratio[has_obs] = neg_count[has_obs] / total[has_obs]
    neg_ratio[invalid_pts] = 1.0
    return mean_err, valid_count, neg_ratio


def _vectorized_tri_angles(map_points, frames, mode="min"):
    """每个点所有相机对的三角化角极值；`min` 取最小角，`max` 取最大角。"""
    N = len(map_points)
    angles = np.full(N, 180.0 if mode == "min" else 0.0, dtype=np.float64)
    max_obs = 0
    for pt in map_points:
        if len(pt.obs) > max_obs:
            max_obs = len(pt.obs)
    if max_obs < 2:
        return angles
    cams = np.full((N, max_obs, 3), np.nan, dtype=np.float64)
    xyz = np.array([p.xyz for p in map_points], dtype=np.float64)
    for i, pt in enumerate(map_points):
        for j, o in enumerate(pt.obs):
            f = frames.get(o.frame_idx)
            if f is not None and f.pose is not None:
                cams[i, j] = f.pose.center
    rays = xyz[:, None, :] - cams
    norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / np.maximum(norms, 1e-12)
    for m in range(max_obs):
        for nn in range(m + 1, max_obs):
            valid = np.isfinite(cams[:, m, 0])
            valid = valid & np.isfinite(cams[:, nn, 0])
            if not np.any(valid):
                continue
            cos = np.einsum("ij,ij->i", rays[:, m], rays[:, nn])
            cos = np.clip(cos, -1.0, 1.0)
            a = np.degrees(np.arccos(cos))
            if mode == "min":
                a = np.where(valid, a, 180.0)
                angles = np.minimum(angles, a)
            else:
                # 无效相机对不参与最大角；一对有效都没有的点保持 0°，会被过滤掉
                a = np.where(valid, a, -1.0)
                angles = np.maximum(angles, a)
    return angles


def filter_observations(map_points, frames, focal, fy, cx, cy,
                        thresh_px=None, min_obs=MIN_OBSERVATIONS):
    """逐观测硬剔除（重投影过大 / 负深度 / 无位姿）。

    与 BA 的 EM 软权重不同，这里真正删除观测，剩余观测不足的点一并删除。
    """
    if not map_points:
        return 0, 0
    if thresh_px is None:
        thresh_px = MAX_REPROJ_ERROR * _cfg().reproj_thresh
    K = _build_K(focal, fy, cx, cy).astype(np.float64)
    n_obs_removed = 0

    for mp in map_points:
        if not np.isfinite(mp.xyz).all() or not mp.obs:
            n_obs_removed += len(mp.obs)
            mp.obs = []
            mp.obs_set = set()
            continue
        X = np.asarray(mp.xyz, np.float64).reshape(3)
        keep = []
        for o in mp.obs:
            fr = frames.get(o.frame_idx)
            if fr is None or fr.pose is None:
                n_obs_removed += 1
                continue
            P = fr.pose
            Xc = P.R.astype(np.float64) @ X + P.t.astype(np.float64).reshape(3)
            if Xc[2] <= 1e-9:                      # 负深度 / 退化
                n_obs_removed += 1
                continue
            uv = K @ Xc
            err = float(np.hypot(uv[0] / uv[2] - o.u, uv[1] / uv[2] - o.v))
            if not np.isfinite(err) or err > thresh_px:
                n_obs_removed += 1
                continue
            keep.append(o)
        if len(keep) != len(mp.obs):
            mp.obs = keep
            mp.obs_set = {(o.frame_idx, o.kp_idx) for o in keep}

    keep_mask = np.array([mp.obs_count >= min_obs for mp in map_points], dtype=bool)
    n_pts_removed = int((~keep_mask).sum())
    if n_pts_removed:
        map_points[:] = [p for p, k in zip(map_points, keep_mask) if k]
        for i, p in enumerate(map_points):
            p.idx = i

    # 以 obs 列表为准重建 feat_map，避免删除观测后留下悬空索引
    for fr in frames.values():
        fm = fr.feat_map
        if fm is not None:
            fm[:] = -1
    for pi, mp in enumerate(map_points):
        for o in mp.obs:
            fr = frames.get(o.frame_idx)
            if (fr is not None and fr.feat_map is not None
                    and 0 <= o.kp_idx < fr.feat_map.size):
                fr.feat_map[o.kp_idx] = pi

    if n_obs_removed or n_pts_removed:
        logger.info("[filter] 剔除 %d 条观测、%d 个点，剩余 %d 点",
                    n_obs_removed, n_pts_removed, len(map_points))
    return n_obs_removed, n_pts_removed


def _check_state_deep():
    """SFM_CHECK_STATE=2：每个注册帧结束后都查一次，定位漂移第一次出现的操作。"""
    return os.environ.get("SFM_CHECK_STATE", "0").strip() == "2"


def _check_state(frames, map_points, tag):
    """状态不变量检查（SFM_CHECK_STATE=1，默认关；只诊断，不修改）。

    校验 MapPoint.idx、obs/obs_set、feat_map 三处互指一致。这类漂移不会自己报错，
    只会让后续 BA 用错数据，因此发现即抛 AssertionError。
    """
    if os.environ.get("SFM_CHECK_STATE", "0").strip() in ("", "0"):
        return
    n = len(map_points)
    problems = []
    for i, p in enumerate(map_points):
        if p.idx != i:
            problems.append(f"point {i}: idx={p.idx}")
        if len(p.obs) != len(p.obs_set):
            problems.append(f"point {i}: len(obs)={len(p.obs)} len(obs_set)={len(p.obs_set)}")
        for o in p.obs:
            fr = frames.get(o.frame_idx)
            if fr is None:
                problems.append(f"point {i}: obs 指向不存在的帧 {o.frame_idx}")
                continue
            fm = fr.feat_map
            if fm is None or not (0 <= o.kp_idx < fm.size):
                problems.append(f"point {i}: kp_idx {o.kp_idx} 越界")
            elif int(fm[o.kp_idx]) != i:
                problems.append(f"point {i}: feat_map[{o.frame_idx}][{o.kp_idx}]="
                                f"{int(fm[o.kp_idx])} 不回指")
    for k, fr in frames.items():
        fm = fr.feat_map
        if fm is None:
            continue
        if fm.size != len(fr.kps):
            problems.append(f"frame {k}: feat_map {fm.size} != kps {len(fr.kps)}")
        bad = (fm >= n) | (fm < -1)
        if np.any(bad):
            idxs = np.where(bad)[0][:5]
            problems.append(f"frame {k}: {int(bad.sum())} 个越界 feat_map，"
                            f"例 {[(int(j), int(fm[j])) for j in idxs]}")
    if problems:
        for msg in problems[:20]:
            logger.error("[state] %s: %s", tag, msg)
        logger.error("[state] %s: 共 %d 处不一致", tag, len(problems))
        raise AssertionError(f"state inconsistency at {tag}: {len(problems)}")
    logger.debug("[state] %s: 一致（%d 帧 / %d 点）", tag, len(frames), n)


def _prune_obs_first(map_points, frames, reproj_thresh, focal, fy, cx, cy):
    """先逐观测剔除，再只按结构判据（三角化角）删点。

    逐观测删不会让单个坏观测带走整条 track，也不依赖 MAD 阈值而随历史漂移。
    """
    n_obs, n_pts = filter_observations(
        map_points, frames, focal, fy, cx, cy,
        thresh_px=MAX_REPROJ_ERROR * reproj_thresh)
    if not map_points:
        return
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    tri_angles = _vectorized_tri_angles(map_points, frames, _tri_angle_mode())
    to_remove = tri_angles < min_tri_deg
    n_angle = int(to_remove.sum())
    if n_angle:
        keep = ~to_remove
        idx_map = np.full(len(map_points), -1, dtype=np.int64)
        idx_map[keep] = np.arange(int(keep.sum()))
        for frame in frames.values():
            fm = frame.feat_map
            if fm is None or fm.size == 0:
                continue
            oob = fm >= len(map_points)
            if np.any(oob):
                fm = np.where(oob, -1, fm)
            valid = fm >= 0
            new_fm = np.full(fm.shape, -1, dtype=np.int64)
            if np.any(valid):
                new_fm[valid] = idx_map[fm[valid]]
            frame.feat_map = new_fm
        new_points = [p for i, p in enumerate(map_points) if keep[i]]
        for new_idx, pt in enumerate(new_points):
            pt.idx = new_idx
        map_points[:] = new_points
    logger.info("[prune] 逐观测：删 %d 条观测 / %d 个点；角度：删 %d 个点；剩余 %d",
                n_obs, n_pts, n_angle, len(map_points))


def prune_map_points(map_points, frames, reproj_thresh, focal, fy, cx, cy):
    if not map_points:
        return
    if os.environ.get("SFM_PRUNE_OBS_ONLY", "0").strip() not in ("", "0"):
        _prune_obs_first(map_points, frames, reproj_thresh, focal, fy, cx, cy)
        return
    poses_by_frame = {f: fr.pose for f, fr in frames.items()}
    mean_err, valid_count, neg_ratio = _compute_point_errors(map_points, poses_by_frame, focal, fy, cx, cy)
    valid_errors = mean_err[np.isfinite(mean_err) & (valid_count >= MIN_OBSERVATIONS)]
    if len(valid_errors) > 10:
        mad = float(np.median(np.abs(valid_errors - np.median(valid_errors))))
        err_thresh = max(MAX_REPROJ_ERROR * reproj_thresh, 3.0 * mad)
    else:
        err_thresh = MAX_REPROJ_ERROR * reproj_thresh
    cfg = _cfg()
    min_tri_deg = cfg.filter_min_tri_angle_deg
    tri_angles = _vectorized_tri_angles(map_points, frames, _tri_angle_mode())
    obs_counts = np.array([p.obs_count for p in map_points])
    to_remove_mask = (
        (obs_counts < MIN_OBSERVATIONS)
        | (neg_ratio >= 0.5)
        | ((valid_count > 0) & (mean_err > err_thresh))
        | (tri_angles < min_tri_deg)
    )
    n_remove = int(to_remove_mask.sum())
    if n_remove == 0:
        return
    keep = ~to_remove_mask
    idx_map = np.full(len(map_points), -1, dtype=np.int64)
    idx_map[keep] = np.arange(int(keep.sum()))
    for frame in frames.values():
        fm = frame.feat_map
        if fm is None or fm.size == 0:
            continue
        oob = fm >= len(map_points)
        if np.any(oob):
            fm = np.where(oob, -1, fm)
        valid = fm >= 0
        new_fm = np.full(fm.shape, -1, dtype=np.int64)
        if np.any(valid):
            new_fm[valid] = idx_map[fm[valid]]
        frame.feat_map = new_fm
    new_points = [p for i, p in enumerate(map_points) if keep[i]]
    for new_idx, pt in enumerate(new_points):
        pt.idx = new_idx
    map_points[:] = new_points
    logger.info("[prune] 删除 %d 个点, 剩余 %d。", n_remove, len(map_points))


def _global_refine(map_points, frames, focal, fy, cx, cy,
                   reproj_thresh, image_size, focal_init, fix_focal,
                   norm_type, covis_neighbors=10,
                   em_iters: int = BA_EM_ITERS,
                   max_iter: Optional[int] = None,
                   complete_frames: Optional[List[int]] = None,
                   focal_prior_sigma: float = 0.0,
                   k1: float = 0.0,
                   optimize_k1: bool = False,
                   ) -> Tuple[float, float, float, float, float]:
    """全局 BA → 硬过滤 → 补 track → 重三角化 → 点合并。

    complete_frames 只传新增帧（O(帧×点)）；增量阶段焦距必须用 focal_prior_sigma 拴住，否则跑偏 +45%。
    """
    # 合并验证 + BA 前删负深度（固定开启）
    _merge_validate = True
    _neg_depth_pre = True
    _refine_loop = (os.environ.get("SFM_REFINE_LOOP", "0").strip()
                    not in ("", "0"))
    all_registered = _registered_ba_frames(frames, map_points)
    if len(all_registered) < 2:
        return focal, fy, cx, cy, k1
    window = all_registered[-max(10, covis_neighbors):]
    if len(window) >= 2:
        _em_reassign_local(window, map_points, frames, focal, fy, cx, cy)
    if _refine_loop:
        # 先清理再解：补 track → 重三角化 → 合并验证，再 BA ↔ 过滤最多 2 轮；
        # 删除量 <0.1% 或平均重投影变化 <5e-4 早停。
        K = _build_K(focal, fy, cx, cy)
        if complete_frames:
            _complete_tracks(complete_frames, frames, map_points, K, reproj_thresh)
        _retriangulate_multiview(map_points, frames, K, min_obs=MULTIVIEW_MIN_OBS)
        _merge_map_points(map_points, frames, norm_type, K=K, k1=k1,
                          validate=_merge_validate)
        prev_err = None
        for _rnd in range(2):
            if _neg_depth_pre:
                # 只剔负深度 / 无位姿观测（阈值放大到 1e9），不动重投影过滤
                filter_observations(map_points, frames, focal, fy, cx, cy,
                                    thresh_px=1e9)
            state = {"success": False}
            focal, fy, cx, cy, k1 = bundle_adjustment(
                all_registered, map_points, frames, focal, fy, cx, cy,
                optimize_points=True, reproj_thresh=reproj_thresh,
                image_size=image_size, is_global=True,
                focal_init=focal_init, optimize_principal=False,
                fix_focal=fix_focal, _state=state,
                max_iter=max_iter, em_iters=em_iters,
                focal_prior=(focal if focal_prior_sigma > 0 else None),
                focal_prior_sigma=focal_prior_sigma,
                k1=k1, optimize_k1=optimize_k1,
            )
            if not state.get("success"):
                break
            n_before = sum(p.obs_count for p in map_points)
            filter_observations(map_points, frames, focal, fy, cx, cy)
            n_after = sum(p.obs_count for p in map_points)
            poses_by_frame = {k: f.pose for k, f in frames.items()
                              if f.pose is not None}
            mean_err, _, _ = _compute_point_errors(
                map_points, poses_by_frame, focal, fy, cx, cy)
            finite = mean_err[np.isfinite(mean_err)]
            cur = float(finite.mean()) if finite.size else 0.0
            removed_frac = (n_before - n_after) / max(n_before, 1)
            if prev_err is not None and (
                    removed_frac < 1e-3
                    or abs(prev_err - cur) <= 5e-4 * max(cur, 1e-12)):
                break
            prev_err = cur
        return focal, fy, cx, cy, k1
    state = {"success": False}
    focal, fy, cx, cy, k1 = bundle_adjustment(
        all_registered, map_points, frames, focal, fy, cx, cy,
        optimize_points=True, reproj_thresh=reproj_thresh,
        image_size=image_size, is_global=True,
        focal_init=focal_init, optimize_principal=False,
        fix_focal=fix_focal, _state=state,
        max_iter=max_iter, em_iters=em_iters,
        focal_prior=(focal if focal_prior_sigma > 0 else None),
        focal_prior_sigma=focal_prior_sigma,
        k1=k1, optimize_k1=optimize_k1,
    )
    if state.get("success"):
        filter_observations(map_points, frames, focal, fy, cx, cy)
    K = _build_K(focal, fy, cx, cy)
    if complete_frames:
        _complete_tracks(complete_frames, frames, map_points, K, reproj_thresh)
    _retriangulate_multiview(map_points, frames, K, min_obs=MULTIVIEW_MIN_OBS)
    _merge_map_points(map_points, frames, norm_type, K=K, k1=k1,
                      validate=_merge_validate)
    return focal, fy, cx, cy, k1


def filter_point_cloud(map_points, poses_by_frame, focal, fy, cx, cy, reproj_thresh):
    if not map_points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
    xyz = np.array([p.xyz for p in map_points], dtype=np.float32)
    mean_err, valid_count, neg_ratio = _compute_point_errors(map_points, poses_by_frame, focal, fy, cx, cy)
    errors = np.where((valid_count >= MIN_OBSERVATIONS) & (neg_ratio < 0.5), mean_err, np.inf)
    finite = np.isfinite(errors)
    if not np.any(finite):
        return xyz, np.zeros(len(map_points), dtype=bool)
    median = float(np.median(errors[finite]))
    thresh = max(median * FILTER_MEDIAN_FACTOR, reproj_thresh * FILTER_REPROJ_FACTOR)
    mask = finite & (errors < thresh)
    return xyz, mask


def _resolve_roots(parent):
    """并查集最终根（路径压缩）。parent 既可能是 list 也可能是 numpy 数组。"""
    n = len(parent)
    out = np.empty(n, dtype=np.int64)
    if isinstance(parent, np.ndarray):
        for i in range(n):
            r = i
            while parent[r] != r:
                r = int(parent[r])
            out[i] = r
        return out
    for i in range(n):
        r = i
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        out[i] = r
    return out


def _merge_pairs_py(map_points, buckets, offsets, pos_thresh, desc_thresh,
                    norm_type, find, union):
    """_merge_map_points 的逐对回退实现（无 Cython 时使用）。

    必须覆盖同格点对与单点 bucket：bucket 边长是 2×pos_thresh，近点可能同格，
    漏掉会留下重复点。
    """
    def _try(i, j):
        if find(i) == find(j):
            return
        pi = map_points[i]
        pj = map_points[j]
        if pi.obs_set == pj.obs_set:
            return
        if np.linalg.norm(pi.xyz - pj.xyz) > pos_thresh:
            return
        if pi.desc is not None and pj.desc is not None:
            if pi.desc.dtype == pj.desc.dtype:
                if norm_type == cv2.NORM_HAMMING:
                    d = int(cv2.norm(pi.desc, pj.desc, cv2.NORM_HAMMING))
                else:
                    d = float(cv2.norm(pi.desc, pj.desc, cv2.NORM_L2))
                if d > desc_thresh:
                    return
        union(i, j)

    for key, ids in buckets.items():
        for dx, dy, dz in offsets:
            nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
            if nkey < key:
                continue
            if nkey == key:
                for ai in range(len(ids)):
                    for bi in range(ai + 1, len(ids)):
                        _try(ids[ai], ids[bi])
                continue
            neighbors = buckets.get(nkey)
            if not neighbors:
                continue
            for i in ids:
                for j in neighbors:
                    _try(i, j)


def _merge_scan_accel(map_points, buckets, offsets, pos_thresh, desc_thresh,
                      norm_type):
    """候选对在 Python 生成，逐对判据与并查集交给 Cython。

    描述子 dtype / 维度不一致时返回 None，交给慢路径。
    """
    n = len(map_points)
    desc_dtype = None
    desc_dim = 0
    for p in map_points:
        d = p.desc
        if d is None or d.ndim < 1:
            return None
        if desc_dtype is None:
            desc_dtype = d.dtype
            desc_dim = d.shape[-1]
        elif d.dtype != desc_dtype or d.shape[-1] != desc_dim:
            return None
    if desc_dtype is None:
        return None
    descs = np.empty((n, desc_dim), dtype=desc_dtype)
    xyz = np.zeros((n, 3), dtype=np.float64)
    for i, p in enumerate(map_points):
        descs[i] = p.desc
        a = p.xyz
        if a.size == 3:
            xyz[i] = a

    # obs_set 相同的点对直接跳过 —— 用组号代替每次集合比较
    gid = {}
    grp = np.empty(n, dtype=np.int64)
    for i, p in enumerate(map_points):
        key = frozenset(p.obs_set)
        g = gid.get(key)
        if g is None:
            g = len(gid)
            gid[key] = g
        grp[i] = g

    arrs = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
    pa_parts = []
    pb_parts = []
    for key, ia in arrs.items():
        for dx, dy, dz in offsets:
            nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
            if nkey < key:
                continue
            if nkey == key:
                # 同格配对（与 _merge_pairs_py 一致）
                if ia.size >= 2:
                    ii, jj = np.triu_indices(ia.size, k=1)
                    pa_parts.append(ia[ii])
                    pb_parts.append(ia[jj])
                continue
            ib = arrs.get(nkey)
            if ib is None:
                continue
            pa_parts.append(np.repeat(ia, ib.size))
            pb_parts.append(np.tile(ib, ia.size))
    if not pa_parts:
        return np.arange(n, dtype=np.int64)
    pa = np.concatenate(pa_parts)
    pb = np.concatenate(pb_parts)
    parent = np.empty(n, dtype=np.int64)
    metric = 1 if norm_type == cv2.NORM_HAMMING else 0
    sfm_accel.merge_scan(xyz, descs, pa, pb, grp, parent,
                         float(pos_thresh), float(desc_thresh), metric, 1)
    return parent


def _reproject_obs_errors(xyz, obs, frames, fx, fy, cx, cy, k1=0.0):
    """一组观测对给定 3D 点的重投影像素误差；无位姿 / 负深度记为 inf。"""
    X = np.asarray(xyz, np.float64).reshape(3)
    errs = np.empty(len(obs), dtype=np.float64)
    for i, o in enumerate(obs):
        fr = frames.get(o.frame_idx)
        if fr is None or fr.pose is None:
            errs[i] = np.inf
            continue
        Xc = (fr.pose.R.astype(np.float64) @ X
              + fr.pose.t.astype(np.float64).reshape(3))
        if Xc[2] <= 1e-9:
            errs[i] = np.inf
            continue
        x = Xc[0] / Xc[2]
        y = Xc[1] / Xc[2]
        rad = 1.0 + float(k1) * (x * x + y * y)
        errs[i] = np.hypot(fx * x * rad + cx - o.u, fy * y * rad + cy - o.v)
    return errs


def _merge_map_points(map_points, frames, norm_type, K=None, k1=0.0,
                      validate=False, max_reproj=4.0, min_inlier_ratio=0.9):
    # 全扫描：同格点对和单点 bucket 也必须参与比较，否则重复点会留到最后。
    if len(map_points) < 2:
        return 0
    xyz = np.array([p.xyz for p in map_points], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    if not np.any(finite):
        return 0
    c = xyz[finite].mean(axis=0)
    scene = float(np.median(np.linalg.norm(xyz[finite] - c, axis=1))) + 1e-6
    pos_thresh = MERGE_MAX_XYZ_DIST_FACTOR * scene
    if norm_type == cv2.NORM_HAMMING:
        desc_thresh = MERGE_MAX_DESC_DIST_ORB
    else:
        sample_desc = None
        for p in map_points:
            if p.desc is not None and len(p.desc) > 0:
                sample_desc = p.desc
                break
        if sample_desc is not None:
            if _is_rootsift_like(sample_desc):
                desc_thresh = MERGE_MAX_DESC_DIST_ROOTSIFT
            else:
                desc_thresh = MERGE_MAX_DESC_DIST_SIFT
        else:
            desc_thresh = MERGE_MAX_DESC_DIST_SIFT
    bucket_size = max(pos_thresh * 2.0, 1e-6)
    buckets = {}
    for i, p in enumerate(map_points):
        if not finite[i]:
            continue
        key = tuple(np.floor(p.xyz / bucket_size).astype(np.int64).tolist())
        buckets.setdefault(key, []).append(i)
    parent = list(range(len(map_points)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    _offsets = [(dx, dy, dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    if sfm_accel is not None and hasattr(sfm_accel, "merge_scan"):
        # 候选对生成留在 Python（仅 27 邻域查表），距离/描述子判断与并查集交给 Cython，
        # 避免逐对的 np.linalg.norm + cv2.norm 开销。
        _merged = _merge_scan_accel(map_points, buckets, _offsets,
                                    pos_thresh, desc_thresh, norm_type)
        if _merged is not None:
            parent = _merged
        else:
            _merge_pairs_py(map_points, buckets, _offsets, pos_thresh,
                            desc_thresh, norm_type, find, union)
    else:
        _merge_pairs_py(map_points, buckets, _offsets, pos_thresh,
                        desc_thresh, norm_type, find, union)
    roots = _resolve_roots(parent)
    groups = {}
    for i in range(len(map_points)):
        groups.setdefault(int(roots[i]), []).append(i)
    merged_count = 0
    new_points = []
    idx_map = np.full(len(map_points), -1, dtype=np.int64)
    if K is not None:
        _K64 = np.asarray(K, dtype=np.float64)
        _fx, _fy = float(_K64[0, 0]), float(_K64[1, 1])
        _cx, _cy = float(_K64[0, 2]), float(_K64[1, 2])
    validate = bool(validate and K is not None)
    for root, members in groups.items():
        if len(members) == 1:
            old_idx = members[0]
            new_idx = len(new_points)
            idx_map[old_idx] = new_idx
            map_points[old_idx].idx = new_idx
            new_points.append(map_points[old_idx])
            continue
        members_sorted = sorted(members, key=lambda m: -map_points[m].obs_count)
        base = map_points[members_sorted[0]]
        if validate:
            # 合并前先按 track 长度加权定新位置，再用全部观测验证重投影：
            # 一票否决，不合并就保持独立——合并错点比不合并更伤后续 BA。
            weights = np.array([max(1, map_points[m].obs_count) for m in members],
                               dtype=np.float64)
            xyzs = np.array([map_points[m].xyz for m in members], dtype=np.float64)
            if np.isfinite(xyzs).all():
                new_xyz = (xyzs * weights[:, None]).sum(axis=0) / weights.sum()
            else:
                new_xyz = np.asarray(base.xyz, dtype=np.float64)
            obs_all = []
            seen_keys = set()
            for m in members:
                for o in map_points[m].obs:
                    key = (o.frame_idx, o.kp_idx)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        obs_all.append(o)
            errs = _reproject_obs_errors(new_xyz, obs_all, frames,
                                         _fx, _fy, _cx, _cy, k1)
            ok = np.isfinite(errs) & (errs <= max_reproj)
            if len(errs) < 2 or float(ok.mean()) < min_inlier_ratio:
                for m in members:
                    new_idx = len(new_points)
                    idx_map[m] = new_idx
                    map_points[m].idx = new_idx
                    new_points.append(map_points[m])
                continue
            base.xyz = new_xyz.astype(np.asarray(base.xyz).dtype, copy=False)
        for m in members_sorted[1:]:
            for o in map_points[m].obs:
                base.add_observation(o)
            merged_count += 1
        new_idx = len(new_points)
        for m in members:
            idx_map[m] = new_idx
        base.idx = new_idx
        new_points.append(base)
    for fr in frames.values():
        fm = fr.feat_map
        if fm is None or fm.size == 0:
            continue
        oob = fm >= len(map_points)
        if np.any(oob):
            fm = np.where(oob, -1, fm)
        valid = fm >= 0
        new_fm = np.full(fm.shape, -1, dtype=np.int64)
        if np.any(valid):
            new_fm[valid] = idx_map[fm[valid]]
        fr.feat_map = new_fm
    map_points[:] = new_points
    return merged_count


# ---- 9. 三角化与初始化 ----
def _triangulate_from_pair(curr_idx, prev_idx, matches, inlier_mask, frames, map_points,
                           focal, fy, cx, cy, triang_thresh):
    # 不要求 prev_idx < curr_idx：按重叠度选下一帧时，参考帧可能在索引上更靠后。
    # 三角化本身与两帧的先后无关，matches 的 query/train 已经固定了方向。
    prev = frames[prev_idx]
    curr = frames[curr_idx]
    if prev.pose is None or curr.pose is None:
        return
    K = _build_K(focal, fy, cx, cy)
    P_prev = K @ np.asarray(prev.pose.RT[:3], dtype=np.float64)
    P_curr = K @ np.asarray(curr.pose.RT[:3], dtype=np.float64)
    if len(inlier_mask) == 0 or len(matches) == 0:
        return
    if len(inlier_mask) != len(matches):
        inlier_mask = np.ones(len(matches), dtype=bool)
    R_prev, t_prev = prev.pose.R, prev.pose.t
    R_curr, t_curr = curr.pose.R, curr.pose.t
    err_thresh = MAX_REPROJ_ERROR * triang_thresh
    err_thresh *= ADD_OBS_ERR_SCALE
    err_sq = err_thresh * err_thresh
    prev_feat_map = prev.feat_map
    curr_feat_map = curr.feat_map
    prev_colors = prev.colors
    curr_colors = curr.colors
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    add_to_prev = []
    add_to_curr = []
    new_match_ids = []
    for k in np.where(inlier_mask)[0]:
        m = matches[k]
        curr_feat = int(m.trainIdx)
        prev_feat = int(m.queryIdx)
        if curr_feat >= len(curr_feat_map):
            continue
        if prev_feat >= len(prev_feat_map):
            continue
        curr_bound = int(curr_feat_map[curr_feat])
        prev_bound = int(prev_feat_map[prev_feat])
        if curr_bound >= 0 and prev_bound >= 0:
            continue
        elif curr_bound >= 0 and prev_bound < 0:
            if curr_bound < len(map_points):
                add_to_prev.append((prev_feat, curr_bound))
        elif curr_bound < 0 and prev_bound >= 0:
            if prev_bound < len(map_points):
                add_to_curr.append((curr_feat, prev_bound))
        else:
            new_match_ids.append(int(k))

    def _try_add(frame, feat, pt_idx, idx_frame, R_f, t_f, colors_f):
        pt = map_points[pt_idx]
        if pt.obs_count >= MAX_OBS_PER_POINT:
            return False
        Xw = pt.xyz
        if Xw.size != 3 or not np.isfinite(Xw).all():
            return False
        if not _check_obs_angle(pt, idx_frame, frames, min_tri_deg):
            return False
        pt_cam = R_f @ Xw.reshape(3, 1) + t_f
        d = float(pt_cam[2, 0])
        if d <= 1e-6:
            return False
        u = focal * float(pt_cam[0, 0]) / d + cx
        v = fy * float(pt_cam[1, 0]) / d + cy
        dx = u - frame.kps[feat].pt[0]
        dy = v - frame.kps[feat].pt[1]
        if dx * dx + dy * dy > err_sq:
            return False
        if colors_f is not None and feat < len(colors_f):
            c = colors_f[feat]
        else:
            c = None
        if pt.add_observation(Observation(idx_frame, int(feat), float(frame.kps[feat].pt[0]),
                                          float(frame.kps[feat].pt[1]),
                                          (c.copy() if c is not None else None))):
            frame.feat_map[feat] = pt_idx
            _update_desc(pt, frame.desc[feat])
            return True
        return False

    for prev_feat, pt_idx in add_to_prev:
        _try_add(prev, prev_feat, pt_idx, prev_idx, R_prev, t_prev, prev_colors)
    for curr_feat, pt_idx in add_to_curr:
        _try_add(curr, curr_feat, pt_idx, curr_idx, R_curr, t_curr, curr_colors)

    if not new_match_ids:
        return
    pts_obs_map = {}
    for k in new_match_ids:
        m = matches[k]
        prev_feat = int(m.queryIdx)
        curr_feat = int(m.trainIdx)
        pts_obs_map[k] = [
            (prev_idx, float(prev.kps[prev_feat].pt[0]), float(prev.kps[prev_feat].pt[1])),
            (curr_idx, float(curr.kps[curr_feat].pt[0]), float(curr.kps[curr_feat].pt[1])),
        ]
    result = _triangulate_multiview_batch(pts_obs_map, frames, K, min_tri_deg, triang_thresh, _MEM_GUARD.current)
    for k, (X, valid) in result.items():
        if not valid:
            continue
        # result 的 key 就是 matches 的下标（pts_obs_map 以 k 为键）
        m = matches[k]
        prev_feat = int(m.queryIdx)
        curr_feat = int(m.trainIdx)
        if curr_feat_map[curr_feat] >= 0:
            continue
        if prev_feat_map[prev_feat] >= 0:
            continue
        pt_idx = len(map_points)
        if prev_colors is not None:
            c_prev = prev_colors[prev_feat] if prev_feat < len(prev_colors) else None
        else:
            c_prev = None
        pt = MapPoint(idx=pt_idx, xyz=X.astype(np.float32), desc=prev.desc[prev_feat].copy())
        pt.add_observation(Observation(prev_idx, prev_feat, float(prev.kps[prev_feat].pt[0]),
                                       float(prev.kps[prev_feat].pt[1]),
                                       (c_prev.copy() if c_prev is not None else None)))
        if curr_colors is not None:
            c_curr = curr_colors[curr_feat] if curr_feat < len(curr_colors) else None
        else:
            c_curr = None
        pt.add_observation(Observation(curr_idx, curr_feat, float(curr.kps[curr_feat].pt[0]),
                                       float(curr.kps[curr_feat].pt[1]),
                                       (c_curr.copy() if c_curr is not None else None)))
        map_points.append(pt)
        prev_feat_map[prev_feat] = pt_idx
        curr_feat_map[curr_feat] = pt_idx


def _init_candidate_pairs(n: int) -> List[Tuple[int, int]]:
    """初始对候选要覆盖整个数据集：图片集无序，相邻图在文件名里可能隔很远。

    n <= 150 取全对（约 8k 对，几秒可接受）；更大时均匀取锚点，锚点互连加局部邻域。
    """
    if n < 2:
        return []
    if n <= 150:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    anchors = list(range(0, n, max(1, n // 100)))
    if anchors[-1] != n - 1:
        anchors.append(n - 1)
    pairs = []
    for i in anchors:
        for d in (1, 2, 3, 5, 8, 13, 21, 34, 55):
            if i + d < n:
                pairs.append((i, i + d))
        for j in anchors:
            if j > i:
                pairs.append((i, j))
    return list(dict.fromkeys(pairs))


def _select_init_pair(frames, K, ransac_thresh, map_points, frames_dict,
                      focal, fy, cx, cy, triang_thresh,
                      init_seed=0, init_band=0, init_skip=0,
                      init_dump=None, init_pass=1):
    cfg = _cfg()
    n = len(frames)
    pairs = _init_candidate_pairs(n)

    def _quick_match(i, j):
        a, b = frames[i], frames[j]
        if a.desc is None or b.desc is None:
            return (i, j, 0)
        if len(a.desc) < 50 or len(b.desc) < 50:
            return (i, j, 0)
        nt, md, _ = get_metric(a.desc)
        n_use = min(500, len(a.desc), len(b.desc))
        try:
            return (i, j, len(_match_ratio_absolute(a.desc[:n_use], b.desc[:n_use], nt, md)))
        except Exception:
            return (i, j, 0)

    scored = Parallel(n_jobs=_MEM_GUARD.current, backend="threading")(delayed(_quick_match)(i, j) for i, j in pairs)
    scored = sorted(scored, key=lambda x: -x[2])
    if not scored:
        return None
    # 取高分候选，避免绝对门限把宽基线像对全部滤掉
    best = max(cnt for _, _, cnt in scored)
    cut = max(20, int(best * 0.15))
    selected = [x for x in scored if x[2] >= cut][:80]
    if not selected:
        selected = scored[:40]
    # 每个索引分段再补一个代表，保证覆盖整个数据集
    n_bands = min(8, n)
    for b in range(n_bands):
        lo = b * n // n_bands
        hi = (b + 1) * n // n_bands
        seg = [x for x in scored if lo <= x[0] < hi]
        if seg:
            selected.append(seg[0])
    pairs = list(dict.fromkeys((i, j) for i, j, _ in selected))
    # ---- 多起点内部开关：默认全 0 时不置换 ----
    # init_band>0 时只重排与最高分相差 <= band 个匹配的候选；
    # seed=0 不做置换，init_skip 跳过前若干个候选（用于强制换盆地）。
    score_map = {(int(i), int(j)): int(c) for i, j, c in scored}
    n_pairs_before = len(pairs)
    band_n = 0
    if init_band > 0 and init_seed > 0:
        thr = int(best) - int(init_band)
        band = [p for p in pairs if score_map.get(p, 0) >= thr]
        rest = [p for p in pairs if score_map.get(p, 0) < thr]
        if len(band) > 1:
            order = np.random.default_rng(int(init_seed)).permutation(len(band))
            pairs = [band[int(k)] for k in order] + rest
        band_n = len(band)
    if init_skip > 0:
        pairs = pairs[int(init_skip):]
    logger.info("[init] pass=%d 候选 %d（band=%d seed=%d skip=%d）",
                init_pass, n_pairs_before, band_n, init_seed, init_skip)

    def _dump(chosen_pair):
        if not init_dump:
            return
        try:
            os.makedirs(init_dump, exist_ok=True)
            payload = {
                "pass": int(init_pass),
                "seed": int(init_seed),
                "band": int(init_band),
                "skip": int(init_skip),
                "best": int(best),
                "cut": int(cut),
                "num_candidates": n_pairs_before,
                "band_size": band_n,
                "top200": [[int(i), int(j), int(c)] for i, j, c in scored[:200]],
                "chosen": ([chosen_pair[0], chosen_pair[1]]
                           if chosen_pair else None),
                "chosen_score": (score_map.get(tuple(chosen_pair))
                                 if chosen_pair else None),
                "gap_to_best": (int(best - score_map[tuple(chosen_pair)])
                                if chosen_pair and tuple(chosen_pair) in score_map
                                else None),
            }
            path = os.path.join(init_dump, f"init_pass{int(init_pass)}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            logger.info("[init] dump -> %s", path)
        except Exception as e:
            logger.warning("[init] dump 失败: %s", e)

    attempts = [
        (cfg.init_min_inliers, cfg.init_min_parallax_deg, 0.30),
        (max(40, cfg.init_min_inliers // 2), cfg.init_min_parallax_deg / 1.5, 0.20),
        (max(25, cfg.init_min_inliers // 3), cfg.init_min_parallax_deg / 2.0, 0.15),
    ]
    for attempt_idx, (min_inl, min_parallax, min_inl_ratio) in enumerate(attempts):
        for (i, j) in pairs:
            a, b = frames[i], frames[j]
            if a.desc is None or b.desc is None:
                continue
            norm_type, md, _ = get_metric(a.desc)
            matches = _match_ratio_absolute(a.desc, b.desc, norm_type, md)
            n_matches = len(matches)
            if n_matches < max(min_inl, 8):
                continue
            pts_a = np.array([a.kps[m.queryIdx].pt for m in matches], dtype=np.float32)
            pts_b = np.array([b.kps[m.trainIdx].pt for m in matches], dtype=np.float32)
            mask_fh, _, _ = _verify_matches_fh(pts_a, pts_b, ransac_thresh, min_inl)
            if mask_fh is None:
                continue
            if int(mask_fh.sum()) / max(n_matches, 1) < min_inl_ratio:
                continue
            E, mask_e = _find_essential(pts_a, pts_b, K, ransac_thresh)
            pose_b_local = pose_inl_local = None
            med_parallax_local = 0.0
            if E is not None and mask_e is not None:
                mask_e = mask_e.ravel().astype(bool)
                if int(mask_e.sum()) >= min_inl:
                    _, R_rel, t_rel, mask_pose = _recover_pose(E, pts_a, pts_b, K)
                    if mask_pose is not None:
                        pose_inl_local = mask_pose.ravel().astype(bool)
                        if int(pose_inl_local.sum()) >= min_inl:
                            ok, med = _check_baseline_sufficient(pts_a, pts_b, R_rel, t_rel, K, pose_inl_local, min_angle_deg=min_parallax)
                            if ok and med >= min_parallax:
                                t_norm = float(np.linalg.norm(t_rel))
                                if t_norm > 1e-12:
                                    pose_b_local = CameraPose(R_rel, (t_rel / t_norm).reshape(3, 1))
                                    med_parallax_local = med
            if pose_b_local is None:
                continue
            if pose_inl_local is None:
                continue
            if int(pose_inl_local.sum()) < min_inl:
                continue
            frames[i].pose = CameraPose(np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
            frames[j].pose = pose_b_local
            temp_points = []
            # 试三角化会往 feat_map 写入临时点索引；temp_points 随即丢弃，
            # 这里必须在返回/继续前恢复，否则真实建图会跳掉这些匹配。
            fm_i = frames_dict[i].feat_map.copy()
            fm_j = frames_dict[j].feat_map.copy()
            try:
                _triangulate_from_pair(j, i, matches, pose_inl_local, frames_dict, temp_points,
                                       focal, fy, cx, cy, triang_thresh)
            finally:
                frames_dict[i].feat_map[:] = fm_i
                frames_dict[j].feat_map[:] = fm_j
            frames[i].pose = None
            frames[j].pose = None
            if len(temp_points) < 30:
                logger.info("[init] (%d,%d) 三角化点数不足 (%d)", i, j, len(temp_points))
                continue
            frames[i].pose = CameraPose(np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
            frames[i].status = FrameStatus.KEYFRAME
            frames[i].is_keyframe = True
            frames[j].pose = pose_b_local
            frames[j].status = FrameStatus.KEYFRAME
            frames[j].is_keyframe = True
            _dump((int(i), int(j)))
            logger.info("[init] pass=%d 选中 pair=(%d,%d) score=%d 候选序号=%d",
                        init_pass, i, j, score_map.get((int(i), int(j)), -1),
                        [k for k, p in enumerate(pairs) if p == (i, j)][:1])
            return i, j, pose_b_local, matches, pose_inl_local, int(pose_inl_local.sum())
    _dump(None)
    return None


def _retriangulate_multiview(map_points, frames, K, min_obs=MULTIVIEW_MIN_OBS,
                             ransac_thresh=3.0, max_points=8000):
    if not map_points:
        return 0
    K64 = np.asarray(K, dtype=np.float64)
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    if len(map_points) > max_points:
        order = sorted(range(len(map_points)), key=lambda i: map_points[i].obs_count, reverse=True)[:max_points]
        candidates = [(i, map_points[i]) for i in order]
    else:
        candidates = list(enumerate(map_points))
    pts_obs_map = {}
    for idx, pt in candidates:
        if pt.obs_count < min_obs:
            continue
        obs_list = []
        for o in pt.obs:
            f = frames.get(o.frame_idx)
            if f is None or f.pose is None:
                continue
            obs_list.append((o.frame_idx, o.u, o.v))
        if len(obs_list) >= min_obs:
            pts_obs_map[idx] = obs_list
    if not pts_obs_map:
        return 0
    result = _triangulate_multiview_batch(pts_obs_map, frames, K64, min_tri_deg, ransac_thresh, _MEM_GUARD.current)
    updated = 0
    for idx, (X, valid) in result.items():
        if not valid:
            continue
        pt = map_points[idx]
        old = pt.xyz.astype(np.float64)
        # 批量矩阵乘与逐帧 `R @ x` 逐位一致；回退路径保持原 `c += ...`
        # 的累加顺序，结果不变。
        obs_R, obs_t, obs_u, obs_v = [], [], [], []
        for o in pt.obs:
            f = frames.get(o.frame_idx)
            if f is None or f.pose is None:
                continue
            obs_R.append(f.pose.R)
            obs_t.append(f.pose.t.flatten())
            obs_u.append(o.u)
            obs_v.append(o.v)
        if obs_R:
            Rs = np.stack(obs_R)
            ts = np.stack(obs_t)
            us = np.asarray(obs_u, dtype=np.float64)
            vs = np.asarray(obs_v, dtype=np.float64)
            _cost_accel = (sfm_accel is not None
                           and hasattr(sfm_accel, "point_cost_from_pc"))

            def _cost(x):
                pc = Rs @ x + ts
                if _cost_accel:
                    return sfm_accel.point_cost_from_pc(
                        pc, us, vs, K64[0, 0], K64[1, 1], K64[0, 2], K64[1, 2])
                # 回退：与批量路径同序，结果逐位一致
                c = 0.0
                m = 0
                for i in range(len(us)):
                    if pc[i, 2] <= 1e-9:
                        return np.inf
                    u = K64[0, 0] * pc[i, 0] / pc[i, 2] + K64[0, 2]
                    v = K64[1, 1] * pc[i, 1] / pc[i, 2] + K64[1, 2]
                    c += (u - us[i]) ** 2 + (v - vs[i]) ** 2
                    m += 1
                return c / m if m else np.inf
        else:
            def _cost(x):
                return np.inf

        oc = _cost(old)
        nc = _cost(X)
        if np.isfinite(oc) and np.isfinite(nc) and nc < oc * MULTIVIEW_ACCEPT_IMPROVE:
            pt.xyz = X.astype(np.float32)
            updated += 1
    return updated


# ---- EM 数据关联 ----
def _em_reassign_local(window, map_points, frames, focal, fy, cx, cy):
    if not window or not map_points:
        return 0
    K64 = _build_K(focal, fy, cx, cy)
    base_radius = float(_cfg().guided_radius_px)
    reassigned = 0

    visible_pts = set()
    for kf in window:
        fr = frames.get(kf)
        if fr is None or fr.feat_map is None:
            continue
        for v in fr.feat_map:
            v = int(v)
            if 0 <= v < len(map_points):
                visible_pts.add(v)
    if not visible_pts:
        return 0
    vis_list = list(visible_pts)
    vis_pos = {p: i for i, p in enumerate(vis_list)}
    n_vis = len(vis_list)
    xyz_all = np.array([map_points[p].xyz for p in vis_list], dtype=np.float64)

    for kf in window:
        fr = frames.get(kf)
        if fr is None or fr.pose is None or len(fr.kps) == 0:
            continue
        fm = fr.feat_map
        if fm is None:
            continue
        R = fr.pose.R.astype(np.float64)
        t = fr.pose.t.astype(np.float64).reshape(3)
        pc = xyz_all @ R.T + t
        depth = pc[:, 2]
        ok = depth > 1e-6
        if not np.any(ok):
            continue
        us = np.full(n_vis, np.nan, dtype=np.float64)
        vs = np.full(n_vis, np.nan, dtype=np.float64)
        us[ok] = K64[0, 0] * pc[ok, 0] / depth[ok] + K64[0, 2]
        vs[ok] = K64[1, 1] * pc[ok, 1] / depth[ok] + K64[1, 2]

        valid_idx = np.where(ok)[0]
        if len(valid_idx) == 0:
            continue
        tree = cKDTree(np.stack([us[valid_idx], vs[valid_idx]], axis=1))

        if fr.kdtree is not None:
            kps_uv = np.asarray(fr.kdtree.data, dtype=np.float64)
        else:
            kps_uv = np.array([kp.pt for kp in fr.kps], dtype=np.float64)
        # 按当前帧特征密度自适应搜索半径：稀疏放宽、稠密收紧
        span = kps_uv.max(axis=0) - kps_uv.min(axis=0) + 1e-6
        area_per_feat = float(span[0] * span[1]) / max(len(kps_uv), 1)
        radius = float(np.clip(np.sqrt(area_per_feat) * 2.0,
                               base_radius * 0.5, base_radius * 2.0))
        # 批量 query：半径内最近邻即 r^2 最小者，省去逐点 query_ball_point。
        nn_dist, nn_idx = tree.query(kps_uv, k=1, distance_upper_bound=radius,
                                     workers=-1)
        hit_kps = np.nonzero(np.isfinite(nn_dist))[0]
        if len(hit_kps) == 0:
            continue
        for kp_idx in hit_kps:
            kp_idx = int(kp_idx)
            kx, ky = float(kps_uv[kp_idx, 0]), float(kps_uv[kp_idx, 1])
            best_pid = vis_list[int(valid_idx[int(nn_idx[kp_idx])])]
            best_r2 = float(nn_dist[kp_idx]) ** 2

            old_pid = int(fm[kp_idx])
            if old_pid == best_pid:
                continue

            # 描述子一致性（避免错关联）
            new_desc = map_points[best_pid].desc
            if new_desc is not None and fr.desc is not None and kp_idx < len(fr.desc):
                q_desc = fr.desc[kp_idx]
                if q_desc.dtype == new_desc.dtype:
                    if q_desc.dtype == np.uint8:
                        dval = int(cv2.norm(q_desc, new_desc, cv2.NORM_HAMMING))
                    else:
                        dval = float(cv2.norm(q_desc, new_desc, cv2.NORM_L2))
                    if dval > _update_dist_for(new_desc):
                        continue

            # 目标点观测数上限
            if map_points[best_pid].obs_count >= MAX_OBS_PER_POINT:
                continue

            old_r2 = float("inf")
            if 0 <= old_pid < len(map_points):
                old_pc = R @ map_points[old_pid].xyz.astype(np.float64) + t
                if old_pc[2] > 1e-6:
                    old_u = K64[0, 0] * old_pc[0] / old_pc[2] + K64[0, 2]
                    old_v = K64[1, 1] * old_pc[1] / old_pc[2] + K64[1, 2]
                    old_r2 = (old_u - kx) ** 2 + (old_v - ky) ** 2

            if best_r2 >= BA_EM_REASSIGN_RATIO * old_r2:
                continue

            fm[kp_idx] = best_pid
            color = fr.colors[kp_idx] if (fr.colors is not None and kp_idx < len(fr.colors)) else None
            new_obs = Observation(kf, kp_idx, float(kx), float(ky),
                                  (color.copy() if color is not None else None))
            if 0 <= old_pid < len(map_points):
                old_pt = map_points[old_pid]
                old_pt.obs = [o for o in old_pt.obs
                              if not (o.frame_idx == kf and o.kp_idx == kp_idx)]
                old_pt.obs_set.discard((kf, kp_idx))
            map_points[best_pid].add_observation(new_obs)
            reassigned += 1

    return reassigned


# ---- 10. 回环检测与验证 ----
class LoopDetector:
    _POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
    _QUANTIZE_CHUNK = 512

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.words = None
        self.idf = None
        self.hists = {}
        self.inv_idx = {}
        self.trained = False
        self._norm_type = cv2.NORM_HAMMING
        # 最近一次 query 的统计：用来判断「回环为 0」是分数不够还是检索结果被筛掉
        self.last_stats = {"n_scored": 0, "top1": 0.0, "top1_kf": -1,
                           "n_pass": 0, "n_kept": 0}

    def train(self, keyframe_descs, norm_type):
        pool = [d for d in keyframe_descs if d is not None and len(d) > 0]
        if not pool:
            return
        all_desc = np.vstack(pool)
        if len(all_desc) < self.vocab_size:
            return
        rng = np.random.default_rng(0)
        cap = 20000
        if len(all_desc) > cap:
            train_desc = all_desc[rng.choice(len(all_desc), size=cap, replace=False)]
        else:
            train_desc = all_desc
        is_hamming = norm_type == cv2.NORM_HAMMING
        data = train_desc.astype(np.float32)
        k = min(self.vocab_size, len(data))
        try:
            mbk = MiniBatchKMeans(n_clusters=k, batch_size=4096, max_iter=100, n_init=3, random_state=0, reassignment_ratio=0.02)
            mbk.fit(data)
            centroids = mbk.cluster_centers_
        except Exception:
            return
        if is_hamming:
            self.words = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
        else:
            self.words = centroids.astype(np.float32)
        df = np.zeros(len(self.words), dtype=np.float64)
        for d in pool:
            words_present = np.unique(self._quantize_assign(d, norm_type))
            df[words_present] += 1.0
        n_docs = max(len(pool), 1)
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        self._norm_type = norm_type
        self.trained = True

    def _hamming_argmin(self, d_uint8, words_uint8):
        N = d_uint8.shape[0]
        assign = np.empty(N, dtype=np.int64)
        for start in range(0, N, self._QUANTIZE_CHUNK):
            end = min(start + self._QUANTIZE_CHUNK, N)
            xor = np.bitwise_xor(d_uint8[start:end, None, :], words_uint8[None, :, :])
            dist = self._POPCOUNT_LUT[xor].sum(axis=2)
            assign[start:end] = np.argmin(dist, axis=1)
        return assign

    def _quantize_assign(self, desc, norm_type):
        if self.words is None or len(desc) == 0:
            return np.zeros(0, dtype=np.int64)
        if norm_type == cv2.NORM_HAMMING:
            return self._hamming_argmin(desc.astype(np.uint8), self.words.astype(np.uint8))
        # 用 ‖d-w‖² = ‖d‖² + ‖w‖² - 2d·w 展开成一次 BLAS 矩阵乘，避免物化
        # (N_kp, vocab, D) 的 GB 级数组；float64 累加保证与 np.linalg.norm 同精度。
        d = np.asarray(desc, dtype=np.float64)
        w = np.asarray(self.words, dtype=np.float64)
        d2 = np.einsum("ij,ij->i", d, d)[:, None]
        w2 = np.einsum("ij,ij->i", w, w)[None, :]
        dist2 = d @ w.T
        dist2 *= -2.0
        dist2 += d2
        dist2 += w2
        return np.argmin(dist2, axis=1).astype(np.int64)

    def _quantize(self, desc):
        if self.words is None or len(desc) == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        assign = self._quantize_assign(desc, self._norm_type)
        tf = np.bincount(assign, minlength=len(self.words)).astype(np.float32)
        if self.idf is not None:
            tf = tf * self.idf.astype(np.float32)
        s = tf.sum()
        if s > 0:
            tf /= s
        return tf

    def add(self, idx, desc):
        if not self.trained:
            return
        tf = self._quantize(desc)
        self.hists[idx] = tf
        for w in np.where(tf > 0)[0]:
            self.inv_idx.setdefault(int(w), []).append((idx, float(tf[w])))

    def query(self, idx, desc, keyframes, skip_recent=LOOP_SKIP_RECENT,
              min_score=LOOP_MIN_SCORE, max_candidates=None):
        if not self.trained:
            return []
        if max_candidates is None:
            max_candidates = _cfg().loop_max_candidates
        q = self._quantize(desc)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return []
        scores = {}
        for w in np.where(q > 0)[0]:
            for kf, wt in self.inv_idx.get(int(w), []):
                scores[kf] = scores.get(kf, 0.0) + wt * float(q[w])
        scored = []
        for kf, s in scores.items():
            if abs(kf - idx) < skip_recent:
                continue
            h = self.hists.get(kf)
            if h is None:
                continue
            hn = np.linalg.norm(h)
            if hn < 1e-12:
                continue
            scored.append((kf, s / (qn * hn)))
        top1_kf, top1 = (-1, 0.0)
        for kf, sc in scored:
            if sc > top1:
                top1_kf, top1 = kf, sc
        ranked = [(kf, sc) for kf, sc in scored if sc >= min_score]
        ranked.sort(key=lambda x: -x[1])
        kept = []
        for kf, s in ranked:
            if all(abs(kf - k) > LOOP_CANDIDATE_NMS_GAP for k, _ in kept):
                kept.append((kf, s))
            if len(kept) >= max_candidates:
                break
        self.last_stats = {
            "n_scored": len(scored), "top1": float(top1),
            "top1_kf": int(top1_kf), "n_pass": len(ranked), "n_kept": len(kept),
        }
        return kept

    def retrieve_topk(self, idx, desc, k=20, min_score=0.15):
        if not self.trained:
            return []
        q = self._quantize(desc)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return []
        scores = {}
        for w in np.where(q > 0)[0]:
            for kf, wt in self.inv_idx.get(int(w), []):
                if kf == idx:
                    continue
                scores[kf] = scores.get(kf, 0.0) + wt * float(q[w])
        ranked = []
        for kf, s in scores.items():
            h = self.hists.get(kf)
            if h is None:
                continue
            hn = np.linalg.norm(h)
            if hn < 1e-12:
                continue
            sc = s / (qn * hn)
            if sc >= min_score:
                ranked.append((kf, sc))
        ranked.sort(key=lambda x: -x[1])
        kept = []
        for kf, s in ranked:
            if all(abs(kf - kk) > 3 for kk, _ in kept):
                kept.append((kf, s))
            if len(kept) >= k:
                break
        return kept


def _check_seq_consistency(curr_idx, cand_kf, keyframes, loop_detector,
                           k=LOOP_SEQ_CHECK_K, thresh=LOOP_SEQ_CHECK_THRESH):
    # 用词袋已登记的帧（按索引排序）而非 keyframes 列表：被剔除的关键帧仍会作为
    # 检索候选返回，按列表位置查会得到 0 并否掉有效回环。
    ids = sorted(loop_detector.hists.keys())
    pos = {f: i for i, f in enumerate(ids)}
    if cand_kf not in pos or curr_idx not in pos:
        return 0
    i, j = pos[cand_kf], pos[curr_idx]
    n_ok = 0
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            ia, jb = i + di, j + dj
            if not (0 <= ia < len(ids)):
                continue
            if not (0 <= jb < len(ids)):
                continue
            a, b = ids[ia], ids[jb]
            if a == b:
                continue
            ha = loop_detector.hists.get(a)
            hb = loop_detector.hists.get(b)
            if ha is None or hb is None:
                continue
            denom = np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-12
            s = float(np.dot(ha, hb) / denom)
            if s > thresh:
                n_ok += 1
    return n_ok


def _detect_and_verify_loops(curr_idx, keyframes, frames, loop_detector, map_points,
                             focal, fy, cx, cy, reproj_thresh):
    cfg = _cfg()
    frame = frames[curr_idx]
    if frame.desc is None or len(frame.desc) == 0:
        return []
    cands = loop_detector.query(curr_idx, frame.desc, keyframes, skip_recent=LOOP_SKIP_RECENT,
                                min_score=LOOP_MIN_SCORE, max_candidates=cfg.loop_max_candidates * 3)
    min_gap = max(40, cfg.loop_min_gap)
    results = []
    # 逐级计数：回环为 0 时能直接看出卡在哪一步
    stage = {"cand": len(cands), "gap": 0, "match": 0, "fh": 0, "p3d": 0,
             "seq": 0, "pnp": 0}
    K64 = _build_K(focal, fy, cx, cy)
    K32 = K64.astype(np.float32)
    for kf, score in cands:
        if abs(kf - curr_idx) < min_gap:
            continue
        stage["gap"] += 1
        if kf not in frames or frames[kf].desc is None:
            continue
        kf_frame = frames[kf]
        nt, md, _ = get_metric(kf_frame.desc)
        matches = _match_ratio_absolute(kf_frame.desc, frame.desc, nt, md)
        if len(matches) < 15:
            continue
        stage["match"] += 1
        pts_a = np.array([kf_frame.kps[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_b = np.array([frame.kps[m.trainIdx].pt for m in matches], dtype=np.float32)
        vmask, _, _ = _verify_matches_fh(pts_a, pts_b, reproj_thresh * 1.5, 10)
        if vmask is None or int(vmask.sum()) < 10:
            continue
        stage["fh"] += 1
        pts3d_list = []
        pts2d_list = []
        kf_fm = kf_frame.feat_map
        for mi, m in enumerate(matches):
            if not vmask[mi]:
                continue
            q = int(m.queryIdx)
            t = int(m.trainIdx)
            if q < 0 or q >= len(kf_fm):
                continue
            pt_idx = int(kf_fm[q])
            if 0 <= pt_idx < len(map_points):
                pts3d_list.append(map_points[pt_idx].xyz)
                pts2d_list.append(frame.kps[t].pt)
        if len(pts3d_list) < cfg.loop_min_inliers:
            continue
        stage["p3d"] += 1
        n_seq = _check_seq_consistency(curr_idx, kf, keyframes, loop_detector)
        if n_seq < LOOP_MIN_SEQ_CONSISTENCY:
            continue
        stage["seq"] += 1
        pts3d = np.asarray(pts3d_list, dtype=np.float32)
        pts2d = np.asarray(pts2d_list, dtype=np.float32)
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                pts3d, pts2d, K32, np.zeros(4, dtype=np.float32),
                iterationsCount=cfg.pnp_ransac_iters,
                reprojectionError=reproj_thresh, confidence=0.99,
            )
        except cv2.error:
            continue
        if not ok or inl is None:
            continue
        if len(inl) < cfg.loop_min_inliers:
            continue
        if len(inl) / max(len(pts3d_list), 1) < LOOP_MIN_INLIER_RATIO:
            continue
        stage["pnp"] += 1
        R, _ = cv2.Rodrigues(rvec)
        pose_loop = CameraPose(R, tvec.reshape(3, 1))
        kf_pose = kf_frame.pose
        s = 1.0
        if kf_pose is not None and len(pts3d_list) >= 10:
            pts_np = np.array(pts3d_list, dtype=np.float64)
            X_in_kf = (kf_pose.R.astype(np.float64) @ pts_np.T).T + kf_pose.t.flatten()
            X_in_curr = (pose_loop.R.astype(np.float64) @ pts_np.T).T + pose_loop.t.flatten()
            sim3 = estimate_sim3(X_in_kf, X_in_curr)
            if sim3 is not None:
                s = float(np.clip(sim3.s, 0.5, 2.0))
                if abs(s - 1.0) < 0.03:
                    s = 1.0
        if kf_pose is not None:
            T_ij = _relative_pose(kf_pose, pose_loop)
        else:
            T_ij = CameraPose(pose_loop.R.copy(), pose_loop.t.copy())
        results.append((kf, Sim3(s, T_ij.R, T_ij.t)))
    _st = loop_detector.last_stats
    if results:
        logger.info("[loop %d] BoW top1=%.3f(kf=%d) 过阈=%d/%d → 命中 %d 处",
                    curr_idx, _st["top1"], _st["top1_kf"], _st["n_pass"],
                    _st["n_scored"], len(results))
    elif _st["n_pass"] == 0:
        # 检索有分数但全被阈值挡掉：阈值定高了，或者词袋根本没区分度
        logger.info("[loop %d] BoW top1=%.3f(kf=%d) < %.2f，%d 个关键帧全部未过阈值",
                    curr_idx, _st["top1"], _st["top1_kf"], LOOP_MIN_SCORE,
                    _st["n_scored"])
    else:
        logger.info("[loop %d] BoW top1=%.3f 过阈=%d → 间隔=%d 匹配=%d F=%d 3D=%d 序列=%d PnP=%d → 0 处",
                    curr_idx, _st["top1"], _st["n_pass"], stage["gap"],
                    stage["match"], stage["fh"], stage["p3d"], stage["seq"],
                    stage["pnp"])
    return results


# ---- 11. PnP 与重定位 ----
def _guided_match_3d2d(frame, kf_indices, map_points, frames, K,
                       proj_radius_px, max_desc_dist=None):
    if frame.pose is None or frame.desc is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []
    if len(frame.kps) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []
    norm_type, md, _ = get_metric(frame.desc)
    if max_desc_dist is None:
        max_desc_dist = md
    R = frame.pose.R
    t = frame.pose.t.flatten()
    K64 = np.asarray(K, dtype=np.float64)
    if frame.kdtree is None:
        frame._build_kdtree()
    tree = frame.kdtree
    kps_uv = np.array([kp.pt for kp in frame.kps], dtype=np.float32)
    out3d, out2d, out_idx = [], [], []
    used_pts, used_kps = set(), set()
    for kf in kf_indices:
        if kf not in frames:
            continue
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        for kp_i, pt_idx in enumerate(kf_fm):
            pt_idx = int(pt_idx)
            if pt_idx < 0 or pt_idx >= len(map_points):
                continue
            if pt_idx in used_pts:
                continue
            Xw = map_points[pt_idx].xyz
            if Xw.size != 3 or not np.isfinite(Xw).all():
                continue
            Xc = R @ Xw.reshape(3, 1) + t.reshape(3, 1)
            d = float(Xc[2, 0])
            if d <= 1e-6:
                continue
            u = K64[0, 0] * float(Xc[0, 0]) / d + K64[0, 2]
            v = K64[1, 1] * float(Xc[1, 0]) / d + K64[1, 2]
            if not (0 <= u < 1e6 and 0 <= v < 1e6):
                continue
            if tree is not None:
                cand = tree.query_ball_point([u, v], proj_radius_px)
            else:
                dx = kps_uv[:, 0] - u
                dy = kps_uv[:, 1] - v
                cand = np.where(dx * dx + dy * dy <= proj_radius_px ** 2)[0].tolist()
            if not cand:
                continue
            desc_pt = map_points[pt_idx].desc
            best_kp, best_d = -1, None
            for ci in cand:
                ci = int(ci)
                if ci in used_kps:
                    continue
                if frame.desc is None or ci >= len(frame.desc):
                    continue
                if desc_pt is None:
                    best_kp, best_d = ci, 0.0
                    break
                if frame.desc.dtype != desc_pt.dtype:
                    continue
                if norm_type == cv2.NORM_HAMMING:
                    dval = int(cv2.norm(frame.desc[ci], desc_pt, cv2.NORM_HAMMING))
                else:
                    dval = float(cv2.norm(frame.desc[ci], desc_pt, cv2.NORM_L2))
                if dval > max_desc_dist:
                    continue
                if best_d is None or dval < best_d:
                    best_d, best_kp = dval, ci
            if best_kp < 0:
                continue
            used_pts.add(pt_idx)
            used_kps.add(best_kp)
            out3d.append(Xw)
            out2d.append(frame.kps[best_kp].pt)
            out_idx.append((pt_idx, best_kp))
    if not out3d:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []
    return np.asarray(out3d, dtype=np.float32), np.asarray(out2d, dtype=np.float32), out_idx


def _add_pnp_observations(frame, idx_pairs, inlier_ids, map_points, frames=None):
    if frame.feat_map is None or len(idx_pairs) == 0:
        return 0
    cfg = _cfg()
    min_angle = cfg.filter_min_tri_angle_deg
    added = 0
    frame_fm = frame.feat_map
    frame_colors = frame.colors
    frame_kps = frame.kps
    for k in inlier_ids:
        k = int(k)
        if k < 0 or k >= len(idx_pairs):
            continue
        pt_idx, kp_idx = idx_pairs[k]
        if not (0 <= pt_idx < len(map_points)):
            continue
        if kp_idx < 0 or kp_idx >= len(frame_fm):
            continue
        existing = int(frame_fm[kp_idx])
        if existing >= 0 and existing != pt_idx:
            continue
        pt = map_points[pt_idx]
        if pt.obs_count >= MAX_OBS_PER_POINT:
            continue
        if not _check_obs_angle(pt, frame.idx, frames, min_angle):
            continue
        if frame_colors is not None and kp_idx < len(frame_colors):
            c = frame_colors[kp_idx]
        else:
            c = None
        if pt.add_observation(Observation(frame.idx, kp_idx,
                                          float(frame_kps[kp_idx].pt[0]),
                                          float(frame_kps[kp_idx].pt[1]),
                                          (c.copy() if c is not None else None))):
            frame_fm[kp_idx] = pt_idx
            _update_desc(pt, frame.desc[kp_idx])
            added += 1
    return added


def _try_pnp_guided(frame, keyframes, map_points, frames, covis_graph,
                    focal, fy, cx, cy, reproj_thresh, loop_detector=None,
                    diag=None):
    """局部地图 PnP。失败时把原因写进 diag，供调用处打印诊断。"""
    cfg = _cfg()
    if diag is not None:
        diag.clear()
        diag.update({"stage": "start", "npts": 0, "inliers": 0, "reason": ""})
    if len(keyframes) <= 1:
        if diag is not None:
            diag["reason"] = "关键帧不足"
        return None, None, 0.0
    K = _build_K(focal, fy, cx, cy)
    K32 = K.astype(np.float32)
    ranked = covis_graph.get_ranked_by_covis(frame.idx)
    neighbors = [k for k, _ in ranked[:cfg.pnp_window]]
    if not neighbors:
        # 共视图尚未覆盖到该帧（例如初始化后的头几帧）时回退到最近的已注册关键帧
        neighbors = [k for k in keyframes if k != frame.idx][-cfg.pnp_window:]
    if loop_detector is not None and loop_detector.trained:
        bow_cands = loop_detector.retrieve_topk(frame.idx, frame.desc, k=cfg.pnp_window, min_score=0.1)
        for kf, _ in bow_cands:
            if kf not in neighbors:
                neighbors.append(kf)
    def _collect(neigh):
        p3 = np.empty((0, 3), dtype=np.float32)
        p2 = np.empty((0, 2), dtype=np.float32)
        ids = []
        if frame.pose is not None:
            p3, p2, ids = _guided_match_3d2d(
                frame, neigh, map_points, frames, K,
                proj_radius_px=cfg.guided_radius_px,
                max_desc_dist=_update_dist_for(frame.desc),
            )
        elif _GUIDED_RELOC_ON and neigh:
            # 重定位帧无位姿：借用共视最强的邻居位姿作预测再做一次 3D-2D 引导匹配，
            # 投影偏一段仍能命中大部分点，RANSAC 兜底。仅在本帧无位姿时走这一支。
            _pred = frames[neigh[0]].pose if neigh[0] in frames else None
            if _pred is not None:
                _saved = frame.pose
                frame.pose = _pred
                try:
                    p3, p2, ids = _guided_match_3d2d(
                        frame, neigh, map_points, frames, K,
                        proj_radius_px=cfg.guided_radius_px,
                        max_desc_dist=_update_dist_for(frame.desc),
                    )
                finally:
                    frame.pose = _saved
        if len(p3) < max(8, cfg.pnp_min_inliers):
            c3, c2, cid = [], [], []
            for kf in neigh:
                if kf not in frames:
                    continue
                kf_fm = frames[kf].feat_map
                if kf_fm is None:
                    continue
                bound = np.where(kf_fm >= 0)[0]
                if len(bound) < 4:
                    continue
                bd = frames[kf].desc[bound]
                nt, md, _ = get_metric(bd)
                matches = _match_ratio_absolute(bd, frame.desc, nt, md)
                for m in matches:
                    pt_idx = int(kf_fm[int(bound[m.queryIdx])])
                    if 0 <= pt_idx < len(map_points):
                        c3.append(map_points[pt_idx].xyz)
                        c2.append(frame.kps[m.trainIdx].pt)
                        cid.append((pt_idx, int(m.trainIdx)))
            if len(c3) >= 8 and len(c3) > len(p3):
                p3 = np.asarray(c3, dtype=np.float32)
                p2 = np.asarray(c2, dtype=np.float32)
                ids = cid
        return p3, p2, ids

    pts3d_g, pts2d_g, idx_g = _collect(neighbors)
    # 无序图片集 / 环形采集下共视邻居覆盖不足，回退到所有已注册帧再试一次，
    # 可利用“非关键帧但强重叠”的图片。
    if len(pts3d_g) < max(12, cfg.pnp_min_inliers):
        posed = [k for k in frames
                 if k != frame.idx and frames[k].pose is not None]
        posed.sort()
        if len(posed) > 60:
            posed = posed[-60:]
        if set(posed) - set(neighbors):
            p3, p2, ids = _collect(posed)
            if len(p3) > len(pts3d_g):
                pts3d_g, pts2d_g, idx_g = p3, p2, ids
                neighbors = posed
    logger.debug("[pnp %d] neighbors=%s guided=%d cand=%d",
                frame.idx, neighbors, len(idx_g), len(pts3d_g))
    if diag is not None:
        diag.update({"stage": "match", "npts": int(len(pts3d_g)),
                     "neighbors": len(neighbors)})
    if len(pts3d_g) < 8:
        if diag is not None:
            diag["reason"] = "匹配点不足"
        return None, None, 0.0
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            np.ascontiguousarray(pts3d_g, dtype=np.float32),
            np.ascontiguousarray(pts2d_g, dtype=np.float32),
            K32, np.zeros(4, dtype=np.float32),
            iterationsCount=cfg.pnp_ransac_iters,
            reprojectionError=reproj_thresh, confidence=0.99,
        )
    except cv2.error:
        logger.debug("[pnp %d] solvePnPRansac error", frame.idx)
        if diag is not None:
            diag["reason"] = "求解异常"
        return None, None, 0.0
    if not ok or inl is None:
        logger.debug("[pnp %d] ransac ok=%s inl=None npts=%d", frame.idx, ok, len(pts3d_g))
        if diag is not None:
            diag["reason"] = "RANSAC 无解"
        return None, None, 0.0
    if len(inl) < cfg.pnp_min_inliers:
        logger.debug("[pnp %d] inliers=%d < %d (npts=%d)",
                    frame.idx, len(inl), cfg.pnp_min_inliers, len(pts3d_g))
        if diag is not None:
            diag.update({"stage": "ransac", "inliers": int(len(inl)),
                         "reason": "内点不足"})
        return None, None, 0.0
    n_inliers = int(len(inl))
    if diag is not None:
        diag.update({"stage": "ok", "inliers": n_inliers, "reason": ""})
    quality = float(n_inliers) / max(1, len(pts3d_g))
    inlier_ids = inl.ravel().astype(np.int64) if inl.ndim > 1 else inl.astype(np.int64)
    if n_inliers >= 8:
        try:
            ok2, rvec2, tvec2 = cv2.solvePnP(
                np.ascontiguousarray(pts3d_g[inlier_ids], dtype=np.float32),
                np.ascontiguousarray(pts2d_g[inlier_ids], dtype=np.float32),
                K32, np.zeros(4, dtype=np.float32),
                rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok2:
                rvec, tvec = rvec2, tvec2
        except cv2.error:
            pass
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    pose = CameraPose(R, t)
    centroid = pts3d_g.mean(axis=0)
    scene_scale = float(np.median(np.linalg.norm(pts3d_g - centroid, axis=1))) + 1e-6
    cam_c = (-R.T @ t).flatten()
    if np.linalg.norm(cam_c - centroid) > 10.0 * scene_scale:
        if diag is not None:
            diag["reason"] = "解偏离地图"
        return None, None, 0.0
    # _check_obs_angle 要求本帧已有位姿，所以调用前先临时挂上 pose，
    # 否则所有观测都会被拒。
    prev_pose = frame.pose
    frame.pose = pose
    _add_pnp_observations(frame, idx_g, inlier_ids, map_points, frames)
    frame.pose = prev_pose
    if diag is not None:
        diag.update({"stage": "ok", "inliers": n_inliers, "reason": ""})
    return pose, None, quality


def _try_global_relocalization(frame, keyframes, map_points, frames, loop_detector,
                               focal, fy, cx, cy, reproj_thresh, diag=None):
    cfg = _cfg()
    if diag is not None:
        diag.clear()
        diag.update({"cands": 0, "best_matches": 0, "best_inliers": 0,
                     "reason": ""})
    if len(keyframes) < 2:
        if diag is not None:
            diag["reason"] = "关键帧不足"
        return None
    if frame.desc is None or len(frame.desc) == 0:
        if diag is not None:
            diag["reason"] = "无描述子"
        return None
    cands = []
    if loop_detector.trained:
        cands = loop_detector.query(frame.idx, frame.desc, keyframes, skip_recent=0,
                                    min_score=LOOP_RELOC_SCORE,
                                    max_candidates=cfg.reloc_max_candidates * 3)
    if not cands:
        # BoW 未训练时，用所有已注册帧（含非关键帧）作为候选
        posed = [k for k in frames if k != frame.idx and frames[k].pose is not None]
        posed.sort()
        if posed:
            cands = [(k, 1.0) for k in posed[-40:]]
    if not cands:
        if diag is not None:
            diag["reason"] = "无候选"
        return None
    if diag is not None:
        diag["cands"] = len(cands)
    K = _build_K(focal, fy, cx, cy).astype(np.float32)
    for kf, _ in cands:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        bound = np.where(kf_fm >= 0)[0]
        if len(bound) < 4:
            continue
        bd = frames[kf].desc[bound]
        nt, md, _ = get_metric(bd)
        matches = _match_ratio_absolute(bd, frame.desc, nt, md)
        if diag is not None and len(matches) > diag["best_matches"]:
            diag["best_matches"] = len(matches)
        if len(matches) < cfg.reloc_min_matches:
            continue
        pts3d, pts2d = [], []
        for m in matches:
            pt_idx = int(kf_fm[int(bound[m.queryIdx])])
            if 0 <= pt_idx < len(map_points):
                pts3d.append(map_points[pt_idx].xyz)
                pts2d.append(frame.kps[m.trainIdx].pt)
        if len(pts3d) < cfg.reloc_min_matches:
            continue
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                np.asarray(pts3d, dtype=np.float32),
                np.asarray(pts2d, dtype=np.float32),
                K, np.zeros(4, dtype=np.float32),
                iterationsCount=150, reprojectionError=reproj_thresh, confidence=0.99,
            )
        except cv2.error:
            continue
        if not ok or inl is None:
            continue
        if diag is not None and len(inl) > diag["best_inliers"]:
            diag["best_inliers"] = int(len(inl))
        if len(inl) < cfg.reloc_min_inliers:
            continue
        R, _ = cv2.Rodrigues(rvec)
        if diag is not None:
            diag.update({"best_inliers": int(len(inl)), "reason": ""})
        return CameraPose(R, tvec.reshape(3, 1)), kf
    if diag is not None and not diag["reason"]:
        diag["reason"] = "候选均未通过"
    return None


# ---- 12. 关键帧与轨迹 ----
def _compute_covis_ratio(curr_idx, keyframes, matches, frames):
    if not matches or len(keyframes) < 2:
        return 0.0
    last_kf = keyframes[-1]
    if last_kf == curr_idx:
        return 1.0
    kf_fm = frames[last_kf].feat_map
    if kf_fm is None:
        return 0.0
    kf_pts = set(int(x) for x in kf_fm if x >= 0)
    if not kf_pts:
        return 0.0
    curr_fm = frames[curr_idx].feat_map
    if curr_fm is None:
        return 0.0
    total = count = 0
    for m in matches:
        t = m.trainIdx
        if t < 0 or t >= len(curr_fm):
            continue
        b = curr_fm[t]
        if b < 0:
            continue
        total += 1
        if int(b) in kf_pts:
            count += 1
    return count / total if total >= 30 else 0.0


def _cull_keyframes(keyframes, frames, curr_idx, covis_graph):
    if len(keyframes) <= KF_CULL_WINDOW:
        return
    recent = keyframes[-KF_CULL_WINDOW:-1]
    curr_fm = frames[curr_idx].feat_map
    if curr_fm is None:
        return
    curr_pts = set(int(x) for x in curr_fm if x >= 0)
    if not curr_pts:
        return
    culled = []
    for kf in recent:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        kf_pts = set(int(x) for x in kf_fm if x >= 0)
        if not kf_pts:
            continue
        denom = max(1, min(len(curr_pts), len(kf_pts)))
        covis = len(curr_pts & kf_pts) / denom
        if covis > KF_CULL_COVIS:
            culled.append(kf)
    # 只从关键帧列表摘掉，保留 feat_map：帧仍是已注册帧，观测要继续参与 BA。
    # 清空 feat_map 会与已挂的观测矛盾，后续建点会把同一 (frame, kp) 重复关联。
    for kf in culled:
        if kf in keyframes:
            keyframes.remove(kf)
        frames[kf].is_keyframe = False
        covis_graph.remove_keyframe(kf)


def _complete_tracks_py(keyframes, frames, map_points, K, reproj_thresh, max_transitivity=5):
    cfg = _cfg()
    if not keyframes or not map_points:
        return 0
    K64 = np.asarray(K, dtype=np.float64)
    err_thresh = MAX_REPROJ_ERROR * reproj_thresh
    min_angle = cfg.filter_min_tri_angle_deg
    added_total = 0
    for kf in keyframes:
        if kf not in frames or frames[kf].pose is None:
            continue
        f = frames[kf]
        fm = f.feat_map
        if fm is None:
            continue
        R = f.pose.R
        t = f.pose.t
        if f.kdtree is None:
            f._build_kdtree()
        tree = f.kdtree
        kps_uv = np.array([kp.pt for kp in f.kps], dtype=np.float32)
        if len(kps_uv) == 0:
            continue
        radius = min(cfg.guided_radius_px * 1.5, TRACK_GATE_PX)
        for pt in map_points:
            if pt.obs_count >= MAX_OBS_PER_POINT:
                continue
            already = False
            for o in pt.obs:
                if o.frame_idx == kf:
                    already = True
                    break
            if already:
                continue
            Xw = pt.xyz
            if Xw.size != 3 or not np.isfinite(Xw).all():
                continue
            Xc = R @ Xw.reshape(3, 1) + t
            d = float(Xc[2, 0])
            if d <= 1e-6:
                continue
            u = K64[0, 0] * float(Xc[0, 0]) / d + K64[0, 2]
            v = K64[1, 1] * float(Xc[1, 0]) / d + K64[1, 2]
            if tree is not None:
                cand = tree.query_ball_point([u, v], radius)
            else:
                dx = kps_uv[:, 0] - u
                dy = kps_uv[:, 1] - v
                cand = np.where(dx * dx + dy * dy <= radius ** 2)[0].tolist()
            if not cand:
                continue
            if not _check_obs_angle(pt, kf, frames, min_angle):
                continue
            desc_pt = pt.desc
            max_dist = _update_dist_for(f.desc)
            best_kp = -1
            best_d = None
            for ci in cand:
                ci = int(ci)
                if fm[ci] >= 0:
                    continue
                if desc_pt is None or f.desc is None:
                    best_kp, best_d = ci, 0.0
                    break
                if f.desc.dtype != desc_pt.dtype:
                    continue
                if f.desc.dtype == np.uint8:
                    dval = int(cv2.norm(f.desc[ci], desc_pt, cv2.NORM_HAMMING))
                else:
                    dval = float(cv2.norm(f.desc[ci], desc_pt, cv2.NORM_L2))
                if dval > max_dist:
                    continue
                if best_d is None or dval < best_d:
                    best_d, best_kp = dval, ci
            if best_kp < 0:
                continue
            if f.colors is not None and best_kp < len(f.colors):
                c = f.colors[best_kp]
            else:
                c = None
            if pt.add_observation(Observation(kf, best_kp,
                                              float(f.kps[best_kp].pt[0]),
                                              float(f.kps[best_kp].pt[1]),
                                              (c.copy() if c is not None else None))):
                fm[best_kp] = pt.idx
                _update_desc(pt, f.desc[best_kp])
                added_total += 1
    return added_total


def _complete_tracks_accel(keyframes, frames, map_points, K, reproj_thresh,
                           max_transitivity=5):
    """_complete_tracks 的加速实现，语义与 Python 版逐条对应。

    处理顺序不变（投影 → 半径候选 → 三角化角 → 描述子匹配），改为批量调用加 Cython 内核。
    """
    from itertools import chain

    cfg = _cfg()
    if not keyframes or not map_points:
        return 0
    n_pts = len(map_points)
    K64 = np.asarray(K, dtype=np.float64)
    kfx, kfy = float(K64[0, 0]), float(K64[1, 1])
    kcx, kcy = float(K64[0, 2]), float(K64[1, 2])
    min_angle = cfg.filter_min_tri_angle_deg
    radius = min(cfg.guided_radius_px * 1.5, TRACK_GATE_PX)

    # ---- 地图点摊平成数组 ----
    xyz = np.zeros((n_pts, 3), dtype=np.float64)
    pt_ok = np.zeros(n_pts, dtype=np.uint8)
    for i, pt in enumerate(map_points):
        a = pt.xyz
        if a.size == 3:
            xyz[i] = a
            if np.isfinite(xyz[i]).all():
                pt_ok[i] = 1

    # ---- 描述子矩阵。要求全地图同 dtype / 同维度，否则退回慢路径 ----
    # 地图点描述子是 1-D (D,)，帧的是 2-D (n_kp, D)。
    desc_dtype = None
    desc_dim = 0
    mixed = False
    for pt in map_points:
        d = pt.desc
        if d is None or d.ndim < 1:
            mixed = True
            break
        if desc_dtype is None:
            desc_dtype = d.dtype
            desc_dim = d.shape[-1]
        elif d.dtype != desc_dtype or d.shape[-1] != desc_dim:
            mixed = True
            break
    if mixed or desc_dtype is None:
        return _complete_tracks_py(keyframes, frames, map_points, K,
                                   reproj_thresh, max_transitivity)
    desc_p = np.empty((n_pts, desc_dim), dtype=desc_dtype)
    for i, pt in enumerate(map_points):
        desc_p[i] = pt.desc
    metric = 1 if desc_dtype == np.uint8 else 0     # 1 = Hamming, 0 = L2

    # ---- 相机中心 / 有效标记（一次建好，整轮不变）----
    n_frames_all = (max(frames) + 1) if frames else 0
    centers = np.zeros((n_frames_all, 3), dtype=np.float64)
    cam_ok = np.zeros(n_frames_all, dtype=np.uint8)
    for fi, fr in frames.items():
        if 0 <= fi < n_frames_all and fr.pose is not None:
            centers[fi] = fr.pose.center
            cam_ok[fi] = 1

    # ---- 观测摊平（三角化角检查用；补进去的观测也追加在这里）----
    obs_pt = np.fromiter(
        (pi for pi, pt in enumerate(map_points) for _ in pt.obs),
        dtype=np.int64)
    n_obs0 = obs_pt.size
    obs_f = np.fromiter(
        (o.frame_idx for pt in map_points for o in pt.obs),
        dtype=np.int64, count=n_obs0)
    obs_cnt = np.bincount(obs_pt, minlength=n_pts).astype(np.int64)
    cap = max(n_obs0 + 8 * n_pts, 1024)
    obs_pt_buf = np.empty(cap, dtype=np.int64)
    obs_f_buf = np.empty(cap, dtype=np.int64)
    obs_pt_buf[:n_obs0] = obs_pt
    obs_f_buf[:n_obs0] = obs_f
    n_obs_cur = n_obs0

    added_total = 0
    for kf in keyframes:
        f = frames.get(kf)
        if f is None or f.pose is None:
            continue
        fm = f.feat_map
        if fm is None or len(fm) == 0:
            continue
        # 描述子 dtype/维度对不上时整段交给慢路径：只回退单帧会让后面几帧
        # 依赖的 desc_p / obs 缓冲与实际状态不一致。
        if (f.desc is None or f.desc.ndim != 2 or f.desc.dtype != desc_dtype
                or f.desc.shape[1] != desc_dim
                or fm.dtype != np.int64 or not fm.flags.c_contiguous):
            return _complete_tracks_py(keyframes, frames, map_points, K,
                                       reproj_thresh, max_transitivity)
        if f.kdtree is None:
            f._build_kdtree()
        tree = f.kdtree
        if tree is None:
            return _complete_tracks_py(keyframes, frames, map_points, K,
                                       reproj_thresh, max_transitivity)

        # 本帧已观测到的点：从 feat_map 反查（与逐点遍历 obs 等价）
        in_frame = np.zeros(n_pts, dtype=bool)
        vv = fm[fm >= 0]
        if vv.size:
            vv = vv[vv < n_pts]
            in_frame[vv] = True
        mask = (pt_ok == 1) & (~in_frame) & (obs_cnt < MAX_OBS_PER_POINT)
        if not mask.any():
            continue

        R = np.ascontiguousarray(f.pose.R, dtype=np.float64)
        t = np.ascontiguousarray(f.pose.t, dtype=np.float64).reshape(-1)[:3]
        uv = np.empty((n_pts, 2), dtype=np.float64)
        depth = np.empty(n_pts, dtype=np.float64)
        sfm_accel.project_points(R, t, xyz, kfx, kfy, kcx, kcy, uv, depth)
        mask &= depth > 1e-6
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            continue

        cand_lists = tree.query_ball_point(uv[idx], radius)
        lens = np.fromiter(map(len, cand_lists), dtype=np.int64, count=idx.size)
        tot = int(lens.sum())
        if tot == 0:
            continue
        cand_kp = np.fromiter(chain.from_iterable(cand_lists), dtype=np.int64,
                              count=tot)
        cand_pt = np.repeat(idx, lens)
        good = (cand_kp >= 0) & (cand_kp < len(fm))
        if not good.all():
            cand_kp = cand_kp[good]
            cand_pt = cand_pt[good]
            if cand_pt.size == 0:
                continue

        ok_ang = np.zeros(n_pts, dtype=np.uint8)
        sfm_accel.check_obs_angle(xyz, obs_pt_buf[:n_obs_cur],
                                  obs_f_buf[:n_obs_cur], centers, cam_ok,
                                  int(kf), min_angle, ok_ang)
        keep = ok_ang[cand_pt].astype(bool)
        if not keep.all():
            cand_pt = cand_pt[keep]
            cand_kp = cand_kp[keep]
            if cand_pt.size == 0:
                continue

        max_dist = float(_update_dist_for(f.desc))
        used = np.zeros(len(fm), dtype=np.uint8)
        best_kp = np.full(n_pts, -1, dtype=np.int64)
        best_d = np.zeros(n_pts, dtype=np.float64)
        sfm_accel.desc_assign_sequential(
            np.ascontiguousarray(f.desc, dtype=desc_dtype), desc_p,
            cand_pt, cand_kp, fm, used, max_dist, metric, best_kp, best_d)

        winners = np.nonzero(best_kp >= 0)[0]
        if winners.size == 0:
            continue
        kps = f.kps
        colors = f.colors
        n_colors = len(colors) if colors is not None else 0
        n_kps = len(kps)
        for p in winners:
            kp = int(best_kp[int(p)])
            if kp >= n_kps:
                continue
            pt = map_points[int(p)]
            c = colors[kp] if kp < n_colors else None
            if not pt.add_observation(Observation(
                    int(kf), kp, float(kps[kp].pt[0]), float(kps[kp].pt[1]),
                    (c.copy() if c is not None else None))):
                continue
            fm[kp] = pt.idx
            _update_desc(pt, f.desc[kp])
            desc_p[int(p)] = pt.desc
            obs_cnt[int(p)] += 1
            if n_obs_cur >= obs_pt_buf.size:      # 扩容
                obs_pt_buf = np.resize(obs_pt_buf, obs_pt_buf.size * 2)
                obs_f_buf = np.resize(obs_f_buf, obs_f_buf.size * 2)
            obs_pt_buf[n_obs_cur] = int(p)
            obs_f_buf[n_obs_cur] = int(kf)
            n_obs_cur += 1
            added_total += 1
    return added_total


def _complete_tracks(keyframes, frames, map_points, K, reproj_thresh,
                     max_transitivity=5):
    if sfm_accel is None:
        return _complete_tracks_py(keyframes, frames, map_points, K,
                                   reproj_thresh, max_transitivity)
    try:
        return _complete_tracks_accel(keyframes, frames, map_points, K,
                                      reproj_thresh, max_transitivity)
    except Exception:                     # noqa: BLE001
        logger.exception("[complete_tracks] 加速路径异常，回退纯 Python")
        return _complete_tracks_py(keyframes, frames, map_points, K,
                                   reproj_thresh, max_transitivity)


def _triangulate_next_best_view(curr_idx, keyframes, frames, map_points,
                                focal, fy, cx, cy, triang_thresh, top_k=8):
    if curr_idx not in frames:
        return 0
    if frames[curr_idx].pose is None:
        return 0
    curr_pose = frames[curr_idx].pose
    curr_fm = frames[curr_idx].feat_map
    if curr_fm is None:
        return 0
    curr_pts = set(int(x) for x in curr_fm if x >= 0)
    scores = []
    for kf in keyframes:
        if kf == curr_idx:
            continue
        kf_pose = frames[kf].pose
        if kf_pose is None:
            continue
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        kf_pts = set(int(x) for x in kf_fm if x >= 0)
        covis = len(curr_pts & kf_pts)
        baseline = float(np.linalg.norm(curr_pose.center - kf_pose.center))
        if baseline < 1e-6:
            continue
        scores.append((kf, float(covis) + 0.1 * float(np.log1p(baseline))))
    if not scores:
        return 0
    scores.sort(key=lambda x: -x[1])
    selected = [kf for kf, _ in scores[:top_k]]
    K = _build_K(focal, fy, cx, cy)
    err_thresh = MAX_REPROJ_ERROR * triang_thresh
    err_thresh *= ADD_OBS_ERR_SCALE
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    new_pts_total = 0
    for kf in selected:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        a = frames[kf]
        b = frames[curr_idx]
        if a.pose is None or b.pose is None:
            continue
        if a.desc is None or b.desc is None:
            continue
        nt, md, _ = get_metric(a.desc)
        if len(a.desc) < 4 or len(b.desc) < 4:
            continue
        matches = _match_ratio_absolute(a.desc, b.desc, nt, md)
        if len(matches) < 8:
            continue
        # 只用 F/H 几何验证过的匹配建点：ratio test 的误匹配会直接变成错点。
        pts_a_all = np.array([a.kps[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_b_all = np.array([b.kps[m.trainIdx].pt for m in matches], dtype=np.float32)
        vmask, _, _ = _verify_matches_fh(pts_a_all, pts_b_all,
                                         _cfg().ransac_thresh * 1.5, 10)
        if vmask is None or int(vmask.sum()) < 8:
            continue
        a_fm_now = frames[kf].feat_map
        curr_fm_now = frames[curr_idx].feat_map
        tri_pairs = []
        for mi, m in enumerate(matches):
            if not vmask[mi]:
                continue
            ka = int(m.queryIdx)
            kb = int(m.trainIdx)
            if ka >= len(a_fm_now) or kb >= len(curr_fm_now):
                continue
            pa = int(a_fm_now[ka])
            pb = int(curr_fm_now[kb])
            if pa < 0 and pb < 0:
                tri_pairs.append((ka, kb))
        if len(tri_pairs) < 4:
            continue
        pts_obs_map = {}
        for k, (ka, kb) in enumerate(tri_pairs):
            pts_obs_map[k] = [
                (kf, float(a.kps[ka].pt[0]), float(a.kps[ka].pt[1])),
                (curr_idx, float(b.kps[kb].pt[0]), float(b.kps[kb].pt[1])),
            ]
        result = _triangulate_multiview_batch(pts_obs_map, frames, K, min_tri_deg, triang_thresh, _MEM_GUARD.current)
        for k, (X, valid) in result.items():
            if not valid:
                continue
            ka, kb = tri_pairs[k]
            if curr_fm_now[kb] >= 0 or a_fm_now[ka] >= 0:
                continue
            pt_idx = len(map_points)
            c_a = a.colors[ka] if (a.colors is not None and ka < len(a.colors)) else None
            c_b = b.colors[kb] if (b.colors is not None and kb < len(b.colors)) else None
            new_pt = MapPoint(idx=pt_idx, xyz=X.astype(np.float32), desc=a.desc[ka].copy())
            new_pt.add_observation(Observation(kf, ka,
                                               float(a.kps[ka].pt[0]),
                                               float(a.kps[ka].pt[1]),
                                               (c_a.copy() if c_a is not None else None)))
            new_pt.add_observation(Observation(curr_idx, kb,
                                               float(b.kps[kb].pt[0]),
                                               float(b.kps[kb].pt[1]),
                                               (c_b.copy() if c_b is not None else None)))
            map_points.append(new_pt)
            a_fm_now[ka] = pt_idx
            curr_fm_now[kb] = pt_idx
            new_pts_total += 1
    return new_pts_total


def _project_and_add_observations(curr_idx, keyframes, frames, map_points,
                                  focal, fy, cx, cy, n_views=10,
                                  proj_radius_px=TRACK_GATE_PX):
    if curr_idx not in frames:
        return 0
    if frames[curr_idx].pose is None:
        return 0
    curr = frames[curr_idx]
    curr_pose = curr.pose
    curr_desc = curr.desc
    curr_fm = curr.feat_map
    if curr_desc is None or curr_fm is None:
        return 0
    if len(curr.kps) == 0:
        return 0
    if len(keyframes) > n_views:
        neighbors = keyframes[-n_views:]
    else:
        neighbors = list(keyframes)
    neighbors = [k for k in neighbors if k != curr_idx and k in frames and frames[k].pose is not None]
    if not neighbors:
        return 0
    cfg = _cfg()
    min_angle = cfg.filter_min_tri_angle_deg
    pts_seen = set()
    candidates = []
    for kf in neighbors:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        for pt_idx in kf_fm:
            pt_idx = int(pt_idx)
            if pt_idx < 0 or pt_idx >= len(map_points):
                continue
            if pt_idx in pts_seen:
                continue
            pts_seen.add(pt_idx)
            pt = map_points[pt_idx]
            if pt.obs_count >= MAX_OBS_PER_POINT:
                continue
            Xw = pt.xyz
            if Xw.size != 3 or not np.isfinite(Xw).all():
                continue
            candidates.append((pt_idx, Xw))
    if not candidates:
        return 0
    R = curr_pose.R
    t = curr_pose.t
    K64 = _build_K(focal, fy, cx, cy)
    max_desc_dist = _update_dist_for(curr_desc)
    if curr.kdtree is None:
        curr._build_kdtree()
    tree = curr.kdtree
    kps_uv = np.array([kp.pt for kp in curr.kps], dtype=np.float32)
    used_kps = set()
    added = 0
    for pt_idx, Xw in candidates:
        Xc = R @ Xw.reshape(3, 1) + t
        d = float(Xc[2, 0])
        if d <= 1e-6:
            continue
        u = K64[0, 0] * float(Xc[0, 0]) / d + K64[0, 2]
        v = K64[1, 1] * float(Xc[1, 0]) / d + K64[1, 2]
        if not (0 <= u < 1e6 and 0 <= v < 1e6):
            continue
        if tree is not None:
            cand = tree.query_ball_point([u, v], proj_radius_px)
        else:
            dx = kps_uv[:, 0] - u
            dy = kps_uv[:, 1] - v
            cand = np.where(dx * dx + dy * dy <= proj_radius_px ** 2)[0].tolist()
        if not cand:
            continue
        pt = map_points[pt_idx]
        if not _check_obs_angle(pt, curr_idx, frames, min_angle):
            continue
        desc_pt = pt.desc
        best_kp = -1
        best_d = None
        for ci in cand:
            ci = int(ci)
            if ci in used_kps or curr_fm[ci] >= 0:
                continue
            if desc_pt is not None and curr_desc is not None and curr_desc.dtype == desc_pt.dtype:
                if curr_desc.dtype == np.uint8:
                    dval = int(cv2.norm(curr_desc[ci], desc_pt, cv2.NORM_HAMMING))
                else:
                    dval = float(cv2.norm(curr_desc[ci], desc_pt, cv2.NORM_L2))
                if best_d is None or dval < best_d:
                    best_d, best_kp = dval, ci
            else:
                if best_kp < 0:
                    best_kp = ci
        if best_kp < 0:
            continue
        if best_d is not None and best_d > max_desc_dist:
            continue
        if curr.colors is not None and best_kp < len(curr.colors):
            c = curr.colors[best_kp]
        else:
            c = None
        if pt.add_observation(Observation(curr_idx, best_kp,
                                          float(curr.kps[best_kp].pt[0]),
                                          float(curr.kps[best_kp].pt[1]),
                                          (c.copy() if c is not None else None))):
            curr_fm[best_kp] = pt_idx
            _update_desc(pt, curr.desc[best_kp])
            used_kps.add(best_kp)
            added += 1
    return added


def _global_match_expand(curr_idx, keyframes, frames, loop_detector, map_points,
                         focal, fy, cx, cy, reproj_thresh):
    frame = frames[curr_idx]
    if frame.pose is None or frame.desc is None:
        return 0, 0
    if not loop_detector.trained:
        return 0, 0
    cands = loop_detector.query(curr_idx, frame.desc, keyframes, skip_recent=0,
                                min_score=LOOP_MIN_SCORE,
                                max_candidates=GLOBAL_MATCH_TOP_K)
    if not cands:
        # 触发条件满足但检索不到候选：这是"开了却什么都不做"最常见的原因
        logger.debug("[global] kf=%d 无 BoW 候选（min_score=%.2f）",
                     curr_idx, LOOP_MIN_SCORE)
        return 0, 0
    K = _build_K(focal, fy, cx, cy)
    K64 = K.astype(np.float64)
    curr_fm = frame.feat_map
    new_pts = 0
    err_thresh = reproj_thresh * MAX_REPROJ_ERROR
    for kf, _ in cands:
        if kf == curr_idx:
            continue
        if abs(kf - curr_idx) < GLOBAL_MATCH_SKIP_RECENT:
            continue
        kf_frame = frames.get(kf)
        if kf_frame is None or kf_frame.pose is None:
            continue
        kf_fm = kf_frame.feat_map
        if kf_fm is None:
            continue
        # 只拿尚未绑定地图点的关键点建新点；写成 >= 0 会与下面的跳过条件互斥，
        # kp_pairs 恒空、新点恒为 0。
        bound = np.where(kf_fm < 0)[0]
        if len(bound) < 4:
            continue
        nt, md, _ = get_metric(kf_frame.desc[bound])
        matches = _match_ratio_absolute(kf_frame.desc[bound], frame.desc, nt, md)
        if len(matches) < GLOBAL_MATCH_MIN_INLIERS:
            continue
        # 同 _triangulate_next_best_view：建点前先做 F/H 几何验证
        pts_a = np.array([kf_frame.kps[int(bound[m.queryIdx])].pt for m in matches],
                         dtype=np.float32)
        pts_b = np.array([frame.kps[m.trainIdx].pt for m in matches], dtype=np.float32)
        vmask, _, _ = _verify_matches_fh(pts_a, pts_b, _cfg().ransac_thresh * 1.5, 10)
        if vmask is None or int(vmask.sum()) < 8:
            continue
        pts_obs_map = {}
        kp_pairs = []
        for mi, m in enumerate(matches):
            if not vmask[mi]:
                continue
            kf_kp = int(bound[m.queryIdx])
            curr_kp = int(m.trainIdx)
            if curr_kp >= len(curr_fm):
                continue
            if curr_fm[curr_kp] >= 0:
                continue
            if 0 <= int(kf_fm[kf_kp]) < len(map_points):
                continue
            kp_pairs.append((kf_kp, curr_kp))
            pts_obs_map[len(kp_pairs) - 1] = [
                (kf, float(kf_frame.kps[kf_kp].pt[0]), float(kf_frame.kps[kf_kp].pt[1])),
                (curr_idx, float(frame.kps[curr_kp].pt[0]), float(frame.kps[curr_kp].pt[1])),
            ]
        if not pts_obs_map:
            continue
        result = _triangulate_multiview_batch(pts_obs_map, frames, K64,
                                              _cfg().filter_min_tri_angle_deg,
                                              err_thresh, _MEM_GUARD.current)
        for k, (X, valid) in result.items():
            if not valid:
                continue
            kf_kp, curr_kp = kp_pairs[k]
            if curr_fm[curr_kp] >= 0:
                continue
            pt_idx = len(map_points)
            new_pt = MapPoint(idx=pt_idx, xyz=X.astype(np.float32), desc=kf_frame.desc[kf_kp].copy())
            new_pt.add_observation(Observation(kf, kf_kp,
                                               float(kf_frame.kps[kf_kp].pt[0]),
                                               float(kf_frame.kps[kf_kp].pt[1])))
            new_pt.add_observation(Observation(curr_idx, curr_kp,
                                               float(frame.kps[curr_kp].pt[0]),
                                               float(frame.kps[curr_kp].pt[1])))
            map_points.append(new_pt)
            curr_fm[curr_kp] = pt_idx
            # 两侧 feat_map 都要回写：只写 curr_fm 会让 (kf, kf_kp) 在 kf 侧没有回指，
            # filter 从 obs 重建 feat_map 时一个关键点会被两个点争用。
            kf_fm[kf_kp] = pt_idx
            new_pts += 1
    # 记录候选数与新增点数，便于判断全局匹配是否真正生效。
    if new_pts:
        logger.info("[global] kf=%d 候选=%d 新点=%d", curr_idx, len(cands), new_pts)
    else:
        logger.debug("[global] kf=%d 候选=%d 新点=0", curr_idx, len(cands))
    return len(cands), new_pts


def _filter_window_by_tri_angle(window, frames, map_points, min_deg=6.0,
                                min_ratio=0.2, min_keep=3):
    """局部 BA 选窗的三角化角筛选：按与当前帧（`window[0]`）的共享点基线角保留帧。

    共享点里基线角 >= `min_deg` 的个数达到 `max(8, min_ratio * 共享点数)` 才保留；
    筛选后不足 `min_keep` 帧就退回原窗口，宁可少筛也不能让局部 BA 失去支撑。
    """
    if len(window) < 3:
        return window
    base = window[0]
    fb = frames.get(base)
    if fb is None or fb.pose is None or fb.feat_map is None:
        return window
    pid_base = {int(x) for x in fb.feat_map if x >= 0}
    if len(pid_base) < 10:
        return window
    cb = np.asarray(fb.pose.center, dtype=np.float64).reshape(3)
    kept = [base]
    n_pts = len(map_points)
    for k in window[1:]:
        fk = frames.get(k)
        if fk is None or fk.pose is None or fk.feat_map is None:
            continue
        pid_k = {int(x) for x in fk.feat_map if x >= 0}
        shared = [p for p in pid_base if p < n_pts and p in pid_k]
        if len(shared) < 8:
            continue
        xyz = np.array([map_points[p].xyz for p in shared], dtype=np.float64)
        ck = np.asarray(fk.pose.center, dtype=np.float64).reshape(3)
        r1 = xyz - cb
        r2 = xyz - ck
        r1 = r1 / np.maximum(np.linalg.norm(r1, axis=1, keepdims=True), 1e-12)
        r2 = r2 / np.maximum(np.linalg.norm(r2, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(
            np.clip(np.einsum("ij,ij->i", r1, r2), -1.0, 1.0)))
        wide = int((ang >= min_deg).sum())
        if wide >= max(8, int(min_ratio * len(shared))):
            kept.append(k)
    if len(kept) < min_keep:
        return window
    return kept


def _local_ba_refinement(window, map_points, frames, focal, fy, cx, cy,
                         reproj_thresh, image_size, focal_init, covis_graph, max_rounds=None,
                         fix_focal=False, k1=0.0, optimize_k1=False):
    """局部 BA 与 EM 数据关联交替迭代：每轮 BA 内 EM 只走 2 次，让数据关联承担收敛。"""
    cfg = _cfg()
    if max_rounds is None:
        max_rounds = max(3, cfg.ba_local_refinements + 1)
    local_em_iters = 2
    prev_weights = None
    prev_rms = None
    success_any = False
    window_set = set(window)
    # 内参 gating（内部开关 SFM_LOCAL_BA_INTR，默认关）：共享内参被所有帧共同约束，
    # 窗口只覆盖一小部分帧时放开 f / k1 会跟着局部几何漂，只有覆盖全部帧才允许放开。
    _intr_gated = False
    if os.environ.get("SFM_LOCAL_BA_INTR", "0").strip() not in ("", "0"):
        _n_registered = sum(1 for _fr in frames.values() if _fr.pose is not None)
        _intr_gated = len(window_set) < _n_registered
    window_pts = set()
    for kf in window:
        fm = frames[kf].feat_map
        if fm is not None:
            window_pts.update(int(x) for x in fm if x >= 0)
    fixed_kfs = set()
    for pt_idx in window_pts:
        if pt_idx >= len(map_points):
            continue
        for o in map_points[pt_idx].obs:
            if o.frame_idx not in window_set:
                fixed_kfs.add(o.frame_idx)
    fixed_pts = set()
    for kf in fixed_kfs:
        fm = frames[kf].feat_map
        if fm is not None:
            fixed_pts.update(int(x) for x in fm if x >= 0)
    for rnd in range(max_rounds):
        ba_state = {"success": False}
        # 窗口外帧对窗口内地图点的观测也纳入残差（只加这些，不把整帧的所有观测
        # 都塞进来——那样局部 BA 规模会翻倍）。位姿由 fixed_kfs 固定。
        _extra_obs = []
        for pt_idx in window_pts:
            if pt_idx >= len(map_points):
                continue
            for o in map_points[pt_idx].obs:
                if o.frame_idx in window_set:
                    continue
                _extra_obs.append((o.frame_idx, pt_idx, o.u, o.v))
        if len(_extra_obs) > 8000:
            _extra_obs = _extra_obs[:8000]
        # 有外部观测时不再冻住窗口点，只固定窗口外帧。
        _fixed_pts_arg = None if _extra_obs else fixed_pts
        _fix_focal_now = fix_focal or _intr_gated
        _opt_k1_now = optimize_k1 and not _intr_gated
        focal, fy, cx, cy, k1 = bundle_adjustment(
            window, map_points, frames, focal, fy, cx, cy,
            optimize_points=True, reproj_thresh=reproj_thresh,
            image_size=image_size, is_global=False,
            focal_init=focal_init, optimize_principal=False,
            fixed_kfs=fixed_kfs, fixed_pts=_fixed_pts_arg, _state=ba_state,
            fix_focal=_fix_focal_now,
            focal_prior=(None if _fix_focal_now else focal),
            focal_prior_sigma=(0.0 if _fix_focal_now
                               else max(0.5, focal * LOCAL_FOCAL_PRIOR_RATIO)),
            em_iters=local_em_iters,
            prev_obs_weights=prev_weights,
            k1=k1, optimize_k1=_opt_k1_now,
            extra_obs=_extra_obs,
        )
        if not ba_state["success"]:
            break
        success_any = True
        prev_weights = ba_state.get("obs_weights")
        K_now = _build_K(focal, fy, cx, cy)
        _retriangulate_multiview(map_points, frames, K_now, min_obs=MULTIVIEW_MIN_OBS)
        _complete_tracks(window, frames, map_points, K_now, reproj_thresh,
                         max_transitivity=cfg.complete_max_transitivity)
        _em_reassign_local(window, map_points, frames, focal, fy, cx, cy)
        # 硬过滤：EM 只降权，外点永远出不了局；这里真正剔除坏观测与坏点
        filter_observations(map_points, frames, focal, fy, cx, cy)
        poses_by_frame = {f: fr.pose for f, fr in frames.items()}
        mean_err, valid_count, neg_ratio = _compute_point_errors(
            map_points, poses_by_frame, focal, fy, cx, cy
        )
        valid_mask = (valid_count >= MIN_OBSERVATIONS) & (neg_ratio < 0.5)
        if np.any(valid_mask):
            rms = float(np.median(mean_err[valid_mask]))
            if prev_rms is not None and rms >= prev_rms * 0.98:
                break
            prev_rms = rms
    return focal, fy, cx, cy, k1, success_any


def _loop_closure_ba(loop_pairs, loop_constraints, frames, map_points, keyframes,
                     covis_graph, focal, fy, cx, cy,
                     ransac_thresh, reproj_thresh, triang_thresh,
                     image_size, focal_init, norm_type, match_dist,
                     fix_focal=False, k1=0.0, optimize_k1=False,
                     max_points_per_loop=300, max_points_total=2000,
                     ba_rounds=1, window_radius=2, detach_old=True):
    """回环闭合 BA：端点独立重三角化 + 邻域窗口 BA。

    用回环测量位姿重新三角化端点匹配，并把关键点从旧地图点摘下挂到新点，
    避免同一像素重复计残差；随后主流程的最终全局 BA 再统一约束。
    """
    stats = {"loops": 0, "matched": 0, "inliers": 0, "tri": 0, "added": 0,
             "claimed": 0, "discard": {}, "ba_ok": False}
    if not loop_pairs or not map_points:
        return focal, fy, cx, cy, k1, stats

    def _discard(reason):
        stats["discard"][reason] = stats["discard"].get(reason, 0) + 1

    def _detach(frame, kp):
        """把关键点从它当前的地图点上摘掉（只改点与 feat_map，不删除点本身）。"""
        fm = frame.feat_map
        if fm is None or kp < 0 or kp >= len(fm):
            return
        pi = int(fm[kp])
        fm[kp] = -1
        if 0 <= pi < len(map_points):
            pt = map_points[pi]
            pt.obs = [o for o in pt.obs
                      if not (o.frame_idx == frame.idx and o.kp_idx == kp)]
            pt.obs_set.discard((frame.idx, kp))

    tmap = {}
    for a, b, T in loop_constraints:
        tmap.setdefault((a, b), T)

    K64 = np.asarray(_build_K(focal, fy, cx, cy), dtype=np.float64)
    min_angle = float(_cfg().filter_min_tri_angle_deg)
    max_reproj = float(triang_thresh)
    claimed = set()
    seen = set()
    total_added = 0
    window_frames = set()

    for a, b in loop_pairs:
        key = (int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        stats["loops"] += 1
        if total_added >= max_points_total:
            _discard("总量上限")
            continue
        T_meas = tmap.get(key)
        fa, fb = frames.get(a), frames.get(b)
        if T_meas is None or fa is None or fb is None:
            _discard("无测量")
            continue
        if fa.pose is None or fb.pose is None:
            _discard("无位姿")
            continue
        if abs(key[1] - key[0]) <= 1:
            _discard("相邻")
            continue
        if fa.desc is None or fb.desc is None:
            _discard("无描述子")
            continue
        nt, md, _ = get_metric(fa.desc)
        matches = _match_ratio_absolute(fa.desc, fb.desc, nt, md)
        if len(matches) < 15:
            _discard("匹配不足")
            continue
        stats["matched"] += 1
        pts_a = np.array([fa.kps[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts_b = np.array([fb.kps[m.trainIdx].pt for m in matches], dtype=np.float64)
        try:
            F, mask_f = cv2.findFundamentalMat(
                pts_a.reshape(-1, 1, 2), pts_b.reshape(-1, 1, 2),
                cv2.FM_RANSAC, max(1.0, reproj_thresh * 1.5), 0.999,
                maxIters=2000)
        except cv2.error:
            F = mask_f = None
        if F is None or mask_f is None:
            _discard("F 失败")
            continue
        mask = mask_f.ravel().astype(bool)
        n_inl = int(mask.sum())
        if n_inl < 10 or n_inl / max(1, len(matches)) < LOOP_MIN_INLIER_RATIO:
            _discard("F 内点不足")
            continue
        stats["inliers"] += n_inl

        obs_map = {}
        kp_by_mi = {}
        for mi in np.flatnonzero(mask):
            m = matches[int(mi)]
            q, t = int(m.queryIdx), int(m.trainIdx)
            if (key[0], q) in claimed or (key[1], t) in claimed:
                stats["claimed"] += 1
                continue
            if len(obs_map) >= max_points_per_loop:
                break
            if total_added + len(obs_map) >= max_points_total:
                break
            kp_by_mi[int(mi)] = (q, t)
            obs_map[int(mi)] = [
                (key[0], float(fa.kps[q].pt[0]), float(fa.kps[q].pt[1])),
                (key[1], float(fb.kps[t].pt[0]), float(fb.kps[t].pt[1])),
            ]
        if not obs_map:
            _discard("无可加点")
            continue

        # 用回环测量位姿三角化：b 的位姿临时换成 a ∘ T_meas，算完立刻还原。
        pose_b_saved = fb.pose
        fb.pose = _compose_pose(fa.pose, T_meas)
        try:
            result = _triangulate_multiview_batch(
                obs_map, frames, K64, min_angle, max_reproj,
                _MEM_GUARD.current)
        finally:
            fb.pose = pose_b_saved

        stats["tri"] += len(result)
        added_here = 0
        for mi, (X, valid) in result.items():
            if not valid:
                continue
            q, t = kp_by_mi[int(mi)]
            claimed.add((key[0], q))
            claimed.add((key[1], t))
            pt_idx = len(map_points)
            pt = MapPoint(idx=pt_idx,
                          xyz=np.asarray(X, dtype=np.float32).reshape(3),
                          desc=fa.desc[q].copy())
            pt.add_observation(Observation(
                key[0], q, float(fa.kps[q].pt[0]), float(fa.kps[q].pt[1])))
            pt.add_observation(Observation(
                key[1], t, float(fb.kps[t].pt[0]), float(fb.kps[t].pt[1])))
            map_points.append(pt)
            if detach_old:
                _detach(fa, q)
                _detach(fb, t)
                fa.feat_map[q] = pt_idx
                fb.feat_map[t] = pt_idx
            added_here += 1
        total_added += added_here
        stats["added"] += added_here
        if added_here:
            window_frames.add(key[0])
            window_frames.add(key[1])

    if total_added and window_frames:
        kf_sorted = sorted(k for k in (set(keyframes) | window_frames)
                           if frames.get(k) is not None and frames[k].pose is not None)
        pos = {k: i for i, k in enumerate(kf_sorted)}
        win = set()
        for k in window_frames:
            if k not in pos:
                continue
            p0 = pos[k]
            for q in range(max(0, p0 - window_radius),
                           min(len(kf_sorted), p0 + window_radius + 1)):
                win.add(kf_sorted[q])
        win = sorted(win)
        if len(win) >= 2:
            focal, fy, cx, cy, k1, ok = _local_ba_refinement(
                win, map_points, frames, focal, fy, cx, cy,
                reproj_thresh, image_size, focal_init, covis_graph,
                max_rounds=max(1, int(ba_rounds)), fix_focal=fix_focal, k1=k1,
                optimize_k1=optimize_k1)
            stats["ba_ok"] = bool(ok)
    return focal, fy, cx, cy, k1, stats


# ---- 13. 主流程 ----
def _remove_frame(idx, frames, map_points, keyframes):
    """把一个帧从重建结果里摘掉：位姿、状态、观测、feat_map、关键帧列表。"""
    fr = frames.get(idx)
    if fr is None:
        return
    fr.pose = None
    fr.status = FrameStatus.LOST
    fr.is_keyframe = False
    if fr.feat_map is not None:
        fr.feat_map[:] = -1
    for pt in map_points:
        if pt.obs and any(o.frame_idx == idx for o in pt.obs):
            pt.obs = [o for o in pt.obs if o.frame_idx != idx]
    if idx in keyframes:
        keyframes.remove(idx)


def _rel_rotation_2view(fa, fb, norm_type, match_dist, K, ransac_thresh):
    """只用两帧图像估计 a→b 的相对旋转（x_b = R x_a + t）；不可靠时返回 None。"""
    if fa.desc is None or fb.desc is None or len(fa.kps) < 8 or len(fb.kps) < 8:
        return None
    m = _match_ratio_absolute(fa.desc, fb.desc, norm_type, match_dist)
    if len(m) < 30:
        return None
    p1 = np.array([fa.kps[x.queryIdx].pt for x in m], dtype=np.float32)
    p2 = np.array([fb.kps[x.trainIdx].pt for x in m], dtype=np.float32)
    E, mask = _find_essential(p1, p2, K, ransac_thresh)
    if E is None or mask is None or int(mask.sum()) < 20:
        return None
    _, R_rel, _, mask_pose = _recover_pose(E, p1, p2, K)
    if mask_pose is None or int(mask_pose.sum()) < 15:
        return None
    return R_rel


def _drop_pose_outliers(frames, map_points, keyframes, focal, fy, cx, cy,
                        norm_type, match_dist, ransac_thresh,
                        max_deg=POSE_OUTLIER_DEG, min_neighbors=2,
                        mark_only=False):
    """用两视图几何独立校验已注册帧，剔除与图像证据矛盾的位姿。

    「自洽地错注册」的帧在 BA 里重投影很小，只有回到原始图像重估相对旋转才能发现；
    判据取与邻居的最小偏差（中位数会被单个错邻居带偏）。连续多帧自洽的错误块无法
    靠局部一致性分离，需要全局约束。
    """
    ids = sorted(k for k, f in frames.items() if f.pose is not None)
    if len(ids) < 6:
        return []
    K = _build_K(focal, fy, cx, cy)
    pos = {f: i for i, f in enumerate(ids)}
    dropped = []
    for k in ids:
        i = pos[k]
        neigh = [ids[j] for j in (i - 2, i - 1, i + 1, i + 2)
                 if 0 <= j < len(ids)]
        devs = []
        for nb in neigh:
            R_rel = _rel_rotation_2view(frames[k], frames[nb], norm_type,
                                        match_dist, K, ransac_thresh)
            if R_rel is None:
                continue
            R_map = frames[nb].pose.R @ frames[k].pose.R.T
            devs.append(_rotation_angle_deg(R_map @ R_rel.T))
        if len(devs) >= min_neighbors:
            worst = float(np.min(devs))
            if worst > max_deg:
                dropped.append((k, worst, len(devs)))
    if not dropped:
        logger.info("[pose-check] %d 个已注册帧与两视图几何一致（阈值 %.0f°）",
                    len(ids), max_deg)
        return []
    logger.warning("[pose-check] 剔除 %d 个位姿与图像证据矛盾的帧（阈值 %.0f°）",
                   len(dropped), max_deg)
    for k, worst, n in dropped:
        logger.warning("  frame %d: 与最近邻居的最小相对旋转偏差 %.1f°（%d 个估计）",
                       k, worst, n)
    for k, _, _ in dropped:
        if mark_only:
            # 只从输出摘掉、不动地图：中途删帧会改变后续 BA / 自标定的数值路径，
            # 可能级联出大幅注册率变化，收益被混沌吞掉。
            fr = frames.get(k)
            if fr is not None:
                fr.pose = None
                fr.status = FrameStatus.LOST
                fr.is_keyframe = False
                if k in keyframes:
                    keyframes.remove(k)
        else:
            _remove_frame(k, frames, map_points, keyframes)
    return [k for k, _, _ in dropped]


def _detect_anomaly_blocks(frames, map_points, min_shared=10, min_block=2,
                           max_ratio=0.10, mark_only=True):
    """共视聚类找出与主图断开的帧块：连续多帧自洽的错误块重投影小、局部旋转一致，
    唯一证据是与主图近乎不共视。以共享点建图求连通分量，最大分量为主体，其余规模
    受限且低于中位数的小簇摘除；单帧分量交给 `_drop_pose_outliers`，只改输出不动地图。
    """
    ids = sorted(k for k, f in frames.items() if f.pose is not None)
    if len(ids) < 4:
        return []
    idx = {k: i for i, k in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 逐点统计共视对：点数 ~5k、单点观测 2–6，O(k^2) 可接受
    shared = {}
    for p in map_points:
        obs = sorted({idx[o.frame_idx] for o in p.obs if o.frame_idx in idx})
        for ai in range(len(obs)):
            for bi in range(ai + 1, len(obs)):
                key = (obs[ai], obs[bi])
                shared[key] = shared.get(key, 0) + 1
    for (a, b), n in shared.items():
        if n >= min_shared:
            union(a, b)

    comp = {}
    for i in range(len(ids)):
        comp.setdefault(find(i), []).append(i)
    groups = sorted(comp.values(), key=len, reverse=True)
    if len(groups) <= 1:
        logger.info("[anomaly] 共视聚类：单一分量 %d 帧，无异常块", len(ids))
        return []
    sizes = np.array([len(g) for g in groups if len(g) >= min_block],
                     dtype=np.float64)
    med = float(np.median(sizes)) if sizes.size else 0.0
    cap = max(min_block, int(max_ratio * len(groups[0])))
    anomaly = []
    for gi, g in enumerate(groups):
        mark = (gi > 0 and min_block <= len(g) <= cap and len(g) < med)
        tag = "主体" if gi == 0 else ("标记" if mark else "保留")
        logger.info("[anomaly] 分量 %d：%d 帧（帧 %d–%d）%s",
                    gi, len(g), ids[g[0]], ids[g[-1]], tag)
        if mark:
            anomaly.extend(ids[i] for i in g)
    if anomaly and mark_only:
        for k in anomaly:
            fr = frames.get(k)
            if fr is not None:
                fr.pose = None
                fr.status = FrameStatus.LOST
                fr.is_keyframe = False
        logger.warning("[anomaly] 从输出摘除 %d 帧的异常块：%s",
                       len(anomaly), sorted(anomaly))
    elif anomaly:
        logger.warning("[anomaly] 检出 %d 帧异常块（未摘除）：%s",
                       len(anomaly), sorted(anomaly))
    else:
        logger.info("[anomaly] 没有需要摘除的异常块")
    return anomaly


def _filter_rotation_edges(edges, frames, max_deg=10.0):
    """旋转边先按当前位姿残差删边，再取最大连通分量。

    两视图 E 矩阵在弱纹理 / 长基线对上偶尔给出错边，而错边在最终 BA 里权重很大，
    会把整片帧拉歪；删掉残差 > max_deg 的边并保留最大分量即可隔离碎片。
    """
    if not edges:
        return edges
    kept = []
    dropped = 0
    for a, b, R_meas in edges:
        fa, fb = frames.get(a), frames.get(b)
        if fa is None or fb is None or fa.pose is None or fb.pose is None:
            continue
        R_map = fb.pose.R @ fa.pose.R.T
        if _rotation_angle_deg(R_map @ R_meas.T) <= max_deg:
            kept.append((a, b, R_meas))
        else:
            dropped += 1
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in kept:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comp = {}
    for a, b, _ in kept:
        comp.setdefault(find(a), set()).update((a, b))
    if len(comp) > 1:
        big = max(comp.values(), key=len)
        kept = [e for e in kept if e[0] in big and e[1] in big]
        logger.info("[rot-edge] 残差过滤：删 %d 条 >%.0f°，最大连通分量 %d 帧，"
                    "保留 %d/%d 条", dropped, max_deg, len(big), len(kept),
                    len(edges))
    else:
        logger.info("[rot-edge] 残差过滤：删 %d 条 >%.0f°，单连通分量，保留 %d/%d 条",
                    dropped, max_deg, len(kept), len(edges))
    return kept



def _visible_3d_keypoints(j, frames, reg, norm_type, match_dist, cache):
    """候选帧 j 中能对上已有 3D 点的关键点集合。

    对 j 附近 ±3 的每个已注册帧做一次匹配（按 ("vis", k, j) 缓存），把匹配到的
    已注册关键点经 `feat_map` 映射到 3D 点；同一个候选关键点只计一次。
    """
    lo = bisect.bisect_left(reg, j - 3)
    hi = bisect.bisect_right(reg, j + 3)
    seen = set()
    for k in reg[lo:hi]:
        key = ("vis", k, j)
        ids = cache.get(key)
        if ids is None:
            fk, fj = frames[k], frames[j]
            if fk.desc is None or fj.desc is None or fk.feat_map is None:
                ids = []
            else:
                matches = _match_ratio_absolute(fk.desc, fj.desc,
                                                norm_type, match_dist)
                fm = fk.feat_map
                ids = []
                for m in matches:
                    q = int(m.queryIdx)
                    t = int(m.trainIdx)
                    if 0 <= q < len(fm) and int(fm[q]) >= 0:
                        ids.append(t)
            cache[key] = ids
        seen.update(ids)
    return seen


def _visibility_pyramid_score(uv, width, height, num_levels=6):
    """可见点的空间覆盖分：6 层网格，最粗 2x2、最细 64x64。

    点落到某层的一个空格时该层贡献 `dim²`，同一格重复落点不重复计分。
    裸可见点数只奖励点多的稠密视角，这一项额外奖励「铺得开」。
    """
    if uv is None or len(uv) == 0 or width <= 0 or height <= 0:
        return 0.0
    max_dim = 1 << num_levels
    xs = np.clip((np.asarray(uv[:, 0], dtype=np.float64) * max_dim / float(width)
                  ).astype(np.int64), 0, max_dim - 1)
    ys = np.clip((np.asarray(uv[:, 1], dtype=np.float64) * max_dim / float(height)
                  ).astype(np.int64), 0, max_dim - 1)
    score = 0.0
    for level in range(num_levels):
        dim = 1 << (level + 1)
        shift = num_levels - 1 - level
        if shift:
            codes = (ys >> shift) * dim + (xs >> shift)
        else:
            codes = ys * max_dim + xs
        n_occupied = int(np.unique(codes).size)
        score += float(n_occupied) * float(dim * dim)
    return score


def _choose_next_frame(pending, frames, seeds, norm_type, match_dist, cache,
                       rng=None, tie_band=0, stats=None, score_mode="match",
                       img_size=None, trials=None, max_trials=0, min_vis=0):
    """从待处理帧里挑下一张：在已注册帧 ±3 内按 score_mode 选，并列按索引取最小。

    `match` / `visible` / `pyramid`（默认）分别用最近帧匹配数、可见 3D 点数、
    6 层可见性金字塔覆盖分；`trials`/rng 非空时分别进入注册重试与并列随机模式。
    """
    if not pending:
        return None
    reg = sorted(k for k, f in frames.items() if f.pose is not None)
    if not reg:
        return min(pending)
    reg_set = set(reg)
    # 重试模式：超出次数上限的帧不再进入候选（正常路径在收回时就已剔除）
    eligible = pending
    if trials is not None and max_trials > 0:
        eligible = {j for j in pending if int(trials.get(j, 0)) < max_trials}
        if not eligible:
            return None
    near = [j for j in eligible
            if j not in reg_set and any((j - d) in reg_set or (j + d) in reg_set
                                        for d in range(1, 4))]
    if not near:
        # 链条断了：按到种子帧的距离往外扩，避免所有帧都挤在远端。
        # 重试模式先挑没试过的帧，免得同一帧一口气把重试次数用完。
        if trials is not None:
            return min(eligible, key=lambda j: (int(trials.get(j, 0)),
                                                min(abs(j - s) for s in seeds), j))
        return min(pending, key=lambda j: (min(abs(j - s) for s in seeds), j))
    scored = []
    for j in near:
        t = 0 if trials is None else int(trials.get(j, 0))
        n_vis = None
        if score_mode in ("visible", "pyramid"):
            ids = _visible_3d_keypoints(j, frames, reg, norm_type, match_dist, cache)
            n_vis = len(ids)
            if score_mode == "pyramid":
                if img_size is not None and ids:
                    fj = frames[j]
                    uv = np.array([fj.kps[t2].pt for t2 in ids], dtype=np.float64)
                    score = _visibility_pyramid_score(uv, img_size[0], img_size[1])
                else:
                    score = 0.0
            else:
                score = float(n_vis)
        else:
            pos = bisect.bisect_left(reg, j)
            cands = [reg[p] for p in (pos - 1, pos) if 0 <= p < len(reg)]
            k = min(cands, key=lambda x: abs(x - j))
            key = (k, j)
            score = cache.get(key)
            if score is None:
                score = len(_match_ratio_absolute(frames[k].desc, frames[j].desc,
                                                  norm_type, match_dist))
                cache[key] = score
            if min_vis > 0:
                # 门限只在重试模式下开启；match 模式本身不需要可见集合
                n_vis = len(_visible_3d_keypoints(j, frames, reg, norm_type,
                                                  match_dist, cache))
        deferred = 1 if (min_vis > 0 and (n_vis is None or n_vis < min_vis)) else 0
        scored.append((j, score, t, deferred))
    if not scored:
        return min(eligible)
    best_j, best_score = None, -1.0
    if trials is not None:
        scored.sort(key=lambda x: (x[3], 1 if x[2] > 0 else 0, -x[1], x[2], x[0]))
        best_j, best_score = scored[0][0], scored[0][1]
    else:
        for j, score, _t, _d in scored:
            if score > best_score or (
                    score_mode in ("visible", "pyramid") and score == best_score
                    and (best_j is None or j < best_j)):
                best_j, best_score = j, score
    if rng is not None and tie_band >= 0:
        band = [j for j, s, _t, _d in scored if s >= best_score - tie_band]
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["multi"] = stats.get("multi", 0) + (1 if len(band) > 1 else 0)
            stats["band_sum"] = stats.get("band_sum", 0) + len(band)
            stats["band_max"] = max(stats.get("band_max", 0), len(band))
        if len(band) > 1:
            _pick = int(band[int(rng.integers(len(band)))])
            if stats is not None:
                stats.setdefault("decisions", []).append(
                    {"best": int(best_j), "band": [int(x) for x in band],
                     "chosen": _pick})
            return _pick
    return best_j


def estimate_poses(frame_paths: List[str], feature_type: str, *,
                   focal_guess: Optional[float] = None, aspect_ratio: float = 1.0,
                   dist_coeffs: Optional[np.ndarray] = None,
                   camera_matrix: Optional[np.ndarray] = None,
                   enable_loop: bool = True, enable_pgo: bool = True,
                   enable_global_matching: bool = False,
                   self_calibrate: bool = True,
                   self_calib_iter: Optional[bool] = None,
                   rot_edge_weight: Optional[float] = None,
                   pose_outlier_deg: Optional[float] = None,
                   _pass: int = 1) -> SfMResult:
    global _RUNTIME_CFG
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    if len(frame_paths) < 2:
        raise ValueError("至少需要 2 帧")
    # 预编译模块分平台（当前 CPython 3.11 / win-amd64）。版本对不上会静默回退
    # 纯 Python：结果一致但慢约 5%，因此这里打一行日志明确说明。
    logger.info("[加速] Cython 内核: %s",
                "已启用 (sfm_accel)" if sfm_accel is not None
                else "未启用，使用纯 Python 实现（结果一致，稍慢）")
    if enable_global_matching and not enable_loop:
        logger.warning("[global] 全局匹配已开启但回环已关闭："
                       "BoW 不会训练，全局匹配不会生效")
    ft = feature_type.lower()
    if ft not in ("sift", "rootsift", "orb"):
        raise ValueError(f"不支持的 feature_type={feature_type!r}")
    t_start = time.time()
    first_img = get_image(frame_paths[0])
    if first_img is None:
        raise FileNotFoundError(f"无法读取图像: {frame_paths[0]}")
    h0, w0 = first_img.shape[:2]
    del first_img
    has_intrinsics = camera_matrix is not None
    has_distortion = dist_coeffs is not None
    # pyceres 2.6 在共享内参块被设为常量时偶发内部崩溃，因此默认让 fx 自由优化；
    # fix_focal 接口保留，待绑定层稳定后再启用。
    fix_focal = False
    # 显式参数优先于模块级默认值（后者来自环境变量）
    if self_calib_iter is None:
        self_calib_iter = SELF_CALIB_ITER
    if rot_edge_weight is None:
        rot_edge_weight = ROT_EDGE_WEIGHT
    if pose_outlier_deg is None:
        pose_outlier_deg = POSE_OUTLIER_DEG
    # SFM_ORDER=seq 可退回原来的「按索引顺序处理」，用于 A/B
    _dyn_order = os.environ.get("SFM_ORDER", "dyn") != "seq"
    # 多起点评测（内部）：只重排初始对候选，默认全 0 时与旧行为逐位一致。
    # 在函数内读取，自标定第二 / 三遍的递归调用会自动继承。
    _init_seed = int(os.environ.get("SFM_INIT_SEED", "0") or 0)
    _init_band = int(os.environ.get("SFM_INIT_BAND", "0") or 0)
    _init_skip = int(os.environ.get("SFM_INIT_SKIP", "0") or 0)
    _init_dump = os.environ.get("SFM_DUMP_INIT", "").strip() or None
    if _init_seed or _init_band or _init_skip:
        logger.info("[init] 多起点: seed=%d band=%d skip=%d（pass=%d）",
                    _init_seed, _init_band, _init_skip, _pass)
    # 多起点（内部）：选帧顺序的近分并列随机化。默认 seed=0 时完全关闭。
    _order_seed = int(os.environ.get("SFM_ORDER_SEED", "0") or 0)
    _order_band = int(os.environ.get("SFM_ORDER_BAND", "2") or 2)
    _order_dump = os.environ.get("SFM_DUMP_ORDER", "").strip() or None
    _order_rng = np.random.default_rng(_order_seed) if _order_seed else None
    _order_stats = {} if _order_seed else None
    if _order_seed:
        logger.info("[order] 多起点: seed=%d band=%d（pass=%d）",
                    _order_seed, _order_band, _pass)
    # 顺序复放（内部）：按给定索引序列出队，用于固定不同轨迹的选帧顺序。
    # 文件按当前 pass 取键；缺键直接报错，避免静默退回动态选帧后对照实验失效。
    _order_seq_path = os.environ.get("SFM_ORDER_SEQ", "").strip()
    _order_seq = None
    if _order_seq_path:
        with open(_order_seq_path, "r", encoding="utf-8") as _f_seq:
            _order_seq_all = json.load(_f_seq)
        _seq_key = f"pass{int(_pass)}"
        if _seq_key not in _order_seq_all:
            raise RuntimeError(
                f"SFM_ORDER_SEQ 缺少 {_seq_key}（{_order_seq_path}）"
                f"，现有键: {sorted(_order_seq_all)}")
        _order_seq = [int(x) for x in _order_seq_all[_seq_key]]
        if _order_seed:
            logger.warning("[order] SFM_ORDER_SEQ 已设置，忽略 SFM_ORDER_SEED=%d",
                           _order_seed)
        logger.info("[order] 顺序复放 pass=%d：%d 帧 <- %s",
                    _pass, len(_order_seq), _order_seq_path)
    # 选帧分数：6 层可见性金字塔覆盖分；match / visible 仍保留在 _choose_next_frame 的 API 里。
    _next_score = "pyramid"
    logger.info("[order] 选帧分数: pyramid（6 层可见性金字塔覆盖分，并列按索引）")
    # 回环闭合 BA（实验开关）：默认关；ITERS=0 时整段跳过、不做任何修改，
    # sham 依赖这一点与基线逐位一致。两遍都会生效（自标定第二遍继承环境变量）。
    _loop_ba_on = os.environ.get("SFM_LOOP_BA", "0").strip() not in ("", "0")
    _loop_ba_iters = int(os.environ.get("SFM_LOOP_BA_ITERS", "1") or 1)
    _loop_ba_max_loop = int(
        os.environ.get("SFM_LOOP_BA_MAX_POINTS_PER_LOOP", "300") or 300)
    _loop_ba_max_total = int(
        os.environ.get("SFM_LOOP_BA_MAX_POINTS_TOTAL", "2000") or 2000)
    # 诊断用：0 = 不把回环关键点从旧地图点上摘下来（保留旧观测）。
    _loop_ba_detach = os.environ.get(
        "SFM_LOOP_BA_DETACH", "1").strip() not in ("", "0")
    # edges（默认）= 回环相对位姿残差加进最终全局 BA；points = 旧的重三角化加点方案。
    _loop_ba_mode = (os.environ.get("SFM_LOOP_BA_MODE", "edges").strip().lower()
                     or "edges")
    _loop_ba_weight_env = os.environ.get("SFM_LOOP_BA_WEIGHT", "").strip()
    _loop_ba_weight = (float(_loop_ba_weight_env) if _loop_ba_weight_env
                       else float(ROT_EDGE_WEIGHT))
    if _loop_ba_on:
        logger.info("[loop-ba] 开启：mode=%s iters=%d 每回环上限=%d 总量上限=%d "
                    "detach=%d 权重=%.0f（pass=%d）",
                    _loop_ba_mode, _loop_ba_iters, _loop_ba_max_loop,
                    _loop_ba_max_total, int(_loop_ba_detach),
                    _loop_ba_weight, _pass)
    # 径向畸变自标定只在「完全没有任何内参输入」时启用：
    # 给了 K / 畸变就说明有标定，此时特征已按给定畸变去畸变，内部模型必须保持针孔。
    optimize_k1 = bool(self_calibrate and camera_matrix is None
                       and dist_coeffs is None and _pass == 1)
    k1 = 0.0
    if optimize_k1:
        logger.info("[self-calib] 无标定输入，在 BA 里把 k1 当共享参数自标定")
    _RUNTIME_CFG = derive_auto_config(len(frame_paths), (h0, w0), feature_type, has_intrinsics, has_distortion)
    # 三角化角阈值分层（固定值）：初始化 16° / 点过滤 1.5° / 局部 BA 选窗 6°。
    _RUNTIME_CFG.init_min_parallax_deg = 16.0
    _RUNTIME_CFG.filter_min_tri_angle_deg = 1.5
    _local_win_deg = 6.0
    logger.info("[tri] 阈值：初始化 %.0f° / 点过滤 %.1f° / 局部选窗 %.0f°",
                _RUNTIME_CFG.init_min_parallax_deg,
                _RUNTIME_CFG.filter_min_tri_angle_deg, _local_win_deg)
    # 清理：合并验证 + BA 前删负深度固定开启；全局精化循环仍是实验开关。
    _merge_validate_on = True
    _neg_depth_pre_on = True
    logger.info("[clean] 合并验证 + BA 前删负深度：固定开启")
    _refine_loop_on = (os.environ.get("SFM_REFINE_LOOP", "0").strip()
                       not in ("", "0"))
    if _refine_loop_on:
        logger.info("[clean] 全局精化循环：先补/合并/重三角化，再 BA<->过滤 最多 2 轮")
    # 局部 BA 内参 gating 仍是实验开关
    if os.environ.get("SFM_LOCAL_BA_INTR", "0").strip() not in ("", "0"):
        logger.info("[ba-intr] 局部 BA gating：窗口未覆盖全部已注册帧时固定 f / k1")
    # 全局 BA 固定用 7-DOF Sim3 gauge
    logger.info("[gauge] 全局 BA 用 7-DOF Sim3 gauge：一机全固定，"
                "另一机只固定基线主轴的一维平移")
    # 异常块 + 旋转边残差过滤（固定开启）
    _anomaly_on = True
    logger.info("[anomaly] 共视聚类摘除异常块 + 旋转边残差过滤（10°）")
    # Create–Continue–Complete 语义：Complete 只在最终模型上跑一次。
    _stages_on = (os.environ.get("SFM_STAGES", "0").strip() not in ("", "0"))
    if _stages_on:
        logger.info("[stages] Complete 只在最终一遍执行（第一遍跳过输出装配）")
    # PGO 逐边淘汰 / BA 鲁棒核 / 逐观测 prune（均为实验开关）
    _pgo_edge_filter_on = (os.environ.get("SFM_PGO_EDGE_FILTER", "0").strip()
                           not in ("", "0"))
    if _pgo_edge_filter_on:
        logger.info("[PGO] 逐边淘汰开启：残差 >10° 或 >5%% 尺度的回环边判无效后重解")
    if os.environ.get("SFM_BA_ROBUST", "0").strip() not in ("", "0"):
        logger.info("[BA] 鲁棒核开启：局部 SOFT_L1(1.0)、全局 Huber(1.0)")
    if os.environ.get("SFM_PRUNE_OBS_ONLY", "0").strip() not in ("", "0"):
        logger.info("[prune] 逐观测优先：先删坏观测，点只在观测不足 / 角度不足时删除")
    cfg = _RUNTIME_CFG
    # 注册重试（内部）：默认关，失败帧计数未到上限就放回池里，等地图变大再试。
    # 复放模式下强制关闭，保证每条序列只出队一次。
    _reg_trials_max = max(0, int(os.environ.get("SFM_REG_TRIALS", "0") or 0))
    _reg_retry_on = bool(_reg_trials_max) and _dyn_order and _order_seq is None
    _reg_min_vis = 0
    _reg_min_vis_src = ""
    if _reg_retry_on:
        _reg_min_vis_env = os.environ.get("SFM_REG_MIN_VIS", "").strip()
        if _reg_min_vis_env == "":
            _reg_min_vis = int(cfg.pnp_min_inliers)
            _reg_min_vis_src = "cfg.pnp_min_inliers"
        else:
            _reg_min_vis = max(0, int(_reg_min_vis_env))
            _reg_min_vis_src = "SFM_REG_MIN_VIS"
        logger.info("[order] 注册重试: 每帧最多 %d 次，可见点门限 %d（%s）",
                    _reg_trials_max, _reg_min_vis, _reg_min_vis_src)
    elif _reg_trials_max and _order_seq is not None:
        logger.warning("[order] SFM_ORDER_SEQ 生效，忽略 SFM_REG_TRIALS=%d",
                       _reg_trials_max)
    _reg_trials = {}
    _reg_requeued = 0
    # 三角化角判据固定为 max：存在一对角度 >= 阈值即可。
    _angle_mode = _tri_angle_mode()
    logger.info("[tri] 三角化角判据: max（存在一对 >= 阈值即可）")

    logger.info("=== 提取特征（joblib 多进程）===")
    _force_workers = os.environ.get("SFM_FEAT_WORKERS", "").strip()
    if _force_workers:
        feat_workers = max(1, int(_force_workers))
        logger.info("[feat] SFM_FEAT_WORKERS=%s 覆盖自动并行度", _force_workers)
    else:
        feat_workers = _auto_feature_workers(h0, w0, feature_type)
    logger.info("[feat] %dx%d %s -> %d workers", w0, h0, feature_type, feat_workers)
    frames_list, img_shape, feature_type = extract_features(
        frame_paths, feature_type, dist_coeffs, camera_matrix,
        max_workers=feat_workers)
    frames = {f.idx: f for f in frames_list}
    h, w = img_shape
    if camera_matrix is not None:
        K_in = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        fx, fy = float(K_in[0, 0]), float(K_in[1, 1])
        cx, cy = float(K_in[0, 2]), float(K_in[1, 2])
    else:
        cx, cy = w / 2.0, h / 2.0
        if focal_guess is not None:
            fx = focal_guess
        else:
            fx = max(w, h) * 1.2
        fy = fx * aspect_ratio
    focal0, fy0 = float(fx), float(fy)
    focal_init = focal0
    image_size = max(w, h)
    reproj_thresh = cfg.reproj_thresh
    ransac_thresh = cfg.ransac_thresh
    triang_thresh = cfg.triang_thresh
    sample_desc = next((f.desc for f in frames_list if f.desc is not None and len(f.desc)), None)
    norm_type, match_dist, _ = get_metric(sample_desc)

    map_points = []
    keyframes = []
    covis_graph = CovisibilityGraph()
    loop_detector = LoopDetector(vocab_size=cfg.loop_vocab_size)
    loop_closures = []
    loop_constraints = []
    K_init = _build_K(focal0, fy0, cx, cy)

    logger.info("=== 初始化 ===")
    init = _select_init_pair(frames_list, K_init, ransac_thresh, map_points, frames,
                             focal0, fy0, cx, cy, triang_thresh,
                             init_seed=_init_seed, init_band=_init_band,
                             init_skip=_init_skip, init_dump=_init_dump,
                             init_pass=_pass)
    if init is None:
        raise RuntimeError("无法初始化 SfM: 未找到合适的初始帧对。")
    idx_a, idx_b, pose_b, matches_init, mask_init, _ = init
    keyframes.append(idx_a)
    keyframes.append(idx_b)
    covis_graph.add_keyframe(idx_a)
    covis_graph.add_keyframe(idx_b)
    _triangulate_from_pair(idx_b, idx_a, matches_init, mask_init, frames, map_points,
                           focal0, fy0, cx, cy, triang_thresh)
    # 初始化后必须填充共视图，否则 _try_pnp_guided 的邻居列表为空，
    # 所有后续帧都无法通过 PnP 注册。
    covis_graph.update_from_frame(idx_a, map_points, frames)
    covis_graph.update_from_frame(idx_b, map_points, frames)
    _check_state(frames, map_points, "init")
    logger.info("[init] pair=(%d,%d) matches=%d inliers=%d points=%d",
                idx_a, idx_b, len(matches_init), int(mask_init.sum()), len(map_points))

    # 初始对之后先做一次全局 BA：让焦距与点云在进入主循环前收敛，
    # 否则后续所有 PnP / 三角化都在用未校正的初始焦距。
    if len(map_points) >= 30:
        ba_init_state = {"success": False}
        focal0, fy0, cx, cy, k1 = bundle_adjustment(
            [idx_a, idx_b], map_points, frames, focal0, fy0, cx, cy,
            optimize_points=True, reproj_thresh=reproj_thresh,
            image_size=image_size, is_global=True,
            focal_init=focal_init, _state=ba_init_state,
            em_iters=1, focal_prior=focal0,
            focal_prior_sigma=max(1.0, focal0 * 0.2),
            k1=k1, optimize_k1=optimize_k1, k1_prior_sigma=K1_PRIOR_SIGMA,
        )
        if ba_init_state.get("success"):
            logger.info("[init-ba] fx=%.1f fy=%.1f pts=%d",
                        focal0, fy0, len(map_points))

    kf_since_ba = 0
    kf_since_multiview = 0
    kf_since_global = 0
    kf_since_global_ba = 0
    gm_announced = False
    gm_calls = 0
    gm_hit_calls = 0
    gm_new_pts = 0
    reg_ok_count = 0
    reg_rel_ok = 0
    reg_pnp_ok = 0
    reg_reloc_ok = 0
    reg_total = 0
    reg_fail_reasons = {}
    # 初始对注册完就以当前帧数 / 点数为基准；从 0 起步会让增长触发永远命中不了。
    last_global_ba_n_kf = len(keyframes)
    last_global_ba_n_pts = len(map_points)

    # 处理顺序按与已注册帧的重叠选择：每轮挑重叠最大的待处理帧（见 _choose_next_frame）
    _replay_order = []
    _replay_skipped = 0
    if _order_seq is not None:
        # 复放优先于动态 / 顺序两种模式：完全按给定序列出队。
        _order_pending = set(range(len(frame_paths))) - {idx_a, idx_b}
        _replay_pos = 0

        def _next_frame():
            nonlocal _replay_pos, _replay_skipped
            while _replay_pos < len(_order_seq):
                j = _order_seq[_replay_pos]
                _replay_pos += 1
                if j not in _order_pending:
                    _replay_skipped += 1
                    logger.warning("[order] 复放跳过已出队帧 %d（第 %d/%d 项）",
                                   j, _replay_pos, len(_order_seq))
                    continue
                _order_pending.discard(j)
                _replay_order.append(j)
                return j
            return None
    elif _dyn_order:
        _order_cache = {}
        _order_pending = set(range(len(frame_paths))) - {idx_a, idx_b}

        def _next_frame():
            if not _order_pending:
                return None
            j = _choose_next_frame(_order_pending, frames, (idx_a, idx_b),
                                   norm_type, match_dist, _order_cache,
                                   rng=_order_rng, tie_band=_order_band,
                                   stats=_order_stats, score_mode=_next_score,
                                   img_size=(w, h),
                                   trials=_reg_trials if _reg_retry_on else None,
                                   max_trials=_reg_trials_max,
                                   min_vis=_reg_min_vis)
            _order_pending.discard(j)
            if _order_stats is not None:
                # 记录完整出队序列（含之后被 min_features 跳过的帧），可直接用于复放；
                # 重试模式下同一帧可能出队多次，只保留首次。
                _ord = _order_stats.setdefault("order", [])
                if int(j) not in _ord:
                    _ord.append(int(j))
            return j
    else:
        _seq = [k for k in range(len(frame_paths)) if k not in (idx_a, idx_b)]
        _seq_pos = 0

        def _next_frame():
            nonlocal _seq_pos
            if _seq_pos >= len(_seq):
                return None
            j = _seq[_seq_pos]
            _seq_pos += 1
            return j

    while True:
        i = _next_frame()
        if i is None:
            break
        frame = frames[i]
        if len(frame.kps) < cfg.min_features:
            frame.status = FrameStatus.INVALID
            continue
        reg_total += 1
        # 取索引上最近的已注册帧（两侧都看）。处理顺序非单调递增，
        # 种子帧之前的帧需要把「后一张」当参考，否则 rel 这条路整段用不上。
        prev_idx = None
        _best_d = None
        for k, fk in frames.items():
            if fk.pose is None:
                continue
            d = abs(k - i)
            if _best_d is None or d < _best_d or (d == _best_d and k < i):
                _best_d, prev_idx = d, k
        has_pose = False
        rel_matches = 0
        rel_ok = pnp_ok = reloc_ok = False
        # 每个失败分支都记一句原因：注册率低时不用再猜是匹配、PnP 还是基线的问题
        rel_d = {"m": 0, "e": 0, "p": 0, "base": None, "why": "无已注册前帧"}
        pnp_d = {}
        reloc_d = {}
        if prev_idx is not None:
            prev = frames[prev_idx]
            matches = _match_ratio_absolute(prev.desc, frame.desc, norm_type, match_dist)
            rel_matches = len(matches)
            rel_d["m"] = rel_matches
            rel_d["why"] = "匹配不足"
            if len(matches) >= cfg.init_min_inliers:
                pts_prev = np.array([prev.kps[m.queryIdx].pt for m in matches], dtype=np.float32)
                pts_curr = np.array([frame.kps[m.trainIdx].pt for m in matches], dtype=np.float32)
                E, mask = _find_essential(pts_prev, pts_curr, K_init, ransac_thresh)
                dynamic_min_e = min(cfg.init_min_inliers, max(15, int(0.20 * len(matches))))
                if E is not None and mask is not None:
                    rel_d["e"] = int(mask.sum())
                    rel_d["why"] = "E 内点不足"
                    if int(mask.sum()) >= dynamic_min_e:
                        _, R_rel, t_rel, mask_pose = _recover_pose(E, pts_prev, pts_curr, K_init)
                        if mask_pose is not None:
                            pose_inlier = mask_pose.ravel().astype(bool)
                            rel_d["p"] = int(pose_inlier.sum())
                            rel_d["why"] = "位姿内点不足"
                            dynamic_min = min(cfg.init_min_inliers, max(15, int(0.15 * len(matches))))
                            if int(pose_inlier.sum()) >= dynamic_min:
                                ok, _med = _check_baseline_sufficient(
                                    pts_prev, pts_curr, R_rel, t_rel, K_init, pose_inlier,
                                    min_angle_deg=BASELINE_MIN_ANGLE_DEG,
                                )
                                rel_d["base"] = bool(ok)
                                logger.debug("[rel %d] matches=%d E_inl=%d pose_inl=%d baseline=%s med=%.2f",
                                            i, len(matches), int(mask.sum()),
                                            int(pose_inlier.sum()), ok, _med)
                                if ok:
                                    frame.pose = CameraPose(R_rel @ prev.pose.R, R_rel @ prev.pose.t + t_rel)
                                    frame.status = FrameStatus.TRACKED
                                    frame.track_quality = float(pose_inlier.sum() / len(matches))
                                    has_pose = True
                                    rel_ok = True
                                    rel_d["why"] = ""
                                    _triangulate_from_pair(i, prev_idx, matches, pose_inlier, frames,
                                                           map_points, focal0, fy0, cx, cy, triang_thresh)
                                else:
                                    rel_d["why"] = "基线不足"
                        else:
                            rel_d["why"] = "位姿恢复失败"
                else:
                    rel_d["why"] = "E 矩阵失败"
        if len(keyframes) > 1:
            pose_pnp, _, quality_pnp = _try_pnp_guided(
                frame, keyframes, map_points, frames, covis_graph,
                focal0, fy0, cx, cy, reproj_thresh, loop_detector=loop_detector,
                diag=pnp_d,
            )
            if pose_pnp is not None:
                pnp_ok = True
                if has_pose and frame.pose is not None:
                    dc = np.linalg.norm(pose_pnp.center - frame.pose.center)
                    scene = 1.0
                    if map_points:
                        recent = np.array([p.xyz for p in map_points[-200:]], dtype=np.float32)
                    else:
                        recent = None
                    if recent is not None and len(recent) > 5:
                        c = recent.mean(axis=0)
                        scene = float(np.median(np.linalg.norm(recent - c, axis=1))) + 1e-6
                    if dc <= 3.0 * scene:
                        frame.pose = pose_pnp
                        has_pose = True
                else:
                    frame.pose = pose_pnp
                    has_pose = True
                    if frame.status != FrameStatus.KEYFRAME:
                        frame.status = FrameStatus.TRACKED
                if quality_pnp > frame.track_quality:
                    frame.track_quality = quality_pnp
        reloc_kf = None
        if (not has_pose) or (frame.track_quality < 0.15):
            reloc_result = _try_global_relocalization(
                frame, keyframes, map_points, frames, loop_detector,
                focal0, fy0, cx, cy, reproj_thresh, diag=reloc_d,
            )
            if reloc_result is not None:
                pose_reloc, reloc_kf = reloc_result
                reloc_ok = True
                frame.pose = pose_reloc
                frame.status = FrameStatus.RELOCATED
                has_pose = True
        _final_why = ""
        if not has_pose:
            _final_why = (pnp_d.get("reason") or rel_d.get("why")
                          or reloc_d.get("reason") or "未知")
            reg_fail_reasons[_final_why] = reg_fail_reasons.get(_final_why, 0) + 1
        logger.info(
            "[reg %d] rel(m=%d e=%d p=%d base=%s%s) pnp(cand=%d inl=%d%s) "
            "reloc(cand=%d m=%d inl=%d%s) -> %s",
            i, rel_d["m"], rel_d["e"], rel_d["p"], rel_d["base"],
            "" if rel_ok else ", " + (rel_d["why"] or "-"),
            pnp_d.get("npts", 0), pnp_d.get("inliers", 0),
            "" if pnp_ok else ", " + (pnp_d.get("reason") or "未执行"),
            reloc_d.get("cands", 0), reloc_d.get("best_matches", 0),
            reloc_d.get("best_inliers", 0),
            "" if reloc_ok else ", " + (reloc_d.get("reason") or "未执行"),
            "LOST:" + _final_why if not has_pose else "OK")
        if has_pose:
            reg_ok_count += 1
        if rel_ok:
            reg_rel_ok += 1
        if pnp_ok:
            reg_pnp_ok += 1
        if reloc_ok:
            reg_reloc_ok += 1
        if not has_pose or frame.pose is None:
            frame.status = FrameStatus.LOST
            if _reg_retry_on:
                # 不立刻出局：地图变大后同一帧可能就能对上，计数未到上限就放回池里。
                _n_try = _reg_trials.get(i, 0) + 1
                _reg_trials[i] = _n_try
                if _n_try < _reg_trials_max:
                    _order_pending.add(i)
                    _reg_requeued += 1
            continue
        last_kf = keyframes[-1]
        last_kf_pose = frames[last_kf].pose
        R_delta = frame.pose.R @ last_kf_pose.R.T
        angle = _rotation_angle_deg(R_delta)
        delta_c = frame.pose.center - last_kf_pose.center
        if len(map_points) >= 5:
            recent = np.array([p.xyz for p in map_points[-200:]], dtype=np.float32)
            c_r = recent.mean(axis=0)
            scene_ref = float(np.median(np.linalg.norm(recent - c_r, axis=1))) + 1e-6
        else:
            scene_ref = 1e-6
        trans_ratio = float(np.linalg.norm(delta_c) / scene_ref)
        covis_ratio = 1.0
        if prev_idx is not None:
            matches_for_covis = _match_ratio_absolute(frames[last_kf].desc, frame.desc, norm_type, match_dist)
            covis_ratio = _compute_covis_ratio(i, keyframes, matches_for_covis, frames)
        is_kf = (angle > KF_ANGLE_DEG or trans_ratio > KF_TRANS_RATIO or covis_ratio < KF_COVIS_RATIO)
        if abs(i - last_kf) < MIN_KF_FRAME_GAP:
            is_kf = False
        reloc_kf_for_loop = None
        if reloc_kf is not None:
            if abs(reloc_kf - i) >= cfg.loop_min_gap:
                reloc_kf_for_loop = reloc_kf
                is_kf = True
        if is_kf and has_pose and frame.pose is not None and len(map_points) > 20:
            _project_and_add_observations(i, keyframes, frames, map_points,
                                          focal0, fy0, cx, cy, n_views=25)
        if is_kf and len(map_points) > 20:
            _cull_keyframes(keyframes, frames, i, covis_graph)
            if i not in keyframes:
                keyframes.append(i)
                covis_graph.add_keyframe(i)
            frame.is_keyframe = True
            frame.status = FrameStatus.KEYFRAME
            kf_since_ba += 1
            kf_since_multiview += 1
            kf_since_global += 1
            kf_since_global_ba += 1
            if reloc_kf_for_loop is not None:
                loop_closures.append((reloc_kf_for_loop, i))
                if frames[reloc_kf_for_loop].pose is not None and frame.pose is not None:
                    T_meas = _relative_pose(frames[reloc_kf_for_loop].pose, frame.pose)
                    loop_constraints.append((reloc_kf_for_loop, i, T_meas))
            covis_graph.update_from_frame(i, map_points, frames)
            if len(keyframes) >= 3:
                _triangulate_next_best_view(i, keyframes, frames, map_points,
                                            focal0, fy0, cx, cy, triang_thresh, top_k=8)
            if len(keyframes) >= 3 and len(map_points) > 30:
                K_now = _build_K(focal0, fy0, cx, cy)
                _complete_tracks(keyframes[-10:], frames, map_points, K_now, reproj_thresh,
                                 max_transitivity=cfg.complete_max_transitivity)
            if enable_loop and len(keyframes) >= 15:
                if not loop_detector.trained:
                    descs = [frames[k].desc for k in keyframes]
                    loop_detector.train(descs, norm_type)
                    if loop_detector.trained:
                        logger.info("[loop] BoW 训练完成：关键帧=%d 词数=%d",
                                    len(keyframes), 0 if loop_detector.words is None
                                    else len(loop_detector.words))
                    else:
                        logger.warning("[loop] BoW 训练失败（描述子不足 %d 个），回环不会生效",
                                       len(descs))
                    for k in keyframes:
                        loop_detector.add(k, frames[k].desc)
                else:
                    loop_detector.add(i, frame.desc)
                loops = _detect_and_verify_loops(
                    i, keyframes, frames, loop_detector, map_points,
                    focal0, fy0, cx, cy, reproj_thresh,
                )
                for cand_kf, S_rel in loops:
                    loop_closures.append((cand_kf, i))
                    loop_constraints.append((cand_kf, i, CameraPose(S_rel.R.copy(), S_rel.t.copy())))
            if (enable_global_matching and len(keyframes) >= 5
                    and kf_since_global >= GLOBAL_MATCH_TRIGGER_EVERY_KF
                    and loop_detector.trained):
                if not gm_announced:
                    logger.info("[global] 全局匹配开始生效（BoW 已训练，%d 个关键帧）",
                                len(keyframes))
                    gm_announced = True
                _n_cand, _n_new = _global_match_expand(
                    i, keyframes, frames, loop_detector, map_points,
                    focal0, fy0, cx, cy, reproj_thresh)
                gm_calls += 1
                if _n_cand:
                    gm_hit_calls += 1
                gm_new_pts += _n_new
                kf_since_global = 0
            if kf_since_ba >= BA_LOCAL_TRIGGER_EVERY and len(keyframes) >= 3:
                window = covis_graph.get_neighbors(i, cfg.ba_local_covis_neighbors)
                if len(window) < 3:
                    # 非关键帧不在共视图里：按共享地图点找共视关键帧，
                    # 让 PnP 注册的普通帧也能进入局部 BA。
                    counts = {}
                    fm_i = frames[i].feat_map
                    if fm_i is not None:
                        for pid in {int(x) for x in fm_i if x >= 0}:
                            if pid >= len(map_points):
                                continue
                            for o in map_points[pid].obs:
                                if o.frame_idx == i:
                                    continue
                                fr_o = frames.get(o.frame_idx)
                                if fr_o is not None and fr_o.is_keyframe:
                                    counts[o.frame_idx] = counts.get(o.frame_idx, 0) + 1
                    window = [i] + [
                        k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])
                    ][:cfg.ba_local_covis_neighbors]
                if _local_win_deg > 0:
                    window = _filter_window_by_tri_angle(
                        window, frames, map_points, _local_win_deg)
                if len(window) >= 3:
                    focal0, fy0, cx, cy, k1, ok_any = _local_ba_refinement(
                        window, map_points, frames, focal0, fy0, cx, cy,
                        reproj_thresh, image_size, focal_init, covis_graph,
                        fix_focal=fix_focal, k1=k1, optimize_k1=optimize_k1,
                    )
                    if ok_any:
                        kf_since_ba = 0
                    else:
                        kf_since_ba = BA_LOCAL_TRIGGER_EVERY - 1
                else:
                    kf_since_ba = 0
            n_pts_now = len(map_points)
            n_reg_now = len(_registered_ba_frames(frames, map_points))
            trigger_global_ba = False
            if kf_since_global_ba >= GLOBAL_BA_EVERY_KF:
                trigger_global_ba = True
            if last_global_ba_n_kf > 0:
                if n_reg_now >= last_global_ba_n_kf * cfg.ba_global_image_growth:
                    trigger_global_ba = True
            if last_global_ba_n_pts > 0:
                if n_pts_now >= last_global_ba_n_pts * cfg.ba_global_point_growth:
                    trigger_global_ba = True
            # 保留 3 作为下限，避免只有初始对时就空跑一次全局 BA。
            if trigger_global_ba and n_reg_now >= 3:
                # 增量阶段用 1 次 EM + 硬过滤控制耗时；收尾的全局 BA 仍跑完整 EM。
                focal0, fy0, cx, cy, k1 = _global_refine(
                    map_points, frames, focal0, fy0, cx, cy,
                    reproj_thresh, image_size, focal_init, fix_focal,
                    norm_type, cfg.ba_local_covis_neighbors,
                    em_iters=1, max_iter=20,
                    focal_prior_sigma=max(1.0, focal0 * 0.02),
                    k1=k1, optimize_k1=optimize_k1)
                kf_since_global_ba = 0
                last_global_ba_n_kf = len(_registered_ba_frames(frames, map_points))
                last_global_ba_n_pts = len(map_points)
        if kf_since_multiview >= cfg.multiview_trigger_every_kf and len(map_points) > 50:
            K_now = _build_K(focal0, fy0, cx, cy)
            _n_retri = _retriangulate_multiview(map_points, frames, K_now, min_obs=MULTIVIEW_MIN_OBS)
            _merge_map_points(map_points, frames, norm_type, K=K_now, k1=k1,
                              validate=_merge_validate_on)
            kf_since_multiview = 0
            # 重三角化改写了 3D 点但位姿还是旧的，接一次局部 BA 把两边拉回一致
            if _n_retri >= 20 and len(keyframes) >= 3:
                _win = covis_graph.get_neighbors(i, cfg.ba_local_covis_neighbors)
                if _local_win_deg > 0:
                    _win = _filter_window_by_tri_angle(
                        _win, frames, map_points, _local_win_deg)
                if len(_win) >= 3:
                    focal0, fy0, cx, cy, k1, _ = _local_ba_refinement(
                        _win, map_points, frames, focal0, fy0, cx, cy,
                        reproj_thresh, image_size, focal_init, covis_graph,
                        fix_focal=fix_focal, k1=k1, optimize_k1=optimize_k1)
        # 按固定帧间隔检查；末轮收尾另有一次无条件 prune。
        if i > 0 and i % PRUNE_EVERY_FRAMES == 0 and len(map_points) > 100:
            prune_map_points(map_points, frames, reproj_thresh, focal0, fy0, cx, cy)
        # 深度状态检查（SFM_CHECK_STATE=2）：逐帧定位 feat_map / obs 漂移的来源
        if _check_state_deep():
            _check_state(frames, map_points, f"loop pass={_pass} frame={i}")

    if _reg_retry_on:
        logger.info("[reg] 主循环尝试 %d 次（重试 %d 次）：注册 %d；"
                    "rel %d / PnP %d / 重定位 %d；失败原因 %s",
                    reg_total, _reg_requeued, reg_ok_count,
                    reg_rel_ok, reg_pnp_ok, reg_reloc_ok,
                    reg_fail_reasons if reg_fail_reasons else "无")
    else:
        logger.info("[reg] 主循环处理 %d 帧：注册 %d；rel %d / PnP %d / 重定位 %d；失败原因 %s",
                    reg_total, reg_ok_count, reg_rel_ok, reg_pnp_ok, reg_reloc_ok,
                    reg_fail_reasons if reg_fail_reasons else "无")
    if _order_stats:
        _calls = _order_stats.get("calls", 0)
        _multi = _order_stats.get("multi", 0)
        logger.info("[order] 选帧 %d 次，多候选 %d 次（%.0f%%），平均 band %.2f，最大 %d",
                    _calls, _multi, 100.0 * _multi / max(1, _calls),
                    _order_stats.get("band_sum", 0) / max(1, _calls),
                    _order_stats.get("band_max", 0))
        if _order_dump:
            try:
                os.makedirs(_order_dump, exist_ok=True)
                payload = {
                    "pass": int(_pass),
                    "seed": int(_order_seed),
                    "band": int(_order_band),
                    "calls": int(_calls),
                    "multi": int(_multi),
                    "decisions": _order_stats.get("decisions", []),
                    "order": _order_stats.get("order", []),
                }
                _path = os.path.join(_order_dump, f"order_pass{int(_pass)}.json")
                with open(_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                logger.info("[order] dump -> %s", _path)
            except Exception as e:
                logger.warning("[order] dump 失败: %s", e)
    if _order_seq is not None:
        logger.info("[order] 复放结束：出队 %d / %d 项，跳过 %d",
                    len(_replay_order), len(_order_seq), _replay_skipped)
        if _order_dump:
            try:
                os.makedirs(_order_dump, exist_ok=True)
                payload = {
                    "pass": int(_pass),
                    "replay": True,
                    "source": _order_seq_path,
                    "requested": len(_order_seq),
                    "skipped": int(_replay_skipped),
                    "order": _replay_order,
                }
                _path = os.path.join(_order_dump, f"order_pass{int(_pass)}.json")
                with open(_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                logger.info("[order] dump -> %s", _path)
            except Exception as e:
                logger.warning("[order] dump 失败: %s", e)

    # 全局匹配开了却没生效时给出明确原因，避免日志沉默。
    if enable_global_matching and gm_announced:
        logger.info("[global] 共调用 %d 次，%d 次检索到候选，累计新增 %d 点",
                    gm_calls, gm_hit_calls, gm_new_pts)
    elif enable_global_matching:
        if not enable_loop:
            _gm_why = "回环已关闭，BoW 未训练"
        elif len(keyframes) < 15:
            _gm_why = "关键帧只有 %d 个（BoW 需 >= 15 才训练）" % len(keyframes)
        else:
            _gm_why = "BoW 训练失败或检索无候选"
        logger.warning("[global] 全局匹配已开启但全程未生效：%s", _gm_why)

    # ---- 第二轮：迭代式重定位丢失帧 ----
    # 顺序遍历会漏掉序列不连续但与其他关键帧重叠的图片；每轮把新注册帧
    # 提升为关键帧，让地图逐轮扩展。
    refined_after_reloc = False
    for _round in range(5):
        lost_ids = [i for i in range(len(frame_paths))
                    if frames[i].pose is None and frames[i].status != FrameStatus.INVALID]
        if not lost_ids or len(keyframes) < 2 or len(map_points) < 20:
            break
        logger.info("[reloc-pass %d] lost=%d keyframes=%d points=%d",
                    _round + 1, len(lost_ids), len(keyframes), len(map_points))
        added = 0
        added_ids = []
        _pass_fail = {}
        for i in lost_ids:
            frame = frames[i]
            if len(frame.kps) < cfg.min_features:
                continue
            _pnp_d = {}
            pose_pnp, _, quality = _try_pnp_guided(
                frame, keyframes, map_points, frames, covis_graph,
                focal0, fy0, cx, cy, reproj_thresh, loop_detector=loop_detector,
                diag=_pnp_d,
            )
            if pose_pnp is None:
                _why = _pnp_d.get("reason") or "未知"
                _pass_fail[_why] = _pass_fail.get(_why, 0) + 1
                logger.debug("[reloc-pass %d] frame %d 失败: %s (匹配点 %d)",
                             _round + 1, i, _why, _pnp_d.get("npts", 0))
                continue
            frame.pose = pose_pnp
            frame.track_quality = quality
            added += 1
            added_ids.append(i)
            if quality >= 0.15 and len(map_points) >= 20:
                frame.status = FrameStatus.KEYFRAME
                frame.is_keyframe = True
                if i not in keyframes:
                    keyframes.append(i)
                    covis_graph.add_keyframe(i)
                covis_graph.update_from_frame(i, map_points, frames)
            else:
                frame.status = FrameStatus.RELOCATED
        logger.info("[reloc-pass %d] registered %d  失败原因 %s",
                    _round + 1, added, _pass_fail if _pass_fail else "无")
        if added == 0 and _round > 0:
            break
        # 每轮重定位之后接一次全局 BA + 重三角化：新注册的帧要先拿到观测才能进 BA，
        # 精修后的位姿又能让下一轮 PnP 多匹配上几帧。
        focal0, fy0, cx, cy, k1 = _global_refine(
            map_points, frames, focal0, fy0, cx, cy,
            reproj_thresh, image_size, focal_init, fix_focal,
            norm_type, cfg.ba_local_covis_neighbors,
            complete_frames=added_ids, k1=k1, optimize_k1=optimize_k1)
        refined_after_reloc = True

    # 收尾：重定位循环里已经精修过就不再重复跑一次全局 BA。
    if not refined_after_reloc:
        focal0, fy0, cx, cy, k1 = _global_refine(
            map_points, frames, focal0, fy0, cx, cy,
            reproj_thresh, image_size, focal_init, fix_focal,
            norm_type, cfg.ba_local_covis_neighbors,
            k1=k1, optimize_k1=optimize_k1)

    valid_kf_set = set(keyframes)
    # 回环端点可能在 _cull_keyframes 里被剔掉，但位姿图必须保留它们，
    # 否则 valid_kf_set 的过滤会把回环边直接丢掉。
    _loop_frames = {f for a, b, _ in loop_constraints for f in (a, b)}
    _readd = [f for f in sorted(_loop_frames)
              if f not in valid_kf_set and frames[f].pose is not None]
    if _readd:
        keyframes.extend(_readd)
        valid_kf_set.update(_readd)
        logger.info("[PGO] 回补 %d 个被剔除的回环端点", len(_readd))
    sorted_kfs = sorted(keyframes)
    pgo_edges = []
    pgo_weights = []
    for a, b in zip(sorted_kfs[:-1], sorted_kfs[1:]):
        if frames[a].pose is None or frames[b].pose is None:
            continue
        T_rel = _relative_pose(frames[a].pose, frames[b].pose)
        pgo_edges.append((a, b, T_rel))
        pgo_weights.append(PGO_SEQ_WEIGHT)
    n_seq_edges = len(pgo_edges)
    if len(map_points) >= 5:
        recent_xyz = np.array([mp.xyz for mp in map_points[-200:]], dtype=np.float32)
        c_r = recent_xyz.mean(axis=0)
        scale_pgo = float(np.median(np.linalg.norm(recent_xyz - c_r, axis=1))) + 1e-6
    else:
        scale_pgo = 1.0
    valid_loop_pairs = []
    for a, b, T_meas in loop_constraints:
        if a not in valid_kf_set or b not in valid_kf_set:
            continue
        if abs(b - a) <= 1:
            continue
        if frames[a].pose is not None and frames[b].pose is not None:
            T_cur = _relative_pose(frames[a].pose, frames[b].pose)
            T_meas_mat = np.eye(4)
            T_meas_mat[:3, :3] = T_meas.R
            T_meas_mat[:3, 3] = T_meas.t.flatten()
            T_cur_mat = np.eye(4)
            T_cur_mat[:3, :3] = T_cur.R
            T_cur_mat[:3, 3] = T_cur.t.flatten()
            T_err = T_meas_mat @ np.linalg.inv(T_cur_mat)
            rv_err, _ = cv2.Rodrigues(T_err[:3, :3])
            trans_err = float(np.linalg.norm(T_err[:3, 3]))
            rot_err = float(np.linalg.norm(rv_err))
            if trans_err > PGO_OUTLIER_REL_POSE_THRESH * scale_pgo or rot_err > 0.5:
                continue
        pgo_edges.append((a, b, T_meas))
        n_inl_est = max(1, covis_graph.graph.get(b, {}).get(a, 10))
        w = float(np.log1p(n_inl_est))
        pgo_weights.append(max(PGO_LOOP_WEIGHT, w))
        valid_loop_pairs.append((a, b))
    loop_closures = valid_loop_pairs
    if enable_pgo and len(pgo_edges) >= 3:
        pre_pgo_poses = {k: frames[k].pose.copy() for k in keyframes if frames[k].pose is not None}
        poses_dict = {k: frames[k].pose for k in keyframes if frames[k].pose is not None}
        weights = {i: w for i, w in enumerate(pgo_weights)}
        if _pgo_edge_filter_on:
            new_poses, _pgo_kept = _pgo_optimize_filtered(
                keyframes, poses_dict, pgo_edges, weights, scale_pgo,
                n_seq=n_seq_edges)
            if new_poses is None:
                new_poses = poses_dict
        else:
            new_poses = pose_graph_optimize(keyframes, poses_dict, pgo_edges, edge_weights=weights)
        max_dc = 0.0
        for k, p in new_poses.items():
            old_p = pre_pgo_poses.get(k)
            if old_p is None or p is None:
                continue
            dc = float(np.linalg.norm(p.center - old_p.center))
            if dc > max_dc:
                max_dc = dc
        if max_dc <= PGO_MAX_DIVERGENCE_RATIO * scale_pgo:
            for k, p in new_poses.items():
                frames[k].pose = p
        else:
            logger.warning("[PGO] 位姿发散 max_dc=%.3f, 放弃 PGO", max_dc)
    # ---- 回环闭合 BA（实验开关，默认关；ITERS=0 时整段跳过）----
    # 只在最终一遍生效：自标定开启时 pass 1 会被 pass 2 覆盖，且 pass 1 的改动会
    # 经 k1 级联到 pass 2；无自标定或单遍运行时 pass 1 本身就是最终结果。
    _loop_ba_final_pass = (_pass >= 2) or (not optimize_k1)
    _loop_edges = []
    if _loop_ba_on and _loop_ba_iters > 0 and _loop_ba_final_pass:
        if _loop_ba_mode == "points":
            if valid_loop_pairs:
                try:
                    focal0, fy0, cx, cy, k1, _lb_stats = _loop_closure_ba(
                        valid_loop_pairs, loop_constraints, frames, map_points, keyframes,
                        covis_graph, focal0, fy0, cx, cy,
                        ransac_thresh, reproj_thresh, triang_thresh,
                        image_size, focal_init, norm_type, match_dist,
                        fix_focal=fix_focal, k1=k1, optimize_k1=optimize_k1,
                        max_points_per_loop=_loop_ba_max_loop,
                        max_points_total=_loop_ba_max_total,
                        ba_rounds=_loop_ba_iters,
                        detach_old=_loop_ba_detach)
                    logger.info(
                        "[loop-ba] 回环 %d/%d，匹配 %d，F 内点 %d，三角化 %d，加入点 %d，"
                        "窗口 BA=%s；丢弃 %s",
                        _lb_stats["loops"], len(valid_loop_pairs), _lb_stats["matched"],
                        _lb_stats["inliers"], _lb_stats["tri"], _lb_stats["added"],
                        _lb_stats["ba_ok"], _lb_stats["discard"] or "无")
                except Exception as _e:            # noqa: BLE001
                    logger.exception("[loop-ba] 异常，治疗臂结果不可用: %s", _e)
        else:
            if _loop_ba_mode != "edges":
                logger.warning("[loop-ba] 未知 mode=%s，按 edges 处理", _loop_ba_mode)
            _tmap = {}
            for _a, _b, _T in loop_constraints:
                _tmap.setdefault((_a, _b), _T)
            _seen_pairs = set()
            for _a, _b in valid_loop_pairs:
                if (_a, _b) in _seen_pairs:
                    continue
                _seen_pairs.add((_a, _b))
                _T_meas = _tmap.get((_a, _b))
                if _T_meas is None:
                    continue
                _loop_edges.append((_a, _b, _T_meas))
            logger.info("[loop-ba] 回环位姿边 %d/%d（权重 %.0f，尺度 %.3f）",
                        len(_loop_edges), len(valid_loop_pairs),
                        _loop_ba_weight, scale_pgo)
    # 先重三角化 + 合并，再做最终 BA：否则新点会用 BA 之前的旧位姿定型。
    K_pre = _build_K(focal0, fy0, cx, cy)
    _retriangulate_multiview(map_points, frames, K_pre, min_obs=MULTIVIEW_MIN_OBS)
    _merge_map_points(map_points, frames, norm_type, K=K_pre, k1=k1,
                      validate=_merge_validate_on)
    _check_state(frames, map_points, "post-final-merge")
    # ---- 两视图旋转约束：只给最终全局 BA 用 ----
    _rot_edges = []
    if rot_edge_weight > 0 and len(keyframes) >= 3:
        _rot_edges = _collect_rotation_edges(
            frames, keyframes, norm_type, match_dist,
            _build_K(focal0, fy0, cx, cy), ransac_thresh)
        logger.info("[rot-edge] 两视图旋转约束 %d 条（权重 %.0f）",
                    len(_rot_edges), rot_edge_weight)
        if _anomaly_on:
            _rot_edges = _filter_rotation_edges(_rot_edges, frames)
    if len(keyframes) >= 3 and len(map_points) > 50:
        all_registered = _registered_ba_frames(frames, map_points)
        window_tail2 = all_registered[-max(10, cfg.ba_local_covis_neighbors):]
        if len(window_tail2) >= 2:
            _em_reassign_local(window_tail2, map_points, frames, focal0, fy0, cx, cy)
        ba_state_g2 = {"success": False}
        focal0, fy0, cx, cy, k1 = bundle_adjustment(
            all_registered, map_points, frames, focal0, fy0, cx, cy,
            optimize_points=True, reproj_thresh=reproj_thresh,
            image_size=image_size, is_global=True,
            focal_init=focal_init, optimize_principal=False,
            fix_focal=fix_focal,
            _state=ba_state_g2, em_iters=BA_EM_ITERS,
            k1=k1, optimize_k1=optimize_k1, k1_prior_sigma=K1_PRIOR_SIGMA,
            rot_edges=_rot_edges, rot_weight=rot_edge_weight,
            loop_edges=_loop_edges, loop_weight=_loop_ba_weight,
            loop_t_scale=scale_pgo,
        )
        if ba_state_g2.get("success"):
            filter_observations(map_points, frames, focal0, fy0, cx, cy)
    # ---- 残余 k1 探测（只在第二遍做）----
    # 特征已按第一遍的 k1 去畸变，残余误差可由 BA 估出（k1_T ≈ k1_1 + k1_res）；
    # 只取 k1、不动模型：BA 前快照，之后还原。
    _k1_res = 0.0
    if (self_calib_iter and _pass == 2 and not self_calibrate
            and dist_coeffs is not None
            and len(frame_paths) >= SELF_CALIB_MIN_FRAMES):
        _reg_probe = _registered_ba_frames(frames, map_points)
        if len(_reg_probe) >= max(5, SELF_CALIB_MIN_KF) and len(map_points) > 50:
            _snap_pose = {k: frames[k].pose.copy() for k in _reg_probe
                          if frames[k].pose is not None}
            _snap_xyz = [p.xyz.copy() for p in map_points]
            _st_k1 = {"success": False}
            try:
                _, _, _, _, _k1_res = bundle_adjustment(
                    _reg_probe, map_points, frames, focal0, fy0, cx, cy,
                    optimize_points=True, reproj_thresh=reproj_thresh,
                    image_size=image_size, is_global=True,
                    focal_init=focal_init, optimize_principal=False,
                    fix_focal=False, _state=_st_k1, em_iters=1,
                    k1=0.0, optimize_k1=True,
                    k1_prior_sigma=K1_RESIDUAL_PRIOR_SIGMA)
            except Exception as _e:            # noqa: BLE001
                logger.warning("[self-calib] 残余 k1 探测失败: %s", _e)
                _k1_res = 0.0
            for k, _p in _snap_pose.items():
                frames[k].pose = _p
            for _pt, _x in zip(map_points, _snap_xyz):
                _pt.xyz = _x
            logger.info("[self-calib] 第二遍残余 k1=%+.5f（第一遍 %+.5f）",
                        _k1_res, float(np.asarray(dist_coeffs).ravel()[0]))

    # ---- Complete 只在最终一遍执行 ----
    # 第二遍会从特征开始完整重跑，第一遍的 prune / 位姿校验 / 输出装配都会被丢弃；
    # 判据与第二遍条件同构，改用直接统计有 pose 的帧，不依赖尚未装配的输出。
    if _stages_on and _pass == 1:
        _n_reg_raw = sum(1 for i in range(len(frame_paths))
                         if frames[i].pose is not None)
        if (optimize_k1 and abs(k1) >= SELF_CALIB_MIN_K1
                and _n_reg_raw >= SELF_CALIB_MIN_FRAMES
                and len(keyframes) >= SELF_CALIB_MIN_KF):
            logger.info("[stages] 第一遍 Create+Continue 完成（注册 %d），"
                        "跳过 Complete，直接进第二遍", _n_reg_raw)
            _RUNTIME_CFG = None
            K_est = _build_K(focal0, fy0, cx, cy)
            return estimate_poses(
                frame_paths, feature_type,
                focal_guess=None, aspect_ratio=aspect_ratio,
                dist_coeffs=np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64),
                camera_matrix=K_est,
                enable_loop=enable_loop, enable_pgo=enable_pgo,
                enable_global_matching=enable_global_matching,
                self_calibrate=False, self_calib_iter=self_calib_iter,
                rot_edge_weight=rot_edge_weight, pose_outlier_deg=pose_outlier_deg,
                _pass=2)

    # ---- 位姿离群检测：只标记、不删帧 ----
    # 中途删帧会改变 BA / 自标定的数值路径并级联出完全不同解，所以放在最后只改输出；
    # 随后的 prune 会顺带丢掉只靠这些帧支撑的点。
    if pose_outlier_deg > 0 and len(frame_paths) >= 10 and len(map_points) > 50:
        _drop_pose_outliers(
            frames, map_points, keyframes, focal0, fy0, cx, cy,
            norm_type, match_dist, ransac_thresh,
            max_deg=pose_outlier_deg, mark_only=True)
    # ---- 异常块：共视聚类摘除（内部开关）----
    # 与位姿校验互补：位姿校验管单帧 / 少帧，这里管彼此自洽但与主图断开的块；
    # 同样只改输出，不动地图。
    if _anomaly_on and len(frame_paths) >= 10 and len(map_points) > 50:
        _anomaly_frames = _detect_anomaly_blocks(
            frames, map_points, mark_only=True)
        if _anomaly_frames:
            _still = set(_anomaly_frames)
            keyframes[:] = [k for k in keyframes if k not in _still]

    _check_state(frames, map_points, "pre-finalize")
    prune_map_points(map_points, frames, reproj_thresh, focal0, fy0, cx, cy)
    # 点云误差统计用原始位姿：被校验摘除的帧 pose=None 直接跳过，
    # 不用「上一个有效位姿」顶替。
    poses_by_frame = {i: frames[i].pose for i in range(len(frame_paths))}
    all_poses = []
    for i in range(len(frame_paths)):
        st = frames[i].status
        if st in (FrameStatus.LOST, FrameStatus.INVALID):
            all_poses.append(None)
        elif frames[i].pose is None:
            all_poses.append(None)
        else:
            all_poses.append(frames[i].pose)
    xyz, mask = filter_point_cloud(map_points, poses_by_frame, focal0, fy0, cx, cy, reproj_thresh)
    if xyz.size == 0 or not np.any(mask):
        xyz_out = np.zeros((0, 3), dtype=np.float32)
        rgb_out = np.zeros((0, 3), dtype=np.uint8)
        obs_out = []
    else:
        kept = [map_points[i] for i in np.where(mask)[0]]
        xyz_out = np.array([p.xyz for p in kept], dtype=np.float32)
        rgb_out = _aggregate_colors(kept)
        obs_out = _aggregate_observations(kept)
    intrinsics = CameraIntrinsics(
        fx=focal0, fy=fy0, cx=cx, cy=cy,
        dist=(np.asarray(dist_coeffs).ravel() if dist_coeffs is not None else None),
    )
    logger.info("=== 结束: 耗时=%.1fs, 点数=%d, 关键帧=%d ===",
                time.time() - t_start, len(xyz_out), len(keyframes))
    if optimize_k1:
        _n_reg = sum(1 for p in all_poses if p is not None)
        logger.info("[self-calib] 第一遍 k1=%+.5f（fx=%.1f, 注册 %d/%d）",
                    k1, focal0, _n_reg, len(frame_paths))


    # ---- 自标定第二遍：把估计到的 k1 当作已知标定，去畸变后重跑 ----
    # 第一遍只有 BA 用畸变模型，PnP / 三角化 / 过滤仍是针孔；第二遍用去畸变特征
    # 走针孔路径（与 --use-gt-intrinsics 一致），统一各环节的相机模型。
    if (optimize_k1 and abs(k1) >= SELF_CALIB_MIN_K1
            and _n_reg >= SELF_CALIB_MIN_FRAMES
            and len(keyframes) >= SELF_CALIB_MIN_KF):
        logger.info("[self-calib] |k1|=%.4f 超过阈值 %.4f，用去畸变特征重跑第二遍",
                    abs(k1), SELF_CALIB_MIN_K1)
        _RUNTIME_CFG = None
        K_est = _build_K(focal0, fy0, cx, cy)
        return estimate_poses(
            frame_paths, feature_type,
            focal_guess=None, aspect_ratio=aspect_ratio,
            dist_coeffs=np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64),
            camera_matrix=K_est,
            enable_loop=enable_loop, enable_pgo=enable_pgo,
            enable_global_matching=enable_global_matching,
            self_calibrate=False, self_calib_iter=self_calib_iter,
            rot_edge_weight=rot_edge_weight, pose_outlier_deg=pose_outlier_deg,
            _pass=2)
    # ---- 自标定第三遍：残差 k1 显著时，用合成后的 k1 重跑 ----
    if (self_calib_iter and _pass == 2 and abs(_k1_res) >= SELF_CALIB_RESIDUAL_MIN
            and not self_calibrate and dist_coeffs is not None):
        _k1_total = float(np.asarray(dist_coeffs).ravel()[0]) + _k1_res
        logger.info("[self-calib] 残余 |k1|=%.5f 超过 %.4f，合成 k1=%.5f 跑第三遍",
                    abs(_k1_res), SELF_CALIB_RESIDUAL_MIN, _k1_total)
        _RUNTIME_CFG = None
        K_est3 = _build_K(focal0, fy0, cx, cy)
        return estimate_poses(
            frame_paths, feature_type,
            focal_guess=None, aspect_ratio=aspect_ratio,
            dist_coeffs=np.array([_k1_total, 0.0, 0.0, 0.0], dtype=np.float64),
            camera_matrix=K_est3,
            enable_loop=enable_loop, enable_pgo=enable_pgo,
            enable_global_matching=enable_global_matching,
            self_calibrate=False, self_calib_iter=self_calib_iter,
            rot_edge_weight=rot_edge_weight, pose_outlier_deg=pose_outlier_deg,
            _pass=3)

    if _PAIR_CACHE_ON:
        _h, _m = _PAIR_CACHE_STATS["hit"], _PAIR_CACHE_STATS["miss"]
        logger.info("[pair-cache] hit=%d miss=%d 命中率=%.1f%%（含后续遍）",
                    _h, _m, 100.0 * _h / max(1, _h + _m))
    _RUNTIME_CFG = None
    return SfMResult(
        intrinsics=intrinsics,
        poses=all_poses,
        xyz=xyz_out,
        rgb=rgb_out,
        obs_per_point=obs_out,
        keyframes=list(keyframes),
        feature_type=feature_type,
        loop_closures=list(loop_closures),
        frame_status=[frames[i].status for i in range(len(frame_paths))],
    )