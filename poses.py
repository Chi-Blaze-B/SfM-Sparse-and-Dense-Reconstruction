"""
增量式运动恢复结构：相机位姿估算与稀疏重建
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from frame_cache import get_image

from scipy.optimize import least_squares
from scipy.special import expit
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree

from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


# =========================================================================
# 运行时自适应配置
# =========================================================================
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
    em_iters: int
    em_pi_in_init: float
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


def derive_auto_config(n_frames, img_shape, feature_type,
                       has_intrinsics, has_distortion) -> AutoConfig:
    h, w = img_shape
    image_size = float(max(h, w))
    n_pixels = float(h * w)

    base_feat = int(np.clip(np.sqrt(n_pixels) * 1.5, 2000, 12000))
    ft = feature_type.lower()
    feature_count = int(base_feat * 1.5) if ft == "orb" else base_feat
    min_features = max(80, int(feature_count * 0.01))

    reproj_thresh = float(np.clip(image_size * 0.0010, 0.5, 2.0))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh
    guided_radius_px = float(np.clip(image_size * 0.008, 5.0, 20.0))

    if n_frames <= 20:
        init_search_ratio = 1.0
        init_candidate_window = max(1, n_frames - 1)
    elif n_frames <= 50:
        init_search_ratio = 0.30
        init_candidate_window = 15
    elif n_frames <= 200:
        init_search_ratio = 0.06
        init_candidate_window = 15
    else:
        init_search_ratio = 0.03
        init_candidate_window = 15
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

    ba_local_covis_neighbors = int(np.clip(
        12 - np.log10(max(n_frames, 10)) * 2, 6, 12))
    ba_min_obs = int(np.clip(n_frames * 2, 50, 150))
    ba_local_iters = 25
    ba_local_refinements = 2
    ba_global_iters = int(np.clip(40 + np.log2(max(n_frames, 2)) * 3, 40, 80))
    ba_global_image_growth = 1.10
    ba_global_point_growth = 1.10

    em_iters = int(np.clip(4 + int(np.log2(max(n_frames, 2))), 4, 8))
    em_pi_in_init = 0.7

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
    filter_min_tri_angle_deg = 1.5

    return AutoConfig(
        image_size=image_size, feature_count=feature_count,
        min_features=min_features, reproj_thresh=reproj_thresh,
        ransac_thresh=ransac_thresh, triang_thresh=triang_thresh,
        guided_radius_px=guided_radius_px,
        init_search_ratio=init_search_ratio,
        init_candidate_window=init_candidate_window,
        init_max_pairs=init_max_pairs, init_min_inliers=init_min_inliers,
        init_min_parallax_deg=init_min_parallax_deg,
        pnp_window=pnp_window, pnp_min_inliers=pnp_min_inliers,
        pnp_ransac_iters=pnp_ransac_iters,
        multiview_trigger_every_kf=multiview_trigger_every_kf,
        ba_local_covis_neighbors=ba_local_covis_neighbors,
        ba_min_obs=ba_min_obs, ba_local_iters=ba_local_iters,
        ba_local_refinements=ba_local_refinements,
        ba_global_iters=ba_global_iters,
        ba_global_image_growth=ba_global_image_growth,
        ba_global_point_growth=ba_global_point_growth,
        em_iters=em_iters,
        em_pi_in_init=em_pi_in_init, loop_vocab_size=loop_vocab_size,
        loop_min_inliers=loop_min_inliers,
        loop_max_candidates=loop_max_candidates,
        loop_min_gap=loop_min_gap,
        reloc_min_matches=reloc_min_matches,
        reloc_min_inliers=reloc_min_inliers,
        reloc_max_candidates=reloc_max_candidates, pgo_iters=pgo_iters,
        complete_max_transitivity=complete_max_transitivity,
        continue_max_angle_deg=continue_max_angle_deg,
        filter_min_tri_angle_deg=filter_min_tri_angle_deg,
    )


_RUNTIME_CFG: Optional[AutoConfig] = None


def _cfg() -> AutoConfig:
    if _RUNTIME_CFG is None:
        return derive_auto_config(100, (1080, 1920), "sift", False, False)
    return _RUNTIME_CFG


# =========================================================================
# 通用常量
# =========================================================================
EPS = 1e-8


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
MATCH_DIST_SIFT = 1.5
MATCH_DIST_UPDATE_ORB = 35
MATCH_DIST_UPDATE_SIFT = 150.0
MATCH_DIST_UPDATE_ROOTSIFT = 0.5

GUIDED_PROJ_RADIUS_RATIO = 0.008

KF_ANGLE_DEG = 5.0
KF_TRANS_RATIO = 0.15
KF_COVIS_RATIO = 0.60
KF_CULL_WINDOW = 10
KF_CULL_COVIS = 0.80
MIN_KF_FRAME_GAP = 2

MIN_TRI_ANGLE_DEG = 2.0
MAX_REPROJ_ERROR = 4.0
MIN_OBSERVATIONS = 2
MULTIVIEW_MIN_OBS = 3
MULTIVIEW_RANSAC_ITERS = 80
MULTIVIEW_RANSAC_THRESH = 3.0
MULTIVIEW_MAX_POINTS_PER_CALL = 8000
SINGLE_TRI_RANSAC_ITERS = 30
SINGLE_TRI_RANSAC_THRESH = 4.0

MULTIVIEW_ACCEPT_IMPROVE = 0.90
ADD_OBS_ERR_SCALE = 3.0

BASELINE_MIN_POS_RATIO = 0.5
BASELINE_MIN_ANGLE_DEG = 2.0
INIT_MIN_PARALLAX_DEG = 8.0

BA_MAX_POINTS = 8000
BA_MAX_OBS = 20000
BA_LOCAL_TRIGGER_EVERY = 1
BA_MAX_INIT_RMS_RATIO = 3.0
BA_OVERFIT_MIN_RMS_LOCAL = 0.02
BA_OVERFIT_MIN_RMS_GLOBAL = 0.005
BA_OVERFIT_MIN_BEFORE = 0.15
BA_MAX_RMS_INCREASE = 1.2
FOCAL_MAX_STEP_RATIO = 0.10
FOCAL_GLOBAL_DRIFT_MAX = 0.05
PRINCIPAL_PT_MAX_STEP = 0.02
PRINCIPAL_PT_FROZEN_EPS = 1e-6

EM_SIGMA_OUT_RATIO_INIT = 4.0
EM_SIGMA_OUT_RATIO = 6.0
EM_GAMMA_FLOOR = 1e-4

EPS_MIN = 1e-8
EPS_MAX = 0.1

FILTER_MEDIAN_FACTOR = 3.0
FILTER_REPROJ_FACTOR = 2.0
SOR_K_NEIGHBORS = 16
SOR_STD_RATIO = 2.0

MERGE_MAX_DESC_DIST_ORB = 20
MERGE_MAX_DESC_DIST_SIFT = 100.0
MERGE_MAX_DESC_DIST_ROOTSIFT = 0.5
MERGE_MAX_XYZ_DIST_FACTOR = 0.02

LOOP_MIN_SCORE = 0.15
LOOP_RELOC_SCORE = 0.05
LOOP_SKIP_RECENT = 15
LOOP_MIN_GAP = 8
LOOP_MIN_SEQ_CONSISTENCY = 2
LOOP_CANDIDATE_NMS_GAP = 5
LOOP_SEQ_CHECK_K = 2
LOOP_SEQ_CHECK_THRESH = 0.3

LOOP_MIN_INLIER_RATIO = 0.20

PGO_CAUCHY_DELTA = 0.15
PGO_SEQ_WEIGHT = 1.0
PGO_LOOP_WEIGHT = 1.5
PGO_COVIS_WEIGHT = 1.5
PGO_MAX_DIVERGENCE_RATIO = 0.3
PGO_OUTLIER_REL_POSE_THRESH = 0.5

GLOBAL_MATCH_TOP_K = 10
GLOBAL_MATCH_SKIP_RECENT = 5
GLOBAL_MATCH_MIN_INLIERS = 12
GLOBAL_MATCH_TRIGGER_EVERY_KF = 1
GLOBAL_BA_EVERY_KF = 20
PROGRESS_LOG_EVERY = 10

MAX_OBS_PER_POINT = 10
MERGE_TRACK_OVERLAP_RATIO = 0.5
CONTINUE_MAX_DEPTH = 10.0

_warned_keys: Set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(message)


# =========================================================================
# 数据结构
# =========================================================================
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[np.ndarray] = None

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.s * (X @ self.R.T) + self.t

    def inverse(self) -> "Sim3":
        R_inv = self.R.T
        s_inv = 1.0 / max(self.s, 1e-12)
        t_inv = -s_inv * (R_inv @ self.t)
        return Sim3(s_inv, R_inv, t_inv)


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
    bow_hist: Optional[np.ndarray] = None
    track_quality: float = 0.0
    is_keyframe: bool = False

    def __post_init__(self):
        if self.feat_map is None:
            self.feat_map = np.full(len(self.kps), -1, dtype=np.int64)


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

    def __iter__(self):
        yield self.intrinsics
        yield self.poses
        yield self.xyz

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (self.intrinsics, self.poses, self.xyz)[idx]


# =========================================================================
# 共视图
# =========================================================================
class CovisibilityGraph:
    def __init__(self):
        self.graph: Dict[int, Dict[int, int]] = {}

    def add_keyframe(self, kf_idx: int) -> None:
        self.graph.setdefault(kf_idx, {})

    def remove_keyframe(self, kf_idx: int) -> None:
        self.graph.pop(kf_idx, None)
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
            if other not in frames or not frames[other].is_keyframe:
                continue
            if other not in self.graph:
                self.graph[other] = {}
            self.graph[kf_idx][other] = c
            self.graph[other][kf_idx] = c

    def get_neighbors(self, seed_kf: int, n: int = 20) -> List[int]:
        if seed_kf not in self.graph:
            return [seed_kf]
        items = sorted(self.graph[seed_kf].items(), key=lambda kv: -kv[1])
        out = [seed_kf]
        for kf, _ in items[:n]:
            out.append(kf)
        return out


# =========================================================================
# 内参 / 颜色 / 畸变
# =========================================================================
def _build_K(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _sample_colors(bgr, kps):
    if not kps:
        return np.zeros((0, 3), dtype=np.uint8)
    h, w = bgr.shape[:2]
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    return bgr[ys, xs][:, ::-1].copy()


def _undistort_kps(kps, K, dist):
    if dist is None or not kps:
        return kps
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
    try:
        undist = cv2.undistortPoints(
            pts, K.astype(np.float64),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1),
            P=K.astype(np.float64)).reshape(-1, 2)
    except cv2.error as e:
        _warn_once("undistort_fail", f"undistortPoints 失败（{e}）")
        return kps
    return [cv2.KeyPoint(float(x), float(y), kp.size, kp.angle,
                         kp.response, kp.octave, kp.class_id)
            for kp, (x, y) in zip(kps, undist)]


# =========================================================================
# Sim(3)
# =========================================================================
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


# =========================================================================
# 特征提取（多线程）
# =========================================================================
_TLS = threading.local()


def _make_detector(feature_type, feature_count):
    ft = feature_type.lower()
    if ft in ("sift", "rootsift"):
        try:
            det = cv2.SIFT_create(nfeatures=feature_count,
                                  contrastThreshold=0.01,
                                  edgeThreshold=10, sigma=1.6)
            return det, ft
        except cv2.error as e:
            logger.warning(f"SIFT 不可用（{e}），回退到 ORB。")
            ft = "orb"
    det = cv2.ORB_create(nfeatures=int(feature_count * 1.5),
                         scaleFactor=1.2, nlevels=8,
                         edgeThreshold=31, patchSize=31)
    return det, ft


def _apply_rootsift(desc):
    if desc is None or len(desc) == 0:
        return desc
    d = desc.astype(np.float32)
    d /= (np.sum(np.abs(d), axis=1, keepdims=True) + 1e-8)
    np.sqrt(d, out=d)
    return d


def _get_detector(feature_type, feature_count):
    """线程局部检测器（cv2.SIFT 非线程安全）。"""
    key = (feature_type, feature_count)
    cache = getattr(_TLS, "detectors", None)
    if cache is None:
        cache = {}
        _TLS.detectors = cache
    if key not in cache:
        det, _ = _make_detector(feature_type, feature_count)
        cache[key] = det
    return cache[key]


def extract_features(paths, feature_type, dist_coeffs=None,
                     camera_matrix=None, max_workers=4):
    cfg = _cfg()
    _, feature_type = _make_detector(feature_type, cfg.feature_count)
    is_sift = feature_type in ("sift", "rootsift")
    empty_desc = (np.zeros((0, 128), dtype=np.float32) if is_sift
                  else np.zeros((0, 32), dtype=np.uint8))
    K_und = None
    if camera_matrix is not None and dist_coeffs is not None:
        K_und = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    def _extract_one(args):
        i, p = args
        detector = _get_detector(feature_type, cfg.feature_count)
        img = get_image(p)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{p}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        kp = kp if kp is not None else []
        if K_und is not None and dist_coeffs is not None:
            kp = _undistort_kps(kp, K_und, dist_coeffs)
        colors = _sample_colors(img, kp)
        if desc is None:
            desc = empty_desc
        if feature_type == "rootsift":
            desc = _apply_rootsift(desc)
        shape = img.shape[:2]
        del img, gray
        return i, p, kp, desc, colors, shape

    n = len(paths)
    if n == 0:
        return [], None, feature_type
    workers = min(max_workers, n, 8)
    if workers <= 1:
        results = [_extract_one((i, p)) for i, p in enumerate(paths)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_extract_one, enumerate(paths)))

    frames = []
    img_shape = results[0][5] if results else None
    for i, p, kp, desc, colors, shape in results:
        frames.append(Frame(idx=i, path=p, kps=kp, desc=desc, colors=colors))
    return frames, img_shape, feature_type


# =========================================================================
# 匹配
# =========================================================================
_MATCHER_CACHE: Dict[int, cv2.BFMatcher] = {}


def _get_bf(norm_type):
    bf = _MATCHER_CACHE.get(norm_type)
    if bf is None:
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        _MATCHER_CACHE[norm_type] = bf
    return bf


def get_metric(desc):
    if desc is None or len(desc) == 0:
        return cv2.NORM_HAMMING, MATCH_DIST_ORB, MATCH_DIST_UPDATE_ORB
    if desc.dtype != np.uint8:
        return cv2.NORM_L2, MATCH_DIST_SIFT, MATCH_DIST_UPDATE_SIFT
    return cv2.NORM_HAMMING, MATCH_DIST_ORB, MATCH_DIST_UPDATE_ORB


def _is_rootsift_like(desc):
    if desc is None or len(desc) == 0:
        return False
    if desc.dtype == np.uint8:
        return False
    try:
        nrm = float(np.linalg.norm(desc[0]))
    except Exception:
        return False
    return abs(nrm - 1.0) < 0.1


def _ratio_for(desc, default=MATCH_RATIO):
    if _is_rootsift_like(desc):
        return MATCH_RATIO_ROOTSIFT
    return default


def _update_dist_for(desc):
    if desc is None or len(desc) == 0:
        return MATCH_DIST_UPDATE_ORB
    if desc.dtype != np.uint8:
        if _is_rootsift_like(desc):
            return MATCH_DIST_UPDATE_ROOTSIFT
        return MATCH_DIST_UPDATE_SIFT
    return MATCH_DIST_UPDATE_ORB


def match_descriptors(desc1, desc2, norm_type, max_dist, ratio=None):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if ratio is None:
        ratio = _ratio_for(desc1)
    if len(desc2) < 2:
        bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)
        return [m for m in bf_cross.match(desc1, desc2)
                if m.distance < max_dist]
    bf = _get_bf(norm_type)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and m.distance < max_dist:
            good.append(m)
    return good


def _match_ratio_absolute(desc1, desc2, norm_type, max_dist):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return match_descriptors(desc1, desc2, norm_type, max_dist)
    bf = _get_bf(norm_type)
    ratio = _ratio_for(desc1)
    m12 = bf.knnMatch(desc1, desc2, k=2)
    m21 = bf.knnMatch(desc2, desc1, k=2)
    keep12 = {}
    for pair in m12:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and m.distance < max_dist:
            keep12[m.queryIdx] = m.trainIdx
    keep21 = {}
    for pair in m21:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and m.distance < max_dist:
            keep21[m.queryIdx] = m.trainIdx
    out = []
    for q, t in keep12.items():
        if keep21.get(t) == q:
            out.append(cv2.DMatch(q, t, 0.0))
    return out


def update_dist_for_desc(desc):
    return _update_dist_for(desc)


def _guided_match_3d2d(frame, kf_indices, map_points, frames, K,
                       proj_radius_px, max_desc_dist=None):
    if frame.pose is None or frame.desc is None or len(frame.kps) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32), [])
    norm_type, md, _ = get_metric(frame.desc)
    if max_desc_dist is None:
        max_desc_dist = md
    R = frame.pose.R
    t = frame.pose.t.flatten()
    K64 = np.asarray(K, dtype=np.float64)
    kps_uv = np.array([kp.pt for kp in frame.kps], dtype=np.float32)
    if len(kps_uv) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32), [])
    try:
        tree = cKDTree(kps_uv)
    except Exception:
        tree = None

    out3d, out2d = [], []
    out_idx = []
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
                cand = np.where(dx * dx + dy * dy <=
                                proj_radius_px ** 2)[0].tolist()
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
                    dval = int(cv2.norm(frame.desc[ci], desc_pt,
                                        cv2.NORM_HAMMING))
                else:
                    dval = float(cv2.norm(frame.desc[ci], desc_pt,
                                          cv2.NORM_L2))
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
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32), [])
    return (np.asarray(out3d, dtype=np.float32),
            np.asarray(out2d, dtype=np.float32),
            out_idx)


# =========================================================================
# 几何验证
# =========================================================================
def _verify_matches_fh(pts1, pts2, thresh, min_inliers):
    if len(pts1) < max(8, min_inliers):
        return None, None, None
    pts1c = pts1.astype(np.float64).reshape(-1, 1, 2)
    pts2c = pts2.astype(np.float64).reshape(-1, 1, 2)
    best_mask, best_count = None, 0
    best_model, best_type = None, None

    H, mask_h = None, None
    try:
        H, mask_h = cv2.findHomography(pts1c, pts2c, cv2.RANSAC,
                                       thresh, maxIters=2000,
                                       confidence=0.999)
    except cv2.error:
        pass
    n_h = int(mask_h.sum()) if mask_h is not None else 0

    F, mask_f = None, None
    try:
        F, mask_f = cv2.findFundamentalMat(pts1c, pts2c, cv2.FM_RANSAC,
                                           thresh, 0.999, maxIters=2000)
    except cv2.error:
        pass
    n_f = int(mask_f.sum()) if mask_f is not None else 0

    if H is not None and mask_h is not None and n_h >= min_inliers:
        best_mask = mask_h.ravel().astype(bool)
        best_count, best_model, best_type = n_h, H, "H"
    if F is not None and mask_f is not None and n_f >= min_inliers:
        if n_f > n_h or best_type is None:
            mask = mask_f.ravel().astype(bool)
            if int(mask.sum()) > best_count:
                best_mask = mask
                best_count = int(mask.sum())
                best_model, best_type = F, "F"
    if best_mask is None or best_count < min_inliers:
        return None, None, None
    return best_mask, best_type, best_model


def _find_essential(pts1, pts2, K, threshold):
    K32 = K.astype(np.float32)
    try:
        return cv2.findEssentialMat(pts1, pts2, cameraMatrix=K32,
                                    method=cv2.RANSAC, prob=0.999,
                                    threshold=threshold)
    except (cv2.error, TypeError):
        pass
    fx = float(K[0, 0]); cx = float(K[0, 2]); cy = float(K[1, 2])
    return cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy),
                                method=cv2.RANSAC, prob=0.999,
                                threshold=threshold)


def _recover_pose(E, pts1, pts2, K, mask=None):
    K32 = K.astype(np.float32)
    try:
        retval, R, t, mask_out = cv2.recoverPose(
            E, pts1, pts2, cameraMatrix=K32)
        return retval, R, t, mask_out
    except (cv2.error, TypeError):
        pass
    fx = float(K[0, 0]); cx = float(K[0, 2]); cy = float(K[1, 2])
    retval, R, t, mask_out = cv2.recoverPose(
        E, pts1, pts2, focal=fx, pp=(cx, cy))
    return retval, R, t, mask_out


def _check_baseline_sufficient(pts_prev, pts_curr, R_rel, t_rel, K,
                               inlier_mask,
                               min_pos_ratio=BASELINE_MIN_POS_RATIO,
                               min_angle_deg=BASELINE_MIN_ANGLE_DEG):
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
    if int(pos.sum()) < 5 or float(pos.mean()) < min_pos_ratio:
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


# =========================================================================
# 批量三角化辅助函数（COLMAP 风格三重过滤）
# =========================================================================
def _triangulate_batch(uvs, projs, cams, min_angle_deg, max_reproj):
    """多点 DLT + 重投影/三角化角/视差三重过滤。

    uvs:   [N, 2] 每视图的像素坐标
    projs: [N, 3, 4] 投影矩阵
    cams:  [N, 3] 相机中心
    返回 (X: [N, 3], valid: [N] bool)
    """
    if len(projs) < 2:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)
    pts4d = cv2.triangulatePoints(projs[0], projs[1],
                                   uvs[0].T.astype(np.float64),
                                   uvs[1].T.astype(np.float64))
    w = pts4d[3]
    X = (pts4d[:3] / np.where(np.abs(w) > 1e-12, w, 1e-12)).T
    N = len(X)
    if N == 0:
        return X, np.zeros(0, dtype=bool)
    Xh = np.hstack([X, np.ones((N, 1))])
    errs = np.zeros((N, len(projs)))
    for k, (P, uv) in enumerate(zip(projs, uvs)):
        pj = Xh @ P.T
        z = pj[:, 2:3]
        pj = pj[:, :2] / np.where(np.abs(z) > 1e-12, z, 1e-12)
        errs[:, k] = np.linalg.norm(pj - uv, axis=1)
    rays = X[None, :, :] - cams[:, None, :]
    norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / np.maximum(norms, 1e-12)
    cos_mat = np.einsum('mni,mni->mn', rays, rays)
    tri_angle = np.full(N, 0.0)
    for i in range(len(projs)):
        for j in range(i + 1, len(projs)):
            c = np.clip(cos_mat[i, j], -1.0, 1.0)
            a = np.degrees(np.arccos(c))
            tri_angle = np.maximum(tri_angle, a)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(errs).all(axis=1)
    valid = (finite
             & (errs.max(axis=1) < max_reproj)
             & (tri_angle > min_angle_deg))
    return X, valid


def _check_obs_angle(pt, new_frame_idx, frames, min_angle_deg):
    """新观测加入后，最小三角化角是否满足阈值。"""
    if new_frame_idx not in frames or frames[new_frame_idx].pose is None:
        return False
    cam_centers = [frames[new_frame_idx].pose.center]
    for o in pt.obs:
        f = frames.get(o.frame_idx)
        if f is not None and f.pose is not None:
            cam_centers.append(f.pose.center)
    if len(cam_centers) < 2:
        return True
    ang = _compute_min_tri_angle(
        np.asarray(pt.xyz, dtype=np.float64),
        np.asarray(cam_centers, dtype=np.float64))
    return ang >= min_angle_deg


# =========================================================================
# 回环检测（BoW + 反向索引）
# =========================================================================
class LoopDetector:
    _POPCOUNT_LUT = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1).astype(np.uint8)
    _QUANTIZE_CHUNK = 512

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.words = None
        self.idf = None
        self.hists = {}
        self.inv_idx = {}   # word -> [(kf, weight)]
        self.trained = False
        self._norm_type = cv2.NORM_HAMMING

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
            sel = rng.choice(len(all_desc), size=cap, replace=False)
            train_desc = all_desc[sel]
        else:
            train_desc = all_desc
        t0 = time.time()
        is_hamming = (norm_type == cv2.NORM_HAMMING)
        data = train_desc.astype(np.float32)
        k = min(self.vocab_size, len(data))
        logger.info(f"[loop] 词袋聚类：{len(data)} × {k} 词")
        try:
            mbk = MiniBatchKMeans(n_clusters=k, batch_size=4096,
                                  max_iter=100, n_init=3,
                                  random_state=0, reassignment_ratio=0.02)
            mbk.fit(data)
            centroids = mbk.cluster_centers_
        except Exception as e:
            logger.warning(f"[loop] 词袋聚类失败：{e}")
            return
        if is_hamming:
            self.words = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
        else:
            self.words = centroids.astype(np.float32)
        df = np.zeros(len(self.words), dtype=np.float64)
        for d in pool:
            assign = self._quantize_assign(d, norm_type)
            words_present = np.unique(assign)
            df[words_present] += 1.0
        n_docs = max(len(pool), 1)
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        self._norm_type = norm_type
        self.trained = True
        logger.info(f"[loop] 词袋训练完成（{time.time() - t0:.1f}s）")

    def _hamming_argmin(self, d_uint8, words_uint8):
        N = d_uint8.shape[0]
        assign = np.empty(N, dtype=np.int64)
        for start in range(0, N, self._QUANTIZE_CHUNK):
            end = min(start + self._QUANTIZE_CHUNK, N)
            xor = np.bitwise_xor(
                d_uint8[start:end, None, :], words_uint8[None, :, :])
            dist = self._POPCOUNT_LUT[xor].sum(axis=2)
            assign[start:end] = np.argmin(dist, axis=1)
        return assign

    def _quantize_assign(self, desc, norm_type):
        if self.words is None or len(desc) == 0:
            return np.zeros(0, dtype=np.int64)
        if norm_type == cv2.NORM_HAMMING:
            return self._hamming_argmin(desc.astype(np.uint8),
                                        self.words.astype(np.uint8))
        d = desc.astype(np.float32)
        w = self.words.astype(np.float32)
        dist = np.linalg.norm(d[:, None, :] - w[None, :, :], axis=2)
        return np.argmin(dist, axis=1)

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

    def query(self, idx, desc, keyframes,
              skip_recent=LOOP_SKIP_RECENT,
              min_score=LOOP_MIN_SCORE,
              max_candidates=None):
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
        ranked = []
        for kf, s in scores.items():
            if abs(kf - idx) < skip_recent:
                continue
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
            if all(abs(kf - k) > LOOP_CANDIDATE_NMS_GAP for k, _ in kept):
                kept.append((kf, s))
            if len(kept) >= max_candidates:
                break
        return kept

    def retrieve_topk(self, idx, desc, k=20, min_score=0.15):
        """供 guided matching / 大场景匹配使用的检索接口。"""
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


# =========================================================================
# SE(3) 与 PGO
# =========================================================================
def _se3_log(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            f"_se3_log 期望 4×4 变换矩阵，收到 shape={T.shape}，"
            f"type={type(T).__name__}")
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    phi = rvec.flatten()
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        return np.concatenate([t, phi])
    phi_hat = cv2.Rodrigues(phi)[0]
    A = np.sin(theta) / theta
    if theta > np.pi - 1e-3:
        C = 0.5 / max(theta * theta, 1e-12)
    else:
        C = (1.0 - A) / max(theta * theta, 1e-12)
    V_inv = (np.eye(3) - 0.5 * phi_hat + C * (phi_hat @ phi_hat))
    rho = V_inv @ t
    return np.concatenate([rho, phi])


def _pose_to_T(pose: CameraPose) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.R
    T[:3, 3] = pose.t.flatten()
    return T


def _T_to_pose(T: np.ndarray) -> CameraPose:
    return CameraPose(T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy())


def pose_graph_optimize(keyframes, poses, edges, iters=None,
                        cauchy_delta=PGO_CAUCHY_DELTA, edge_weights=None):
    if iters is None:
        iters = _cfg().pgo_iters
    if not keyframes or len(edges) < 2:
        return poses
    kf_to_pos = {kf: i for i, kf in enumerate(keyframes)}
    n = len(keyframes)
    if n < 2:
        return poses
    param = np.zeros(6 * n, dtype=np.float64)
    for kf, i in kf_to_pos.items():
        if kf not in poses or poses[kf] is None:
            continue
        rv, _ = cv2.Rodrigues(poses[kf].R)
        param[i * 6:i * 6 + 3] = rv.flatten()
        param[i * 6 + 3:i * 6 + 6] = poses[kf].t.flatten()
    valid_edges = []
    for idx_e, (a, b, T) in enumerate(edges):
        if a in kf_to_pos and b in kf_to_pos:
            w = 1.0 if edge_weights is None else float(
                edge_weights.get(idx_e, 1.0))
            valid_edges.append((a, b, T, w))
    if len(valid_edges) < 2:
        return poses
    edge_i = np.array([kf_to_pos[e[0]] for e in valid_edges], dtype=np.int64)
    edge_j = np.array([kf_to_pos[e[1]] for e in valid_edges], dtype=np.int64)
    T_meas = np.stack([_pose_to_T(e[2]) for e in valid_edges])
    edge_w = np.array([e[3] for e in valid_edges], dtype=np.float64)
    sqrt_w = np.sqrt(edge_w)
    fixed_pos = 0

    def _params_to_T(params):
        T = np.empty((n, 4, 4), dtype=np.float64)
        for i in range(n):
            rv = params[i * 6:i * 6 + 3]
            R, _ = cv2.Rodrigues(rv)
            T[i, :3, :3] = R
            T[i, :3, 3] = params[i * 6 + 3:i * 6 + 6]
            T[i, 3, :] = [0, 0, 0, 1]
        return T

    def _residuals(params):
        params_fixed = params.copy()
        params_fixed[fixed_pos * 6:fixed_pos * 6 + 6] = param[
            fixed_pos * 6:fixed_pos * 6 + 6]
        T = _params_to_T(params_fixed)
        T_i = T[edge_i]
        T_j = T[edge_j]
        T_i_inv = np.linalg.inv(T_i)
        T_pred = np.einsum('nij,njk->nik', T_j, T_i_inv)
        T_err = np.einsum('nij,njk->nik', np.linalg.inv(T_meas), T_pred)
        res = np.empty((len(valid_edges), 6), dtype=np.float64)
        for k in range(len(valid_edges)):
            res[k] = _se3_log(T_err[k])
        res *= sqrt_w[:, None]
        return res.flatten()

    try:
        result = least_squares(
            _residuals, param, method='trf',
            loss='cauchy', f_scale=cauchy_delta,
            max_nfev=iters, verbose=0)
        param = result.x
    except Exception as e:
        logger.warning(f"[PGO] 优化失败：{e}，保留原位姿。")
        return poses
    params_fixed = param.copy()
    params_fixed[fixed_pos * 6:fixed_pos * 6 + 6] = param[
        fixed_pos * 6:fixed_pos * 6 + 6]
    T = _params_to_T(params_fixed)
    new_poses = dict(poses)
    for kf, i in kf_to_pos.items():
        new_poses[kf] = _T_to_pose(T[i])
    logger.info(f"[PGO] 完成：{n} 节点，{len(valid_edges)} 边（Cauchy）。")
    return new_poses


# =========================================================================
# 三角化基础
# =========================================================================
def _compute_min_tri_angle(X, cam_centers):
    n = len(cam_centers)
    if n < 2:
        return 0.0
    rays = X[None, :] - cam_centers
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays / np.maximum(norms, 1e-9)
    cos_mat = rays @ rays.T
    tri_angle = 180.0
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.clip(cos_mat[i, j], -1.0, 1.0))
            a = float(np.degrees(np.arccos(c)))
            if a < tri_angle:
                tri_angle = a
    return tri_angle if tri_angle < 180.0 else 0.0


def _reproj_cost_for_point(pt, xyz, frames, K64):
    cost = 0.0
    n_valid = 0
    for o in pt.obs:
        f = frames.get(o.frame_idx)
        if f is None or f.pose is None:
            continue
        P = K64 @ np.asarray(f.pose.RT[:3], dtype=np.float64)
        proj = P @ np.append(xyz, 1.0)
        if abs(proj[2]) < 1e-12:
            return np.inf
        u, v = proj[0] / proj[2], proj[1] / proj[2]
        cost += (u - o.u) ** 2 + (v - o.v) ** 2
        n_valid += 1
    if n_valid == 0:
        return np.inf
    return cost / n_valid


def _retriangulate_multiview(map_points, frames, K,
                             min_obs=MULTIVIEW_MIN_OBS,
                             ransac_iters=MULTIVIEW_RANSAC_ITERS,
                             ransac_thresh=MULTIVIEW_RANSAC_THRESH,
                             max_points=MULTIVIEW_MAX_POINTS_PER_CALL):
    if not map_points:
        return 0
    K64 = np.asarray(K, dtype=np.float64)
    updated = 0
    if len(map_points) > max_points:
        order = sorted(range(len(map_points)),
                       key=lambda i: map_points[i].obs_count,
                       reverse=True)[:max_points]
        candidates = [map_points[i] for i in order]
    else:
        candidates = map_points
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    for pt in candidates:
        obs = pt.obs
        if len(obs) < min_obs:
            continue
        proj_mats, uvs, centers_list = [], [], []
        for o in obs:
            f = frames.get(o.frame_idx)
            if f is None or f.pose is None:
                continue
            P = K64 @ np.asarray(f.pose.RT[:3], dtype=np.float64)
            proj_mats.append(P)
            uvs.append((o.u, o.v))
            centers_list.append(f.pose.center)
        if len(uvs) < min_obs:
            continue
        uvs_arr = np.asarray(uvs, dtype=np.float64)
        centers_arr = np.asarray(centers_list, dtype=np.float64)
        Xs, valid = _triangulate_batch(
            [uvs_arr], proj_mats, centers_arr, min_tri_deg,
            ransac_thresh)
        if not np.any(valid):
            continue
        best_X = Xs[int(np.argmax(valid))]
        old_xyz = pt.xyz.astype(np.float64)
        old_cost = _reproj_cost_for_point(pt, old_xyz, frames, K64)
        new_cost = _reproj_cost_for_point(pt, best_X, frames, K64)
        if not np.isfinite(old_cost) or not np.isfinite(new_cost):
            continue
        if new_cost < old_cost * MULTIVIEW_ACCEPT_IMPROVE:
            pt.xyz = best_X.astype(np.float32)
            updated += 1
    logger.info(f"[multiview] DLT 重三角化更新 {updated} / {len(candidates)} 点")
    return updated


# =========================================================================
# 点合并
# =========================================================================
def _merge_map_points(map_points, frames, norm_type):
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
        if sample_desc is not None and _is_rootsift_like(sample_desc):
            desc_thresh = MERGE_MAX_DESC_DIST_ROOTSIFT
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

    for key, ids in buckets.items():
        if len(ids) < 2:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if nkey <= key:
                        continue
                    neighbors = buckets.get(nkey)
                    if not neighbors:
                        continue
                    for i in ids:
                        pi = map_points[i]
                        for j in neighbors:
                            if find(i) == find(j):
                                continue
                            pj = map_points[j]
                            # 完全共视的两点不合并
                            if pi.obs_set == pj.obs_set:
                                continue
                            if np.linalg.norm(pi.xyz - pj.xyz) > pos_thresh:
                                continue
                            if (pi.desc is not None and pj.desc is not None
                                    and pi.desc.dtype == pj.desc.dtype):
                                if norm_type == cv2.NORM_HAMMING:
                                    d = int(cv2.norm(pi.desc, pj.desc,
                                                     cv2.NORM_HAMMING))
                                else:
                                    d = float(cv2.norm(pi.desc, pj.desc,
                                                       cv2.NORM_L2))
                                if d > desc_thresh:
                                    continue
                            union(i, j)
    groups = {}
    for i in range(len(map_points)):
        groups.setdefault(find(i), []).append(i)
    merged_count = 0
    new_points = []
    idx_map = np.full(len(map_points), -1, dtype=np.int64)
    for root, members in groups.items():
        if len(members) == 1:
            old_idx = members[0]
            new_idx = len(new_points)
            idx_map[old_idx] = new_idx
            map_points[old_idx].idx = new_idx
            new_points.append(map_points[old_idx])
            continue
        members_sorted = sorted(members,
                                key=lambda m: -map_points[m].obs_count)
        base = map_points[members_sorted[0]]
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
    logger.info(f"[merge] 合并 {merged_count} 个点，剩余 {len(map_points)}。")
    return merged_count


# =========================================================================
# COLMAP 风格 BA
# =========================================================================
@dataclass
class BAConfig:
    """BA 配置（COLMAP 风格）。"""
    loss_type: str = "huber"           # huber | cauchy | soft_l1 | linear
    loss_scale: float = 1.0
    max_iter: Optional[int] = None
    refine_focal_length: bool = True
    refine_principal_point: bool = False
    use_em_refine: bool = True
    em_iters: int = 3
    em_pi_init: float = 0.7
    em_min_inlier_ratio: float = 0.3
    em_sigma_out_ratio: float = 4.0
    em_gamma_floor: float = 1e-4
    verbose: bool = True


def _mad_sigma(residuals_flat):
    abs_r = np.abs(residuals_flat)
    if abs_r.size == 0:
        return 0.5
    mad = float(np.median(np.abs(abs_r - np.median(abs_r))))
    return max(1.4826 * mad, 0.5)


def _em_refine(res_flat, sigma_init, pi_init, n_iters,
               sigma_out_ratio=EM_SIGMA_OUT_RATIO,
               gamma_floor=EM_GAMMA_FLOOR, verbose=False):
    """EM 内点/外点分离，返回每观测的内点后验 γ。"""
    res = np.asarray(res_flat, dtype=np.float64)
    n_obs = len(res) // 2
    if n_obs == 0:
        return np.zeros(0, dtype=np.float64)
    r2 = (res.reshape(n_obs, 2) ** 2).sum(axis=1)

    sigma_in = float(np.clip(sigma_init, 0.1, 10.0))
    pi_in = float(np.clip(pi_init, 0.05, 0.99))
    sigma_out = sigma_in * 3.0
    sigma_out_cap = sigma_in * sigma_out_ratio

    gamma = np.ones(n_obs, dtype=np.float64)
    for it in range(n_iters):
        log_ratio = (np.log(max(pi_in, 1e-12) / max(1.0 - pi_in, 1e-12))
                     + 2.0 * np.log(max(sigma_out, 1e-12)
                                    / max(sigma_in, 1e-12))
                     + r2 * (1.0 / (2.0 * sigma_out ** 2)
                             - 1.0 / (2.0 * sigma_in ** 2)))
        gamma = np.maximum(expit(log_ratio), gamma_floor)
        pi_in = float(np.clip(gamma.mean(), 0.05, 0.99))
        hi = gamma > 0.9
        if hi.sum() >= 5:
            s2_in = float((r2[hi] * gamma[hi]).sum() / (2.0 * hi.sum()))
        else:
            s2_in = float((r2 * gamma).sum() / (2.0 * gamma.sum() + 1e-12))
        sigma_in = float(np.clip(np.sqrt(max(s2_in, 1e-6)), 0.1, 10.0))
        lo = gamma < 0.1
        if lo.sum() >= 5:
            s2_out = float(r2[lo].mean() / 2.0)
            sigma_out = float(np.sqrt(max(s2_out, 1e-6)))
        else:
            sigma_out = sigma_in * 3.0
        sigma_out = min(max(sigma_out, sigma_in * 1.5), sigma_out_cap)
        if verbose:
            logger.info(
                f"[BA-EM] 轮 {it+1}/{n_iters}: π_in={pi_in:.3f}, "
                f"σ_in={sigma_in:.3f}, σ_out={sigma_out:.3f}, "
                f"γ>0.5={float((gamma > 0.5).mean()):.2%}")
    return gamma


def bundle_adjustment(
    keyframe_ids, map_points, frames,
    focal, fy, cx, cy,
    optimize_points=True,
    reproj_thresh=1.0,
    image_size=1920,
    max_iter=None,
    is_global=False,
    focal_init=None,
    optimize_principal=False,
    fixed_kfs: Optional[Set[int]] = None,
    ba_cfg: Optional[BAConfig] = None,
    _retry_depth: int = 0,
    _state: Optional[Dict[str, bool]] = None,
) -> Tuple[float, float, float, float]:
    """COLMAP 风格 BA。

    正确性要点：
      - Gauge Fixing: TWO_CAMS_FROM_WORLD
        · gauge_a（观测最多）：位姿 6 DoF 全固定
        · gauge_b（观测次多）：只固定旋转 3 DoF
        · 剩余 3 平移 + 6 旋转 + 1 尺度 = 7 DoF 自由
      - 主点默认不进参数向量（optimize_principal=False）
      - 焦距边界绑定 focal_init，防全局漂移
      - EM 只作为最后一步的加权精修，不参与主目标
    """
    cfg = _cfg()
    if ba_cfg is None:
        ba_cfg = BAConfig()
    if _state is not None:
        _state["success"] = False
    if max_iter is None:
        max_iter = ba_cfg.max_iter if ba_cfg.max_iter is not None else (
            cfg.ba_global_iters if is_global else cfg.ba_local_iters)
    if focal_init is None:
        focal_init = focal

    # ---------- 1. 收集观测 ----------
    valid_kfs = [f for f in keyframe_ids
                 if f in frames and frames[f].pose is not None]
    if len(valid_kfs) < 2:
        return focal, fy, cx, cy
    keyframe_ids = valid_kfs

    obs_raw = []
    for f_idx in keyframe_ids:
        fr = frames[f_idx]
        fm = fr.feat_map
        if fm is None:
            continue
        for kp_idx, pt_idx in enumerate(fm):
            pt_idx = int(pt_idx)
            if 0 <= pt_idx < len(map_points):
                u, v = fr.kps[kp_idx].pt
                obs_raw.append((f_idx, pt_idx, float(u), float(v)))

    if len(obs_raw) < cfg.ba_min_obs:
        if _state is not None:
            _state["success"] = True
        return focal, fy, cx, cy

    if len(obs_raw) > BA_MAX_OBS:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(obs_raw), BA_MAX_OBS, replace=False)
        obs_raw = [obs_raw[i] for i in sel]

    obs_count = {f: 0 for f in keyframe_ids}
    for f, _, _, _ in obs_raw:
        obs_count[f] += 1

    pt_kf = {}
    for f, pt, _, _ in obs_raw:
        pt_kf.setdefault(pt, set()).add(f)
    eligible = {pt for pt, kfs in pt_kf.items() if len(kfs) >= 2}
    if not eligible:
        if _state is not None:
            _state["success"] = True
        return focal, fy, cx, cy
    obs = [o for o in obs_raw if o[1] in eligible]

    all_pt_ids = sorted(eligible)
    if len(all_pt_ids) > BA_MAX_POINTS:
        cnt = {pid: 0 for pid in all_pt_ids}
        for _, pid, _, _ in obs:
            cnt[pid] += 1
        all_pt_ids = sorted(all_pt_ids, key=lambda x: cnt[x],
                            reverse=True)[:BA_MAX_POINTS]
        pset = set(all_pt_ids)
        obs = [o for o in obs if o[1] in pset]

    n_obs = len(obs)
    if n_obs < 10:
        return focal, fy, cx, cy

    # ---------- 2. Gauge Fixing (COLMAP TWO_CAMS_FROM_WORLD) ----------
    sorted_by_obs = sorted(obs_count.keys(), key=lambda x: -obs_count[x])
    if fixed_kfs:
        uf = [k for k in sorted(fixed_kfs) if k in obs_count]
        if len(uf) >= 2:
            gauge_a, gauge_b = uf[0], uf[1]
        elif len(uf) == 1:
            gauge_a = uf[0]
            others = [k for k in sorted_by_obs if k != gauge_a]
            gauge_b = others[0] if others else gauge_a
        else:
            gauge_a, gauge_b = sorted_by_obs[0], sorted_by_obs[1]
    elif is_global:
        # 全局：首尾两帧，条件数最好
        srt = sorted(obs_count.keys())
        gauge_a, gauge_b = srt[0], srt[-1]
    else:
        gauge_a, gauge_b = sorted_by_obs[0], sorted_by_obs[1]

    # ---------- 3. 参数布局 ----------
    opt_focal = bool(ba_cfg.refine_focal_length)
    opt_pp = bool(optimize_principal and ba_cfg.refine_principal_point)

    idx = 0
    fx_i = fy_i = cx_i = cy_i = -1
    if opt_focal:
        fx_i = idx; idx += 1
        fy_i = idx; idx += 1
    if opt_pp:
        cx_i = idx; idx += 1
        cy_i = idx; idx += 1

    kf_pose_i = {}
    for k in keyframe_ids:
        kf_pose_i[k] = idx
        idx += 6
    pt_xyz_i = {}
    for pid in all_pt_ids:
        pt_xyz_i[pid] = idx
        idx += 3
    n_params = idx

    free_mask = np.ones(n_params, dtype=bool)
    ia = kf_pose_i[gauge_a]
    free_mask[ia:ia + 6] = False
    if gauge_b != gauge_a:
        ib = kf_pose_i[gauge_b]
        free_mask[ib:ib + 3] = False
    if not optimize_points:
        for pid in all_pt_ids:
            i = pt_xyz_i[pid]
            free_mask[i:i + 3] = False

    free_idx = np.where(free_mask)[0]
    n_free = len(free_idx)
    if n_free == 0:
        return focal, fy, cx, cy

    p0 = np.zeros(n_params, dtype=np.float64)
    if opt_focal:
        p0[fx_i] = focal
        p0[fy_i] = fy
    if opt_pp:
        p0[cx_i] = cx
        p0[cy_i] = cy
    for k in keyframe_ids:
        i = kf_pose_i[k]
        pose = frames[k].pose
        rv, _ = cv2.Rodrigues(pose.R)
        p0[i:i + 3] = rv.flatten()
        p0[i + 3:i + 6] = pose.t.flatten()
    for pid in all_pt_ids:
        i = pt_xyz_i[pid]
        p0[i:i + 3] = map_points[pid].xyz

    lb = np.full(n_params, -np.inf)
    ub = np.full(n_params, np.inf)
    if opt_focal:
        fx_lo = max(focal * (1 - FOCAL_MAX_STEP_RATIO),
                    focal_init * (1 - FOCAL_GLOBAL_DRIFT_MAX))
        fx_hi = min(focal * (1 + FOCAL_MAX_STEP_RATIO),
                    focal_init * (1 + FOCAL_GLOBAL_DRIFT_MAX))
        fy_lo = max(fy * (1 - FOCAL_MAX_STEP_RATIO),
                    focal_init * (1 - FOCAL_GLOBAL_DRIFT_MAX))
        fy_hi = min(fy * (1 + FOCAL_MAX_STEP_RATIO),
                    focal_init * (1 + FOCAL_GLOBAL_DRIFT_MAX))
        if fx_lo >= fx_hi or fy_lo >= fy_hi:
            logger.warning("[BA] 焦距边界交叉，跳过。")
            return focal, fy, cx, cy
        lb[fx_i], ub[fx_i] = fx_lo, fx_hi
        lb[fy_i], ub[fy_i] = fy_lo, fy_hi
    if opt_pp:
        step = PRINCIPAL_PT_MAX_STEP * image_size
        lb[cx_i], ub[cx_i] = cx - step, cx + step
        lb[cy_i], ub[cy_i] = cy - step, cy + step

    xyz_all = np.array([map_points[pid].xyz for pid in all_pt_ids],
                       dtype=np.float64)
    scene_scale = 1.0
    if len(xyz_all):
        scene_scale = float(
            np.linalg.norm(xyz_all.max(0) - xyz_all.min(0))) + 1e-6
    t_bound = max(scene_scale * 5.0, 10.0)
    for k in keyframe_ids:
        i = kf_pose_i[k]
        lb[i + 3:i + 6] = -t_bound
        ub[i + 3:i + 6] = t_bound

    kf_list = list(keyframe_ids)
    kf_to_row = {k: i for i, k in enumerate(kf_list)}
    kf_pose_cols = np.array(
        [[kf_pose_i[k] + j for j in range(6)] for k in kf_list],
        dtype=np.int64)
    pt_xyz_cols = np.array(
        [[pt_xyz_i[pid] + j for j in range(3)] for pid in all_pt_ids],
        dtype=np.int64)
    pt_row_map = {pid: i for i, pid in enumerate(all_pt_ids)}

    obs_kf_row = np.fromiter(
        (kf_to_row[f] for f, _, _, _ in obs), dtype=np.int64, count=n_obs)
    obs_pt_row = np.fromiter(
        (pt_row_map[pid] for _, pid, _, _ in obs),
        dtype=np.int64, count=n_obs)
    obs_u = np.fromiter((u for _, _, u, _ in obs),
                        dtype=np.float64, count=n_obs)
    obs_v = np.fromiter((v for _, _, _, v in obs),
                        dtype=np.float64, count=n_obs)

    def _residuals(p_free):
        p = p0.copy()
        p[free_idx] = p_free
        if opt_focal:
            fx_l, fy_l = p[fx_i], p[fy_i]
        else:
            fx_l, fy_l = focal, fy
        if opt_pp:
            cx_l, cy_l = p[cx_i], p[cy_i]
        else:
            cx_l, cy_l = cx, cy
        pose_p = p[kf_pose_cols]
        Rs = Rot.from_rotvec(pose_p[:, :3]).as_matrix()
        ts = pose_p[:, 3:]
        pts = p[pt_xyz_cols]
        R_o = Rs[obs_kf_row]
        t_o = ts[obs_kf_row]
        X_o = pts[obs_pt_row]
        X_cam = np.einsum('nij,nj->ni', R_o, X_o) + t_o
        z = X_cam[:, 2]
        z_safe = np.where(z > 1e-6, z, 1e-6)
        u_proj = fx_l * X_cam[:, 0] / z_safe + cx_l
        v_proj = fy_l * X_cam[:, 1] / z_safe + cy_l
        res = np.empty((n_obs, 2), dtype=np.float64)
        res[:, 0] = u_proj - obs_u
        res[:, 1] = v_proj - obs_v
        bad = z <= 1e-6
        if np.any(bad):
            res[bad, 0] = 1e4
            res[bad, 1] = 1e4
        return res.ravel()

    def _build_sparsity():
        n_res = 2 * n_obs
        sp = lil_matrix((n_res, n_free), dtype=np.int8)
        free_pos = -np.ones(n_params, dtype=np.int64)
        free_pos[free_idx] = np.arange(n_free)
        if opt_focal:
            for c in (free_pos[fx_i], free_pos[fy_i]):
                if c >= 0:
                    sp[:, c] = 1
        if opt_pp:
            for c in (free_pos[cx_i], free_pos[cy_i]):
                if c >= 0:
                    sp[:, c] = 1
        for i_o in range(n_obs):
            r0, r1 = 2 * i_o, 2 * i_o + 1
            base = kf_pose_i[kf_list[obs_kf_row[i_o]]]
            for j in range(6):
                c = free_pos[base + j]
                if c >= 0:
                    sp[r0, c] = 1
                    sp[r1, c] = 1
            pid = all_pt_ids[obs_pt_row[i_o]]
            base_p = pt_xyz_i[pid]
            for j in range(3):
                c = free_pos[base_p + j]
                if c >= 0:
                    sp[r0, c] = 1
                    sp[r1, c] = 1
        return sp.tocsr()

    sparsity = _build_sparsity()

    p_free0 = p0[free_idx].copy()
    res0 = _residuals(p_free0).reshape(n_obs, 2)
    err0 = np.linalg.norm(res0, axis=1)
    rms_before = float(np.sqrt(np.median(err0 ** 2)))
    if not np.isfinite(rms_before):
        rms_before = 0.0

    loss_type = ba_cfg.loss_type
    loss_scale = ba_cfg.loss_scale
    if rms_before > reproj_thresh * BA_MAX_INIT_RMS_RATIO:
        logger.warning(f"[BA] 初值 RMS={rms_before:.2f}px 过差，切 Cauchy")
        loss_type = "cauchy"
        loss_scale = max(rms_before, reproj_thresh)
    if loss_type not in ("huber", "cauchy", "soft_l1", "linear"):
        loss_type = "huber"

    if ba_cfg.verbose:
        logger.info(
            f"[BA-LM] loss={loss_type}, scale={loss_scale:.3f}px, "
            f"KFs={len(keyframe_ids)} (gauge {gauge_a},{gauge_b}), "
            f"pts={len(all_pt_ids)}, obs={n_obs}, "
            f"fx={'opt' if opt_focal else 'fix'}, "
            f"pp={'opt' if opt_pp else 'fix'}, RMS0={rms_before:.3f}")

    # ---------- 4. LM + 损失 ----------
    try:
        result = least_squares(
            _residuals, p_free0,
            bounds=(lb[free_idx], ub[free_idx]),
            method='trf',
            loss=loss_type,
            f_scale=max(loss_scale, 0.5),
            max_nfev=max_iter,
            verbose=0, ftol=1e-5, xtol=1e-5, gtol=1e-5,
            jac_sparsity=sparsity)
        p_free_opt = result.x
    except Exception as e:
        logger.warning(f"[BA-LM] 优化失败：{e}")
        return focal, fy, cx, cy

    p_opt = p0.copy()
    p_opt[free_idx] = p_free_opt

    def _decode(p):
        if opt_focal:
            f_new, fy_new = float(p[fx_i]), float(p[fy_i])
        else:
            f_new, fy_new = focal, fy
        if opt_pp:
            cx_new, cy_new = float(p[cx_i]), float(p[cy_i])
        else:
            cx_new, cy_new = cx, cy
        return f_new, fy_new, cx_new, cy_new

    focal_new, fy_new, cx_new, cy_new = _decode(p_opt)

    # ---------- 5. EM 最后做 ----------
    res_cur = _residuals(p_free_opt).reshape(n_obs, 2)
    gamma_final = None
    if ba_cfg.use_em_refine and n_obs >= 50:
        err_cur = np.linalg.norm(res_cur, axis=1)
        gamma = _em_refine(
            res_cur.ravel(),
            sigma_init=_mad_sigma(err_cur),
            pi_init=ba_cfg.em_pi_init,
            n_iters=ba_cfg.em_iters,
            sigma_out_ratio=ba_cfg.em_sigma_out_ratio,
            gamma_floor=ba_cfg.em_gamma_floor,
            verbose=ba_cfg.verbose)
        gamma_final = gamma
        inlier_ratio = float((gamma >= 0.5).mean())
        if ba_cfg.verbose:
            logger.info(f"[BA-EM] 内点率={inlier_ratio:.2%}")
        if (inlier_ratio >= ba_cfg.em_min_inlier_ratio
                and inlier_ratio < 0.99):
            sqrt_w = np.repeat(np.sqrt(np.maximum(gamma, 1e-6)), 2)

            def _residuals_w(p_free):
                return _residuals(p_free) * sqrt_w

            try:
                result2 = least_squares(
                    _residuals_w, p_free_opt,
                    bounds=(lb[free_idx], ub[free_idx]),
                    method='trf', loss='linear',
                    max_nfev=max(5, max_iter // 2),
                    verbose=0, ftol=1e-5, xtol=1e-5, gtol=1e-5,
                    jac_sparsity=sparsity)
                p_free_opt = result2.x
                p_opt = p0.copy()
                p_opt[free_idx] = p_free_opt
                focal_new, fy_new, cx_new, cy_new = _decode(p_opt)
            except Exception as e:
                logger.warning(f"[BA-EM] 加权精修失败：{e}")

    # ---------- 6. 结果 ----------
    res_final = _residuals(p_free_opt).reshape(n_obs, 2)
    err_final = np.linalg.norm(res_final, axis=1)
    rms_after = float(np.sqrt(np.median(err_final ** 2)))

    pose_p = p_opt[kf_pose_cols]
    new_poses = {}
    for i, k in enumerate(kf_list):
        R, _ = cv2.Rodrigues(pose_p[i, :3])
        new_poses[k] = CameraPose(R, pose_p[i, 3:].reshape(3, 1))
    pts_p = p_opt[pt_xyz_cols]
    new_pts = {pid: pts_p[i].copy() for i, pid in enumerate(all_pt_ids)}

    # ---------- 7. 过拟合 / 变差 → 完整回滚 ----------
    overfit_min = (BA_OVERFIT_MIN_RMS_GLOBAL if is_global
                   else BA_OVERFIT_MIN_RMS_LOCAL)
    obs_per_point = n_obs / max(len(all_pt_ids), 1)
    overfit = (rms_after < overfit_min
               and rms_before > BA_OVERFIT_MIN_BEFORE
               and obs_per_point < 1.8)
    worse = rms_after > rms_before * BA_MAX_RMS_INCREASE
    if overfit or worse:
        reason = "疑似过拟合" if overfit else "结果变差"
        logger.warning(f"[BA] {reason}"
                       f"（RMS {rms_before:.3f}→{rms_after:.3f}），回滚")
        return focal, fy, cx, cy

    # ---------- 8. 提交 ----------
    for k, pose in new_poses.items():
        frames[k].pose = pose
    for pid, xyz in new_pts.items():
        map_points[pid].xyz = xyz
    if _state is not None:
        _state["success"] = True
    if ba_cfg.verbose:
        n_in = (int((gamma_final >= 0.5).sum())
                if gamma_final is not None else n_obs)
        logger.info(
            f"[BA] 完成：RMS {rms_before:.3f}→{rms_after:.3f}px，"
            f"fx {focal:.1f}→{focal_new:.1f}，内点 {n_in}/{n_obs}")
    return focal_new, fy_new, cx_new, cy_new


# =========================================================================
# 地图点管理
# =========================================================================
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
    pt_ids = pt_ids[keep]; f_ids = f_ids[keep]
    us = us[keep]; vs = vs[keep]
    xyz_o = xyz_all[pt_ids]
    R_o = R_all[f_ids]; t_o = t_all[f_ids]
    pt_cam = np.einsum('nij,nj->ni', R_o, xyz_o) + t_o
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


def _remove_bad_observations(map_points, frames, poses_by_frame,
                             focal, fy, cx, cy, err_thresh):
    removed = 0
    for pt in map_points:
        if pt.obs_count < MIN_OBSERVATIONS:
            continue
        xyz = pt.xyz
        if xyz.size != 3 or not np.isfinite(xyz).all():
            continue
        new_obs = []
        new_set = set()
        removed_keys = []
        for o in pt.obs:
            pose = poses_by_frame.get(o.frame_idx)
            if pose is None:
                removed_keys.append((o.frame_idx, o.kp_idx))
                removed += 1
                continue
            pt_cam = pose.R @ xyz.reshape(3, 1) + pose.t
            d = float(pt_cam[2, 0])
            if d <= 1e-6:
                removed_keys.append((o.frame_idx, o.kp_idx))
                removed += 1
                continue
            u = focal * float(pt_cam[0, 0]) / d + cx
            v = fy * float(pt_cam[1, 0]) / d + cy
            err = float(np.hypot(u - o.u, v - o.v))
            if err > err_thresh:
                removed_keys.append((o.frame_idx, o.kp_idx))
                removed += 1
                continue
            new_obs.append(o)
            new_set.add((o.frame_idx, o.kp_idx))
        if len(new_obs) != len(pt.obs):
            pt.obs = new_obs
            pt.obs_set = new_set
            for fr_idx, kp_idx in removed_keys:
                fr = frames.get(fr_idx)
                if fr is None or fr.feat_map is None:
                    continue
                if not (0 <= kp_idx < len(fr.feat_map)):
                    continue
                if fr.feat_map[kp_idx] == pt.idx:
                    fr.feat_map[kp_idx] = -1
    if removed > 0:
        logger.info(f"[obs-clean] 剔除 {removed} 条坏观测。")


def _statistical_outlier_mask(xyz, k=SOR_K_NEIGHBORS, std_ratio=SOR_STD_RATIO):
    """PCL 风格 StatisticalOutlierRemoval。"""
    if len(xyz) < k + 1:
        return np.ones(len(xyz), dtype=bool)
    finite = np.isfinite(xyz).all(axis=1)
    if not np.any(finite):
        return np.zeros(len(xyz), dtype=bool)
    valid_xyz = xyz[finite]
    try:
        tree = cKDTree(valid_xyz)
        d, _ = tree.query(valid_xyz, k=k + 1)
        mean_d = d[:, 1:].mean(axis=1)
        mu, sigma = float(mean_d.mean()), float(mean_d.std())
        valid_mask = mean_d < mu + std_ratio * sigma
    except Exception:
        valid_mask = np.ones(len(valid_xyz), dtype=bool)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[finite] = valid_mask
    return mask


def prune_map_points(map_points, frames, reproj_thresh, focal, fy, cx, cy):
    if not map_points:
        return
    poses_by_frame = {f: fr.pose for f, fr in frames.items()}
    err_thresh = MAX_REPROJ_ERROR * reproj_thresh
    _remove_bad_observations(map_points, frames, poses_by_frame,
                             focal, fy, cx, cy, err_thresh)
    n_old = len(map_points)
    mean_err, valid_count, neg_ratio = _compute_point_errors(
        map_points, poses_by_frame, focal, fy, cx, cy)
    xyz_arr = np.array([p.xyz for p in map_points], dtype=np.float64)
    sor_mask = _statistical_outlier_mask(xyz_arr)
    to_remove = []
    for i, pt in enumerate(map_points):
        if not sor_mask[i]:
            to_remove.append(i)
            continue
        if pt.obs_count < MIN_OBSERVATIONS:
            to_remove.append(i)
        elif neg_ratio[i] >= 0.5:
            to_remove.append(i)
        elif valid_count[i] > 0 and mean_err[i] > err_thresh:
            to_remove.append(i)
    if not to_remove:
        return
    keep = np.ones(n_old, dtype=bool)
    keep[np.asarray(to_remove, dtype=np.int64)] = False
    idx_map = np.full(n_old, -1, dtype=np.int64)
    idx_map[keep] = np.arange(int(keep.sum()))
    for frame in frames.values():
        fm = frame.feat_map
        if fm is None or fm.size == 0:
            continue
        oob = fm >= n_old
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
    logger.info(f"[prune] 删除 {len(to_remove)} 个点，剩余 {len(map_points)}。")


def filter_point_cloud(map_points, poses_by_frame, focal, fy, cx, cy,
                       reproj_thresh):
    if not map_points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
    xyz = np.array([p.xyz for p in map_points], dtype=np.float32)
    mean_err, valid_count, neg_ratio = _compute_point_errors(
        map_points, poses_by_frame, focal, fy, cx, cy)
    errors = np.where((valid_count >= MIN_OBSERVATIONS) & (neg_ratio < 0.5),
                      mean_err, np.inf)
    finite = np.isfinite(errors)
    if not np.any(finite):
        return xyz, np.zeros(len(map_points), dtype=bool)
    median = float(np.median(errors[finite]))
    thresh = max(median * FILTER_MEDIAN_FACTOR,
                 reproj_thresh * FILTER_REPROJ_FACTOR)
    mask = finite & (errors < thresh)
    return xyz, mask


# =========================================================================
# 辅助
# =========================================================================
def _update_desc(pt, new_desc):
    old = pt.desc
    if old is None:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
        return
    up_dist = _update_dist_for(new_desc)
    if new_desc.dtype != old.dtype:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
        return
    if old.dtype == np.uint8:
        d = int(cv2.norm(old, new_desc, cv2.NORM_HAMMING))
    else:
        d = float(cv2.norm(old, new_desc, cv2.NORM_L2))
    if pt.obs_count < 3 or pt.desc_age > 5 or d >= up_dist * 1.2:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
    else:
        pt.desc_age += 1


def _rotation_angle_deg(R):
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv) * 180.0 / np.pi)


def _relative_pose(pose_i, pose_j):
    R = pose_j.R @ pose_i.R.T
    t = pose_j.t - R @ pose_i.t
    return CameraPose(R, t)


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
    total = 0
    count = 0
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
    if total < 30:
        return 0.0
    return count / total


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
        covis = len(curr_pts & kf_pts) / max(1, min(len(curr_pts),
                                                    len(kf_pts)))
        if covis > KF_CULL_COVIS:
            culled.append(kf)
    for kf in culled:
        if kf in keyframes:
            keyframes.remove(kf)
        frames[kf].feat_map = np.full(len(frames[kf].kps), -1,
                                       dtype=np.int64)
        frames[kf].is_keyframe = False
        covis_graph.remove_keyframe(kf)


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
    return [[(o.frame_idx, o.kp_idx, o.u, o.v) for o in pt.obs]
            for pt in points]


# =========================================================================
# 两视图三角化（批量）
# =========================================================================
def _triangulate_from_pair(curr_idx, prev_idx, matches, inlier_mask,
                           frames, map_points, focal, fy, cx, cy,
                           triang_thresh):
    if prev_idx >= curr_idx:
        raise RuntimeError(f"prev_idx={prev_idx} 必须 < curr_idx={curr_idx}")
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
    err_thresh = MAX_REPROJ_ERROR * triang_thresh * ADD_OBS_ERR_SCALE
    err_sq = err_thresh * err_thresh
    prev_feat_map = prev.feat_map
    curr_feat_map = curr.feat_map
    prev_colors = prev.colors
    curr_colors = curr.colors
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    add_to_prev = []
    add_to_curr = []
    new_match_ids = []
    inlier_ids = np.where(inlier_mask)[0]
    for k in inlier_ids:
        m = matches[k]
        curr_feat = int(m.trainIdx)
        prev_feat = int(m.queryIdx)
        if curr_feat >= len(curr_feat_map) or prev_feat >= len(prev_feat_map):
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
        if (u - frame.kps[feat].pt[0]) ** 2 + \
           (v - frame.kps[feat].pt[1]) ** 2 > err_sq:
            return False
        c = (colors_f[feat] if colors_f is not None
             and feat < len(colors_f) else None)
        if pt.add_observation(Observation(
                frame_idx=idx_frame, kp_idx=int(feat),
                u=float(frame.kps[feat].pt[0]),
                v=float(frame.kps[feat].pt[1]),
                color=(c.copy() if c is not None else None))):
            frame.feat_map[feat] = pt_idx
            _update_desc(pt, frame.desc[feat])
            return True
        return False

    for prev_feat, pt_idx in add_to_prev:
        _try_add(prev, prev_feat, pt_idx, prev_idx, R_prev, t_prev,
                 prev_colors)
    for curr_feat, pt_idx in add_to_curr:
        _try_add(curr, curr_feat, pt_idx, curr_idx, R_curr, t_curr,
                 curr_colors)

    if not new_match_ids:
        return
    pts_prev = np.array([prev.kps[matches[k].queryIdx].pt
                         for k in new_match_ids], dtype=np.float64)
    pts_curr = np.array([curr.kps[matches[k].trainIdx].pt
                         for k in new_match_ids], dtype=np.float64)
    cam_prev_c = prev.pose.center
    cam_curr_c = curr.pose.center
    uvs = [pts_prev, pts_curr]
    projs = [P_prev, P_curr]
    cams = np.asarray([cam_prev_c, cam_curr_c], dtype=np.float64)
    Xs, valid = _triangulate_batch(uvs, projs, cams, min_tri_deg,
                                   triang_thresh)
    for k in np.where(valid)[0]:
        mid = new_match_ids[k]
        m = matches[mid]
        prev_feat = int(m.queryIdx)
        curr_feat = int(m.trainIdx)
        if curr_feat_map[curr_feat] >= 0 or prev_feat_map[prev_feat] >= 0:
            continue
        pt_idx = len(map_points)
        c_prev = (prev_colors[prev_feat] if prev_colors is not None
                  and prev_feat < len(prev_colors) else None)
        pt = MapPoint(idx=pt_idx, xyz=Xs[k].astype(np.float32),
                      desc=prev.desc[prev_feat].copy())
        pt.add_observation(Observation(
            frame_idx=prev_idx, kp_idx=prev_feat,
            u=float(prev.kps[prev_feat].pt[0]),
            v=float(prev.kps[prev_feat].pt[1]),
            color=(c_prev.copy() if c_prev is not None else None)))
        c_curr = (curr_colors[curr_feat] if curr_colors is not None
                  and curr_feat < len(curr_colors) else None)
        pt.add_observation(Observation(
            frame_idx=curr_idx, kp_idx=curr_feat,
            u=float(curr.kps[curr_feat].pt[0]),
            v=float(curr.kps[curr_feat].pt[1]),
            color=(c_curr.copy() if c_curr is not None else None)))
        map_points.append(pt)
        prev_feat_map[prev_feat] = pt_idx
        curr_feat_map[curr_feat] = pt_idx


# =========================================================================
# next-best-view 三角化
# =========================================================================
def _triangulate_next_best_view(curr_idx, keyframes, frames, map_points,
                                focal, fy, cx, cy, triang_thresh, top_k=8):
    if curr_idx not in frames or frames[curr_idx].pose is None:
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
        score = float(covis) + 0.1 * float(np.log1p(baseline))
        scores.append((kf, score))
    if not scores:
        return 0
    scores.sort(key=lambda x: -x[1])
    selected = [kf for kf, _ in scores[:top_k]]
    K = _build_K(focal, fy, cx, cy)
    err_thresh = MAX_REPROJ_ERROR * triang_thresh * ADD_OBS_ERR_SCALE
    err_th_sq = err_thresh * err_thresh
    new_pts_total = 0
    added_obs_total = 0
    min_tri_deg = _cfg().filter_min_tri_angle_deg
    for kf in selected:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        a = frames[kf]
        b = frames[curr_idx]
        if a.pose is None or b.pose is None:
            continue
        kp_desc_a = a.desc
        kp_desc_b = b.desc
        if kp_desc_a is None or kp_desc_b is None:
            continue
        nt, md, _ = get_metric(kp_desc_a)
        if len(kp_desc_a) < 4 or len(kp_desc_b) < 4:
            continue
        matches = _match_ratio_absolute(kp_desc_a, kp_desc_b, nt, md)
        if len(matches) < 8:
            continue
        a_fm_now = frames[kf].feat_map
        curr_fm_now = frames[curr_idx].feat_map
        add_to_a = []
        add_to_b = []
        tri_pairs = []
        for m in matches:
            ka = int(m.queryIdx)
            kb = int(m.trainIdx)
            if ka >= len(a_fm_now) or kb >= len(curr_fm_now):
                continue
            pa = int(a_fm_now[ka])
            pb = int(curr_fm_now[kb])
            if pa >= 0 and pb >= 0:
                continue
            elif pa >= 0 and pb < 0:
                add_to_b.append((kb, pa))
            elif pa < 0 and pb >= 0:
                add_to_a.append((ka, pb))
            else:
                tri_pairs.append((ka, kb))

        def _try_add_obs(frame, feat, pt_idx, idx_frame, colors_f):
            if pt_idx >= len(map_points):
                return False
            pt = map_points[pt_idx]
            if pt.obs_count >= MAX_OBS_PER_POINT:
                return False
            Xw = pt.xyz
            if Xw.size != 3 or not np.isfinite(Xw).all():
                return False
            if not _check_obs_angle(pt, idx_frame, frames, min_tri_deg):
                return False
            pt_cam = frame.pose.R @ Xw.reshape(3, 1) + frame.pose.t
            d = float(pt_cam[2, 0])
            if d <= 1e-6:
                return False
            u = focal * float(pt_cam[0, 0]) / d + cx
            v = fy * float(pt_cam[1, 0]) / d + cy
            if (u - frame.kps[feat].pt[0]) ** 2 + \
               (v - frame.kps[feat].pt[1]) ** 2 > err_th_sq:
                return False
            c = (colors_f[feat] if colors_f is not None
                 and feat < len(colors_f) else None)
            if pt.add_observation(Observation(
                    frame_idx=idx_frame, kp_idx=feat,
                    u=float(frame.kps[feat].pt[0]),
                    v=float(frame.kps[feat].pt[1]),
                    color=(c.copy() if c is not None else None))):
                frame.feat_map[feat] = pt_idx
                _update_desc(pt, frame.desc[feat])
                return True
            return False

        for kp_b, pt_idx in add_to_b:
            if _try_add_obs(b, kp_b, pt_idx, curr_idx, b.colors):
                added_obs_total += 1
        for kp_a, pt_idx in add_to_a:
            if _try_add_obs(a, kp_a, pt_idx, kf, a.colors):
                added_obs_total += 1

        if len(tri_pairs) < 4:
            continue
        src_kp_idx = [ka for ka, _ in tri_pairs]
        dst_kp_idx = [kb for _, kb in tri_pairs]
        pts_a = np.array([a.kps[k].pt for k in src_kp_idx], dtype=np.float64)
        pts_b = np.array([b.kps[k].pt for k in dst_kp_idx], dtype=np.float64)
        P_a = K @ np.asarray(a.pose.RT[:3], dtype=np.float64)
        P_b = K @ np.asarray(b.pose.RT[:3], dtype=np.float64)
        cams = np.asarray([a.pose.center, b.pose.center], dtype=np.float64)
        Xs, valid = _triangulate_batch([pts_a, pts_b], [P_a, P_b],
                                       cams, min_tri_deg, triang_thresh)
        for k in np.where(valid)[0]:
            ka = src_kp_idx[k]
            kb = dst_kp_idx[k]
            if curr_fm_now[kb] >= 0 or a_fm_now[ka] >= 0:
                continue
            pt_idx = len(map_points)
            c_a = (a.colors[ka] if a.colors is not None
                   and ka < len(a.colors) else None)
            c_b = (b.colors[kb] if b.colors is not None
                   and kb < len(b.colors) else None)
            new_pt = MapPoint(idx=pt_idx, xyz=Xs[k].astype(np.float32),
                              desc=a.desc[ka].copy())
            new_pt.add_observation(Observation(
                frame_idx=kf, kp_idx=ka,
                u=float(a.kps[ka].pt[0]), v=float(a.kps[ka].pt[1]),
                color=(c_a.copy() if c_a is not None else None)))
            new_pt.add_observation(Observation(
                frame_idx=curr_idx, kp_idx=kb,
                u=float(b.kps[kb].pt[0]), v=float(b.kps[kb].pt[1]),
                color=(c_b.copy() if c_b is not None else None)))
            map_points.append(new_pt)
            a_fm_now[ka] = pt_idx
            curr_fm_now[kb] = pt_idx
            new_pts_total += 1
    if new_pts_total > 0 or added_obs_total > 0:
        logger.info(f"[nbv-tri] 帧 {curr_idx}：从 {len(selected)} 视图新增 "
                    f"{new_pts_total} 点，追加 {added_obs_total} 观测")
    return new_pts_total


# =========================================================================
# CompleteTracks + FindTransitivity
# =========================================================================
def _find_transitivity(curr_idx, keyframes, frames, map_points,
                       max_transitivity=1):
    if curr_idx not in frames or frames[curr_idx].feat_map is None:
        return []
    curr_fm = frames[curr_idx].feat_map
    curr_bound = {int(pt): kp for kp, pt in enumerate(curr_fm) if pt >= 0}
    results = []
    neighbor_window = keyframes[-max_transitivity * 3 - 1:-1]
    for kf in neighbor_window:
        if kf not in frames or frames[kf].feat_map is None:
            continue
        kf_fm = frames[kf].feat_map
        for kp_kf, pt_idx in enumerate(kf_fm):
            pt_idx = int(pt_idx)
            if pt_idx < 0 or pt_idx >= len(map_points):
                continue
            if pt_idx in curr_bound:
                results.append((kp_kf, curr_bound[pt_idx]))
    return results


def _complete_tracks(keyframes, frames, map_points, K, reproj_thresh,
                     max_transitivity=5):
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
        kps_uv = np.array([kp.pt for kp in f.kps], dtype=np.float32)
        if len(kps_uv) == 0:
            continue
        try:
            tree = cKDTree(kps_uv)
        except Exception:
            tree = None
        radius = cfg.guided_radius_px * 1.5
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
            max_dist = update_dist_for_desc(f.desc)
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
                    dval = int(cv2.norm(f.desc[ci], desc_pt,
                                        cv2.NORM_HAMMING))
                else:
                    dval = float(cv2.norm(f.desc[ci], desc_pt, cv2.NORM_L2))
                if dval > max_dist:
                    continue
                if best_d is None or dval < best_d:
                    best_d, best_kp = dval, ci
            if best_kp < 0:
                continue
            c = (f.colors[best_kp] if f.colors is not None
                 and best_kp < len(f.colors) else None)
            if pt.add_observation(Observation(
                    frame_idx=kf, kp_idx=best_kp,
                    u=float(f.kps[best_kp].pt[0]),
                    v=float(f.kps[best_kp].pt[1]),
                    color=(c.copy() if c is not None else None))):
                fm[best_kp] = pt.idx
                _update_desc(pt, f.desc[best_kp])
                added_total += 1
    if added_total > 0:
        logger.info(f"[complete-tracks] 补全 {added_total} 条观测")
    return added_total


# =========================================================================
# 投影追加观测
# =========================================================================
def _project_and_add_observations(curr_idx, keyframes, frames, map_points,
                                   focal, fy, cx, cy, n_views=10,
                                   proj_radius_px=20.0):
    if curr_idx not in frames or frames[curr_idx].pose is None:
        return 0
    curr = frames[curr_idx]
    curr_pose = curr.pose
    curr_desc = curr.desc
    curr_fm = curr.feat_map
    if curr_desc is None or curr_fm is None or len(curr.kps) == 0:
        return 0
    neighbors = (keyframes[-n_views:] if len(keyframes) > n_views
                 else list(keyframes))
    neighbors = [k for k in neighbors
                 if k != curr_idx and k in frames
                 and frames[k].pose is not None]
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
    max_desc_dist = update_dist_for_desc(curr_desc)
    kps_uv = np.array([kp.pt for kp in curr.kps], dtype=np.float32)
    try:
        tree = cKDTree(kps_uv)
    except Exception:
        tree = None
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
            cand = np.where(dx * dx + dy * dy <=
                            proj_radius_px ** 2)[0].tolist()
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
            if ci in used_kps:
                continue
            if curr_fm[ci] >= 0:
                continue
            if (desc_pt is not None and curr_desc is not None
                    and curr_desc.dtype == desc_pt.dtype):
                if curr_desc.dtype == np.uint8:
                    dval = int(cv2.norm(curr_desc[ci], desc_pt,
                                        cv2.NORM_HAMMING))
                else:
                    dval = float(cv2.norm(curr_desc[ci], desc_pt,
                                          cv2.NORM_L2))
                if best_d is None or dval < best_d:
                    best_d, best_kp = dval, ci
            else:
                if best_kp < 0:
                    best_kp = ci
        if best_kp < 0:
            continue
        if best_d is not None and best_d > max_desc_dist:
            continue
        c = (curr.colors[best_kp] if curr.colors is not None
             and best_kp < len(curr.colors) else None)
        if pt.add_observation(Observation(
                frame_idx=curr_idx, kp_idx=best_kp,
                u=float(curr.kps[best_kp].pt[0]),
                v=float(curr.kps[best_kp].pt[1]),
                color=(c.copy() if c is not None else None))):
            curr_fm[best_kp] = pt_idx
            _update_desc(pt, curr.desc[best_kp])
            used_kps.add(best_kp)
            added += 1
    if added > 0:
        logger.info(f"[proj-obs] 帧 {curr_idx}: 从 {len(neighbors)} KF "
                    f"投影追加 {added} 观测")
    return added


# =========================================================================
# PnP AddObservation（含角度检查）
# =========================================================================
def _add_pnp_observations(frame, idx_pairs, inlier_ids, map_points,
                          frames=None):
    if frame.feat_map is None or len(idx_pairs) == 0:
        return 0
    cfg = _cfg()
    min_angle = cfg.filter_min_tri_angle_deg
    added = 0
    frame_fm = frame.feat_map
    frame_colors = frame.colors
    frame_kps = frame.kps
    pts_np = np.asarray([map_points[p].xyz for p, _ in idx_pairs[:200]
                         if 0 <= p < len(map_points)])
    if len(pts_np) > 5:
        c_pts = pts_np.mean(axis=0)
        scene_scale = float(np.median(np.linalg.norm(
            pts_np - c_pts, axis=1))) + 1e-6
    else:
        scene_scale = 1.0
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
            if existing < len(map_points):
                d = float(np.linalg.norm(
                    map_points[existing].xyz - map_points[pt_idx].xyz))
                if d > 0.05 * scene_scale:
                    continue
        pt = map_points[pt_idx]
        if pt.obs_count >= MAX_OBS_PER_POINT:
            continue
        if not _check_obs_angle(pt, frame.idx, frames, min_angle):
            continue
        c = (frame_colors[kp_idx]
             if frame_colors is not None and kp_idx < len(frame_colors)
             else None)
        if pt.add_observation(Observation(
                frame_idx=frame.idx, kp_idx=kp_idx,
                u=float(frame_kps[kp_idx].pt[0]),
                v=float(frame_kps[kp_idx].pt[1]),
                color=(c.copy() if c is not None else None))):
            frame_fm[kp_idx] = pt_idx
            _update_desc(pt, frame.desc[kp_idx])
            added += 1
    return added


# =========================================================================
# PnP / 重定位
# =========================================================================
def _try_pnp_guided_once(frame, keyframes, map_points, frames, covis_graph,
                          focal, fy, cx, cy, reproj_thresh,
                          reproj_scale=1.0, ransac_scale=1.0,
                          guided_scale=1.0, loop_detector=None):
    cfg = _cfg()
    if len(keyframes) <= 1:
        return None, None, 0.0
    K = _build_K(focal, fy, cx, cy)
    K32 = K.astype(np.float32)
    anchor = keyframes[-1]
    neighbors = covis_graph.get_neighbors(anchor, cfg.pnp_window)
    # BoW 检索补充
    if loop_detector is not None and loop_detector.trained:
        bow_cands = loop_detector.retrieve_topk(
            frame.idx, frame.desc, k=cfg.pnp_window, min_score=0.1)
        for kf, _ in bow_cands:
            if kf not in neighbors:
                neighbors.append(kf)
    if len(neighbors) < 4:
        for kf in keyframes[-cfg.pnp_window:]:
            if kf not in neighbors:
                neighbors.append(kf)
    pts3d_g = np.empty((0, 3), dtype=np.float32)
    pts2d_g = np.empty((0, 2), dtype=np.float32)
    idx_g = []
    if frame.pose is not None:
        pts3d_g, pts2d_g, idx_g = _guided_match_3d2d(
            frame, neighbors, map_points, frames, K,
            proj_radius_px=cfg.guided_radius_px * guided_scale,
            max_desc_dist=update_dist_for_desc(frame.desc))
    if len(pts3d_g) < max(8, cfg.pnp_min_inliers):
        pts3d_c = []
        pts2d_c = []
        idx_c = []
        for kf in neighbors:
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
                    pts3d_c.append(map_points[pt_idx].xyz)
                    pts2d_c.append(frame.kps[m.trainIdx].pt)
                    idx_c.append((pt_idx, int(m.trainIdx)))
        if len(pts3d_c) >= 8 and len(pts3d_c) > len(pts3d_g):
            pts3d_g = np.asarray(pts3d_c, dtype=np.float32)
            pts2d_g = np.asarray(pts2d_c, dtype=np.float32)
            idx_g = idx_c
    if len(pts3d_g) < 8:
        return None, None, 0.0
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            np.ascontiguousarray(pts3d_g, dtype=np.float32),
            np.ascontiguousarray(pts2d_g, dtype=np.float32),
            K32, np.zeros(4, dtype=np.float32),
            iterationsCount=int(cfg.pnp_ransac_iters * ransac_scale),
            reprojectionError=reproj_thresh * reproj_scale,
            confidence=0.99)
    except cv2.error:
        return None, None, 0.0
    if not ok or inl is None or len(inl) < cfg.pnp_min_inliers:
        return None, None, 0.0
    n_inliers = int(len(inl))
    quality = float(n_inliers) / max(1, len(pts3d_g))
    inlier_ids = (inl.ravel().astype(np.int64) if inl.ndim > 1
                  else inl.astype(np.int64))
    if n_inliers >= 8:
        try:
            ok2, rvec2, tvec2 = cv2.solvePnP(
                np.ascontiguousarray(pts3d_g[inlier_ids], dtype=np.float32),
                np.ascontiguousarray(pts2d_g[inlier_ids], dtype=np.float32),
                K32, np.zeros(4, dtype=np.float32),
                rvec, tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                rvec, tvec = rvec2, tvec2
        except cv2.error:
            pass
    _add_pnp_observations(frame, idx_g, inlier_ids, map_points, frames)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    pose = CameraPose(R, t)
    centroid = pts3d_g.mean(axis=0)
    scene_scale = float(np.median(np.linalg.norm(pts3d_g - centroid,
                                                 axis=1))) + 1e-6
    cam_c = (-R.T @ t).flatten()
    if np.linalg.norm(cam_c - centroid) > 10.0 * scene_scale:
        return None, None, 0.0
    return pose, None, quality


def _try_pnp_guided(frame, keyframes, map_points, frames, covis_graph,
                    focal, fy, cx, cy, reproj_thresh, loop_detector=None):
    cfg = _cfg()
    n_kps = max(1, len(frame.kps))
    expected_inliers = max(cfg.pnp_min_inliers, int(n_kps * 0.10))

    pose, sim3, quality = _try_pnp_guided_once(
        frame, keyframes, map_points, frames, covis_graph,
        focal, fy, cx, cy, reproj_thresh,
        reproj_scale=1.0, ransac_scale=1.0, guided_scale=1.0,
        loop_detector=loop_detector)

    n_inliers_1 = int(quality * n_kps)
    if pose is not None and n_inliers_1 >= expected_inliers:
        return pose, sim3, quality

    pose2, sim3_2, quality2 = _try_pnp_guided_once(
        frame, keyframes, map_points, frames, covis_graph,
        focal, fy, cx, cy, reproj_thresh,
        reproj_scale=1.5, ransac_scale=1.5, guided_scale=1.5,
        loop_detector=loop_detector)
    if pose2 is not None:
        n_inliers_2 = int(quality2 * n_kps)
        if n_inliers_2 > n_inliers_1 * 1.3:
            logger.info(
                f"[PnP-retry] 帧 {frame.idx}: 第 2 轮 1.5× 成功"
                f"（内点 {n_inliers_1}→{n_inliers_2}，"
                f"期望≥{expected_inliers}）")
            return pose2, sim3_2, quality2
    if pose is None or (pose2 is not None and quality2 > quality):
        return pose2, sim3_2, quality2
    return pose, sim3, quality


def _try_global_relocalization(frame, keyframes, map_points, frames,
                               loop_detector, focal, fy, cx, cy,
                               reproj_thresh):
    cfg = _cfg()
    if len(keyframes) < 5:
        return None
    if frame.desc is None or len(frame.desc) == 0:
        return None
    cands = []
    if loop_detector.trained:
        cands = loop_detector.query(
            frame.idx, frame.desc, keyframes,
            skip_recent=0, min_score=LOOP_RELOC_SCORE,
            max_candidates=cfg.reloc_max_candidates * 3)
    if not cands:
        recent_kfs = [k for k in keyframes[-30:] if k != frame.idx]
        if recent_kfs:
            cands = [(k, 1.0) for k in recent_kfs]
    if not cands:
        return None
    K = _build_K(focal, fy, cx, cy).astype(np.float32)
    for kf, _score in cands:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        bound = np.where(kf_fm >= 0)[0]
        if len(bound) < 4:
            continue
        bd = frames[kf].desc[bound]
        nt, md, _ = get_metric(bd)
        matches = _match_ratio_absolute(bd, frame.desc, nt, md)
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
                iterationsCount=150, reprojectionError=reproj_thresh,
                confidence=0.99)
        except cv2.error:
            continue
        if not ok or inl is None or len(inl) < cfg.reloc_min_inliers:
            continue
        R, _ = cv2.Rodrigues(rvec)
        return CameraPose(R, tvec.reshape(3, 1)), kf
    return None


def _global_match_expand(curr_idx, keyframes, frames, loop_detector,
                         map_points, focal, fy, cx, cy, reproj_thresh):
    frame = frames[curr_idx]
    if frame.pose is None or frame.desc is None or len(frame.desc) == 0:
        return 0, 0
    if not loop_detector.trained:
        return 0, 0
    cands = loop_detector.query(
        curr_idx, frame.desc, keyframes,
        skip_recent=0, min_score=LOOP_MIN_SCORE,
        max_candidates=GLOBAL_MATCH_TOP_K)
    if not cands:
        return 0, 0
    K = _build_K(focal, fy, cx, cy)
    P_curr = K @ np.asarray(frame.pose.RT[:3], dtype=np.float64)
    new_obs_count = 0
    new_pt_count = 0
    curr_feat_map = frame.feat_map
    curr_colors = frame.colors
    err_thresh = reproj_thresh * MAX_REPROJ_ERROR
    for kf, _score in cands:
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
        bound = np.where(kf_fm >= 0)[0]
        if len(bound) < 4:
            continue
        bd = kf_frame.desc[bound]
        nt, md, _ = get_metric(bd)
        matches = _match_ratio_absolute(bd, frame.desc, nt, md)
        if len(matches) < GLOBAL_MATCH_MIN_INLIERS:
            continue
        pts_a = np.array([kf_frame.kps[int(bound[m.queryIdx])].pt
                          for m in matches], dtype=np.float32)
        pts_b = np.array([frame.kps[m.trainIdx].pt for m in matches],
                         dtype=np.float32)
        vmask, _, _ = _verify_matches_fh(
            pts_a, pts_b, reproj_thresh * 1.5, GLOBAL_MATCH_MIN_INLIERS)
        if vmask is None:
            continue
        P_kf = K @ np.asarray(kf_frame.pose.RT[:3], dtype=np.float64)
        reuse_obs = []
        new_tri = []
        for mi, m in enumerate(matches):
            if not vmask[mi]:
                continue
            kf_kp = int(bound[m.queryIdx])
            curr_kp = int(m.trainIdx)
            if curr_kp >= len(curr_feat_map):
                continue
            if curr_feat_map[curr_kp] >= 0:
                continue
            pt_idx = int(kf_fm[kf_kp])
            if 0 <= pt_idx < len(map_points):
                xyz = map_points[pt_idx].xyz
                pt_cam = frame.pose.R @ xyz.reshape(3, 1) + frame.pose.t
                d = float(pt_cam[2, 0])
                if d <= 1e-6:
                    continue
                u = focal * float(pt_cam[0, 0]) / d + cx
                v = fy * float(pt_cam[1, 0]) / d + cy
                err = float(np.hypot(u - frame.kps[curr_kp].pt[0],
                                     v - frame.kps[curr_kp].pt[1]))
                if err > err_thresh:
                    continue
                reuse_obs.append((mi, pt_idx, curr_kp))
            else:
                new_tri.append(mi)
        for _mi, pt_idx, curr_kp in reuse_obs:
            pt = map_points[pt_idx]
            if pt.obs_count >= MAX_OBS_PER_POINT:
                continue
            c = (curr_colors[curr_kp] if curr_colors is not None
                 and curr_kp < len(curr_colors) else None)
            if pt.add_observation(Observation(
                    frame_idx=curr_idx, kp_idx=curr_kp,
                    u=float(frame.kps[curr_kp].pt[0]),
                    v=float(frame.kps[curr_kp].pt[1]),
                    color=(c.copy() if c is not None else None))):
                curr_feat_map[curr_kp] = pt_idx
                _update_desc(pt, frame.desc[curr_kp])
                new_obs_count += 1
        if new_tri:
            pts_a_t = np.array(
                [kf_frame.kps[int(bound[matches[mi].queryIdx])].pt
                 for mi in new_tri], dtype=np.float64)
            pts_b_t = np.array(
                [frame.kps[matches[mi].trainIdx].pt
                 for mi in new_tri], dtype=np.float64)
            cams = np.asarray([kf_frame.pose.center, frame.pose.center],
                              dtype=np.float64)
            Xs, valid = _triangulate_batch(
                [pts_a_t, pts_b_t], [P_kf, P_curr], cams,
                _cfg().filter_min_tri_angle_deg, err_thresh)
            for k, mi in enumerate(new_tri):
                if not valid[k]:
                    continue
                m = matches[mi]
                kf_kp = int(bound[m.queryIdx])
                curr_kp = int(m.trainIdx)
                pt_idx = len(map_points)
                c_a = (kf_frame.colors[kf_kp]
                       if kf_frame.colors is not None
                       and kf_kp < len(kf_frame.colors) else None)
                pt = MapPoint(
                    idx=pt_idx, xyz=Xs[k].astype(np.float32),
                    desc=kf_frame.desc[kf_kp].copy())
                pt.add_observation(Observation(
                    frame_idx=kf, kp_idx=kf_kp,
                    u=float(kf_frame.kps[kf_kp].pt[0]),
                    v=float(kf_frame.kps[kf_kp].pt[1]),
                    color=(c_a.copy() if c_a is not None else None)))
                c_b = (curr_colors[curr_kp]
                       if curr_colors is not None
                       and curr_kp < len(curr_colors) else None)
                pt.add_observation(Observation(
                    frame_idx=curr_idx, kp_idx=curr_kp,
                    u=float(frame.kps[curr_kp].pt[0]),
                    v=float(frame.kps[curr_kp].pt[1]),
                    color=(c_b.copy() if c_b is not None else None)))
                map_points.append(pt)
                curr_feat_map[curr_kp] = pt_idx
                new_pt_count += 1
    if new_obs_count > 0 or new_pt_count > 0:
        logger.info(
            f"[global-match] 帧 {curr_idx}："
            f"新增观测 {new_obs_count}，新点 {new_pt_count}")
    return new_obs_count, new_pt_count


# =========================================================================
# 回环检测（含序列一致性）
# =========================================================================
def _check_seq_consistency(curr_idx, cand_kf, keyframes, loop_detector,
                            k=LOOP_SEQ_CHECK_K,
                            thresh=LOOP_SEQ_CHECK_THRESH):
    kf_pos = {kf: i for i, kf in enumerate(keyframes)}
    if cand_kf not in kf_pos or curr_idx not in kf_pos:
        return 0
    i, j = kf_pos[cand_kf], kf_pos[curr_idx]
    n_ok = 0
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            ia, jb = i + di, j + dj
            if not (0 <= ia < len(keyframes) and 0 <= jb < len(keyframes)):
                continue
            a, b = keyframes[ia], keyframes[jb]
            if a == b:
                continue
            ha = loop_detector.hists.get(a)
            hb = loop_detector.hists.get(b)
            if ha is None or hb is None:
                continue
            s = float(np.dot(ha, hb)
                      / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-12))
            if s > thresh:
                n_ok += 1
    return n_ok


def _detect_and_verify_loops(curr_idx, keyframes, frames, loop_detector,
                             map_points, focal, fy, cx, cy, reproj_thresh):
    cfg = _cfg()
    frame = frames[curr_idx]
    if frame.desc is None or len(frame.desc) == 0:
        return []
    cands = loop_detector.query(
        curr_idx, frame.desc, keyframes,
        skip_recent=LOOP_SKIP_RECENT,
        min_score=LOOP_MIN_SCORE,
        max_candidates=cfg.loop_max_candidates * 3)
    min_gap = max(40, cfg.loop_min_gap)
    results = []
    n_rej_gap = 0
    n_rej_none = 0
    n_rej_match = 0
    n_rej_geom = 0
    n_rej_pnp = 0
    n_rej_ratio = 0
    n_rej_seq = 0
    K64 = _build_K(focal, fy, cx, cy)
    K32 = K64.astype(np.float32)
    for kf, score in cands:
        if abs(kf - curr_idx) < min_gap:
            n_rej_gap += 1
            continue
        if kf not in frames or frames[kf].desc is None:
            n_rej_none += 1
            continue
        kf_frame = frames[kf]
        nt, md, _ = get_metric(kf_frame.desc)
        matches = _match_ratio_absolute(kf_frame.desc, frame.desc, nt, md)
        if len(matches) < 15:
            n_rej_match += 1
            continue
        pts_a = np.array([kf_frame.kps[m.queryIdx].pt for m in matches],
                         dtype=np.float32)
        pts_b = np.array([frame.kps[m.trainIdx].pt for m in matches],
                         dtype=np.float32)
        vmask, _mt, _mdl = _verify_matches_fh(
            pts_a, pts_b, reproj_thresh * 1.5, 10)
        if vmask is None or int(vmask.sum()) < 10:
            n_rej_geom += 1
            continue
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
            n_rej_pnp += 1
            continue
        # 序列一致性
        if _check_seq_consistency(curr_idx, kf, keyframes,
                                   loop_detector) < LOOP_MIN_SEQ_CONSISTENCY:
            n_rej_seq += 1
            continue
        pts3d = np.asarray(pts3d_list, dtype=np.float32)
        pts2d = np.asarray(pts2d_list, dtype=np.float32)
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                pts3d, pts2d, K32, np.zeros(4, dtype=np.float32),
                iterationsCount=cfg.pnp_ransac_iters,
                reprojectionError=reproj_thresh,
                confidence=0.99)
        except cv2.error:
            n_rej_pnp += 1
            continue
        if not ok or inl is None or len(inl) < cfg.loop_min_inliers:
            n_rej_pnp += 1
            continue
        inlier_ratio = len(inl) / max(len(pts3d_list), 1)
        if inlier_ratio < LOOP_MIN_INLIER_RATIO:
            n_rej_ratio += 1
            continue
        R, _ = cv2.Rodrigues(rvec)
        pose_loop = CameraPose(R, tvec.reshape(3, 1))
        kf_pose = kf_frame.pose
        s = 1.0
        if kf_pose is not None and len(pts3d_list) >= 10:
            pts_np = np.array(pts3d_list, dtype=np.float64)
            X_in_kf = (kf_pose.R.astype(np.float64) @ pts_np.T).T \
                      + kf_pose.t.flatten()
            X_in_curr = (pose_loop.R.astype(np.float64) @ pts_np.T).T \
                        + pose_loop.t.flatten()
            sim3 = estimate_sim3(X_in_kf, X_in_curr)
            if sim3 is not None:
                s = float(np.clip(sim3.s, 0.5, 2.0))
                if abs(s - 1.0) < 0.03:
                    s = 1.0
        if kf_pose is not None:
            T_ij = _relative_pose(kf_pose, pose_loop)
        else:
            T_ij = CameraPose(pose_loop.R.copy(), pose_loop.t.copy())
        S_rel = Sim3(s, T_ij.R, T_ij.t)
        if s != 1.0:
            logger.info(f"[回环] {kf}↔{curr_idx} 尺度比 s={s:.4f}")
        results.append((kf, S_rel))
    if len(cands) > 0 and len(results) == 0:
        logger.info(
            f"[loop-dbg] 帧 {curr_idx}: {len(cands)} 候选全被拒"
            f"（gap={n_rej_gap}, 匹配数不足={n_rej_match}, "
            f"几何={n_rej_geom}, PnP={n_rej_pnp}, "
            f"内点率不足={n_rej_ratio}, 序列不一致={n_rej_seq}, "
            f"None描述子={n_rej_none}）")
    return results


# =========================================================================
# 初始化（COLMAP 风格评分）
# =========================================================================
def _score_init_pair(n_inliers, med_parallax_deg, n_matches):
    return (float(n_inliers)
            * (1.0 + np.log1p(max(med_parallax_deg, 0.0)))
            * (1.0 + 0.1 * np.log1p(max(n_matches, 0))))


def _select_init_pair(frames, K, ransac_thresh):
    cfg = _cfg()
    n = len(frames)
    search_limit = max(2, min(n, int(n * cfg.init_search_ratio) + 1))
    pairs = []
    for i in range(0, search_limit):
        j_end = min(n, i + cfg.init_candidate_window)
        for j in range(i + 1, j_end):
            pairs.append((i, j))
            if len(pairs) >= cfg.init_max_pairs:
                break
        if len(pairs) >= cfg.init_max_pairs:
            break
    scored = []
    for (i, j) in pairs:
        a, b = frames[i], frames[j]
        if a.desc is None or b.desc is None:
            continue
        if len(a.desc) < 50 or len(b.desc) < 50:
            continue
        nt, md, _ = get_metric(a.desc)
        n_use = min(500, len(a.desc), len(b.desc))
        try:
            quick = _match_ratio_absolute(
                a.desc[:n_use], b.desc[:n_use], nt, md)
        except Exception:
            continue
        scored.append((i, j, len(quick)))
    scored.sort(key=lambda x: -x[2])
    top_k = min(len(scored), 40)
    pairs = [(i, j) for i, j, cnt in scored[:top_k] if cnt >= 20]
    if not pairs:
        pairs = [(i, j) for i, j, _ in scored[:top_k]]
    logger.info(f"[init] 预筛后 {len(pairs)} 对候选（原 {len(scored)} 对）")
    attempts = [
        (cfg.init_min_inliers, cfg.init_min_parallax_deg, 0.30),
        (max(40, cfg.init_min_inliers // 2),
         cfg.init_min_parallax_deg / 1.5, 0.20),
        (max(25, cfg.init_min_inliers // 3),
         cfg.init_min_parallax_deg / 2.0, 0.15),
    ]
    best = None
    best_score = -1.0
    for attempt_idx, (min_inl, min_parallax, min_inl_ratio) in enumerate(attempts):
        logger.info(f"[init] 第 {attempt_idx+1}/{len(attempts)} 轮："
                    f"内点≥{min_inl}，视差≥{min_parallax:.2f}°，"
                    f"内点率≥{min_inl_ratio:.0%}")
        n_tried = 0
        n_success = 0
        for (i, j) in pairs:
            n_tried += 1
            a, b = frames[i], frames[j]
            if a.desc is None or b.desc is None:
                continue
            norm_type, md, _ = get_metric(a.desc)
            matches = _match_ratio_absolute(a.desc, b.desc, norm_type, md)
            n_matches = len(matches)
            if n_matches < max(min_inl, 8):
                continue
            pts_a = np.array([a.kps[m.queryIdx].pt for m in matches],
                             dtype=np.float32)
            pts_b = np.array([b.kps[m.trainIdx].pt for m in matches],
                             dtype=np.float32)
            mask_fh, model_type, model = _verify_matches_fh(
                pts_a, pts_b, ransac_thresh, min_inl)
            if mask_fh is None:
                continue
            n_inl_geom = int(mask_fh.sum())
            if n_inl_geom / max(n_matches, 1) < min_inl_ratio:
                continue
            E, mask_e = _find_essential(pts_a, pts_b, K, ransac_thresh)
            pose_b_local = None
            pose_inl_local = None
            med_parallax_local = 0.0
            if E is not None and mask_e is not None:
                mask_e = mask_e.ravel().astype(bool)
                if int(mask_e.sum()) >= min_inl:
                    _, R_rel, t_rel, mask_pose = _recover_pose(
                        E, pts_a, pts_b, K)
                    if mask_pose is not None:
                        pose_inl_local = mask_pose.ravel().astype(bool)
                        if int(pose_inl_local.sum()) >= min_inl:
                            ok, med = _check_baseline_sufficient(
                                pts_a, pts_b, R_rel, t_rel, K,
                                pose_inl_local,
                                min_pos_ratio=BASELINE_MIN_POS_RATIO,
                                min_angle_deg=min_parallax)
                            if ok and med >= min_parallax:
                                t_norm = float(np.linalg.norm(t_rel))
                                if t_norm > 1e-12:
                                    t_unit = t_rel / t_norm
                                    pose_b_local = CameraPose(
                                        R_rel, t_unit.reshape(3, 1))
                                    med_parallax_local = med
            if pose_b_local is None or pose_inl_local is None:
                continue
            if int(pose_inl_local.sum()) < min_inl:
                continue
            n_success += 1
            score = _score_init_pair(
                int(pose_inl_local.sum()), med_parallax_local, n_matches)
            if score > best_score:
                best_score = score
                best = (i, j, pose_b_local, matches, pose_inl_local,
                        int(pose_inl_local.sum()))
        logger.info(f"[init] 第 {attempt_idx+1} 轮：尝试 {n_tried} 对，"
                    f"几何成功 {n_success}")
        if best is not None:
            break
    logger.info(f"[init] 最佳={'有' if best else '无'}")
    if best is None:
        return None
    i, j, pose_b, matches, mask, _ = best
    frames[i].pose = CameraPose(np.eye(3, dtype=np.float32),
                                np.zeros((3, 1), dtype=np.float32))
    frames[i].status = FrameStatus.KEYFRAME
    frames[i].is_keyframe = True
    frames[j].pose = pose_b
    frames[j].status = FrameStatus.KEYFRAME
    frames[j].is_keyframe = True
    return best


# =========================================================================
# 局部 BA 迭代精化
# =========================================================================
def _local_ba_refinement(window, map_points, frames, focal, fy, cx, cy,
                         reproj_thresh, image_size, focal_init,
                         covis_graph, max_rounds=None):
    cfg = _cfg()
    if max_rounds is None:
        max_rounds = cfg.ba_local_refinements
    prev_rms = None
    success_any = False
    for rnd in range(max_rounds):
        ba_state = {"success": False}
        focal, fy, cx, cy = bundle_adjustment(
            window, map_points, frames,
            focal, fy, cx, cy,
            optimize_points=True,
            reproj_thresh=reproj_thresh,
            image_size=image_size,
            is_global=False,
            focal_init=focal_init,
            optimize_principal=False,
            _state=ba_state)
        if not ba_state["success"]:
            break
        success_any = True
        K_now = _build_K(focal, fy, cx, cy)
        _retriangulate_multiview(map_points, frames, K_now,
                                 min_obs=MULTIVIEW_MIN_OBS)
        _complete_tracks(window, frames, map_points, K_now,
                         reproj_thresh,
                         max_transitivity=cfg.complete_max_transitivity)
        poses_by_frame = {f: fr.pose for f, fr in frames.items()}
        mean_err, valid_count, neg_ratio = _compute_point_errors(
            map_points, poses_by_frame, focal, fy, cx, cy)
        valid_mask = (valid_count >= MIN_OBSERVATIONS) & (neg_ratio < 0.5)
        if np.any(valid_mask):
            rms = float(np.median(mean_err[valid_mask]))
            if prev_rms is not None and rms >= prev_rms * 0.98:
                break
            prev_rms = rms
    return focal, fy, cx, cy, success_any


# =========================================================================
# 主流水线
# =========================================================================
def estimate_poses(
    frame_paths: List[str],
    feature_type: str,
    *,
    focal_guess: Optional[float] = None,
    aspect_ratio: float = 1.0,
    dist_coeffs: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    enable_loop: bool = True,
    enable_pgo: bool = True,
    enable_global_matching: bool = False,
) -> SfMResult:
    global _RUNTIME_CFG
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    if len(frame_paths) < 2:
        raise ValueError("至少需要 2 帧")
    ft = feature_type.lower()
    if ft not in ("sift", "rootsift", "orb"):
        raise ValueError(
            f"不支持的 feature_type={feature_type!r}，"
            f"可选：'sift' / 'rootsift' / 'orb'")
    t_start = time.time()
    first_img = get_image(frame_paths[0])
    if first_img is None:
        raise FileNotFoundError(f"无法读取图像：{frame_paths[0]}")
    h0, w0 = first_img.shape[:2]
    del first_img
    has_intrinsics = camera_matrix is not None
    has_distortion = dist_coeffs is not None
    _RUNTIME_CFG = derive_auto_config(
        n_frames=len(frame_paths),
        img_shape=(h0, w0),
        feature_type=feature_type,
        has_intrinsics=has_intrinsics,
        has_distortion=has_distortion)
    cfg = _RUNTIME_CFG
    logger.info(f"=== 配置（{w0}×{h0}，{len(frame_paths)} 帧，"
                f"用户指定 {feature_type}，"
                f"内参={'有' if has_intrinsics else '无'}）===")
    logger.info("=== 提取特征（多线程）===")
    frames_list, img_shape, feature_type = extract_features(
        frame_paths, feature_type, dist_coeffs, camera_matrix)
    frames = {f.idx: f for f in frames_list}
    h, w = img_shape
    if camera_matrix is not None:
        K_in = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        fx = float(K_in[0, 0])
        fy = float(K_in[1, 1])
        cx = float(K_in[0, 2])
        cy = float(K_in[1, 2])
    else:
        cx, cy = w / 2.0, h / 2.0
        fx = focal_guess if focal_guess is not None else max(w, h) * 1.2
        fy = fx * aspect_ratio
    focal0 = float(fx)
    fy0 = float(fy)
    focal_init = focal0
    image_size = max(w, h)
    reproj_thresh = cfg.reproj_thresh
    ransac_thresh = cfg.ransac_thresh
    triang_thresh = cfg.triang_thresh
    norm_type, match_dist, _ = get_metric(
        next((f.desc for f in frames_list
              if f.desc is not None and len(f.desc)), None))
    map_points = []
    keyframes = []
    covis_graph = CovisibilityGraph()
    loop_detector = LoopDetector(vocab_size=cfg.loop_vocab_size)
    loop_closures = []
    loop_constraints = []
    K_init = _build_K(focal0, fy0, cx, cy)
    init = _select_init_pair(frames, K_init, ransac_thresh)
    if init is None:
        raise RuntimeError("无法初始化 SfM：未找到合适的初始帧对。")
    idx_a, idx_b, pose_b, matches_init, mask_init, _ = init
    logger.info(f"初始化成功：帧 {idx_a} + {idx_b}，内点={int(mask_init.sum())}")
    keyframes.append(idx_a)
    keyframes.append(idx_b)
    covis_graph.add_keyframe(idx_a)
    covis_graph.add_keyframe(idx_b)
    _triangulate_from_pair(
        idx_b, idx_a, matches_init, mask_init,
        frames, map_points, focal0, fy0, cx, cy, triang_thresh)
    kf_since_ba = 0
    kf_since_multiview = 0
    kf_since_global = 0
    kf_since_global_ba = 0
    last_global_ba_n_kf = 0
    last_global_ba_n_pts = 0

    for i in range(len(frame_paths)):
        if i == idx_a or i == idx_b:
            continue
        t_frame = time.time()
        frame = frames[i]
        if len(frame.kps) < cfg.min_features:
            logger.warning(f"[帧 {i}] 特征过少（{len(frame.kps)}），标记 INVALID。")
            frame.status = FrameStatus.INVALID
            continue
        prev_idx = None
        for k in range(i - 1, -1, -1):
            if frames[k].pose is not None:
                prev_idx = k
                break
        has_pose = False
        prev_matches_count = 0
        if prev_idx is not None:
            prev = frames[prev_idx]
            matches = _match_ratio_absolute(prev.desc, frame.desc,
                                            norm_type, match_dist)
            prev_matches_count = len(matches)
            if len(matches) >= cfg.init_min_inliers:
                pts_prev = np.array(
                    [prev.kps[m.queryIdx].pt for m in matches],
                    dtype=np.float32)
                pts_curr = np.array(
                    [frame.kps[m.trainIdx].pt for m in matches],
                    dtype=np.float32)
                E, mask = _find_essential(pts_prev, pts_curr, K_init,
                                          ransac_thresh)
                dynamic_min_e = min(cfg.init_min_inliers,
                                    max(15, int(0.20 * len(matches))))
                if E is not None and mask is not None and \
                        int(mask.sum()) >= dynamic_min_e:
                    _, R_rel, t_rel, mask_pose = _recover_pose(
                        E, pts_prev, pts_curr, K_init)
                    if mask_pose is not None:
                        pose_inlier = mask_pose.ravel().astype(bool)
                        dynamic_min = min(cfg.init_min_inliers,
                                          max(15, int(0.15 * len(matches))))
                        if int(pose_inlier.sum()) >= dynamic_min:
                            ok, _ = _check_baseline_sufficient(
                                pts_prev, pts_curr, R_rel, t_rel, K_init,
                                pose_inlier,
                                min_angle_deg=cfg.init_min_parallax_deg)
                            if ok:
                                R_curr = R_rel @ prev.pose.R
                                t_curr = R_rel @ prev.pose.t + t_rel
                                frame.pose = CameraPose(R_curr, t_curr)
                                frame.status = FrameStatus.TRACKED
                                frame.track_quality = float(
                                    pose_inlier.sum() / len(matches))
                                has_pose = True
                                _triangulate_from_pair(
                                    i, prev_idx, matches, pose_inlier,
                                    frames, map_points,
                                    focal0, fy0, cx, cy, triang_thresh)
        if len(keyframes) > 1:
            pose_pnp, _, quality_pnp = _try_pnp_guided(
                frame, keyframes, map_points, frames, covis_graph,
                focal0, fy0, cx, cy, reproj_thresh,
                loop_detector=loop_detector)
            if pose_pnp is not None:
                if has_pose and frame.pose is not None:
                    dc = np.linalg.norm(pose_pnp.center - frame.pose.center)
                    scene = 1.0
                    recent = (np.array([p.xyz for p in map_points[-200:]],
                                       dtype=np.float32) if map_points else None)
                    if recent is not None and len(recent) > 5:
                        c = recent.mean(axis=0)
                        scene = float(np.median(np.linalg.norm(
                            recent - c, axis=1))) + 1e-6
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
                focal0, fy0, cx, cy, reproj_thresh)
            if reloc_result is not None:
                pose_reloc, reloc_kf = reloc_result
                frame.pose = pose_reloc
                frame.status = FrameStatus.RELOCATED
                logger.info(f"[帧 {i}] 全局重定位成功（匹配 KF={reloc_kf}）。")
                has_pose = True
        if not has_pose or frame.pose is None:
            frame.status = FrameStatus.LOST
            if i % 5 == 0:
                logger.info(
                    f"[帧 {i}] LOST：prev={prev_idx}, "
                    f"desc_matches={prev_matches_count}, "
                    f"kps={len(frame.kps)}, track_q={frame.track_quality:.2f}")
            continue
        last_kf = keyframes[-1]
        last_kf_pose = frames[last_kf].pose
        R_delta = frame.pose.R @ last_kf_pose.R.T
        angle = _rotation_angle_deg(R_delta)
        delta_c = frame.pose.center - last_kf_pose.center
        if len(map_points) >= 5:
            recent = np.array([p.xyz for p in map_points[-200:]],
                              dtype=np.float32)
            c_r = recent.mean(axis=0)
            scene_ref = float(np.median(np.linalg.norm(
                recent - c_r, axis=1))) + 1e-6
        else:
            scene_ref = 1e-6
        trans_ratio = float(np.linalg.norm(delta_c) / scene_ref)
        covis_ratio = 1.0
        if prev_idx is not None:
            matches_for_covis = _match_ratio_absolute(
                frames[last_kf].desc, frame.desc, norm_type, match_dist)
            covis_ratio = _compute_covis_ratio(i, keyframes,
                                               matches_for_covis, frames)
        is_kf = (angle > KF_ANGLE_DEG or trans_ratio > KF_TRANS_RATIO
                 or covis_ratio < KF_COVIS_RATIO)
        if i - last_kf < MIN_KF_FRAME_GAP:
            is_kf = False
        reloc_kf_for_loop = None
        if reloc_kf is not None and abs(reloc_kf - i) >= cfg.loop_min_gap:
            reloc_kf_for_loop = reloc_kf
            is_kf = True
        if is_kf and has_pose and frame.pose is not None and len(map_points) > 20:
            _project_and_add_observations(
                i, keyframes, frames, map_points,
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
                if (frames[reloc_kf_for_loop].pose is not None
                        and frame.pose is not None):
                    T_meas = _relative_pose(
                        frames[reloc_kf_for_loop].pose, frame.pose)
                    loop_constraints.append(
                        (reloc_kf_for_loop, i, T_meas))
                logger.info(f"[重定位→回环] 帧 {reloc_kf_for_loop} ↔ {i}")
            covis_graph.update_from_frame(i, map_points, frames)
            if len(keyframes) >= 3:
                _triangulate_next_best_view(
                    i, keyframes, frames, map_points,
                    focal0, fy0, cx, cy, triang_thresh, top_k=8)
            if len(keyframes) >= 3 and len(map_points) > 30:
                K_now = _build_K(focal0, fy0, cx, cy)
                _complete_tracks(
                    keyframes[-10:], frames, map_points, K_now,
                    reproj_thresh,
                    max_transitivity=cfg.complete_max_transitivity)
            if enable_loop and len(keyframes) >= 15:
                if not loop_detector.trained:
                    descs = [frames[k].desc for k in keyframes]
                    loop_detector.train(descs, norm_type)
                    for k in keyframes:
                        loop_detector.add(k, frames[k].desc)
                else:
                    loop_detector.add(i, frame.desc)
                loops = _detect_and_verify_loops(
                    i, keyframes, frames, loop_detector, map_points,
                    focal0, fy0, cx, cy, reproj_thresh)
                for cand_kf, S_rel in loops:
                    loop_closures.append((cand_kf, i))
                    loop_constraints.append(
                        (cand_kf, i, CameraPose(S_rel.R.copy(),
                                                S_rel.t.copy())))
                    logger.info(f"[回环] 帧 {cand_kf} ↔ {i} (s={S_rel.s:.4f})")
            if (enable_global_matching
                    and len(keyframes) >= 5
                    and kf_since_global >= GLOBAL_MATCH_TRIGGER_EVERY_KF
                    and loop_detector.trained):
                _global_match_expand(
                    i, keyframes, frames, loop_detector, map_points,
                    focal0, fy0, cx, cy, reproj_thresh)
                kf_since_global = 0
            if (kf_since_ba >= BA_LOCAL_TRIGGER_EVERY
                    and len(keyframes) >= 3):
                window = covis_graph.get_neighbors(
                    i, cfg.ba_local_covis_neighbors)
                if len(window) >= 3:
                    focal0, fy0, cx, cy, ok_any = _local_ba_refinement(
                        window, map_points, frames,
                        focal0, fy0, cx, cy,
                        reproj_thresh, image_size, focal_init,
                        covis_graph)
                    if ok_any:
                        kf_since_ba = 0
                    else:
                        kf_since_ba = BA_LOCAL_TRIGGER_EVERY - 1
                        logger.info("[BA] 局部 BA 精化失败，下一帧重试。")
                else:
                    kf_since_ba = 0
            n_kf_now = len([k for k in keyframes
                            if frames[k].pose is not None])
            n_pts_now = len(map_points)
            trigger_global_ba = False
            if kf_since_global_ba >= GLOBAL_BA_EVERY_KF:
                trigger_global_ba = True
            if (last_global_ba_n_kf > 0
                    and n_kf_now >= last_global_ba_n_kf *
                    cfg.ba_global_image_growth):
                trigger_global_ba = True
            if (last_global_ba_n_pts > 0
                    and n_pts_now >= last_global_ba_n_pts *
                    cfg.ba_global_point_growth):
                trigger_global_ba = True
            if trigger_global_ba and n_kf_now >= 15:
                all_registered_ba = [
                    k for k in keyframes
                    if frames[k].pose is not None]
                logger.info(
                    f"[全局精化] 注册 {len(all_registered_ba)} 帧 / "
                    f"{n_pts_now} 点，触发周期性全局 BA")
                ba_state_p = {"success": False}
                (focal0, fy0, cx, cy) = bundle_adjustment(
                    all_registered_ba, map_points, frames,
                    focal0, fy0, cx, cy,
                    optimize_points=True,
                    reproj_thresh=reproj_thresh,
                    image_size=image_size,
                    is_global=True,
                    focal_init=focal_init,
                    optimize_principal=False,
                    _state=ba_state_p,
                    max_iter=20)
                if ba_state_p["success"]:
                    kf_since_global_ba = 0
                    last_global_ba_n_kf = n_kf_now
                    last_global_ba_n_pts = n_pts_now
                else:
                    logger.warning(
                        "[全局精化] 周期性 BA 失败，下轮重试。")
        if (kf_since_multiview >= cfg.multiview_trigger_every_kf
                and len(map_points) > 50):
            K_now = _build_K(focal0, fy0, cx, cy)
            _retriangulate_multiview(map_points, frames, K_now,
                                     min_obs=MULTIVIEW_MIN_OBS)
            _merge_map_points(map_points, frames, norm_type)
            kf_since_multiview = 0
        if i % 200 == 0 and len(map_points) > 100:
            prune_map_points(map_points, frames, reproj_thresh,
                             focal0, fy0, cx, cy)
        if (i == 1 or i % PROGRESS_LOG_EVERY == 0
                or i == len(frame_paths) - 1):
            dt = time.time() - t_frame
            logger.info(
                f"[帧 {i}/{len(frame_paths)-1}] 状态={frame.status} "
                f"跟踪质量={frame.track_quality:.2f} 点数={len(map_points)} "
                f"关键帧数={len(keyframes)} 耗时={dt:.2f}s")

    K_final = _build_K(focal0, fy0, cx, cy)
    _retriangulate_multiview(map_points, frames, K_final,
                             min_obs=MULTIVIEW_MIN_OBS)
    if len(keyframes) >= 3 and len(map_points) > 50:
        logger.info("=== 全局 BA ===")
        all_registered = [k for k in keyframes
                          if frames[k].pose is not None]
        ba_state_g = {"success": False}
        (focal0, fy0, cx, cy) = bundle_adjustment(
            all_registered, map_points, frames,
            focal0, fy0, cx, cy,
            optimize_points=True,
            reproj_thresh=reproj_thresh,
            image_size=image_size,
            is_global=True,
            focal_init=focal_init,
            optimize_principal=False,
            _state=ba_state_g)
        if not ba_state_g["success"]:
            logger.warning("[全局 BA] 失败，跳过。")
        K_final = _build_K(focal0, fy0, cx, cy)
    valid_kf_set = set(keyframes)
    sorted_kfs = sorted(keyframes)
    pgo_edges = []
    pgo_weights = []
    for a, b in zip(sorted_kfs[:-1], sorted_kfs[1:]):
        if frames[a].pose is None or frames[b].pose is None:
            continue
        T_rel = _relative_pose(frames[a].pose, frames[b].pose)
        pgo_edges.append((a, b, T_rel))
        pgo_weights.append(PGO_SEQ_WEIGHT)
    if len(map_points) >= 5:
        recent_xyz = np.array([mp.xyz for mp in map_points[-200:]],
                              dtype=np.float32)
        c_r = recent_xyz.mean(axis=0)
        scale_pgo = float(np.median(np.linalg.norm(
            recent_xyz - c_r, axis=1))) + 1e-6
    else:
        scale_pgo = 1.0
    valid_loop_pairs = []
    n_pgo_rejected = 0
    n_pgo_accepted = 0
    for a, b, T_meas in loop_constraints:
        if a not in valid_kf_set or b not in valid_kf_set:
            continue
        if abs(b - a) <= 1:
            continue
        if (frames[a].pose is not None and frames[b].pose is not None):
            T_cur = _relative_pose(frames[a].pose, frames[b].pose)
            T_meas_mat = _pose_to_T(T_meas)
            T_cur_mat = _pose_to_T(T_cur)
            T_err = T_meas_mat @ np.linalg.inv(T_cur_mat)
            err_vec = _se3_log(T_err)
            trans_err = float(np.linalg.norm(err_vec[:3]))
            rot_err = float(np.linalg.norm(err_vec[3:]))
            if (trans_err > PGO_OUTLIER_REL_POSE_THRESH * scale_pgo
                    or rot_err > 0.5):
                n_pgo_rejected += 1
                continue
        pgo_edges.append((a, b, T_meas))
        n_inl_est = max(1, covis_graph.graph.get(b, {}).get(a, 10))
        w = float(np.log1p(n_inl_est))
        pgo_weights.append(max(PGO_LOOP_WEIGHT, w))
        valid_loop_pairs.append((a, b))
        n_pgo_accepted += 1
    if n_pgo_rejected > 0 or n_pgo_accepted > 0:
        logger.info(
            f"[PGO] 回环约束：接受 {n_pgo_accepted} 条，"
            f"剔除 {n_pgo_rejected} 条离群")
    edge_weights_list = pgo_weights
    loop_closures = valid_loop_pairs
    if enable_pgo and len(pgo_edges) >= 3:
        logger.info("=== 位姿图优化 ===")
        pre_pgo_poses = {k: frames[k].pose.copy() for k in keyframes
                         if frames[k].pose is not None}
        poses_dict = {k: frames[k].pose for k in keyframes
                      if frames[k].pose is not None}
        weights = {i: w for i, w in enumerate(edge_weights_list)}
        new_poses = pose_graph_optimize(keyframes, poses_dict, pgo_edges,
                                        edge_weights=weights)
        pgo_ok = True
        max_dc = 0.0
        for k, p in new_poses.items():
            old_p = pre_pgo_poses.get(k)
            if old_p is None or p is None:
                continue
            dc = float(np.linalg.norm(p.center - old_p.center))
            if dc > max_dc:
                max_dc = dc
        if max_dc > PGO_MAX_DIVERGENCE_RATIO * scale_pgo:
            pgo_ok = False
            logger.warning(
                f"[PGO] 检测到位姿发散（max_dc={max_dc:.3f}），放弃 PGO。")
        if pgo_ok:
            for k, p in new_poses.items():
                frames[k].pose = p
    if len(keyframes) >= 3 and len(map_points) > 50:
        logger.info("=== 全局 BA（PGO 后）===")
        all_registered = [k for k in keyframes
                          if frames[k].pose is not None]
        ba_state_g2 = {"success": False}
        (focal0, fy0, cx, cy) = bundle_adjustment(
            all_registered, map_points, frames,
            focal0, fy0, cx, cy,
            optimize_points=True,
            reproj_thresh=reproj_thresh,
            image_size=image_size,
            is_global=True,
            focal_init=focal_init,
            optimize_principal=False,
            _state=ba_state_g2)
        if not ba_state_g2["success"]:
            logger.warning("[全局 BA(PGO后)] 失败，跳过。")
    K_final = _build_K(focal0, fy0, cx, cy)
    _retriangulate_multiview(map_points, frames, K_final,
                             min_obs=MULTIVIEW_MIN_OBS)
    _merge_map_points(map_points, frames, norm_type)
    prune_map_points(map_points, frames, reproj_thresh,
                     focal0, fy0, cx, cy)
    filled_poses = []
    last_valid = frames[keyframes[0]].pose
    for i in range(len(frame_paths)):
        p = frames[i].pose
        if p is None:
            p = last_valid
        else:
            last_valid = p
        filled_poses.append(p)
    all_poses = []
    for i in range(len(frame_paths)):
        st = frames[i].status
        if st in (FrameStatus.LOST, FrameStatus.INVALID) \
                or frames[i].pose is None:
            all_poses.append(None)
        else:
            all_poses.append(frames[i].pose)
    poses_by_frame = {i: filled_poses[i] for i in range(len(filled_poses))}
    xyz, mask = filter_point_cloud(map_points, poses_by_frame,
                                   focal0, fy0, cx, cy, reproj_thresh)
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
        dist=(np.asarray(dist_coeffs).ravel()
              if dist_coeffs is not None else None))
    logger.info(
        f"=== 结束：fx={focal0:.2f}, fy={fy0:.2f}, "
        f"点数={len(xyz_out)}, 关键帧={len(keyframes)}, "
        f"回环={len(loop_closures)}, "
        f"总耗时={time.time() - t_start:.1f}s ===")
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
        frame_status=[frames[i].status for i in range(len(frame_paths))])