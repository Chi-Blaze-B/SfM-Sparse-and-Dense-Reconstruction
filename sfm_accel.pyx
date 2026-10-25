# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True, nonecheck=False, embedsignature=False
"""SfM 流水线的 Cython 热点内核。

只处理展平后的 numpy 数组：地图仍是 Python 对象图，这里的函数不碰它。
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, hypot, acos, M_PI, NAN
from libc.stdint cimport int64_t, uint8_t
from libc.stdlib cimport malloc, calloc, free

cnp.import_array()

ctypedef fused desc_t:
    cnp.uint8_t
    cnp.float32_t


# ---------------------------------------------------------------------------
# Hamming 距离的 popcount 查表（导入时建一次）
# ---------------------------------------------------------------------------
cdef unsigned char _POPCNT[256]


cdef void _init_popcnt() noexcept:
    cdef int i, v, c
    for i in range(256):
        v = i
        c = 0
        while v:
            c += v & 1
            v >>= 1
        _POPCNT[i] = <unsigned char>c


_init_popcnt()


# ---------------------------------------------------------------------------
# 0. 批量投影
# ---------------------------------------------------------------------------
def project_points(const double[:, ::1] R, const double[::1] t,
                   const double[:, ::1] X,
                   double f, double fy, double cx, double cy,
                   double[:, ::1] uv, double[::1] depth):
    """uv = K (R X + t)，depth = (R X + t)_z。深度非正的点 uv 写 NaN，调用方只看 depth。"""
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t i
    cdef double x, y, z, pc0, pc1, pc2
    cdef double r00 = R[0, 0], r01 = R[0, 1], r02 = R[0, 2]
    cdef double r10 = R[1, 0], r11 = R[1, 1], r12 = R[1, 2]
    cdef double r20 = R[2, 0], r21 = R[2, 1], r22 = R[2, 2]
    cdef double t0 = t[0], t1 = t[1], t2 = t[2]
    for i in range(n):
        x = X[i, 0]; y = X[i, 1]; z = X[i, 2]
        pc0 = r00 * x + r01 * y + r02 * z + t0
        pc1 = r10 * x + r11 * y + r12 * z + t1
        pc2 = r20 * x + r21 * y + r22 * z + t2
        depth[i] = pc2
        if pc2 > 1e-9:
            uv[i, 0] = f * pc0 / pc2 + cx
            uv[i, 1] = fy * pc1 / pc2 + cy
        else:
            uv[i, 0] = NAN
            uv[i, 1] = NAN


# ---------------------------------------------------------------------------
# 1. 散乱候选表上的描述子匹配
# ---------------------------------------------------------------------------
cdef inline double _desc_dist(const desc_t[::1] a, const desc_t[::1] b,
                              Py_ssize_t dim, int metric) noexcept nogil:
    cdef Py_ssize_t k
    cdef double s = 0.0
    cdef unsigned int h = 0
    if metric == 0:                      # L2，与 cv2.NORM_L2 一致（返回开方值）
        for k in range(dim):
            s += (<double>a[k] - <double>b[k]) * (<double>a[k] - <double>b[k])
        return sqrt(s)
    else:                                # Hamming
        for k in range(dim):
            h += _POPCNT[<unsigned char>(a[k]) ^ <unsigned char>(b[k])]
        return <double>h


# ---------------------------------------------------------------------------
# 2. 顺序指派（复刻 Python 循环的先后规则）
# ---------------------------------------------------------------------------
def desc_assign_sequential(const desc_t[:, ::1] desc_f,
                           const desc_t[:, ::1] desc_p,
                           const int64_t[::1] cand_pt,     # 非递减，长度 M
                           const int64_t[::1] cand_kp,     # 长度 M
                           const int64_t[::1] fm,          # 本帧 feat_map，只读
                           uint8_t[::1] used,              # 关键点占用标记，会被写
                           double max_dist,
                           int metric,
                           int64_t[::1] best_kp_out,       # 长度 n_pts，-1 表示没有
                           double[::1] best_d_out):        # 长度 n_pts
    """按点序逐个指派，先到的点优先占住关键点。

    原 Python 循环边扫边写 `feat_map[kp] = pt`，所以后面的点抢不到已被占用的关键点；
    这里用 `used` 复刻同一规则。距离相同则保留先遇到的候选。
    """
    cdef Py_ssize_t m = cand_pt.shape[0]
    cdef Py_ssize_t i, k, dim = desc_f.shape[1]
    cdef int64_t p, kp
    cdef double d, best
    cdef int64_t best_i

    for i in range(best_kp_out.shape[0]):
        best_kp_out[i] = -1
        best_d_out[i] = 0.0

    i = 0
    while i < m:
        p = cand_pt[i]
        best = max_dist
        best_i = -1
        k = i
        while k < m and cand_pt[k] == p:
            kp = cand_kp[k]
            if fm[kp] < 0 and used[kp] == 0:
                d = _desc_dist(desc_f[kp], desc_p[p], dim, metric)
                if d <= max_dist and (best_i < 0 or d < best):
                    best = d
                    best_i = kp
            k += 1
        if best_i >= 0:
            best_kp_out[p] = best_i
            best_d_out[p] = best
            used[best_i] = 1
        i = k
    return 0


# ---------------------------------------------------------------------------
# 3. 三角化角
# ---------------------------------------------------------------------------
cdef inline double _min_tri_angle_one(const double[:, ::1] X, Py_ssize_t p,
                                      const int64_t* obs_f,
                                      Py_ssize_t a0, Py_ssize_t a1,
                                      int64_t new_fr,
                                      const double[:, ::1] centers,
                                      const uint8_t[::1] cam_ok) noexcept nogil:
    cdef Py_ssize_t a, b, n = 0, nf = a1 - a0
    cdef Py_ssize_t ncam = cam_ok.shape[0]
    cdef int64_t fa, fb
    cdef double ax, ay, az, bx, by, bz, na, nb, c, best, ang
    cdef double deg = 180.0 / M_PI
    cdef int64_t idxs[64]
    if new_fr >= 0 and new_fr < ncam and cam_ok[new_fr] != 0:
        idxs[n] = new_fr
        n += 1
    for a in range(nf):
        fa = obs_f[a0 + a]
        if fa >= 0 and fa < ncam and cam_ok[fa] != 0 and n < 64:
            idxs[n] = fa
            n += 1
    if n < 2:
        return -1.0                     # 少于两相机：调用方按「通过」处理
    best = 180.0
    for a in range(n):
        fa = idxs[a]
        ax = X[p, 0] - centers[fa, 0]
        ay = X[p, 1] - centers[fa, 1]
        az = X[p, 2] - centers[fa, 2]
        na = sqrt(ax * ax + ay * ay + az * az)
        if na < 1e-9:
            na = 1e-9
        ax /= na; ay /= na; az /= na
        for b in range(a + 1, n):
            fb = idxs[b]
            bx = X[p, 0] - centers[fb, 0]
            by = X[p, 1] - centers[fb, 1]
            bz = X[p, 2] - centers[fb, 2]
            nb = sqrt(bx * bx + by * by + bz * bz)
            if nb < 1e-9:
                nb = 1e-9
            c = (ax * bx + ay * by + az * bz) / nb
            if c > 1.0:
                c = 1.0
            elif c < -1.0:
                c = -1.0
            ang = acos(c) * deg
            if ang < best:
                best = ang
    return best if best < 180.0 else 0.0


def check_obs_angle(const double[:, ::1] X,
                    const int64_t[::1] obs_pt,      # 已按点排好序的 CSR
                    const int64_t[::1] obs_f,
                    const double[:, ::1] centers,   # (n_frames, 3)
                    const uint8_t[::1] cam_ok,      # (n_frames,)
                    int64_t new_fr,
                    double min_angle,
                    uint8_t[::1] ok_out):           # (n_pts,)
    """一次性判定所有点的三角化角，等价于逐点调 `_check_obs_angle`。

    最小视角不小于 min_angle 即通过；可用相机少于两个时 Python 版直接返回 True，这里同样通过。
    """
    cdef Py_ssize_t n_pts = X.shape[0]
    cdef Py_ssize_t i
    cdef int64_t p
    cdef double ang
    cdef Py_ssize_t *starts = NULL
    cdef Py_ssize_t *cursor = NULL
    cdef int64_t *fof = NULL
    cdef Py_ssize_t nobs = obs_pt.shape[0]
    # 观测是边补边追加的，顺序不保证；这里自己桶排序，不要求调用方预排序。
    starts = <Py_ssize_t*>calloc(n_pts + 1, sizeof(Py_ssize_t))
    cursor = <Py_ssize_t*>malloc((n_pts + 1) * sizeof(Py_ssize_t))
    fof = <int64_t*>malloc((nobs + 1) * sizeof(int64_t))
    if starts == NULL or cursor == NULL or fof == NULL:
        if starts != NULL: free(starts)
        if cursor != NULL: free(cursor)
        if fof != NULL: free(fof)
        raise MemoryError()
    try:
        for i in range(nobs):
            p = obs_pt[i]
            if p >= 0 and p < n_pts:
                starts[p + 1] += 1
        for i in range(n_pts):
            starts[i + 1] += starts[i]
            cursor[i] = starts[i]
        for i in range(nobs):
            p = obs_pt[i]
            if p >= 0 and p < n_pts:
                fof[cursor[p]] = obs_f[i]
                cursor[p] += 1
        for i in range(n_pts):
            ang = _min_tri_angle_one(X, i, fof, starts[i], starts[i + 1],
                                     new_fr, centers, cam_ok)
            ok_out[i] = 1 if (ang < 0.0 or ang >= min_angle) else 0
    finally:
        free(starts)
        free(cursor)
        free(fof)
    return 0


# ---------------------------------------------------------------------------
# 4. 地图点合并的并查集
# ---------------------------------------------------------------------------
def merge_scan(const double[:, ::1] xyz,
               const desc_t[:, ::1] descs,
               const int64_t[::1] pair_a,
               const int64_t[::1] pair_b,
               const int64_t[::1] grp,
               int64_t[::1] parent,
               double pos_thresh,
               double desc_thresh,
               int metric,
               int use_desc):
    """逐对判据与并查集，逐条对应 `_merge_map_points` 的内层循环。

    跳过条件依次是：obs_set 相同（用组号 grp 比较）、空间距离超限、描述子距离超限、已连通。
    返回实际合并次数。
    """
    cdef Py_ssize_t m = pair_a.shape[0]
    cdef Py_ssize_t k, dim = descs.shape[1]
    cdef int64_t i, j, ra, rb
    cdef double dx, dy, dz, dd
    cdef long n_union = 0
    cdef Py_ssize_t n = parent.shape[0]
    for k in range(n):
        parent[k] = k
    for k in range(m):
        i = pair_a[k]
        j = pair_b[k]
        if grp[i] == grp[j]:
            continue
        dx = xyz[i, 0] - xyz[j, 0]
        dy = xyz[i, 1] - xyz[j, 1]
        dz = xyz[i, 2] - xyz[j, 2]
        if dx * dx + dy * dy + dz * dz > pos_thresh * pos_thresh:
            continue
        if use_desc:
            dd = _desc_dist(descs[i], descs[j], dim, metric)
            if dd > desc_thresh:
                continue
        ra = i
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        rb = j
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        if ra != rb:
            parent[rb] = ra
            n_union += 1
    return n_union
