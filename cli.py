"""视频 / 图片集转 SfM 稀疏/稠密重建 CLI 端。"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import psutil

from frames import extract_frames, import_images, list_images, exif_focal_ratio
from poses import estimate_poses, CameraPose, FrameStatus
from exporter import (
    intr_from_K,
    write_ply,
    write_cameras_txt,
    write_intrinsics_json,
)
import frame_meta

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def set_affinity_to_all_cores() -> None:
    try:
        p = psutil.Process(os.getpid())
        all_cpus = list(range(psutil.cpu_count()))
        p.cpu_affinity(all_cpus)
        logger.info("CPU 亲和性设置为 %d 个核心", len(all_cpus))
    except Exception as e:
        logger.warning("无法设置 CPU 亲和性: %s", e)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video-to-SfM Pipeline (sparse / dense reconstruction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入 / 输出：视频与图片集二选一
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="输入视频文件路径")
    src.add_argument(
        "--image-dir", type=str,
        help="输入图片集目录（与 --video 二选一；顶层无图片时会递归查找子目录）",
    )
    parser.add_argument("--output-dir", type=str, default="./sfm_output",
                        help="输出目录（存放点云 / 相机位姿 / 内参）")
    parser.add_argument("--workdir", type=str, default="./workdir",
                        help="中间文件工作目录")

    # 运行模式
    parser.add_argument(
        "--mode", type=str,
        choices=["sparse", "dense"], default="sparse",
        help="运行模式：sparse=只跑 SfM 稀疏重建；dense=SfM + SGBM 稠密重建",
    )

    # 帧提取（仅 --video）
    parser.add_argument(
        "--sampling-mode", type=str,
        choices=["uniform", "smart", "two-stage"], default="uniform",
        help="帧采样策略（仅视频）：uniform / smart（光流）/ two-stage（视差+光流+纹理）",
    )
    parser.add_argument("--fps", type=float, default=15.0,
                        help="目标帧率（仅视频，均匀采样模式）")
    parser.add_argument("--scale", type=float, default=None,
                        help="缩放系数 0<scale<=1（视频默认 0.5；图片集默认 1.0=直接引用原图）")
    parser.add_argument("--min-frames", type=int, default=None,
                        help="最少提取帧数（仅视频，默认 30）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="最多帧数/图片数（视频默认 200；图片集默认 0=全部，>0 时均匀抽取）")

    # 相机标定（可选）
    parser.add_argument("--camera-matrix", type=str, default=None,
                        help="3x3 内参矩阵 .npy（可选；图片集有标定时强烈建议提供）")
    parser.add_argument("--dist-coeffs", type=str, default=None,
                        help="畸变系数 .npy：4/5/8 值走 OpenCV，12 值走 COLMAP THIN_PRISM_FISHEYE")

    # 位姿估计
    parser.add_argument(
        "--feature-type", type=str,
        choices=["orb", "sift", "rootsift"], default="orb",
        help="特征描述子（orb=快速二进制，sift=鲁棒浮点，rootsift=SIFT 归一化，较慢）",
    )
    parser.add_argument(
        "--use-focal-guess", action="store_true",
        help="以 1.0×图像长边作为初始像素焦距（约 53° 长边方向 FOV）",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="关闭回环检测（短序列 / 纯前向拍摄可关，加速且减少误回环风险）",
    )
    parser.add_argument(
        "--no-pgo", action="store_true",
        help="关闭位姿图优化（无回环时收益有限）",
    )
    parser.add_argument(
        "--enable-global-matching", action="store_true",
        help="启用全局词袋检索匹配（更稳但更慢，长序列建议开启）",
    )

    # 稠密重建参数
    dense_group = parser.add_argument_group("稠密重建（--mode dense 时生效）")
    dense_group.add_argument("--dense-downscale", type=int, default=2,
                             help="稠密重建时帧降采样倍数，越大越快，精度越低")
    dense_group.add_argument("--dense-max-pairs", type=int, default=60,
                             help="稠密重建最多处理帧对数")
    dense_group.add_argument("--dense-voxel-ratio", type=float, default=0.005,
                             help="稠密点云体素大小 / 场景尺度")
    dense_group.add_argument("--dense-min-views", type=int, default=1,
                             help="稠密融合：一个体素至少被多少个帧对命中才保留；"
                                  "2 可去掉单帧对产生的重复点，但会稀疏一些")

    # 缓存复用
    parser.add_argument("--reuse-dir", type=str, default=None,
                        help="复用指定工作目录的中间产物（帧 / 位姿缓存），跳过已完成步骤")

    # 帧图像内存缓存
    parser.add_argument(
        "--frame-cache-mb", type=int, default=1500,
        help="帧图像内存缓存上限（MiB）；0 表示禁用。用于加速重复读取",
    )

    # 内部评测口：不对外暴露，不进 --help
    parser.add_argument(
        "--gt-model", type=str, default=None, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-self-calibrate", action="store_true",
        help="关闭无标定输入的径向畸变自标定。开启时若估计到显著 k1，会用去畸变"
             "特征再跑一遍（耗时约翻倍）",
    )
    parser.add_argument(
        "--self-calib-iter", action="store_const", const=True, default=None,
        help="自标定后再估一次残余 k1，显著时跑第三遍（k1/中心更好，旋转可能变差，"
             "耗时约 +20%%）",
    )
    parser.add_argument(
        "--rot-edge-weight", type=float, default=None,
        help="最终 BA 中两视图旋转约束的权重（默认 100）；0 关闭",
    )
    parser.add_argument(
        "--pose-outlier-deg", type=float, default=None,
        help="位姿校验阈值：与邻居的最小相对旋转偏差（默认 12°）；0 关闭",
    )

    return parser


def load_poses(poses_data: np.ndarray) -> list:
    """从 [n,4,4] 定长数组还原位姿列表，NaN 行表示位姿缺失。"""
    poses = []
    for p in poses_data:
        if np.isnan(p).any():
            poses.append(None)
        else:
            poses.append(CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()))
    return poses


def save_poses(poses: list, path: Path) -> None:
    """位姿存成 [n,4,4] 定长数组，缺失位姿填 NaN。"""
    poses_arr = np.full((len(poses), 4, 4), np.nan, dtype=np.float32)
    for i, p in enumerate(poses):
        if p is not None:
            poses_arr[i] = p.RT
    np.save(path, poses_arr)


# ---- 输入解析 ----
def _resolve_source_args(args: argparse.Namespace) -> None:
    image_mode = args.image_dir is not None
    if args.scale is None:
        args.scale = 1.0 if image_mode else 0.5
    if args.min_frames is None:
        args.min_frames = 2 if image_mode else 30
    if args.max_frames is None:
        args.max_frames = 0 if image_mode else 200


def _load_calibration(
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    camera_matrix = np.load(args.camera_matrix) if args.camera_matrix else None
    dist_coeffs = np.load(args.dist_coeffs) if args.dist_coeffs else None
    return camera_matrix, dist_coeffs


# ---- 主流程 ----
def run_pipeline(args: argparse.Namespace) -> None:
    overall_start = time.time()
    enable_dense = (args.mode == "dense")
    n_steps = 4 if enable_dense else 3
    image_mode = args.image_dir is not None

    workdir = Path(args.workdir)
    if args.reuse_dir is not None:
        workdir = Path(args.reuse_dir)
        if not workdir.exists():
            logger.error("复用目录不存在: %s", args.reuse_dir)
            sys.exit(1)
        logger.info("复用工作目录: %s", workdir)
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = workdir / "frames"
    poses_dir = workdir / "poses"
    poses_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("  %s → SfM %s重建",
                "图片集" if image_mode else "视频",
                "稀疏 + 稠密" if enable_dense else "稀疏")
    logger.info("=" * 60)
    logger.info("运行模式: %s", args.mode)
    if image_mode:
        logger.info("图片集: %s", os.path.abspath(args.image_dir))
    else:
        logger.info("视频: %s", os.path.abspath(args.video))

    # ---- 步骤 1：抽帧 / 导入图片集 ----
    logger.info("[1/%d] %s...", n_steps,
                "正在导入图片集" if image_mode else "正在提取视频帧")
    t0 = time.time()

    frame_paths_file = workdir / "frame_paths.txt"
    meta_path = workdir / "frame_meta.json"
    sfm_meta_path = workdir / "sfm_meta.json"

    can_reuse_frames = frame_meta.can_reuse_frames(workdir)

    current_meta = {
        "source_type": "images" if image_mode else "video",
        "video": None if image_mode else os.path.abspath(args.video),
        "image_dir": os.path.abspath(args.image_dir) if image_mode else None,
        "scale": args.scale,
        "fps": args.fps,
        "sampling_mode": args.sampling_mode,
        "feature_type": args.feature_type,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
    }

    if can_reuse_frames and meta_path.exists():
        mismatch_reasons = frame_meta.check_meta(meta_path, current_meta)
        if mismatch_reasons:
            logger.warning(
                "  输入/提取参数变化（%s），重新准备帧并作废下游缓存（位姿 / 点云 / 稠密帧对）",
                "; ".join(mismatch_reasons),
            )
            can_reuse_frames = False
            for name in frame_meta.invalidate_downstream(workdir):
                logger.info("  🗑  已删除过期文件 %s", name)
    elif can_reuse_frames:
        logger.info("  提示: 复用旧帧缓存（无 frame_meta.json，未校验提取参数）")

    if can_reuse_frames:
        frame_paths = [p.strip() for p in frame_paths_file.read_text().splitlines()]
        logger.info("已从 %s 加载 %d 帧（跳过抽帧）", workdir, len(frame_paths))
    elif image_mode:
        frame_paths = import_images(
            args.image_dir,
            output_dir=str(frame_dir),
            scale=args.scale,
            max_frames=args.max_frames,
        )
        frame_paths_file.write_text("\n".join(frame_paths))
        frame_meta.write_meta(meta_path, current_meta)
        logger.info("已导入 %d 张图片（耗时 %.1fs）",
                    len(frame_paths), time.time() - t0)
    else:
        smart_sampling = args.sampling_mode != "uniform"
        two_stage = args.sampling_mode == "two-stage"

        frame_paths = extract_frames(
            video_path=args.video,
            output_dir=str(frame_dir),
            fps=args.fps,
            scale=args.scale,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            smart_sampling=smart_sampling,
            two_stage=two_stage,
            poses_output_dir=str(poses_dir / "coarse_poses") if two_stage else None,
            optical_flow_method="farneback",
            feature_type=args.feature_type,
        )
        frame_paths_file.write_text("\n".join(frame_paths))
        frame_meta.write_meta(meta_path, current_meta)
        logger.info("已提取 %d 帧（耗时 %.1fs）", len(frame_paths), time.time() - t0)

    if len(frame_paths) < 2:
        logger.error("至少需要 2 帧。")
        sys.exit(1)

    # ---- 帧缓存：先建实例，稍后预加载 ----
    cache_mb = int(getattr(args, "frame_cache_mb", 1500))
    try:
        from frame_cache import (
            configure as _configure_cache,
            cache_stats as _cache_stats,
            get_image as _get_image,
        )
    except ImportError:
        _configure_cache = None
        _cache_stats = None
        _get_image = None

    if _configure_cache is not None and cache_mb > 0:
        _configure_cache(max_bytes=cache_mb * 1024 * 1024)

    # 读首帧测尺寸；缓存禁用时退化为 cv2.imread
    if _get_image is not None:
        _img = _get_image(frame_paths[0])
    else:
        import cv2 as _cv2
        _img = _cv2.imread(frame_paths[0])
    if _img is None:
        logger.error("无法读取首帧: %s", frame_paths[0])
        sys.exit(1)
    h, w = _img.shape[:2]
    logger.info("帧分辨率: %dx%d", w, h)

    # ---- 帧缓存：这里只设上限，不预加载 ----
    # 位姿估计阶段每帧只解码一次，预加载省不到任何工作量；反而会占住
    # _auto_feature_workers 用来估算并行度的可用内存，白白少开一个 worker。
    # 真正的预加载推迟到稠密重建之前（见步骤 3）。
    if _configure_cache is not None and cache_mb <= 0:
        _configure_cache(max_bytes=0)
        logger.info("帧缓存已禁用（--frame-cache-mb 0）")

    # ---- 步骤 2：位姿估计 ----
    logger.info("[2/%d] 正在估计相机位姿...", n_steps)
    t0 = time.time()

    intrinsics_file = workdir / "intrinsics.npy"
    poses_file = workdir / "poses.npy"
    sparse_file = workdir / "sparse_points.npy"

    sfm_result = None
    cached_keyframes = None
    cached_loop_closures = []
    cached_frame_status = None
    cached_dist = None

    if intrinsics_file.exists() and poses_file.exists() and sparse_file.exists():
        K = np.load(intrinsics_file)
        sparse_points = np.load(sparse_file)
        poses = load_poses(np.load(poses_file))
        if len(poses) > len(frame_paths):
            poses = poses[:len(frame_paths)]

        if sfm_meta_path.exists():
            try:
                _m = json.loads(sfm_meta_path.read_text())
                cached_keyframes = _m.get("keyframes")
                cached_loop_closures = [tuple(x) for x in _m.get("loop_closures", [])]
                cached_frame_status = _m.get("frame_status")
                cached_dist = _m.get("dist")
            except Exception as e:
                logger.warning("读取 sfm_meta.json 失败: %s", e)

        logger.info(
            "已从 %s 加载 SfM 快照（intrinsics / poses / sparse_points），跳过位姿估计",
            workdir,
        )
    else:
        camera_matrix, dist_coeffs = _load_calibration(args)
        if camera_matrix is not None:
            if image_mode and args.scale < 0.999:
                logger.warning(
                    "  提供了内参但 scale=%.2f：内参对应的是原始分辨率，"
                    "请改用 --scale 1.0 或自行缩放内参", args.scale)
            logger.info("使用给定内参: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                        camera_matrix[0, 0], camera_matrix[1, 1],
                        camera_matrix[0, 2], camera_matrix[1, 2])
        if dist_coeffs is not None:
            n_d = int(np.asarray(dist_coeffs).size)
            logger.info("使用给定畸变系数: %d 个（%s）", n_d,
                        "COLMAP THIN_PRISM_FISHEYE" if n_d == 12 else "OpenCV")

        # 焦距初值优先级：显式内参 > --use-focal-guess > EXIF > 内置启发（1.2×长边）
        focal_guess = None
        if camera_matrix is not None:
            pass
        elif args.use_focal_guess:
            focal_guess = float(max(w, h))
            fov_deg = 2.0 * np.degrees(np.arctan(max(w, h) / (2.0 * focal_guess)))
            axis = "水平" if w >= h else "垂直"
            logger.info("初始焦距猜测: %.1fpx（约 %.0f° %s FOV）",
                        focal_guess, fov_deg, axis)
        elif image_mode:
            _src = list_images(args.image_dir)
            _ratio = exif_focal_ratio(_src[0]) if _src else None
            if _ratio is not None:
                focal_guess = _ratio * w
                logger.info("按 EXIF 估计初始焦距: fx=%.1f px（%.2f×宽度）",
                            focal_guess, _ratio)
            else:
                logger.info("图片无 EXIF 焦距，使用默认启发式 fx=%.1f px（1.2×长边）",
                            max(w, h) * 1.2)
        else:
            logger.info("视频输入无 EXIF 焦距，使用默认启发式 fx=%.1f px（1.2×长边）",
                        max(w, h) * 1.2)

        try:
            sfm_result = estimate_poses(
                frame_paths,
                args.feature_type,
                focal_guess=focal_guess,
                aspect_ratio=1.0,
                dist_coeffs=dist_coeffs,
                camera_matrix=camera_matrix,
                enable_loop=not args.no_loop,
                enable_pgo=not args.no_pgo,
                enable_global_matching=args.enable_global_matching,
                self_calibrate=not args.no_self_calibrate,
                self_calib_iter=args.self_calib_iter,
                rot_edge_weight=args.rot_edge_weight,
                pose_outlier_deg=args.pose_outlier_deg,
            )
            K = sfm_result.intrinsics.K
            poses = sfm_result.poses
            sparse_points = sfm_result.xyz
        except RuntimeError as e:
            logger.error("位姿估算失败: %s", e)
            logger.error(
                "建议：换用平移充分、纹理丰富、无大面积运动模糊的素材。"
            )
            sys.exit(1)

        if sfm_result.frame_status is not None:
            st = Counter(sfm_result.frame_status)
            logger.info("SfM 帧状态: %s", dict(st))
            logger.info("SfM 关键帧: %d / %d",
                        len(sfm_result.keyframes), len(frame_paths))
            logger.info("SfM 回环: %d 处", len(sfm_result.loop_closures))
            if len(sfm_result.loop_closures) > 0:
                for a, b in sfm_result.loop_closures[:10]:
                    logger.info("  loop: %d ↔ %d", a, b)
            n_lost = st.get(FrameStatus.LOST, 0) + st.get(FrameStatus.INVALID, 0)
            if n_lost > len(frame_paths) * 0.4:
                logger.warning(
                    "超过 40%% 的帧丢失/无效（%d / %d），建议更换素材。",
                    n_lost, len(frame_paths),
                )

        if args.gt_model:
            if not image_mode:
                logger.warning("--gt-model 只对图片集输入有意义（视频帧名无法与参考模型对应）")
            from eval_tools import evaluate_run, format_summary
            logger.info("=== 参考模型评测 ===")
            eval_metrics = evaluate_run(
                sfm_result, frame_paths, args.gt_model,
                img_scale=args.scale if image_mode else 1.0)
            for _line in format_summary(eval_metrics).splitlines():
                logger.info(_line)
            try:
                (output_dir / "evaluation.json").write_text(
                    json.dumps(eval_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            except Exception as e:
                logger.warning("写入 evaluation.json 失败: %s", e)

        np.save(intrinsics_file, K)
        save_poses(poses, poses_file)
        if sparse_points is None:
            sparse_points = np.zeros((0, 3), dtype=np.float32)
        np.save(sparse_file, sparse_points)

        try:
            sfm_meta_path.write_text(json.dumps({
                "keyframes": list(sfm_result.keyframes),
                "loop_closures": [[int(a), int(b)]
                                  for a, b in sfm_result.loop_closures],
                "frame_status": (list(sfm_result.frame_status)
                                 if sfm_result.frame_status else None),
                # 自标定 / 用户给定的畸变模型必须落盘：复用目录时稠密阶段要靠它反畸变
                "dist": (None if sfm_result.intrinsics.dist is None
                         else [float(v) for v in
                               np.asarray(sfm_result.intrinsics.dist).ravel()]),
            }, ensure_ascii=False))
        except Exception as e:
            logger.warning("写入 sfm_meta.json 失败: %s", e)

    while len(poses) < len(frame_paths):
        poses.append(None)

    if sfm_result is not None and sfm_result.frame_status is not None:
        lost_or_invalid = sum(
            1 for st in sfm_result.frame_status
            if st in (FrameStatus.LOST, FrameStatus.INVALID)
        )
        valid_count = len(frame_paths) - lost_or_invalid
    else:
        valid_count = sum(1 for p in poses if p is not None)
    logger.info("共 %d 帧，其中 %d 帧位姿有效（耗时 %.1fs）",
                len(frame_paths), valid_count, time.time() - t0)

    if valid_count < 3:
        logger.error("有效位姿过少。请检查素材质量（平移、纹理、模糊）。")
        sys.exit(1)

    # ---- 步骤 3：稠密重建 ----
    import threading
    import signal

    dense_result = None
    if enable_dense:
        logger.info("[3/%d] 正在稠密重建...", n_steps)
        t0 = time.time()
        # 稠密重建会跨帧对反复解码同一帧，这里才是缓存的用武之地。
        if _configure_cache is not None and cache_mb > 0:
            t_cache = time.time()
            _configure_cache(max_bytes=cache_mb * 1024 * 1024,
                             preload=frame_paths)
            logger.info("帧图像已预加载到内存（上限 %d MiB，耗时 %.1fs）",
                        cache_mb, time.time() - t_cache)
        try:
            from dense import dense_reconstruct, DenseConfig, DenseAbortedError
        except ImportError as e:
            logger.warning("无法导入稠密模块 dense.py: %s，跳过稠密重建", e)
            enable_dense = False
            n_steps = 3
        else:
            if sfm_result is not None and sfm_result.keyframes:
                _kf = list(sfm_result.keyframes)
            else:
                _kf = cached_keyframes
            _intr = sfm_result.intrinsics if sfm_result else intr_from_K(K)
            if sfm_result is None:
                _, _dist_dense = _load_calibration(args)
                if _dist_dense is not None:
                    _intr.dist = np.asarray(_dist_dense, dtype=np.float64)
                elif cached_dist:
                    _intr.dist = np.asarray(cached_dist, dtype=np.float64)

            dense_cfg = DenseConfig(
                downscale=args.dense_downscale,
                max_pairs=args.dense_max_pairs,
                voxel_ratio=args.dense_voxel_ratio,
                fusion_min_views=args.dense_min_views,
            )
            dense_path = output_dir / "dense_points.ply"

            stop_event = threading.Event()
            _old_sigint = signal.getsignal(signal.SIGINT)

            def _on_sigint(signum, frame):
                if not stop_event.is_set():
                    logger.info("\n检测到 Ctrl+C，正在中断（已算帧对会落盘）...")
                    stop_event.set()

            try:
                signal.signal(signal.SIGINT, _on_sigint)

                dense_result = dense_reconstruct(
                    frame_paths, _intr, poses, str(dense_path),
                    keyframes=_kf,
                    config=dense_cfg,
                    cache_dir=str(workdir / "dense_pairs"),
                    stop_event=stop_event,
                )
                logger.info(
                    "稠密点云已导出: %s（%d 点，%d/%d 对成功，缓存命中 %d，耗时 %.1fs）",
                    dense_path, dense_result.num_points,
                    dense_result.num_pairs_used, dense_result.num_pairs_total,
                    dense_result.num_pairs_cached,
                    dense_result.elapsed_sec,
                )
            except DenseAbortedError:
                logger.info("稠密重建已中断，缓存已保存到 %s/dense_pairs", workdir)
                logger.info("继续导出稀疏结果……")
                dense_result = None
            except Exception as e:
                logger.warning("稠密重建失败: %s", e)
            finally:
                signal.signal(signal.SIGINT, _old_sigint)

    # ---- 步骤 4：导出 ----
    logger.info("[%d/%d] 正在导出结果...", n_steps, n_steps)
    t0 = time.time()

    ply_path = output_dir / "sparse_points.ply"
    cam_path = output_dir / "cameras.txt"
    intr_path = output_dir / "intrinsics.json"
    report_path = output_dir / "sfm_report.json"

    try:
        write_ply(str(ply_path), np.asarray(sparse_points))
        logger.info("稀疏点云已导出: %s（%d 点）",
                    ply_path, int(np.asarray(sparse_points).shape[0]))
    except Exception as e:
        logger.warning("稀疏点云导出失败: %s", e)

    try:
        write_cameras_txt(str(cam_path), poses)
        logger.info("相机位姿已导出: %s", cam_path)
    except Exception as e:
        logger.warning("相机导出失败: %s", e)

    try:
        write_intrinsics_json(str(intr_path), K, w, h)
        logger.info("内参已导出: %s", intr_path)
    except Exception as e:
        logger.warning("内参导出失败: %s", e)

    try:
        report = {
            "source_type": "images" if image_mode else "video",
            "video": None if image_mode else os.path.abspath(args.video),
            "image_dir": os.path.abspath(args.image_dir) if image_mode else None,
            "run_mode": args.mode,
            "num_frames": len(frame_paths),
            "num_valid_poses": int(valid_count),
            "num_sparse_points": int(np.asarray(sparse_points).shape[0]),
            "num_dense_points": int(dense_result.num_points) if dense_result else 0,
            "dense_pairs_used": int(dense_result.num_pairs_used) if dense_result else 0,
            "num_keyframes": (len(sfm_result.keyframes) if sfm_result
                              else (len(cached_keyframes) if cached_keyframes else 0)),
            "num_loop_closures": (len(sfm_result.loop_closures) if sfm_result
                                  else len(cached_loop_closures)),
            "feature_type": args.feature_type,
            "image_width": int(w),
            "image_height": int(h),
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("报告已导出: %s", report_path)
    except Exception as e:
        logger.warning("报告导出失败: %s", e)

    total = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("完成！输出目录: %s", os.path.abspath(output_dir))
    logger.info("总耗时: %.1fs", total)
    if _cache_stats is not None:
        try:
            _st = _cache_stats()
            logger.info(
                "帧缓存统计: 命中 %d / 未命中 %d（命中率 %.1f%%），"
                "驱逐 %d，当前占用 %.1f / %.1f MiB",
                _st["hits"], _st["misses"], _st["hit_rate"] * 100,
                _st["evictions"],
                _st["bytes"] / 1048576, _st["max_bytes"] / 1048576,
            )
        except Exception as e:
            logger.debug("读取帧缓存统计失败: %s", e)
    logger.info("=" * 60)


def cli(argv: list[str] = None) -> None:
    setup_logging()
    set_affinity_to_all_cores()
    parser = build_parser()
    args = parser.parse_args(argv)

    _resolve_source_args(args)

    if args.image_dir is not None:
        if not os.path.isdir(args.image_dir):
            logger.error("图片集目录不存在: %s", args.image_dir)
            sys.exit(1)
    elif not os.path.isfile(args.video):
        logger.error("视频文件不存在: %s", args.video)
        sys.exit(1)

    if args.camera_matrix and not os.path.isfile(args.camera_matrix):
        logger.error("内参文件不存在: %s", args.camera_matrix)
        sys.exit(1)
    if args.dist_coeffs and not os.path.isfile(args.dist_coeffs):
        logger.error("畸变系数文件不存在: %s", args.dist_coeffs)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    cli()