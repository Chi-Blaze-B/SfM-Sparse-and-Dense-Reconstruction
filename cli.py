"""视频转 SfM 稀疏/稠密重建 CLI 端。"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

from frames import extract_frames
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

    # 输入 / 输出
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
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

    # 帧提取
    parser.add_argument(
        "--sampling-mode", type=str,
        choices=["uniform", "smart", "two-stage"], default="uniform",
        help="帧采样策略：uniform / smart（光流）/ two-stage（视差+光流+纹理）",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="目标帧率（均匀采样模式）")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放系数 (0<scale<=1)")
    parser.add_argument("--min-frames", type=int, default=30, help="最少提取帧数")
    parser.add_argument("--max-frames", type=int, default=200, help="最多提取帧数")

    # 位姿估计
    parser.add_argument(
        "--feature-type", type=str,
        choices=["orb", "sift"], default="orb",
        help="特征描述子（orb=快速二进制，sift=鲁棒浮点，较慢）",
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

    # 稠密重建参数（仅 --mode dense 时生效）
    dense_group = parser.add_argument_group("稠密重建（--mode dense 时生效）")
    dense_group.add_argument("--dense-downscale", type=int, default=2,
                             help="稠密重建时帧降采样倍数，越大越快，精度越低")
    dense_group.add_argument("--dense-max-pairs", type=int, default=60,
                             help="稠密重建最多处理帧对数")
    dense_group.add_argument("--dense-voxel-ratio", type=float, default=0.005,
                             help="稠密点云体素大小 / 场景尺度")

    # 缓存复用
    parser.add_argument("--reuse-dir", type=str, default=None,
                        help="复用指定工作目录的中间产物（帧 / 位姿缓存），跳过已完成步骤")

    return parser


def load_poses(poses_data: np.ndarray) -> list:
    """从定长 [n,4,4] 数组还原位姿列表，NaN 行表示位姿缺失。"""
    poses = []
    for p in poses_data:
        if np.isnan(p).any():
            poses.append(None)
        else:
            poses.append(CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()))
    return poses


def save_poses(poses: list, path: Path) -> None:
    """位姿列表保存为定长 [n,4,4] 数组，缺失位姿填 NaN。"""
    poses_arr = np.full((len(poses), 4, 4), np.nan, dtype=np.float32)
    for i, p in enumerate(poses):
        if p is not None:
            poses_arr[i] = p.RT
    np.save(path, poses_arr)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    overall_start = time.time()
    enable_dense = (args.mode == "dense")
    n_steps = 4 if enable_dense else 3

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
    logger.info("  视频 → SfM %s重建",
                "稀疏 + 稠密" if enable_dense else "稀疏")
    logger.info("=" * 60)
    logger.info("运行模式: %s", args.mode)

    # ---- 步骤 1：抽帧 ----
    logger.info("[1/%d] 正在提取帧...", n_steps)
    t0 = time.time()

    frame_paths_file = workdir / "frame_paths.txt"
    meta_path = workdir / "frame_meta.json"

    can_reuse_frames = frame_paths_file.exists()

    current_meta = {
        "video": os.path.abspath(args.video),
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
                "  提取参数变化（%s），重新提取帧并作废下游缓存（位姿 / 点云）",
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

    import cv2
    _img = cv2.imread(frame_paths[0])
    if _img is None:
        logger.error("无法读取首帧: %s", frame_paths[0])
        sys.exit(1)
    h, w = _img.shape[:2]
    logger.info("帧分辨率: %dx%d", w, h)

    # ---- 步骤 2：位姿估计 ----
    logger.info("[2/%d] 正在估计相机位姿...", n_steps)
    t0 = time.time()

    intrinsics_file = workdir / "intrinsics.npy"
    poses_file = workdir / "poses.npy"
    sparse_file = workdir / "sparse_points.npy"

    sfm_result = None

    if intrinsics_file.exists() and poses_file.exists() and sparse_file.exists():
        K = np.load(intrinsics_file)
        sparse_points = np.load(sparse_file)
        poses = load_poses(np.load(poses_file))
        if len(poses) > len(frame_paths):
            poses = poses[:len(frame_paths)]
        logger.info(
            "已从 %s 加载 SfM 快照（intrinsics / poses / sparse_points），跳过位姿估计",
            workdir,
        )
    else:
        focal_guess = None
        if args.use_focal_guess:
            focal_guess = float(max(w, h))
            fov_deg = 2.0 * np.degrees(np.arctan(max(w, h) / (2.0 * focal_guess)))
            axis = "水平" if w >= h else "垂直"
            logger.info("初始焦距猜测: %.1fpx（约 %.0f° %s FOV）",
                        focal_guess, fov_deg, axis)

        try:
            sfm_result = estimate_poses(
                frame_paths,
                min_inliers=25,
                feature_type=args.feature_type,
                focal_guess=focal_guess,
                aspect_ratio=1.0,
                enable_loop=not args.no_loop,
                enable_pgo=not args.no_pgo,
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
            if n_lost > len(frame_paths) * 0.3:
                logger.warning(
                    "超过 30%% 的帧丢失/无效（%d / %d），建议更换素材。",
                    n_lost, len(frame_paths),
                )

        np.save(intrinsics_file, K)
        save_poses(poses, poses_file)
        if sparse_points is None:
            sparse_points = np.zeros((0, 3), dtype=np.float32)
        np.save(sparse_file, sparse_points)

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
        logger.error("有效位姿过少。请检查视频质量。")
        sys.exit(1)

    # ---- 步骤 3：稠密重建（仅 dense 模式） ----
    import threading
    import signal

    dense_result = None
    if enable_dense:
        logger.info("[3/%d] 正在稠密重建...", n_steps)
        t0 = time.time()
        try:
            from dense import dense_reconstruct, DenseConfig, DenseAbortedError
        except ImportError as e:
            logger.warning("无法导入稠密模块 dense.py: %s，跳过稠密重建", e)
            enable_dense = False
            n_steps = 3
        else:
            _kf = list(sfm_result.keyframes) if (sfm_result and sfm_result.keyframes) else None
            _intr = sfm_result.intrinsics if sfm_result else intr_from_K(K)

            dense_cfg = DenseConfig(
                downscale=args.dense_downscale,
                max_pairs=args.dense_max_pairs,
                voxel_ratio=args.dense_voxel_ratio,
            )
            dense_path = output_dir / "dense_points.ply"

            # Ctrl+C → 置位 stop_event；dense 内部会 flush 后抛 DenseAbortedError
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
                # 稠密中断不丢弃稀疏成果：继续走导出流程
                logger.info("稠密重建已中断，缓存已保存到 %s/dense_pairs", workdir)
                logger.info("继续导出稀疏结果……")
                dense_result = None
            except Exception as e:
                logger.warning("稠密重建失败: %s", e)
            finally:
                signal.signal(signal.SIGINT, _old_sigint)

    # ---- 步骤 4（或 3）：导出 ----
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
            "video": os.path.abspath(args.video),
            "run_mode": args.mode,
            "num_frames": len(frame_paths),
            "num_valid_poses": int(valid_count),
            "num_sparse_points": int(np.asarray(sparse_points).shape[0]),
            "num_dense_points": int(dense_result.num_points) if dense_result else 0,
            "dense_pairs_used": int(dense_result.num_pairs_used) if dense_result else 0,
            "num_keyframes": (len(sfm_result.keyframes) if sfm_result else 0),
            "num_loop_closures": (len(sfm_result.loop_closures) if sfm_result else 0),
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
    logger.info("=" * 60)


def cli(argv: list[str] = None) -> None:
    setup_logging()
    set_affinity_to_all_cores()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.video):
        logger.error("视频文件不存在: %s", args.video)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    cli()