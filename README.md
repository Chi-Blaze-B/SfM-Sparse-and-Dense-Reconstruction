<h1 align="center"><strong>SfM Sparse & Dense Reconstruction</strong></h1>

<p align="center">
  <img src="https://img.shields.io/badge/🌟_SfM-Sparse_%26_Dense-FF6F00?style=flat-square&logo=github&logoColor=white" alt="Project">
  <br><br>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/SciPy-1.10+-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy">
  <br><br>
  <img src="https://img.shields.io/badge/GUI-PySide6-8A2BE2?style=flat-square&logo=qt&logoColor=white" alt="GUI">
  <img src="https://img.shields.io/badge/SfM-ORB_|_SIFT-00A98F?style=flat-square" alt="SfM">
  <img src="https://img.shields.io/badge/MVS-SGBM-FF4500?style=flat-square" alt="MVS">
  <br><br>
  <img src="https://img.shields.io/badge/Platform-Windows_|_Linux_|_macOS-0078D4?style=flat-square&logo=windows&logoColor=white" alt="Platform">
  <img src="https://img.shields.io/badge/License-Apache_2.0-1E90FF?style=flat-square&logo=apache&logoColor=white" alt="License">
</p>

基于 Python 的视频转 SfM 稀疏 / 稠密点云工作流。输入一段视频，输出稀疏点云或稀疏 + 稠密点云，可用 MeshLab、CloudCompare 等工具浏览。

**纯 OpenCV + SciPy + NumPy，无 PyTorch、无 CUDA、无外部 SfM / MVS 二进制**。抽帧、位姿估计、三角化、BA、回环、PGO、稠密立体匹配全部内置。

- **自研增量式 SfM**：ORB / SIFT → 本质矩阵初始化 → 三角化 / PnP 重定位 → EM 加权 BA → BoW 回环 → PGO，不依赖 COLMAP。
- **稠密重建**：相邻帧对 SGBM 立体匹配 + 深度过滤 + 体素降采样 + 统计离群点过滤，输出带 RGB 的点云。
- **智能采样**：均匀 / 光流驱动（smart）/ 两阶段（视差 + 光流 + 清晰度）三种策略，含时间覆盖兜底与分层加权。
- **帧对级缓存**：稠密中断后已算帧对落盘，下次启动自动跳过；正常完成不写盘，省磁盘寿命。
- **暗色主题 GUI**：相机轨迹 / 点云可视化、帧预览、日志输出，回环 / PGO 开关齐备。
- **无 GPU 依赖**：纯 CPU 即可完整运行稀疏与稠密重建。

> CLI 额外依赖 `psutil`；GUI 额外依赖 `PySide6`、`matplotlib`、`psutil`。

---

## 📦 安装

- Python 3.11，无 GPU 要求，纯 CPU 可运行。

```bash
# 可选：创建 Conda 环境
conda create -n sfm python=3.11
conda activate sfm

# 安装依赖
pip install -r requirements.txt
# 或手动安装
pip install numpy scipy opencv-python-headless PySide6 matplotlib "psutil>=5.9.0"

# 克隆仓库
git clone https://github.com/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction
```

> **注意**：OpenCV 必须用 `headless` 版本，避免与 PySide6 的 Qt 库冲突。GUI 用 PySide6 显示图像，headless 版本足够；CLI 只用 `psutil`，不会导入 PySide6 / matplotlib。

| 库 | 用途 |
|---|---|
| numpy | 矩阵运算、点云、位姿数组 |
| scipy | least_squares（BA / PGO）、expit、稀疏矩阵、cKDTree |
| opencv-python-headless | 抽帧、特征、匹配、光流、三角化、PnP、SGBM |
| PySide6 | 桌面 GUI（仅 GUI 需要） |
| matplotlib | 相机轨迹 / 点云可视化（Agg 后端，仅 GUI 需要） |
| psutil | CPU 亲和性设置、硬件探测 |

---

## 🚀 使用方式

### 1. 命令行接口（CLI）

```bash
# 只跑稀疏（默认）
python cli.py --video input.mp4 --output-dir ./sfm_output

# 稀疏 + 稠密全程
python cli.py --video input.mp4 --output-dir ./sfm_output --mode dense
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--video` | 输入视频路径 | 必填 |
| `--output-dir` | 输出目录（点云 / 位姿 / 内参） | `./sfm_output` |
| `--workdir` | 工作目录 | `./workdir` |
| `--mode` | `sparse` / `dense` | `sparse` |
| `--sampling-mode` | `uniform` / `smart` / `two-stage` | `uniform` |
| `--fps` | 采样帧率 | `15.0` |
| `--scale` | 画面缩放（0~1） | `0.5` |
| `--min-frames` / `--max-frames` | 最少 / 最多提取帧数 | `30` / `200` |
| `--feature-type` | `orb` / `sift` | `orb` |
| `--use-focal-guess` | 以 1.0×长边作初始像素焦距（约 53° FOV） | `False` |
| `--no-loop` | 关闭回环检测（短序列 / 纯前向拍摄可关） | 不传时开启 |
| `--no-pgo` | 关闭位姿图优化（无回环时收益有限） | 不传时开启 |
| `--dense-downscale` | 稠密帧降采样倍数 | `2` |
| `--dense-max-pairs` | 稠密最多处理帧对数 | `60` |
| `--dense-voxel-ratio` | 稠密体素大小 / 场景尺度 | `0.005` |
| `--reuse-dir` | 复用工作目录的中间产物，跳过已完成步骤 | `None` |

```bash
# 智能采样 + SIFT
python cli.py --video input.mp4 --mode sparse \
    --sampling-mode smart --feature-type sift

# 两阶段采样 + 稠密重建
python cli.py --video input.mp4 --mode dense \
    --sampling-mode two-stage \
    --dense-downscale 2 --dense-max-pairs 80

# 短序列 / 纯前向拍摄：关闭回环与 PGO 加速
python cli.py --video input.mp4 --no-loop --no-pgo

# 复用上次的帧 / 位姿缓存，只重跑导出
python cli.py --video input.mp4 --reuse-dir ./workdir
```

> `--reuse-dir` 指向的工作目录必须存在，CLI 会读取
> `frame_paths.txt` / `intrinsics.npy` / `poses.npy` / `sparse_points.npy`
> 以及可选的 `sfm_meta.json`；若抽帧参数与 `frame_meta.json` 不一致，
> 会自动重新抽帧并作废下游缓存。

### 2. 图形界面（GUI）

```bash
python gui.py
```

- 视频、输出目录、工作目录选择
- 运行模式：稀疏重建 / 稀疏 + 稠密
- 采样策略、帧率、缩放、帧数范围
- 特征描述子（ORB / SIFT）、初始焦距猜测、回环 / PGO 开关
- 稠密参数（降采样倍数、最多帧对、体素比例），仅在 dense 模式可见
- 帧缩略图预览、分页浏览（窗口缩放自适应布局）
- 相机轨迹 + 稀疏点云俯视图（重建完成后自动切换）
- 日志输出、中断保存缓存

> **默认值差异**：CLI 默认 `sparse` / `orb`，初始焦距猜测需显式开启；
> GUI 默认「稀疏重建」/「ORB（快）」，初始焦距猜测、回环、PGO 均默认开启。

---

## 🧩 核心模块与实现要点

| 模块 | 功能 |
|---|---|
| `frames.py` | 视频抽帧，uniform / smart / two-stage 采样；清晰度 + 光流门控，时间覆盖兜底与分层加权 |
| `poses.py` | 增量式 SfM（ORB / SIFT），E 矩阵初始化、三角化 / PnP 重定位、EM 加权 BA、BoW 回环、PGO、多视图 DLT 重三角化、尺度归一化 |
| `dense.py` | 稠密重建：SGBM 立体匹配 + 深度 / 基线过滤 + 体素降采样 + 统计离群点过滤；帧对级缓存续跑 |
| `exporter.py` | 二进制 PLY、`cameras.txt`、`intrinsics.json` 导出 |
| `frame_meta.py` | `frame_meta.json` 读写、抽帧参数校验、缓存复用判断与下游作废 |
| `gui.py` / `cli.py` | PySide6 暗色 GUI 与命令行入口，sparse / dense 双模式 |

**实现要点**

- **抽帧**：拉普拉斯方差清晰度 + Farneback / LK 光流，门控阈值随视频自适应，分层选择段内按 清晰度 × 光流 加权。
- **稠密**：`stereoRectify` + `remap` 校正，`reprojectImageTo3D` 反投影，深度 / 基线比例过滤；体素降采样用线性索引 + `np.bincount` 加权，离群点过滤用 `cKDTree` 近邻平均距离。
- **CPU 亲和性**：启动时自动绑定所有逻辑核心。

---

## 📋 输入数据要求

| 维度 | 建议 |
|---|---|
| 视角数 | 稀疏 ≥ 30 帧（推荐 50~60）；稀疏 + 稠密 ≥ 60 帧（推荐 100~200） |
| 帧间重叠 | 相邻两帧像素重叠 ≥ 70%；`--fps` 过高会引入冗余帧、拖慢 SfM |
| 分辨率 | 特征匹配 ≥ 480p；`--scale 0.5` 时 1080p 降为 960×540；稠密对分辨率更敏感 |
| 曝光 / 白平衡 | 建议锁定或预处理，自动跳变会干扰 SfM 与 BA；RAW 或手动曝光素材上限更高 |
| 环绕度 | 单面可见只能重建单面；360° 建议每 ~30° 一个机位；垂直方向缺失会导致地板 / 天花板欠拟合 |
| 场景 | SfM 假设静态场景，移动物体会污染匹配与三角化；反射 / 镜面表面重建效果差 |

---

## 🎯 特征描述子选择

| 特征 | 命令 | 适用场景 |
|---|---|---|
| ORB（默认） | `--feature-type orb` | 纹理丰富，速度最快 |
| SIFT | `--feature-type sift` | 纹理不足或短序列，更稳健但慢 |

- 短序列（< 60 帧）：ORB 后端足够。
- 低纹理 / 室内弱光：优先 SIFT。
- 不确定：先用默认 ORB。
- 回环检测与 PGO 均可通过 CLI / GUI 控制，默认开启。

---

## 🧠 稠密重建参数建议

| 参数 | 作用 | 建议 |
|---|---|---|
| `--dense-downscale` | 帧降采样倍数 | 1 = 精度最高，2 = 平衡，4 = 极快 |
| `--dense-max-pairs` | 最多处理帧对数 | 60~120；越多覆盖率越高，耗时线性增长 |
| `--dense-voxel-ratio` | 体素大小 / 场景尺度 | 0.003~0.01；越小越密，也越慢 |

- 稠密基于相邻帧对，视差不够大时深度不准；纯前向拍摄效果有限，建议环绕或多角度拍摄。
- 体素降采样后点数通常在百万级，`--dense-voxel-ratio 0.005` 是通用起点。
- 帧对选择：若提供关键帧且数量 ≥ 2，只用关键帧相邻对；否则用全部相邻帧对。超过 `max_pairs` 时均匀截断。

---

## 💾 缓存复用与中断续跑

工作目录内容：

| 文件 / 目录 | 内容 |
|---|---|
| `frame_paths.txt` | 帧路径列表 |
| `frame_meta.json` | 抽帧参数（video / scale / fps / sampling_mode / feature_type / min_frames / max_frames） |
| `intrinsics.npy` / `poses.npy` / `sparse_points.npy` | 内参、位姿、稀疏点云 |
| `sfm_meta.json` | 关键帧 / 回环 / 帧状态（缓存复用时用于还原） |
| `frames/` | 抽帧结果 |
| `dense_pairs/` | 稠密帧对缓存与 `meta.json` 指纹 |

### 稀疏阶段

```bash
python cli.py --video input.mp4 --reuse-dir ./workdir
```

- CLI 与 GUI 统一判断：`frame_paths.txt` 存在 **且** `frames/` 目录非空才复用抽帧缓存。
- 重新抽帧前会清理 `frames/` 下的旧 `frame_*.png`，避免残留。
- `intrinsics.npy` + `poses.npy` + `sparse_points.npy` 存在 → 跳过位姿估计。
- `sfm_meta.json` 存在时同时读回关键帧 / 回环 / 帧状态，稠密重建优先使用关键帧相邻对。
- `frame_meta.json` 存在时，抽帧参数变化会自动作废下游缓存（`intrinsics` / `poses` / `sparse_points` / `sfm_meta` / `dense_pairs`）；无该文件则不校验。

### 稠密阶段

- 主循环把已算完的帧对累积在内存，正常完成不写盘；用户主动中断（Ctrl+C / 点「停止」）时一次性批量落盘。
- 下次启动扫描 `dense_pairs/`，参数指纹一致时，已成功和已标记失败的帧对都跳过，只补缺失的帧对。
- 若要重算失败帧对，需删除 `dense_pairs/` 或修改参数指纹。
- 参数指纹覆盖 `downscale` / `num_disparities` / `block_size` / `min_disparity` / `scene_scale` / `n_frames` / 帧 mtime / **`intrinsics` + `poses` 内容哈希**；任意一项变化都会自动清空缓存。
- `voxel_ratio` / `outlier_std` / `max_points` 不参与指纹——帧对复用，只重跑后处理。
- `kill -9` 或断电无法拦截，内存里未落盘的帧对会丢失。

### 输出文件

| 文件 | sparse | dense | 说明 |
|---|:---:|:---:|---|
| `sparse_points.ply` | ✓ | ✓ | 仅含 XYZ，无 RGB |
| `cameras.txt` | ✓ | ✓ | 相机位姿（tx ty tz qx qy qz qw），缺失位姿的帧被跳过 |
| `intrinsics.json` | ✓ | ✓ | 内参 |
| `sfm_report.json` | ✓ | ✓ | 统计报告 |
| `dense_points.ply` | — | ✓ | 含 XYZ + RGB |

---

## ⚙️ 高级参数建议

- `--sampling-mode two-stage`：快速运动或视角变化剧烈时使用。
- `--feature-type sift`：低纹理 / 短序列更稳健，速度慢。
- `--no-loop` / `--no-pgo`：短序列或纯前向拍摄可关，加速并减少误回环风险。
- `--dense-downscale 1`：稠密精度最高，耗时最长。
- `--dense-max-pairs 120`：覆盖更多帧对，点云更完整。
- `--dense-voxel-ratio 0.003`：点云更密，输出文件更大。
- `--scale 0.25`：内存 / CPU 吃紧时降采样抽帧，加快 SfM。

---

## 📝 注意事项与限制

- 通常 100~200 帧效果较好；长序列建议 `--sampling-mode two-stage`，避免冗余帧。
- 稠密重建对纯旋转、纯前向拍摄效果有限，建议环绕拍摄。
- 启动时自动绑定所有逻辑核心；纯 CPU 运行，无需 CUDA 扩展。
- 回环检测与 PGO 默认开启；短序列或纯前向拍摄可关闭以加速。
- `SfMResult.poses` 中 LOST / INVALID 帧为 `None`，下游（稠密 / 导出 / GUI）据此跳过无效帧；`poses.npy` 以 NaN 行表示缺失位姿。
- 关键帧过多时，为控制 PGO 规模会剔除共视度过高的旧关键帧，相关位姿图边与回环记录同步作废。
- 动态场景 / 运动物体会产生重影或形变；反射 / 镜面表面重建效果不理想；稠密基于 SGBM，视差范围有限，超远距离深度估计会截断。
- 性能数字因硬件、分辨率、场景而异。

---

## 📄 许可证

Apache-2.0，欢迎自由使用和修改。

## 🙏 致谢

OpenCV、SciPy、NumPy、PySide6、matplotlib 等优秀开源库。

如有问题，欢迎提 Issue 或 PR。Happy Reconstructing!