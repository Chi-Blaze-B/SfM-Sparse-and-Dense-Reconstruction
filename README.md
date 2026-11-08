<h1 align="center"><strong>SfM Sparse &amp; Dense Reconstruction</strong></h1>

<p align="center">
  <!-- 项目 -->
  <img src="https://img.shields.io/badge/🌟_SfM_Reconstruction-Sparse_%26_Dense-FF6F00?style=flat-square&logo=github&logoColor=white" alt="Project">
  <br><br>
  <!-- 核心依赖 -->
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <br><br>
  <img src="https://img.shields.io/badge/NumPy-1.24.0%2B-4DABCF?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-1.10.0%2B-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/scikit--learn-1.3.0%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/OpenCV-headless_%7C_4.8.0%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/PySide6-6.5.0%2B-41CD52?style=flat-square&logo=qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/Matplotlib-3.7.0%2B-11557C?style=flat-square&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/psutil-5.9.0%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="psutil">
  <img src="https://img.shields.io/badge/joblib-1.3.0%2B-FFB000?style=flat-square" alt="joblib">
  <img src="https://img.shields.io/badge/pyceres-2.6.0%2B-00A98F?style=flat-square" alt="pyceres">
  <br><br>
  <!-- 核心功能 -->
  <img src="https://img.shields.io/badge/GUI-PySide6-41CD52?style=flat-square&logo=qt&logoColor=white" alt="GUI">
  <img src="https://img.shields.io/badge/SfM-ORB_%7C_SIFT_%7C_RootSIFT-00A98F?style=flat-square" alt="SfM">
  <img src="https://img.shields.io/badge/MVS-SGBM-FF4500?style=flat-square" alt="MVS">
  <br><br>
  <!-- 兼容性与许可证 -->
  <img src="https://img.shields.io/badge/Platform-Windows_%7C_Linux_%7C_macOS-0078D4?style=flat-square&logo=windows&logoColor=white" alt="Platform">
  <img src="https://img.shields.io/badge/License-Apache_2.0-1E90FF?style=flat-square&logo=apache&logoColor=white" alt="License">
  <br><br>
  <!-- 社区动态 -->
  <a href="https://github.com/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction/stargazers">
    <img src="https://img.shields.io/github/stars/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction?style=flat-square&color=yellow&logo=github" alt="Stars">
  </a>
  <a href="https://github.com/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction/network/members">
    <img src="https://img.shields.io/github/forks/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction?style=flat-square&color=blue&logo=github" alt="Forks">
  </a>
  <a href="https://github.com/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction/issues">
    <img src="https://img.shields.io/github/issues/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction?style=flat-square&color=red&logo=github" alt="Issues">
  </a>
  <a href="https://github.com/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction/commits">
    <img src="https://img.shields.io/github/last-commit/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction?style=flat-square&color=green&logo=github" alt="Last Commit">
  </a>
</p>

基于 Python 的视频 / 图片集转 SfM 稀疏 / 稠密点云工作流。输入一段视频或一个图片集目录，输出稀疏点云或稀疏 + 稠密点云，可用 MeshLab、CloudCompare 等工具浏览。

**核心为 OpenCV + NumPy + SciPy + pyceres（BA / PGO），回环词袋使用 scikit-learn MiniBatchKMeans；无 PyTorch、无 CUDA、无外部 SfM / MVS 二进制**。抽帧、位姿估计、三角化、BA、回环、PGO、稠密立体匹配全部内置。

- **自研增量式 SfM**：ORB / SIFT / RootSIFT → 最佳初始帧对（E 矩阵 + 退化检测）→ 按可见性金字塔覆盖分选下一帧 + 相邻帧跟踪 / 局部地图 PnP 重定位 → 多视图 DLT 重三角化 + 合并验证的点合并 → EM 加权 BA（默认优化 fx / fy，主点优化内部可选，全局 7-DOF gauge）→ TF-IDF BoW 回环 + F/H 几何验证 → PGO（Cauchy 核）→ 全局 BA，不依赖 COLMAP。
- **稠密重建**：相邻帧对 SGBM 立体匹配 + 深度过滤 + 体素降采样 + 统计离群点过滤，输出带 RGB 的点云。
- **智能采样**：均匀 / 光流驱动（smart）/ 两阶段（视差 + 光流 + 清晰度）三种策略，含时间覆盖兜底与分层加权。
- **帧对级缓存**：稠密中断后已算帧对落盘，下次启动自动跳过；正常完成不写盘，省磁盘寿命。
- **暗色主题 GUI**：相机轨迹 / 点云可视化、帧预览、日志输出，回环 / PGO / 全局匹配开关齐备。
- **无 GPU 依赖**：纯 CPU 即可完整运行稀疏与稠密重建。

> CLI 与 GUI 均依赖 `numpy` / `scipy` / `opencv-python-headless` / `scikit-learn` / `joblib` / `pyceres` / `psutil`；GUI 额外依赖 `PySide6`、`matplotlib`。`scikit-learn` 用于回环词袋 MiniBatchKMeans、`pyceres` 用于 BA / PGO，二者都在 `poses.py` 顶层导入，因此即使关闭回环也必须安装。

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
pip install numpy scipy opencv-python-headless scikit-learn joblib pyceres PySide6 matplotlib "psutil>=5.9.0"

# 克隆仓库
git clone https://github.com/Chi-Blaze-B/SfM-Sparse-and-Dense-Reconstruction
```

> **注意**：OpenCV 必须用 `headless` 版本，避免与 PySide6 的 Qt 库冲突。GUI 用 PySide6 显示图像，headless 版本足够；CLI 不会导入 PySide6 / matplotlib，但仍会通过 `poses.py` 顶层导入 `scikit-learn` / `joblib` / `pyceres`。

| 库 | 用途 |
|---|---|
| numpy | 矩阵运算、点云、位姿数组 |
| scipy | `cKDTree`（近邻查询）、`Rotation`（旋转表示转换） |
| opencv-python-headless | 抽帧、特征、匹配、光流、三角化、PnP、SGBM |
| scikit-learn | MiniBatchKMeans（回环词袋训练） |
| joblib | 特征提取多进程（loky 后端），按可用内存自动决定 worker 数 |
| pyceres | 光束法平差 / PGO / Sim3（自定义 Python CostFunction + 解析雅可比） |
| PySide6 | 桌面 GUI（仅 GUI 需要） |
| matplotlib | 相机轨迹 / 点云可视化（Agg 后端，仅 GUI 需要） |
| Pillow | 读 EXIF 的 35mm 等效焦距；matplotlib 已依赖，通常无需单独安装 |
| psutil | CPU 亲和性设置、硬件探测 |

### 预编译加速模块（可选，仓库自带）

`sfm_accel.pyd` 是 Cython 写的热点内核（批量投影、描述子指派、三角化角检查、地图点合并）。
**已随仓库提供预编译版本，不需要 Cython，也不需要编译器。** 它是自包含的，
只依赖 Windows 自带的 UCRT，不附带任何 MinGW 运行时 DLL。

| 项目 | 说明 |
|---|---|
| 适用环境 | CPython **3.11** / **win-amd64** |
| 环境不匹配时 | 导入失败 → `poses.py` 自动回退到纯 Python 实现，**结果完全一致**（128 张压测慢约 5%） |
| 怎么确认 | 启动日志里有 `[加速] Cython 内核: 已启用 (sfm_accel)` 或 `未启用…` |
| 强制关掉 | `set SFM_ACCEL=0` |
| 自行重编 | `pip install cython && python build_accel.py`（需 gcc / MinGW-w64） |

> 加速只针对数组与对象循环那一层。BA（pyceres）和特征匹配（OpenCV）本来就在 C++ 里，
> 换不换这个模块都不影响它们的耗时。

---

## 🚀 使用方式

### 1. 命令行接口（CLI）

```bash
# 只跑稀疏（默认）
python cli.py --video input.mp4 --output-dir ./sfm_output

# 稀疏 + 稠密全程
python cli.py --video input.mp4 --output-dir ./sfm_output --mode dense

# 直接指定图片集（与 --video 二选一）
python cli.py --image-dir ./images --output-dir ./sfm_output

# 图片集 + 标定（例如 ETH3D 的 THIN_PRISM_FISHEYE）
python cli.py --image-dir scene/images/dslr_images \
    --camera-matrix K.npy --dist-coeffs dist.npy
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--video` | 输入视频路径（与 `--image-dir` 二选一） | 必填其一 |
| `--image-dir` | 输入图片集目录（与 `--video` 二选一） | 必填其一 |
| `--output-dir` | 输出目录（点云 / 位姿 / 内参） | `./sfm_output` |
| `--workdir` | 工作目录 | `./workdir` |
| `--mode` | `sparse` / `dense` | `sparse` |
| `--sampling-mode` | `uniform` / `smart` / `two-stage`（仅视频） | `uniform` |
| `--fps` | 采样帧率（仅视频） | `15.0` |
| `--scale` | 画面缩放（0~1） | 视频 `0.5`，图片集 `1.0` |
| `--min-frames` / `--max-frames` | 最少 / 最多帧数（图片集：最多图片数，0=全部） | 视频 `30` / `200`，图片集 `2` / `0` |
| `--camera-matrix` | 3×3 内参矩阵 `.npy`（可选） | `None` |
| `--dist-coeffs` | 畸变系数 `.npy`；4/5/8 值走 OpenCV，12 值走 COLMAP THIN_PRISM_FISHEYE | `None` |
| `--feature-type` | `orb` / `sift` / `rootsift` | `orb` |
| `--use-focal-guess` | 以 1.0×长边作初始像素焦距（约 53° FOV）；会覆盖 EXIF | `False` |
| `--no-loop` | 关闭回环检测（短序列 / 纯前向拍摄可关） | 不传时开启 |
| `--no-pgo` | 关闭位姿图优化（无回环时收益有限） | 不传时开启 |
| `--enable-global-matching` | 启用全局词袋检索匹配（更稳但更慢） | `False` |
| `--dense-downscale` | 稠密帧降采样倍数 | `2` |
| `--dense-max-pairs` | 稠密最多处理帧对数 | `60` |
| `--dense-voxel-ratio` | 稠密体素大小 / 场景尺度 | `0.005` |
| `--reuse-dir` | 复用工作目录的中间产物，跳过已完成步骤 | `None` |
| `--frame-cache-mb` | 帧图像内存缓存上限（MiB）；0 表示禁用。只在稠密阶段预加载，位姿估计阶段不用缓存 | `1500` |
| `--no-self-calibrate` | 关闭无标定输入的 k1 自标定 | 不传时开启 |
| `--self-calib-iter` | 自标定后再估一次残余 k1，显著时跑第三遍（+20% 耗时） | 不传时关闭 |
| `--rot-edge-weight` | 最终 BA 中两视图旋转约束的权重；0 关闭 | `100` |
| `--pose-outlier-deg` | 位姿校验阈值（与邻居的最小相对旋转偏差）；0 关闭 | `12` |
| `--dense-min-views` | 稠密融合所需最少帧对数；2 可去重但更稀疏 | `1` |

图片集模式的约定：

- 目录内的图片按**文件名自然排序**（`DSC_2` 排在 `DSC_10` 之前）；
- 顶层没有图片时会**自动递归**查找子目录，因此可以直接指向
  ETH3D 的 `<scene>` 或 `<scene>/images`；
- `--scale 1.0`（默认）**直接引用原图**，不复制、不解码，不会占双份磁盘；
  `--scale < 1` 时才解码缩放并写入 `workdir/frames/frame_XXXXX.png`；
- 图片集模式下 `--sampling-mode` / `--fps` / `--min-frames` 不生效；
  `--max-frames > 0` 表示按自然顺序**均匀抽取**该数量的图片。

**焦距与内参的来源**（优先级从高到低）：

| 来源 | 何时生效 | 说明 |
|---|---|---|
| `--camera-matrix` / `--dist-coeffs` | 显式提供标定 | 最好，且省掉自标定的第二遍；不提供时会自标定 k1（见下） |
| `--use-focal-guess` | 显式要求用 1.0×长边 | 约 53° FOV 的先验，会覆盖 EXIF |
| 图片 EXIF 的 35mm 等效焦距 | 默认；图片带 EXIF 时 | 这组图推出 fx=2485，真值 2560（**-2.9%**） |
| 内置启发（1.2×长边） | 无 EXIF（截图 / 视频抽帧） | 约 41mm 等效，估不准 |

> EXIF 读取依赖 Pillow（matplotlib 已依赖，装完依赖就有）；读不到时静默退回启发值。
> **有标定一定要给**——相机模型（尤其径向畸变 k1）对轨迹精度的影响大于任何其他参数。

**径向畸变自标定**：完全没有内参输入时，BA 会把 `k1` 当共享参数一起估计（SIMPLE_RADIAL，
带弱先验）。若估计出的 `|k1| ≥ 0.004` 且注册帧足够多，会自动用**去畸变后的特征再跑一遍**，
让 PnP / 三角化 / 过滤也处在一致的针孔模型下——代价是**耗时约翻倍**。用 `--no-self-calibrate`
或 GUI 复选框关闭。有标定数据时请直接给 `--dist-coeffs`。

另有一个默认关闭的迭代选项（CLI `--self-calib-iter` / GUI「k1 迭代」）：在第二遍之后再估一次残余 k1，
显著时用合成值跑第三遍。它能进一步压低径向形变，但会改变 k1 的收敛路径、增加约 20% 耗时，
且不保证旋转指标同步改善，因此默认关闭；能接受这个权衡时再开启。

```bash
# 智能采样 + SIFT
python cli.py --video input.mp4 --mode sparse \
    --sampling-mode smart --feature-type sift

# 两阶段采样 + 稠密重建
python cli.py --video input.mp4 --mode dense \
    --sampling-mode two-stage \
    --dense-downscale 2 --dense-max-pairs 80

# 长序列 / 回环场景：开启全局匹配
python cli.py --video input.mp4 --enable-global-matching

# 低纹理 / 室内弱光：SIFT + 全局匹配
python cli.py --video input.mp4 --feature-type sift --enable-global-matching

# 短序列 / 纯前向拍摄：关闭回环与 PGO 加速
python cli.py --video input.mp4 --no-loop --no-pgo

# 图片集：只取 50 张、缩放到一半、跑稠密
python cli.py --image-dir ./images --scale 0.5 --max-frames 50 --mode dense

# 复用上次的帧 / 位姿缓存，只重跑导出
python cli.py --video input.mp4 --reuse-dir ./workdir
```

> `--reuse-dir` 指向的工作目录必须存在，CLI 会读取
> `frame_paths.txt` / `intrinsics.npy` / `poses.npy` / `sparse_points.npy`
> 以及可选的 `sfm_meta.json`；若输入或抽帧参数与 `frame_meta.json` 不一致，
> 会自动重新准备帧并作废下游缓存。

### 2. 图形界面（GUI）

```bash
python gui.py
```

- 视频 / 图片集输入切换（视频文件或图片集目录）、输出目录、工作目录选择
- 运行模式：稀疏重建 / 稀疏 + 稠密
- 采样策略、帧率、缩放、帧数范围（图片集模式下自动隐藏仅视频有效的选项，最多图片数 0 表示全部）
- 特征描述子（ORB / SIFT / RootSIFT）、初始焦距猜测、回环 / PGO / 全局匹配开关
- 内参来源：自动（EXIF 35mm 等效焦距 → 1.2×长边启发）/ 手动指定相机矩阵与畸变系数 `.npy`
- 无内参时自标定径向畸变 k1（估计显著时会自动去畸变重跑一遍），可选残余 k1 迭代
- 位姿校验阈值、两视图旋转约束权重（对照「与邻居旋转是否自洽」「是否压住累积漂移」）
- 稠密参数（降采样倍数、最多帧对、体素比例），仅在 dense 模式可见
- 帧图像内存缓存上限（MiB，0 禁用）
- 帧缩略图预览、分页浏览（窗口缩放自适应布局）
- 相机轨迹 + 稀疏点云俯视图（重建完成后自动切换）
- 日志输出、中断保存缓存

> **默认值差异**：CLI 默认 `sparse` / `orb`，初始焦距猜测、全局匹配需显式开启；
> GUI 默认「稀疏重建」/「ORB（快）」，初始焦距猜测、回环、PGO 均默认开启，
> 全局匹配默认关闭。
> CLI 与 GUI 默认帧缓存上限均为 1500 MiB；CLI 可用 `--frame-cache-mb 0` 禁用，
> GUI 可将性能页缓存上限设为 0。全局匹配依赖回环 BoW，关闭回环时不会生效，GUI 里取消「回环检测」后「全局匹配」会自动置灰。

---

## 🧩 核心模块与实现要点

| 模块 | 功能 |
|---|---|
| `frames.py` | 视频抽帧（uniform / smart / two-stage 采样；清晰度 + 光流门控，时间覆盖兜底与分层加权）与图片集导入（自然排序、递归查找、按需缩放） |
| `poses.py` | 增量式 SfM（ORB / SIFT / RootSIFT），最佳初始帧对 + 退化检测、可见性金字塔选帧、相邻帧跟踪 + 局部地图 PnP、F/H 几何验证、多视图 DLT 重三角化、带验证的点合并、EM 加权 BA（fx / fy；主点内部可选；无标定时自标定 k1；全局 7-DOF gauge）、MiniBatchKMeans BoW 回环 + 时间一致性、PGO（加权 Cauchy）、全局 BA |
| `dense.py` | 稠密重建：SGBM 立体匹配 + 深度 / 基线过滤 + 体素降采样（可按命中帧对数做融合过滤）+ 统计离群点过滤；帧对级缓存续跑 |
| `exporter.py` | 二进制 PLY、`cameras.txt`、`intrinsics.json` 导出 |
| `frame_meta.py` | `frame_meta.json` 读写、抽帧参数校验、缓存复用判断与下游作废 |
| `frame_cache.py` | 进程内线程安全 LRU 帧图像内存缓存；支持预加载、命中率统计，CLI / GUI 共用。特征提取在 spawn 出来的子进程里跑且每帧只读一次，因此子进程缓存关闭、父进程预加载推迟到稠密阶段 |
| `gui.py` / `cli.py` | PySide6 暗色 GUI 与命令行入口，sparse / dense 双模式 |
| `sfm_accel.pyx` / `build_accel.py` | Cython 加速内核（批量投影、描述子指派、三角化角检查、地图点合并）与构建脚本；仓库自带 CPython 3.11 / win-amd64 的预编译 `sfm_accel.pyd`，导入失败时 `poses.py` 自动回退纯 Python，结果完全一致 |

**实现要点**

- **初始化**：在起始窗口内枚举所有候选帧对，按内点数打分选出最佳初始 pair；用 E 矩阵恢复相对位姿，并做正深度比例 + 中位视差角退化检测，失败可自动换下一对。
- **选帧顺序**：主循环按「与已注册帧的可见重叠」挑下一帧——把能看到已有 3D 点的关键点投影到 6 层可见性金字塔上算覆盖分，偏好空间铺得开的视角，而不是单纯点最多的视角；并列按帧索引，参考帧可取索引两侧。
- **匹配与验证**：ORB 用 Hamming + ratio test，SIFT / RootSIFT 用 L2；描述子匹配采用互最近邻 + 双向 ratio；关键决策（回环、局部 PnP 候选）用 F/H RANSAC 做几何验证。新建地图点前同样过 F/H 验证，补观测 / 补 track 要求重投影 **< 3 px**。三角化角判据是「任意一对相机达到阈值即可」：初始化 16°、点过滤 1.5°、局部 BA 选窗 6°。
- **位姿校验**：末尾用两视图 E 矩阵独立估计每帧与最近邻居的相对旋转，和地图里的相对旋转比对；与所有邻居的最小偏差超过阈值（默认 12°）的帧会被标记为 LOST，不再出现在输出里。这一步专门抓「重投影很小但位姿错了」的帧——它们和错误的三维点互相自洽，BA 看不出来。校验只标记输出、不改地图，避免影响后续 BA 与自标定。随后按共视关系聚类，把与主图几乎不共享地图点、且规模远小于主体的帧块整体从输出中摘除。
- **BA**：pyceres（自定义 Python CostFunction，解析雅可比）；默认最小二乘 + EM 软内点加权（鲁棒核为可选开关）；默认联合优化 `fx` / `fy`，主点优化为内部可选且当前入口默认关闭；代价上升时自动回滚本轮。全局 BA 用 7 自由度 Sim3 gauge——固定一机，另一机只固定基线主轴方向的一维平移，避免多钉自由度把两台参考机之间的微小不一致变成内应力。无标定数据时 `k1` 作为共享参数一起优化（SIMPLE_RADIAL），估计显著则去畸变重跑一遍。局部 BA 会把**窗口外帧对窗口内地图点的观测**作为额外残差（位姿固定），给窗口边界上的点外部约束；重三角化之后接一次局部 BA，最终 BA 前也先重三角化。最终全局 BA 还会加入**相邻关键帧的两视图相对旋转约束**（E 矩阵，独立于地图，权重 `--rot-edge-weight`，默认 100，设 0 关闭），并在求解前按残差剔除异常旋转边——回环和 PGO 的测量都打在地图点上，只能让模型自洽，纠正不了累积旋转。
- **回环**：MiniBatchKMeans 训练视觉词 + TF-IDF 加权直方图 + 余弦相似度检索；候选通过 F/H RANSAC、PnP RANSAC 与序列一致性验证。
- **地图管理**：周期性多视图 DLT 重三角化；点合并基于描述子 + 3D 位置，合并前按 track 长度加权定位置、并用全部观测做重投影验证（>4 px 或内点 <90% 则不合并）；先剔除坏观测再删点，`feat_map` 与观测同步更新。
- **PGO**：顺序边与回环边带不同权重（回环边权重随共视计数调整），Cauchy 核；PGO 之后再跑一次全局 BA，提升重投影一致性。
- **全局匹配**：基于回环 BoW 词袋在历史关键帧中检索候选；依赖回环词袋已训练，关闭回环时不会生效。
- **抽帧**：拉普拉斯方差清晰度 + Farneback / LK 光流，门控阈值随视频自适应，分层选择段内按 清晰度 × 光流 加权。
- **稠密**：`stereoRectify` + `remap` 校正，`reprojectImageTo3D` 反投影，深度 / 基线比例过滤；体素降采样用线性索引 + `np.bincount` 加权，离群点过滤用 `cKDTree` 近邻平均距离。
- **CPU 亲和性**：启动时自动绑定所有逻辑核心。
- **性能**：热点循环已向量化或 Cython 化 —— 补 track 用「一次批量投影 + 一次批量 kd-tree 查询 + C 里逐点指派」，点合并的逐对判据进 Cython，BoW 量化用 `‖d‖²+‖w‖²−2d·w` 的单次 BLAS 乘代替 GB 级广播。帧缓存只在稠密阶段预加载（特征提取每帧只解码一次，预加载收益很小，还会挤占特征提取的并行预算）。south-building 128 张 @1600 的参考耗时：约 10 分钟（关闭自标定）/ 约 21 分钟（默认两遍自标定），注册 102/128；实际耗时随硬件、分辨率和可用内存变化。

---

## 📋 输入数据要求

| 维度 | 建议 |
|---|---|
| 视角数 | 稀疏 ≥ 30 帧（推荐 50-60）；稀疏 + 稠密 ≥ 60 帧（推荐 100-200） |
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
| RootSIFT | `--feature-type rootsift` | SIFT 归一化版本，对曝光变化更鲁棒；**阈值未按新管线重新标定，暂无优势数据** |

- 短序列（< 60 帧）：ORB 后端足够。
- 低纹理 / 室内弱光：优先 **SIFT**。
- 匹配困难、回环易误检：SIFT + `--enable-global-matching`。
- 不确定：先用默认 ORB。
- 回环检测、PGO、全局匹配均可通过 CLI / GUI 控制，回环与 PGO 默认开启，全局匹配默认关闭。
- 全局匹配依赖回环 BoW，需保持回环开启；关闭回环时，全局匹配不会生效。

> **RootSIFT 说明**：RootSIFT 的匹配阈值尚未在当前管线上重新标定，也没有显示它优于 SIFT 的结果。
> 不确定时先用 `sift`；需要尝试时再用 `rootsift`。

---

## 🌐 全局匹配

`--enable-global-matching` 会启用基于 BoW 的全局检索匹配：除相邻帧外，还会在历史关键帧中检索共视候选做描述子匹配，用于提升宽基线、长序列、回环场景的稳定性。

> 注意：该功能复用回环检测的 BoW 词袋；若关闭回环检测（`--no-loop` 或 GUI 取消回环），词袋不会训练，全局匹配不会生效（GUI 里会直接置灰该开关）。
> 词袋要等关键帧达到 15 个才开始训练，所以短序列上这个开关可能整个流程都不生效。
> 日志里出现 `[global] 全局匹配开始生效` 表示它真正开始工作；一直没有这行就是没跑起来。

| 场景 | 建议 |
|---|---|
| 短序列（< 60 帧）、纯前向拍摄 | 关闭（默认），速度优先 |
| 长序列（≥ 200 帧） | 开启，避免只靠相邻帧断链 |
| 大回环 / 环绕拍摄 | 开启 |
| 低纹理 / 匹配数偏低 | 开启，并配合 SIFT |

开启后整体耗时会明显增加，具体取决于关键帧数量与检索窗口。

---

## 🧠 稠密重建参数建议

| 参数 | 作用 | 建议 |
|---|---|---|
| `--dense-downscale` | 帧降采样倍数 | 1 = 精度最高，2 = 平衡，4 = 极快 |
| `--dense-max-pairs` | 最多处理帧对数 | 60~120；越多覆盖率越高，耗时线性增长 |
| `--dense-voxel-ratio` | 体素大小 / 场景尺度 | 0.003~0.01；越小越密，也越慢 |
| `--dense-min-views` | 一个体素至少被多少个帧对命中才保留 | 默认 1（不过滤）；2 去掉单帧对的重复点，点云会更稀疏 |

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
| `sfm_meta.json` | 关键帧 / 回环 / 帧状态 / 畸变系数（缓存复用时用于还原） |
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
- `--feature-type sift`：低纹理 / 短序列更稳健，速度慢（RootSIFT 阈值未按新管线复测，建议先用 sift）。
- `--enable-global-matching`：长序列、宽基线、回环场景建议开启；需保持回环开启。
- `--no-loop` / `--no-pgo`：短序列或纯前向拍摄可关，加速并减少误回环风险。
- `--dense-downscale 1`：稠密精度最高，耗时最长。
- `--dense-max-pairs 120`：覆盖更多帧对，点云更完整。
- `--dense-voxel-ratio 0.003`：点云更密，输出文件更大。
- `--scale 0.25`：内存 / CPU 吃紧时降采样抽帧，加快 SfM。
- `--frame-cache-mb 0`：禁用帧图像内存缓存，降低内存占用（缓存只在稠密阶段生效，位姿估计阶段本来就不预加载）。

---

## 📝 注意事项与限制

- 通常 100~200 帧效果较好；长序列建议 `--sampling-mode two-stage`，避免冗余帧。
- 稠密重建对纯旋转、纯前向拍摄效果有限，建议环绕拍摄。
- 启动时自动绑定所有逻辑核心；纯 CPU 运行，无需 CUDA 扩展。
- 回环检测与 PGO 默认开启；短序列或纯前向拍摄可关闭以加速。全局匹配默认关闭；全局匹配依赖回环 BoW，关闭回环时不会生效。
- 回环词袋使用 scikit-learn MiniBatchKMeans，因此 scikit-learn 为必需依赖。
- `SfMResult.poses` 中 LOST / INVALID 帧为 `None`，下游（稠密 / 导出 / GUI）据此跳过无效帧；`poses.npy` 以 NaN 行表示缺失位姿。
- 初始化在起始窗口内自动选最佳初始帧对（不固定第 0 帧）；`keyframes[0]` 不一定为 0。
- 关键帧过多时，为控制 PGO 规模会把共视度过高的旧关键帧从关键帧列表里移除；它仍是已注册帧，观测继续参与 BA，若仍作为回环端点会补回位姿图再参与 PGO。
- 动态场景 / 运动物体会产生重影或形变；反射 / 镜面表面重建效果不理想；稠密基于 SGBM，视差范围有限，超远距离深度估计会截断。
- 性能数字因硬件、分辨率、场景而异。

---

## 📄 许可证

Apache-2.0，欢迎自由使用和修改。

仓库内的预编译二进制 `sfm_accel.pyd` 由同目录的 `sfm_accel.pyx` 编译产生，适用同样的
Apache-2.0 许可。它只依赖 Windows 自带的 UCRT，不附带 GCC 的 GPL 运行时库
（`libgomp` / `libwinpthread` 均未使用）。

## 🙏 致谢

OpenCV、SciPy、NumPy、scikit-learn、PySide6、matplotlib 等优秀开源库。

如有问题，欢迎提 Issue 或 PR。Happy Reconstructing!