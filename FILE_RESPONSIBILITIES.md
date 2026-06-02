# 文件职责扫描报告

## 项目概述
**项目名称**: group5-canny-cutile  
**目标**: 使用 cuTile（NVIDIA GPU 并行编程 DSL）实现并优化 Canny 边缘检测算法  
**技术栈**: Python, NumPy, CuPy, cuTile, OpenCV, CUDA  
**团队成员分工**:
- jiaxi liu: cuTile GPU 实现 + 实时视频
- xudong ma: 纯 Python Baseline + 性能测试 + 报告素材
- shiying yang: 依赖分析、参数优化、集成

---

## 文件职责详解

### 📁 根目录文件

#### `README.md`
- **职责**: 项目文档和使用说明
- **内容**: 
  - 项目概述和 cuTile 背景
  - 完整的 Canny 管道架构说明
  - 文件结构描述
  - 运行方式和命令示例
  - cuTile 实现细节

#### `assignmentProjects2026.md`
- **职责**: 课程作业描述文档
- **内容**: SoftEng 751 2026 的三个项目选项说明（Canny Edge Detection 是其中之一）

#### `requirement.txt`
- **职责**: Python 依赖项列表
- **依赖**: numpy, scipy, matplotlib, opencv-python, pillow, cupy

#### `bsds500_canny_test.py`
- **职责**: 可靠性测试和验证
- **功能**:
  - 从 Berkeley Segmentation Dataset 500 下载测试图像
  - 与 OpenCV 的 cv2.Canny 进行对比验证
  - 包含纯 NumPy 的完整 Canny 实现（不依赖 GPU）
  - 计算每张图像和整体的可靠性指标
  - 可视化对比结果

#### `LICENSE`
- **职责**: 项目许可证

#### `.gitignore`
- **职责**: Git 版本控制配置

#### `.idea/`
- **职责**: PyCharm IDE 配置文件（自动生成）

#### `__pycache__/`
- **职责**: Python 字节码缓存（自动生成）

#### `venv/`
- **职责**: Python 虚拟环境目录

#### `.git/`, `.claude/`
- **职责**: 版本控制和 AI 助手配置

---

### 📁 `data/` 目录
- **职责**: 测试图像存储
- **文件**:
  - `IMG_6860.JPG` - 测试图像 1
  - `test.jpg` - 测试图像 2（主要用于快速测试）

---

### 📁 `report/` 目录
- **职责**: 基准测试结果、输出图像和分析资料存储
- **内容类型**:
  - PNG 图像: 中间处理结果（模糊、梯度、NMS、最终边缘等）
  - CSV 文件: 性能基准数据
  - 包含多个阶段的对比: CPU vs GPU vs cuTile 实现

---

### 📁 `notebooks/` 目录

#### `01_canny_baseline 1.ipynb`
- **职责**: 纯 NumPy Canny 基线实现
- **内容**:
  - 完整的 Python Canny 算法实现
  - 中间结果可视化
  - 性能基准测试
  - 教学和参考用途

#### `pure_numpy_fixed .ipynb`
- **职责**: 修复后的纯 NumPy 实现
- **用途**: 调试和改进基线算法

---

### 📁 `src/gpu/` 目录 - 核心实现文件

#### **基础工具库**

##### `gaussian_benchmark.py`
- **职责**: Gaussian 模糊的基准测试和实现
- **主要函数**:
  - `load_grayscale_image()` - 加载灰度图像
  - `make_gaussian_kernel()` - 生成 1D 高斯核（可分离卷积）
  - `gaussian_blur_cpu()` - CPU 实现
  - `gaussian_blur_gpu_compute_only()` - CuPy GPU 实现
  - `save_uint8_image()` - 保存图像
  - 命令行基准测试界面
- **特点**: 分离式 1D 高斯模糊（高效）

##### `sobel_benchmark.py`
- **职责**: Sobel 梯度计算的基准测试
- **主要函数**:
  - `load_grayscale_image()` - 加载图像
  - `sobel_cpu()` - CPU 实现（计算 Gx, Gy, 幅度, 角度）
  - `sobel_gpu_compute_only()` - CuPy GPU 实现
  - `normalize_to_uint8()` - 标准化到 uint8 范围
  - 详细的性能对比（CPU vs GPU）
- **基准对象**: Sobel 3×3 卷积核

##### `canny_frontend_benchmark.py`
- **职责**: Canny 前端（高斯模糊 + Sobel）基准测试
- **主要函数**:
  - `canny_frontend_cpu()` - CPU 实现
  - `canny_frontend_gpu_compute_only()` - GPU 实现
  - 对比前两个管道阶段的性能
- **用途**: 评估 Gaussian + Sobel 的性能

##### `nms_benchmark.py`
- **职责**: 非极大值抑制（Non-Maximum Suppression）基准测试
- **主要函数**:
  - `non_max_suppression_cpu()` - CPU 实现
  - `non_max_suppression_gpu()` - CuPy GPU 实现
  - 完整的 Canny 到 NMS 的管道
- **用途**: 评估 NMS 阶段的性能

---

#### **cuTile 优化实现**

##### `sobel_cutile_benchmark.py`
- **职责**: cuTile Sobel 内核实现和基准测试
- **主要组件**:
  - `@ct.kernel sobel_magnitude_from_neighbors_cutile()` - cuTile Sobel 核函数
  - `launch_sobel_cutile()` - 启动 cuTile 核
  - `sobel_magnitude_cupy_compute_only()` - CuPy 计算参考
  - 平铺大小扫描（64, 128, 256, 512）
- **特点**: 
  - 邻域像素预提取优化（避免线程间共享数据）
  - 与 CuPy 性能对比
- **基准参数**: 多个平铺大小的详细测试

##### `vector_add_cutile.py`
- **职责**: cuTile 学习和验证示例
- **内容**: 
  - 简单的向量加法 cuTile 实现
  - 用于验证 cuTile 环境和基础概念
- **用途**: 开发早期的 cuTile 验证代码

---

#### **GPU 基线实现**

##### `sobel_gpu_baseline.py`
- **职责**: CuPy 版 Sobel 基线
- **用途**: 在使用 cuTile 之前建立 GPU 参考实现
- **内容**: Sobel 梯度的 CuPy 实现

##### `sobel_cupy_only.py`
- **职责**: 专门的 CuPy Sobel 实现
- **内容**: Sobel 幅度和角度的 CuPy 计算

---

#### **参考实现**

##### `canny_pipeline.py`
- **职责**: CuPy 参考 Canny 管道实现
- **内容**:
  - 完整的 Canny 算法（使用 CuPy）
  - 包括：Gaussian 模糊、Sobel、NMS、阈值、滞后
  - 用于对比和验证
- **用途**: 基准参考实现

---

#### **完整管道实现**

##### `cutile_full_pipeline.py`
- **职责**: 最终优化的 Canny 完整管道
- **管道结构**:
  - Gaussian 模糊: cuTile (k=5) 或 CuPy 回退
  - Sobel: cuTile 或 CuPy 回退
  - NMS: CuPy
  - 阈值: NumPy
  - 滞后: CPU (OpenCV 连通分量)
- **命令行界面**:
  - `--image` - 输入图像路径
  - `--runs` - 基准测试次数
  - `--tile-size` - cuTile 平铺大小
  - `--output` - 输出边缘图像路径
  - `--no-display` - 无 GUI 模式
- **特点**: 最大化 cuTile 覆盖率，必要时回退

##### `cutile_canny_pipeline_benchmark.py`
- **职责**: 完整管道基准测试（通过 NMS）
- **主要函数**:
  - `sobel_angle_cupy_compute_only()` - Sobel 角度计算
  - `non_max_suppression_cupy_compute_only()` - NMS GPU 版
  - `canny_pipeline_cpu()` - CPU 完整管道
  - `canny_pipeline_gpu_with_input_transfer()` - GPU 管道含数据传输计时
- **测试覆盖**: 
  - 高斯模糊、Sobel、NMS 三个主要阶段
  - 详细的性能分析

##### `complete_canny_benchmark.py`
- **职责**: 从加载到最终边缘的完整 Canny 基准测试
- **功能**:
  - `select_thresholds()` - 智能阈值选择（百分位或固定值）
  - `hysteresis_cpu()` - 滞后处理（CPU）
  - 完整的端到端基准测试
  - 多个运行和统计
- **输出**: 详细的性能和结果分析

---

#### **实时应用**

##### `video_stream_demo.py`
- **职责**: 实时视频流 Canny 边缘检测演示
- **功能**:
  - 支持摄像头输入（整数索引）
  - 支持视频文件输入
  - 支持单张图像循环播放模式
  - 实时帧处理和性能统计
  - CSV 性能日志记录
- **命令行选项**:
  - `--source` - 摄像头索引或视频路径
  - `--image-loop` - 循环播放的图像路径
  - `--max-frames` - 最大处理帧数
  - `--no-display` - 无显示模式
  - 帧率显示和输出配置
- **用途**: 展示 Canny 的实时应用能力

---

## 工作流总结

### 📊 开发阶段
1. **基线开发** (`notebooks/`, `bsds500_canny_test.py`)
   - 纯 Python 实现
   - 功能验证

2. **GPU 基线** (`*_benchmark.py` 工具库)
   - 单个阶段的 CPU vs GPU 对比
   - 建立性能基准

3. **cuTile 优化** (`*_cutile_benchmark.py`, `*_cutile_pipeline*`)
   - 用 cuTile 替换 GPU 内核
   - 平铺参数优化
   - 性能测试

4. **整合** (`cutile_full_pipeline.py`)
   - 完整管道集成
   - 错误处理和回退机制

5. **应用演示** (`video_stream_demo.py`)
   - 实时处理演示
   - 端到端验证

### 📈 测试和基准策略
- **单阶段基准**: 评估单个组件（高斯、Sobel、NMS）
- **前端基准**: 评估前两个阶段
- **完整基准**: 评估完整管道
- **可靠性测试**: 与 OpenCV 对比验证
- **性能调优**: 平铺大小扫描和参数优化

### 🎯 关键优化点
1. **分离式高斯模糊**: 1D 水平 + 1D 垂直通道
2. **邻域预提取**: Sobel 在 cuTile 中避免线程间通信
3. **平铺参数**: 多个平铺大小的性能对比
4. **数据传输优化**: 包含和不包含内存传输的基准
5. **回退机制**: 在 cuTile 不可用时使用 CuPy/NumPy

---

## 核心算法管道

```
输入图像
    ↓
[1] Gaussian 模糊 (k=5, σ=1.4)
    ├─ cuTile 实现 (优先)
    └─ CuPy 回退
    ↓
[2] Sobel 梯度
    ├─ 计算 Gx, Gy
    ├─ 幅度 = √(Gx² + Gy²)
    ├─ 角度 = atan2(Gy, Gx)
    ├─ cuTile 实现 (优先)
    └─ CuPy 回退
    ↓
[3] 非极大值抑制 (NMS)
    └─ CuPy 实现
    ↓
[4] 双阈值处理
    └─ NumPy (百分位 or 固定值)
    ↓
[5] 滞后处理
    └─ CPU + OpenCV (连通分量)
    ↓
输出边缘图像 (uint8)
```

---

## 文件依赖关系

```
核心库:
├─ gaussian_benchmark.py (高斯模糊工具)
├─ sobel_benchmark.py (Sobel 工具)
├─ sobel_cutile_benchmark.py (cuTile Sobel)
└─ sobel_cupy_only.py (CuPy Sobel)

单阶段基准:
├─ gaussian_benchmark.py
├─ sobel_benchmark.py
├─ sobel_cupy_only.py
└─ sobel_gpu_baseline.py

管道基准:
├─ canny_frontend_benchmark.py
│   ├─ gaussian_benchmark.py
│   └─ sobel_benchmark.py
├─ nms_benchmark.py
│   ├─ canny_frontend_benchmark.py
│   └─ gaussian_benchmark.py
├─ cutile_canny_pipeline_benchmark.py
│   ├─ gaussian_benchmark.py
│   └─ sobel_cutile_benchmark.py
└─ complete_canny_benchmark.py
    └─ cutile_canny_pipeline_benchmark.py

完整实现:
├─ cutile_full_pipeline.py
│   ├─ gaussian_benchmark.py
│   └─ sobel_cutile_benchmark.py
└─ canny_pipeline.py

应用:
└─ video_stream_demo.py
    ├─ cutile_canny_pipeline_benchmark.py
    └─ gaussian_benchmark.py

验证:
└─ bsds500_canny_test.py (独立，自包含)
```

---

## 使用场景

| 文件 | 主要使用场景 | 用户角色 |
|------|-----------|--------|
| `cutile_full_pipeline.py` | 快速处理单张图像 | 最终用户 |
| `video_stream_demo.py` | 实时演示应用 | 演示/展示 |
| `bsds500_canny_test.py` | 验证算法正确性 | QA/验证 |
| `*_benchmark.py` | 性能分析 | 开发者/优化 |
| `notebooks/` | 学习和教学 | 学生/新人 |
| `canny_pipeline.py` | 参考实现 | 开发者/对比 |

---

生成时间: 2026-05-12
