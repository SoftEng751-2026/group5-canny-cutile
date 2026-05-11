# group5-canny-cutile
Canny Edge Detection implementation with cuTile for SoftEng 751 2026


## 项目分工
- jiaxi liu:  cuTile GPU 实现 + 实时视频
- xudong ma :纯 Python Baseline + 性能测试 + 报告素材
- shiying yang: 依赖分析、参数优化、集成

## 当前进度 (2026.4)
- ✅ 纯 Python Canny Baseline 已完成（包含所有步骤 + 中间结果可视化 + Benchmark）
- 测试图片位于 `data/` 文件夹
- 中间结果保存在 `report/` 文件夹

## 如何运行 Baseline

```bash
# 1. 激活环境
.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行 Jupyter Notebook
jupyter notebook
```

---

## Overview

This project implements Canny edge detection optimised with [cuTile](https://docs.nvidia.com/cuda/cutile-python/), NVIDIA's Python DSL for tile-parallel GPU programming, developed for SoftEng 751.

## Pipeline Architecture

| Stage | Implementation | Notes |
|-------|---------------|-------|
| Gaussian blur | **cuTile** (k=5) / CuPy fallback | Separable 1D passes |
| Sobel magnitude | **cuTile** / CuPy fallback | 3×3 stencil via neighbour arrays |
| Sobel angle | CuPy | Required for NMS direction |
| Non-maximum suppression | CuPy | Boolean mask operations |
| Double threshold | NumPy | Percentile-based or fixed |
| Hysteresis | CPU — OpenCV connected components | Inherently sequential |

Gaussian blur and Sobel magnitude are implemented as cuTile kernels using a shared design pattern: neighbouring pixel arrays are pre-extracted and flattened to 1D before the kernel launch, so each tile operates on fully independent data with no inter-tile dependencies.  NMS and later stages run on CuPy or CPU; hysteresis is kept on CPU because the BFS graph traversal does not map cleanly to the tile execution model.

## File Structure

```
notebooks/
  01_canny_baseline.ipynb          Pure NumPy Canny baseline with visualisation
src/gpu/
  cutile_full_pipeline.py          Complete pipeline (cuTile + CuPy + CPU)
  sobel_cutile_benchmark.py        cuTile Sobel kernel + tile-size sweep
  gaussian_benchmark.py            CuPy Gaussian vs CPU benchmark
  sobel_benchmark.py               CuPy Sobel vs CPU benchmark
  canny_frontend_benchmark.py      Gaussian + Sobel frontend benchmark
  cutile_canny_pipeline_benchmark.py  Pipeline through NMS benchmark
  complete_canny_benchmark.py      Full pipeline benchmark (all stages)
  video_stream_demo.py             Real-time video stream demo
  canny_pipeline.py                CuPy reference pipeline
bsds500_canny_test.py              Reliability test vs OpenCV on BSDS500
report/                            Benchmark CSVs, output images, analysis notes
data/                              Sample images
```

## How to Run

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# Complete cuTile pipeline on a single image
python src/gpu/cutile_full_pipeline.py --image data/test.jpg

# Benchmark with 10 timed runs (no GUI window)
python src/gpu/cutile_full_pipeline.py --image data/test.jpg --runs 10 --no-display

# Explore tile sizes and save edge output
python src/gpu/cutile_full_pipeline.py --image data/test.jpg --tile-size 128 --output report/edges.png --no-display

# Real-time webcam demo
python src/gpu/video_stream_demo.py --source 0

# Video demo using a static image as a fake stream (no webcam needed)
python src/gpu/video_stream_demo.py --image-loop data/test.jpg --max-frames 30 --no-display

# Individual stage benchmarks
python src/gpu/sobel_cutile_benchmark.py --tile-sizes 64,128,256,512 --runs 30
python src/gpu/complete_canny_benchmark.py --image data/test.jpg
```

## cuTile Implementation Details

### Gaussian Blur

The separable Gaussian blur runs two cuTile passes (horizontal then vertical). Each pass pre-extracts `k` shifted views of the image and passes them as separate 1D arrays to the kernel, which computes the weighted sum tile-by-tile. Only k=5 uses the cuTile kernel; other sizes fall back to CuPy automatically.

### Sobel Magnitude

Eight 3×3 neighbour arrays are extracted from the blurred image interior and flattened to 1D. The cuTile kernel computes `sqrt(Gx² + Gy²)` for each interior pixel with no inter-tile dependencies. A tile-size sweep (64, 128, 256, 512) is benchmarked in `sobel_cutile_benchmark.py`; results are written to `report/17_sobel_cutile_tile_sweep_*.csv`.

### Hardware Note

The cuTile Sobel kernel requires Turing architecture or newer (RTX 20-series+). On Pascal GPUs (GTX 10-series) the kernel falls back to an equivalent CuPy implementation automatically. See git commit `07f36ca` for details.

## Benchmark Results

Benchmark CSVs and output images are written to `report/`:

| File pattern | Content |
|---|---|
| `17_sobel_cutile_tile_sweep_*` | cuTile Sobel tile-size sweep |
| `19_cutile_canny_pipeline_*` | Gaussian + Sobel + NMS pipeline |
| `22_complete_canny_benchmark_*` | Full Canny (all stages) CPU vs GPU |
| `25_video_stream_fps_*` | Real-time video stream FPS |

For a detailed analysis of data dependencies, parallelism ratings, and memory access patterns for each stage, see `report/dependency_analysis.md`.