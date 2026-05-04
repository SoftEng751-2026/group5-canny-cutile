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

## Step 2: GPU/cuTile Canny Implementation

This step contains the GPU and cuTile implementation work for the Canny edge detection project.

The main GPU/cuTile scripts are located under:

```text
src/gpu/

Test images should be placed in the repository-level data/ folder, for example:

data/test.jpg
data/IMG_6860.JPG

The scripts resolve paths relative to the repository root, so they can be run either from the project root or from src/gpu.

2.1 cuTile environment test

Run:

python src/gpu/vector_add_cutile.py

Expected output:

cuTile vector_add test passed!

This checks that Python, CuPy, cuTile, and the NVIDIA GPU environment are working.

2.2 Sobel CPU/GPU benchmark

Run:

python src/gpu/sobel_benchmark.py --image data/test.jpg

This compares CPU Sobel and GPU Sobel. It reports CPU time, GPU compute-only time, GPU time with input transfer, speedup, and output difference.

2.3 Gaussian blur CPU/GPU benchmark

Run:

python src/gpu/gaussian_benchmark.py --image data/test.jpg

Optional parameter test:

python src/gpu/gaussian_benchmark.py --image data/test.jpg --kernel-size 7 --sigma 1.6

This compares CPU Gaussian blur and GPU Gaussian blur, and supports different Gaussian kernel sizes and sigma values.

2.4 Gaussian + Sobel frontend benchmark

Run:

python src/gpu/canny_frontend_benchmark.py --image data/test.jpg

This benchmarks the first two Canny stages:

Gaussian blur -> Sobel gradient

It compares CPU, GPU compute-only, GPU with input transfer, and GPU end-to-end timing.

2.5 cuTile Sobel magnitude benchmark

Run:

python src/gpu/sobel_cutile_benchmark.py --image data/test.jpg --tile-sizes 64,128,256,512

This is the first true cuTile kernel implementation. It computes Sobel magnitude using cuTile and tests different tile sizes.

Example with a larger tile-size sweep:

python src/gpu/sobel_cutile_benchmark.py --image data/test.jpg --tile-sizes 32,64,128,256,512,1024
2.6 cuTile Canny frontend pipeline

Run:

python src/gpu/cutile_canny_pipeline_benchmark.py --image data/test.jpg

This pipeline uses:

CuPy Gaussian blur
-> cuTile Sobel magnitude
-> CuPy Sobel angle
-> CuPy non-maximum suppression

It does not yet include double threshold or hysteresis.

2.7 Complete Canny prototype

Run:

python src/gpu/complete_canny_benchmark.py --image data/test.jpg

This runs:

Gaussian blur
-> cuTile Sobel magnitude
-> Sobel angle
-> non-maximum suppression
-> double threshold
-> hysteresis

The frontend uses GPU/cuTile, while double threshold and hysteresis are currently CPU post-processing steps.

2.8 Video stream FPS demo

Image-loop test:

python src/gpu/video_stream_demo.py --image-loop data/test.jpg --max-frames 30 --no-display

Larger image-loop test:

python src/gpu/video_stream_demo.py --image-loop data/IMG_6860.JPG --resize-width 640 --max-frames 20 --no-display

Webcam test:

python src/gpu/video_stream_demo.py --source 0 --resize-width 640

Press q to quit the display window.

2.9 Notes on timing

GPU timing uses warm-up runs and CUDA stream synchronisation. Warm-up runs avoid measuring one-off GPU initialisation overhead. Synchronisation ensures GPU computation has finished before the timer stops.

Several benchmarks distinguish between:

compute-only time
input-transfer time
end-to-end time

This is important because OpenCV frames usually arrive on the CPU and need to be copied to the GPU for processing.

2.10 Recent video-stream results
Source	Resize width	Frames	Average FPS
data/test.jpg image-loop	640	30	13.64
data/IMG_6860.JPG image-loop	640	20	12.76
webcam source 0	640	752	78.49