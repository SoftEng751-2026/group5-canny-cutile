# Member A GPU/cuTile Implementation Notes

## Role

Member A focused on the GPU/cuTile part of the Canny edge detection project.

The implemented work includes:

1. cuTile environment test
2. GPU Sobel baseline
3. CPU vs GPU Sobel benchmark
4. CPU vs GPU Gaussian blur benchmark
5. Gaussian + Sobel frontend benchmark
6. cuTile Sobel magnitude benchmark with tile-size sweep
7. cuTile Canny frontend pipeline benchmark
8. Full Canny prototype benchmark
9. Video stream FPS demo

## Main files

### `src/gpu/vector_add_cutile.py`

A minimal cuTile vector addition test.

Purpose:

- check that Python, CuPy, cuTile, and NVIDIA GPU execution work correctly
- confirm that a cuTile kernel can be launched successfully

Run:

```powershell
python src/gpu/vector_add_cutile.py

Expected output:

cuTile vector_add test passed!
src/gpu/sobel_benchmark.py

Benchmarks CPU Sobel against GPU Sobel using CuPy.

It measures:

CPU Sobel time
GPU compute-only Sobel time
GPU Sobel time including CPU-to-GPU transfer
CPU/GPU output difference

Example:

python src/gpu/sobel_benchmark.py
python src/gpu/sobel_benchmark.py --image data/IMG_6860.JPG --runs 20 --warmup 3
src/gpu/gaussian_benchmark.py

Benchmarks CPU Gaussian blur against GPU Gaussian blur using CuPy.

It supports different image inputs, kernel sizes, and sigma values.

Example:

python src/gpu/gaussian_benchmark.py
python src/gpu/gaussian_benchmark.py --image data/IMG_6860.JPG --runs 20 --warmup 3
python src/gpu/gaussian_benchmark.py --image data/test.jpg --kernel-size 7 --sigma 1.6 --runs 30 --warmup 3
src/gpu/canny_frontend_benchmark.py

Benchmarks the first two Canny stages:

Gaussian blur -> Sobel gradient

It compares:

CPU frontend
GPU compute-only frontend
GPU frontend with input transfer
GPU end-to-end frontend

Example:

python src/gpu/canny_frontend_benchmark.py
python src/gpu/canny_frontend_benchmark.py --image data/IMG_6860.JPG --runs 10 --warmup 3
src/gpu/sobel_cutile_benchmark.py

Implements the first true cuTile Sobel magnitude kernel.

It tests different cuTile tile sizes and writes a CSV result file.

Example:

python src/gpu/sobel_cutile_benchmark.py
python src/gpu/sobel_cutile_benchmark.py --image data/IMG_6860.JPG --runs 10 --warmup 3
python src/gpu/sobel_cutile_benchmark.py --image data/test.jpg --tile-sizes 32,64,128,256,512,1024 --runs 30 --warmup 5

Important observation:

Small images may not benefit much from cuTile because overhead dominates.
Larger images benefit more from cuTile parallel execution.
The best tile size depends on image size and workload.
src/gpu/cutile_canny_pipeline_benchmark.py

Benchmarks a Canny frontend pipeline using:

CuPy Gaussian blur
-> cuTile Sobel magnitude
-> CuPy Sobel angle
-> CuPy non-maximum suppression

It does not include double threshold or hysteresis yet.

Example:

python src/gpu/cutile_canny_pipeline_benchmark.py
python src/gpu/cutile_canny_pipeline_benchmark.py --image data/IMG_6860.JPG --runs 3 --warmup 1
python src/gpu/cutile_canny_pipeline_benchmark.py --image data/test.jpg --kernel-size 7 --sigma 1.6 --tile-sizes 64,128,256,512 --runs 10 --warmup 3
src/gpu/complete_canny_benchmark.py

Benchmarks a full Canny prototype:

Gaussian blur
-> cuTile Sobel magnitude
-> Sobel angle
-> non-maximum suppression
-> double threshold
-> hysteresis

The frontend uses GPU/cuTile. Double threshold and hysteresis are currently CPU post-processing steps.

Example:

python src/gpu/complete_canny_benchmark.py
python src/gpu/complete_canny_benchmark.py --image data/IMG_6860.JPG --runs 1 --warmup 1
python src/gpu/complete_canny_benchmark.py --image data/test.jpg --kernel-size 7 --sigma 1.6 --tile-sizes 64,128,256,512 --runs 3 --warmup 1

Important observation:

CPU and GPU prototype edge maps matched in the tested cases.
Small NMS differences around 0.000031 or 0.000061 are expected from float32 GPU arithmetic.
Final edge pixel difference was 0.0000% in the tested cases.
src/gpu/video_stream_demo.py

Runs the Canny pipeline on a stream of input frames.

It supports:

webcam input
video file input
image-loop mode for repeatable testing
FPS CSV output
optional video output

Examples:

python src/gpu/video_stream_demo.py --image-loop data/test.jpg --max-frames 30 --no-display

python src/gpu/video_stream_demo.py --image-loop data/IMG_6860.JPG --resize-width 640 --max-frames 20 --no-display

python src/gpu/video_stream_demo.py --source 0 --resize-width 640

Recent FPS results:

Source	Resize width	Frames	Average FPS
data/test.jpg image-loop	640	30	13.64
data/IMG_6860.JPG image-loop	640	20	12.76
webcam source 0	640	752	78.49
Notes for the report
Correctness checking

For the benchmark scripts, CPU results were used as reference outputs.

The GPU or cuTile output was compared with the CPU output using maximum absolute difference or final edge-pixel difference.

Observed differences were zero or very small, which indicates that the GPU/cuTile versions preserve the expected result.

Performance measurement

GPU timing used warm-up runs and CUDA stream synchronisation.

Warm-up runs avoid measuring one-off GPU initialisation overhead.

CUDA stream synchronisation ensures that asynchronous GPU computation has completed before the timer stops.

Transfer cost

Several benchmarks distinguish:

compute-only time
time with CPU-to-GPU input transfer
end-to-end time including output transfer

This is important because real-time video frames usually arrive from OpenCV on the CPU and may need to be copied to the GPU.

Parameter exploration

The main explored parameters include:

image size
Gaussian kernel size
Gaussian sigma
cuTile tile size
resize width for video stream
threshold percentile and low threshold ratio

The experiments showed that the best tile size is not fixed and can depend on image size and workload.