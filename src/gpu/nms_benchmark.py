import argparse
import time
from pathlib import Path

from paths import resolve_project_path

import cv2
import cupy as cp
import numpy as np

from canny_frontend_benchmark import (
    canny_frontend_cpu,
    canny_frontend_gpu_compute_only,
)
from gaussian_benchmark import load_grayscale_image, make_gaussian_kernel
from sobel_benchmark import normalize_to_uint8


def non_max_suppression_cpu(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Apply non-maximum suppression on CPU.

    Each pixel is compared with two neighbours along the gradient direction.
    If the pixel is not a local maximum, it is suppressed to zero.
    """
    if magnitude.shape != angle.shape:
        raise ValueError("magnitude and angle must have the same shape.")

    angle = angle % 180.0
    output = np.zeros_like(magnitude, dtype=np.float32)

    center = magnitude[1:-1, 1:-1]
    direction = angle[1:-1, 1:-1]

    left = magnitude[1:-1, :-2]
    right = magnitude[1:-1, 2:]

    top = magnitude[:-2, 1:-1]
    bottom = magnitude[2:, 1:-1]

    top_left = magnitude[:-2, :-2]
    top_right = magnitude[:-2, 2:]
    bottom_left = magnitude[2:, :-2]
    bottom_right = magnitude[2:, 2:]

    direction_0 = (direction < 22.5) | (direction >= 157.5)
    direction_45 = (direction >= 22.5) & (direction < 67.5)
    direction_90 = (direction >= 67.5) & (direction < 112.5)
    direction_135 = (direction >= 112.5) & (direction < 157.5)

    keep_0 = direction_0 & (center >= left) & (center >= right)
    keep_45 = direction_45 & (center >= top_right) & (center >= bottom_left)
    keep_90 = direction_90 & (center >= top) & (center >= bottom)
    keep_135 = direction_135 & (center >= top_left) & (center >= bottom_right)

    keep = keep_0 | keep_45 | keep_90 | keep_135

    output[1:-1, 1:-1] = np.where(keep, center, 0.0)

    return output


def non_max_suppression_gpu_compute_only(magnitude_gpu, angle_gpu):
    """
    Apply non-maximum suppression on GPU using CuPy.

    The magnitude and angle arrays are already on the GPU. This matches the
    intended Canny pipeline, where Sobel produces magnitude and angle on GPU
    and NMS consumes them without copying them back to CPU.
    """
    if magnitude_gpu.shape != angle_gpu.shape:
        raise ValueError("magnitude and angle must have the same shape.")

    angle_gpu = angle_gpu % 180.0
    output_gpu = cp.zeros_like(magnitude_gpu, dtype=cp.float32)

    center = magnitude_gpu[1:-1, 1:-1]
    direction = angle_gpu[1:-1, 1:-1]

    left = magnitude_gpu[1:-1, :-2]
    right = magnitude_gpu[1:-1, 2:]

    top = magnitude_gpu[:-2, 1:-1]
    bottom = magnitude_gpu[2:, 1:-1]

    top_left = magnitude_gpu[:-2, :-2]
    top_right = magnitude_gpu[:-2, 2:]
    bottom_left = magnitude_gpu[2:, :-2]
    bottom_right = magnitude_gpu[2:, 2:]

    direction_0 = (direction < 22.5) | (direction >= 157.5)
    direction_45 = (direction >= 22.5) & (direction < 67.5)
    direction_90 = (direction >= 67.5) & (direction < 112.5)
    direction_135 = (direction >= 112.5) & (direction < 157.5)

    keep_0 = direction_0 & (center >= left) & (center >= right)
    keep_45 = direction_45 & (center >= top_right) & (center >= bottom_left)
    keep_90 = direction_90 & (center >= top) & (center >= bottom)
    keep_135 = direction_135 & (center >= top_left) & (center >= bottom_right)

    keep = keep_0 | keep_45 | keep_90 | keep_135

    output_gpu[1:-1, 1:-1] = cp.where(keep, center, 0.0)

    return output_gpu


def benchmark_cpu_nms(
    magnitude_cpu: np.ndarray,
    angle_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark CPU NMS only."""
    for _ in range(warmup):
        output_cpu = non_max_suppression_cpu(magnitude_cpu, angle_cpu)

    start = time.perf_counter()

    for _ in range(runs):
        output_cpu = non_max_suppression_cpu(magnitude_cpu, angle_cpu)

    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, output_cpu


def benchmark_gpu_nms_compute_only(
    magnitude_gpu,
    angle_gpu,
    runs: int,
    warmup: int,
):
    """Benchmark GPU NMS only, with magnitude and angle already on GPU."""
    cp.cuda.Stream.null.synchronize()

    for _ in range(warmup):
        output_gpu = non_max_suppression_gpu_compute_only(magnitude_gpu, angle_gpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        output_gpu = non_max_suppression_gpu_compute_only(magnitude_gpu, angle_gpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, output_gpu


def make_output_tag(image_path: Path, kernel_size: int, sigma: float) -> str:
    """Create a readable output tag that avoids overwriting results."""
    sigma_text = f"{sigma:g}".replace(".", "p")
    return f"{image_path.stem}_k{kernel_size}_s{sigma_text}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and GPU non-maximum suppression for Canny."
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/test.jpg"),
        help="Input image path.",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Odd Gaussian kernel size used before Sobel.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma value used before Sobel.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of timed runs.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up runs before timing.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Directory for output images.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive.")

    if args.warmup < 0:
        raise ValueError("warmup must not be negative.")

    image_path = resolve_project_path(args.image)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_cpu = load_grayscale_image(image_path)
    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    print(f"Input image: {image_path}")
    print(f"Image shape: {image_cpu.shape}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Sigma: {args.sigma}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")

    print("Preparing CPU Gaussian + Sobel inputs for NMS...")
    _, magnitude_cpu, angle_cpu = canny_frontend_cpu(image_cpu, kernel_cpu)

    print("Preparing GPU Gaussian + Sobel inputs for NMS...")
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)

    _, magnitude_gpu, angle_gpu = canny_frontend_gpu_compute_only(
        image_gpu,
        kernel_gpu,
    )
    cp.cuda.Stream.null.synchronize()

    cpu_nms_time, nms_cpu = benchmark_cpu_nms(
        magnitude_cpu=magnitude_cpu,
        angle_cpu=angle_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_nms_time, nms_gpu = benchmark_gpu_nms_compute_only(
        magnitude_gpu=magnitude_gpu,
        angle_gpu=angle_gpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    nms_gpu_cpu = cp.asnumpy(nms_gpu)
    max_abs_diff = float(np.max(np.abs(nms_cpu - nms_gpu_cpu)))

    output_tag = make_output_tag(image_path, args.kernel_size, args.sigma)

    cpu_output_path = output_dir / f"15_nms_cpu_edges_{output_tag}.png"
    gpu_output_path = output_dir / f"16_nms_gpu_edges_{output_tag}.png"

    cv2.imwrite(str(cpu_output_path), normalize_to_uint8(nms_cpu))
    cv2.imwrite(str(gpu_output_path), normalize_to_uint8(nms_gpu_cpu))

    print()
    print("Non-maximum suppression benchmark results")
    print("-----------------------------------------")
    print(f"CPU NMS average time: {cpu_nms_time:.6f} seconds")
    print(f"GPU NMS compute-only average time: {gpu_nms_time:.6f} seconds")

    if gpu_nms_time > 0:
        print(f"Compute-only GPU speedup over CPU: {cpu_nms_time / gpu_nms_time:.2f}x")

    print(f"Max absolute difference between CPU and GPU NMS: {max_abs_diff:.6f}")
    print(f"CPU NMS output saved to: {cpu_output_path}")
    print(f"GPU NMS output saved to: {gpu_output_path}")

    print()
    print(
        "Note: this benchmarks the NMS stage after Gaussian blur and Sobel. "
        "GPU NMS is measured as compute-only because magnitude and angle are "
        "intended to stay on the GPU from the previous Sobel stage."
    )


if __name__ == "__main__":
    main()