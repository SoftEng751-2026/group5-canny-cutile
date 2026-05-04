import argparse
import time
from pathlib import Path
from paths import resolve_project_path

import cv2
import cupy as cp
import numpy as np


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Load an image as grayscale float32."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image.astype(np.float32)


def sobel_cpu(image: np.ndarray):
    """Compute Sobel gradient magnitude and direction on CPU using NumPy."""
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    gx[1:-1, 1:-1] = (
        -image[:-2, :-2]
        + image[:-2, 2:]
        - 2.0 * image[1:-1, :-2]
        + 2.0 * image[1:-1, 2:]
        - image[2:, :-2]
        + image[2:, 2:]
    )

    gy[1:-1, 1:-1] = (
        image[:-2, :-2]
        + 2.0 * image[:-2, 1:-1]
        + image[:-2, 2:]
        - image[2:, :-2]
        - 2.0 * image[2:, 1:-1]
        - image[2:, 2:]
    )

    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    return magnitude, angle


def sobel_gpu_compute_only(image_gpu):
    """
    Compute Sobel gradient magnitude and direction on GPU.

    The input is already on the GPU, so this measures the Sobel computation
    without CPU-to-GPU transfer cost.
    """
    gx = cp.zeros_like(image_gpu, dtype=cp.float32)
    gy = cp.zeros_like(image_gpu, dtype=cp.float32)

    gx[1:-1, 1:-1] = (
        -image_gpu[:-2, :-2]
        + image_gpu[:-2, 2:]
        - 2.0 * image_gpu[1:-1, :-2]
        + 2.0 * image_gpu[1:-1, 2:]
        - image_gpu[2:, :-2]
        + image_gpu[2:, 2:]
    )

    gy[1:-1, 1:-1] = (
        image_gpu[:-2, :-2]
        + 2.0 * image_gpu[:-2, 1:-1]
        + image_gpu[:-2, 2:]
        - image_gpu[2:, :-2]
        - 2.0 * image_gpu[2:, 1:-1]
        - image_gpu[2:, 2:]
    )

    magnitude = cp.sqrt(gx * gx + gy * gy)
    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0

    return magnitude, angle


def sobel_gpu_with_transfer(image_cpu: np.ndarray):
    """
    Copy the image from CPU to GPU, then compute Sobel on GPU.

    This is useful because real video frames usually arrive on CPU from OpenCV,
    so CPU-to-GPU transfer can affect real-time performance.
    """
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    return sobel_gpu_compute_only(image_gpu)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to uint8 for saving."""
    max_value = float(image.max())

    if max_value == 0.0:
        return image.astype(np.uint8)

    return (image / max_value * 255.0).astype(np.uint8)


def benchmark_cpu(image_cpu: np.ndarray, runs: int, warmup: int):
    """Benchmark CPU Sobel."""
    for _ in range(warmup):
        magnitude, angle = sobel_cpu(image_cpu)

    start = time.perf_counter()

    for _ in range(runs):
        magnitude, angle = sobel_cpu(image_cpu)

    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, magnitude, angle


def benchmark_gpu_compute_only(image_cpu: np.ndarray, runs: int, warmup: int):
    """Benchmark GPU Sobel when the image is already on the GPU."""
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()

    for _ in range(warmup):
        magnitude_gpu, angle_gpu = sobel_gpu_compute_only(image_gpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        magnitude_gpu, angle_gpu = sobel_gpu_compute_only(image_gpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, magnitude_gpu, angle_gpu


def benchmark_gpu_with_transfer(image_cpu: np.ndarray, runs: int, warmup: int):
    """Benchmark GPU Sobel including CPU-to-GPU transfer."""
    for _ in range(warmup):
        magnitude_gpu, angle_gpu = sobel_gpu_with_transfer(image_cpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        magnitude_gpu, angle_gpu = sobel_gpu_with_transfer(image_cpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, magnitude_gpu, angle_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and GPU Sobel gradient computation."
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/test.jpg"),
        help="Input image path.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=50,
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

    image_path = resolve_project_path(args.image)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input image: {image_path}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")

    image_cpu = load_grayscale_image(image_path)

    cpu_time, cpu_mag, cpu_angle = benchmark_cpu(
        image_cpu=image_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_compute_time, gpu_mag_compute, gpu_angle_compute = benchmark_gpu_compute_only(
        image_cpu=image_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_transfer_time, gpu_mag_transfer, gpu_angle_transfer = benchmark_gpu_with_transfer(
        image_cpu=image_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_mag_cpu = cp.asnumpy(gpu_mag_compute)
    max_abs_diff = float(np.max(np.abs(cpu_mag - gpu_mag_cpu)))

    image_stem = image_path.stem

    cpu_output = output_dir / f"07_sobel_cpu_magnitude_{image_stem}.png"
    gpu_output = output_dir / f"08_sobel_gpu_magnitude_{image_stem}.png"

    cv2.imwrite(str(cpu_output), normalize_to_uint8(cpu_mag))
    cv2.imwrite(str(gpu_output), normalize_to_uint8(gpu_mag_cpu))

    print()
    print("Sobel benchmark results")
    print("-----------------------")
    print(f"Image shape: {image_cpu.shape}")
    print(f"CPU Sobel average time: {cpu_time:.6f} seconds")
    print(f"GPU Sobel compute-only average time: {gpu_compute_time:.6f} seconds")
    print(f"GPU Sobel with transfer average time: {gpu_transfer_time:.6f} seconds")

    if gpu_compute_time > 0:
        print(f"Compute-only GPU speedup over CPU: {cpu_time / gpu_compute_time:.2f}x")

    if gpu_transfer_time > 0:
        print(f"GPU-with-transfer speedup over CPU: {cpu_time / gpu_transfer_time:.2f}x")

    print(f"Max absolute difference between CPU and GPU magnitude: {max_abs_diff:.6f}")
    print(f"CPU magnitude saved to: {cpu_output}")
    print(f"GPU magnitude saved to: {gpu_output}")

    print()
    print(
        "Note: compute-only time excludes CPU-to-GPU transfer, while with-transfer "
        "time includes copying the input image to the GPU."
    )


if __name__ == "__main__":
    main()