import argparse
import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Load an image as grayscale float32."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image.astype(np.float32)


def make_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Create a 1D Gaussian kernel.

    We use separable convolution:
    2D Gaussian blur = horizontal 1D blur + vertical 1D blur.
    This is faster than applying a full 2D kernel.
    """
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd, for example 3, 5, 7.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)

    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)


def gaussian_blur_cpu(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur on CPU using NumPy.

    Boundary handling uses edge padding, so the output image has the same size
    as the input image.
    """
    radius = kernel.size // 2
    height, width = image.shape

    # Horizontal pass
    padded_horizontal = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    temp = np.zeros_like(image, dtype=np.float32)

    for offset, weight in enumerate(kernel):
        temp += weight * padded_horizontal[:, offset : offset + width]

    # Vertical pass
    padded_vertical = np.pad(temp, ((radius, radius), (0, 0)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for offset, weight in enumerate(kernel):
        output += weight * padded_vertical[offset : offset + height, :]

    return output


def gaussian_blur_gpu_compute_only(image_gpu, kernel_gpu):
    """
    Apply Gaussian blur on GPU using CuPy.

    The input image and kernel are already on the GPU, so this measures
    computation without CPU-to-GPU transfer cost.
    """
    radius = kernel_gpu.size // 2
    height, width = image_gpu.shape

    # Horizontal pass
    padded_horizontal = cp.pad(image_gpu, ((0, 0), (radius, radius)), mode="edge")
    temp = cp.zeros_like(image_gpu, dtype=cp.float32)

    for offset in range(kernel_gpu.size):
        temp += kernel_gpu[offset] * padded_horizontal[:, offset : offset + width]

    # Vertical pass
    padded_vertical = cp.pad(temp, ((radius, radius), (0, 0)), mode="edge")
    output = cp.zeros_like(image_gpu, dtype=cp.float32)

    for offset in range(kernel_gpu.size):
        output += kernel_gpu[offset] * padded_vertical[offset : offset + height, :]

    return output


def gaussian_blur_gpu_with_transfer(image_cpu: np.ndarray, kernel_cpu: np.ndarray):
    """
    Copy image and kernel from CPU to GPU, then apply Gaussian blur.

    This is useful for real-time video experiments because OpenCV frames usually
    arrive on CPU before being copied to GPU.
    """
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)

    return gaussian_blur_gpu_compute_only(image_gpu, kernel_gpu)


def save_uint8_image(output_path: Path, image: np.ndarray) -> None:
    """
    Save a float32 image as uint8.

    Gaussian blur should preserve the original intensity range, so we clip to
    [0, 255] instead of normalising by the maximum value.
    """
    image_uint8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), image_uint8)


def benchmark_cpu(image_cpu: np.ndarray, kernel_cpu: np.ndarray, runs: int, warmup: int):
    """Benchmark CPU Gaussian blur."""
    for _ in range(warmup):
        output_cpu = gaussian_blur_cpu(image_cpu, kernel_cpu)

    start = time.perf_counter()

    for _ in range(runs):
        output_cpu = gaussian_blur_cpu(image_cpu, kernel_cpu)

    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, output_cpu


def benchmark_gpu_compute_only(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark GPU Gaussian blur when image and kernel are already on GPU."""
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()

    for _ in range(warmup):
        output_gpu = gaussian_blur_gpu_compute_only(image_gpu, kernel_gpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        output_gpu = gaussian_blur_gpu_compute_only(image_gpu, kernel_gpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, output_gpu


def benchmark_gpu_with_transfer(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark GPU Gaussian blur including CPU-to-GPU transfer."""
    for _ in range(warmup):
        output_gpu = gaussian_blur_gpu_with_transfer(image_cpu, kernel_cpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        output_gpu = gaussian_blur_gpu_with_transfer(image_cpu, kernel_cpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs
    return average_time, output_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and GPU Gaussian blur."
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
        help="Odd Gaussian kernel size, for example 3, 5, or 7.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma value.",
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


def make_output_tag(image_path: Path, kernel_size: int, sigma: float) -> str:
    sigma_text = f"{sigma:g}".replace(".", "p")
    return f"{image_path.stem}_k{kernel_size}_s{sigma_text}"


def main():
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive.")

    if args.warmup < 0:
        raise ValueError("warmup must not be negative.")

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_cpu = load_grayscale_image(image_path)
    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    print(f"Input image: {image_path}")
    print(f"Image shape: {image_cpu.shape}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Sigma: {args.sigma}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")
    print(f"Gaussian kernel: {kernel_cpu}")

    cpu_time, cpu_output = benchmark_cpu(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_compute_time, gpu_output_compute = benchmark_gpu_compute_only(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_transfer_time, gpu_output_transfer = benchmark_gpu_with_transfer(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_output_cpu = cp.asnumpy(gpu_output_compute)
    max_abs_diff = float(np.max(np.abs(cpu_output - gpu_output_cpu)))

    output_tag = make_output_tag(image_path, args.kernel_size, args.sigma)

    cpu_output_path = output_dir / f"09_gaussian_cpu_blur_{output_tag}.png"
    gpu_output_path = output_dir / f"10_gaussian_gpu_blur_{output_tag}.png"

    save_uint8_image(cpu_output_path, cpu_output)
    save_uint8_image(gpu_output_path, gpu_output_cpu)

    print()
    print("Gaussian blur benchmark results")
    print("--------------------------------")
    print(f"CPU Gaussian average time: {cpu_time:.6f} seconds")
    print(f"GPU Gaussian compute-only average time: {gpu_compute_time:.6f} seconds")
    print(f"GPU Gaussian with transfer average time: {gpu_transfer_time:.6f} seconds")

    if gpu_compute_time > 0:
        print(f"Compute-only GPU speedup over CPU: {cpu_time / gpu_compute_time:.2f}x")

    if gpu_transfer_time > 0:
        print(f"GPU-with-transfer speedup over CPU: {cpu_time / gpu_transfer_time:.2f}x")

    print(f"Max absolute difference between CPU and GPU blur: {max_abs_diff:.6f}")
    print(f"CPU blur saved to: {cpu_output_path}")
    print(f"GPU blur saved to: {gpu_output_path}")

    print()
    print(
        "Note: compute-only time excludes CPU-to-GPU transfer, while with-transfer "
        "time includes copying the input image and kernel to the GPU."
    )


if __name__ == "__main__":
    main()