import argparse
import time
from pathlib import Path
from paths import resolve_project_path
import cupy as cp
import numpy as np

from gaussian_benchmark import (
    gaussian_blur_cpu,
    gaussian_blur_gpu_compute_only,
    load_grayscale_image,
    make_gaussian_kernel,
    save_uint8_image,
)
from sobel_benchmark import normalize_to_uint8, sobel_cpu, sobel_gpu_compute_only


def canny_frontend_cpu(image_cpu: np.ndarray, kernel_cpu: np.ndarray):
    """
    Run the first two stages of Canny on CPU:
    Gaussian blur followed by Sobel gradient.
    """
    blurred_cpu = gaussian_blur_cpu(image_cpu, kernel_cpu)
    magnitude_cpu, angle_cpu = sobel_cpu(blurred_cpu)

    return blurred_cpu, magnitude_cpu, angle_cpu


def canny_frontend_gpu_compute_only(image_gpu, kernel_gpu):
    """
    Run Gaussian blur followed by Sobel gradient on GPU.

    The image and kernel are already on the GPU, so this excludes CPU-to-GPU
    transfer cost. This measures the compute part of the GPU pipeline.
    """
    blurred_gpu = gaussian_blur_gpu_compute_only(image_gpu, kernel_gpu)
    magnitude_gpu, angle_gpu = sobel_gpu_compute_only(blurred_gpu)

    return blurred_gpu, magnitude_gpu, angle_gpu


def canny_frontend_gpu_with_input_transfer(image_cpu: np.ndarray, kernel_cpu: np.ndarray):
    """
    Copy the input image and kernel to GPU, then run Gaussian + Sobel.

    The output remains on the GPU. This is useful for a multi-stage pipeline
    where later stages also stay on the GPU.
    """
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)

    return canny_frontend_gpu_compute_only(image_gpu, kernel_gpu)


def canny_frontend_gpu_end_to_end(image_cpu: np.ndarray, kernel_cpu: np.ndarray):
    """
    Copy input to GPU, run Gaussian + Sobel, then copy output magnitude back to CPU.

    This is closer to real-time display use cases, where OpenCV reads frames on
    CPU and the edge image may need to be displayed or saved on CPU.
    """
    blurred_gpu, magnitude_gpu, angle_gpu = canny_frontend_gpu_with_input_transfer(
        image_cpu,
        kernel_cpu,
    )

    magnitude_cpu = cp.asnumpy(magnitude_gpu)

    return magnitude_cpu


def benchmark_cpu(image_cpu: np.ndarray, kernel_cpu: np.ndarray, runs: int, warmup: int):
    """Benchmark CPU Gaussian + Sobel frontend."""
    for _ in range(warmup):
        blurred_cpu, magnitude_cpu, angle_cpu = canny_frontend_cpu(image_cpu, kernel_cpu)

    start = time.perf_counter()

    for _ in range(runs):
        blurred_cpu, magnitude_cpu, angle_cpu = canny_frontend_cpu(image_cpu, kernel_cpu)

    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, blurred_cpu, magnitude_cpu, angle_cpu


def benchmark_gpu_compute_only(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark GPU frontend with image and kernel already on GPU."""
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()

    for _ in range(warmup):
        blurred_gpu, magnitude_gpu, angle_gpu = canny_frontend_gpu_compute_only(
            image_gpu,
            kernel_gpu,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        blurred_gpu, magnitude_gpu, angle_gpu = canny_frontend_gpu_compute_only(
            image_gpu,
            kernel_gpu,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, blurred_gpu, magnitude_gpu, angle_gpu


def benchmark_gpu_with_input_transfer(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark GPU frontend including CPU-to-GPU input transfer."""
    for _ in range(warmup):
        blurred_gpu, magnitude_gpu, angle_gpu = canny_frontend_gpu_with_input_transfer(
            image_cpu,
            kernel_cpu,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        blurred_gpu, magnitude_gpu, angle_gpu = canny_frontend_gpu_with_input_transfer(
            image_cpu,
            kernel_cpu,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, blurred_gpu, magnitude_gpu, angle_gpu


def benchmark_gpu_end_to_end(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark GPU frontend including input transfer and output transfer."""
    for _ in range(warmup):
        magnitude_cpu = canny_frontend_gpu_end_to_end(image_cpu, kernel_cpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        magnitude_cpu = canny_frontend_gpu_end_to_end(image_cpu, kernel_cpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, magnitude_cpu


def make_output_tag(image_path: Path, kernel_size: int, sigma: float) -> str:
    """Create a readable output tag that avoids overwriting results."""
    sigma_text = f"{sigma:g}".replace(".", "p")
    return f"{image_path.stem}_k{kernel_size}_s{sigma_text}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Gaussian blur + Sobel gradient Canny frontend."
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

    cpu_time, cpu_blurred, cpu_magnitude, cpu_angle = benchmark_cpu(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_compute_time, gpu_blurred, gpu_magnitude, gpu_angle = benchmark_gpu_compute_only(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_input_transfer_time, _, _, _ = benchmark_gpu_with_input_transfer(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_end_to_end_time, gpu_magnitude_end_to_end_cpu = benchmark_gpu_end_to_end(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    gpu_blurred_cpu = cp.asnumpy(gpu_blurred)
    gpu_magnitude_cpu = cp.asnumpy(gpu_magnitude)

    blur_max_abs_diff = float(np.max(np.abs(cpu_blurred - gpu_blurred_cpu)))
    magnitude_max_abs_diff = float(np.max(np.abs(cpu_magnitude - gpu_magnitude_cpu)))
    end_to_end_max_abs_diff = float(
        np.max(np.abs(cpu_magnitude - gpu_magnitude_end_to_end_cpu))
    )

    output_tag = make_output_tag(image_path, args.kernel_size, args.sigma)

    cpu_blur_output = output_dir / f"11_frontend_cpu_blur_{output_tag}.png"
    gpu_blur_output = output_dir / f"12_frontend_gpu_blur_{output_tag}.png"
    cpu_magnitude_output = output_dir / f"13_frontend_cpu_magnitude_{output_tag}.png"
    gpu_magnitude_output = output_dir / f"14_frontend_gpu_magnitude_{output_tag}.png"

    save_uint8_image(cpu_blur_output, cpu_blurred)
    save_uint8_image(gpu_blur_output, gpu_blurred_cpu)
    save_uint8_image(cpu_magnitude_output, normalize_to_uint8(cpu_magnitude))
    save_uint8_image(gpu_magnitude_output, normalize_to_uint8(gpu_magnitude_cpu))

    print()
    print("Canny frontend benchmark results")
    print("--------------------------------")
    print(f"CPU Gaussian+Sobel average time: {cpu_time:.6f} seconds")
    print(f"GPU compute-only average time: {gpu_compute_time:.6f} seconds")
    print(f"GPU with input transfer average time: {gpu_input_transfer_time:.6f} seconds")
    print(f"GPU end-to-end average time: {gpu_end_to_end_time:.6f} seconds")

    if gpu_compute_time > 0:
        print(f"Compute-only GPU speedup over CPU: {cpu_time / gpu_compute_time:.2f}x")

    if gpu_input_transfer_time > 0:
        print(
            "GPU-with-input-transfer speedup over CPU: "
            f"{cpu_time / gpu_input_transfer_time:.2f}x"
        )

    if gpu_end_to_end_time > 0:
        print(f"End-to-end GPU speedup over CPU: {cpu_time / gpu_end_to_end_time:.2f}x")

    print(f"Max absolute difference between CPU and GPU blur: {blur_max_abs_diff:.6f}")
    print(
        "Max absolute difference between CPU and GPU magnitude: "
        f"{magnitude_max_abs_diff:.6f}"
    )
    print(
        "Max absolute difference for end-to-end GPU magnitude: "
        f"{end_to_end_max_abs_diff:.6f}"
    )

    print(f"CPU blur saved to: {cpu_blur_output}")
    print(f"GPU blur saved to: {gpu_blur_output}")
    print(f"CPU magnitude saved to: {cpu_magnitude_output}")
    print(f"GPU magnitude saved to: {gpu_magnitude_output}")

    print()
    print(
        "Note: this is the Canny frontend only: Gaussian blur followed by Sobel. "
        "It does not yet include non-maximum suppression, double threshold, or hysteresis."
    )


if __name__ == "__main__":
    main()