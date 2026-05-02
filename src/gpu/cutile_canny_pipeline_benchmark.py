import argparse
import csv
import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

from gaussian_benchmark import (
    gaussian_blur_cpu,
    gaussian_blur_gpu_compute_only,
    load_grayscale_image,
    make_gaussian_kernel,
)
from sobel_benchmark import normalize_to_uint8, sobel_cpu
from sobel_cutile_benchmark import launch_sobel_cutile


def sobel_angle_cupy_compute_only(image_gpu):
    """
    Compute Sobel gradient angle on GPU using CuPy.

    The cuTile Sobel kernel currently computes magnitude only. NMS also needs
    gradient direction, so this function computes only the angle using the same
    Sobel stencil as the CPU reference.
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

    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0

    return angle


def non_max_suppression_cpu(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Apply non-maximum suppression on CPU.

    Each pixel is compared with two neighbours along the quantised gradient
    direction. Non-local maxima are suppressed to zero.
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

    This is used after Gaussian blur and cuTile Sobel magnitude while the arrays
    are still on the GPU.
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


def canny_pipeline_cpu(image_cpu: np.ndarray, kernel_cpu: np.ndarray):
    """
    CPU reference pipeline:
    Gaussian blur -> Sobel magnitude/angle -> non-maximum suppression.
    """
    blurred_cpu = gaussian_blur_cpu(image_cpu, kernel_cpu)
    magnitude_cpu, angle_cpu = sobel_cpu(blurred_cpu)
    nms_cpu = non_max_suppression_cpu(magnitude_cpu, angle_cpu)

    return blurred_cpu, magnitude_cpu, angle_cpu, nms_cpu


def canny_pipeline_gpu_compute_only(image_gpu, kernel_gpu, tile_size: int):
    """
    GPU pipeline:
    CuPy Gaussian blur -> cuTile Sobel magnitude -> CuPy Sobel angle -> CuPy NMS.

    The input image and Gaussian kernel are already on the GPU, so this excludes
    CPU-to-GPU transfer cost.
    """
    blurred_gpu = gaussian_blur_gpu_compute_only(image_gpu, kernel_gpu)

    magnitude_gpu = launch_sobel_cutile(
        image_gpu=blurred_gpu,
        tile_size=tile_size,
    )

    angle_gpu = sobel_angle_cupy_compute_only(blurred_gpu)

    nms_gpu = non_max_suppression_gpu_compute_only(
        magnitude_gpu=magnitude_gpu,
        angle_gpu=angle_gpu,
    )

    return blurred_gpu, magnitude_gpu, angle_gpu, nms_gpu


def canny_pipeline_gpu_with_input_transfer(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
):
    """
    Copy input image and Gaussian kernel to GPU, then run the GPU pipeline.
    Output remains on GPU.
    """
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)

    return canny_pipeline_gpu_compute_only(
        image_gpu=image_gpu,
        kernel_gpu=kernel_gpu,
        tile_size=tile_size,
    )


def canny_pipeline_gpu_end_to_end(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
):
    """
    Copy input to GPU, run the GPU pipeline, then copy NMS output back to CPU.

    This is closer to real-time display use cases because OpenCV usually reads
    frames on CPU and displays CPU images.
    """
    _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        tile_size=tile_size,
    )

    nms_cpu = cp.asnumpy(nms_gpu)

    return nms_cpu


def benchmark_cpu_pipeline(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    runs: int,
    warmup: int,
):
    """Benchmark CPU Gaussian + Sobel + NMS pipeline."""
    for _ in range(warmup):
        blurred_cpu, magnitude_cpu, angle_cpu, nms_cpu = canny_pipeline_cpu(
            image_cpu,
            kernel_cpu,
        )

    start = time.perf_counter()

    for _ in range(runs):
        blurred_cpu, magnitude_cpu, angle_cpu, nms_cpu = canny_pipeline_cpu(
            image_cpu,
            kernel_cpu,
        )

    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, blurred_cpu, magnitude_cpu, angle_cpu, nms_cpu


def benchmark_gpu_pipeline_for_tile_size(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
    runs: int,
    warmup: int,
):
    """
    Benchmark one tile size with three GPU timing modes:
    compute-only, with input transfer, and end-to-end.
    """
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)

    cp.cuda.Stream.null.synchronize()

    for _ in range(warmup):
        _, _, _, nms_gpu = canny_pipeline_gpu_compute_only(
            image_gpu=image_gpu,
            kernel_gpu=kernel_gpu,
            tile_size=tile_size,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        _, _, _, nms_gpu = canny_pipeline_gpu_compute_only(
            image_gpu=image_gpu,
            kernel_gpu=kernel_gpu,
            tile_size=tile_size,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    compute_only_time = (end - start) / runs
    nms_compute_only_cpu = cp.asnumpy(nms_gpu)

    for _ in range(warmup):
        _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=tile_size,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=tile_size,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    input_transfer_time = (end - start) / runs

    for _ in range(warmup):
        nms_end_to_end_cpu = canny_pipeline_gpu_end_to_end(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=tile_size,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        nms_end_to_end_cpu = canny_pipeline_gpu_end_to_end(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=tile_size,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    end_to_end_time = (end - start) / runs

    return (
        compute_only_time,
        input_transfer_time,
        end_to_end_time,
        nms_compute_only_cpu,
        nms_end_to_end_cpu,
    )


def parse_tile_sizes(tile_sizes_text: str):
    tile_sizes = []

    for item in tile_sizes_text.split(","):
        value = int(item.strip())

        if value <= 0:
            raise ValueError("tile sizes must be positive integers.")

        tile_sizes.append(value)

    return tile_sizes


def make_output_tag(image_path: Path, kernel_size: int, sigma: float) -> str:
    sigma_text = f"{sigma:g}".replace(".", "p")
    return f"{image_path.stem}_k{kernel_size}_s{sigma_text}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Canny frontend pipeline using CuPy Gaussian blur, "
            "cuTile Sobel magnitude, and GPU non-maximum suppression."
        )
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
        help="Odd Gaussian kernel size.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma value.",
    )

    parser.add_argument(
        "--tile-sizes",
        type=str,
        default="64,128,256,512",
        help="Comma-separated cuTile tile sizes to test.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed runs.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up runs before timing.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Directory for output images and CSV results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive.")

    if args.warmup < 0:
        raise ValueError("warmup must not be negative.")

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_sizes = parse_tile_sizes(args.tile_sizes)

    image_cpu = load_grayscale_image(image_path)
    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    print(f"Input image: {image_path}")
    print(f"Image shape: {image_cpu.shape}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Sigma: {args.sigma}")
    print(f"Tile sizes: {tile_sizes}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")

    cpu_time, _, _, _, nms_cpu = benchmark_cpu_pipeline(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    results = []
    best_end_to_end_result = None

    for tile_size in tile_sizes:
        (
            compute_only_time,
            input_transfer_time,
            end_to_end_time,
            nms_compute_only_cpu,
            nms_end_to_end_cpu,
        ) = benchmark_gpu_pipeline_for_tile_size(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=tile_size,
            runs=args.runs,
            warmup=args.warmup,
        )

        compute_only_diff = float(np.max(np.abs(nms_cpu - nms_compute_only_cpu)))
        end_to_end_diff = float(np.max(np.abs(nms_cpu - nms_end_to_end_cpu)))

        result = {
            "tile_size": tile_size,
            "cpu_time_seconds": cpu_time,
            "gpu_compute_only_seconds": compute_only_time,
            "gpu_with_input_transfer_seconds": input_transfer_time,
            "gpu_end_to_end_seconds": end_to_end_time,
            "speedup_compute_only": cpu_time / compute_only_time
            if compute_only_time > 0
            else float("inf"),
            "speedup_with_input_transfer": cpu_time / input_transfer_time
            if input_transfer_time > 0
            else float("inf"),
            "speedup_end_to_end": cpu_time / end_to_end_time
            if end_to_end_time > 0
            else float("inf"),
            "max_abs_diff_compute_only": compute_only_diff,
            "max_abs_diff_end_to_end": end_to_end_diff,
            "nms_output": nms_end_to_end_cpu,
        }

        results.append(result)

        if (
            best_end_to_end_result is None
            or end_to_end_time < best_end_to_end_result["gpu_end_to_end_seconds"]
        ):
            best_end_to_end_result = result

    output_tag = make_output_tag(image_path, args.kernel_size, args.sigma)

    csv_output_path = output_dir / f"19_cutile_canny_pipeline_{output_tag}.csv"
    cpu_output_path = output_dir / f"20_pipeline_cpu_nms_{output_tag}.png"
    gpu_output_path = (
        output_dir
        / f"21_pipeline_gpu_nms_{output_tag}_tile{best_end_to_end_result['tile_size']}.png"
    )

    with csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "tile_size",
                "cpu_time_seconds",
                "gpu_compute_only_seconds",
                "gpu_with_input_transfer_seconds",
                "gpu_end_to_end_seconds",
                "speedup_compute_only",
                "speedup_with_input_transfer",
                "speedup_end_to_end",
                "max_abs_diff_compute_only",
                "max_abs_diff_end_to_end",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "tile_size": result["tile_size"],
                    "cpu_time_seconds": f"{result['cpu_time_seconds']:.8f}",
                    "gpu_compute_only_seconds": (
                        f"{result['gpu_compute_only_seconds']:.8f}"
                    ),
                    "gpu_with_input_transfer_seconds": (
                        f"{result['gpu_with_input_transfer_seconds']:.8f}"
                    ),
                    "gpu_end_to_end_seconds": (
                        f"{result['gpu_end_to_end_seconds']:.8f}"
                    ),
                    "speedup_compute_only": f"{result['speedup_compute_only']:.4f}",
                    "speedup_with_input_transfer": (
                        f"{result['speedup_with_input_transfer']:.4f}"
                    ),
                    "speedup_end_to_end": f"{result['speedup_end_to_end']:.4f}",
                    "max_abs_diff_compute_only": (
                        f"{result['max_abs_diff_compute_only']:.8f}"
                    ),
                    "max_abs_diff_end_to_end": (
                        f"{result['max_abs_diff_end_to_end']:.8f}"
                    ),
                }
            )

    cv2.imwrite(str(cpu_output_path), normalize_to_uint8(nms_cpu))
    cv2.imwrite(
        str(gpu_output_path),
        normalize_to_uint8(best_end_to_end_result["nms_output"]),
    )

    print()
    print("cuTile Canny pipeline benchmark results")
    print("--------------------------------------")
    print(f"CPU Gaussian+Sobel+NMS time: {cpu_time:.6f} seconds")

    for result in results:
        print(
            f"tile_size={result['tile_size']:>4} | "
            f"GPU compute-only={result['gpu_compute_only_seconds']:.6f}s | "
            f"GPU input-transfer={result['gpu_with_input_transfer_seconds']:.6f}s | "
            f"GPU end-to-end={result['gpu_end_to_end_seconds']:.6f}s | "
            f"end-to-end speedup={result['speedup_end_to_end']:.2f}x | "
            f"max diff={result['max_abs_diff_end_to_end']:.6f}"
        )

    print()
    print(f"Best end-to-end tile size: {best_end_to_end_result['tile_size']}")
    print(f"CSV results saved to: {csv_output_path}")
    print(f"CPU NMS output saved to: {cpu_output_path}")
    print(f"Best GPU NMS output saved to: {gpu_output_path}")

    print()
    print(
        "Note: this pipeline uses CuPy for Gaussian blur, cuTile for Sobel "
        "magnitude, CuPy for Sobel angle, and CuPy for non-maximum suppression. "
        "It does not yet include double threshold or hysteresis."
    )


if __name__ == "__main__":
    main()