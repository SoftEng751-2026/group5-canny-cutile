import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

from cutile_canny_pipeline_benchmark import (
    canny_pipeline_cpu,
    canny_pipeline_gpu_with_input_transfer,
)
from gaussian_benchmark import load_grayscale_image, make_gaussian_kernel


def select_thresholds(
    nms_image: np.ndarray,
    mode: str,
    high_percentile: float,
    low_ratio: float,
    low_threshold: float | None,
    high_threshold: float | None,
):
    """
    Select low/high thresholds for Canny.

    percentile mode adapts to each image, avoiding hardcoded thresholds.
    fixed mode allows explicit threshold values for controlled experiments.
    """
    if mode == "fixed":
        if low_threshold is None or high_threshold is None:
            raise ValueError("fixed mode requires --low-threshold and --high-threshold")

        if low_threshold < 0 or high_threshold < 0:
            raise ValueError("thresholds must be non-negative")

        if low_threshold > high_threshold:
            raise ValueError("low-threshold must be <= high-threshold")

        return float(low_threshold), float(high_threshold)

    if not 0.0 <= high_percentile <= 100.0:
        raise ValueError("high-percentile must be between 0 and 100")

    if not 0.0 < low_ratio <= 1.0:
        raise ValueError("low-ratio must be in (0, 1]")

    positive_values = nms_image[nms_image > 0.0]

    if positive_values.size == 0:
        return float("inf"), float("inf")

    high = float(np.percentile(positive_values, high_percentile))
    low = float(low_ratio * high)

    return low, high


def hysteresis_cpu(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """
    Canny hysteresis using breadth-first search.

    Strong edge pixels are kept. Weak pixels are kept only if they are connected
    to a strong edge through an 8-neighbour path.
    """
    if strong.shape != weak.shape:
        raise ValueError("strong and weak masks must have the same shape")

    height, width = strong.shape
    edges = np.zeros_like(strong, dtype=bool)

    queue = deque()

    strong_positions = np.argwhere(strong)

    for row, col in strong_positions:
        edges[row, col] = True
        queue.append((int(row), int(col)))

    neighbour_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while queue:
        row, col = queue.popleft()

        for row_offset, col_offset in neighbour_offsets:
            next_row = row + row_offset
            next_col = col + col_offset

            if next_row < 0 or next_row >= height:
                continue

            if next_col < 0 or next_col >= width:
                continue

            if weak[next_row, next_col] and not edges[next_row, next_col]:
                edges[next_row, next_col] = True
                queue.append((next_row, next_col))

    return edges


def double_threshold_and_hysteresis_cpu(
    nms_image: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """
    Apply double threshold and hysteresis to an NMS image.
    """
    strong = nms_image >= high_threshold
    weak = (nms_image >= low_threshold) & ~strong

    edges = hysteresis_cpu(strong, weak)

    return edges


def complete_canny_cpu(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    threshold_mode: str,
    high_percentile: float,
    low_ratio: float,
    low_threshold: float | None,
    high_threshold: float | None,
):
    """
    Full CPU reference:
    Gaussian -> Sobel -> NMS -> double threshold -> hysteresis.
    """
    _, _, _, nms_cpu = canny_pipeline_cpu(image_cpu, kernel_cpu)

    low, high = select_thresholds(
        nms_cpu,
        threshold_mode,
        high_percentile,
        low_ratio,
        low_threshold,
        high_threshold,
    )

    edges_cpu = double_threshold_and_hysteresis_cpu(nms_cpu, low, high)

    return nms_cpu, edges_cpu, low, high


def complete_canny_gpu_frontend_cpu_postprocess(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
    threshold_mode: str,
    high_percentile: float,
    low_ratio: float,
    low_threshold: float | None,
    high_threshold: float | None,
):
    """
    Full prototype pipeline:
    GPU/cuTile frontend -> copy NMS to CPU -> double threshold + hysteresis on CPU.

    This is useful before integrating a dedicated GPU hysteresis implementation.
    """
    _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        tile_size=tile_size,
    )

    nms_cpu = cp.asnumpy(nms_gpu)

    low, high = select_thresholds(
        nms_cpu,
        threshold_mode,
        high_percentile,
        low_ratio,
        low_threshold,
        high_threshold,
    )

    edges_cpu = double_threshold_and_hysteresis_cpu(nms_cpu, low, high)

    return nms_cpu, edges_cpu, low, high


def benchmark_cpu(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    args,
):
    for _ in range(args.warmup):
        nms_cpu, edges_cpu, low, high = complete_canny_cpu(
            image_cpu,
            kernel_cpu,
            args.threshold_mode,
            args.high_percentile,
            args.low_ratio,
            args.low_threshold,
            args.high_threshold,
        )

    start = time.perf_counter()

    for _ in range(args.runs):
        nms_cpu, edges_cpu, low, high = complete_canny_cpu(
            image_cpu,
            kernel_cpu,
            args.threshold_mode,
            args.high_percentile,
            args.low_ratio,
            args.low_threshold,
            args.high_threshold,
        )

    end = time.perf_counter()

    return (end - start) / args.runs, nms_cpu, edges_cpu, low, high


def complete_canny_gpu_full(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
    threshold_mode: str,
    high_percentile: float,
    low_ratio: float,
    low_threshold: float | None,
    high_threshold: float | None,
):
    """
    Fully GPU pipeline: GPU frontend + GPU threshold + GPU hysteresis via cupyx.
    No NMS→CPU transfer; hysteresis runs on GPU with cupyx.scipy.ndimage.label.
    """
    from cupyx.scipy import ndimage as cpnd

    _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        tile_size=tile_size,
    )

    if threshold_mode == "fixed":
        if low_threshold is None or high_threshold is None:
            raise ValueError("fixed mode requires low_threshold and high_threshold")
        low = float(low_threshold)
        high = float(high_threshold)
    else:
        pos = nms_gpu[nms_gpu > 0]
        if pos.size == 0:
            return np.zeros(nms_gpu.shape, dtype=bool), float("inf"), float("inf")
        high = float(cp.percentile(pos, high_percentile))
        low = float(low_ratio * high)

    strong = nms_gpu >= high
    weak = (nms_gpu >= low) & ~strong
    candidate = (strong | weak).astype(cp.uint8)
    labels, n = cpnd.label(candidate, structure=cp.ones((3, 3), dtype=cp.int32))
    keep = cp.zeros(n + 1, dtype=cp.bool_)
    keep[cp.unique(labels[strong])] = True
    keep[0] = False
    edges_gpu = keep[labels]

    edges_cpu = cp.asnumpy(edges_gpu)
    nms_cpu = cp.asnumpy(nms_gpu)
    return nms_cpu, edges_cpu, low, high


def benchmark_gpu_full(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
    args,
):
    for _ in range(args.warmup):
        complete_canny_gpu_full(
            image_cpu, kernel_cpu, tile_size,
            args.threshold_mode, args.high_percentile, args.low_ratio,
            args.low_threshold, args.high_threshold,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(args.runs):
        nms_cpu, edges_cpu, low, high = complete_canny_gpu_full(
            image_cpu, kernel_cpu, tile_size,
            args.threshold_mode, args.high_percentile, args.low_ratio,
            args.low_threshold, args.high_threshold,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / args.runs, nms_cpu, edges_cpu, low, high


def benchmark_gpu_frontend(
    image_cpu: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
    args,
):
    for _ in range(args.warmup):
        nms_cpu, edges_cpu, low, high = complete_canny_gpu_frontend_cpu_postprocess(
            image_cpu,
            kernel_cpu,
            tile_size,
            args.threshold_mode,
            args.high_percentile,
            args.low_ratio,
            args.low_threshold,
            args.high_threshold,
        )

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(args.runs):
        nms_cpu, edges_cpu, low, high = complete_canny_gpu_frontend_cpu_postprocess(
            image_cpu,
            kernel_cpu,
            tile_size,
            args.threshold_mode,
            args.high_percentile,
            args.low_ratio,
            args.low_threshold,
            args.high_threshold,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / args.runs, nms_cpu, edges_cpu, low, high


def parse_tile_sizes(tile_sizes_text: str):
    tile_sizes = []

    for item in tile_sizes_text.split(","):
        value = int(item.strip())

        if value <= 0:
            raise ValueError("tile sizes must be positive integers")

        tile_sizes.append(value)

    return tile_sizes


def make_output_tag(image_path: Path, kernel_size: int, sigma: float, threshold_mode: str):
    sigma_text = f"{sigma:g}".replace(".", "p")
    return f"{image_path.stem}_k{kernel_size}_s{sigma_text}_{threshold_mode}"


def save_edge_image(path: Path, edges: np.ndarray):
    image = edges.astype(np.uint8) * 255
    cv2.imwrite(str(path), image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a full Canny prototype with GPU/cuTile frontend."
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
        help="Comma-separated cuTile tile sizes.",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["percentile", "fixed"],
        default="percentile",
        help="How to choose Canny thresholds.",
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=90.0,
        help="High threshold percentile for percentile mode.",
    )

    parser.add_argument(
        "--low-ratio",
        type=float,
        default=0.5,
        help="Low threshold = low-ratio * high threshold.",
    )

    parser.add_argument(
        "--low-threshold",
        type=float,
        default=None,
        help="Fixed low threshold for fixed mode.",
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=None,
        help="Fixed high threshold for fixed mode.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Output directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive")

    if args.warmup < 0:
        raise ValueError("warmup must not be negative")

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
    print(f"Threshold mode: {args.threshold_mode}")
    print(f"High percentile: {args.high_percentile}")
    print(f"Low ratio: {args.low_ratio}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")

    cpu_time, cpu_nms, cpu_edges, cpu_low, cpu_high = benchmark_cpu(
        image_cpu,
        kernel_cpu,
        args,
    )

    results = []
    best_result = None

    for tile_size in tile_sizes:
        gpu_hybrid_time, gpu_nms, gpu_hybrid_edges, gpu_low, gpu_high = benchmark_gpu_frontend(
            image_cpu, kernel_cpu, tile_size, args,
        )
        gpu_full_time, _, gpu_full_edges, gf_low, gf_high = benchmark_gpu_full(
            image_cpu, kernel_cpu, tile_size, args,
        )

        nms_max_abs_diff = float(np.max(np.abs(cpu_nms - gpu_nms)))
        diff_hybrid = int(np.count_nonzero(cpu_edges != gpu_hybrid_edges))
        diff_full   = int(np.count_nonzero(cpu_edges != gpu_full_edges))

        result = {
            "tile_size": tile_size,
            "cpu_time_seconds": cpu_time,
            "gpu_frontend_cpu_postprocess_seconds": gpu_hybrid_time,
            "gpu_full_seconds": gpu_full_time,
            "speedup": cpu_time / gpu_hybrid_time if gpu_hybrid_time > 0 else float("inf"),
            "speedup_full_gpu": cpu_time / gpu_full_time if gpu_full_time > 0 else float("inf"),
            "cpu_low_threshold": cpu_low,
            "cpu_high_threshold": cpu_high,
            "gpu_low_threshold": gpu_low,
            "gpu_high_threshold": gpu_high,
            "nms_max_abs_diff": nms_max_abs_diff,
            "different_pixels": diff_hybrid,
            "different_pixel_ratio": diff_hybrid / cpu_edges.size,
            "different_pixels_full_gpu": diff_full,
            "different_pixel_ratio_full_gpu": diff_full / cpu_edges.size,
            "gpu_hybrid_edges": gpu_hybrid_edges,
            "gpu_full_edges": gpu_full_edges,
        }

        results.append(result)

        if best_result is None or gpu_full_time < best_result["gpu_full_seconds"]:
            best_result = result

    output_tag = make_output_tag(image_path, args.kernel_size, args.sigma, args.threshold_mode)

    csv_output_path = output_dir / f"22_complete_canny_benchmark_{output_tag}.csv"
    cpu_edge_output_path = output_dir / f"23_complete_canny_cpu_edges_{output_tag}.png"
    gpu_hybrid_edge_output_path = (
        output_dir / f"24_complete_canny_gpu_edges_{output_tag}_tile{best_result['tile_size']}.png"
    )
    gpu_full_edge_output_path = (
        output_dir / f"24b_complete_canny_gpu_full_edges_{output_tag}_tile{best_result['tile_size']}.png"
    )

    with csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "tile_size",
                "cpu_time_seconds",
                "gpu_frontend_cpu_postprocess_seconds",
                "gpu_full_seconds",
                "speedup",
                "speedup_full_gpu",
                "cpu_low_threshold",
                "cpu_high_threshold",
                "gpu_low_threshold",
                "gpu_high_threshold",
                "nms_max_abs_diff",
                "different_pixels",
                "different_pixel_ratio",
                "different_pixels_full_gpu",
                "different_pixel_ratio_full_gpu",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "tile_size": result["tile_size"],
                "cpu_time_seconds": f"{result['cpu_time_seconds']:.8f}",
                "gpu_frontend_cpu_postprocess_seconds": f"{result['gpu_frontend_cpu_postprocess_seconds']:.8f}",
                "gpu_full_seconds": f"{result['gpu_full_seconds']:.8f}",
                "speedup": f"{result['speedup']:.4f}",
                "speedup_full_gpu": f"{result['speedup_full_gpu']:.4f}",
                "cpu_low_threshold": f"{result['cpu_low_threshold']:.8f}",
                "cpu_high_threshold": f"{result['cpu_high_threshold']:.8f}",
                "gpu_low_threshold": f"{result['gpu_low_threshold']:.8f}",
                "gpu_high_threshold": f"{result['gpu_high_threshold']:.8f}",
                "nms_max_abs_diff": f"{result['nms_max_abs_diff']:.8f}",
                "different_pixels": result["different_pixels"],
                "different_pixel_ratio": f"{result['different_pixel_ratio']:.8f}",
                "different_pixels_full_gpu": result["different_pixels_full_gpu"],
                "different_pixel_ratio_full_gpu": f"{result['different_pixel_ratio_full_gpu']:.8f}",
            })

    save_edge_image(cpu_edge_output_path, cpu_edges)
    save_edge_image(gpu_hybrid_edge_output_path, best_result["gpu_hybrid_edges"])
    save_edge_image(gpu_full_edge_output_path, best_result["gpu_full_edges"])

    print()
    print("Complete Canny benchmark results (3-way comparison)")
    print("----------------------------------------------------")
    print(f"CPU full Canny time: {cpu_time * 1000:.2f} ms")
    print()
    print(f"{'Tile':>5} | {'Hybrid(ms)':>10} | {'Speedup':>8} | {'Full GPU(ms)':>12} | {'Speedup':>8} | {'NMS diff':>9} | {'Edge diff%':>10}")
    print("-" * 85)
    for result in results:
        print(
            f"{result['tile_size']:>5} | "
            f"{result['gpu_frontend_cpu_postprocess_seconds']*1000:>10.2f} | "
            f"{result['speedup']:>8.2f}x | "
            f"{result['gpu_full_seconds']*1000:>12.2f} | "
            f"{result['speedup_full_gpu']:>8.2f}x | "
            f"{result['nms_max_abs_diff']:>9.6f} | "
            f"{result['different_pixel_ratio_full_gpu']*100:>10.4f}%"
        )

    print()
    print(f"Best tile size (full GPU): {best_result['tile_size']}")
    print(f"CSV results saved to: {csv_output_path}")
    print(f"CPU edge image saved to: {cpu_edge_output_path}")
    print(f"Best hybrid GPU edge image saved to: {gpu_hybrid_edge_output_path}")
    print(f"Best full GPU edge image saved to: {gpu_full_edge_output_path}")


if __name__ == "__main__":
    main()