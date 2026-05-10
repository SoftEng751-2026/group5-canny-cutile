import argparse
import csv
import time
from pathlib import Path
 
import cv2
import cupy as cp
import numpy as np
 
from gaussian_benchmark import load_grayscale_image
from sobel_benchmark import normalize_to_uint8, sobel_cpu
 
 
def sobel_magnitude_cupy(image_gpu):
    """Compute Sobel magnitude using CuPy (GPU)."""
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
    return magnitude
 
 
def benchmark_cupy_sobel(image_gpu, runs: int, warmup: int):
    """Benchmark CuPy Sobel magnitude."""
    for _ in range(warmup):
        magnitude_gpu = sobel_magnitude_cupy(image_gpu)
 
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
 
    for _ in range(runs):
        magnitude_gpu = sobel_magnitude_cupy(image_gpu)
 
    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()
 
    average_time = (end - start) / runs
    return average_time, magnitude_gpu
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Sobel magnitude using CuPy (GPU)."
    )
    parser.add_argument("--image", type=Path, default=Path("data/test.jpg"))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("report"))
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
 
    image_cpu = load_grayscale_image(image_path)
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
 
    print(f"Input image: {image_path}")
    print(f"Image shape: {image_cpu.shape}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")
 
    # CPU reference
    cpu_magnitude, _ = sobel_cpu(image_cpu)
 
    # CuPy GPU benchmark
    cupy_time, cupy_magnitude_gpu = benchmark_cupy_sobel(
        image_gpu=image_gpu,
        runs=args.runs,
        warmup=args.warmup,
    )
 
    cupy_magnitude_cpu = cp.asnumpy(cupy_magnitude_gpu)
    max_abs_diff = float(np.max(np.abs(cpu_magnitude - cupy_magnitude_cpu)))
 
    # Save output image
    output_image_path = output_dir / f"sobel_cupy_{image_path.stem}.png"
    cv2.imwrite(str(output_image_path), normalize_to_uint8(cupy_magnitude_cpu))
 
    # Save CSV
    csv_path = output_dir / f"sobel_cupy_benchmark_{image_path.stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["runs", "avg_time_seconds", "max_abs_diff_vs_cpu"])
        writer.writeheader()
        writer.writerow({
            "runs": args.runs,
            "avg_time_seconds": f"{cupy_time:.8f}",
            "max_abs_diff_vs_cpu": f"{max_abs_diff:.8f}",
        })
 
    print()
    print("CuPy Sobel Benchmark Results")
    print("-----------------------------")
    print(f"Average time : {cupy_time:.6f} seconds")
    print(f"Max diff vs CPU: {max_abs_diff:.6f}")
    print(f"Output image : {output_image_path}")
    print(f"CSV saved    : {csv_path}")
 
 
if __name__ == "__main__":
    main()