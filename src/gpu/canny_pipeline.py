import argparse
import time
from pathlib import Path
import cv2
import cupy as cp
import numpy as np


def make_gaussian_kernel(kernel_size, sigma):
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_gpu(image_gpu, kernel_gpu):
    radius = kernel_gpu.size // 2
    height, width = image_gpu.shape
    # Horizontal pass
    padded_horizontal = cp.pad(image_gpu, ((0, 0), (radius, radius)), mode="edge")
    temp = cp.zeros_like(image_gpu, dtype=cp.float32)
    for offset in range(kernel_gpu.size):
        temp += kernel_gpu[offset] * padded_horizontal[:, offset:offset + width]
    # Vertical pass
    padded_vertical = cp.pad(temp, ((radius, radius), (0, 0)), mode="edge")
    output = cp.zeros_like(image_gpu, dtype=cp.float32)
    for offset in range(kernel_gpu.size):
        output += kernel_gpu[offset] * padded_vertical[offset:offset + height, :]
    return output

def sobel_gpu(image_gpu):
    gx = cp.zeros_like(image_gpu, dtype=cp.float32)
    gy = cp.zeros_like(image_gpu, dtype=cp.float32)
    gx[1:-1, 1:-1] = (
        -image_gpu[:-2, :-2] + image_gpu[:-2, 2:]
        - 2.0 * image_gpu[1:-1, :-2] + 2.0 * image_gpu[1:-1, 2:]
        - image_gpu[2:, :-2] + image_gpu[2:, 2:])
    gy[1:-1, 1:-1] = (
        image_gpu[:-2, :-2] + 2.0 * image_gpu[:-2, 1:-1] + image_gpu[:-2, 2:]
        - image_gpu[2:, :-2] - 2.0 * image_gpu[2:, 1:-1] - image_gpu[2:, 2:])
    magnitude = cp.sqrt(gx * gx + gy * gy)
    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0
    return magnitude, angle


def nms_gpu(magnitude_gpu, angle_gpu):
    angle = angle_gpu % 180.0
    output = cp.zeros_like(magnitude_gpu, dtype=cp.float32)
    center = magnitude_gpu[1:-1, 1:-1]
    direction = angle[1:-1, 1:-1]
    d0   = (direction < 22.5)   | (direction >= 157.5)
    d45  = (direction >= 22.5)  & (direction < 67.5)
    d90  = (direction >= 67.5)  & (direction < 112.5)
    d135 = (direction >= 112.5) & (direction < 157.5)
    keep = (
        (d0   & (center >= magnitude_gpu[1:-1, :-2]) & (center >= magnitude_gpu[1:-1, 2:])) |
        (d45  & (center >= magnitude_gpu[:-2, 2:])   & (center >= magnitude_gpu[2:, :-2]))  |
        (d90  & (center >= magnitude_gpu[:-2, 1:-1]) & (center >= magnitude_gpu[2:, 1:-1])) |
        (d135 & (center >= magnitude_gpu[:-2, :-2])  & (center >= magnitude_gpu[2:, 2:]))
    )
    output[1:-1, 1:-1] = cp.where(keep, center, 0.0)
    return output


def double_threshold(nms_cpu, mode, high_pct, low_ratio, low_fix, high_fix):
    if mode == "fixed":
        low, high = float(low_fix), float(high_fix)
    else:
        pos = nms_cpu[nms_cpu > 0]
        if pos.size == 0:
            return float("inf"), float("inf"), np.zeros_like(nms_cpu, dtype=bool), np.zeros_like(nms_cpu, dtype=bool)
        high = float(np.percentile(pos, high_pct))
        low = float(low_ratio * high)
    strong = nms_cpu >= high
    weak = (nms_cpu >= low) & ~strong
    return low, high, strong, weak


def hysteresis(strong, weak):
    if not np.any(strong):
        return np.zeros_like(strong, dtype=bool)
    candidate = (strong | weak).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)
    keep = np.zeros(num_labels, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def run_canny_pipeline(image_cpu, kernel_size=5, sigma=1.4,
                       threshold_mode="percentile", high_percentile=90.0,
                       low_ratio=0.5, low_threshold=None, high_threshold=None):
    times = {}

    t0 = time.perf_counter()
    kernel_cpu = make_gaussian_kernel(kernel_size, sigma)
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)
    blurred_gpu = gaussian_blur_gpu(image_gpu, kernel_gpu)
    cp.cuda.Stream.null.synchronize()
    times["gaussian"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    magnitude_gpu, angle_gpu = sobel_gpu(blurred_gpu)
    cp.cuda.Stream.null.synchronize()
    times["sobel"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nms_result = nms_gpu(magnitude_gpu, angle_gpu)
    cp.cuda.Stream.null.synchronize()
    times["nms"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nms_cpu = cp.asnumpy(nms_result)
    times["transfer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    low, high, strong, weak = double_threshold(
        nms_cpu, threshold_mode, high_percentile, low_ratio, low_threshold, high_threshold)
    times["threshold"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges = hysteresis(strong, weak)
    times["hysteresis"] = time.perf_counter() - t0

    times["total"] = sum(times.values())
    return {"edges": edges, "low": low, "high": high, "times": times,
            "blurred_gpu": blurred_gpu, "magnitude_gpu": magnitude_gpu,
            "nms_gpu": nms_result, "nms_cpu": nms_cpu}


def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end Canny Pipeline")
    p.add_argument("--image", type=Path, default=Path("data/test.jpg"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--sigma", type=float, default=1.4)
    p.add_argument("--threshold-mode", choices=["percentile", "fixed"], default="percentile")
    p.add_argument("--high-percentile", type=float, default=90.0)
    p.add_argument("--low-ratio", type=float, default=0.5)
    p.add_argument("--low-threshold", type=float, default=None)
    p.add_argument("--high-threshold", type=float, default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    image_cpu = load_grayscale(args.image.resolve())
    print(f"Image: {args.image}  Shape: {image_cpu.shape}")
    kwargs = dict(
        kernel_size=args.kernel_size, sigma=args.sigma,
        threshold_mode=args.threshold_mode, high_percentile=args.high_percentile,
        low_ratio=args.low_ratio, low_threshold=args.low_threshold,
        high_threshold=args.high_threshold)

    if args.benchmark:
        for _ in range(args.warmup):
            run_canny_pipeline(image_cpu, **kwargs)
        results = [run_canny_pipeline(image_cpu, **kwargs) for _ in range(args.runs)]
        avg = {k: float(np.mean([r["times"][k] for r in results])) for k in results[0]["times"]}
        print(f"Benchmark ({args.runs} runs):")
        for k, v in avg.items():
            print(f"  {k:<12} {v*1000:8.3f} ms")
    else:
        result = run_canny_pipeline(image_cpu, **kwargs)
        print("Pipeline timing:")
        for k, v in result["times"].items():
            print(f"  {k:<12} {v*1000:8.3f} ms")
        print(f"  low={result['low']:.2f}  high={result['high']:.2f}  edges={result['edges'].sum()}")
        edges_uint8 = result["edges"].astype(np.uint8) * 255
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), edges_uint8)
            print(f"  Saved: {args.output}")
        if not args.no_display:
            cv2.imshow("Canny Edges", edges_uint8)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()