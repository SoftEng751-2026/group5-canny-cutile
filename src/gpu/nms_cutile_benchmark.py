"""
cuTile Non-Maximum Suppression kernel and tile-size sweep benchmark.

For each interior pixel the kernel loads the centre magnitude, its 8 neighbours,
and the quantised gradient angle, then applies the four-direction local-maximum
test in a single tile-parallel pass — mirroring the structure of
sobel_cutile_benchmark.py.

Pipeline integration
--------------------
launch_nms_cutile(magnitude_gpu, angle_gpu, tile_size) -> nms_gpu
  Drop-in replacement for non_max_suppression_gpu_compute_only().
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

try:
    import cuda.tile as ct
    _HAS_CUTILE = True
except Exception:
    _HAS_CUTILE = False

from gaussian_benchmark import load_grayscale_image, make_gaussian_kernel
from sobel_benchmark import normalize_to_uint8
from nms_benchmark import (
    non_max_suppression_cpu,
    non_max_suppression_gpu_compute_only,
    benchmark_cpu_nms,
    benchmark_gpu_nms_compute_only,
)
from canny_frontend_benchmark import (
    canny_frontend_cpu,
    canny_frontend_gpu_compute_only,
)


# ── cuTile NMS kernel ─────────────────────────────────────────────────────────

if _HAS_CUTILE:
    @ct.kernel
    def _nms_cutile_kernel(
        mag_c,
        mag_l, mag_r,
        mag_t, mag_b,
        mag_tl, mag_tr,
        mag_bl, mag_br,
        ang_c,
        output,
        tile_size: ct.Constant[int],
    ):
        """
        cuTile NMS kernel.

        Each tile processes tile_size consecutive interior pixels (row-major
        flattened). For each pixel:
          - angle in [0°, 22.5°) or [157.5°, 180°)  -> compare left/right
          - angle in [22.5°, 67.5°)                  -> compare top-right/bottom-left
          - angle in [67.5°, 112.5°)                 -> compare top/bottom
          - angle in [112.5°, 157.5°)                -> compare top-left/bottom-right
        """
        b = ct.bid(0)
        s = (tile_size,)

        mc  = ct.load(mag_c,  index=(b,), shape=s)
        ml  = ct.load(mag_l,  index=(b,), shape=s)
        mr  = ct.load(mag_r,  index=(b,), shape=s)
        mt  = ct.load(mag_t,  index=(b,), shape=s)
        mb  = ct.load(mag_b,  index=(b,), shape=s)
        mtl = ct.load(mag_tl, index=(b,), shape=s)
        mtr = ct.load(mag_tr, index=(b,), shape=s)
        mbl = ct.load(mag_bl, index=(b,), shape=s)
        mbr = ct.load(mag_br, index=(b,), shape=s)
        a   = ct.load(ang_c,  index=(b,), shape=s)

        d0   = (a < 22.5)   | (a >= 157.5)
        d45  = (a >= 22.5)  & (a < 67.5)
        d90  = (a >= 67.5)  & (a < 112.5)
        d135 = (a >= 112.5) & (a < 157.5)

        keep = (
            (d0   & (mc >= ml)  & (mc >= mr))  |
            (d45  & (mc >= mtr) & (mc >= mbl)) |
            (d90  & (mc >= mt)  & (mc >= mb))  |
            (d135 & (mc >= mtl) & (mc >= mbr))
        )

        ct.store(output, index=(b,), tile=ct.where(keep, mc, 0.0))


# ── Helper: pad to tile_size multiple ────────────────────────────────────────

def _pad(arr_gpu, tile_size):
    flat = cp.ascontiguousarray(arr_gpu.ravel())
    rem = flat.size % tile_size
    if rem:
        flat = cp.pad(flat, (0, tile_size - rem), mode="constant", constant_values=0)
    return flat


# ── Neighbour array preparation ───────────────────────────────────────────────

def prepare_nms_arrays(magnitude_gpu, angle_gpu, tile_size):
    """
    Slice and flatten the 9 magnitude neighbour views and 1 angle view needed
    by the cuTile NMS kernel.

    Only the interior region (H-2, W-2) is processed; boundary pixels remain 0.
    """
    if magnitude_gpu.shape != angle_gpu.shape:
        raise ValueError("magnitude and angle must have the same shape.")
    if magnitude_gpu.ndim != 2:
        raise ValueError("magnitude must be a 2D array.")

    H, W = magnitude_gpu.shape
    if H < 3 or W < 3:
        raise ValueError("image must be at least 3x3 for NMS.")

    mag_views = [
        magnitude_gpu[1:-1, 1:-1],   # center
        magnitude_gpu[1:-1, :-2],    # left
        magnitude_gpu[1:-1, 2:],     # right
        magnitude_gpu[:-2,  1:-1],   # top
        magnitude_gpu[2:,   1:-1],   # bottom
        magnitude_gpu[:-2,  :-2],    # top-left
        magnitude_gpu[:-2,  2:],     # top-right
        magnitude_gpu[2:,   :-2],    # bottom-left
        magnitude_gpu[2:,   2:],     # bottom-right
    ]
    ang_view = angle_gpu[1:-1, 1:-1]

    orig_size = mag_views[0].size
    padded_mags = [_pad(v, tile_size) for v in mag_views]
    padded_ang  = _pad(ang_view, tile_size)

    return padded_mags, padded_ang, orig_size, H, W


# ── Launch helper ─────────────────────────────────────────────────────────────

def launch_nms_cutile(magnitude_gpu, angle_gpu, tile_size):
    """
    Run the cuTile NMS kernel and return a full-size output image.

    Falls back silently to CuPy NMS if cuTile is unavailable.
    """
    if not _HAS_CUTILE:
        return non_max_suppression_gpu_compute_only(magnitude_gpu, angle_gpu)

    # Angle must be in [0, 180) — normalise on GPU before extracting views.
    angle_norm = angle_gpu % 180.0

    padded_mags, padded_ang, orig_size, H, W = prepare_nms_arrays(
        magnitude_gpu, angle_norm, tile_size
    )

    padded_size = padded_mags[0].size
    output_padded = cp.zeros(padded_size, dtype=cp.float32)
    grid = (padded_size // tile_size, 1, 1)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        _nms_cutile_kernel,
        (
            padded_mags[0],   # center
            padded_mags[1],   # left
            padded_mags[2],   # right
            padded_mags[3],   # top
            padded_mags[4],   # bottom
            padded_mags[5],   # top-left
            padded_mags[6],   # top-right
            padded_mags[7],   # bottom-left
            padded_mags[8],   # bottom-right
            padded_ang,
            output_padded,
            tile_size,
        ),
    )

    interior = output_padded[:orig_size].reshape(H - 2, W - 2)
    full_output = cp.zeros((H, W), dtype=cp.float32)
    full_output[1:-1, 1:-1] = interior
    return full_output


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def benchmark_cutile_nms(magnitude_gpu, angle_gpu, tile_size, runs, warmup):
    for _ in range(warmup):
        launch_nms_cutile(magnitude_gpu, angle_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        out = launch_nms_cutile(magnitude_gpu, angle_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / runs, out


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _parse_tile_sizes(text):
    values = []
    for item in text.split(","):
        v = int(item.strip())
        if v <= 0:
            raise ValueError("tile sizes must be positive integers.")
        values.append(v)
    return values


def _make_output_tag(image_path, kernel_size, sigma):
    sigma_text = f"{sigma:g}".replace(".", "p")
    return f"{image_path.stem}_k{kernel_size}_s{sigma_text}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark cuTile NMS vs CuPy NMS — tile-size sweep."
    )
    p.add_argument("--image",       type=Path,  default=Path("data/test.jpg"))
    p.add_argument("--kernel-size", type=int,   default=5)
    p.add_argument("--sigma",       type=float, default=1.4)
    p.add_argument("--tile-sizes",  type=str,   default="64,128,256,512")
    p.add_argument("--runs",        type=int,   default=30)
    p.add_argument("--warmup",      type=int,   default=5)
    p.add_argument("--output-dir",  type=Path,  default=Path("report"))
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive.")
    if args.warmup < 0:
        raise ValueError("warmup must not be negative.")

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_sizes = _parse_tile_sizes(args.tile_sizes)

    image_cpu  = load_grayscale_image(image_path)
    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    print(f"cuTile available : {_HAS_CUTILE}")
    print(f"Input image      : {image_path}")
    print(f"Image shape      : {image_cpu.shape}")
    print(f"Kernel size      : {args.kernel_size}")
    print(f"Sigma            : {args.sigma}")
    print(f"Tile sizes       : {tile_sizes}")
    print(f"Runs / warmup    : {args.runs} / {args.warmup}")
    print()

    # ── Reference CPU NMS ────────────────────────────────────────────────────
    print("Preparing CPU Gaussian + Sobel inputs for NMS...")
    _, magnitude_cpu, angle_cpu = canny_frontend_cpu(image_cpu, kernel_cpu)

    # ── GPU inputs ────────────────────────────────────────────────────────────
    print("Preparing GPU Gaussian + Sobel inputs for NMS...")
    image_gpu  = cp.asarray(image_cpu,  dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_cpu, dtype=cp.float32)
    _, magnitude_gpu, angle_gpu = canny_frontend_gpu_compute_only(image_gpu, kernel_gpu)
    cp.cuda.Stream.null.synchronize()

    # ── CPU benchmark ─────────────────────────────────────────────────────────
    cpu_time, nms_cpu = benchmark_cpu_nms(magnitude_cpu, angle_cpu, args.runs, args.warmup)

    # ── CuPy benchmark (no tile-size dependence) ──────────────────────────────
    cupy_time, nms_cupy_gpu = benchmark_gpu_nms_compute_only(
        magnitude_gpu, angle_gpu, args.runs, args.warmup
    )
    nms_cupy_cpu = cp.asnumpy(nms_cupy_gpu)
    cupy_vs_cpu_diff = float(np.max(np.abs(nms_cpu - nms_cupy_cpu)))

    # ── cuTile tile-size sweep ────────────────────────────────────────────────
    results = []
    best_result = None

    for tile_size in tile_sizes:
        cutile_time, nms_cutile_gpu = benchmark_cutile_nms(
            magnitude_gpu, angle_gpu, tile_size, args.runs, args.warmup
        )
        nms_cutile_cpu = cp.asnumpy(nms_cutile_gpu)

        diff_vs_cpu   = float(np.max(np.abs(nms_cpu       - nms_cutile_cpu)))
        diff_vs_cupy  = float(np.max(np.abs(nms_cupy_cpu  - nms_cutile_cpu)))

        result = {
            "tile_size":             tile_size,
            "cutile_time_seconds":   cutile_time,
            "cupy_time_seconds":     cupy_time,
            "cpu_time_seconds":      cpu_time,
            "speedup_vs_cupy":       cupy_time  / cutile_time if cutile_time > 0 else float("inf"),
            "speedup_vs_cpu":        cpu_time   / cutile_time if cutile_time > 0 else float("inf"),
            "max_abs_diff_vs_cpu":   diff_vs_cpu,
            "max_abs_diff_vs_cupy":  diff_vs_cupy,
            "output": nms_cutile_cpu,
        }
        results.append(result)

        if best_result is None or cutile_time < best_result["cutile_time_seconds"]:
            best_result = result

    # ── Output ────────────────────────────────────────────────────────────────
    tag = _make_output_tag(image_path, args.kernel_size, args.sigma)

    csv_path      = output_dir / f"27_nms_cutile_tile_sweep_{tag}.csv"
    cpu_img_path  = output_dir / f"15_nms_cpu_edges_{tag}.png"
    cupy_img_path = output_dir / f"16_nms_gpu_edges_{tag}.png"
    best_img_path = output_dir / f"28_nms_cutile_edges_{tag}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tile_size",
            "cutile_time_seconds",
            "cupy_time_seconds",
            "cpu_time_seconds",
            "speedup_vs_cupy",
            "speedup_vs_cpu",
            "max_abs_diff_vs_cpu",
            "max_abs_diff_vs_cupy",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "tile_size":            r["tile_size"],
                "cutile_time_seconds":  f"{r['cutile_time_seconds']:.8f}",
                "cupy_time_seconds":    f"{r['cupy_time_seconds']:.8f}",
                "cpu_time_seconds":     f"{r['cpu_time_seconds']:.8f}",
                "speedup_vs_cupy":      f"{r['speedup_vs_cupy']:.4f}",
                "speedup_vs_cpu":       f"{r['speedup_vs_cpu']:.4f}",
                "max_abs_diff_vs_cpu":  f"{r['max_abs_diff_vs_cpu']:.8f}",
                "max_abs_diff_vs_cupy": f"{r['max_abs_diff_vs_cupy']:.8f}",
            })

    cv2.imwrite(str(cpu_img_path),  normalize_to_uint8(nms_cpu))
    cv2.imwrite(str(cupy_img_path), normalize_to_uint8(nms_cupy_cpu))
    cv2.imwrite(str(best_img_path), normalize_to_uint8(best_result["output"]))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("NMS benchmark results")
    print("─────────────────────────────────────────────────────────────────")
    print(f"CPU   NMS time    : {cpu_time * 1000:8.3f} ms")
    print(f"CuPy  NMS time    : {cupy_time * 1000:8.3f} ms  "
          f"(speedup vs CPU: {cpu_time/cupy_time:.2f}×,  "
          f"max diff vs CPU: {cupy_vs_cpu_diff:.6f})")
    print()
    print(f"{'Tile':>5} | {'cuTile(ms)':>10} | {'vs CuPy':>8} | {'vs CPU':>8} | "
          f"{'diff vs CPU':>12} | {'diff vs CuPy':>13}")
    print("─" * 75)
    for r in results:
        print(
            f"{r['tile_size']:>5} | "
            f"{r['cutile_time_seconds']*1000:>10.3f} | "
            f"{r['speedup_vs_cupy']:>8.2f}× | "
            f"{r['speedup_vs_cpu']:>8.2f}× | "
            f"{r['max_abs_diff_vs_cpu']:>12.6f} | "
            f"{r['max_abs_diff_vs_cupy']:>13.6f}"
        )

    print()
    print(f"Best tile size   : {best_result['tile_size']}")
    print(f"CSV saved to     : {csv_path}")
    print(f"CPU edge image   : {cpu_img_path}")
    print(f"CuPy edge image  : {cupy_img_path}")
    print(f"cuTile edge image: {best_img_path}")


if __name__ == "__main__":
    main()
