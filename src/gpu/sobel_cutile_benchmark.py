import argparse
import csv
import time
from pathlib import Path

import cv2
import cupy as cp
import cuda.tile as ct
import numpy as np

from gaussian_benchmark import load_grayscale_image
from sobel_benchmark import normalize_to_uint8, sobel_cpu


@ct.kernel
def sobel_magnitude_from_neighbors_cutile(
    top_left,
    top_center,
    top_right,
    middle_left,
    middle_right,
    bottom_left,
    bottom_center,
    bottom_right,
    magnitude_output,
    tile_size: ct.Constant[int],
):
    """
    cuTile kernel for Sobel magnitude.

    Each tile processes a 1D chunk of flattened interior pixels.
    Boundary pixels are handled outside the kernel by only flattening the image interior.
    """
    block_id = ct.bid(0)

    tl = ct.load(top_left, index=(block_id,), shape=(tile_size,))
    tc = ct.load(top_center, index=(block_id,), shape=(tile_size,))
    tr = ct.load(top_right, index=(block_id,), shape=(tile_size,))

    ml = ct.load(middle_left, index=(block_id,), shape=(tile_size,))
    mr = ct.load(middle_right, index=(block_id,), shape=(tile_size,))

    bl = ct.load(bottom_left, index=(block_id,), shape=(tile_size,))
    bc = ct.load(bottom_center, index=(block_id,), shape=(tile_size,))
    br = ct.load(bottom_right, index=(block_id,), shape=(tile_size,))

    gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br
    gy = tl + 2.0 * tc + tr - bl - 2.0 * bc - br

    magnitude = ct.sqrt(gx * gx + gy * gy)

    ct.store(magnitude_output, index=(block_id,), tile=magnitude)


def pad_to_multiple_gpu(array_gpu, multiple: int):
    """
    Pad a 1D GPU array so cuTile can process full tiles.

    The extra padded values are ignored after the kernel finishes.
    """
    if multiple <= 0:
        raise ValueError("multiple must be positive.")

    original_size = array_gpu.size
    remainder = original_size % multiple

    if remainder == 0:
        return array_gpu, original_size

    pad_width = multiple - remainder
    padded = cp.pad(array_gpu, (0, pad_width), mode="constant", constant_values=0)

    return padded, original_size


def prepare_sobel_neighbor_arrays(image_gpu, tile_size: int):
    """
    Prepare flattened 3x3 neighbour arrays for the Sobel cuTile kernel.

    For an H x W image, the kernel processes only the interior region with
    shape (H-2, W-2), then the result is placed back into a full H x W image.
    This avoids hardcoding image dimensions.
    """
    if image_gpu.ndim != 2:
        raise ValueError("image_gpu must be a 2D grayscale image.")

    height, width = image_gpu.shape

    if height < 3 or width < 3:
        raise ValueError("image must be at least 3x3 for Sobel.")

    neighbours = [
        cp.ascontiguousarray(image_gpu[:-2, :-2].ravel()),
        cp.ascontiguousarray(image_gpu[:-2, 1:-1].ravel()),
        cp.ascontiguousarray(image_gpu[:-2, 2:].ravel()),
        cp.ascontiguousarray(image_gpu[1:-1, :-2].ravel()),
        cp.ascontiguousarray(image_gpu[1:-1, 2:].ravel()),
        cp.ascontiguousarray(image_gpu[2:, :-2].ravel()),
        cp.ascontiguousarray(image_gpu[2:, 1:-1].ravel()),
        cp.ascontiguousarray(image_gpu[2:, 2:].ravel()),
    ]

    padded_neighbours = []
    original_size = None

    for neighbour in neighbours:
        padded, size = pad_to_multiple_gpu(neighbour, tile_size)
        padded_neighbours.append(padded)

        if original_size is None:
            original_size = size
        elif original_size != size:
            raise ValueError("all neighbour arrays must have the same size.")

    return padded_neighbours, original_size, height, width


def launch_sobel_cutile(image_gpu, tile_size: int):
    """
    Run the cuTile Sobel magnitude kernel and return a full-size magnitude image.
    """
    padded_neighbours, original_size, height, width = prepare_sobel_neighbor_arrays(
        image_gpu,
        tile_size,
    )

    padded_size = padded_neighbours[0].size
    output_padded = cp.zeros(padded_size, dtype=cp.float32)

    grid = ((padded_size + tile_size - 1) // tile_size, 1, 1)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        sobel_magnitude_from_neighbors_cutile,
        (
            padded_neighbours[0],
            padded_neighbours[1],
            padded_neighbours[2],
            padded_neighbours[3],
            padded_neighbours[4],
            padded_neighbours[5],
            padded_neighbours[6],
            padded_neighbours[7],
            output_padded,
            tile_size,
        ),
    )

    interior = output_padded[:original_size].reshape((height - 2, width - 2))

    full_output = cp.zeros((height, width), dtype=cp.float32)
    full_output[1:-1, 1:-1] = interior

    return full_output


@ct.kernel
def sobel_fused_from_neighbors_cutile(
    top_left,
    top_center,
    top_right,
    middle_left,
    middle_right,
    bottom_left,
    bottom_center,
    bottom_right,
    magnitude_output,
    gx_output,
    gy_output,
    tile_size: ct.Constant[int],
):
    """
    Fused cuTile Sobel kernel: magnitude + gx + gy in one pass.

    Reads each of the 8 neighbour arrays exactly once and writes three outputs.
    This eliminates the redundant second neighbour read that the original pipeline
    required to compute gradient angle via CuPy after the magnitude-only kernel.
    """
    b = ct.bid(0)
    s = (tile_size,)

    tl = ct.load(top_left,     index=(b,), shape=s)
    tc = ct.load(top_center,   index=(b,), shape=s)
    tr = ct.load(top_right,    index=(b,), shape=s)
    ml = ct.load(middle_left,  index=(b,), shape=s)
    mr = ct.load(middle_right, index=(b,), shape=s)
    bl = ct.load(bottom_left,  index=(b,), shape=s)
    bc = ct.load(bottom_center,index=(b,), shape=s)
    br = ct.load(bottom_right, index=(b,), shape=s)

    gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br
    gy =  tl + 2.0 * tc + tr - bl - 2.0 * bc - br

    ct.store(magnitude_output, index=(b,), tile=ct.sqrt(gx * gx + gy * gy))
    ct.store(gx_output,        index=(b,), tile=gx)
    ct.store(gy_output,        index=(b,), tile=gy)


def launch_sobel_fused_cutile(image_gpu, tile_size: int):
    """
    Run the fused cuTile Sobel kernel and return (magnitude, gx, gy).

    All three outputs share the same neighbour arrays, so each of the 8 neighbour
    arrays is read exactly once — half the memory bandwidth of the original two-pass
    approach (magnitude-only cuTile kernel + separate CuPy pass for angle).
    """
    padded_neighbours, original_size, height, width = prepare_sobel_neighbor_arrays(
        image_gpu, tile_size
    )

    padded_size = padded_neighbours[0].size
    mag_padded = cp.zeros(padded_size, dtype=cp.float32)
    gx_padded  = cp.zeros(padded_size, dtype=cp.float32)
    gy_padded  = cp.zeros(padded_size, dtype=cp.float32)

    grid = ((padded_size + tile_size - 1) // tile_size, 1, 1)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        sobel_fused_from_neighbors_cutile,
        (
            padded_neighbours[0],
            padded_neighbours[1],
            padded_neighbours[2],
            padded_neighbours[3],
            padded_neighbours[4],
            padded_neighbours[5],
            padded_neighbours[6],
            padded_neighbours[7],
            mag_padded,
            gx_padded,
            gy_padded,
            tile_size,
        ),
    )

    interior_shape = (height - 2, width - 2)

    def _to_full(padded):
        interior = padded[:original_size].reshape(interior_shape)
        full = cp.zeros((height, width), dtype=cp.float32)
        full[1:-1, 1:-1] = interior
        return full

    return _to_full(mag_padded), _to_full(gx_padded), _to_full(gy_padded)


def sobel_magnitude_cupy_compute_only(image_gpu):
    """
    Compute Sobel magnitude only using CuPy.

    This is used as a fair comparison against the cuTile Sobel magnitude kernel.
    It does not compute gradient angle.
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

    return magnitude


def benchmark_cutile_kernel_only(
    image_gpu,
    tile_size: int,
    runs: int,
    warmup: int,
):
    """
    Benchmark cuTile Sobel magnitude.

    This includes neighbour preparation and the cuTile kernel launch.
    It does not include disk I/O or loading the image from CPU.
    """
    for _ in range(warmup):
        output_gpu = launch_sobel_cutile(image_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        output_gpu = launch_sobel_cutile(image_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, output_gpu


def benchmark_cupy_sobel_magnitude(image_gpu, runs: int, warmup: int):
    """
    Benchmark CuPy Sobel magnitude only.

    This is a fair comparison against the cuTile Sobel magnitude kernel because
    both implementations compute the same output: gradient magnitude only.
    """
    for _ in range(warmup):
        magnitude_gpu = sobel_magnitude_cupy_compute_only(image_gpu)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        magnitude_gpu = sobel_magnitude_cupy_compute_only(image_gpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / runs

    return average_time, magnitude_gpu


def _two_pass_sobel(image_gpu, tile_size):
    """
    Non-fused two-pass approach (old pipeline):
      Pass 1 — cuTile magnitude-only kernel
      Pass 2 — CuPy re-reads 8 neighbours to compute gx, gy, arctan2
    """
    mag = launch_sobel_cutile(image_gpu, tile_size)
    b = image_gpu
    gx = cp.zeros_like(b)
    gy = cp.zeros_like(b)
    gx[1:-1, 1:-1] = (-b[:-2,:-2] + b[:-2,2:] - 2*b[1:-1,:-2]
                      + 2*b[1:-1,2:] - b[2:,:-2] + b[2:,2:])
    gy[1:-1, 1:-1] = (b[:-2,:-2] + 2*b[:-2,1:-1] + b[:-2,2:]
                      - b[2:,:-2] - 2*b[2:,1:-1] - b[2:,2:])
    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0
    return mag, angle


def _fused_sobel(image_gpu, tile_size):
    """
    Fused single-pass approach (new pipeline):
      Pass 1 — cuTile fused kernel writes mag + gx + gy
      arctan2 on pre-computed gradients (no second neighbour read)
    """
    mag, gx, gy = launch_sobel_fused_cutile(image_gpu, tile_size)
    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0
    return mag, angle


def benchmark_two_pass_sobel(image_gpu, tile_size, runs, warmup):
    """Benchmark non-fused two-pass Sobel (magnitude cuTile + angle CuPy)."""
    for _ in range(warmup):
        _two_pass_sobel(image_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        mag, angle = _two_pass_sobel(image_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / runs, mag, angle


def benchmark_fused_sobel(image_gpu, tile_size, runs, warmup):
    """Benchmark fused single-pass Sobel (cuTile mag+gx+gy + arctan2)."""
    for _ in range(warmup):
        _fused_sobel(image_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        mag, angle = _fused_sobel(image_gpu, tile_size)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / runs, mag, angle


def benchmark_cupy_full_sobel(image_gpu, runs, warmup):
    """Benchmark full CuPy Sobel: magnitude + angle in one vectorised pass."""
    def _run():
        b = image_gpu
        gx = cp.zeros_like(b)
        gy = cp.zeros_like(b)
        gx[1:-1, 1:-1] = (-b[:-2,:-2] + b[:-2,2:] - 2*b[1:-1,:-2]
                          + 2*b[1:-1,2:] - b[2:,:-2] + b[2:,2:])
        gy[1:-1, 1:-1] = (b[:-2,:-2] + 2*b[:-2,1:-1] + b[:-2,2:]
                          - b[2:,:-2] - 2*b[2:,1:-1] - b[2:,2:])
        mag = cp.sqrt(gx * gx + gy * gy)
        angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0
        return mag, angle

    for _ in range(warmup):
        _run()

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    for _ in range(runs):
        mag, angle = _run()

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / runs, mag, angle


def make_output_tag(image_path: Path) -> str:
    return image_path.stem


def parse_tile_sizes(tile_sizes_text: str):
    tile_sizes = []

    for item in tile_sizes_text.split(","):
        value = int(item.strip())

        if value <= 0:
            raise ValueError("tile sizes must be positive integers.")

        tile_sizes.append(value)

    return tile_sizes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Sobel magnitude implemented with cuTile."
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/test.jpg"),
        help="Input image path.",
    )

    parser.add_argument(
        "--tile-sizes",
        type=str,
        default="64,128,256,512",
        help="Comma-separated tile sizes to test.",
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
    image_gpu = cp.asarray(image_cpu, dtype=cp.float32)

    print(f"Input image  : {image_path}")
    print(f"Image shape  : {image_cpu.shape}")
    print(f"Tile sizes   : {tile_sizes}")
    print(f"Runs/warmup  : {args.runs} / {args.warmup}")

    cpu_magnitude, _ = sobel_cpu(image_cpu)
    output_tag = make_output_tag(image_path)

    # ── Part 1: magnitude-only sweep (original benchmark) ────────────────────
    cupy_mag_time, cupy_mag_gpu = benchmark_cupy_sobel_magnitude(
        image_gpu=image_gpu, runs=args.runs, warmup=args.warmup,
    )
    cupy_mag_cpu = cp.asnumpy(cupy_mag_gpu)
    cupy_mag_diff = float(np.max(np.abs(cpu_magnitude - cupy_mag_cpu)))

    results_mag = []
    best_mag = None

    for tile_size in tile_sizes:
        t, mag_gpu = benchmark_cutile_kernel_only(
            image_gpu=image_gpu, tile_size=tile_size,
            runs=args.runs, warmup=args.warmup,
        )
        mag_cpu = cp.asnumpy(mag_gpu)
        diff = float(np.max(np.abs(cpu_magnitude - mag_cpu)))
        r = {
            "tile_size": tile_size,
            "cutile_time_seconds": t,
            "cupy_time_seconds": cupy_mag_time,
            "speedup_vs_cupy": cupy_mag_time / t if t > 0 else float("inf"),
            "max_abs_diff_vs_cpu": diff,
            "output": mag_cpu,
        }
        results_mag.append(r)
        if best_mag is None or t < best_mag["cutile_time_seconds"]:
            best_mag = r

    csv_mag = output_dir / f"17_sobel_cutile_tile_sweep_{output_tag}.csv"
    with csv_mag.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tile_size", "cutile_time_seconds", "cupy_time_seconds",
            "speedup_vs_cupy", "max_abs_diff_vs_cpu"])
        w.writeheader()
        for r in results_mag:
            w.writerow({
                "tile_size":            r["tile_size"],
                "cutile_time_seconds":  f"{r['cutile_time_seconds']:.8f}",
                "cupy_time_seconds":    f"{r['cupy_time_seconds']:.8f}",
                "speedup_vs_cupy":      f"{r['speedup_vs_cupy']:.4f}",
                "max_abs_diff_vs_cpu":  f"{r['max_abs_diff_vs_cpu']:.8f}",
            })

    cv2.imwrite(
        str(output_dir / f"18_sobel_cutile_magnitude_{output_tag}.png"),
        normalize_to_uint8(best_mag["output"]),
    )

    # ── Part 2: fused vs non-fused sweep (new benchmark) ─────────────────────
    # CuPy full-Sobel baseline (magnitude + angle, single vectorised pass)
    cupy_full_time, cupy_full_mag, cupy_full_ang = benchmark_cupy_full_sobel(
        image_gpu=image_gpu, runs=args.runs, warmup=args.warmup,
    )

    results_fused = []
    best_fused = None

    for tile_size in tile_sizes:
        two_t, two_mag, two_ang = benchmark_two_pass_sobel(
            image_gpu=image_gpu, tile_size=tile_size,
            runs=args.runs, warmup=args.warmup,
        )
        fus_t, fus_mag, fus_ang = benchmark_fused_sobel(
            image_gpu=image_gpu, tile_size=tile_size,
            runs=args.runs, warmup=args.warmup,
        )

        fus_mag_cpu = cp.asnumpy(fus_mag)
        two_mag_cpu = cp.asnumpy(two_mag)
        diff_mag = float(np.max(np.abs(cpu_magnitude - fus_mag_cpu)))

        r = {
            "tile_size":                   tile_size,
            "two_pass_time_seconds":        two_t,
            "fused_time_seconds":           fus_t,
            "cupy_full_time_seconds":       cupy_full_time,
            "speedup_fused_vs_two_pass":    two_t / fus_t if fus_t > 0 else float("inf"),
            "speedup_fused_vs_cupy_full":   cupy_full_time / fus_t if fus_t > 0 else float("inf"),
            "max_abs_diff_fused_vs_cpu":    diff_mag,
            "fused_output": fus_mag_cpu,
        }
        results_fused.append(r)
        if best_fused is None or fus_t < best_fused["fused_time_seconds"]:
            best_fused = r

    csv_fused = output_dir / f"29_sobel_fused_tile_sweep_{output_tag}.csv"
    with csv_fused.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tile_size",
            "two_pass_time_seconds",
            "fused_time_seconds",
            "cupy_full_time_seconds",
            "speedup_fused_vs_two_pass",
            "speedup_fused_vs_cupy_full",
            "max_abs_diff_fused_vs_cpu",
        ])
        w.writeheader()
        for r in results_fused:
            w.writerow({
                "tile_size":                  r["tile_size"],
                "two_pass_time_seconds":       f"{r['two_pass_time_seconds']:.8f}",
                "fused_time_seconds":          f"{r['fused_time_seconds']:.8f}",
                "cupy_full_time_seconds":      f"{r['cupy_full_time_seconds']:.8f}",
                "speedup_fused_vs_two_pass":   f"{r['speedup_fused_vs_two_pass']:.4f}",
                "speedup_fused_vs_cupy_full":  f"{r['speedup_fused_vs_cupy_full']:.4f}",
                "max_abs_diff_fused_vs_cpu":   f"{r['max_abs_diff_fused_vs_cpu']:.8f}",
            })

    cv2.imwrite(
        str(output_dir / f"30_sobel_fused_magnitude_{output_tag}.png"),
        normalize_to_uint8(best_fused["fused_output"]),
    )

    # ── Print summaries ───────────────────────────────────────────────────────
    print()
    print("Part 1 — Sobel magnitude-only: cuTile vs CuPy")
    print("─────────────────────────────────────────────")
    print(f"CuPy magnitude-only: {cupy_mag_time*1000:8.3f} ms  "
          f"(max diff vs CPU: {cupy_mag_diff:.6f})")
    print()
    print(f"{'Tile':>5} | {'cuTile(ms)':>10} | {'vs CuPy':>8} | {'max diff':>10}")
    print("─" * 45)
    for r in results_mag:
        print(f"{r['tile_size']:>5} | {r['cutile_time_seconds']*1000:>10.3f} | "
              f"{r['speedup_vs_cupy']:>8.3f}× | {r['max_abs_diff_vs_cpu']:>10.6f}")
    print(f"\nCSV → {csv_mag}")

    print()
    print("Part 2 — Fused Sobel (mag+gx+gy): fused vs two-pass vs CuPy")
    print("────────────────────────────────────────────────────────────")
    print(f"CuPy full (mag+angle): {cupy_full_time*1000:8.3f} ms")
    print()
    print(f"{'Tile':>5} | {'2-pass(ms)':>10} | {'Fused(ms)':>9} | "
          f"{'Fused/2-pass':>12} | {'Fused/CuPy':>10} | {'max diff':>10}")
    print("─" * 70)
    for r in results_fused:
        print(
            f"{r['tile_size']:>5} | "
            f"{r['two_pass_time_seconds']*1000:>10.3f} | "
            f"{r['fused_time_seconds']*1000:>9.3f} | "
            f"{r['speedup_fused_vs_two_pass']:>12.3f}× | "
            f"{r['speedup_fused_vs_cupy_full']:>10.3f}× | "
            f"{r['max_abs_diff_fused_vs_cpu']:>10.6f}"
        )
    print(f"\nCSV → {csv_fused}")


if __name__ == "__main__":
    main()