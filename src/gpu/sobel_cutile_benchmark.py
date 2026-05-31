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

    print(f"Input image: {image_path}")
    print(f"Image shape: {image_cpu.shape}")
    print(f"Tile sizes: {tile_sizes}")
    print(f"Runs: {args.runs}")
    print(f"Warm-up runs: {args.warmup}")

    cpu_magnitude, _ = sobel_cpu(image_cpu)

    cupy_time, cupy_magnitude_gpu = benchmark_cupy_sobel_magnitude(
        image_gpu=image_gpu,
        runs=args.runs,
        warmup=args.warmup,
    )

    cupy_magnitude_cpu = cp.asnumpy(cupy_magnitude_gpu)
    cupy_max_abs_diff = float(np.max(np.abs(cpu_magnitude - cupy_magnitude_cpu)))

    results = []
    best_result = None

    for tile_size in tile_sizes:
        cutile_time, cutile_magnitude_gpu = benchmark_cutile_kernel_only(
            image_gpu=image_gpu,
            tile_size=tile_size,
            runs=args.runs,
            warmup=args.warmup,
        )

        cutile_magnitude_cpu = cp.asnumpy(cutile_magnitude_gpu)
        max_abs_diff = float(np.max(np.abs(cpu_magnitude - cutile_magnitude_cpu)))

        result = {
            "tile_size": tile_size,
            "cutile_time_seconds": cutile_time,
            "cupy_time_seconds": cupy_time,
            "speedup_vs_cupy": cupy_time / cutile_time
            if cutile_time > 0
            else float("inf"),
            "max_abs_diff_vs_cpu": max_abs_diff,
            "output": cutile_magnitude_cpu,
        }

        results.append(result)

        if best_result is None or cutile_time < best_result["cutile_time_seconds"]:
            best_result = result

    output_tag = make_output_tag(image_path)

    csv_output_path = output_dir / f"17_sobel_cutile_tile_sweep_{output_tag}.csv"
    best_image_output_path = output_dir / f"18_sobel_cutile_magnitude_{output_tag}.png"

    with csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "tile_size",
                "cutile_time_seconds",
                "cupy_time_seconds",
                "speedup_vs_cupy",
                "max_abs_diff_vs_cpu",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "tile_size": result["tile_size"],
                    "cutile_time_seconds": f"{result['cutile_time_seconds']:.8f}",
                    "cupy_time_seconds": f"{result['cupy_time_seconds']:.8f}",
                    "speedup_vs_cupy": f"{result['speedup_vs_cupy']:.4f}",
                    "max_abs_diff_vs_cpu": f"{result['max_abs_diff_vs_cpu']:.8f}",
                }
            )

    cv2.imwrite(
        str(best_image_output_path),
        normalize_to_uint8(best_result["output"]),
    )

    print()
    print("Sobel cuTile benchmark results")
    print("------------------------------")
    print(f"CuPy Sobel magnitude-only time: {cupy_time:.6f} seconds")
    print(f"CuPy max absolute difference vs CPU: {cupy_max_abs_diff:.6f}")

    for result in results:
        print(
            f"tile_size={result['tile_size']:>4} | "
            f"cuTile time={result['cutile_time_seconds']:.6f}s | "
            f"speedup vs CuPy={result['speedup_vs_cupy']:.2f}x | "
            f"max diff vs CPU={result['max_abs_diff_vs_cpu']:.6f}"
        )

    print()
    print(f"Best tile size: {best_result['tile_size']}")
    print(f"CSV results saved to: {csv_output_path}")
    print(f"Best cuTile magnitude image saved to: {best_image_output_path}")

    print()
    print(
        "Note: this is the first true cuTile Sobel implementation. "
        "It computes Sobel magnitude for image interior pixels and leaves image "
        "boundaries as zero, matching the CPU reference behaviour. The CuPy "
        "comparison also computes magnitude only, so the comparison is fair."
    )


if __name__ == "__main__":
    main()