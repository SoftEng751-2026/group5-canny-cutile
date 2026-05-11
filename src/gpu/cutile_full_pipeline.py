"""
Canny edge detection maximising cuTile coverage.

Pipeline:
  Gaussian blur  -- cuTile (k=5) or CuPy fallback
  Sobel          -- cuTile        or CuPy fallback
  NMS            -- CuPy
  Threshold      -- NumPy
  Hysteresis     -- CPU (OpenCV connected components)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gaussian_benchmark import load_grayscale_image, make_gaussian_kernel
from sobel_cutile_benchmark import launch_sobel_cutile, sobel_magnitude_cupy_compute_only

try:
    import cuda.tile as ct
    _HAS_CUTILE = True
except Exception:
    _HAS_CUTILE = False


# ── cuTile Gaussian kernel (k=5) and launch helper ────────────────────────────
# The kernel computes a weighted sum of five shifted 1D views (one per
# kernel tap), identical for both the horizontal and vertical blur passes.

if _HAS_CUTILE:
    @ct.kernel
    def _gauss_k5(n0, n1, n2, n3, n4, out,
                  w0: ct.Constant[float], w1: ct.Constant[float],
                  w2: ct.Constant[float], w3: ct.Constant[float],
                  w4: ct.Constant[float],
                  tile_size: ct.Constant[int]):
        b = ct.bid(0)
        s = (tile_size,)
        v = (w0 * ct.load(n0, index=(b,), shape=s)
           + w1 * ct.load(n1, index=(b,), shape=s)
           + w2 * ct.load(n2, index=(b,), shape=s)
           + w3 * ct.load(n3, index=(b,), shape=s)
           + w4 * ct.load(n4, index=(b,), shape=s))
        ct.store(out, index=(b,), tile=v)

    def _launch_gauss_k5(views, weights, tile_size):
        """Pad, launch the cuTile kernel, and return unpadded 1D output."""
        def pad(v):
            flat = cp.ascontiguousarray(v.ravel())
            rem = flat.size % tile_size
            if rem:
                flat = cp.pad(flat, (0, tile_size - rem))
            return flat

        pads = [pad(v) for v in views]
        orig = views[0].size
        out = cp.zeros(pads[0].size, dtype=cp.float32)
        grid = (pads[0].size // tile_size, 1, 1)
        ct.launch(cp.cuda.get_current_stream(), grid, _gauss_k5,
                  (*pads, out, *[float(w) for w in weights], tile_size))
        return out[:orig]


# ── Stage functions ───────────────────────────────────────────────────────────

def _gaussian_cupy(img, kernel_gpu):
    r, H, W = kernel_gpu.size // 2, img.shape[0], img.shape[1]
    p = cp.pad(img, ((0, 0), (r, r)), mode='edge')
    t = sum(kernel_gpu[i] * p[:, i:i + W] for i in range(kernel_gpu.size))
    p = cp.pad(t, ((r, r), (0, 0)), mode='edge')
    return sum(kernel_gpu[i] * p[i:i + H, :] for i in range(kernel_gpu.size))


def gaussian_blur(img_gpu, kernel_cpu, tile_size):
    k = kernel_cpu.size
    r, H, W = k // 2, img_gpu.shape[0], img_gpu.shape[1]

    if _HAS_CUTILE and k == 5:
        ph = cp.pad(img_gpu, ((0, 0), (r, r)), mode='edge')
        temp = _launch_gauss_k5([ph[:, i:i + W] for i in range(k)],
                                 kernel_cpu, tile_size).reshape(H, W)
        pv = cp.pad(temp, ((r, r), (0, 0)), mode='edge')
        return _launch_gauss_k5([pv[i:i + H, :] for i in range(k)],
                                 kernel_cpu, tile_size).reshape(H, W)

    return _gaussian_cupy(img_gpu, cp.asarray(kernel_cpu, dtype=cp.float32))


def sobel(blurred_gpu, tile_size):
    try:
        mag = launch_sobel_cutile(blurred_gpu, tile_size)
    except Exception:
        mag = sobel_magnitude_cupy_compute_only(blurred_gpu)

    # Gradient angle is required by NMS; computed via CuPy.
    gx = cp.zeros_like(blurred_gpu)
    gy = cp.zeros_like(blurred_gpu)
    b = blurred_gpu
    gx[1:-1, 1:-1] = -b[:-2, :-2] + b[:-2, 2:] - 2*b[1:-1, :-2] + 2*b[1:-1, 2:] - b[2:, :-2] + b[2:, 2:]
    gy[1:-1, 1:-1] =  b[:-2, :-2] + 2*b[:-2, 1:-1] + b[:-2, 2:] - b[2:, :-2] - 2*b[2:, 1:-1] - b[2:, 2:]
    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0
    return mag, angle


def nms(mag_gpu, angle_gpu):
    d, out = angle_gpu % 180.0, cp.zeros_like(mag_gpu)
    c, a = mag_gpu[1:-1, 1:-1], d[1:-1, 1:-1]
    keep = (
        ((a < 22.5) | (a >= 157.5)) & (c >= mag_gpu[1:-1, :-2]) & (c >= mag_gpu[1:-1, 2:]) |
        ((a >= 22.5) & (a < 67.5))  & (c >= mag_gpu[:-2, 2:])   & (c >= mag_gpu[2:, :-2])  |
        ((a >= 67.5) & (a < 112.5)) & (c >= mag_gpu[:-2, 1:-1]) & (c >= mag_gpu[2:, 1:-1]) |
        ((a >= 112.5) & (a < 157.5))& (c >= mag_gpu[:-2, :-2])  & (c >= mag_gpu[2:, 2:])
    )
    out[1:-1, 1:-1] = cp.where(keep, c, 0.0)
    return out


def threshold_and_hysteresis(nms_cpu, high_pct, low_ratio):
    pos = nms_cpu[nms_cpu > 0]
    if pos.size == 0:
        return np.zeros_like(nms_cpu, bool), 0.0, 0.0
    high = float(np.percentile(pos, high_pct))
    low = float(low_ratio * high)
    strong = nms_cpu >= high
    weak = (nms_cpu >= low) & ~strong
    n, labels = cv2.connectedComponents((strong | weak).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels], low, high


# ── End-to-end run ────────────────────────────────────────────────────────────

def run(image_cpu, kernel_cpu, tile_size=256, high_pct=90.0, low_ratio=0.5):
    t = {}
    img_gpu = cp.asarray(image_cpu, dtype=cp.float32)

    t0 = time.perf_counter()
    blurred = gaussian_blur(img_gpu, kernel_cpu, tile_size)
    cp.cuda.Stream.null.synchronize()
    t['gaussian'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mag, angle = sobel(blurred, tile_size)
    cp.cuda.Stream.null.synchronize()
    t['sobel'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nms_gpu = nms(mag, angle)
    cp.cuda.Stream.null.synchronize()
    t['nms'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nms_cpu = cp.asnumpy(nms_gpu)
    t['transfer'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges, low, high = threshold_and_hysteresis(nms_cpu, high_pct, low_ratio)
    t['hysteresis'] = time.perf_counter() - t0

    t['total'] = sum(t.values())
    return edges, low, high, t


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Canny edge detection with cuTile')
    p.add_argument('--image', type=Path, default=Path('data/test.jpg'))
    p.add_argument('--kernel-size', type=int, default=5)
    p.add_argument('--sigma', type=float, default=1.4)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--high-percentile', type=float, default=90.0)
    p.add_argument('--low-ratio', type=float, default=0.5)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    image_cpu = load_grayscale_image(args.image.resolve())
    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    gauss_impl = 'cuTile' if _HAS_CUTILE and args.kernel_size == 5 else 'CuPy'
    print(f'cuTile available : {_HAS_CUTILE}')
    print(f'Gaussian (k={args.kernel_size}) : {gauss_impl}')
    print(f'Sobel            : cuTile (CuPy fallback on incompatible GPU)')
    print(f'NMS              : CuPy')
    print(f'Hysteresis       : CPU (connected components)')

    # Warm-up run to exclude JIT/CUDA init cost from reported timings.
    run(image_cpu, kernel_cpu, args.tile_size, args.high_percentile, args.low_ratio)

    records = [run(image_cpu, kernel_cpu, args.tile_size,
                   args.high_percentile, args.low_ratio)
               for _ in range(args.runs)]

    edges, low, high, _ = records[-1]
    avg = {k: float(np.mean([r[3][k] for r in records])) for k in records[0][3]}

    print(f'\nTiming (mean over {args.runs} run(s)):')
    for k, v in avg.items():
        print(f'  {k:<12} {v * 1000:8.3f} ms')
    print(f'\n  low={low:.2f}  high={high:.2f}  edge_pixels={edges.sum()}')

    edges_u8 = edges.astype(np.uint8) * 255
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), edges_u8)
        print(f'  Saved: {args.output}')
    if not args.no_display:
        cv2.imshow('Canny (cuTile)', edges_u8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()