"""
BSDS500 Canny reliability test.

Downloads images from the Berkeley Segmentation Dataset 500 and compares:
  - Pure-NumPy Canny (CPU reference)
  - Full GPU / cuTile Canny   (if CuPy is available)
  - OpenCV cv2.Canny          (external reference)

Displays visual results and prints per-image and aggregate reliability summaries.
"""

import sys
from collections import deque
from pathlib import Path

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import csv

# ── Optional GPU pipeline ─────────────────────────────────────────────────────
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent / "src" / "gpu"))
    from cutile_full_pipeline import run as _gpu_run
    from gaussian_benchmark import make_gaussian_kernel as _make_kernel_gpu
    HAS_GPU = True
except Exception as _e:
    HAS_GPU = False
    print(f"[info] GPU pipeline not available ({_e}); GPU column will be skipped.",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Pure-NumPy Canny pipeline (self-contained, no GPU / cuTile dependencies)
# ---------------------------------------------------------------------------

def _make_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _gaussian_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.size // 2
    h, w = image.shape
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.zeros((h, w), dtype=np.float32)
    for i, weight in enumerate(kernel):
        tmp += weight * padded[:, i : i + w]
    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros((h, w), dtype=np.float32)
    for i, weight in enumerate(kernel):
        out += weight * padded[i : i + h, :]
    return out


def _sobel(blurred: np.ndarray):
    gx = np.zeros_like(blurred)
    gy = np.zeros_like(blurred)
    b = blurred
    gx[1:-1, 1:-1] = (
        -b[:-2, :-2] + b[:-2, 2:]
        - 2 * b[1:-1, :-2] + 2 * b[1:-1, 2:]
        - b[2:, :-2] + b[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        b[:-2, :-2] + 2 * b[:-2, 1:-1] + b[:-2, 2:]
        - b[2:, :-2] - 2 * b[2:, 1:-1] - b[2:, 2:]
    )
    magnitude = np.hypot(gx, gy).astype(np.float32)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0
    return magnitude, angle


def _non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    out = np.zeros_like(magnitude)
    c = magnitude[1:-1, 1:-1]
    d = angle[1:-1, 1:-1]
    keep = (
        ((d < 22.5) | (d >= 157.5))
        & (c >= magnitude[1:-1, :-2]) & (c >= magnitude[1:-1, 2:])
    ) | (
        ((d >= 22.5) & (d < 67.5))
        & (c >= magnitude[:-2, 2:]) & (c >= magnitude[2:, :-2])
    ) | (
        ((d >= 67.5) & (d < 112.5))
        & (c >= magnitude[:-2, 1:-1]) & (c >= magnitude[2:, 1:-1])
    ) | (
        ((d >= 112.5) & (d < 157.5))
        & (c >= magnitude[:-2, :-2]) & (c >= magnitude[2:, 2:])
    )
    out[1:-1, 1:-1] = np.where(keep, c, 0.0)
    return out


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    h, w = strong.shape
    edges = strong.copy()
    queue = deque(map(tuple, np.argwhere(strong)))
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and weak[nr, nc] and not edges[nr, nc]:
                edges[nr, nc] = True
                queue.append((nr, nc))
    return edges


def numpy_canny(
    gray_uint8: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.4,
    high_percentile: float = 90.0,
    low_ratio: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """
    Full Canny on a uint8 grayscale image.
    Returns (edges_bool, low_threshold, high_threshold).
    """
    img = gray_uint8.astype(np.float32)
    kernel = _make_gaussian_kernel(kernel_size, sigma)
    blurred = _gaussian_blur(img, kernel)
    magnitude, angle = _sobel(blurred)
    nms = _non_max_suppression(magnitude, angle)

    positive = nms[nms > 0.0]
    if positive.size == 0:
        return np.zeros_like(gray_uint8, dtype=bool), 0.0, 0.0

    high = float(np.percentile(positive, high_percentile))
    low = float(low_ratio * high)

    strong = nms >= high
    weak = (nms >= low) & ~strong
    edges = _hysteresis(strong, weak)
    return edges, low, high


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _download_bsds500() -> Path:
    print("Downloading BSDS500 dataset via kagglehub (cached if already downloaded)...")
    path = kagglehub.dataset_download("balraj98/berkeley-segmentation-dataset-500-bsds500")
    print(f"Dataset root: {path}")
    return Path(path)


def _find_test_images(root: Path, max_images: int = None) -> list[Path]:
    # Try known sub-paths for the standard BSDS500 Kaggle layout
    candidates = [
        "BSR/BSDS500/data/images/test",
        "BSDS500/data/images/test",
        "data/images/test",
        "images/test",
        "test",
    ]
    for sub in candidates:
        folder = root / sub
        if folder.is_dir():
            imgs = sorted(folder.glob("*.jpg"))
            if max_images:
                imgs = imgs[:max_images]
            if imgs:
                print(f"Images found at: {folder}")
                return imgs
    # Fallback: any JPEG in the tree
    imgs = sorted(root.rglob("*.jpg"))
    if max_images:
        imgs = imgs[:max_images]
    if imgs:
        print("Images found via recursive search.")
    return imgs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(np_edges: np.ndarray, oc_edges: np.ndarray) -> dict:
    np_bin = np_edges.astype(bool)
    oc_bin = (oc_edges > 0)

    agreement = float(np.mean(np_bin == oc_bin))
    tp = int(np.sum(np_bin & oc_bin))
    fp = int(np.sum(np_bin & ~oc_bin))
    fn = int(np.sum(~np_bin & oc_bin))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    diff_map = np.zeros((*np_bin.shape, 3), dtype=np.uint8)
    diff_map[np_bin & oc_bin] = [220, 220, 220]      # grey  - both agree
    diff_map[np_bin & ~oc_bin] = [220, 60, 60]        # red   - NumPy only
    diff_map[~np_bin & oc_bin] = [60, 60, 220]        # blue  - OpenCV only

    return {
        "agreement": agreement,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "numpy_edge_pct": float(np.mean(np_bin)) * 100,
        "opencv_edge_pct": float(np.mean(oc_bin)) * 100,
        "diff_map": diff_map,
    }


# ---------------------------------------------------------------------------
# GPU Canny wrapper
# ---------------------------------------------------------------------------

def gpu_canny(
    gray_uint8: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.4,
    high_percentile: float = 90.0,
    low_ratio: float = 0.5,
):
    """Run the full GPU/cuTile Canny pipeline. Returns (edges_bool, low, high)."""
    kernel_cpu = _make_kernel_gpu(kernel_size, sigma)
    img_f32 = gray_uint8.astype("float32")
    edges, low, high, _ = _gpu_run(
        image_cpu=img_f32,
        kernel_cpu=kernel_cpu,
        tile_size=256,
        high_pct=high_percentile,
        low_ratio=low_ratio,
        gpu_hysteresis=True,
    )
    return edges.astype(bool), float(low), float(high)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _show_grid(records: list[dict]) -> None:
    n = len(records)
    has_gpu = any("gpu_edges" in r for r in records)
    n_cols = 5 if has_gpu else 4

    fig, axes = plt.subplots(n, n_cols, figsize=(4.2 * n_cols, 4.2 * n), squeeze=False)

    col_titles = ["Original (grayscale)", "NumPy Canny"]
    if has_gpu:
        col_titles.append("GPU/cuTile Canny")
    col_titles += [
        "OpenCV Canny",
        "Diff NumPy vs GPU\n(red=NumPy only, blue=GPU only, grey=both)"
        if has_gpu else
        "Diff NumPy vs OpenCV\n(red=NumPy only, blue=OpenCV only, grey=both)",
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for row, rec in enumerate(records):
        m   = rec["metrics_vs_opencv"]
        mg  = rec.get("metrics_vs_gpu")

        col = 0
        axes[row, col].imshow(rec["gray"], cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_ylabel(rec["name"], fontsize=8, rotation=0,
                                  labelpad=90, va="center")
        col += 1

        axes[row, col].imshow(rec["np_edges"], cmap="gray")
        axes[row, col].set_xlabel(
            f"edges: {m['numpy_edge_pct']:.2f}%  (low={rec['low']:.1f}, high={rec['high']:.1f})",
            fontsize=8)
        col += 1

        if has_gpu and "gpu_edges" in rec:
            axes[row, col].imshow(rec["gpu_edges"], cmap="gray")
            axes[row, col].set_xlabel(
                f"edges: {float(np.mean(rec['gpu_edges']))*100:.2f}%", fontsize=8)
            col += 1

        axes[row, col].imshow(rec["oc_edges"], cmap="gray")
        axes[row, col].set_xlabel(
            f"edges: {m['opencv_edge_pct']:.2f}%", fontsize=8)
        col += 1

        # diff map: NumPy vs GPU if available, else NumPy vs OpenCV
        if mg is not None:
            axes[row, col].imshow(mg["diff_map"])
            axes[row, col].set_xlabel(
                f"NumPy↔GPU agree: {mg['agreement']*100:.2f}%  F1: {mg['f1']:.3f}",
                fontsize=8)
        else:
            axes[row, col].imshow(m["diff_map"])
            axes[row, col].set_xlabel(
                f"NumPy↔OpenCV agree: {m['agreement']*100:.2f}%  F1: {m['f1']:.3f}",
                fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    title_str = "BSDS500 — NumPy vs GPU/cuTile vs OpenCV Canny" if has_gpu \
                else "BSDS500 — NumPy Canny vs OpenCV Canny"
    fig.suptitle(title_str + "\nThresholds computed from NumPy NMS output.",
                 fontsize=12, y=1.002)
    plt.tight_layout()
    plt.show()


def _print_summary(records: list[dict]) -> None:
    has_gpu = any("metrics_vs_gpu" in r and r["metrics_vs_gpu"] is not None
                  for r in records)

    sep = "=" * 90

    # ── NumPy vs OpenCV ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("RELIABILITY SUMMARY — NumPy Canny vs OpenCV Canny on BSDS500")
    print(sep)
    print(f"{'Image':<30} {'Agreement':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 76)

    oc_agreements, oc_precisions, oc_recalls, oc_f1s = [], [], [], []
    for rec in records:
        m = rec["metrics_vs_opencv"]
        name = rec["name"][:28]
        print(f"{name:<30} {m['agreement']*100:>9.2f}%"
              f" {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}")
        oc_agreements.append(m["agreement"]); oc_precisions.append(m["precision"])
        oc_recalls.append(m["recall"]);        oc_f1s.append(m["f1"])

    print("-" * 76)
    print(f"{'MEAN':<30} {np.mean(oc_agreements)*100:>9.2f}%"
          f" {np.mean(oc_precisions):>10.3f} {np.mean(oc_recalls):>10.3f} {np.mean(oc_f1s):>8.3f}")
    print(f"{'STDEV':<30} {np.std(oc_agreements)*100:>9.2f}%"
          f" {np.std(oc_precisions):>10.3f} {np.std(oc_recalls):>10.3f} {np.std(oc_f1s):>8.3f}")
    print(sep)

    # ── NumPy vs GPU ──────────────────────────────────────────────────────────
    gp_agreements, gp_precisions, gp_recalls, gp_f1s = [], [], [], []
    if has_gpu:
        print(f"\n{sep}")
        print("ACCURACY SUMMARY — NumPy Canny vs GPU/cuTile Canny on BSDS500")
        print(sep)
        print(f"{'Image':<30} {'Agreement':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
        print("-" * 76)

        for rec in records:
            mg = rec.get("metrics_vs_gpu")
            if mg is None:
                continue
            name = rec["name"][:28]
            print(f"{name:<30} {mg['agreement']*100:>9.2f}%"
                  f" {mg['precision']:>10.3f} {mg['recall']:>10.3f} {mg['f1']:>8.3f}")
            gp_agreements.append(mg["agreement"]); gp_precisions.append(mg["precision"])
            gp_recalls.append(mg["recall"]);        gp_f1s.append(mg["f1"])

        if gp_agreements:
            print("-" * 76)
            print(f"{'MEAN':<30} {np.mean(gp_agreements)*100:>9.2f}%"
                  f" {np.mean(gp_precisions):>10.3f} {np.mean(gp_recalls):>10.3f} {np.mean(gp_f1s):>8.3f}")
            print(f"{'STDEV':<30} {np.std(gp_agreements)*100:>9.2f}%"
                  f" {np.std(gp_precisions):>10.3f} {np.std(gp_recalls):>10.3f} {np.std(gp_f1s):>8.3f}")
        print(sep)
        print("\nHigh agreement / F1 between NumPy and GPU/cuTile confirms that the "
              "GPU pipeline is numerically equivalent to the CPU reference.\n")

    print(
        "\nMetric guide:\n"
        "  Agreement  - fraction of pixels where both agree (edge or not)\n"
        "  Precision  - fraction of reference edge pixels also in comparison\n"
        "  Recall     - fraction of comparison edge pixels also in reference\n"
        "  F1         - harmonic mean of precision & recall (1.0 = identical)\n"
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = Path("canny_reliability_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["Image",
                   "NP_vs_OC_Agreement", "NP_vs_OC_Precision",
                   "NP_vs_OC_Recall",    "NP_vs_OC_F1"]
        if has_gpu:
            headers += ["NP_vs_GPU_Agreement", "NP_vs_GPU_Precision",
                        "NP_vs_GPU_Recall",    "NP_vs_GPU_F1"]
        writer.writerow(headers)

        for rec in records:
            m  = rec["metrics_vs_opencv"]
            mg = rec.get("metrics_vs_gpu")
            row = [rec["name"],
                   f"{m['agreement']:.6f}", f"{m['precision']:.6f}",
                   f"{m['recall']:.6f}",    f"{m['f1']:.6f}"]
            if has_gpu:
                if mg:
                    row += [f"{mg['agreement']:.6f}", f"{mg['precision']:.6f}",
                            f"{mg['recall']:.6f}",    f"{mg['f1']:.6f}"]
                else:
                    row += ["N/A"] * 4
            writer.writerow(row)

    print(f"Detailed results saved to: {csv_path.absolute()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(max_images: int = None, kernel_size: int = 5, sigma: float = 1.4,
         high_percentile: float = 90.0, low_ratio: float = 0.5) -> None:

    dataset_root = _download_bsds500()
    image_paths = _find_test_images(dataset_root, max_images)

    if not image_paths:
        print("ERROR: no JPEG images found in the downloaded dataset.", file=sys.stderr)
        sys.exit(1)

    max_display = 5  # Only display first 5 images to avoid too many plots
    print(f"\nTesting on {len(image_paths)} image(s).\n")
    print(f"Parameters: kernel={kernel_size}, sigma={sigma}, "
          f"high_pct={high_percentile}, low_ratio={low_ratio}\n")
    print(f"Note: Full grid visualization will show first {max_display} images only.\n")

    print(f"GPU/cuTile pipeline: {'available' if HAS_GPU else 'NOT available (CPU only)'}\n")

    records = []
    for idx, img_path in enumerate(image_paths):
        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name} ...", end=" ", flush=True)

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("SKIP (could not read)")
            continue

        # --- NumPy Canny (adaptive percentile thresholds) ---
        np_edges, low, high = numpy_canny(
            gray, kernel_size=kernel_size, sigma=sigma,
            high_percentile=high_percentile, low_ratio=low_ratio,
        )

        # --- GPU/cuTile Canny (same thresholds via percentile on GPU) ---
        gpu_edges = None
        metrics_vs_gpu = None
        if HAS_GPU:
            try:
                gpu_edges, _, _ = gpu_canny(
                    gray, kernel_size=kernel_size, sigma=sigma,
                    high_percentile=high_percentile, low_ratio=low_ratio,
                )
                metrics_vs_gpu = _compute_metrics(np_edges, gpu_edges.astype(np.uint8) * 255)
            except Exception as exc:
                print(f"[gpu error: {exc}] ", end="")

        # --- OpenCV Canny (same thresholds, L2 gradient) ---
        oc_edges = cv2.Canny(
            gray, threshold1=low, threshold2=high,
            apertureSize=3, L2gradient=True,
        )

        metrics_vs_opencv = _compute_metrics(np_edges, oc_edges)

        gpu_info = ""
        if metrics_vs_gpu is not None:
            gpu_info = (f"  GPU↔NumPy agree={metrics_vs_gpu['agreement']*100:.1f}%"
                        f" F1={metrics_vs_gpu['f1']:.3f}")

        print(
            f"OC agree={metrics_vs_opencv['agreement']*100:.1f}%"
            f" F1={metrics_vs_opencv['f1']:.3f}"
            f"  (low={low:.1f}, high={high:.1f})"
            f"{gpu_info}"
        )

        records.append({
            "name":             img_path.name,
            "gray":             gray,
            "np_edges":         np_edges,
            "gpu_edges":        gpu_edges,
            "oc_edges":         oc_edges,
            "low":              low,
            "high":             high,
            "metrics_vs_opencv": metrics_vs_opencv,
            "metrics_vs_gpu":    metrics_vs_gpu,
        })

    if not records:
        print("No images could be processed.", file=sys.stderr)
        sys.exit(1)

    _print_summary(records)

    display_records = records[:max_display]
    _show_grid(display_records)


if __name__ == "__main__":
    main()