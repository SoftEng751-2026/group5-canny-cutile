"""
BSDS500 Canny reliability test.

Downloads images from the Berkeley Segmentation Dataset 500 and compares our
pure-NumPy Canny implementation against OpenCV's cv2.Canny on those images.
Displays visual results and prints a per-image and aggregate reliability summary.
"""

import sys
from collections import deque
from pathlib import Path

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime


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
# Display
# ---------------------------------------------------------------------------

def _show_grid(records: list[dict]) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.2 * n), squeeze=False)

    col_titles = [
        "Original (grayscale)",
        "NumPy Canny",
        "OpenCV Canny",
        "Difference\n(red=NumPy only, blue=OpenCV only, grey=both)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row, rec in enumerate(records):
        m = rec["metrics"]

        axes[row, 0].imshow(rec["gray"], cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_ylabel(rec["name"], fontsize=8, rotation=0,
                                labelpad=90, va="center")

        axes[row, 1].imshow(rec["np_edges"], cmap="gray")
        axes[row, 1].set_xlabel(
            f"edge pixels: {m['numpy_edge_pct']:.2f}%  "
            f"(low={rec['low']:.1f}, high={rec['high']:.1f})",
            fontsize=8,
        )

        axes[row, 2].imshow(rec["oc_edges"], cmap="gray")
        axes[row, 2].set_xlabel(
            f"edge pixels: {m['opencv_edge_pct']:.2f}%",
            fontsize=8,
        )

        axes[row, 3].imshow(m["diff_map"])
        axes[row, 3].set_xlabel(
            f"pixel agreement: {m['agreement']*100:.2f}%   F1: {m['f1']:.3f}",
            fontsize=8,
        )

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "BSDS500 - Pure-NumPy Canny vs OpenCV Canny\n"
        "Thresholds are computed from the NumPy NMS output and applied to both implementations.",
        fontsize=12,
        y=1.005,
    )
    plt.tight_layout()
    plt.show()


def _print_summary(records: list[dict]) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print("RELIABILITY SUMMARY -- Pure NumPy Canny vs OpenCV Canny on BSDS500")
    print(sep)
    print(f"{'Image':<30} {'Agreement':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 76)

    agreements, precisions, recalls, f1s = [], [], [], []
    for rec in records:
        m = rec["metrics"]
        name = rec["name"][:28]
        print(
            f"{name:<30} {m['agreement']*100:>9.2f}%"
            f" {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}"
        )
        agreements.append(m["agreement"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    print("-" * 76)
    print(
        f"{'MEAN':<30} {np.mean(agreements)*100:>9.2f}%"
        f" {np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f} {np.mean(f1s):>8.3f}"
    )
    print(
        f"{'STDEV':<30} {np.std(agreements)*100:>9.2f}%"
        f" {np.std(precisions):>10.3f} {np.std(recalls):>10.3f} {np.std(f1s):>8.3f}"
    )
    print(
        f"{'MIN':<30} {np.min(agreements)*100:>9.2f}%"
        f" {np.min(precisions):>10.3f} {np.min(recalls):>10.3f} {np.min(f1s):>8.3f}"
    )
    print(
        f"{'MAX':<30} {np.max(agreements)*100:>9.2f}%"
        f" {np.max(precisions):>10.3f} {np.max(recalls):>10.3f} {np.max(f1s):>8.3f}"
    )
    print(sep)
    print(
        "\nMetric guide:\n"
        "  Agreement  - fraction of pixels where both implementations agree (edge or not)\n"
        "  Precision  - fraction of NumPy edge pixels also detected by OpenCV\n"
        "  Recall     - fraction of OpenCV edge pixels also detected by NumPy\n"
        "  F1         - harmonic mean of precision & recall (1.0 = identical)\n"
        "\nExpected differences arise from:\n"
        "  * NumPy uses edge-padding; OpenCV uses a different border strategy\n"
        "  * OpenCV computes Sobel on uint8 internally before converting to float\n"
        "  * Minor floating-point ordering differences in NMS\n"
    )
    
    # Save results to CSV
    csv_path = Path("canny_reliability_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Agreement", "Precision", "Recall", "F1"])
        for rec in records:
            m = rec["metrics"]
            writer.writerow([
                rec["name"],
                f"{m['agreement']:.6f}",
                f"{m['precision']:.6f}",
                f"{m['recall']:.6f}",
                f"{m['f1']:.6f}",
            ])
        writer.writerow([])  # blank line
        writer.writerow(["AGGREGATED STATISTICS"])
        writer.writerow(["Metric", "Mean", "Stdev", "Min", "Max", "Count"])
        writer.writerow([
            "Agreement",
            f"{np.mean(agreements):.6f}",
            f"{np.std(agreements):.6f}",
            f"{np.min(agreements):.6f}",
            f"{np.max(agreements):.6f}",
            len(agreements),
        ])
        writer.writerow([
            "Precision",
            f"{np.mean(precisions):.6f}",
            f"{np.std(precisions):.6f}",
            f"{np.min(precisions):.6f}",
            f"{np.max(precisions):.6f}",
            len(precisions),
        ])
        writer.writerow([
            "Recall",
            f"{np.mean(recalls):.6f}",
            f"{np.std(recalls):.6f}",
            f"{np.min(recalls):.6f}",
            f"{np.max(recalls):.6f}",
            len(recalls),
        ])
        writer.writerow([
            "F1",
            f"{np.mean(f1s):.6f}",
            f"{np.std(f1s):.6f}",
            f"{np.min(f1s):.6f}",
            f"{np.max(f1s):.6f}",
            len(f1s),
        ])
    
    print(f"\n✓ 详细结果已保存至: {csv_path.absolute()}")
    return {
        "mean_agreement": np.mean(agreements),
        "mean_precision": np.mean(precisions),
        "mean_recall": np.mean(recalls),
        "mean_f1": np.mean(f1s),
        "std_agreement": np.std(agreements),
        "std_precision": np.std(precisions),
        "std_recall": np.std(recalls),
        "std_f1": np.std(f1s),
        "count": len(records),
    }


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

    records = []
    for idx, img_path in enumerate(image_paths):
        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name} ...", end=" ", flush=True)

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("SKIP (could not read)")
            continue

        # --- NumPy Canny (adaptive percentile thresholds) ---
        np_edges, low, high = numpy_canny(
            gray,
            kernel_size=kernel_size,
            sigma=sigma,
            high_percentile=high_percentile,
            low_ratio=low_ratio,
        )

        # --- OpenCV Canny (same thresholds, L2 gradient to match our hypot) ---
        # We pass the raw uint8 image so OpenCV handles its own border internally.
        # Thresholds are in gradient-magnitude units (same scale as ours since both
        # operate on 0-255 intensity values).
        oc_edges = cv2.Canny(
            gray,
            threshold1=low,
            threshold2=high,
            apertureSize=3,
            L2gradient=True,
        )

        metrics = _compute_metrics(np_edges, oc_edges)

        print(
            f"agreement={metrics['agreement']*100:.1f}%  "
            f"F1={metrics['f1']:.3f}  "
            f"(low={low:.1f}, high={high:.1f})"
        )

        records.append({
            "name": img_path.name,
            "gray": gray,
            "np_edges": np_edges,
            "oc_edges": oc_edges,
            "low": low,
            "high": high,
            "metrics": metrics,
        })

    if not records:
        print("No images could be processed.", file=sys.stderr)
        sys.exit(1)

    summary = _print_summary(records)
    
    # Only display the first max_display images
    display_records = records[:max_display]
    _show_grid(display_records)


if __name__ == "__main__":
    main()