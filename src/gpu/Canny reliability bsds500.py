"""
BSDS500 Canny reliability test.

Downloads images from the Berkeley Segmentation Dataset 500 and compares our
pure-NumPy Canny implementation against OpenCV's cv2.Canny on those images.

- Runs on ALL images in the test set (up to 500)
- Displays a visual grid of 10 evenly-sampled images
- Prints a per-image and aggregate reliability summary
- Saves full results to CSV
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
    diff_map[np_bin & oc_bin] = [220, 220, 220]   # grey  - both agree
    diff_map[np_bin & ~oc_bin] = [220, 60, 60]    # red   - NumPy only
    diff_map[~np_bin & oc_bin] = [60, 60, 220]    # blue  - OpenCV only

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
# Display — evenly sampled 10 images from all results
# ---------------------------------------------------------------------------

def _sample_display_records(records: list[dict], n_display: int = 10) -> list[dict]:
    """
    Pick n_display records evenly spread across the full results list.
    If fewer records than n_display, return all.
    """
    total = len(records)
    if total <= n_display:
        return records
    # Evenly spaced indices covering the full range
    indices = np.linspace(0, total - 1, n_display, dtype=int)
    return [records[i] for i in indices]


def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2-D (grayscale/bool) or 3-D (RGB) array to (h, w) using cv2."""
    dtype = arr.dtype
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    if dtype == bool:
        return resized > 127
    return resized


def _show_grid(records: list[dict], title_suffix: str = "",
               thumb_w: int = 240, thumb_h: int = 160) -> None:
    """
    Display a clean grid of results.

    Layout uses n+2 rows in GridSpec:
      row 0  — title + legend (text-only, tall)
      row 1  — column headers (text-only, short)
      rows 2+ — image data rows
    This ensures title, legend and column headers never overlap.
    """
    n = len(records)

    DPI      = 100
    label_px = 120    # filename column width
    gap_px   = 6      # gap between columns
    ann_px   = 28     # space below each image row for annotation
    title_px = 32     # height of title row
    hdr_px   = 22     # height of column-header row

    fig_w_px = label_px + (thumb_w + gap_px) * 4
    fig_h_px = title_px + hdr_px + n * (thumb_h + ann_px + gap_px)

    fig_w = fig_w_px / DPI
    fig_h = fig_h_px / DPI

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    label_frac = label_px / fig_w_px
    img_frac   = thumb_w  / fig_w_px
    title_frac = title_px / fig_h_px
    hdr_frac   = hdr_px   / fig_h_px
    data_frac  = 1.0 - title_frac - hdr_frac

    # Outer vertical split: [title | header | data]
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    outer = GridSpec(
        3, 1, figure=fig,
        height_ratios=[title_frac, hdr_frac, data_frac],
        hspace=0,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )

    # ── Row 0: title + legend ────────────────────────────────────────────────
    ax_title = fig.add_subplot(outer[0])
    ax_title.axis("off")
    ax_title.text(0.5, 0.75,
                  f"BSDS500 — NumPy Canny vs OpenCV Canny  {title_suffix}",
                  ha="center", va="center", fontsize=10, fontweight="bold",
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.20,
                  "Diff key:  grey = both agree  ·  red = NumPy only  ·  blue = OpenCV only",
                  ha="center", va="center", fontsize=7, style="italic",
                  transform=ax_title.transAxes)

    # ── Row 1: column headers ────────────────────────────────────────────────
    hdr_gs = GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1],
        width_ratios=[label_frac, img_frac, img_frac, img_frac, img_frac],
        wspace=gap_px / thumb_w,
    )
    col_labels = ["", "Original", "NumPy Canny", "OpenCV Canny", "Difference"]
    for c, lbl in enumerate(col_labels):
        ax_h = fig.add_subplot(hdr_gs[0, c])
        ax_h.axis("off")
        ax_h.text(0.5, 0.5, lbl,
                  ha="center", va="center", fontsize=8, fontweight="bold",
                  transform=ax_h.transAxes)

    # ── Rows 2+: image data ──────────────────────────────────────────────────
    data_gs = GridSpecFromSubplotSpec(
        n, 5, subplot_spec=outer[2],
        width_ratios=[label_frac, img_frac, img_frac, img_frac, img_frac],
        hspace=ann_px / thumb_h,
        wspace=gap_px / thumb_w,
    )
    axes = [[fig.add_subplot(data_gs[r, c]) for c in range(5)] for r in range(n)]

    for row, rec in enumerate(records):
        m  = rec["metrics"]
        ax = axes[row]

        # col 0 – filename
        ax[0].axis("off")
        ax[0].text(0.98, 0.5, rec["name"],
                   ha="right", va="center", fontsize=7,
                   transform=ax[0].transAxes)

        # col 1 – original (keep true aspect ratio via "equal", let axes shrink)
        ax[1].imshow(_resize_to(rec["gray"], thumb_h, thumb_w),
                     cmap="gray", vmin=0, vmax=255, aspect="equal")

        # col 2 – NumPy Canny
        ax[2].imshow(_resize_to(rec["np_edges"].astype(np.uint8) * 255, thumb_h, thumb_w),
                     cmap="gray", aspect="equal")
        ax[2].set_xlabel(
            f"edges {m['numpy_edge_pct']:.1f}%  thr [{rec['low']:.0f}–{rec['high']:.0f}]",
            fontsize=7, labelpad=2)

        # col 3 – OpenCV Canny
        ax[3].imshow(_resize_to(rec["oc_edges"], thumb_h, thumb_w),
                     cmap="gray", aspect="equal")
        ax[3].set_xlabel(f"edges {m['opencv_edge_pct']:.1f}%", fontsize=7, labelpad=2)

        # col 4 – diff map
        ax[4].imshow(_resize_to(m["diff_map"], thumb_h, thumb_w), aspect="equal")
        ax[4].set_xlabel(
            f"agree {m['agreement']*100:.1f}%   F1 {m['f1']:.3f}",
            fontsize=7, labelpad=2)

        # Remove ticks
        for col in range(1, 5):
            ax[col].set_xticks([])
            ax[col].set_yticks([])

    out_path = "canny_reliability_grid.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    print(f"✓ Grid saved to {out_path}")


# ---------------------------------------------------------------------------
# Summary + CSV export
# ---------------------------------------------------------------------------

def _print_summary(records: list[dict]) -> dict:
    sep = "=" * 76
    print(f"\n{sep}")
    print("RELIABILITY SUMMARY -- Pure NumPy Canny vs OpenCV Canny on BSDS500")
    print(f"Total images tested: {len(records)}")
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
        writer.writerow([])
        writer.writerow(["AGGREGATED STATISTICS"])
        writer.writerow(["Metric", "Mean", "Stdev", "Min", "Max", "Count"])
        for label, vals in [
            ("Agreement", agreements),
            ("Precision", precisions),
            ("Recall", recalls),
            ("F1", f1s),
        ]:
            writer.writerow([
                label,
                f"{np.mean(vals):.6f}",
                f"{np.std(vals):.6f}",
                f"{np.min(vals):.6f}",
                f"{np.max(vals):.6f}",
                len(vals),
            ])

    print(f"\n✓ Full results saved to: {csv_path.absolute()}")

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

def main(
    max_images: int = None,       # None = run on ALL images (full BSDS500 test set)
    n_display: int = 10,          # how many images to show in the visual grid
    kernel_size: int = 5,
    sigma: float = 1.4,
    high_percentile: float = 90.0,
    low_ratio: float = 0.5,
) -> dict:
    """
    Run reliability test on the full BSDS500 test set.

    Parameters
    ----------
    max_images : int or None
        Cap on number of images to process. None = all (recommended for full test).
    n_display : int
        Number of images to show in the visual grid. They are evenly sampled
        from the full results so you get a representative spread.
    kernel_size, sigma, high_percentile, low_ratio
        Canny hyperparameters.

    Returns
    -------
    dict with aggregate statistics (mean/std/min/max for agreement, precision, recall, F1)
    """
    dataset_root = _download_bsds500()
    image_paths = _find_test_images(dataset_root, max_images)

    if not image_paths:
        print("ERROR: no JPEG images found in the downloaded dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTesting on {len(image_paths)} image(s).")
    print(f"Parameters: kernel={kernel_size}, sigma={sigma}, "
          f"high_pct={high_percentile}, low_ratio={low_ratio}")
    print(f"Visual grid will show {min(n_display, len(image_paths))} evenly-sampled images.\n")

    records = []
    for idx, img_path in enumerate(image_paths):
        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name} ...", end=" ", flush=True)

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("SKIP (could not read)")
            continue

        np_edges, low, high = numpy_canny(
            gray,
            kernel_size=kernel_size,
            sigma=sigma,
            high_percentile=high_percentile,
            low_ratio=low_ratio,
        )

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

    # Evenly sample n_display images for the visual grid
    display_records = _sample_display_records(records, n_display)
    title_suffix = f" ({len(records)} images tested, {len(display_records)} shown)"
    _show_grid(display_records, title_suffix=title_suffix)

    return summary


if __name__ == "__main__":
    main()