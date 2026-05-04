import argparse
import csv
import time
from pathlib import Path

from paths import resolve_project_path

import cv2
import cupy as cp
import numpy as np

from cutile_canny_pipeline_benchmark import canny_pipeline_gpu_with_input_transfer
from gaussian_benchmark import make_gaussian_kernel


def parse_video_source(source: str):
    """Use integer camera index if source is a digit, otherwise use it as a video path."""
    return int(source) if source.isdigit() else source


def make_source_tag(source: str, image_loop: Path | None) -> str:
    """Create a readable output tag without hardcoding a specific source name."""
    if image_loop is not None:
        return image_loop.stem

    if source.isdigit():
        return f"camera{source}"

    return Path(source).stem


def resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    """Resize frame while preserving aspect ratio. If resize_width <= 0, keep original size."""
    if resize_width <= 0:
        return frame

    height, width = frame.shape[:2]

    if width == resize_width:
        return frame

    scale = resize_width / width
    resize_height = int(round(height * scale))

    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def frame_to_grayscale_float32(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to grayscale float32 image."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def select_thresholds(
    nms_image: np.ndarray,
    threshold_mode: str,
    high_percentile: float,
    low_ratio: float,
    low_threshold: float | None,
    high_threshold: float | None,
):
    """
    Select Canny thresholds.

    Percentile mode adapts to each frame, avoiding hardcoded thresholds.
    Fixed mode is available for controlled experiments.
    """
    if threshold_mode == "fixed":
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


def hysteresis_connected_components(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """
    Fast CPU hysteresis using connected components.

    Weak pixels are kept only if they are 8-connected to at least one strong pixel.
    This matches the Canny hysteresis idea and is faster than a Python BFS loop.
    """
    if strong.shape != weak.shape:
        raise ValueError("strong and weak masks must have the same shape")

    candidate_edges = strong | weak

    if not np.any(strong):
        return np.zeros_like(strong, dtype=bool)

    num_labels, labels = cv2.connectedComponents(
        candidate_edges.astype(np.uint8),
        connectivity=8,
    )

    strong_labels = np.unique(labels[strong])

    keep_label = np.zeros(num_labels, dtype=bool)
    keep_label[strong_labels] = True
    keep_label[0] = False

    return keep_label[labels]


def double_threshold_and_hysteresis(
    nms_image: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Apply double threshold and hysteresis to an NMS image."""
    strong = nms_image >= high_threshold
    weak = (nms_image >= low_threshold) & ~strong

    return hysteresis_connected_components(strong, weak)


def process_frame(
    frame_bgr: np.ndarray,
    kernel_cpu: np.ndarray,
    tile_size: int,
    threshold_mode: str,
    high_percentile: float,
    low_ratio: float,
    low_threshold: float | None,
    high_threshold: float | None,
    resize_width: int,
):
    """
    Process one frame using:
    resize -> grayscale -> GPU/cuTile frontend -> CPU threshold+hysteresis.
    """
    resized_frame = resize_frame(frame_bgr, resize_width)
    image_cpu = frame_to_grayscale_float32(resized_frame)

    _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
        image_cpu=image_cpu,
        kernel_cpu=kernel_cpu,
        tile_size=tile_size,
    )

    cp.cuda.Stream.null.synchronize()
    nms_cpu = cp.asnumpy(nms_gpu)

    low, high = select_thresholds(
        nms_image=nms_cpu,
        threshold_mode=threshold_mode,
        high_percentile=high_percentile,
        low_ratio=low_ratio,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    edges_bool = double_threshold_and_hysteresis(nms_cpu, low, high)
    edges_uint8 = edges_bool.astype(np.uint8) * 255

    edge_pixel_ratio = float(np.count_nonzero(edges_bool) / edges_bool.size)

    return resized_frame, edges_uint8, low, high, edge_pixel_ratio


def open_frame_source(source: str, image_loop: Path | None):
    """
    Return either a repeated image source or an OpenCV VideoCapture source.
    """
    if image_loop is not None:
        frame = cv2.imread(str(image_loop), cv2.IMREAD_COLOR)

        if frame is None:
            raise FileNotFoundError(f"Could not load image-loop frame: {image_loop}")

        return "image_loop", frame, None

    capture = cv2.VideoCapture(parse_video_source(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. "
            "Use --source 0 for webcam, or --image-loop data/test.jpg for testing."
        )

    return "capture", None, capture


def read_next_frame(source_type: str, loop_frame: np.ndarray | None, capture):
    """Read one frame from image-loop or VideoCapture."""
    if source_type == "image_loop":
        return True, loop_frame.copy()

    return capture.read()


def create_video_writer(output_video: Path | None, frame_shape):
    """Create an OpenCV video writer if output_video is provided."""
    if output_video is None:
        return None

    output_video.parent.mkdir(parents=True, exist_ok=True)

    height, width = frame_shape[:2]
    suffix = output_video.suffix.lower()

    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(str(output_video), fourcc, 20.0, (width, height), isColor=True)

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    return writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time/video-stream Canny demo using GPU/cuTile frontend."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index, for example 0, or path to a video file.",
    )

    parser.add_argument(
        "--image-loop",
        type=Path,
        default=None,
        help="Repeat a single image as a fake video stream, useful for testing without webcam.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to process. 0 means until video ends or user quits.",
    )

    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Resize frames to this width before processing. Use 0 to keep original size.",
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
        "--tile-size",
        type=int,
        default=256,
        help="cuTile tile size for Sobel magnitude.",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["percentile", "fixed"],
        default="percentile",
        help="Threshold selection mode.",
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
        "--no-display",
        action="store_true",
        help="Disable OpenCV windows. Useful for benchmarking or when display is unavailable.",
    )

    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path to save edge video, for example report/canny_demo.mp4.",
    )

    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-frame timing results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.image_loop is not None:
        args.image_loop = resolve_project_path(args.image_loop)

    if args.output_video is not None:
        args.output_video = resolve_project_path(args.output_video)

    if args.results_csv is not None:
        args.results_csv = resolve_project_path(args.results_csv)

    if args.max_frames < 0:
        raise ValueError("max-frames must be >= 0")

    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")

    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    source_tag = make_source_tag(args.source, args.image_loop)

    if args.results_csv is None:
        sigma_text = f"{args.sigma:g}".replace(".", "p")
        args.results_csv = resolve_project_path(
            Path("report")
            / f"25_video_stream_fps_{source_tag}_tile{args.tile_size}_k{args.kernel_size}_s{sigma_text}.csv"
        )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Video stream Canny demo")
    print("-----------------------")
    print(f"Source: {args.image_loop if args.image_loop is not None else args.source}")
    print(f"Resize width: {args.resize_width}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Sigma: {args.sigma}")
    print(f"Tile size: {args.tile_size}")
    print(f"Threshold mode: {args.threshold_mode}")
    print(f"Results CSV: {args.results_csv}")
    print("Press q in the display window to quit.")

    source_type, loop_frame, capture = open_frame_source(args.source, args.image_loop)

    video_writer = None
    rows = []
    frame_index = 0

    try:
        while True:
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            ok, frame_bgr = read_next_frame(source_type, loop_frame, capture)

            if not ok:
                break

            start = time.perf_counter()

            resized_frame, edges_uint8, low, high, edge_pixel_ratio = process_frame(
                frame_bgr=frame_bgr,
                kernel_cpu=kernel_cpu,
                tile_size=args.tile_size,
                threshold_mode=args.threshold_mode,
                high_percentile=args.high_percentile,
                low_ratio=args.low_ratio,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
                resize_width=args.resize_width,
            )

            end = time.perf_counter()

            processing_time = end - start
            fps = 1.0 / processing_time if processing_time > 0 else float("inf")

            edge_bgr = cv2.cvtColor(edges_uint8, cv2.COLOR_GRAY2BGR)

            if video_writer is None and args.output_video is not None:
                video_writer = create_video_writer(args.output_video, edge_bgr.shape)

            if video_writer is not None:
                video_writer.write(edge_bgr)

            rows.append(
                {
                    "frame_index": frame_index,
                    "processing_time_seconds": processing_time,
                    "fps": fps,
                    "low_threshold": low,
                    "high_threshold": high,
                    "edge_pixel_ratio": edge_pixel_ratio,
                }
            )

            if not args.no_display:
                display_frame = np.hstack((resized_frame, edge_bgr))
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.2f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Original | Canny edges", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

    finally:
        if capture is not None:
            capture.release()

        if video_writer is not None:
            video_writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

    with args.results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "processing_time_seconds",
                "fps",
                "low_threshold",
                "high_threshold",
                "edge_pixel_ratio",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "frame_index": row["frame_index"],
                    "processing_time_seconds": f"{row['processing_time_seconds']:.8f}",
                    "fps": f"{row['fps']:.4f}",
                    "low_threshold": f"{row['low_threshold']:.8f}",
                    "high_threshold": f"{row['high_threshold']:.8f}",
                    "edge_pixel_ratio": f"{row['edge_pixel_ratio']:.8f}",
                }
            )

    if rows:
        average_time = float(np.mean([row["processing_time_seconds"] for row in rows]))
        average_fps = 1.0 / average_time if average_time > 0 else float("inf")
        average_edge_ratio = float(np.mean([row["edge_pixel_ratio"] for row in rows]))

        print()
        print("Video stream summary")
        print("--------------------")
        print(f"Processed frames: {len(rows)}")
        print(f"Average processing time: {average_time:.6f} seconds/frame")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Average edge pixel ratio: {average_edge_ratio:.6f}")
        print(f"Results saved to: {args.results_csv}")

        if args.output_video is not None:
            print(f"Output video saved to: {args.output_video}")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()