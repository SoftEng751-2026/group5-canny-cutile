import csv
import queue
import threading
import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

from cutile_full_pipeline import run as canny_run_full_gpu
from gaussian_benchmark import make_gaussian_kernel
from video_stream_demo import (
    create_video_writer,
    frame_to_grayscale_float32,
    make_source_tag,
    open_frame_source,
    parse_args,
    read_next_frame,
    resize_frame,
)


SENTINEL = object()


def producer_worker(args, source_type, loop_frame, capture, input_queue, stop_event):
    """
    Stage 0: read frames from camera/video/image-loop.

    This worker feeds frames into the pipeline. It is separate from the GPU worker
    so frame capture can overlap with GPU/CPU processing.
    """
    frame_index = 0

    try:
        while not stop_event.is_set():
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            ok, frame_bgr = read_next_frame(source_type, loop_frame, capture)

            if not ok:
                break

            input_queue.put(
                {
                    "frame_index": frame_index,
                    "frame_bgr": frame_bgr,
                    "submit_time": time.perf_counter(),
                }
            )

            frame_index += 1

    finally:
        input_queue.put(SENTINEL)


def gpu_full_worker(args, kernel_cpu, input_queue, result_queue, stop_event):
    """
    Stage 1: Full GPU Canny pipeline.

    For each frame:
      resize -> grayscale -> Gaussian (cuTile) -> Sobel fused (cuTile) ->
      NMS (cuTile) -> threshold (CuPy) -> hysteresis (cupyx) -> edges CPU

    All computation stays on GPU; only the final bool edge map is transferred
    back to CPU for display/write. No separate CPU post-processing stage needed.
    """
    while not stop_event.is_set():
        item = input_queue.get()

        if item is SENTINEL:
            result_queue.put(SENTINEL)
            break

        stage_start = time.perf_counter()

        resized_frame = resize_frame(item["frame_bgr"], args.resize_width)
        image_cpu = frame_to_grayscale_float32(resized_frame)

        edges, low, high, _ = canny_run_full_gpu(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=args.tile_size,
            high_pct=args.high_percentile,
            low_ratio=args.low_ratio,
            gpu_hysteresis=True,
        )

        cp.cuda.Stream.null.synchronize()
        stage_end = time.perf_counter()

        edges_uint8 = edges.astype(np.uint8) * 255
        edge_pixel_ratio = float(np.count_nonzero(edges) / edges.size)

        result_queue.put(
            {
                "frame_index": item["frame_index"],
                "submit_time": item["submit_time"],
                "resized_frame": resized_frame,
                "edges_uint8": edges_uint8,
                "low_threshold": low,
                "high_threshold": high,
                "edge_pixel_ratio": edge_pixel_ratio,
                "gpu_full_seconds": stage_end - stage_start,
            }
        )


def main():
    args = parse_args()

    if args.max_frames < 0:
        raise ValueError("max-frames must be >= 0")

    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")

    kernel_cpu = make_gaussian_kernel(args.kernel_size, args.sigma)

    source_tag = make_source_tag(args.source, args.image_loop)

    if args.results_csv is None:
        sigma_text = f"{args.sigma:g}".replace(".", "p")
        args.results_csv = (
            Path("report")
            / f"26_video_stream_pipeline_fps_{source_tag}_tile{args.tile_size}_k{args.kernel_size}_s{sigma_text}.csv"
        )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Pipelined video stream Canny demo")
    print("---------------------------------")
    print(f"Source: {args.image_loop if args.image_loop is not None else args.source}")
    print(f"Resize width: {args.resize_width}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Sigma: {args.sigma}")
    print(f"Tile size: {args.tile_size}")
    print(f"Threshold mode: {args.threshold_mode}")
    print(f"Results CSV: {args.results_csv}")
    print()
    print("Pipeline design:")
    print("Producer : read frames")
    print("GPU worker: resize/grayscale + full GPU Canny (cuTile Gaussian+Sobel+NMS, cupyx hysteresis)")
    print("Main     : display/write only")
    print("Press q in the display window to quit.")

    source_type, loop_frame, capture = open_frame_source(args.source, args.image_loop)

    input_queue  = queue.Queue(maxsize=4)
    result_queue = queue.Queue(maxsize=4)
    stop_event   = threading.Event()

    video_writer = None
    rows = []
    processed_frames = 0

    producer = threading.Thread(
        target=producer_worker,
        args=(args, source_type, loop_frame, capture, input_queue, stop_event),
        daemon=True,
    )

    gpu_worker = threading.Thread(
        target=gpu_full_worker,
        args=(args, kernel_cpu, input_queue, result_queue, stop_event),
        daemon=True,
    )

    overall_start = time.perf_counter()

    try:
        producer.start()
        gpu_worker.start()

        while True:
            item = result_queue.get()

            if item is SENTINEL:
                break

            edge_bgr = cv2.cvtColor(item["edges_uint8"], cv2.COLOR_GRAY2BGR)

            if video_writer is None and args.output_video is not None:
                video_writer = create_video_writer(args.output_video, edge_bgr.shape)

            if video_writer is not None:
                video_writer.write(edge_bgr)

            frame_done_time = time.perf_counter()
            latency_seconds = frame_done_time - item["submit_time"]

            rows.append(
                {
                    "frame_index":       item["frame_index"],
                    "gpu_full_seconds":  item["gpu_full_seconds"],
                    "latency_seconds":   latency_seconds,
                    "low_threshold":     item["low_threshold"],
                    "high_threshold":    item["high_threshold"],
                    "edge_pixel_ratio":  item["edge_pixel_ratio"],
                }
            )

            processed_frames += 1

            if not args.no_display:
                display_frame = np.hstack((item["resized_frame"], edge_bgr))

                recent_elapsed = time.perf_counter() - overall_start
                throughput_fps = processed_frames / recent_elapsed if recent_elapsed > 0 else 0.0

                cv2.putText(
                    display_frame,
                    f"Full-GPU FPS: {throughput_fps:.2f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Original | Full-GPU Canny edges", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    finally:
        stop_event.set()

        if capture is not None:
            capture.release()

        if video_writer is not None:
            video_writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

    overall_end = time.perf_counter()
    total_wall_time = overall_end - overall_start

    with args.results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "frame_index",
            "gpu_full_seconds",
            "latency_seconds",
            "low_threshold",
            "high_threshold",
            "edge_pixel_ratio",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "frame_index":      row["frame_index"],
                    "gpu_full_seconds": f"{row['gpu_full_seconds']:.8f}",
                    "latency_seconds":  f"{row['latency_seconds']:.8f}",
                    "low_threshold":    f"{row['low_threshold']:.8f}",
                    "high_threshold":   f"{row['high_threshold']:.8f}",
                    "edge_pixel_ratio": f"{row['edge_pixel_ratio']:.8f}",
                }
            )

    if rows:
        avg_gpu    = float(np.mean([r["gpu_full_seconds"] for r in rows]))
        avg_lat    = float(np.mean([r["latency_seconds"]  for r in rows]))
        avg_edges  = float(np.mean([r["edge_pixel_ratio"] for r in rows]))
        throughput = len(rows) / total_wall_time if total_wall_time > 0 else float("inf")

        print()
        print("Full-GPU pipelined video stream summary")
        print("----------------------------------------")
        print(f"Processed frames          : {len(rows)}")
        print(f"Total wall time           : {total_wall_time:.6f} s")
        print(f"Throughput FPS            : {throughput:.2f}")
        print(f"Avg full-GPU time/frame   : {avg_gpu*1000:.2f} ms")
        print(f"Avg end-to-end latency    : {avg_lat*1000:.2f} ms")
        print(f"Avg edge pixel ratio      : {avg_edges:.6f}")
        print(f"Results saved to          : {args.results_csv}")

        if args.output_video is not None:
            print(f"Output video saved to     : {args.output_video}")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()