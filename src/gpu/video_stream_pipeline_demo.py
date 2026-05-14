import csv
import queue
import threading
import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

from cutile_canny_pipeline_benchmark import canny_pipeline_gpu_with_input_transfer
from gaussian_benchmark import make_gaussian_kernel
from video_stream_demo import (
    create_video_writer,
    double_threshold_and_hysteresis,
    frame_to_grayscale_float32,
    make_source_tag,
    open_frame_source,
    parse_args,
    read_next_frame,
    resize_frame,
    select_thresholds,
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


def gpu_frontend_worker(args, kernel_cpu, input_queue, nms_queue, stop_event):
    """
    Stage 1: GPU frontend.

    For each frame:
    resize -> grayscale -> GPU Gaussian/Sobel/NMS -> copy NMS result back to CPU.

    The CPU post-processing stage can run in parallel with this worker on the
    previous frame.
    """
    while not stop_event.is_set():
        item = input_queue.get()

        if item is SENTINEL:
            nms_queue.put(SENTINEL)
            break

        stage_start = time.perf_counter()

        resized_frame = resize_frame(item["frame_bgr"], args.resize_width)
        image_cpu = frame_to_grayscale_float32(resized_frame)

        _, _, _, nms_gpu = canny_pipeline_gpu_with_input_transfer(
            image_cpu=image_cpu,
            kernel_cpu=kernel_cpu,
            tile_size=args.tile_size,
        )

        # Ensure GPU work is finished before copying the NMS output to CPU.
        cp.cuda.Stream.null.synchronize()
        nms_cpu = cp.asnumpy(nms_gpu)

        stage_end = time.perf_counter()

        nms_queue.put(
            {
                "frame_index": item["frame_index"],
                "submit_time": item["submit_time"],
                "resized_frame": resized_frame,
                "nms_cpu": nms_cpu,
                "gpu_frontend_seconds": stage_end - stage_start,
            }
        )


def postprocess_frame(args, nms_item):
    """
    Stage 2: CPU post-processing.

    Apply threshold selection, double thresholding and hysteresis.
    """
    stage_start = time.perf_counter()

    low, high = select_thresholds(
        nms_image=nms_item["nms_cpu"],
        threshold_mode=args.threshold_mode,
        high_percentile=args.high_percentile,
        low_ratio=args.low_ratio,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    edges_bool = double_threshold_and_hysteresis(nms_item["nms_cpu"], low, high)
    edges_uint8 = edges_bool.astype(np.uint8) * 255
    edge_pixel_ratio = float(np.count_nonzero(edges_bool) / edges_bool.size)

    stage_end = time.perf_counter()

    return {
        "edges_uint8": edges_uint8,
        "low_threshold": low,
        "high_threshold": high,
        "edge_pixel_ratio": edge_pixel_ratio,
        "cpu_postprocess_seconds": stage_end - stage_start,
    }


def run_pipeline(args, kernel_cpu):
    """
    Execute the pipelined video stream and return (rows, total_wall_time_seconds).

    args must be a Namespace with: source, image_loop, max_frames, resize_width,
    tile_size, threshold_mode, high_percentile, low_ratio, low_threshold,
    high_threshold, no_display, output_video.
    """
    source_type, loop_frame, capture = open_frame_source(args.source, args.image_loop)

    input_queue = queue.Queue(maxsize=4)
    nms_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    video_writer = None
    rows = []
    processed_frames = 0

    producer = threading.Thread(
        target=producer_worker,
        args=(args, source_type, loop_frame, capture, input_queue, stop_event),
        daemon=True,
    )

    gpu_worker = threading.Thread(
        target=gpu_frontend_worker,
        args=(args, kernel_cpu, input_queue, nms_queue, stop_event),
        daemon=True,
    )

    overall_start = time.perf_counter()

    try:
        producer.start()
        gpu_worker.start()

        while True:
            nms_item = nms_queue.get()

            if nms_item is SENTINEL:
                break

            post_result = postprocess_frame(args, nms_item)

            edge_bgr = cv2.cvtColor(post_result["edges_uint8"], cv2.COLOR_GRAY2BGR)

            if video_writer is None and args.output_video is not None:
                video_writer = create_video_writer(args.output_video, edge_bgr.shape)

            if video_writer is not None:
                video_writer.write(edge_bgr)

            frame_done_time = time.perf_counter()
            latency_seconds = frame_done_time - nms_item["submit_time"]

            rows.append(
                {
                    "frame_index": nms_item["frame_index"],
                    "gpu_frontend_seconds": nms_item["gpu_frontend_seconds"],
                    "cpu_postprocess_seconds": post_result["cpu_postprocess_seconds"],
                    "latency_seconds": latency_seconds,
                    "low_threshold": post_result["low_threshold"],
                    "high_threshold": post_result["high_threshold"],
                    "edge_pixel_ratio": post_result["edge_pixel_ratio"],
                }
            )

            processed_frames += 1

            if not args.no_display:
                display_frame = np.hstack((nms_item["resized_frame"], edge_bgr))

                recent_elapsed = time.perf_counter() - overall_start
                throughput_fps = processed_frames / recent_elapsed if recent_elapsed > 0 else 0.0

                cv2.putText(
                    display_frame,
                    f"Pipeline FPS: {throughput_fps:.2f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Original | Pipelined Canny edges", display_frame)

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

    return rows, total_wall_time


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
    print("Producer: read frames")
    print("GPU worker: resize/grayscale + Gaussian/Sobel/NMS")
    print("CPU/main: threshold + hysteresis + display/write")
    print("Press q in the display window to quit.")

    rows, total_wall_time = run_pipeline(args, kernel_cpu)

    with args.results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "frame_index",
            "gpu_frontend_seconds",
            "cpu_postprocess_seconds",
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
                    "frame_index": row["frame_index"],
                    "gpu_frontend_seconds": f"{row['gpu_frontend_seconds']:.8f}",
                    "cpu_postprocess_seconds": f"{row['cpu_postprocess_seconds']:.8f}",
                    "latency_seconds": f"{row['latency_seconds']:.8f}",
                    "low_threshold": f"{row['low_threshold']:.8f}",
                    "high_threshold": f"{row['high_threshold']:.8f}",
                    "edge_pixel_ratio": f"{row['edge_pixel_ratio']:.8f}",
                }
            )

    if rows:
        avg_gpu = float(np.mean([row["gpu_frontend_seconds"] for row in rows]))
        avg_cpu = float(np.mean([row["cpu_postprocess_seconds"] for row in rows]))
        avg_latency = float(np.mean([row["latency_seconds"] for row in rows]))
        avg_edge_ratio = float(np.mean([row["edge_pixel_ratio"] for row in rows]))
        throughput_fps = len(rows) / total_wall_time if total_wall_time > 0 else float("inf")

        print()
        print("Pipelined video stream summary")
        print("------------------------------")
        print(f"Processed frames: {len(rows)}")
        print(f"Total wall time: {total_wall_time:.6f} seconds")
        print(f"Throughput FPS: {throughput_fps:.2f}")
        print(f"Average GPU frontend time: {avg_gpu:.6f} seconds/frame")
        print(f"Average CPU postprocess time: {avg_cpu:.6f} seconds/frame")
        print(f"Average end-to-end latency: {avg_latency:.6f} seconds/frame")
        print(f"Average edge pixel ratio: {avg_edge_ratio:.6f}")
        print(f"Results saved to: {args.results_csv}")

        if args.output_video is not None:
            print(f"Output video saved to: {args.output_video}")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()