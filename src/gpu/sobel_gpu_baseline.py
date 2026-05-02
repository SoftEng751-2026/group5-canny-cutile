import time
from pathlib import Path

import cv2
import cupy as cp
import numpy as np


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """
    Load an image from disk and convert it to grayscale float32.
    The CPU only does image loading; the Sobel computation is done on the GPU.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image.astype(np.float32)


def sobel_gpu(image_cpu: np.ndarray):
    """
    Compute Sobel gradients on the GPU using CuPy.

    This is a GPU baseline version. It helps us verify the Sobel logic and
    image input/output before replacing the core calculation with cuTile kernels.
    """
    image = cp.asarray(image_cpu, dtype=cp.float32)

    gx = cp.zeros_like(image)
    gy = cp.zeros_like(image)

    # Sobel X kernel:
    # [-1, 0, 1]
    # [-2, 0, 2]
    # [-1, 0, 1]
    gx[1:-1, 1:-1] = (
        -image[:-2, :-2]
        + image[:-2, 2:]
        - 2 * image[1:-1, :-2]
        + 2 * image[1:-1, 2:]
        - image[2:, :-2]
        + image[2:, 2:]
    )

    # Sobel Y kernel:
    # [ 1,  2,  1]
    # [ 0,  0,  0]
    # [-1, -2, -1]
    gy[1:-1, 1:-1] = (
        image[:-2, :-2]
        + 2 * image[:-2, 1:-1]
        + image[:-2, 2:]
        - image[2:, :-2]
        - 2 * image[2:, 1:-1]
        - image[2:, 2:]
    )

    magnitude = cp.sqrt(gx * gx + gy * gy)
    angle = (cp.arctan2(gy, gx) * 180.0 / cp.pi) % 180.0

    return magnitude, angle


def normalize_to_uint8(image_gpu) -> np.ndarray:
    """
    Convert a GPU array to a CPU uint8 image for saving.
    """
    image_cpu = cp.asnumpy(image_gpu)

    max_value = image_cpu.max()
    if max_value == 0:
        return image_cpu.astype(np.uint8)

    image_cpu = image_cpu / max_value * 255.0
    return image_cpu.astype(np.uint8)


def main():
    project_root = Path(__file__).resolve().parents[2]
    image_path = project_root / "data" / "test.jpg"
    output_path = project_root / "report" / "06_sobel_gpu_magnitude.png"

    print(f"Loading image: {image_path}")
    image_cpu = load_grayscale_image(image_path)

    print("Running Sobel gradient computation on GPU...")

    # Warm-up run: this avoids measuring one-off GPU/CuPy initialisation overhead.
    magnitude_gpu, angle_gpu = sobel_gpu(image_cpu)
    cp.cuda.Stream.null.synchronize()

    num_runs = 20
    start = time.perf_counter()

    for _ in range(num_runs):
        magnitude_gpu, angle_gpu = sobel_gpu(image_cpu)

    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    average_time = (end - start) / num_runs

    magnitude_image = normalize_to_uint8(magnitude_gpu)
    cv2.imwrite(str(output_path), magnitude_image)

    print(f"Input shape: {image_cpu.shape}")
    print(f"Average GPU Sobel time over {num_runs} runs: {average_time:.6f} seconds")
    print(f"Estimated FPS for Sobel step only: {1.0 / average_time:.2f}")
    print(f"Saved magnitude image to: {output_path}")
    print(f"Angle range: {float(cp.min(angle_gpu)):.2f} to {float(cp.max(angle_gpu)):.2f} degrees")


if __name__ == "__main__":
    main()