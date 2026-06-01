# group5-canny-cutile

Canny Edge Detection accelerated with [cuTile](https://docs.nvidia.com/cuda/cutile-python/) —
NVIDIA's Python DSL for tile-parallel GPU programming.
**SoftEng 751, Group 5, 2026.**

Team: Jiaxi Liu · Xudong Ma · Shiying Yang

---

## What this project implements

A complete, GPU-accelerated Canny edge-detection pipeline with three layers of parallelism:

| Layer | Where | Mechanism |
|:---|:---|:---|
| **Pixel-level data parallelism** | Gaussian / Sobel / NMS | `N_blocks = pixels / tile_size` blocks run concurrently on GPU SMs |
| **cuTile fused kernels** | Sobel (row-view) / NMS (row-view) / Gaussian | One pass reads neighbours as zero-copy column-offset views, writes mag+gx+gy; eliminates 8–9 full-image copies |
| **Cross-frame pipeline parallelism** | Video stream | GPU frontend of frame K+1 overlaps CPU post-processing of frame K via `threading` |

The stage-4/5 improvement moves `threshold + hysteresis` from CPU (OpenCV connected components)
to GPU (`cupyx.scipy.ndimage.label`), eliminating the Amdahl bottleneck identified in the
analysis notebook.

---

## Pipeline

```
Input image
    │
    ▼
[1] Gaussian blur (k=5, σ=1.4)
    ├─ cuTile fused 5-tap kernel  (k=5 only; CuPy fallback for other sizes)
    │
    ▼
[2] Sobel gradient (mag + gx + gy)
    ├─ cuTile row-view fused kernel  (1 pad copy → 8 zero-copy col-offset views)
    │   → writes magnitude, gx, gy in one pass; CuPy fallback
    │
    ▼
[3] Non-Maximum Suppression
    ├─ cuTile row-view kernel  (1 pad copy → 9 zero-copy views; CuPy fallback)
    │
    ▼
[4] Double threshold  (percentile-based or fixed)
    ├─ CuPy (GPU)
    │
    ▼
[5] Hysteresis
    ├─ GPU: cupyx.scipy.ndimage.label  ← default, removes the Amdahl bottleneck
    └─ CPU: OpenCV connectedComponents ← --cpu-hysteresis flag, kept for comparison
    │
    ▼
Output edge image (bool / uint8)
```

---

## File structure

```
notebooks/
  evaluate_performance_en.ipynb   ← PRIMARY analysis notebook (English, live benchmarks)
  evaluate_performance.ipynb      ← Chinese companion notebook
  01_canny_baseline 1.ipynb       ← Pure NumPy Canny baseline with visualisation
  pure_numpy_fixed .ipynb         ← Fixed pure-NumPy version (debugging reference)

src/gpu/
  ── Pipeline ──────────────────────────────────────────────────────────────────
  cutile_full_pipeline.py         End-to-end Canny: cuTile stages 1-3,
                                  GPU or CPU stage 4-5; CLI entry point

  ── cuTile kernels & benchmarks ───────────────────────────────────────────────
  sobel_cutile_benchmark.py       Three Sobel variants:
                                    • launch_sobel_cutile      (old: 8 full copies)
                                    • launch_sobel_fused_cutile (fused, still copies)
                                    • launch_sobel_row_view    ← pipeline uses this
                                  Tile-size sweep; produces report/17*, 29*, 31* CSVs

  nms_cutile_benchmark.py         Two NMS variants:
                                    • launch_nms_cutile        (old: 9 full copies)
                                    • launch_nms_row_view      ← pipeline uses this
                                  Tile-size sweep; produces report/27*, 32* CSVs

  ── CuPy baseline benchmarks ──────────────────────────────────────────────────
  gaussian_benchmark.py           CuPy Gaussian vs CPU; load_grayscale_image,
                                  make_gaussian_kernel (shared utilities)
  sobel_benchmark.py              CuPy Sobel vs CPU; normalize_to_uint8 utility
  nms_cupy_benchmark.py           CuPy NMS vs CPU
  canny_frontend_benchmark.py     Gaussian + Sobel frontend benchmark
  cutile_canny_pipeline_benchmark.py  Gaussian + Sobel + NMS pipeline benchmark;
                                  produces report/19* CSVs
  complete_canny_benchmark.py     Full pipeline (all 5 stages) CPU vs GPU benchmark;
                                  produces report/22* CSVs

  ── Video demos ───────────────────────────────────────────────────────────────
  video_stream_demo.py            Sequential per-frame Canny demo
                                  (camera / video file / image-loop);
                                  produces report/25* CSVs
  video_stream_pipeline_demo.py   Cross-frame pipelined demo: GPU frontend thread
                                  overlaps CPU postprocess thread;
                                  exposes run_pipeline() used by the notebook

  ── Learning / reference ──────────────────────────────────────────────────────
  vector_add_cutile.py            Minimal cuTile hello-world (vector add)

bsds500_canny_test.py             Correctness test: runs the pipeline on all 200
                                  BSDS500 test images and compares against OpenCV Canny

data/
  test.jpg        512×512 test image (tile-aligned; used for quick benchmarks)
  IMG_6860.JPG    3024×4032 large image (12.2 MP; used for large-scale benchmarks)
  size.jpg        Reference photo for the controlled size-sweep experiment (§3 notebook)

report/
  ── Benchmark CSVs ────────────────────────────────────────────────────────────
  17_sobel_cutile_tile_sweep_*    cuTile Sobel (old copy-based) tile-size sweep
  27_nms_cutile_tile_sweep_*      cuTile NMS  (old copy-based) tile-size sweep
  29_sobel_fused_tile_sweep_*     Fused Sobel: 2-pass vs fused vs CuPy
  31_sobel_row_view_tile_sweep_*  Row-view Sobel vs CuPy  ← current implementation
  32_nms_row_view_tile_sweep_*    Row-view NMS vs CuPy    ← current implementation
  19_cutile_canny_pipeline_*      Gaussian+Sobel+NMS frontend speedup
  22_complete_canny_benchmark_*   Full pipeline CPU vs GPU+CPU hybrid vs full-GPU
  25_video_stream_fps_*           Sequential video stream FPS
  26_video_stream_pipelined_*     Cross-frame pipelined video stream FPS
  30_bsds500_gpu_benchmark.csv    BSDS500 per-image GPU hybrid speedup (200 images)
  ── Output images ─────────────────────────────────────────────────────────────
  04_suppressed.png / 05_final_edges.png   NMS and final edge maps
  06-08_sobel_*.png                        Sobel magnitude comparisons
  15-16_nms_*edges_*.png                   NMS CPU vs GPU edges
  18_sobel_cutile_magnitude_*.png          Best-tile cuTile Sobel output
  28_nms_cutile_edges_*.png                Best-tile cuTile NMS output
  30_sobel_fused_magnitude_*.png           Fused Sobel output
```

---

## Quick start

```powershell
# Activate the virtual environment (Windows)
.venv\Scripts\activate        # or: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# Run the complete cuTile pipeline on a single image (displays result)
$env:PYTHONPATH = "src/gpu"
python src/gpu/cutile_full_pipeline.py --image data/test.jpg

# Benchmark mode: 10 timed runs, no GUI, save edge output
python src/gpu/cutile_full_pipeline.py --image data/test.jpg --runs 10 --no-display --output report/edges.png

# Try a different tile size (64, 128, 256, 512 are tested in the notebooks)
python src/gpu/cutile_full_pipeline.py --image data/IMG_6860.JPG --tile-size 128 --no-display
```

---

## Running benchmarks

All benchmark scripts run from the repo root with `PYTHONPATH=src/gpu`.

```powershell
$env:PYTHONPATH = "src/gpu"

# Sobel cuTile tile-size sweep (produces report/17*, 29*, 31* CSVs + PNGs)
python src/gpu/sobel_cutile_benchmark.py --image data/IMG_6860.JPG --tile-sizes 64,128,256,512

# NMS cuTile tile-size sweep (produces report/27*, 32* CSVs + PNGs)
python src/gpu/nms_cutile_benchmark.py --image data/IMG_6860.JPG --tile-sizes 64,128,256,512

# Gaussian + Sobel + NMS pipeline sweep (produces report/19* CSVs)
python src/gpu/cutile_canny_pipeline_benchmark.py --image data/IMG_6860.JPG

# Full pipeline benchmark — CPU vs GPU+CPU hybrid vs full-GPU (produces report/22* CSVs)
python src/gpu/complete_canny_benchmark.py --image data/test.jpg

# BSDS500 correctness + speed test (downloads dataset automatically)
python bsds500_canny_test.py
```

---

## Video stream demos

```powershell
$env:PYTHONPATH = "src/gpu"

# Sequential demo — webcam
python src/gpu/video_stream_demo.py --source 0 --max-frames 100

# Sequential demo — image loop (no webcam needed)
python src/gpu/video_stream_demo.py --image-loop data/test.jpg --max-frames 50 --no-display

# Cross-frame pipelined demo — image loop
python src/gpu/video_stream_pipeline_demo.py --image-loop data/test.jpg --max-frames 100 --no-display --results-csv report/pipeline_test.csv
```

---

## Analysis notebooks

### `notebooks/evaluate_performance_en.ipynb` — primary notebook

**Run all cells from top to bottom.** Every figure is generated live; no pre-existing CSV is
read. All stages are exercised through the project's own functions — nothing is re-implemented
inside the notebook.

The notebook has 9 sections:

| § | Title | What is evaluated |
|:---|:---|:---|
| 1 | **Pixel-level data parallelism** | Structural relationship between image size, tile size, and `N_blocks`; plotted from actual image pixel counts |
| 2 | **Tile size as the parallel-granularity knob** | All three cuTile stages (Gaussian, Sobel, NMS) swept over tile sizes 64–1024 on a 1080p frame; each compared against its CuPy baseline; best tile size per stage identified |
| 3 | **Image size vs GPU frontend speedup** | One real image (`data/size.jpg`) resized to 480p / 720p / 1080p / 4K — content held constant, only pixel count varies; `benchmark_cpu_pipeline` vs `benchmark_gpu_pipeline_for_tile_size`; shows speedup growing from ~3× to ~20× compute-only |
| 4 | **Per-stage and whole-frontend CuPy vs cuTile** | Synthetic images 512²–8192² (0.26–67 MP), including non-tile-aligned widths (1500, 3000); Gaussian / Sobel / NMS individually, plus summed whole-frontend; correctness verified (max diff ≈ 0 at every size); OOM-guarded |
| 5 | **Amdahl's law — serial hysteresis bottleneck** | `complete_canny_benchmark` functions time GPU frontend + CPU post-processing; serial fraction *f* measured live; measured speedup points plotted on the Amdahl curve; shows the CPU hysteresis caps end-to-end speedup |
| 6 | **BSDS500 dataset benchmark** | Up to 200 real photographs (upscaled 4× to ~2.5 MP so the GPU is not overhead-bound); CPU total vs GPU+CPU hybrid; speedup distribution histogram; Amdahl curve with measured mean point |
| 7 | **Sequential vs cross-frame pipelined video** | Uses a real video in `data/` if present, otherwise image-loop; sequential: `video_stream_demo.process_frame`; pipelined: `video_stream_pipeline_demo.run_pipeline`; measures GPU and CPU stage times per frame; compares measured FPS to theory `(t_gpu+t_cpu)/max(t_gpu,t_cpu)` |
| 8 | **GPU stage 4-5 (our improvement)** | Directly compares `run_full_pipeline(..., gpu_hysteresis=False)` (CPU OpenCV connected components) vs `run_full_pipeline(..., gpu_hysteresis=True)` (GPU `cupyx.scipy.ndimage.label`); measures stage-4/5 cost separately and total pipeline FPS; shows that moving hysteresis to GPU removes the Amdahl bottleneck from §5 |
| 9 | **Summary table** | All six parallelism axes collected |

### `notebooks/evaluate_performance.ipynb` — Chinese companion

Same sections as the English notebook, but reads pre-computed CSVs from `report/` for its plots
(rather than running live benchmarks). Also contains an additional `§3.5` cell (added during
development) that sweeps all three stages across synthetic image sizes and generates the same
table and plots as §4 of the English notebook, produced live.

### `notebooks/01_canny_baseline 1.ipynb`

Pure NumPy Canny implementation with step-by-step intermediate visualisations. Educational
reference; no GPU required.

---

## cuTile implementation notes

### Row-view data layout (current, used in pipeline)

Both Sobel and NMS previously allocated 8–9 separate contiguous copies of the image interior
(one per neighbour direction), costing ~77% of total runtime on copies alone. The current
implementation uses a **row-view layout**:

1. Pad the image once: `cp.pad(img, ((1, 2), (1, 1)))` — one allocation.
2. Extract three contiguous row-range views (`top_flat / ctr_flat / bot_flat`) — `ravel()` on
   a C-contiguous slice is a view, zero copy.
3. Eight (Sobel) or nine (NMS) column-offset slices (`flat[0:] / flat[1:] / flat[2:]`) are
   also views — pointer offset only, zero copy.
4. The same `_nms_cutile_kernel` / `sobel_fused_from_neighbors_cutile` kernel is reused
   unchanged.

The extra bottom-padding row (`((1,2),...)` instead of `((1,1),...))`) guards the `+2` column
offset of the bottom-right view from reading past the allocation on the final tile (UB fix).

Angle padding in NMS row-view uses **identical padding dimensions** as magnitude so both arrays
share the same row stride `Wp`. Mismatched strides caused silent output corruption on
tile-aligned image widths (e.g. 512×512) — fixed and verified (max diff = 0 on all sizes).

### Gaussian

The separable k=5 Gaussian runs two cuTile passes (horizontal then vertical). Each pass
pre-extracts `k=5` shifted 1D views and computes the weighted sum tile-by-tile. Widths that are
not multiples of `tile_size` trigger an extra column-padding copy; k≠5 falls back to CuPy.

### Hardware requirement

cuTile kernels require Turing architecture (RTX 20-series) or newer. The pipeline falls back
silently to CuPy on older GPUs. Tested on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB).

---

## Benchmark results (representative, tile_size=256, RTX 3050 Laptop)

| Image | Stage | CuPy | cuTile (row-view) | Speedup |
|:---|:---|:---|:---|:---|
| 1024² (1 MP) | Gaussian | 1.31 ms | 0.62 ms | **2.1×** |
| 1024² (1 MP) | Sobel | 1.58 ms | 0.79 ms | **2.0×** |
| 1024² (1 MP) | NMS | 1.16 ms | 0.56 ms | **2.1×** |
| 4096² (17 MP) | Gaussian | 19.3 ms | 8.9 ms | **2.2×** |
| 4096² (17 MP) | Sobel | 23.6 ms | 11.0 ms | **2.1×** |
| 4096² (17 MP) | NMS | 17.3 ms | 7.4 ms | **2.3×** |

All max-diff values vs CuPy are 0.0000 (bit-exact on these sizes).

Video stream (image-loop, 512-wide frames, 100 frames):

| Mode | FPS |
|:---|:---|
| Sequential (`video_stream_demo.py`) | ~10 FPS |
| Cross-frame pipelined | ~40–80 FPS (theory: `1/max(t_gpu, t_cpu)`) |

---

## Dependencies

```
numpy
scipy
matplotlib
opencv-python
cupy-cuda12x      # match your CUDA version
cuda-python       # provides cuda.tile
cupyx             # bundled with cupy
kagglehub         # optional, for BSDS500 download
```

Install: `pip install -r requirement.txt`
