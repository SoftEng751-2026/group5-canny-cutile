# Dependency Analysis: GPU-Accelerated Canny Edge Detection

**Project**: group5-canny-cutile — SoftEng 751, Group 5, 2026  
**Authors**: Jiaxi Liu, Xudong Ma, Shiying Yang

---

## Part 1: Edge Detection Steps — Data Dependency Analysis

### 1.1 The Five-Stage Canny Pipeline

The Canny edge detection algorithm consists of five sequential stages. Each stage consumes the output of the previous one, establishing a chain of **inter-stage data dependencies** that prevent the stages themselves from running in parallel with each other. However, within each stage there exist rich **intra-stage data-parallel** opportunities that cuTile exploits directly.

```
Raw Image (H × W float32)
       │
       ▼
┌─────────────────────────┐
│  Stage 1: Gaussian Blur │  ← Eliminates noise
└─────────────────────────┘
       │  blurred (H × W)
       ▼
┌──────────────────────────┐
│  Stage 2: Sobel Gradient │  ← Detects intensity change
└──────────────────────────┘
       │  magnitude + gx + gy (H × W each)
       ▼
┌────────────────────────────────────────┐
│  Stage 3: Non-Maximum Suppression (NMS)│  ← Thins edges to 1 px
└────────────────────────────────────────┘
       │  thinned magnitude (H × W)
       ▼
┌────────────────────────────┐
│  Stage 4: Double Threshold │  ← Classifies strong/weak/none
└────────────────────────────┘
       │  strong + weak masks (H × W bool each)
       ▼
┌────────────────────────────┐
│  Stage 5: Hysteresis       │  ← Connects weak to strong edges
└────────────────────────────┘
       │
       ▼
  Edge Map (H × W bool)
```

### 1.2 Inter-Stage Dependencies

The table below records the precise data contracts between stages. Because every stage reads the **entire output** of the previous stage before it can produce any output, these dependencies are **strict sequential** — there is no room to overlap stage K with stage K+1 within a single frame.

| Stage | Reads | Writes | Dependency Type |
|:------|:------|:-------|:----------------|
| 1 — Gaussian Blur | raw image `I[r,c]` | `G[r,c]` (blurred) | Full-image read-after-write from raw |
| 2 — Sobel Gradient | `G[r,c]` | `Mag[r,c]`, `Gx[r,c]`, `Gy[r,c]` | Full-image RAW from Stage 1 |
| 3 — NMS | `Mag[r,c]`, `Gx[r,c]`, `Gy[r,c]` | `NMS[r,c]` | Full-image RAW from Stage 2 |
| 4 — Double Threshold | `NMS[r,c]` | `strong[r,c]`, `weak[r,c]` | Full-image RAW from Stage 3 |
| 5 — Hysteresis | `strong[r,c]`, `weak[r,c]` | `edges[r,c]` | Full-image RAW from Stage 4 |

> **RAW = Read-After-Write.** Stage K+1 must not read any pixel until Stage K has finished writing all pixels. This is the fundamental constraint that makes the five stages run sequentially per frame.

### 1.3 Intra-Stage Dependencies and cuTile Parallelization

Although stages must run sequentially, *within* each stage most pixel computations are **independent of each other** — a pixel at `(r, c)` depends only on a small local neighbourhood, not on any other output pixel being computed in the same stage. This is the independence structure that cuTile exploits.

#### Stage 1 — Gaussian Blur

**Algorithm**: Separable 2-D convolution decomposed into two 1-D passes (horizontal then vertical). Kernel size k = 5, σ = 1.4.

**Intra-stage dependencies**:

- *Horizontal pass*: output pixel `H[r,c]` depends only on `I[r, c-2 .. c+2]` — five pixels in the same row. No dependency on any other row. Rows are fully independent of each other.
- *Vertical pass*: output pixel `G[r,c]` depends only on `H[r-2 .. r+2, c]` — five pixels in the same column. No dependency on any other column.
- Within each pass, every output pixel is **embarrassingly parallel**: zero write-after-read conflicts across the output array.

**cuTile parallelization** (`_gauss_k5` in `cutile_full_pipeline.py:33–69`):

- The image is padded once per pass, then five zero-copy column-offset (horizontal pass) or row-offset (vertical pass) views are extracted.
- A single cuTile kernel launch creates `N_blocks = (H × W_padded) / tile_size` independent blocks, each computing one contiguous tile of the flattened output row.
- All blocks run concurrently on GPU streaming multiprocessors (SMs); no inter-block communication is needed because the stencil radius (2) is fully captured in the padded input views passed to the kernel.

```
Horizontal pass kernel receives 5 shifted views of the padded row:
  n0[i] = padded[r, c-2]   (view offset 0)
  n1[i] = padded[r, c-1]   (view offset 1)
  ...
  n4[i] = padded[r, c+2]   (view offset 4)

Each block b computes: out[b*tile:(b+1)*tile] = Σ w_k * n_k[b*tile:(b+1)*tile]
All blocks are independent → fully parallel on GPU.
```

**What cuTile enables**: The `ct.kernel` decorator compiles the Python-level stencil expression directly to a GPU kernel; `ct.load` / `ct.store` map to vectorised SIMD loads/stores on the GPU's warp. No manual thread-indexing is required, and the kernel is portable across Turing / Ampere / Ada GPUs.

---

#### Stage 2 — Sobel Gradient

**Algorithm**: 3×3 Sobel operators applied to the blurred image to compute gradient magnitude, Gx, and Gy simultaneously.

**Intra-stage dependencies**:

- Output pixel `(Mag, Gx, Gy)[r,c]` depends on a 3×3 neighbourhood of `G`: the 8 pixels at offsets `{-1, 0, +1} × {-1, 0, +1}` around `(r, c)`.
- No output pixel depends on any other output pixel being computed in this stage. The stage is **embarrassingly parallel** at pixel granularity.

**cuTile parallelization** (`launch_sobel_row_view` in `sobel_cutile_benchmark.py:315–355`):

The key optimization is the **row-view layout** that avoids copying the padded array multiple times:

```
Step 1 — Pad once:
  padded = cp.pad(G, ((1, 2), (1, 1)), mode="edge")   # 1 HW-element copy

Step 2 — Extract 3 contiguous row-range views (zero copy, pointer offset):
  top_flat = padded[0:H,   :].ravel()   # rows 0..H-1
  ctr_flat = padded[1:H+1, :].ravel()   # rows 1..H
  bot_flat = padded[2:H+2, :].ravel()   # rows 2..H+1

Step 3 — Extract 8 column-offset slices (zero copy, pointer offset):
  top_left   = top_flat[0:],  top_center = top_flat[1:],  top_right   = top_flat[2:]
  mid_left   = ctr_flat[0:],               mid_right   = ctr_flat[2:]
  bot_left   = bot_flat[0:],  bot_center = bot_flat[1:],  bot_right   = bot_flat[2:]

Step 4 — Fused kernel writes all 3 outputs in one pass:
  gx  = -tl + tr - 2*ml + 2*mr - bl + br
  gy  =  tl + 2*tc + tr - bl - 2*bc - br
  mag = sqrt(gx² + gy²)
```

This reduces memory bandwidth from ~8–9 full-image copies (old approach) to ~1 pad copy, a **~5× bandwidth reduction**. The kernel `sobel_fused_from_neighbors_cutile` (`sobel_cutile_benchmark.py:158–198`) receives all 8 neighbour views and writes `magnitude`, `gx`, `gy` in a single kernel launch — eliminating two extra passes over global memory.

**Parallelism**: `N_blocks = H × (W_padded / tile_size)` blocks run concurrently. Each block is data-independent from every other block.

---

#### Stage 3 — Non-Maximum Suppression (NMS)

**Algorithm**: For each pixel, retain the magnitude only if it is a local maximum along the gradient direction (one of four quantised directions: 0°, 45°, 90°, 135°). Otherwise suppress to zero.

**Intra-stage dependencies**:

- Output pixel `NMS[r,c]` depends on `Mag[r,c]` (centre), the two neighbours along the gradient direction from Stage 2, and the angle array `Ang[r,c]` computed from `Gx`, `Gy`.
- In the worst case (diagonal direction), neighbours at `(r-1, c+1)` and `(r+1, c-1)` are accessed. The **read footprint** is a subset of the 3×3 neighbourhood.
- No output pixel depends on another output pixel. The stage is **embarrassingly parallel** at pixel granularity.

**cuTile parallelization** (`launch_nms_row_view` in `nms_cutile_benchmark.py:208–277`):

Identical row-view layout to Sobel, but now 9 views of the magnitude array (centre + all 8 neighbours) plus the angle array are passed to the kernel:

```
Pad magnitude + angle arrays identically (matching row strides).
Extract 9 magnitude views:  mag_c, mag_l, mag_r, mag_t, mag_b,
                             mag_tl, mag_tr, mag_bl, mag_br
Extract 1 angle view:       ang_c

Kernel branch:
  d0   → compare mag_c vs mag_l,  mag_r
  d45  → compare mag_c vs mag_tr, mag_bl
  d90  → compare mag_c vs mag_t,  mag_b
  d135 → compare mag_c vs mag_tl, mag_br

  keep = (d0 & mc≥ml & mc≥mr) | (d45 & mc≥mtr & mc≥mbl) | ...
  output = where(keep, mc, 0.0)
```

Memory bandwidth: ~2 HW copies (1 for mag pad + 1 for angle pad) versus ~10 HW copies in the old approach.

**An important subtlety**: the angle array must be padded with the same row stride as the magnitude array. If padding is not matched, the column-offset slices for angle and magnitude will be misaligned on tile-width boundaries, causing silent numerical corruption. This is handled explicitly in the implementation (`nms_cutile_benchmark.py:208–230`).

---

#### Stage 4 — Double Threshold

**Algorithm**: Classify each NMS pixel as *strong* (above `high`), *weak* (between `low` and `high`), or *non-edge* (below `low`). Thresholds are computed as percentiles of the NMS magnitude histogram.

**Intra-stage dependencies**:

- **Percentile computation**: Requires a global reduction over all NMS pixels — a serial dependency that prevents pixel-level parallelism in the threshold-selection step.
- **Classification**: Once thresholds are fixed, every pixel is classified independently. The classification step is **embarrassingly parallel**.

**cuTile parallelization**:

- Threshold selection uses `cp.percentile(NMS, [low_pct, high_pct])` — CuPy's parallel histogram/sort kernel handles the reduction in one GPU call.
- Classification uses vectorised CuPy comparison operators, which are internally parallelised across SMs.
- No cuTile custom kernel is needed here because CuPy's built-in operations are already optimal for this pattern.

---

#### Stage 5 — Hysteresis

**Algorithm**: Retain a weak edge pixel if and only if it is 8-connected to at least one strong edge pixel (possibly via a chain of other weak pixels). This is essentially a connected-components labelling problem.

**Intra-stage dependencies**:

- This stage has **global data dependencies**: whether pixel `(r, c)` is kept depends on the entire connected component it belongs to, which may span the full image.
- Iterative flood-fill is inherently sequential in the general case.

**cuTile / GPU parallelization**:

- The project uses `cupyx.scipy.ndimage.label` (GPU-accelerated connected-components labelling) to perform hysteresis entirely on the GPU.
- GPU connected-components algorithms exploit **BFS-like wavefront parallelism**: each BFS frontier level is processed in parallel across SMs. While not embarrassingly parallel, the GPU version is substantially faster than a CPU flood-fill for large images.
- The original CPU implementation (`cv2.connectedComponents`) was identified as an **Amdahl bottleneck**: for large images the CPU hysteresis time (~100+ ms) dominated the GPU frontend time (~5–30 ms), capping the overall speedup at ~2–5× regardless of the per-stage GPU speedups. Moving hysteresis to the GPU removes this serial fraction.

---

### 1.4 Dependency Summary Table

| Stage | Parallelism Type | cuTile Used? | GPU Mechanism | Serial Fraction |
|:------|:----------------|:-------------|:--------------|:----------------|
| 1 — Gaussian | Embarrassingly parallel (separable 1-D stencils) | **Yes** — `_gauss_k5` kernel | N_blocks = pixels/tile_size; 5 zero-copy views | None |
| 2 — Sobel | Embarrassingly parallel (3×3 stencil, 3 outputs) | **Yes** — `sobel_fused_from_neighbors_cutile` | 1 pad + 8 zero-copy views; fused 3-output write | None |
| 3 — NMS | Embarrassingly parallel (directional local max) | **Yes** — `_nms_cutile_kernel` | 1 pad + 9 zero-copy views; angle-aligned padding | None |
| 4 — Double Threshold | Reduction (percentile) + embarrassingly parallel (classify) | No — CuPy | `cp.percentile` + vectorised comparison | Percentile reduction (small) |
| 5 — Hysteresis | Global (connected components) | No — CuPy/cupyx | `cupyx.scipy.ndimage.label` BFS wavefront | Moderate (bounded by BFS depth) |

### 1.5 Amdahl's Law Perspective

Let `t_GPU` = GPU frontend time (Stages 1–3) and `t_serial` = CPU hysteresis time (Stage 5). The serial fraction is:

```
f = t_serial / (t_GPU + t_serial)
```

Maximum achievable speedup (Amdahl):

```
S_max = 1 / (f + (1-f)/S_parallel)
```

With `t_serial ≈ 100 ms` (CPU hysteresis on 4K image) and `t_GPU ≈ 30 ms`, `f ≈ 0.77`. Even with an infinite GPU speedup on stages 1–3, `S_max ≈ 1.3×`. Moving hysteresis to `cupyx` reduces `t_serial` to ~5–10 ms, bringing `f` below 0.2 and allowing `S_max > 4×` for the full pipeline.

---

## Part 2: Video Stream Pipeline — Parallelization Strategies

### 2.1 Sequential Baseline

The simplest approach (`video_stream_demo.py`) processes frames **one at a time**:

```
Frame 0: [Decode] → [Gaussian] → [Sobel] → [NMS] → [Threshold+Hysteresis] → [Display]
Frame 1:                                                                        [Decode] → ...
```

Throughput is limited by the sum of all stage latencies per frame:

```
FPS_seq = 1 / (t_decode + t_Gauss + t_Sobel + t_NMS + t_thresh + t_hysteresis)
```

On typical hardware (RTX 3050 Laptop, 1080p frames) this yields ~10 FPS — insufficient for real-time video at 30 FPS.

### 2.2 Cross-Frame Pipeline Parallelism (Producer–Consumer)

The core insight is that frame K+1's GPU work and frame K's CPU post-processing **access disjoint data** and can therefore run on separate hardware resources simultaneously: the GPU handles frame K+1 while the CPU handles frame K.

#### 2.2.1 Two-Stage Cross-Frame Pipeline

Implemented in `video_stream_pipeline_demo.py` using three Python threads and two bounded queues:

```
                   ┌──────────────┐     input_queue (maxsize=4)
Thread: Producer   │ Decode frame │──────────────────────────────────────►
                   └──────────────┘                                        │
                                                                           ▼
                                              ┌──────────────────────────────────────────┐
Thread: GPU worker                            │ Resize + Grayscale + Gaussian + Sobel +  │
  (frame K+1)                                 │ NMS (all cuTile)  +  H→D transfer only   │
                                              └──────────────────────────────────────────┘
                                                           │  post_queue (maxsize=2)
                                                           ▼
                                        ┌────────────────────────────────┐
Thread: CPU worker                      │ Threshold + Hysteresis (CPU)   │
  (frame K)                             │ → edge map ready for display   │
                                        └────────────────────────────────┘
```

**Dependency constraints satisfied**:

1. The GPU worker reads from `input_queue` and writes to `post_queue`. It never touches `result_queue` directly.
2. The CPU worker reads from `post_queue` and writes to `result_queue`. It only reads the NMS output that the GPU worker already finished writing — the bounded queue acts as the synchronisation barrier.
3. The `post_queue` is bounded (`maxsize=2`) so the GPU worker applies back-pressure if the CPU worker falls behind, preventing unbounded memory growth.

**Throughput** (with pipelining):

```
FPS_pipeline ≈ 1 / max(t_GPU_frontend, t_CPU_postprocess)
```

When the GPU frontend time and CPU post-processing time are balanced, this approaches twice the sequential throughput. On typical hardware the pipelined mode achieves **40–80 FPS** at 1080p, compared to ~10 FPS sequential.

**Latency** is slightly higher than sequential (one frame of pipeline delay), but throughput is the primary metric for video processing.

#### 2.2.2 Full-GPU Mode (No CPU Post-Processing Stage)

A second design in the same file (`gpu_full_worker`) moves threshold and hysteresis entirely to the GPU (`cupyx.scipy.ndimage.label`), collapsing the two pipeline stages back into one. The pipeline becomes:

```
Thread: Producer ──► input_queue ──► Thread: GPU worker (all 5 stages on GPU) ──► result_queue ──► Display
```

With no CPU stage to overlap, pipelining gain comes only from overlapping frame **decode** with GPU processing. However, the GPU latency per frame is much lower because:
- `t_hysteresis_GPU ≈ 5–10 ms` vs `t_hysteresis_CPU ≈ 100 ms`
- The single GPU thread can process frames at ~100–200 FPS when the GPU is not the bottleneck.

This mode is preferable when the CPU is also needed for other workloads (e.g., UI rendering, audio) and when the GPU is powerful enough that hysteresis is cheap.

### 2.3 Potential Further Parallelization Opportunities

#### 2.3.1 Multi-Stream GPU Concurrency (CUDA Streams)

The current implementation uses the default CUDA stream (`Stream.null`), which serialises all GPU operations. Launching Stage 1 of frame K+1 on a **separate CUDA stream** concurrently with Stage 5 of frame K (running on stream 0) would enable intra-GPU concurrency when SM occupancy is below 100%.

```
Stream 0:  [Gaussian K] ─► [Sobel K] ─► [NMS K] ─► [Thresh K] ─► [Hysteresis K]
Stream 1:              [Gaussian K+1] ─► ...
```

This requires that each frame's GPU allocations are independent (no aliasing), which is already guaranteed by the current design (each frame allocates fresh CuPy arrays). The main risk is memory pressure: running two frames on-GPU simultaneously doubles the working set.

#### 2.3.2 Spatial Tile Parallelism Within a Single Frame (cuTile Extension)

Stages 1–3 are already spatially parallel at the pixel level via cuTile blocks. A further opportunity is to split the image into large **spatial tiles** (e.g., quadrants) and process them on separate CUDA streams with independent SM assignments. This would allow the GPU scheduler to interleave the four quadrant kernels across SMs more evenly when SM occupancy would otherwise leave some SMs idle.

This is beneficial only for very large images (≥ 4K) where individual kernel launches saturate fewer than half the available SMs.

#### 2.3.3 Decode–Compute Overlap with GPU Video Decoding

The current decoder is `cv2.VideoCapture` (CPU-based). Replacing it with **NVDEC** (NVIDIA hardware video decoder, accessible via `cv2.VideoCapture` with the `FFMPEG` + CUDA backend or via `PyNvVideoCodec`) would:

- Decode frames directly into GPU memory, eliminating the CPU→GPU transfer.
- Allow the decode and the Gaussian kernel to overlap on different hardware units (video engine vs SM array).

The pipeline would become:

```
NVDEC engine:  [Decode K+1 → GPU buffer] (hardware unit, runs concurrently with SM work)
SM array:      [Gaussian K] ─► [Sobel K] ─► [NMS K] ─► ...
```

This removes the `frame_to_grayscale_float32` + `cp.asarray` cost (currently ~2–5 ms per 1080p frame) from the critical path.

#### 2.3.4 Multi-GPU Distribution

For throughput-critical applications (broadcast-quality 4K video at 60 FPS), frames can be distributed round-robin across multiple GPUs:

```
Frame 0 → GPU 0
Frame 1 → GPU 1
Frame 2 → GPU 0
Frame 3 → GPU 1
...
```

Each GPU runs the full 5-stage pipeline independently (no cross-GPU communication needed, since frames are independent). Throughput scales linearly with the number of GPUs. CuPy supports device selection via `cp.cuda.Device(i)`.

#### 2.3.5 Batch Processing (Multiple Frames Per Kernel Launch)

cuTile kernels could be extended to process a batch of B frames in a single launch by stacking frames along a third array dimension. The kernel's block grid would be `N_blocks = B × H × (W / tile_size)`. This amortises the kernel-launch overhead across B frames and allows the GPU's memory controller to coalesce accesses across frames.

This trades per-frame latency (higher: must wait for B frames) for throughput (lower per-frame kernel-launch overhead). Suitable for offline video processing but not for real-time streaming where low latency is required.

### 2.4 Pipeline Design Comparison

| Design | Threads | GPU Load | CPU Load | FPS (1080p, RTX 3050L) | Latency | Best For |
|:-------|:--------|:---------|:---------|:-----------------------|:--------|:---------|
| Sequential | 1 | Low | Low | ~10 | 1 frame | Debugging |
| Cross-frame (CPU post) | 3 | Medium | Medium | ~40–80 | 2 frames | Balanced CPU+GPU systems |
| Full-GPU + producer thread | 2 | High | Minimal | ~100+ | 1.5 frames | GPU-heavy workloads |
| Multi-stream (CUDA streams) | 2 | Very High | Low | ~150+ (theoretical) | 2 frames | High-end GPUs |
| Multi-GPU | N | Distributed | Low | N × single-GPU FPS | 1 frame/GPU | Servers |
| Batch cuTile | 1 | Very High | Low | High throughput, high latency | B frames | Offline |

### 2.5 Theoretical Maximum Throughput

The theoretical maximum throughput of the cross-frame pipeline is bounded by the slowest stage (the pipeline bottleneck). For 1080p on an RTX 3050 Laptop:

| Stage | Approx. Time | Thread |
|:------|:------------|:-------|
| Decode + resize | ~3 ms | Producer |
| Gaussian + Sobel + NMS (cuTile) | ~8 ms | GPU worker |
| Threshold + Hysteresis (CPU) | ~15–20 ms | CPU worker |

The CPU post-processing stage is the bottleneck. The theoretical maximum is:

```
FPS_max = 1 / max(3, 8, 20) ms = 50 FPS
```

Moving hysteresis to the GPU (`cupyx`) reduces the CPU stage to ~0 ms, making the GPU the bottleneck:

```
FPS_max = 1 / 8 ms ≈ 125 FPS
```

This is why the full-GPU mode (`gpu_full_worker`) achieves significantly higher throughput than the cross-frame pipeline mode in practice.

---

*Analysis based on source code in `src/gpu/`, performance data from `notebooks/evaluate_performance_en.ipynb`, and architectural description in `README.md`.*