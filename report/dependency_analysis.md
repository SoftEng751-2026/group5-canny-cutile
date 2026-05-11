# Dependency Analysis: Canny Edge Detection Pipeline

**Project:** Group 5 — Optimising Canny Edge Detection in cuTile  
**Author:** Shiying Yang  
**Date:** 2026-05-11

---

## 1. Introduction

This document analyses the data dependencies within and between every stage of
the Canny edge detection algorithm. Understanding these dependencies is the
foundation for deciding which stages can be parallelised, which require
synchronisation barriers, and how cuTile's tile-based execution model fits the
computation. The analysis covers five stages:

1. Gaussian Blur
2. Sobel Gradient (magnitude + angle)
3. Non-Maximum Suppression (NMS)
4. Double Thresholding
5. Hysteresis

---

## 2. Pipeline-Level (Inter-Stage) Dependencies

The five stages form a strict linear chain. Each stage reads the complete output
of the preceding stage before it can begin writing its own output. There are no
backward edges and no cycles.

```
Raw image
    │
    ▼
┌─────────────────┐
│  Gaussian Blur  │   consumes: raw image (H×W float32)
│  (smoothing)    │   produces: blurred image (H×W float32)
└────────┬────────┘
         │ blurred image
         ▼
┌─────────────────┐
│  Sobel Gradient │   consumes: blurred image (H×W float32)
│  magnitude+angle│   produces: magnitude (H×W float32)
│                 │             angle     (H×W float32)
└────────┬────────┘
         │ magnitude, angle
         ▼
┌─────────────────┐
│      NMS        │   consumes: magnitude (H×W float32)
│                 │             angle     (H×W float32)
│                 │   produces: thinned magnitude (H×W float32)
└────────┬────────┘
         │ thinned magnitude
         ▼
┌─────────────────┐
│ Double Threshold│   consumes: thinned magnitude (H×W float32)
│                 │             scalar low threshold
│                 │             scalar high threshold
│                 │   produces: strong mask (H×W bool)
│                 │             weak mask   (H×W bool)
└────────┬────────┘
         │ strong mask, weak mask
         ▼
┌─────────────────┐
│   Hysteresis    │   consumes: strong mask (H×W bool)
│                 │             weak mask   (H×W bool)
│                 │   produces: final edge map (H×W bool)
└─────────────────┘
```

**Consequence for implementation:** a global synchronisation barrier is required
between every pair of adjacent stages. No stage may begin until its predecessor
has finished writing all of its output. In a GPU pipeline, this synchronisation
is enforced by `cp.cuda.Stream.null.synchronize()` before transferring between
stages, or implicitly by CuPy kernel sequencing on the same stream.

---

## 3. Stage-by-Stage Data Dependency Analysis

### 3.1 Stage 1 — Gaussian Blur

**Purpose:** suppress high-frequency noise before gradient computation.

**Algorithm (separable 1D convolution):**  
The implementation uses two sequential 1D passes rather than a single 2D kernel.

- **Horizontal pass:** each output pixel `temp[r, c]` depends on input pixels
  `image[r, c-radius .. c+radius]` — a 1D window of `kernel_size` pixels along
  the same row.
- **Vertical pass:** each output pixel `blurred[r, c]` depends on intermediate
  pixels `temp[r-radius .. r+radius, c]` — a 1D window of `kernel_size` pixels
  along the same column.

**Intra-pass dependencies:**  
Within a single pass, every output pixel is independent of every other output
pixel at the same pass. Output `temp[r, c]` is a function of a fixed set of
*input* pixels only; it does not depend on any other *output* pixel in the same
pass. This makes the horizontal pass and the vertical pass each embarrassingly
parallel.

**Cross-pass dependency:**  
The vertical pass reads from `temp`, which is fully written by the horizontal
pass. The two passes must be sequentially ordered.

**Boundary handling:**  
Edge padding (`mode="edge"`) replicates border pixels before each pass,
eliminating conditional branching at image boundaries. All interior and border
output pixels follow the same computation path.

**Neighbourhood footprint:**

```
Horizontal pass:  output[r, c]  ←  input[r, c-radius .. c+radius]
                                    (1 × kernel_size window)

Vertical pass:    output[r, c]  ←  temp[r-radius .. r+radius, c]
                                    (kernel_size × 1 window)

Combined 2D footprint:            kernel_size × kernel_size region centred
                                  on (r, c), accessed in two passes
```

**Parallelism rating:** ★★★★★ — both passes are fully data-parallel across
all pixels.

---

### 3.2 Stage 2 — Sobel Gradient

**Purpose:** compute the gradient magnitude and direction at each pixel.

**Algorithm:** a 3×3 stencil applied to the blurred image.

The Sobel operator computes two convolutions simultaneously:

```
Gx kernel:          Gy kernel:
  -1  0  +1           +1  +2  +1
  -2  0  +2            0   0   0
  -1  0  +1           -1  -2  -1

Gx[r,c] = -B[r-1,c-1] + B[r-1,c+1]
           -2·B[r,c-1]  + 2·B[r,c+1]
           -B[r+1,c-1] + B[r+1,c+1]

Gy[r,c] =  B[r-1,c-1] + 2·B[r-1,c] + B[r-1,c+1]
           -B[r+1,c-1] - 2·B[r+1,c] - B[r+1,c+1]

magnitude[r,c] = sqrt(Gx² + Gy²)
angle[r,c]     = arctan2(Gy, Gx) × (180/π)  mod 180°
```

where `B` is the blurred image from Stage 1.

**Intra-stage dependencies:**  
Each output pixel `(r, c)` reads exactly 8 neighbours of `B` (corners and
horizontal/vertical neighbours; the centre pixel cancels in both kernels) and
writes to `magnitude[r, c]` and `angle[r, c]`. No output pixel depends on any
other output pixel within this stage. The computation is embarrassingly parallel
across all interior pixels.

**Boundary pixels:**  
Interior is `[1:-1, 1:-1]` (one-pixel border). Border pixels are left as zero,
matching the CPU reference. This means the 3×3 stencil never reads out-of-bounds.

**cuTile mapping:**  
The cuTile kernel (`sobel_magnitude_from_neighbors_cutile`) pre-extracts the
eight 3×3 neighbour arrays by slicing the blurred image into eight shifted
views, flattens them, and passes them as separate input arrays. Each cuTile tile
processes `tile_size` consecutive interior pixels. Because the neighbour arrays
are constructed *before* the kernel launch, there are **no inter-tile data
dependencies** inside the cuTile kernel itself — every tile operates on
independent, pre-fetched data.

**Neighbourhood footprint:**

```
output[r, c]  ←  blurred[r-1, c-1], blurred[r-1, c], blurred[r-1, c+1]
                  blurred[r,   c-1],                   blurred[r,   c+1]
                  blurred[r+1, c-1], blurred[r+1, c], blurred[r+1, c+1]
                  (3×3 region, centre not used)
```

**Parallelism rating:** ★★★★★ — fully data-parallel across all interior pixels.

---

### 3.3 Stage 3 — Non-Maximum Suppression (NMS)

**Purpose:** thin edges to one pixel wide by suppressing non-local-maximum
gradient pixels.

**Algorithm:** for each interior pixel, compare its magnitude to two neighbours
along the quantised gradient direction. If the pixel is not a local maximum,
suppress it to zero.

The gradient direction is quantised into four bins:

| Angle range              | Comparison direction | Neighbours compared |
|--------------------------|----------------------|---------------------|
| 0–22.5° or 157.5–180°   | Horizontal (0°)      | left, right         |
| 22.5–67.5°              | Diagonal (45°)       | top-right, bottom-left |
| 67.5–112.5°             | Vertical (90°)       | top, bottom         |
| 112.5–157.5°            | Diagonal (135°)      | top-left, bottom-right |

**Intra-stage dependencies:**  
Each output pixel `nms[r, c]` reads `magnitude[r, c]` and two neighbours from
`magnitude`, plus `angle[r, c]`. It does not read from any other output pixel.
The computation is embarrassingly parallel across all interior pixels.

**Key observation — read-only stencil on magnitude:**  
Unlike Gaussian blur or Sobel, NMS does not accumulate into the output from
multiple source pixels; it makes a single binary keep/suppress decision per
pixel. The neighbour reads are all from the *input* `magnitude` array, not the
*output* array, so there is no write-after-read hazard within the stage.

**Neighbourhood footprint:**

```
output[r, c]  ←  magnitude[r, c]           (centre)
                  + 2 neighbours selected by angle[r, c]
                  (from the 8-connected neighbourhood)
```

**Parallelism rating:** ★★★★★ — fully data-parallel across all interior pixels.

---

### 3.4 Stage 4 — Double Thresholding

**Purpose:** classify each surviving NMS pixel as either a *strong* edge or a
*weak* (candidate) edge using two scalar thresholds.

**Algorithm:**

```
strong[r, c] = (nms[r, c] >= high_threshold)
weak[r, c]   = (nms[r, c] >= low_threshold)  AND  NOT strong[r, c]
```

The two thresholds can be determined in two ways:
- **Percentile mode** (default): compute the *p*-th percentile of the
  positive-valued NMS pixels, set `high = percentile`, `low = ratio × high`.
- **Fixed mode**: supply explicit scalar values.

**Dependency on NMS output:**  
In percentile mode, the thresholds cannot be computed until the entire NMS
output is available. This introduces a data reduction step (computing a
percentile over all positive pixels) before the pixel-wise classification can
begin. This reduction is itself parallelisable but requires a final aggregation
step.

**Intra-stage dependencies:**  
Once the scalar thresholds are known, each pixel's classification is
independent of every other pixel. The classification is embarrassingly parallel.

**Parallelism rating (classification):** ★★★★★  
**Parallelism rating (threshold computation):** ★★★☆☆ — requires a parallel
reduction, introducing a synchronisation point.

---

### 3.5 Stage 5 — Hysteresis

**Purpose:** keep weak edges that are connected to at least one strong edge,
discard all isolated weak edges.

**Algorithm:** breadth-first search (BFS) starting from all strong-edge pixels,
propagating to 8-connected weak-edge neighbours.

```python
edges = strong.copy()
queue = deque(all strong pixel positions)

while queue:
    (r, c) = queue.popleft()
    for each 8-neighbour (nr, nc):
        if weak[nr, nc] and not edges[nr, nc]:
            edges[nr, nc] = True
            queue.append((nr, nc))
```

An alternative implementation (`video_stream_demo.py`) uses
`cv2.connectedComponents` to label all candidate edge components (strong ∪ weak)
and then keeps only components that contain at least one strong pixel.

**Intra-stage dependencies:**  
This is the only stage with **true data-flow dependencies within the stage**.
The output `edges[nr, nc]` written in one BFS step is immediately read by the
next step to decide whether to enqueue further neighbours. The computation
cannot be trivially split into independent pixel-parallel tasks because:

- Whether a weak pixel is kept depends on the BFS frontier, which evolves
  dynamically as the algorithm runs.
- In the worst case, an edge chain of length *L* requires *L* sequential BFS
  iterations before the last pixel can be determined.

**Read-after-write (RAW) hazard:**  
`edges[nr, nc] = True` is written in step *k*; the check `not edges[nr, nc]`
in step *k+1* must see this write. This creates a strict ordering between BFS
iterations that prevents naive data parallelism.

**Why GPU parallelisation is hard:**  
A naïve pixel-parallel GPU kernel that reads `edges` and writes `edges`
simultaneously would produce race conditions. Correct parallel implementations
typically use:
1. **Multi-pass label propagation** — iteratively update a labelling array until
   convergence (each pass is parallel; convergence requires O(diameter) passes).
2. **Connected-components algorithms** — run a GPU-parallel connected-components
   on the union of strong and weak pixels, then filter components by whether
   they contain a strong pixel (the approach used in `video_stream_demo.py`).
3. **Work-queue approaches** — atomic operations and GPU work queues, which are
   complex to implement in cuTile's current programming model.

**Current implementation choice:**  
Both current CPU implementations avoid the race condition by using a serial BFS
(`complete_canny_benchmark.py`) or `cv2.connectedComponents`
(`video_stream_demo.py`). Neither runs on the GPU, which is the primary
bottleneck for a fully GPU-accelerated Canny pipeline.

**Parallelism rating:** ★☆☆☆☆ — inherently sequential per BFS frontier; only
parallelisable with specialised algorithms at the cost of implementation complexity.

---

## 4. Inter-Stage Data Flow Summary

```
Stage            Reads from       Writes to         Data volume
─────────────────────────────────────────────────────────────────────
Gaussian Blur    raw image        blurred            1 array  (H×W f32)
Sobel Gradient   blurred          magnitude, angle   2 arrays (H×W f32 each)
NMS              magnitude, angle nms                1 array  (H×W f32)
Double Threshold nms              strong, weak       2 arrays (H×W bool each)
Hysteresis       strong, weak     edges              1 array  (H×W bool)
─────────────────────────────────────────────────────────────────────
```

In the GPU pipeline, Stages 1–3 keep all arrays on the GPU throughout (no
CPU–GPU transfer between them). Only after NMS does the current implementation
copy data back to CPU for Stages 4–5. This PCIe transfer is a latency cost that
could be eliminated if Stages 4–5 were moved to the GPU.

---

## 5. cuTile-Specific Dependency Considerations

cuTile exposes a tile-based execution model: a kernel is launched over a 1D
grid of blocks, each block processing a contiguous tile of `tile_size` elements.
The following dependency properties matter for cuTile mapping.

### 5.1 Sobel Magnitude Kernel

The implemented cuTile Sobel kernel (`sobel_magnitude_from_neighbors_cutile`)
pre-extracts eight shifted views of the interior image region and flattens each
into a 1D array. The kernel tile `t` reads positions `[t × tile_size ..
(t+1) × tile_size - 1]` from each of the eight neighbour arrays and writes
the same range of the output array.

**There are no inter-tile data dependencies.** Tile `t` never reads output
written by tile `t-1` or `t+1`. This makes the tile assignment trivially
correct for any `tile_size` value.

**Tile-size effect on performance (not correctness):**  
Smaller tiles → more blocks → potentially better occupancy on small images but
higher kernel-launch overhead per pixel.  
Larger tiles → fewer blocks → lower launch overhead but may exceed shared memory
or register limits per block. The measured optimal tile size depends on image
resolution and GPU architecture.

### 5.2 Why Gaussian Blur and NMS Fit cuTile

Both Gaussian blur (per-pass) and NMS have the same structural property as
Sobel: each output pixel is a function of a fixed-size neighbourhood of *input*
pixels. The neighbour-extraction trick used for Sobel can be applied to either
stage to create independent tile inputs with no inter-tile dependencies.

### 5.3 Why Hysteresis Does Not Fit the Current cuTile Model

The cuTile kernel model (as used here) assumes that each tile computes its
output from tile-local input data only. Hysteresis violates this assumption:
the output of tile `t` influences whether pixels in adjacent tiles should be
marked as edges (through connected-component propagation). A correct cuTile
hysteresis kernel would require either:
- **Iterative re-launching** of a kernel until convergence, which reintroduces
  global synchronisation between launches.
- **Halo / ghost-cell regions** exchanged between tiles, which the current
  cuTile API does not directly support.

---

## 6. Memory Access Pattern Analysis

### 6.1 Access Locality

| Stage           | Access pattern              | L2/shared-memory friendliness |
|-----------------|-----------------------------|---------------------------------|
| Gaussian (horiz)| Sequential row scan         | ★★★★★ — coalesced reads        |
| Gaussian (vert) | Column scan (stride = width)| ★★☆☆☆ — strided, cache-unfriendly for large images |
| Sobel           | 3×3 neighbourhood           | ★★★☆☆ — partially coalesced; neighbour rows add stride accesses |
| NMS             | Centre + 2 conditional nbrs | ★★★☆☆ — branch-dependent gather; mostly sequential |
| Double Threshold| Per-element read            | ★★★★★ — sequential scan        |
| Hysteresis (BFS)| Irregular queue-driven      | ★☆☆☆☆ — random access, cache-hostile |

### 6.2 The Vertical Gaussian Pass

The vertical pass reads `temp[r, c]` for `r` values that are `width` elements
apart in memory. For a 640-pixel-wide image stored in row-major order, adjacent
column elements are 2560 bytes (4 bytes × 640) apart. This stride causes cache
line waste and is the primary memory bottleneck of the Gaussian stage on both
CPU and GPU.

**Mitigation strategies considered:**
- Transpose the image, apply a horizontal pass, transpose back (two transpositions
  plus one horizontal pass, but all accesses are coalesced).
- Use GPU texture memory, which caches 2D spatial locality.
- Accumulate vertical sums in shared memory (cuTile does not currently expose
  explicit shared-memory management in the Python API).

### 6.3 Sobel Neighbour Arrays in cuTile

By pre-extracting and flattening the eight neighbour arrays before the kernel
launch, the cuTile Sobel kernel reads eight sequential streams simultaneously.
GPU hardware can service multiple concurrent memory transactions, so this layout
is generally coalesced — all threads in a warp read the same relative position
from each neighbour array, and those positions are contiguous in memory.

The cost of pre-extracting the neighbour arrays (eight CuPy slicing operations)
is measured as part of the kernel's wall-clock time and is included in the tile-
size sweep benchmarks.

---

## 7. Parallelism Classification Summary

| Stage           | Parallel type              | GPU suitable | cuTile suitable | Current GPU impl |
|-----------------|----------------------------|:------------:|:---------------:|:----------------:|
| Gaussian Blur   | Embarrassingly parallel    | Yes          | Yes             | CuPy             |
| Sobel Gradient  | Embarrassingly parallel    | Yes          | Yes             | cuTile (benchmarked), CuPy (pipeline) |
| NMS             | Embarrassingly parallel    | Yes          | Yes             | CuPy             |
| Double Threshold| Embarrassingly parallel (+ reduction) | Yes | Yes         | CPU              |
| Hysteresis      | Irregular / frontier-based | Difficult    | Difficult       | CPU              |

**Note on Sobel:** The cuTile Sobel kernel is implemented and benchmarked in
`src/gpu/sobel_cutile_benchmark.py` and produces correct results. However, it
is not integrated into the main pipeline (`cutile_canny_pipeline_benchmark.py`)
because the Pascal-architecture GPU used for development does not support the
cuTile instruction set required by this kernel. The pipeline falls back to the
equivalent CuPy implementation. See git commit `07f36ca` for details.

---

## 8. Implications for Further Optimisation

### 8.1 Stages that would benefit most from cuTile

1. **Sobel Gradient** — already implemented; blocked by GPU compatibility.
   Would be the first stage to integrate once a supported GPU is available.
2. **NMS** — the same neighbour-extraction approach used for Sobel applies
   directly. Eight shifted views can be pre-extracted and a tile kernel can
   classify each pixel independently.
3. **Gaussian Blur (horizontal pass)** — sequential access pattern is ideal
   for cuTile tiles; low implementation risk.

### 8.2 Stages that require special handling

4. **Double Threshold** — the parallel reduction for percentile computation
   is straightforward with CuPy (`cp.percentile`); the pixel classification
   itself is trivially parallel. Moving both to GPU removes the NMS→CPU transfer.
5. **Hysteresis** — the most difficult stage. The connected-components approach
   (`cv2.connectedComponents`) is already faster than the BFS loop for the CPU
   path. A GPU connected-components algorithm (e.g., via CuPy's sparse graph
   routines or a custom label-propagation kernel) would close the last gap and
   allow a fully GPU-resident pipeline.

### 8.3 The GPU–CPU Transfer Bottleneck

The current pipeline transfers the NMS result from GPU to CPU (`cp.asnumpy`)
before Stages 4–5. For a 640×480 frame (4-byte float32), this is ~1.2 MB per
frame. At a PCIe 3.0 ×16 bandwidth of ~12 GB/s, this transfer alone costs
~0.1 ms — small but non-negligible when targeting 60+ FPS. Moving Stages 4–5
to the GPU would eliminate this transfer entirely and reduce per-frame overhead.