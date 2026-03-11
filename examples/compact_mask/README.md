# CompactMask — Memory-Efficient Mask Storage

This example benchmarks `CompactMask`, a new mask representation introduced in
`supervision` that replaces dense `(N, H, W)` boolean arrays with a crop-scoped
Run-Length Encoding (RLE). The benchmark demonstrates full API compatibility,
massive memory savings, and order-of-magnitude annotation speedups — with no
change to your existing `Detections` code.

---

## The Problem

Instance segmentation models return one boolean mask per detected object.
`supervision` stores these as a stacked `(N, H, W)` numpy array.

For a 4K image with 1 000 detected objects:

```
1 000 x 3840 x 2160 x 1 byte = 8.3 GB
```

At this scale, typical pipelines crash with `MemoryError` before a single frame
is annotated. Aerial imagery, satellite tiles, and high-density crowd scenes all
hit this wall.

---

## The Solution — Crop-RLE Storage

`CompactMask` stores each mask as a run-length encoding of its **bounding-box
crop** rather than the full image canvas.

```
dense (N,H,W) mask   →   N x crop_RLE + N x (x1,y1) offset
8.3 GB               →   ~280 KB
```

The bounding boxes are already present in `Detections.xyxy`, so no extra
metadata is required from the caller.

### Theoretical analysis (4K scene, 80x80 px objects, ~65% fill per bbox)

Assumptions used throughout the PR design analysis:

| Parameter              | Value                    |
| ---------------------- | ------------------------ |
| Image size             | 4K — 3840x2160 = 8.29 MP |
| Avg bounding box       | 80x80 px = 6 400 px²     |
| Fill ratio within bbox | ~65%                     |
| Avg contour vertices   | ~400 pts                 |
| Avg RLE runs / mask    | ~240 (3 runs x 80 rows)  |

#### Space comparison

| Format              | Per object     | N=100  | N=1 000    | vs Dense  |
| ------------------- | -------------- | ------ | ---------- | --------- |
| **Dense** (current) | 8.29 MB        | 829 MB | **8.3 GB** | 1x        |
| Local Crop + Offset | 6.4 KB         | 640 KB | 6.4 MB     | 1 300x    |
| **Crop-RLE** ✓      | ~2 KB          | 200 KB | **2 MB**   | 4 000x    |
| Polygon ⚠ lossy     | ~3.2 KB        | 320 KB | 3.2 MB     | 2 600x    |
| memmap              | 8.29 MB (disk) | 829 MB | 8.3 GB     | 1x (disk) |

Crop-RLE beats Local Crop because it only encodes actual pixel runs, skipping
the ~35% background pixels within each bounding box.

#### Encode time: dense array → format

| Format              | Complexity                        | N=10    | N=100   | N=1 000   |
| ------------------- | --------------------------------- | ------- | ------- | --------- |
| Local Crop + Offset | O(A) — strided slice from xyxy    | ~0.1 ms | ~1 ms   | ~10 ms    |
| **Crop RLE**        | O(A) — scan crop rows for runs    | ~0.2 ms | ~2 ms   | ~20 ms    |
| Polygon             | O(P) — `cv2.findContours` on crop | ~2 ms   | ~20 ms  | ~200 ms   |
| memmap              | O(I) — write 8.29 MB to disk      | ~80 ms  | ~800 ms | ~8 000 ms |

#### Decode time: format → full (H, W) mask

Required by `MaskAnnotator`, `mask_iou_batch`, `merge()`, etc.
Dominant cost at 4K is **allocating and zeroing a 8.29 MB array**, which is
identical across all in-memory formats once full materialisation is needed.

| Format                | N=10   | N=100   | N=1 000   |
| --------------------- | ------ | ------- | --------- |
| Local Crop / Crop RLE | ~3 ms  | ~30 ms  | ~300 ms   |
| Polygon               | ~5 ms  | ~50 ms  | ~500 ms   |
| memmap                | ~80 ms | ~800 ms | ~8 000 ms |

#### Decode time: crop-only path (optimised)

When callers need only the bounding-box region — `MaskAnnotator` crop-paint
path, `.area`, `contains_holes`, `filter_segments_by_distance`:

| Format              | Complexity                       | N=10     | N=100   | N=1 000   |
| ------------------- | -------------------------------- | -------- | ------- | --------- |
| Local Crop + Offset | O(1) — already stored            | ~0 ms    | ~0 ms   | ~0 ms     |
| **Crop RLE** ✓      | O(A) — expand ~240 runs          | ~0.02 ms | ~0.2 ms | ~2 ms     |
| Polygon             | O(A) — `fillPoly` on crop canvas | ~2 ms    | ~20 ms  | ~200 ms   |
| memmap              | N/A — always full-size           | ~80 ms   | ~800 ms | ~8 000 ms |

Crop RLE's `.crop()` method powers the `MaskAnnotator` optimisation — it never
allocates the full image canvas, which is the entire source of the annotation
speedup.

#### IoU / NMS at 1 % bbox overlap rate (sparse aerial scene)

| Format              | Strategy                              | N=1 000    |
| ------------------- | ------------------------------------- | ---------- |
| Dense (current)     | All pairs, 640² pixel AND             | ~10 000 ms |
| Local Crop + Offset | Bbox pre-filter → pixel IoU           | **~5 ms**  |
| Crop RLE            | Bbox pre-filter → expand intersection | **~15 ms** |

At N=1 000 with 1 % overlap, bbox pre-filter reduces 499 500 candidate pairs to
~5 000 overlapping pairs — a ~2 000x reduction in pixel-level work.

---

## Why Crop-RLE Was Chosen over Local Crop

Both formats compress extremely well; the deciding factors for Crop-RLE are:

1. **~3x smaller** for masks that are themselves sparse within their bounding box.
2. **COCO RLE interop path** — row-major crop RLE can be re-encoded to
    column-major full-image RLE for `pycocotools` if needed.
3. `.area` computed directly from run lengths — no materialisation, no allocation.

The main trade-off: crop-only decode is O(A) rather than O(1). For the common
solid-fill segmentation mask this is negligible (\<0.1 ms per mask).

---

## Drop-In Compatibility

`CompactMask` implements the same duck-typed interface as `np.ndarray`:

```python
import supervision as sv
from supervision.detection.compact_mask import CompactMask

# Build from an existing dense (N, H, W) bool array:
compact = CompactMask.from_dense(masks_dense, xyxy, image_shape=(H, W))

# Use exactly like a dense mask — no other code changes needed:
detections = sv.Detections(xyxy=xyxy, mask=compact, class_id=class_ids)

# Filtering, merging, area — all work transparently:
filtered = detections[confidence > 0.5]
areas = detections.area  # RLE sum, no materialisation
merged = sv.Detections.merge([det_a, det_b])

# MaskAnnotator works without any change:
annotated = sv.MaskAnnotator().annotate(frame, detections)

# Materialise back to dense when you need raw numpy:
dense_again = compact.to_dense()  # (N, H, W) bool
```

Supported indexing patterns:

| Expression         | Returns                      |
| ------------------ | ---------------------------- |
| `mask[i]` (int)    | Dense `(H, W)` bool array    |
| `mask[bool_array]` | New `CompactMask` (filtered) |
| `mask[slice]`      | New `CompactMask`            |
| `np.asarray(mask)` | Dense `(N, H, W)` bool array |

---

## Benchmark

Run on any machine — no GPU or real model required:

```bash
uv run python examples/compact_mask/benchmark.py
```

Three image tiers x three fill fractions (5 / 10 / 20 %):

| Tier | Resolution | Typical use-case                    |
| ---- | ---------- | ----------------------------------- |
| FHD  | 1920x1080  | Video surveillance, robotics        |
| 4K   | 3840x2160  | Drone footage, cinema               |
| SAT  | 8192x8192  | Sentinel-2 / GeoTIFF benchmark tile |

Dense timing is skipped automatically when the array would exceed 12 GB
(`DENSE_SKIP_GB`), preventing swap thrashing on SAT scenarios. Memory is still
reported as theoretical `NxHxW` bytes.

### Sample results (macOS, Apple M-series, REPS=5)

| Scenario    | Dense mem | Compact theor. | Compact actual | Mem x   | Area x | Annot x |
| ----------- | --------- | -------------- | -------------- | ------- | ------ | ------- |
| FHD-100-5%  | 207 MB    | 33 KB          | 62 KB          | 6 300x  | 280x   | 70x     |
| FHD-100-20% | 207 MB    | 67 KB          | 137 KB         | 3 100x  | 267x   | 27x     |
| 4K-500-5%   | 4 147 MB  | 139 KB         | 250 KB         | 30 000x | 1 087x | 383x    |
| 4K-1000-10% | 8 294 MB  | 277 KB         | 498 KB         | 30 000x | 1 120x | 439x    |
| SAT-200-5%  | 13 422 MB | 271 KB         | 485 KB         | 49 000x | N/A    | N/A     |

- **Compact theor.** — sum of internal numpy buffer `nbytes`
- **Compact actual** — `tracemalloc` peak during `CompactMask.from_dense()`, including Python object overhead (~2x theoretical for small object counts)
- **Mem x** — dense / compact theoretical ratio
- **Area x** — `.area` speedup; RLE sums True-pixel counts with no materialisation
- **Annot ×** — `MaskAnnotator` speedup; crop-paint avoids full-frame allocation
- **N/A** — dense timing skipped (array > 12 GB)

All non-skipped scenarios pass: pixel-perfect annotation, exact area,
lossless `to_dense()` roundtrip.

---

## Use-Cases

- **Aerial / satellite imagery** — thousands of small objects on large tiles;
    dense masks exhaust RAM before inference completes.
- **High-density crowd / cell segmentation** — N > 500 on FHD already requires
    several GB of mask storage per batch.
- **Real-time annotation pipelines** — crop-paint cuts annotation from seconds
    to milliseconds at 4K resolution.
- **Long-running tracking** — accumulated `Detections` across many frames stay
    in kilobytes rather than gigabytes.
- **`InferenceSlicer`** — `with_offset()` adjusts crop origins directly when
    stitching tile results; no dense materialisation needed.

---

## Limitations

- `CompactMask` is **not** a full `np.ndarray`. Call `.to_dense()` before
    passing to code that requires arbitrary ndarray methods (`astype`, `reshape`,
    `ravel`, `any`, `all`, …).
- RLE format is **row-major (C-order), crop-scoped** — incompatible with
    pycocotools / COCO API RLEs (column-major, full-image-scoped). Use
    `.to_dense()` first if you need pycocotools interop.
- `from_dense()` requires the input `(N, H, W)` array to fit in memory.
    For truly OOM-scale data, build `CompactMask` per-detection directly from
    model output crops rather than from a pre-allocated dense stack.

---

## Files

| File           | Description                                      |
| -------------- | ------------------------------------------------ |
| `benchmark.py` | Full benchmark across FHD / 4K / satellite tiers |
| `README.md`    | This file                                        |
