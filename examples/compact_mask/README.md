# CompactMask — Memory-Efficient Mask Storage

This example benchmarks `CompactMask`, a new mask representation introduced in `supervision` that replaces dense `(N, H, W)` boolean arrays with a crop-scoped Run-Length Encoding (RLE). The benchmark demonstrates full API compatibility, massive memory savings, and order-of-magnitude annotation speedups — with no change to your existing `Detections` code.

---

## The Problem

Instance segmentation models return one boolean mask per detected object. `supervision` stores these as a stacked `(N, H, W)` numpy array.

For a 4K image with 1 000 detected objects:

```
1 000 x 3840 x 2160 x 1 byte = 8.3 GB
```

At this scale, typical pipelines crash with `MemoryError` before a single frame is annotated. Aerial imagery, satellite tiles, and high-density crowd scenes all hit this wall.

---

## The Solution — Crop-RLE Storage

`CompactMask` stores each mask as a run-length encoding of its **bounding-box crop** rather than the full image canvas.

```
dense (N,H,W) mask   →   N x crop_RLE + N x (x1,y1) offset
8.3 GB               →   ~280 KB
```

The bounding boxes are already present in `Detections.xyxy`, so no extra metadata is required from the caller.

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

Crop-RLE beats Local Crop because it only encodes actual pixel runs, skipping the ~35% background pixels within each bounding box.

#### Encode time: dense array → format

| Format              | Complexity                        | N=10    | N=100   | N=1 000   |
| ------------------- | --------------------------------- | ------- | ------- | --------- |
| Local Crop + Offset | O(A) — strided slice from xyxy    | ~0.1 ms | ~1 ms   | ~10 ms    |
| **Crop RLE**        | O(A) — scan crop rows for runs    | ~0.2 ms | ~2 ms   | ~20 ms    |
| Polygon             | O(P) — `cv2.findContours` on crop | ~2 ms   | ~20 ms  | ~200 ms   |
| memmap              | O(I) — write 8.29 MB to disk      | ~80 ms  | ~800 ms | ~8 000 ms |

#### Decode time: format → full (H, W) mask

Required by `MaskAnnotator`, `mask_iou_batch`, `merge()`, etc. Dominant cost at 4K is **allocating and zeroing a 8.29 MB array**, which is identical across all in-memory formats once full materialisation is needed.

| Format                | N=10   | N=100   | N=1 000   |
| --------------------- | ------ | ------- | --------- |
| Local Crop / Crop RLE | ~3 ms  | ~30 ms  | ~300 ms   |
| Polygon               | ~5 ms  | ~50 ms  | ~500 ms   |
| memmap                | ~80 ms | ~800 ms | ~8 000 ms |

#### Decode time: crop-only path (optimised)

When callers need only the bounding-box region — `MaskAnnotator` crop-paint path, `.area`, `contains_holes`, `filter_segments_by_distance`:

| Format              | Complexity                       | N=10     | N=100   | N=1 000   |
| ------------------- | -------------------------------- | -------- | ------- | --------- |
| Local Crop + Offset | O(1) — already stored            | ~0 ms    | ~0 ms   | ~0 ms     |
| **Crop RLE** ✓      | O(A) — expand ~240 runs          | ~0.02 ms | ~0.2 ms | ~2 ms     |
| Polygon             | O(A) — `fillPoly` on crop canvas | ~2 ms    | ~20 ms  | ~200 ms   |
| memmap              | N/A — always full-size           | ~80 ms   | ~800 ms | ~8 000 ms |

Crop RLE's `.crop()` method powers the `MaskAnnotator` optimisation — it never allocates the full image canvas, which is the entire source of the annotation speedup.

#### IoU / NMS at 1 % bbox overlap rate (sparse aerial scene)

| Format              | Strategy                              | N=1 000    |
| ------------------- | ------------------------------------- | ---------- |
| Dense (current)     | All pairs, 640² pixel AND             | ~10 000 ms |
| Local Crop + Offset | Bbox pre-filter → pixel IoU           | **~5 ms**  |
| Crop RLE            | Bbox pre-filter → expand intersection | **~15 ms** |

At N=1 000 with 1 % overlap, bbox pre-filter reduces 499 500 candidate pairs to ~5 000 overlapping pairs — a ~2 000x reduction in pixel-level work.

---

## Why Crop-RLE Was Chosen over Local Crop

Both formats compress extremely well; the deciding factors for Crop-RLE are:

1. **~3x smaller** for masks that are themselves sparse within their bounding box.
2. **COCO RLE interop path** — row-major crop RLE can be re-encoded to column-major full-image RLE for `pycocotools` if needed.
3. `.area` computed directly from run lengths — no materialisation, no allocation.

The main trade-off: crop-only decode is O(A) rather than O(1). For the common solid-fill segmentation mask this is negligible (\<0.1 ms per mask).

---

## Operation-by-Operation Speedup Analysis

This section walks through every `Detections` operation that touches masks and shows exactly why `CompactMask` is faster. All code snippets are taken from the actual implementation. Numbers use the **4K-500-5 %** scenario unless noted (3840 x 2160 image, 500 detections, each mask filling ~5 % of the frame).

At 5 % fill on a 4K image each mask's bounding box is roughly 450 x 450 px, producing ~4 RLE runs per row (smooth polygon edge) x 450 rows = ~1 800 runs.

---

### Memory

Dense stores one full-resolution bool array per mask:

```
N x H x W x 1 byte
500 x 2160 x 3840 x 1 = 4.1 GB
```

Compact stores three lightweight structures:

```python
self._rles: list[npt.NDArray[np.int32]]  # N Python references to small int32 arrays
self._crop_shapes: npt.NDArray[np.int32]  # (N, 2) — crop (h, w) per mask
self._offsets: npt.NDArray[np.int32]  # (N, 2) — (x1, y1) origin per mask
```

Per-mask RLE size at 5 % fill: ~1 800 int32 run lengths x 4 bytes = ~7.2 KB. Per-mask dense size: 3840 x 2160 x 1 = 8.3 MB. Per-mask ratio: 8.3 MB / 7.2 KB = **~1 150x**.

Scaled to N=500: 500 x 7.2 KB = 3.6 MB of RLE data, plus `_crop_shapes` (4 KB) and `_offsets` (4 KB). Python list + array object overhead roughly doubles the footprint for small N, giving ~7 MB actual vs 4.1 GB dense.

| Component       | Dense      | Compact   | Ratio     |
| --------------- | ---------- | --------- | --------- |
| Mask data       | 4.1 GB     | 3.6 MB    | 1 150x    |
| Python overhead | negligible | ~3.4 MB   | --        |
| **Total**       | **4.1 GB** | **~7 MB** | **~600x** |

At 20 % fill, crops grow and RLE runs increase — the ratio drops to ~200x. At the benchmark's 4K-500-5 % scenario the measured ratio is 30 000x because the synthetic benchmark uses smaller objects (80 x 80 px crops) with fewer runs than the 450 x 450 assumption above.

---

### `.area`

Dense `Detections.area` reads every pixel of every mask:

```python
# detection/core.py — dense path
return np.array([np.sum(mask) for mask in self.mask])
# N masks x H x W boolean sums = 500 x 8.3 M = 4.15 billion reads
```

Compact delegates to `_rle_area`, which sums only the odd-indexed run lengths (the True-pixel runs) in each RLE:

```python
# detection/compact_mask.py — _rle_area
return int(np.sum(rle[1::2]))
```

```python
# detection/compact_mask.py — CompactMask.area
return np.array([_rle_area(r) for r in self._rles], dtype=np.int64)
```

At 4K-500-5 %: 500 x ~900 odd-indexed int32 sums = ~450 000 operations, vs 500 x 8.3 M = 4.15 billion boolean reads.

| Factor                             | Reduction   |
| ---------------------------------- | ----------- |
| RLE sums vs full-frame pixel reads | ~4 600x     |
| int32 arithmetic vs bool reduction | ~2x         |
| No (H, W) allocation per mask      | latency     |
| **Combined**                       | **~1 000x** |

Benchmark column "Area x" shows 1 087x at 4K-500-5 %, consistent with this analysis.

---

### `filter` / `__getitem__` (boolean index)

Dense: `masks[bool_array]` triggers NumPy fancy indexing, which allocates a new `(K, H, W)` bool array and copies K full frames:

```python
# detection/core.py — Detections.__getitem__
mask = (self.mask[index] if self.mask is not None else None,)
# For dense ndarray, numpy allocates (K, 2160, 3840) and memcpy's K frames
```

Compact `CompactMask.__getitem__` converts the boolean index to integer positions and builds a new `CompactMask` from Python list indexing and NumPy fancy indexing on small `(N, 2)` arrays:

```python
# detection/compact_mask.py — CompactMask.__getitem__
if isinstance(index, np.ndarray) and index.dtype == bool:
    idx_arr = np.where(index)[0]
# ...
new_rles = [self._rles[int(i)] for i in idx_arr]
new_crop_shapes: npt.NDArray[np.int32] = self._crop_shapes[idx_arr]
new_offsets: npt.NDArray[np.int32] = self._offsets[idx_arr]
return CompactMask(new_rles, new_crop_shapes, new_offsets, self._image_shape)
```

Keeping K=250 of 500 at 4K:

|             | Dense                         | Compact                               |
| ----------- | ----------------------------- | ------------------------------------- |
| Data copied | 250 x 3840 x 2160 = **2 GB**  | 250 Python references + 250 x 8 bytes |
| Allocation  | new `(250, 2160, 3840)` array | new `CompactMask` shell (~trivial)    |
| **Speedup** |                               | **~10 000x less data moved**          |

---

### `annotate` (`MaskAnnotator`)

Dense: for each mask, `MaskAnnotator` indexes the full `(H, W)` array and applies a boolean mask across the entire scene:

```python
# annotators/core.py — dense path
mask = np.asarray(detections.mask[detection_idx], dtype=bool)
colored_mask[mask] = color.as_bgr()
```

Each `detections.mask[detection_idx]` for a dense array yields a full `(2160, 3840)` view, and the boolean indexing scans all 8.3 M pixels.

Compact: the annotator detects `CompactMask` and paints only the crop region:

```python
# annotators/core.py — compact path
x1 = int(compact_mask.offsets[detection_idx, 0])
y1 = int(compact_mask.offsets[detection_idx, 1])
crop_m = compact_mask.crop(detection_idx)
crop_h, crop_w = crop_m.shape
colored_mask[y1 : y1 + crop_h, x1 : x1 + crop_w][crop_m] = color.as_bgr()
```

`compact_mask.crop()` decodes the RLE into a `(crop_h, crop_w)` array — at 5 % fill, roughly 450 x 450 = 200 K pixels vs 8.3 M for the full frame.

| Factor                                             | Reduction      |
| -------------------------------------------------- | -------------- |
| Crop decode vs full-frame boolean index (per mask) | ~42x           |
| No full `(H, W)` allocation per integer index      | latency        |
| x N=500 masks                                      | compounds      |
| **Combined**                                       | **~40 – 400x** |

Benchmark column "Annot x" shows 383x at 4K-500-5 %.

---

### IoU (`mask_iou_batch` / `compact_mask_iou_batch`)

Dense `mask_iou_batch` on N=500, 4K:

```python
# detection/utils/iou_and_nms.py — _mask_iou_batch_split
intersection_area = np.logical_and(masks_true[:, None], masks_detection).sum(
    axis=(2, 3)
)
# shape (500, 500, 2160, 3840) — 2 trillion boolean ops
# .sum(axis=(2,3)) for intersection counts
# memory_limit splits this into chunks capped at 5 GB scratch
```

Compact `compact_mask_iou_batch` — three layered optimisations:

**1. Vectorised bbox pre-filter — O(N²) array ops, zero decoding**

```python
ix1: npt.NDArray[np.int32] = np.maximum(x1a[:, None], x1b[None, :])
iy1: npt.NDArray[np.int32] = np.maximum(y1a[:, None], y1b[None, :])
ix2: npt.NDArray[np.int32] = np.minimum(x2a[:, None], x2b[None, :])
iy2: npt.NDArray[np.int32] = np.minimum(y2a[:, None], y2b[None, :])
bbox_overlap: npt.NDArray[np.bool_] = (ix1 <= ix2) & (iy1 <= iy2)
```

At 5 % fill, two random masks overlap with probability ~4 %. ~96 % of the 250 000 pairs get IoU = 0 for free — no pixel work at all.

**2. Sub-crop decode — compare only the intersection region**

```python
ox_a, oy_a = int(x1a[i]), int(y1a[i])
sub_a = crops_a[i][ly1 - oy_a : ly2 - oy_a + 1, lx1 - ox_a : lx2 - ox_a + 1]

ox_b, oy_b = int(x1b[j]), int(y1b[j])
sub_b = crops_b[j][ly1 - oy_b : ly2 - oy_b + 1, lx1 - ox_b : lx2 - ox_b + 1]

inter = int(np.logical_and(sub_a, sub_b).sum())
```

Typical crop at 4K / 5 % fill is ~450 x 450 px. The intersection sub-region of two overlapping crops is typically ~200 x 200 = 40 000 ops vs 8.3 M for a full frame AND.

**3. Crop caching — each mask decoded at most once**

```python
if i not in crops_a:
    crops_a[i] = masks_true.crop(i)
```

Area is obtained from `_rle_area` (sum odd-indexed runs), never touching the pixel grid:

```python
areas_a: npt.NDArray[np.int64] = masks_true.area
```

| Factor                               | Reduction   |
| ------------------------------------ | ----------- |
| ~4 % of pairs need pixel work        | 25x         |
| Sub-crop vs full frame per pair      | ~200x       |
| Area from RLE, not `sum(axis=(1,2))` | ~10x        |
| No 5 GB scratch allocation           | latency     |
| **Combined**                         | **~1 100x** |

At 20 % fill the gaps close — more pairs overlap, larger crops — speedup drops from ~1 100x to ~130x.

---

### NMS (`mask_non_max_suppression`)

Dense: resizes all N masks to 640 x 640 (`resize_masks`), then runs the greedy NMS loop where every IoU step performs a 640 x 640 boolean AND:

```python
# detection/utils/iou_and_nms.py — dense NMS path
masks_resized = resize_masks(masks, mask_dimension)
ious = mask_iou_batch(masks_resized, masks_resized, overlap_metric)
```

`resize_masks` for N=500 at 4K creates a `(500, 640, 640)` intermediate (~200 MB) via meshgrid fancy indexing — a significant allocation and computation just to prepare for the IoU step.

Compact: `mask_non_max_suppression` detects `CompactMask` and calls `compact_mask_iou_batch` directly on the original crop coordinates, skipping the resize entirely:

```python
# detection/utils/iou_and_nms.py — compact NMS path
if isinstance(masks, CompactMask):
    ious = compact_mask_iou_batch(masks, masks, overlap_metric)
```

All three IoU optimisations (bbox pre-filter, sub-crop decode, crop caching) apply. The resize step is eliminated completely.

| Factor                                             | Reduction                            |
| -------------------------------------------------- | ------------------------------------ |
| Skip resize_masks (N x 640 x 640 alloc + meshgrid) | ~200 MB saved + compute              |
| Bbox pre-filter eliminates ~96 % of pairs          | 25x                                  |
| Sub-crop decode for remaining pairs                | ~200x                                |
| **Combined**                                       | **same as IoU: ~1 100x at 5 % fill** |

---

### `merge` (`Detections.merge`)

Dense: `np.vstack` allocates a new `(N1+N2, H, W)` array and copies both halves:

```python
# detection/core.py — dense merge path
return np.vstack([np.asarray(m) for m in masks])
# Merging two 250-mask sets at 4K: 2 x 250 x 8.3 MB = 4.1 GB copied
```

Compact: `CompactMask.merge` extends a Python list and concatenates two small int32 arrays:

```python
# detection/compact_mask.py — CompactMask.merge
new_rles: list[npt.NDArray[np.int32]] = []
for m in masks_list:
    new_rles.extend(m._rles)

new_crop_shapes: npt.NDArray[np.int32] = np.concatenate(
    [m._crop_shapes for m in masks_list], axis=0
)
new_offsets: npt.NDArray[np.int32] = np.concatenate(
    [m._offsets for m in masks_list], axis=0
)
```

`list.extend` copies N reference pointers. `np.concatenate` on `(N, 2)` int32 arrays copies N x 8 bytes per array.

|             | Dense                         | Compact                        |
| ----------- | ----------------------------- | ------------------------------ |
| Data moved  | 2 x 250 x 8.3 MB = **4.1 GB** | 500 references + 500 x 8 bytes |
| Allocation  | new `(500, 2160, 3840)` array | new `CompactMask` shell        |
| **Speedup** |                               | **effectively free**           |

**Note:** `Detections.merge` calls `is_empty()` on each input. Before the `len(xyxy) > 0` short-circuit was added, `is_empty()` invoked `__eq__` which called `np.array_equal(self.to_dense(), ...)` — materialising the entire `(N, H, W)` CompactMask to dense just to check emptiness. The fix:

```python
# detection/core.py — Detections.is_empty (fixed)
if len(self.xyxy) > 0:
    return False
```

This O(1) check avoids the O(N x H x W) dense materialisation that previously dominated compact merge time.

---

### `offset` / `with_offset` (`InferenceSlicer` tile stitching)

Dense `move_masks`: allocates a new `(N, new_H, new_W)` array and copies each mask with shifted slice coordinates — O(N x H x W):

```python
# detection/utils/masks.py — move_masks
mask_array = np.full((masks.shape[0], resolution_wh[1], resolution_wh[0]), False)
# ... source/destination slicing logic ...
mask_array[:, dst_y1:dst_y2, dst_x1:dst_x2] = masks[:, src_y1:src_y2, src_x1:src_x2]
```

Compact `with_offset(dx, dy)`: vectorised bounds check first. All new bounding-box positions are computed in a single numpy op. When none overflow the new canvas — the common case in `InferenceSlicer` — the RLE data is not touched at all:

```python
# detection/compact_mask.py — CompactMask.with_offset (fast path)
new_offsets = self._offsets + np.array([dx, dy], dtype=np.int32)  # O(N) numpy
needs_clip = (x1s < 0) | (y1s < 0) | (x2s >= new_w) | (y2s >= new_h)
if not needs_clip.any():
    return CompactMask(
        list(self._rles), self._crop_shapes.copy(), new_offsets, new_image_shape
    )
```

When a crop does overflow (e.g. object at a tile edge), only that crop is decoded, sliced, and re-encoded. Masks fully outside bounds get a 1x1 all-False stub without any decoding.

|                   | Dense                                  | Compact (no-clip fast path)          |
| ----------------- | -------------------------------------- | ------------------------------------ |
| Work per mask     | allocate `(new_H, new_W)` + copy H x W | add scalar to offset row — O(1)      |
| N=500 at 4K       | 500 x 8.3 MB = **4.1 GB** alloc + copy | two numpy ops on `(N, 2)` int32      |
| Output allocation | new `(N, new_H, new_W)` = 4.1 GB       | shared RLE list + new `(N, 2)` array |
| **Speedup**       |                                        | **effectively free (>1 000x)**       |

In the `InferenceSlicer` pipeline the canvas is always expanded by the tile offset, so no crop ever overflows — the fast path is always taken. Clipping only activates for objects that genuinely straddle the image boundary.

---

### `centroids` (`calculate_masks_centroids`)

Dense: `np.tensordot` reads every pixel of every mask to compute weighted coordinate sums:

```python
# detection/utils/masks.py — dense centroid path
vertical_indices, horizontal_indices = np.indices((height, width)) + 0.5
# np.tensordot(masks, indices, axes=([1, 2], [0, 1]))
# reads all N x H x W values = 500 x 8.3 M = 4.15 billion
```

Compact: per-crop loop decodes only the bounding-box region and computes centroids within that crop:

```python
# detection/utils/masks.py — compact centroid path
crop = masks.crop(i)
crop_h, crop_w = crop.shape
x1 = int(masks.offsets[i, 0])
y1 = int(masks.offsets[i, 1])
# ...
crop_rows, crop_cols = np.indices((crop_h, crop_w))
cx = float(np.sum((crop_cols + 0.5)[crop])) / total + x1
cy = float(np.sum((crop_rows + 0.5)[crop])) / total + y1
```

At 5 % fill each crop is ~450 x 450 = 200 K pixels vs 8.3 M for the full frame.

| Factor                                    | Reduction            |
| ----------------------------------------- | -------------------- |
| Crop area vs full frame (per mask)        | ~42x                 |
| No global `np.indices((H, W))` allocation | saves ~63 MB float64 |
| **Combined (N=500)**                      | **~40x**             |

---

### Summary

Estimated speedups at the **4K-500-5 %** operating point. Dense baseline = 1x.

| Operation         | Dense cost                   | Compact cost                | Speedup          |
| ----------------- | ---------------------------- | --------------------------- | ---------------- |
| Memory            | 4.1 GB                       | ~7 MB                       | ~600x            |
| `.area`           | N x H x W reads              | N x ~900 int32 sums         | ~1 000x          |
| `filter` (K=250)  | 2 GB copy                    | 250 references              | ~10 000x         |
| `annotate`        | N x 8.3 M px scan            | N x 200 K px crop           | ~400x            |
| `mask_iou_batch`  | N² x H x W (chunked)         | bbox pre-filter + sub-crop  | ~1 100x          |
| NMS               | resize to 640² + N² IoU      | direct crop IoU             | ~1 100x          |
| `merge` (2 x 250) | 4.1 GB vstack                | list.extend + concat (N, 2) | effectively free |
| `with_offset`     | N x H x W copy + giant alloc | O(N) offset arithmetic      | >1 000x          |
| `centroids`       | N x H x W tensordot          | N x crop_area indices       | ~40x             |

All speedups diminish as fill fraction grows: at 20 % fill, crops are larger, more bbox pairs overlap, and RLEs contain more runs. The IoU speedup drops from ~1 100x to ~130x. Memory savings drop from ~600x to ~200x.

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

Dense timing is skipped automatically when the array would exceed 12 GB (`DENSE_SKIP_GB`), preventing swap thrashing on SAT scenarios. Memory is still reported as theoretical `NxHxW` bytes.

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
- **Annot x** — `MaskAnnotator` speedup; crop-paint avoids full-frame allocation
- **N/A** — dense timing skipped (array > 12 GB)

All non-skipped scenarios pass: pixel-perfect annotation, exact area, lossless `to_dense()` roundtrip.

---

## Use-Cases

- **Aerial / satellite imagery** — thousands of small objects on large tiles; dense masks exhaust RAM before inference completes.
- **High-density crowd / cell segmentation** — N > 500 on FHD already requires several GB of mask storage per batch.
- **Real-time annotation pipelines** — crop-paint cuts annotation from seconds to milliseconds at 4K resolution.
- **Long-running tracking** — accumulated `Detections` across many frames stay in kilobytes rather than gigabytes.
- **`InferenceSlicer`** — `with_offset()` adjusts crop origins directly when stitching tile results; no dense materialisation needed.

---

## Limitations

- `CompactMask` is **not** a full `np.ndarray`. Call `.to_dense()` before passing to code that requires arbitrary ndarray methods (`astype`, `reshape`, `ravel`, `any`, `all`, …).
- RLE format is **row-major (C-order), crop-scoped** — incompatible with pycocotools / COCO API RLEs (column-major, full-image-scoped). Use `.to_dense()` first if you need pycocotools interop.
- `from_dense()` requires the input `(N, H, W)` array to fit in memory. For truly OOM-scale data, build `CompactMask` per-detection directly from model output crops rather than from a pre-allocated dense stack.

---

## Files

| File           | Description                                      |
| -------------- | ------------------------------------------------ |
| `benchmark.py` | Full benchmark across FHD / 4K / satellite tiers |
| `README.md`    | This file                                        |
