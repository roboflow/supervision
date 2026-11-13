# CompactMask — Memory-Efficient Mask Storage

This example benchmarks `CompactMask`, a new mask representation introduced in `supervision` that replaces dense `(N, H, W)` boolean arrays with a crop-scoped Run-Length Encoding (RLE). The benchmark demonstrates full API compatibility, massive memory savings, and order-of-magnitude annotation speedups — with no change to your existing `Detections` code.

______________________________________________________________________

## The Problem

Instance segmentation models return one boolean mask per detected object. `supervision` stores these as a stacked `(N, H, W)` numpy array.

For a 4K image with 1 000 detected objects:

```
1 000 x 3840 x 2160 x 1 byte = 8.3 GB
```

At this scale, typical pipelines crash with `MemoryError` before a single frame is annotated. Aerial imagery, satellite tiles, and high-density crowd scenes all hit this wall.

______________________________________________________________________

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

| Format              | Per object     | N=100      | N=1 000  | vs Dense   |
| ------------------- | -------------- | ---------- | -------- | ---------- |
| **Dense** (current) | 8.29 MB        | 829 MB     | 8.3 GB   | 1x         |
| Local Crop + Offset | 6.4 KB         | 640 KB     | 6.4 MB   | 1 300x     |
| **Crop-RLE** ✓      | **~2 KB**      | **200 KB** | **2 MB** | **4 000x** |
| Polygon ⚠ lossy     | ~3.2 KB        | 320 KB     | 3.2 MB   | 2 600x     |
| memmap              | 8.29 MB (disk) | 829 MB     | 8.3 GB   | 1x (disk)  |

Crop-RLE beats Local Crop because it only encodes actual pixel runs, skipping the ~35% background pixels within each bounding box.

#### Encode time: dense array → format

| Format              | Complexity                        | N=10        | N=100     | N=1 000    |
| ------------------- | --------------------------------- | ----------- | --------- | ---------- |
| Local Crop + Offset | O(A) — strided slice from xyxy    | **~0.1 ms** | **~1 ms** | **~10 ms** |
| **Crop RLE**        | O(A) — scan crop rows for runs    | ~0.2 ms     | ~2 ms     | ~20 ms     |
| Polygon             | O(P) — `cv2.findContours` on crop | ~2 ms       | ~20 ms    | ~200 ms    |
| memmap              | O(I) — write 8.29 MB to disk      | ~80 ms      | ~800 ms   | ~8 000 ms  |

#### Decode time: format → full (H, W) mask

Required by `MaskAnnotator`, `mask_iou_batch`, `merge()`, etc. Dominant cost at 4K is **allocating and zeroing a 8.29 MB array**, which is identical across all in-memory formats once full materialisation is needed.

| Format                | N=10      | N=100      | N=1 000     |
| --------------------- | --------- | ---------- | ----------- |
| Local Crop / Crop RLE | **~3 ms** | **~30 ms** | **~300 ms** |
| Polygon               | ~5 ms     | ~50 ms     | ~500 ms     |
| memmap                | ~80 ms    | ~800 ms    | ~8 000 ms   |

#### Decode time: crop-only path (optimised)

When callers need only the bounding-box region — `MaskAnnotator` crop-paint path, `.area`, `contains_holes`, `filter_segments_by_distance`:

| Format              | Complexity                       | N=10      | N=100     | N=1 000   |
| ------------------- | -------------------------------- | --------- | --------- | --------- |
| Local Crop + Offset | O(1) — already stored            | **~0 ms** | **~0 ms** | **~0 ms** |
| **Crop RLE** ✓      | O(A) — expand ~240 runs          | ~0.02 ms  | ~0.2 ms   | ~2 ms     |
| Polygon             | O(A) — `fillPoly` on crop canvas | ~2 ms     | ~20 ms    | ~200 ms   |
| memmap              | N/A — always full-size           | ~80 ms    | ~800 ms   | ~8 000 ms |

Crop RLE's `.crop()` method powers the `MaskAnnotator` optimisation — it never allocates the full image canvas, which is the entire source of the annotation speedup.

#### IoU / NMS at 1 % bbox overlap rate (sparse aerial scene)

| Format              | Strategy                              | N=1 000    |
| ------------------- | ------------------------------------- | ---------- |
| Dense (current)     | All pairs, 640² pixel AND             | ~10 000 ms |
| Local Crop + Offset | Bbox pre-filter → pixel IoU           | **~5 ms**  |
| Crop RLE            | Bbox pre-filter → expand intersection | ~15 ms     |

At N=1 000 with 1 % overlap, bbox pre-filter reduces 499 500 candidate pairs to ~5 000 overlapping pairs — a ~2 000x reduction in pixel-level work.

______________________________________________________________________

## Why Crop-RLE Was Chosen over Local Crop

Both formats compress extremely well; the deciding factors for Crop-RLE are:

1. **~3x smaller** for masks that are themselves sparse within their bounding box.
2. **COCO RLE interop path** — crop RLE uses column-major (F-order) pixel scan, matching `pycocotools`; to interoperate, you still need to construct a full-image COCO RLE from the crop-scoped encoding (for example by padding/merging runs onto the full-image canvas, or by materialising the crop in the full image and re-encoding).
3. `.area` computed directly from run lengths — no materialisation, no allocation.

The main trade-off: crop-only decode is O(A) rather than O(1). For the common solid-fill segmentation mask this is negligible (\<0.1 ms per mask).

______________________________________________________________________

## Operation-by-Operation Speedup Analysis

This section walks through every `Detections` operation that touches masks and shows exactly why `CompactMask` is faster. All code snippets are taken from the actual implementation. Numbers use the **FHD-200-50%-v600** scenario unless noted (1920 x 1080 image, 200 detections, each mask filling ~50% of the frame, 600-vertex polygons — a realistic hard case with dense fill and complex object boundaries).

At 50% fill on an FHD image each mask's bounding box covers a large portion of the frame, producing many RLE runs per row.

______________________________________________________________________

### Memory

Dense stores one full-resolution bool array per mask:

```
N x H x W x 1 byte
200 x 1080 x 1920 x 1 = 414 MB
```

Compact stores three lightweight structures:

```python
self._rles: list[npt.NDArray[np.int32]]  # N Python references to small int32 arrays
self._crop_shapes: npt.NDArray[np.int32]  # (N, 2) — crop (h, w) per mask
self._offsets: npt.NDArray[np.int32]  # (N, 2) — (x1, y1) origin per mask
```

Per-mask RLE size at 50% fill with 600-vertex polygons: ~4.7 KB (933 KB / 200). Per-mask dense size: 1920 x 1080 x 1 = 2.1 MB. Per-mask ratio: 2.1 MB / 4.7 KB = **~445x**.

Scaled to N=200: 200 x 4.7 KB = ~933 KB of RLE data, plus `_crop_shapes` (1.6 KB) and `_offsets` (1.6 KB). Python list + array object overhead roughly doubles the footprint for small N.

| Component       | Dense          | Compact     | Ratio     |
| --------------- | -------------- | ----------- | --------- |
| Mask data       | 414 MB         | **~933 KB** | ~445x     |
| Python overhead | **negligible** | ~933 KB     | --        |
| **Total**       | **414 MB**     | **~1.9 MB** | **~392x** |

At 5% fill with 8-vertex polygons, the ratio reaches 10 000x–20 000x because crops are tiny and RLEs are extremely short. The benchmark's 4K-200-5%-v8 scenario measures 21 786x (theory) / ~6 000x (malloc). The SAT-200-5%-v8 scenario reaches 62 968x theoretical.

______________________________________________________________________

### `.area`

Dense `Detections.area` reads every pixel of every mask:

```python
# detection/core.py — dense path
return np.array([np.sum(mask) for mask in self.mask])
# N masks x H x W boolean sums = 200 x 2.1 M = 420 million reads
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

At FHD-200-50%-v600, dense `.area` takes 84.66 ms; compact takes 0.48 ms — a **71x speedup**. At SAT-200-20%-v128 the measured speedup reaches **1 204x** because the dense array is 13.4 GB and each sum must scan the entire canvas.

| Factor                             | Reduction   |
| ---------------------------------- | ----------- |
| RLE sums vs full-frame pixel reads | ~4 600x     |
| int32 arithmetic vs bool reduction | ~2x         |
| No (H, W) allocation per mask      | latency     |
| **Combined**                       | **~1 000x** |

______________________________________________________________________

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

At FHD-200-50%-v600, dense `filter` takes 14.56 ms; compact takes 0.03 ms — a **500x speedup**. At SAT-200-20%-v128 the speedup reaches **14 757x**.

|             | Dense                   | Compact                             |
| ----------- | ----------------------- | ----------------------------------- |
| Data copied | K x H x W (full frames) | K Python references + K x 8 bytes   |
| Allocation  | new `(K, H, W)` array   | new `CompactMask` shell (~trivial)  |
| **Speedup** |                         | **hundreds to tens of thousands x** |

______________________________________________________________________

### `annotate` (`MaskAnnotator`)

Dense: for each mask, `MaskAnnotator` indexes the full `(H, W)` array and applies a boolean mask across the entire scene:

```python
# annotators/core.py — dense path
mask = np.asarray(detections.mask[detection_idx], dtype=bool)
colored_mask[mask] = color.as_bgr()
```

Each `detections.mask[detection_idx]` for a dense array yields a full `(H, W)` view, and the boolean indexing scans all pixels.

Compact: the annotator detects `CompactMask` and paints only the crop region:

```python
# annotators/core.py — compact path
x1 = int(compact_mask.offsets[detection_idx, 0])
y1 = int(compact_mask.offsets[detection_idx, 1])
crop_m = compact_mask.crop(detection_idx)
crop_h, crop_w = crop_m.shape
colored_mask[y1 : y1 + crop_h, x1 : x1 + crop_w][crop_m] = color.as_bgr()
```

`compact_mask.crop()` decodes the RLE into a `(crop_h, crop_w)` array. At FHD-200-50%-v600, dense `annotate` takes 848.95 ms; compact takes 32.67 ms — a **22x speedup**. At SAT-200-20%-v128 the speedup reaches **89x**.

| Factor                                             | Reduction           |
| -------------------------------------------------- | ------------------- |
| Crop decode vs full-frame boolean index (per mask) | crop-size dependent |
| No full `(H, W)` allocation per integer index      | latency             |
| x N masks                                          | compounds           |
| **Combined**                                       | **~26 – 400x**      |

______________________________________________________________________

### IoU (`mask_iou_batch` / `compact_mask_iou_batch`)

Dense `mask_iou_batch` on N=200, FHD:

```python
# detection/utils/iou_and_nms.py — _mask_iou_batch_split
intersection_area = np.logical_and(masks_true[:, None], masks_detection).sum(
    axis=(2, 3)
)
# shape (200, 200, 1080, 1920) — ~80 billion boolean ops
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

At 5% fill, two random masks overlap with probability ~4%. ~96% of the N² pairs get IoU = 0 for free — no pixel work at all.

**2. Sub-crop decode — compare only the intersection region**

```python
ox_a, oy_a = int(x1a[i]), int(y1a[i])
sub_a = crops_a[i][ly1 - oy_a : ly2 - oy_a + 1, lx1 - ox_a : lx2 - ox_a + 1]

ox_b, oy_b = int(x1b[j]), int(y1b[j])
sub_b = crops_b[j][ly1 - oy_b : ly2 - oy_b + 1, lx1 - ox_b : lx2 - ox_b + 1]

inter = int(np.logical_and(sub_a, sub_b).sum())
```

The intersection sub-region of two overlapping crops is typically far smaller than the full frame.

**3. Crop caching — each mask decoded at most once**

```python
if i not in crops_a:
    crops_a[i] = masks_true.crop(i)
```

Area is obtained from `_rle_area` (sum odd-indexed runs), never touching the pixel grid:

```python
areas_a: npt.NDArray[np.int64] = masks_true.area
```

At FHD-200-50%-v600, dense IoU takes 23 915 ms; compact takes 51.58 ms — a **446x speedup**. At 5% fill / sparse scenarios the speedup is even larger because fewer bbox pairs overlap.

| Factor                               | Reduction       |
| ------------------------------------ | --------------- |
| Bbox pre-filter at sparse fill       | 25x             |
| Sub-crop vs full frame per pair      | ~200x           |
| Area from RLE, not `sum(axis=(1,2))` | ~10x            |
| No 5 GB scratch allocation           | latency         |
| **Combined**                         | **~100 – 500x** |

At 20% fill the gaps close — more pairs overlap, larger crops — speedup drops toward the lower end of the range.

______________________________________________________________________

### NMS (`mask_non_max_suppression`)

Both dense and compact paths now call `mask_iou_batch(masks, masks)` directly, computing exact mask IoU on the original (unresized) masks. There is no intermediate resize step.

```python
# detection/utils/iou_and_nms.py — NMS (both paths)
ious = mask_iou_batch(masks, masks, overlap_metric)
```

`mask_iou_batch` dispatches internally: when passed a `CompactMask` it calls `compact_mask_iou_batch`, applying all three IoU optimisations (bbox pre-filter, sub-crop decode, crop caching). When passed a dense ndarray it runs the chunked pixel-AND path.

All three IoU optimisations apply to the compact path:

| Factor                                | Reduction                    |
| ------------------------------------- | ---------------------------- |
| Bbox pre-filter eliminates most pairs | 25x at sparse fill           |
| Sub-crop decode for remaining pairs   | ~200x                        |
| Area from RLE, not pixel sum          | ~10x                         |
| **Combined**                          | **same as IoU: ~100 – 500x** |

At FHD-200-50%-v600, dense NMS takes 5 231 ms; compact takes 48.15 ms — a **481x speedup**. Dense IoU/NMS is skipped for scenarios above 1 GB (4K-200 and SAT-200 tiers); compact NMS still runs on those.

______________________________________________________________________

### `merge` (`Detections.merge`)

Dense: `np.vstack` allocates a new `(N1+N2, H, W)` array and copies both halves:

```python
# detection/core.py — dense merge path
return np.vstack([np.asarray(m) for m in masks])
# Merging two 100-mask sets at FHD: 2 x 100 x 2.1 MB = 414 MB copied
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

At FHD-200-50%-v600, dense merge takes 29.71 ms; compact takes 0.03 ms — a **929x speedup**. At SAT-200-20%-v128 the speedup reaches **89 046x**.

|             | Dense                   | Compact                    |
| ----------- | ----------------------- | -------------------------- |
| Data moved  | N x H x W (full frames) | N references + N x 8 bytes |
| Allocation  | new `(N, H, W)` array   | new `CompactMask` shell    |
| **Speedup** |                         | **effectively free**       |

**Note:** `Detections.merge` calls `is_empty()` on each input. Before the `len(xyxy) > 0` short-circuit was added, `is_empty()` invoked `__eq__` which called `np.array_equal(self.to_dense(), ...)` — materialising the entire `(N, H, W)` CompactMask to dense just to check emptiness. The fix:

```python
# detection/core.py — Detections.is_empty (fixed)
if len(self.xyxy) > 0:
    return False
```

This O(1) check avoids the O(N x H x W) dense materialisation that previously dominated compact merge time.

______________________________________________________________________

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

At FHD-200-50%-v600, dense offset takes 42.30 ms; compact takes 0.02 ms — a **2 016x speedup**. At SAT-200-20%-v128 the speedup reaches **290 779x**.

|                   | Dense                                  | Compact (no-clip fast path)          |
| ----------------- | -------------------------------------- | ------------------------------------ |
| Work per mask     | allocate `(new_H, new_W)` + copy H x W | add scalar to offset row — O(1)      |
| N=200 at FHD      | 200 x 2.1 MB = **414 MB** alloc + copy | two numpy ops on `(N, 2)` int32      |
| Output allocation | new `(N, new_H, new_W)`                | shared RLE list + new `(N, 2)` array |
| **Speedup**       |                                        | **effectively free (>1 000x)**       |

In the `InferenceSlicer` pipeline the canvas is always expanded by the tile offset, so no crop ever overflows — the fast path is always taken. Clipping only activates for objects that genuinely straddle the image boundary.

______________________________________________________________________

### `centroids` (`calculate_masks_centroids`)

Dense: `np.tensordot` reads every pixel of every mask to compute weighted coordinate sums:

```python
# detection/utils/masks.py — dense centroid path
vertical_indices, horizontal_indices = np.indices((height, width)) + 0.5
# np.tensordot(masks, indices, axes=([1, 2], [0, 1]))
# reads all N x H x W values
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

At FHD-200-50%-v600, dense centroids takes 1 133.68 ms; compact takes 60.39 ms — a **13x speedup**. At SAT-200-20%-v128 the speedup reaches **857x** because the dense path must allocate and scan a 13.4 GB array.

| Factor                                    | Reduction           |
| ----------------------------------------- | ------------------- |
| Crop area vs full frame (per mask)        | fill-dependent      |
| No global `np.indices((H, W))` allocation | saves large float64 |
| **Combined (N=200)**                      | **~19 – 1 000x**    |

______________________________________________________________________

### Summary

Measured speedups at the **FHD-200-50%-v600** operating point (dense fill, complex polygons — a realistic hard case). Dense baseline = 1x.

| Operation        | Dense cost  | Compact cost | Speedup |
| ---------------- | ----------- | ------------ | ------- |
| Memory           | 414 MB      | **~1.9 MB**  | ~392x   |
| `.area`          | 84.66 ms    | **0.48 ms**  | 71x     |
| `filter`         | 14.56 ms    | **0.03 ms**  | 500x    |
| `annotate`       | 848.95 ms   | **32.67 ms** | 22x     |
| `mask_iou_batch` | 23 915 ms   | **51.58 ms** | 446x    |
| NMS              | 5 231 ms    | **48.15 ms** | 481x    |
| `merge`          | 29.71 ms    | **0.03 ms**  | 929x    |
| `with_offset`    | 42.30 ms    | **0.02 ms**  | 2 016x  |
| `centroids`      | 1 133.68 ms | **60.39 ms** | 13x     |

All speedups are larger at sparser fill fractions and larger resolutions. At SAT-200-20%-v128, `.area` reaches 1 204x and `merge` reaches 89 046x. At the sparsest scenarios (5% fill, 8-vertex polygons), memory ratios exceed 60 000x.

______________________________________________________________________

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

______________________________________________________________________

## Benchmark

Run on any machine — no GPU or real model required:

```bash
uv run python examples/compact_mask/benchmark.py
```

For a focused benchmark of the Roboflow inference-result parser API, run:

```bash
uv run python examples/compact_mask/bench_inference_api.py
```

This script downloads all supervision image assets plus the middle frame from every supervision video asset by default, runs one real segmentation inference per source image, requests native RLE masks from Inference, freezes that result, and then compares parser performance:

```python
sv.Detections.from_inference(result)
sv.Detections.from_inference(result, compact_masks=True)
```

Timing repetitions, warmups, confidence, IoU, response mask format, and the default model live as constants in `bench_inference_api.py`.

Inference runs and segmentation-derived box fields are outside the timed benchmark loop. By default the script uses `rfdetr-seg-large` with `response_mask_format="rle"`; set `BENCH_INFERENCE_MODEL_ID` to override the model. Set `ROBOFLOW_API_KEY` when your model requires authentication. Sources where the model returns no native RLE segmentation masks are skipped because there is no RLE parser work to benchmark. `rfdetr-large` is a valid local Inference model id, but it is object detection only; use an `rfdetr-seg-*` model for instance segmentation.

Run one specific supervision image or video asset with `--asset`:

```bash
uv run python examples/compact_mask/bench_inference_api.py --asset people-walking
uv run python examples/compact_mask/bench_inference_api.py --asset soccer
uv run python examples/compact_mask/bench_inference_api.py --asset vehicles
uv run python examples/compact_mask/bench_inference_api.py --asset people-walking-video
```

The output reports image size, segmented objects, median parser time, peak traced allocations, mask storage, and parser speedup (`dense parser time / compact parser time`).

**Speedup column:** The `speedup` value reflects allocation savings — how much time is saved by skipping the dense `(N, H, W)` bool-stack allocation — not a faster RLE decode. Compact RLE arithmetic is typically slower than the dense NumPy path. The net result:

- **Compact is faster** only when the dense `(N, H, W)` bool-stack allocation dominates — large images with many sparse masks where avoiding that allocation outweighs the RLE arithmetic cost.
- **Compact is slower** for small images or dense/overlapping masks, where Python RLE arithmetic dominates and the allocation cost is negligible.
- **The primary guaranteed benefit is memory**: compact masks use roughly 99% less memory than dense stacks for typical segmentation output, regardless of which parse direction is faster.

The default run includes a `synthetic-dense-64` row (64×64 image, 4 fully-filled masks) to demonstrate the adversarial regime where compact is slower than dense. For each real source with segmentation masks, the script also writes a validation overlay to `examples/compact_mask/outputs/*_segmentations.jpg`.

### Sample results — inference API

Measured on macOS Apple M4 Max, 50 reps after 3 warmups, using `rfdetr-seg-large` via Roboflow Inference.

| src                        | res       | seg | dense ms | CM ms     | speedup | peak MB (dense/compact) | mask MB (dense/compact) | ok  |
| -------------------------- | --------- | --- | -------- | --------- | ------- | ----------------------- | ----------------------- | --- |
| synthetic-dense-64         | 64×64     | 4   | **0.03** | 0.11      | 0.31×   | 0.04 / 0.05             | 0.02 / 0.00             | ✓   |
| people-walking.jpg         | 1920×1080 | 53  | 85.56    | **12.55** | 6.82×   | 219.86 / 0.11           | 109.90 / 0.02           | ✓   |
| soccer.jpg                 | 398×224   | 21  | 1.36     | **1.07**  | 1.27×   | 3.77 / 0.05             | 1.87 / 0.00             | ✓   |
| vehicles.mp4#269           | 3840×2160 | 7   | 46.03    | **2.60**  | 18×     | 116.13 / 0.07           | 58.06 / 0.00            | ✓   |
| milk-bottling-plant.mp4#94 | 1920×1080 | 9   | 15.61    | **11.57** | 1.35×   | 37.34 / 0.53            | 18.66 / 0.03            | ✓   |
| vehicles-2.mp4#637         | 1920×1080 | 47  | 76.87    | **13.59** | 5.66×   | 194.97 / 0.13           | 97.46 / 0.03            | ✓   |
| grocery-store.mp4#501      | 3840×2160 | 4   | 27.20    | **4.36**  | 6.24×   | 66.36 / 0.22            | 33.18 / 0.01            | ✓   |
| subway.mp4#649             | 2160×3840 | 42  | 325.71   | **32.21** | 10×     | 696.78 / 0.80           | 348.36 / 0.09           | ✓   |
| market-square.mp4#237      | 2160×3840 | 96  | 732.98   | **27.24** | 27×     | 1592.61 / 0.22          | 796.26 / 0.05           | ✓   |
| people-walking.mp4#170     | 1920×1080 | 60  | 100.99   | **12.69** | 7.96×   | 248.89 / 0.12           | 124.42 / 0.02           | ✓   |
| beach-1.mp4#223            | 3840×2160 | 33  | 223.50   | **13.39** | 17×     | 547.47 / 0.12           | 273.72 / 0.02           | ✓   |
| basketball-1.mp4#238       | 1920×1080 | 2   | 3.61     | **2.05**  | 1.76×   | 8.30 / 0.15             | 4.15 / 0.01             | ✓   |
| skiing.mp4#176             | 1920×1080 | 11  | 16.47    | **3.07**  | 5.37×   | 45.63 / 0.08            | 22.81 / 0.01            | ✓   |

- **seg** — number of instance segmentations returned by the model
- **dense ms / CM ms** — median parse time for `from_inference()` vs `from_inference(compact_masks=True)`
- **speedup** — dense / compact parse time; values below 1× (e.g., synthetic-dense-64) indicate the adversarial regime where RLE arithmetic cost exceeds allocation savings
- **peak MB** — peak traced allocations during parsing (dense / compact)
- **mask MB** — mask storage only (dense / compact); compact is typically 100–5 000× smaller
- **ok** — `compact.to_dense()` pixel-exactly matches dense masks

Six image tiers x three fill fractions (5 / 20 / 50 %) x three vertex counts (8 / 128 / 600):

| Tier    | Resolution | Objects | Dense array | Notes                                |
| ------- | ---------- | ------- | ----------- | ------------------------------------ |
| FHD-100 | 1920x1080  | 100     | 0.21 GB     | Full operations including IoU+NMS    |
| FHD-200 | 1920x1080  | 200     | 0.41 GB     | Full operations including IoU+NMS    |
| FHD-400 | 1920x1080  | 400     | 0.83 GB     | Full operations including IoU+NMS    |
| 4K-100  | 3840x2160  | 100     | 0.83 GB     | Full operations including IoU+NMS    |
| 4K-200  | 3840x2160  | 200     | 1.66 GB     | Dense IoU+NMS skipped (array > 1 GB) |
| SAT-200 | 8192x8192  | 200     | 13.4 GB     | Dense IoU+NMS skipped (array > 1 GB) |

Dense timing is skipped automatically when the dense IoU/NMS array would exceed 1 GB (`IOU_DENSE_SKIP_GB`), preventing swap thrashing. All dense ops are skipped above 16 GB (`DENSE_SKIP_GB`); no scenario in the current matrix reaches that threshold. Memory is always reported as theoretical `NxHxW` bytes.

### Sample results (macOS, Apple M4 Max, REPS=4)

| Scenario         | Dense mem | Compact theor. | Mem x   | Area x | Filter x | Annot x | IoU x | NMS x | Merge x  | Offset x | Centroids x |
| ---------------- | --------- | -------------- | ------- | ------ | -------- | ------- | ----- | ----- | -------- | -------- | ----------- |
| FHD-100-5%-v8    | 207 MB    | 28 KB          | 7 418x  | —      | —        | —       | —     | —     | —        | —        | —           |
| FHD-100-50%-v600 | 207 MB    | 913 KB         | 227x    | —      | —        | —       | —     | —     | —        | —        | —           |
| FHD-200-50%-v600 | 415 MB    | 933 KB         | 445x    | 71x    | 500x     | 22x     | 446x  | 481x  | 929x     | 2 016x   | 13x         |
| FHD-400-5%-v8    | 829 MB    | 60 KB          | 13 937x | —      | —        | —       | —     | —     | —        | —        | —           |
| 4K-100-5%-v8     | 829 MB    | 53 KB          | 15 554x | —      | —        | —       | —     | —     | —        | —        | —           |
| 4K-100-20%-v128  | 829 MB    | 586 KB         | 1 415x  | —      | —        | —       | —     | —     | —        | —        | —           |
| 4K-200-5%-v8     | 1 659 MB  | 76 KB          | 21 786x | —      | —        | —       | —     | —     | —        | —        | —           |
| SAT-200-5%-v8    | 13 422 MB | 213 KB         | 62 968x | 6 942x | 30 255x  | 204x    | †     | †     | 105 545x | 251 629x | 2 173x      |
| SAT-200-20%-v128 | 13 422 MB | 2 596 KB       | 5 171x  | 1 204x | 14 757x  | 89x     | †     | †     | 89 046x  | 290 779x | 857x        |
| SAT-200-50%-v600 | 13 422 MB | 14 222 KB      | 944x    | —      | —        | —       | †     | †     | —        | —        | —           |

- **Compact theor.** — sum of internal numpy buffer `nbytes`
- **Mem x** — dense / compact theoretical ratio
- **Area x / Filter x / Annot x / IoU x / NMS x / Merge x / Offset x / Centroids x** — compact speedup over dense for each operation
- **†** — dense IoU+NMS skipped (dense array > 1 GB); compact still runs and is timed
- **—** — not shown; full per-scenario tables are printed by the benchmark script

All non-skipped scenarios pass: pixel-perfect annotation, exact area, lossless `to_dense()` roundtrip.

______________________________________________________________________

## Use-Cases

- **Aerial / satellite imagery** — thousands of small objects on large tiles; dense masks exhaust RAM before inference completes.
- **High-density crowd / cell segmentation** — N > 500 on FHD already requires several GB of mask storage per batch.
- **Real-time annotation pipelines** — crop-paint cuts annotation from seconds to milliseconds at 4K resolution.
- **Long-running tracking** — accumulated `Detections` across many frames stay in kilobytes rather than gigabytes.
- **`InferenceSlicer`** — `with_offset()` adjusts crop origins directly when stitching tile results; no dense materialisation needed.

______________________________________________________________________

## Limitations

- `CompactMask` is **not** a full `np.ndarray`. Call `.to_dense()` before passing to code that requires arbitrary ndarray methods (`astype`, `reshape`, `ravel`, `any`, `all`, …).
- RLE format is **column-major (F-order), crop-scoped** — pixel-scan order matches COCO / pycocotools, but crop scope differs from full-image scope. Use `.to_dense()` to materialize a full-image dense mask, then encode that mask to COCO RLE before passing it to pycocotools.
- `from_dense()` requires the input `(N, H, W)` array to fit in memory. For truly OOM-scale data, build `CompactMask` per-detection directly from model output crops rather than from a pre-allocated dense stack.

______________________________________________________________________

## Non-Maximum Merging Benchmark

Run `python examples/compact_mask/benchmark_nmm.py --canvas 512 2048 4096` on each revision to compare compact `Detections.with_nmm`. It holds twelve 64×64 mask crops fixed while varying the logical canvas. Solid, checkerboard and seeded random masks expose the effect of RLE fragmentation. No model, image download or GPU is required.

The output table includes input RLE count, median uninstrumented wall time, peak traced allocation measured in separate calls, and output count/area. The memory number is Python/NumPy allocation tracked by `tracemalloc`, not process RSS. Input construction is excluded. `--crop-size`, `--duplicates` (default: three per group) and `--repeats` control crop size, group size and repetitions. After the compact NMM union change, `--canvas 100000` can demonstrate that increasing canvas dimensions alone no longer allocates full image unions. Avoid that size when benchmarking the old implementation, which materializes several complete images.

The RLE union sorts `K` foreground column intervals. Mask fragmentation and the columns foreground runs cross determine `K`; enlarging the canvas with fixed encoded crops does not. A cheap O(masks) run-count check (`_should_union_densely`) flags mask sets fragmented enough that the sort would cost more than a single dense pass, and unions those via a bbox-local dense OR instead — bounded to a 16 Mi-pixel bbox so it never re-introduces the full-canvas materialization this union function exists to avoid. Above that bbox cap, the interval path runs regardless of fragmentation, since a dense fallback there would cost more memory, not less. NMM overlap evaluation still decodes overlapping crops, and large or loose crops remain expensive.

Measured on Python 3.13.15 / NumPy 2.3.1, comparing the parent of `035079270` (dense unions) with the current fragmentation-aware implementation. Both revisions used the same benchmark script, with four disjoint groups in a 2×2 grid. Peak columns are maximum traced allocation over three complete NMM calls, excluding input construction; time columns are median wall time of the same calls, uninstrumented.

| Crop / canvas       | Objects | Pattern      | Dense-only mem. peak (MiB) | Current mem. peak (MiB) | Dense-only time (ms) | Current time (ms) |
| ------------------- | ------- | ------------ | -------------------------- | ----------------------- | -------------------- | ----------------- |
| 32×32 / 512×512     | 200     | checkerboard | 25.83                      | **1.69** (0.07×)        | 12.69                | **9.87** (0.78×)  |
| 32×32 / 512×512     | 200     | solid        | 25.07                      | **0.27** (0.01×)        | **11.06**            | 11.10 (1.00×)     |
| 400×400 / 1024×1024 | 12      | checkerboard | 17.37                      | 17.37 (1.00×)           | 18.38                | **18.12** (0.99×) |
| 400×400 / 1024×1024 | 12      | solid        | 6.01                       | **0.69** (0.11×)        | 5.64                 | **2.35** (0.42×)  |
| 400×400 / 2048×2048 | 12      | checkerboard | 31.93                      | **17.37** (0.54×)       | 20.75                | **17.16** (0.83×) |
| 400×400 / 2048×2048 | 12      | solid        | 24.01                      | **0.69** (0.03×)        | 8.62                 | **2.28** (0.26×)  |
| 400×400 / 4096×4096 | 12      | checkerboard | 103.93                     | **17.37** (0.17×)       | 33.85                | **17.20** (0.51×) |
| 400×400 / 4096×4096 | 12      | solid        | 96.01                      | **0.69** (0.01×)        | 21.67                | **2.32** (0.11×)  |

Ratios in parentheses are interval/dense; under 1× means the interval path wins. The 32×32/512×512 row uses `--duplicates 50` (200 small objects total, instead of the default three per group) — many small fragmented crops on a small canvas. The 400×400 rows previously showed a genuine weak side: at 1024×1024, a single large checkerboard crop cost the old interval-only union 1.79× the dense implementation's memory and 4.5× its time, since sorting `K` ≈ 1.9M foreground intervals cost more than the dense array's single `O(area)` pass. The fragmentation-aware dispatch above closes that gap: every checkerboard row now flags as fragmented (run count ≈ crop area, since checkerboard flips every pixel) and unions via a bbox-local dense OR — matching or beating the dense-only baseline on both memory and time at every measured scale, while keeping peak memory flat (~17.37 MiB) as canvas grows from 1024×1024 to 4096×4096, unlike the dense-only baseline, which scales with canvas area. Solid rows are unaffected (their run count is one column per instance, far under the fragmentation threshold), so they still take the interval path and keep the same numbers as before. A single mask right at the dense fallback's 16 Mi-pixel bbox cap (a 4000×4000 checkerboard-fragmented crop, three duplicates) still favors dense decisively: 488 MiB / 136 ms with the fallback versus 2,197 MiB / 1,258 ms on the pure interval path, forced for comparison. The fallback's own encode step is not free on a maximally-fragmented result — several bytes of transient array per pixel, not one bit — but far cheaper than sorting the same fragmentation.

This does not eliminate the tradeoff in general, only bounds where it can occur: the dense fallback is capped at a 16 Mi-pixel union bbox specifically so a handful of tiny, widely separated fragmented crops cannot force a large dense allocation. A single mask crop larger than 4096×4096 pixels exceeds that cap and takes the interval path regardless of fragmentation, paying the full `K log K` sort with no dense fallback — measured at 4100×4100 (just past the cap, three duplicates): 2,309 MiB / 1,371 ms, the same order of cost as the forced-interval comparison above. The theoretical worst case from before persists only past this exact boundary, for a single object that size — not exercised by the table rows.

Reproduce the small-object row with `python examples/compact_mask/benchmark_nmm.py --crop-size 32 --canvas 512 --duplicates 50 --pattern checkerboard solid --repeats 3`, and the large-crop rows with `python examples/compact_mask/benchmark_nmm.py --crop-size 400 --canvas 1024 2048 4096 --pattern checkerboard solid --repeats 3`, on the current implementation; check out the parent of `035079270` for the dense-union baseline. Foreground area matched between revisions per row: 2,048 (32×32 checkerboard) / 4,096 (32×32 solid) / 320,000 (400×400 checkerboard) / 640,000 (400×400 solid). Absolute measurements depend on the Python/NumPy environment.

______________________________________________________________________

## Files

| File                     | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `benchmark.py`           | Full benchmark across FHD / 4K / satellite tiers         |
| `bench_inference_api.py` | Focused dense vs compact `from_inference` benchmark      |
| `benchmark_nmm.py`       | Compact NMM canvas size and mask fragmentation benchmark |
| `README.md`              | This file                                                |
