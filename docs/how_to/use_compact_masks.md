---
comments: true
---

# Use Compact Masks for Memory-Efficient Segmentation

[CompactMask][supervision.detection.compact_mask.CompactMask] stores each instance mask as a run-length encoding of its bounding-box **crop** rather than a full `(H, W)` boolean frame. For high-resolution images with many sparse masks this can reduce memory from tens of gigabytes to tens of megabytes, and eliminates full-frame decode work in annotators that only need the cropped region.

This guide covers the four main integration points:

1. [Ingesting COCO RLE payloads directly as CompactMask](#ingest-coco-rle-payloads)
2. [Parsing Roboflow Inference results without a dense stack](#parse-inference-results)
3. [Skipping mask materialisation for box/label annotators](#skip-unnecessary-materialisation)
4. [Merging mixed dense and compact detections](#merge-mixed-detections)

---

## Ingest COCO RLE Payloads

If your model or API returns masks in the COCO RLE format (`{"size": [H, W], "counts": "..."}`) you can convert them directly to `CompactMask` without allocating an `(N, H, W)` boolean array:

```python
import numpy as np
import supervision as sv
from supervision.detection.compact_mask import CompactMask

# Example: two COCO RLE masks for a 720×1280 frame.
# Replace the counts strings with actual compressed RLE payloads from your
# model or API — e.g., from pycocotools mask.encode() or an Inference response.
rles = [
    {"size": [720, 1280], "counts": "YOUR_RLE_COUNTS_STRING_HERE"},
    {"size": [720, 1280], "counts": "YOUR_RLE_COUNTS_STRING_HERE"},
]
xyxy = np.array(
    [
        [100.0, 50.0, 400.0, 300.0],
        [500.0, 200.0, 900.0, 600.0],
    ]
)

compact = CompactMask.from_coco_rle(rles, xyxy, image_shape=(720, 1280))

detections = sv.Detections(
    xyxy=xyxy,
    mask=compact,
    class_id=np.array([0, 1]),
)
```

`from_coco_rle` uses run-length arithmetic scoped to each bounding box so no dense pixel array is ever created. Uncompressed integer count lists are also accepted in place of compressed strings.

---

## Parse Inference Results

`Detections.from_inference` accepts a `compact_masks=True` flag that routes the Roboflow RLE payload through `CompactMask.from_coco_rle` instead of decoding to a dense stack:

```python
import supervision as sv

# result: a Roboflow Inference v2 response dict with instance masks.
detections = sv.Detections.from_inference(result, compact_masks=True)

from supervision.detection.compact_mask import CompactMask

assert isinstance(detections.mask, CompactMask)
```

!!! Warning

    `compact_masks=True` crops each mask to its detector bounding box. Pixels outside the box are silently dropped. For masks that extend meaningfully beyond the reported bounding box, use the default `compact_masks=False` (dense decode) to preserve all pixels.

To convert an existing dense-mask `Detections` to compact at any point:

```python
detections_compact = detections.to_compact_masks()
```

---

## Skip Unnecessary Materialisation

Annotators that do not draw masks (box, label, circle, ellipse, trace, keypoint) expose `requires_mask = False`. Integrations can branch on this flag to avoid decoding compact or RLE masks before annotation:

```python
import supervision as sv

annotators = [
    sv.BoxAnnotator(),
    sv.LabelAnnotator(),
    sv.MaskAnnotator(),  # requires_mask = True
]

for ann in annotators:
    if ann.requires_mask:
        # Annotator reads mask pixels — CompactMask decodes lazily per crop.
        scene = ann.annotate(scene, detections)
    else:
        # Annotator ignores masks — strip mask field to eliminate any decode cost.
        det_no_mask = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
        )
        scene = ann.annotate(scene, det_no_mask)
```

Annotators that set `requires_mask = True`: [MaskAnnotator][supervision.annotators.core.MaskAnnotator], [PolygonAnnotator][supervision.annotators.core.PolygonAnnotator], [HaloAnnotator][supervision.annotators.core.HaloAnnotator].

All others default to `requires_mask = False`.

!!! Note

    `PolygonAnnotator` and `MaskAnnotator` both operate directly on `CompactMask` without materialising the full `(N, H, W)` frame — passing compact detections to them is already efficient.

---

## Merge Mixed Detections

When merging `Detections` objects that mix dense `ndarray` masks and `CompactMask` instances, `Detections.merge` converts dense inputs to `CompactMask` automatically. No full `(N, H, W)` stack is allocated:

```python
import numpy as np
import supervision as sv
from supervision.detection.compact_mask import CompactMask

H, W = 720, 1280

# Compact detections from an RLE-based source.
# Replace the counts string with a real compressed RLE payload from your model or API.
rles = [{"size": [H, W], "counts": "YOUR_RLE_COUNTS_STRING_HERE"}]
xyxy_a = np.array([[100.0, 50.0, 400.0, 300.0]])
cm = CompactMask.from_coco_rle(rles, xyxy_a, image_shape=(H, W))
det_a = sv.Detections(xyxy=xyxy_a, mask=cm, class_id=np.array([0]))

# Dense detections from a different source.
masks_b = np.zeros((1, H, W), dtype=bool)
masks_b[0, 200:400, 500:800] = True
xyxy_b = np.array([[500.0, 200.0, 799.0, 399.0]])
det_b = sv.Detections(xyxy=xyxy_b, mask=masks_b, class_id=np.array([1]))

# Output is CompactMask regardless of input order.
merged = sv.Detections.merge([det_a, det_b])
assert isinstance(merged.mask, CompactMask)
assert len(merged) == 2
```

Merge rules:

| Inputs                                | Output mask type                |
| ------------------------------------- | ------------------------------- |
| All `CompactMask`                     | `CompactMask`                   |
| Mixed `CompactMask` + dense `ndarray` | `CompactMask`                   |
| All dense `ndarray`                   | `ndarray` (backward compatible) |

All `CompactMask` inputs must share the same `image_shape`; mismatches raise `ValueError`.

---

## Performance Notes

These estimates apply to the **parsing and annotation stage**, not end-to-end pipeline FPS. Model inference typically dominates total runtime.

| Optimisation                 | Realistic gain             | Applies when                                                  |
| ---------------------------- | -------------------------- | ------------------------------------------------------------- |
| `from_coco_rle` ingestion    | 25–60% faster parse        | Full-frame COCO RLE payload; current dense decode path        |
| `MaskAnnotator` ROI blending | 10–35% faster annotation   | Many small, sparse masks on high-res frames                   |
| `PolygonAnnotator` crop path | 15–45% faster polygon draw | Many compact masks; full-frame materialise was the bottleneck |
| Mixed-mask merge             | 5–20% faster merge         | Mix of compact and dense sources (e.g. multi-camera stitch)   |

Upper-end gains assume: ≥1080p frames, tens to hundreds of instances, masks covering less than ~20% of total pixels.

---

## API Reference

- [CompactMask][supervision.detection.compact_mask.CompactMask]
- [CompactMask.from_coco_rle][supervision.detection.compact_mask.CompactMask.from_coco_rle]
- [CompactMask.from_dense][supervision.detection.compact_mask.CompactMask.from_dense]
- [Detections.from_inference][supervision.detection.core.Detections.from_inference]
- [Detections.to_compact_masks][supervision.detection.core.Detections.to_compact_masks]
- [Detections.merge][supervision.detection.core.Detections.merge]
- [BaseAnnotator.requires_mask][supervision.annotators.base.BaseAnnotator]
