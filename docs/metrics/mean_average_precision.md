---
comments: true
description: API reference for MeanAveragePrecision and KeyPointMeanAveragePrecision — COCO mAP for detection (boxes, masks, oriented boxes) and pose estimation (keypoints, OKS).
---

# Mean Average Precision

Install the metrics extra before using this API:

```bash
pip install "supervision[metrics]"
```

`MeanAveragePrecision` scores detection models on `sv.Detections` (boxes, masks or oriented boxes) matched by IoU. `KeyPointMeanAveragePrecision` scores pose estimation models on `sv.KeyPoints` matched by Object Keypoint Similarity (OKS); see [Pose estimation](#pose-estimation). Both follow COCO evaluation and return results with the same `map50_95`, `map50`, `map75`, `plot` and `to_pandas` interface.

## Evaluate targets from a PyTorch DataLoader

A DataLoader can return any annotation format defined by its dataset and collate function. Convert each image's annotations to a separate `sv.Detections` object, then pair it with that image's predictions. Ground truth needs `xyxy` boxes and integer `class_id` values; only predictions need confidence scores.

For Hugging Face DETR, `Detections.from_transformers` accepts **post-processed predictions** with `boxes`, `labels`, and `scores`. Training targets instead have `boxes` and `class_labels`. With annotation conversion enabled, their boxes use normalized `(center_x, center_y, width, height)` coordinates. Convert them with `sv.xcycwh_to_xyxy` and scale to the same image dimensions used to post-process predictions. `sv.xywh_to_xyxy` is for boxes whose first two coordinates describe the top-left corner, so it is not the right converter here.

### Convert normalized targets

The following helper converts image-normalized boxes into pixel coordinates. `image_size_hw` is `(height, width)`, while the scale applied to boxes is `(width, height, width, height)`.

```python
import numpy as np
import numpy.typing as npt
import supervision as sv


def normalized_targets_to_detections(
    boxes: npt.NDArray[np.float32],
    class_ids: npt.NDArray[np.int64],
    image_size_hw: tuple[int, int],
) -> sv.Detections:
    """Convert normalized center-based targets to pixel corner coordinates."""
    height, width = image_size_hw
    xyxy = sv.xcycwh_to_xyxy(boxes)
    xyxy *= np.array([width, height, width, height], dtype=xyxy.dtype)
    return sv.Detections(xyxy=xyxy, class_id=class_ids)


target = normalized_targets_to_detections(
    boxes=np.array([[0.5, 0.5, 0.25, 0.5]], dtype=np.float32),
    class_ids=np.array([0], dtype=np.int64),
    image_size_hw=(480, 640),
)
print(target.xyxy)
# [[240. 120. 400. 360.]]
```

An image without objects should still have one target entry: pass boxes with shape `(0, 4)` and class IDs with shape `(0,)`. The helper preserves these shapes and does not modify the input arrays.

### Run evaluation

This example assumes you already have a DETR `model`, its `image_processor`, and `TEST_DATALOADER`. Each batch contains `pixel_values`, `pixel_mask`, and a list of per-image `labels` dictionaries. `classes` lists class names in class-ID order, read here from the model's `id2label` mapping; both predictions and targets must use IDs from `0` to `len(classes) - 1`. Remap both sides first if your dataset uses sparse category IDs.

**Coordinate assumptions:** the targets below are normalized relative to each image before batch padding, and evaluation preprocessing only resizes images. Under these assumptions, `orig_size` restores original-image pixel coordinates, as in the [Hugging Face evaluation example](https://huggingface.co/docs/transformers/tasks/object_detection#preparing-function-to-compute-map). Check your processor and collate function: processing multiple images together can update annotations to the padded canvas. Such targets need padding and resize transforms reversed before using `orig_size`; multiplying by `orig_size` alone does not undo padding. Likewise, crops or other geometric augmentations require their transforms to be accounted for. Keep validation preprocessing deterministic.

```python
import torch
from supervision.metrics import MeanAveragePrecision

device = next(model.parameters()).device
classes = [
    model.config.id2label[class_id] for class_id in sorted(model.config.id2label)
]
model.eval()
predictions: list[sv.Detections] = []
targets: list[sv.Detections] = []

with torch.no_grad():
    for batch in TEST_DATALOADER:
        labels = batch["labels"]
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            pixel_mask=batch["pixel_mask"].to(device),
        )
        target_sizes = torch.stack([label["orig_size"] for label in labels]).to(device)
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.0,
        )

        for result, label in zip(results, labels, strict=True):
            height, width = label["orig_size"].detach().cpu().tolist()
            targets.append(
                normalized_targets_to_detections(
                    boxes=label["boxes"].detach().cpu().numpy().astype(np.float32),
                    class_ids=label["class_labels"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64),
                    image_size_hw=(height, width),
                )
            )
            predictions.append(sv.Detections.from_transformers(result))

map_result = MeanAveragePrecision().update(predictions, targets).compute()
print(map_result.map50_95)

confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=classes,
    conf_threshold=0.5,
)
print(confusion_matrix.matrix)
```

Keep low-confidence predictions when computing mAP: filtering at `0.5` before evaluation discards part of the precision-recall curve. The confusion matrix uses its own confidence threshold. Iterate over the complete loader, preserving one prediction/target pair per image even when either side is empty.

## Pose estimation

`KeyPointMeanAveragePrecision` scores pose estimation models the way COCO keypoint evaluation does: predictions are matched to targets by Object Keypoint Similarity (OKS) instead of IoU. Pass one `sv.KeyPoints` per image for predictions and targets. COCO visibility `v=1` (occluded) and `v=2` (visible) both map to `KeyPoints.visible=True` and `v=0` to `False`, since `pycocotools` counts every keypoint with `v>0`. To reproduce COCO numbers, store each target's annotated area in `targets.data["area"]`; otherwise the area of the box spanning its visible keypoints is used. Targets with no visible keypoint are COCO ignore regions only when their boxes are given as `targets.data["xyxy"]` in `(x_min, y_min, x_max, y_max)`; without boxes they are skipped, so a prediction on one counts as a false positive, and a class whose only targets are skipped scores `-1`. A prediction box in `predictions.data["xyxy"]` sets its object-size bucket, as the `bbox` of a COCO result entry does in `pycocotools`. Mark COCO crowd targets with `targets.data["iscrowd"]`; like `pycocotools`, they are ignore regions that any number of predictions may match. Predictions are ranked by `detection_confidence`, not keypoint `confidence`. Non-finite (NaN or infinite) coordinates are not rejected: a non-finite target keypoint counts as unlabelled, like `visible=False`, and a non-finite prediction keypoint adds nothing to OKS while its finite keypoints still count; `pycocotools` instead returns a NaN OKS when a labelled keypoint of the pair is NaN. A target area must be finite and non-negative; a zero area is accepted but leaves only exact-match keypoints counting. Clean such labels first.

```python
import supervision as sv
from supervision.metrics import KeyPointMeanAveragePrecision

metric = KeyPointMeanAveragePrecision()  # COCO sigmas for 17-point skeletons
for predictions, targets in zip(PREDICTIONS, TARGETS):
    metric.update(predictions, targets)

result = metric.compute()
print(result.map50_95)
```

## API reference

<div class="md-typeset">
    <h2><a href="#supervision.metrics.mean_average_precision.MeanAveragePrecision">MeanAveragePrecision</a></h2>
</div>

:::supervision.metrics.mean_average_precision.MeanAveragePrecision

<div class="md-typeset">
    <h2><a href="#supervision.metrics.mean_average_precision.MeanAveragePrecisionResult">MeanAveragePrecisionResult</a></h2>
</div>

:::supervision.metrics.mean_average_precision.MeanAveragePrecisionResult

<div class="md-typeset">
    <h2><a href="#supervision.dataset.formats.coco.get_coco_class_index_mapping">get_coco_class_index_mapping</a></h2>
</div>

:::supervision.dataset.formats.coco.get_coco_class_index_mapping

<div class="md-typeset">
    <h2><a href="#supervision.metrics.keypoint_mean_average_precision.KeyPointMeanAveragePrecision">KeyPointMeanAveragePrecision</a></h2>
</div>

:::supervision.metrics.keypoint_mean_average_precision.KeyPointMeanAveragePrecision

<div class="md-typeset">
    <h2><a href="#supervision.metrics.keypoint_mean_average_precision.KeyPointMeanAveragePrecisionResult">KeyPointMeanAveragePrecisionResult</a></h2>
</div>

:::supervision.metrics.keypoint_mean_average_precision.KeyPointMeanAveragePrecisionResult
