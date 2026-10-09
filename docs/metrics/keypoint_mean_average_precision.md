---
comments: true
description: API reference for KeypointMeanAveragePrecision, the COCO keypoint mAP based on Object Keypoint Similarity (OKS) for pose estimation models.
---

# Keypoint Mean Average Precision

Install the metrics extra before using this API:

```bash
pip install "supervision[metrics]"
```

`KeypointMeanAveragePrecision` scores pose estimation models the way COCO keypoint evaluation does: predictions are matched to targets by Object Keypoint Similarity (OKS) instead of IoU. Pass one `sv.KeyPoints` per image for predictions and targets. COCO visibility `v=1` (occluded) and `v=2` (visible) both map to `KeyPoints.visible=True` and `v=0` to `False`, since `pycocotools` counts every keypoint with `v>0`. To reproduce COCO numbers, store each target's annotated area in `targets.data["area"]`; otherwise the area of the box spanning its visible keypoints is used. Targets with no visible keypoint are COCO ignore regions only when their boxes are given as `targets.data["xyxy"]` in `(x_min, y_min, x_max, y_max)`; without boxes they are skipped, so a prediction on one counts as a false positive. A prediction box in `predictions.data["xyxy"]` sets its object-size bucket, as the `bbox` of a COCO result entry does in `pycocotools`. Mark COCO crowd targets with `targets.data["iscrowd"]`; like `pycocotools`, they are ignore regions that any number of predictions may match. Predictions are ranked by `detection_confidence`, not keypoint `confidence`. Non-finite (NaN or infinite) coordinates are not rejected: a non-finite target keypoint counts as unlabelled, like `visible=False`, and a non-finite prediction keypoint adds nothing to OKS while its finite keypoints still count; `pycocotools` instead returns a NaN OKS when a labelled keypoint of the pair is NaN. A target area must be finite and non-negative; a zero area is accepted but leaves only exact-match keypoints counting. Clean such labels first.

```python
import supervision as sv
from supervision.metrics import KeypointMeanAveragePrecision

metric = KeypointMeanAveragePrecision()  # COCO sigmas for 17-point skeletons
for predictions, targets in zip(PREDICTIONS, TARGETS):
    metric.update(predictions, targets)

result = metric.compute()
print(result.map50_95)
```

<div class="md-typeset">
    <h2><a href="#supervision.metrics.keypoint_mean_average_precision.KeypointMeanAveragePrecision">KeypointMeanAveragePrecision</a></h2>
</div>

:::supervision.metrics.keypoint_mean_average_precision.KeypointMeanAveragePrecision

<div class="md-typeset">
    <h2><a href="#supervision.metrics.keypoint_mean_average_precision.KeypointMeanAveragePrecisionResult">KeypointMeanAveragePrecisionResult</a></h2>
</div>

:::supervision.metrics.keypoint_mean_average_precision.KeypointMeanAveragePrecisionResult
