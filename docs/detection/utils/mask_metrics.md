---
comments: true
status: new
---

# Mask Metrics

These utilities compare a predicted binary segmentation mask with a target mask.
They are useful for segmentation model debugging, annotation QA, dataset
inspection, and contour-sensitive evaluation workflows.

Region-overlap metrics such as Dice and IoU focus on how much foreground area is
shared. Boundary metrics focus on how well the mask contours align. The
`tolerance` parameter controls how forgiving boundary matching is in pixel space.

Empty-mask behavior is explicit across all functions:

- both masks empty: `1.0`
- one mask empty: `0.0`

Current limitations:

- single mask pair only
- integer pixel tolerance only
- no ratio-based tolerance
- no spacing-aware tolerance
- no visualization or report helpers in this PR

Future work:

- `compare_masks` quality report helper
- false-positive / false-negative / true-positive mask decomposition
- boundary error maps
- visualization helper or `ComparisonAnnotator` integration
- dataset-level segmentation QA cookbook

<div class="md-typeset">
  <h2><a href="#supervision.detection.utils.mask_metrics.mask_iou">mask_iou</a></h2>
</div>

:::supervision.detection.utils.mask_metrics.mask_iou

<div class="md-typeset">
  <h2><a href="#supervision.detection.utils.mask_metrics.dice_coefficient">dice_coefficient</a></h2>
</div>

:::supervision.detection.utils.mask_metrics.dice_coefficient

<div class="md-typeset">
  <h2><a href="#supervision.detection.utils.mask_metrics.boundary_iou">boundary_iou</a></h2>
</div>

:::supervision.detection.utils.mask_metrics.boundary_iou

<div class="md-typeset">
  <h2><a href="#supervision.detection.utils.mask_metrics.boundary_f_score">boundary_f_score</a></h2>
</div>

:::supervision.detection.utils.mask_metrics.boundary_f_score
