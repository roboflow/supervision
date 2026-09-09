---
comments: true
---

# Match Detections

Match two `sv.Detections` objects into one-to-one pairs by greedy, highest-IoU-first assignment. The function returns index arrays, so the result composes with `sv.Detections` slicing.

```python
import supervision as sv

matched_pairs, unmatched_a, unmatched_b = sv.match_detections(
    detections_a,
    detections_b,
    iou_threshold=0.5,
    class_agnostic=False,
)
```

`matched_pairs` has shape `(M, 2)`: column 0 indexes `detections_a`, column 1 indexes `detections_b`. `unmatched_a` and `unmatched_b` hold the indices of the remaining detections on each side.

<div class="md-typeset">
    <h2><a href="#supervision.metrics.utils.matching.match_detections">match_detections</a></h2>
</div>

:::supervision.metrics.utils.matching.match_detections
