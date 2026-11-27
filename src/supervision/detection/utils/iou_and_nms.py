"""Compatibility import path for the overlap and suppression helpers.

The implementations live in :mod:`supervision.detection.utils.iou` and
:mod:`supervision.detection.utils.nms`; this module re-exports their public names so
existing imports keep working.
"""

from __future__ import annotations

from supervision.detection.utils.iou import (
    OverlapFilter,
    OverlapMetric,
    OverlapOn,
    box_iou,
    box_iou_batch,
    box_iou_batch_with_jaccard,
    compact_mask_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.detection.utils.nms import (
    box_non_max_merge,
    box_non_max_suppression,
    box_soft_non_max_suppression,
    box_weighted_box_fusion,
    mask_non_max_merge,
    mask_non_max_suppression,
    mask_soft_non_max_suppression,
    oriented_box_non_max_merge,
    oriented_box_non_max_suppression,
    oriented_box_soft_non_max_suppression,
)
from supervision.utils.internal import warn_deprecated

warn_deprecated(
    "`supervision.detection.utils.iou_and_nms` is deprecated in `0.33.0` and will be "
    "removed in `0.36.0`. Import from `supervision.detection.utils.iou` or "
    "`supervision.detection.utils.nms` instead."
)

__all__ = [
    "OverlapFilter",
    "OverlapMetric",
    "OverlapOn",
    "box_iou",
    "box_iou_batch",
    "box_iou_batch_with_jaccard",
    "box_non_max_merge",
    "box_non_max_suppression",
    "box_soft_non_max_suppression",
    "box_weighted_box_fusion",
    "compact_mask_iou_batch",
    "mask_iou_batch",
    "mask_non_max_merge",
    "mask_non_max_suppression",
    "mask_soft_non_max_suppression",
    "oriented_box_iou_batch",
    "oriented_box_non_max_merge",
    "oriented_box_non_max_suppression",
    "oriented_box_soft_non_max_suppression",
]
