from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.boxes import obb_polygon_area
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)

if TYPE_CHECKING:
    from supervision.detection.core import Detections


def detection_area(detections: Detections) -> npt.NDArray[np.generic]:
    """
    Calculate detection areas using the richest geometry present.

    Selection order:

    1. If ``mask`` is set, return the area of each mask.
    2. Else, if ``data[ORIENTED_BOX_COORDINATES]`` is set, return the area of
       each oriented box.
    3. Otherwise, return the axis-aligned box area.

    Args:
        detections: Detections whose geometry should be measured.

    Returns:
        A 1-D array containing the area of each detection.

    Example:
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> detections = Detections(
        ...     xyxy=np.array([[0, 0, 2, 3]], dtype=np.float32)
        ... )
        >>> detection_area(detections)  # doctest: +ELLIPSIS
        array([6.], dtype=float32)
    """
    if detections.mask is not None:
        if isinstance(detections.mask, CompactMask):
            return detections.mask.area
        mask_area = np.count_nonzero(detections.mask, axis=(1, 2)).astype(
            np.int64, copy=False
        )
        return cast(npt.NDArray[np.int64], mask_area)
    if ORIENTED_BOX_COORDINATES in detections.data:
        return obb_polygon_area(
            cast(npt.NDArray[np.number], detections.data[ORIENTED_BOX_COORDINATES])
        )
    return detections.box_area


def detection_iou(
    detections_true: Detections,
    detections_detection: Detections,
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """
    Calculate pairwise IoU using the richest geometry both operands share.

    Selection order:

    1. If both operands have masks, use mask IoU.
    2. Else, if both operands have oriented-box coordinates, use OBB IoU.
    3. Otherwise, use axis-aligned box IoU.

    Args:
        detections_true: Reference detections.
        detections_detection: Candidate detections.
        overlap_metric: Metric used to compute the degree of overlap.

    Returns:
        A matrix of pairwise overlaps with shape
            ``(len(detections_true), len(detections_detection))``.

    Example:
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> detections = Detections(
        ...     xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32)
        ... )
        >>> detection_iou(detections, detections)  # doctest: +ELLIPSIS
        array([[1.]], dtype=float32)
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    if detections_true.mask is not None and detections_detection.mask is not None:
        return mask_iou_batch(
            detections_true.mask,
            detections_detection.mask,
            overlap_metric=overlap_metric,
        )

    if (
        ORIENTED_BOX_COORDINATES in detections_true.data
        and ORIENTED_BOX_COORDINATES in detections_detection.data
    ):
        return oriented_box_iou_batch(
            cast(
                npt.NDArray[np.number],
                detections_true.data[ORIENTED_BOX_COORDINATES],
            ),
            cast(
                npt.NDArray[np.number],
                detections_detection.data[ORIENTED_BOX_COORDINATES],
            ),
            overlap_metric=overlap_metric,
        )

    return box_iou_batch(
        detections_true.xyxy,
        detections_detection.xyxy,
        overlap_metric=overlap_metric,
    )
