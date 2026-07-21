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
from supervision.detection.utils.masks import count_mask_pixels

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
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> detections = Detections(
        ...     xyxy=np.array([[0, 0, 2, 3]], dtype=np.float32)
        ... )
        >>> detection_area(detections)  # doctest: +ELLIPSIS
        array([6.], dtype=float32)

        ```
    """
    if detections.mask is not None:
        if isinstance(detections.mask, CompactMask):
            return detections.mask.area
        return count_mask_pixels(detections.mask)
    if ORIENTED_BOX_COORDINATES in detections.data:
        return obb_polygon_area(
            cast(npt.NDArray[np.number], detections.data[ORIENTED_BOX_COORDINATES])
        )
    return detections.box_area


def detection_iou(
    detections_a: Detections,
    detections_b: Detections,
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """
    Calculate pairwise IoU using the richest geometry both operands share.

    Selection order:

    1. If both operands have masks, use mask IoU.
    2. Else, if both operands have oriented-box coordinates, use OBB IoU.
    3. Otherwise, use axis-aligned box IoU.

    This is a **shared**-geometry dispatch, unlike `detection_area`'s
    per-operand dispatch: if the two operands carry different kinds of
    richer geometry (e.g. one has only a mask, the other only oriented-box
    coordinates), neither is used and the axis-aligned box IoU is computed
    instead.

    Args:
        detections_a: First set of detections.
        detections_b: Second set of detections.
        overlap_metric: Metric used to compute the degree of overlap.

    Returns:
        A matrix of pairwise overlaps with shape
            ``(len(detections_a), len(detections_b))``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> detections = Detections(
        ...     xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32)
        ... )
        >>> detection_iou(detections, detections)  # doctest: +ELLIPSIS
        array([[1.]], dtype=float32)

        ```
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    if detections_a.mask is not None and detections_b.mask is not None:
        return mask_iou_batch(
            detections_a.mask,
            detections_b.mask,
            overlap_metric=overlap_metric,
        )

    if (
        ORIENTED_BOX_COORDINATES in detections_a.data
        and ORIENTED_BOX_COORDINATES in detections_b.data
    ):
        return oriented_box_iou_batch(
            cast(
                npt.NDArray[np.number],
                detections_a.data[ORIENTED_BOX_COORDINATES],
            ),
            cast(
                npt.NDArray[np.number],
                detections_b.data[ORIENTED_BOX_COORDINATES],
            ),
            overlap_metric=overlap_metric,
        )

    return box_iou_batch(
        detections_a.xyxy,
        detections_b.xyxy,
        overlap_metric=overlap_metric,
    )
