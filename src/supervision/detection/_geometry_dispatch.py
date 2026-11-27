from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.boxes import obb_polygon_area
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    OverlapOn,
    _box_non_max_merge_from_arrays,
    _box_non_max_suppression_from_arrays,
    _box_soft_non_max_suppression_from_arrays,
    box_iou_batch,
    mask_iou_batch,
    mask_non_max_merge,
    mask_non_max_suppression,
    mask_soft_non_max_suppression,
    oriented_box_iou_batch,
    oriented_box_non_max_merge,
    oriented_box_non_max_suppression,
    oriented_box_soft_non_max_suppression,
)
from supervision.detection.utils.masks import count_mask_pixels

if TYPE_CHECKING:
    from supervision.detection.core import Detections


def detection_area(detections: Detections) -> npt.NDArray[np.generic]:
    """Calculate detection areas using the richest geometry present.

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
    """Calculate pairwise IoU using the richest geometry both operands share.

    Selection order:

    1. If both operands have masks, use mask IoU.
    2. Else, if both operands have oriented-box coordinates, use OBB IoU.
    3. Otherwise, use axis-aligned box IoU.

    This is a **shared**-geometry dispatch, unlike
    :func:`~supervision.detection._geometry_dispatch.detection_area`'s
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


class _Geometry(Enum):
    """Geometry a selection operation compares detections on."""

    MASK = "mask"
    ORIENTED_BOX = "oriented_box"
    BOX = "box"


def resolve_overlap_geometry(
    detections: Detections, overlap_on: OverlapOn | None
) -> _Geometry:
    """Pick the geometry an overlap-based operation compares detections on.

    With `overlap_on=None` the richest geometry present wins: mask, then
    oriented box, then axis-aligned box. `OverlapOn.BOX` skips the mask rung and
    `OverlapOn.MASK` requires a mask.

    Args:
        detections: Detections the operation runs on.
        overlap_on: Requested geometry, or `None` to use the richest one present.

    Returns:
        The geometry to compare on.

    Raises:
        ValueError: If `overlap_on` is `OverlapOn.MASK` and `detections` has no
            mask.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> detections = Detections(
        ...     xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32)
        ... )
        >>> resolve_overlap_geometry(detections, None)
        <_Geometry.BOX: 'box'>

        ```
    """
    if overlap_on is OverlapOn.MASK:
        if detections.mask is None:
            raise ValueError(
                "overlap_on='mask' requires `Detections.mask` to be set; "
                "pass overlap_on='box' to compare boxes instead."
            )
        return _Geometry.MASK
    if overlap_on is None and detections.mask is not None:
        return _Geometry.MASK
    if ORIENTED_BOX_COORDINATES in detections.data:
        return _Geometry.ORIENTED_BOX
    return _Geometry.BOX


def _kernel_operands(
    detections: Detections,
    geometry: _Geometry,
    class_agnostic: bool,
    operation_name: str,
) -> dict[str, Any]:
    """Build the geometry-specific keyword operands of a selection kernel.

    Args:
        detections: Detections the operation runs on; `confidence` must be set.
        geometry: Geometry the kernel compares on.
        class_agnostic: Whether class ids are ignored.
        operation_name: Operation label used in missing-`class_id` errors.

    Returns:
        Keyword arguments for the kernel registered for `geometry`.
    """
    if geometry is _Geometry.MASK:
        return {
            "predictions": detections._build_nms_predictions(
                class_agnostic, operation_name
            ),
            "masks": detections.mask,
        }
    if geometry is _Geometry.ORIENTED_BOX:
        return {
            "predictions": detections._build_nms_predictions(
                class_agnostic, operation_name
            ),
            "oriented_boxes": np.asarray(detections.data[ORIENTED_BOX_COORDINATES]),
        }
    return {
        "boxes": detections.xyxy,
        "scores": detections.confidence,
        "categories": detections._resolve_nms_categories(
            class_agnostic, operation_name
        ),
    }


#: Kernel per geometry for each overlap-based operation.
_NMS_KERNELS: dict[_Geometry, Callable[..., Any]] = {
    _Geometry.MASK: mask_non_max_suppression,
    _Geometry.ORIENTED_BOX: oriented_box_non_max_suppression,
    _Geometry.BOX: _box_non_max_suppression_from_arrays,
}
_SOFT_NMS_KERNELS: dict[_Geometry, Callable[..., Any]] = {
    _Geometry.MASK: mask_soft_non_max_suppression,
    _Geometry.ORIENTED_BOX: oriented_box_soft_non_max_suppression,
    _Geometry.BOX: _box_soft_non_max_suppression_from_arrays,
}
_NMM_KERNELS: dict[_Geometry, Callable[..., Any]] = {
    _Geometry.MASK: mask_non_max_merge,
    _Geometry.ORIENTED_BOX: oriented_box_non_max_merge,
    _Geometry.BOX: _box_non_max_merge_from_arrays,
}


def nms_keep_mask(
    detections: Detections,
    geometry: _Geometry,
    class_agnostic: bool,
    iou_threshold: float,
    overlap_metric: OverlapMetric | str,
) -> npt.NDArray[np.bool_]:
    """Run non-max suppression on the geometry and return the keep mask.

    Args:
        detections: Detections with `confidence` set.
        geometry: Geometry to compare on.
        class_agnostic: Whether class ids are ignored.
        iou_threshold: Overlap threshold above which a detection is suppressed.
        overlap_metric: Metric measuring the overlap.

    Returns:
        A boolean array, `True` for each detection that survives.
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    operands = _kernel_operands(detections, geometry, class_agnostic, "NMS")
    return cast(
        npt.NDArray[np.bool_],
        _NMS_KERNELS[geometry](
            **operands, iou_threshold=iou_threshold, overlap_metric=overlap_metric
        ),
    )


def soft_nms_confidence(
    detections: Detections,
    geometry: _Geometry,
    class_agnostic: bool,
    sigma: float,
    overlap_metric: OverlapMetric | str,
) -> npt.NDArray[np.floating]:
    """Run Gaussian soft-NMS on the geometry and return the decayed confidence.

    Args:
        detections: Detections with `confidence` set.
        geometry: Geometry to compare on.
        class_agnostic: Whether class ids are ignored.
        sigma: Strength of the confidence decay.
        overlap_metric: Metric measuring the overlap.

    Returns:
        The decayed confidence, one value per detection in input order.
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    operands = _kernel_operands(detections, geometry, class_agnostic, "Soft-NMS")
    return cast(
        npt.NDArray[np.floating],
        _SOFT_NMS_KERNELS[geometry](
            **operands, sigma=sigma, overlap_metric=overlap_metric
        ),
    )


def nmm_groups(
    detections: Detections,
    geometry: _Geometry,
    class_agnostic: bool,
    iou_threshold: float,
    overlap_metric: OverlapMetric | str,
) -> list[list[int]]:
    """Run non-max merging on the geometry and return the merge groups.

    Args:
        detections: Detections with `confidence` set.
        geometry: Geometry to compare on.
        class_agnostic: Whether class ids are ignored.
        iou_threshold: Overlap threshold above which detections are merged.
        overlap_metric: Metric measuring the overlap.

    Returns:
        Lists of detection indices, each list being one merge group.
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    operands = _kernel_operands(detections, geometry, class_agnostic, "NMM")
    return cast(
        list[list[int]],
        _NMM_KERNELS[geometry](
            **operands, iou_threshold=iou_threshold, overlap_metric=overlap_metric
        ),
    )
