from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


def _mask_operands(
    detections: Detections, class_agnostic: bool, operation_name: str
) -> dict[str, Any]:
    """Build the keyword operands of a mask selection kernel.

    Args:
        detections: Detections with `mask` and `confidence` set.
        class_agnostic: Whether class ids are ignored.
        operation_name: Operation label used in missing-`class_id` errors.

    Returns:
        Keyword arguments for the mask kernels.
    """
    return {
        "predictions": detections._build_nms_predictions(
            class_agnostic, operation_name
        ),
        "masks": detections.mask,
    }


def _oriented_box_operands(
    detections: Detections, class_agnostic: bool, operation_name: str
) -> dict[str, Any]:
    """Build the keyword operands of an oriented-box selection kernel.

    Args:
        detections: Detections with oriented-box corners and `confidence` set.
        class_agnostic: Whether class ids are ignored.
        operation_name: Operation label used in missing-`class_id` errors.

    Returns:
        Keyword arguments for the oriented-box kernels.
    """
    return {
        "predictions": detections._build_nms_predictions(
            class_agnostic, operation_name
        ),
        "oriented_boxes": np.asarray(detections.data[ORIENTED_BOX_COORDINATES]),
    }


def _box_operands(
    detections: Detections, class_agnostic: bool, operation_name: str
) -> dict[str, Any]:
    """Build the keyword operands of an axis-aligned box selection kernel.

    Args:
        detections: Detections with `confidence` set.
        class_agnostic: Whether class ids are ignored.
        operation_name: Operation label used in missing-`class_id` errors.

    Returns:
        Keyword arguments for the axis-aligned box kernels.
    """
    return {
        "boxes": detections.xyxy,
        "scores": detections.confidence,
        "categories": detections._resolve_nms_categories(
            class_agnostic, operation_name
        ),
    }


@dataclass(frozen=True)
class OverlapStrategy:
    """How one geometry builds its operands and which kernels it runs.

    Attributes:
        name: Label of the geometry, for example `"oriented_box"`.
        family: Public `OverlapOn` value that can request this geometry.
        is_available: Whether a `Detections` carries the operand this geometry needs.
        operands: Builds the keyword operands shared by the three kernels.
        nms_kernel: Non-max suppression kernel returning a keep mask.
        soft_nms_kernel: Gaussian soft-NMS kernel returning decayed confidence.
        nmm_kernel: Non-max merge kernel returning merge groups.
    """

    name: str
    family: OverlapOn
    is_available: Callable[[Detections], bool]
    operands: Callable[[Detections, bool, str], dict[str, Any]]
    nms_kernel: Callable[..., Any]
    soft_nms_kernel: Callable[..., Any]
    nmm_kernel: Callable[..., Any]

    def nms_keep_mask(
        self,
        detections: Detections,
        class_agnostic: bool,
        iou_threshold: float,
        overlap_metric: OverlapMetric | str,
    ) -> npt.NDArray[np.bool_]:
        """Run non-max suppression and return the keep mask.

        Args:
            detections: Detections with `confidence` set.
            class_agnostic: Whether class ids are ignored.
            iou_threshold: Overlap threshold above which a detection is suppressed.
            overlap_metric: Metric measuring the overlap.

        Returns:
            A boolean array, `True` for each detection that survives.
        """
        overlap_metric = OverlapMetric.from_value(overlap_metric)
        operands = self.operands(detections, class_agnostic, "NMS")
        return cast(
            npt.NDArray[np.bool_],
            self.nms_kernel(
                **operands, iou_threshold=iou_threshold, overlap_metric=overlap_metric
            ),
        )

    def soft_nms_confidence(
        self,
        detections: Detections,
        class_agnostic: bool,
        sigma: float,
        overlap_metric: OverlapMetric | str,
    ) -> npt.NDArray[np.floating]:
        """Run Gaussian soft-NMS and return the decayed confidence.

        Args:
            detections: Detections with `confidence` set.
            class_agnostic: Whether class ids are ignored.
            sigma: Strength of the confidence decay.
            overlap_metric: Metric measuring the overlap.

        Returns:
            The decayed confidence, one value per detection in input order.
        """
        overlap_metric = OverlapMetric.from_value(overlap_metric)
        operands = self.operands(detections, class_agnostic, "Soft-NMS")
        return cast(
            npt.NDArray[np.floating],
            self.soft_nms_kernel(
                **operands, sigma=sigma, overlap_metric=overlap_metric
            ),
        )

    def nmm_groups(
        self,
        detections: Detections,
        class_agnostic: bool,
        iou_threshold: float,
        overlap_metric: OverlapMetric | str,
    ) -> list[list[int]]:
        """Run non-max merging and return the merge groups.

        Args:
            detections: Detections with `confidence` set.
            class_agnostic: Whether class ids are ignored.
            iou_threshold: Overlap threshold above which detections are merged.
            overlap_metric: Metric measuring the overlap.

        Returns:
            Lists of detection indices, each list being one merge group.
        """
        overlap_metric = OverlapMetric.from_value(overlap_metric)
        operands = self.operands(detections, class_agnostic, "NMM")
        return cast(
            list[list[int]],
            self.nmm_kernel(
                **operands, iou_threshold=iou_threshold, overlap_metric=overlap_metric
            ),
        )


#: Strategies in priority order: the richest geometry present wins.
_OVERLAP_STRATEGIES: tuple[OverlapStrategy, ...] = (
    OverlapStrategy(
        name="mask",
        family=OverlapOn.MASK,
        is_available=lambda detections: detections.mask is not None,
        operands=_mask_operands,
        nms_kernel=mask_non_max_suppression,
        soft_nms_kernel=mask_soft_non_max_suppression,
        nmm_kernel=mask_non_max_merge,
    ),
    OverlapStrategy(
        name="oriented_box",
        family=OverlapOn.BOX,
        is_available=lambda detections: ORIENTED_BOX_COORDINATES in detections.data,
        operands=_oriented_box_operands,
        nms_kernel=oriented_box_non_max_suppression,
        soft_nms_kernel=oriented_box_soft_non_max_suppression,
        nmm_kernel=oriented_box_non_max_merge,
    ),
    OverlapStrategy(
        name="box",
        family=OverlapOn.BOX,
        is_available=lambda detections: True,
        operands=_box_operands,
        nms_kernel=_box_non_max_suppression_from_arrays,
        soft_nms_kernel=_box_soft_non_max_suppression_from_arrays,
        nmm_kernel=_box_non_max_merge_from_arrays,
    ),
)


def resolve_overlap_strategy(
    detections: Detections, overlap_on: OverlapOn | None
) -> OverlapStrategy:
    """Pick the geometry an overlap-based operation compares detections on.

    The first strategy whose family `overlap_on` allows and whose operand the
    detections carry wins. With `overlap_on=None` every family is allowed, so
    the richest geometry present is used: mask, then oriented box, then
    axis-aligned box. `OverlapOn.BOX` skips the mask strategy and
    `OverlapOn.MASK` requires a mask.

    Args:
        detections: Detections the operation runs on.
        overlap_on: Requested geometry, or `None` to use the richest one present.

    Returns:
        The strategy to compare with.

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
        >>> resolve_overlap_strategy(detections, None).name
        'box'

        ```
    """
    for strategy in _OVERLAP_STRATEGIES:
        if overlap_on in (None, strategy.family) and strategy.is_available(detections):
            return strategy
    # The box strategy is always available, so only a mask request can fall through.
    raise ValueError(
        "overlap_on='mask' requires `Detections.mask` to be set; "
        "pass overlap_on='box' to compare boxes instead."
    )
