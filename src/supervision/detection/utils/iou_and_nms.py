from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision import _cv2 as cv2
from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.converters import mask_to_xyxy
from supervision.utils.internal import warn_deprecated


class OverlapFilter(Enum):
    """
    Enum specifying the strategy for filtering overlapping detections.

    Attributes:
        NONE: Do not filter detections based on overlap.
        NON_MAX_SUPPRESSION: Filter detections using non-max suppression. This means,
            detections that overlap by more than a set threshold will be discarded,
            except for the one with the highest confidence.
        NON_MAX_MERGE: Merge detections with non-max merging. This means,
            detections that overlap by more than a set threshold will be merged
            into a single detection.
    """

    NONE = "none"
    NON_MAX_SUPPRESSION = "non_max_suppression"
    NON_MAX_MERGE = "non_max_merge"

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda member: member.value, cls))

    @classmethod
    def from_value(cls, value: OverlapFilter | str) -> OverlapFilter:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
        )


class OverlapMetric(Enum):
    """
    Enum specifying the metric for measuring overlap between detections.

    Attributes:
        IOU: Intersection over Union. A region-overlap metric that compares
            two shapes (usually bounding boxes or masks) by normalising the
            shared area with the area of their union.
        IOS: Intersection over Smaller, a region-overlap metric that compares
            two shapes (usually bounding boxes or masks) by normalising the
            shared area with the smaller of the two shapes.
    """

    IOU = "IOU"
    IOS = "IOS"

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda member: member.value, cls))

    @classmethod
    def from_value(cls, value: OverlapMetric | str) -> OverlapMetric:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.upper()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
        )


def _validate_iou_threshold(iou_threshold: float) -> None:
    """Raise `ValueError` when an IoU threshold falls outside `[0, 1]`."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError(
            "Value of `iou_threshold` must be in the closed range from 0 to 1, "
            f"{iou_threshold} given."
        )


def box_iou(
    box_true: list[float] | npt.NDArray[np.floating],
    box_detection: list[float] | npt.NDArray[np.floating],
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> float:
    """
    Compute overlap metric between two bounding boxes.

    Supports standard IOU (intersection-over-union) and IOS
    (intersection-over-smaller-area) metrics. Returns the overlap value in range
    `[0, 1]`.

    Args:
        box_true: Ground truth box in format
          `(x_min, y_min, x_max, y_max)`.
        box_detection: Detected box in format
          `(x_min, y_min, x_max, y_max)`.
        overlap_metric: Overlap type.
          Use `OverlapMetric.IOU` for IOU or
          `OverlapMetric.IOS` for IOS. Defaults to `OverlapMetric.IOU`.

    Returns:
        Overlap value between boxes in `[0, 1]`.

    Raises:
        ValueError: If `overlap_metric` is not IOU or IOS.

    Examples:
        ```pycon
        >>> import supervision as sv
        >>> box_true = [100, 100, 200, 200]
        >>> box_detection = [150, 150, 250, 250]
        >>> sv.box_iou(box_true, box_detection, overlap_metric=sv.OverlapMetric.IOU)
        0.142857...
        >>> sv.box_iou(box_true, box_detection, overlap_metric=sv.OverlapMetric.IOS)
        0.25

        ```
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    x_min_true, y_min_true, x_max_true, y_max_true = np.array(box_true)
    x_min_det, y_min_det, x_max_det, y_max_det = np.array(box_detection)

    x_min_inter = max(x_min_true, x_min_det)
    y_min_inter = max(y_min_true, y_min_det)
    x_max_inter = min(x_max_true, x_max_det)
    y_max_inter = min(y_max_true, y_max_det)

    inter_w = max(0.0, x_max_inter - x_min_inter)
    inter_h = max(0.0, y_max_inter - y_min_inter)

    area_inter = inter_w * inter_h

    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    area_det = (x_max_det - x_min_det) * (y_max_det - y_min_det)

    if overlap_metric == OverlapMetric.IOU:
        area_norm = area_true + area_det - area_inter
    elif overlap_metric == OverlapMetric.IOS:
        area_norm = min(area_true, area_det)
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )

    if area_norm <= 0.0:
        return 0.0

    return float(area_inter / area_norm)


def box_iou_batch(
    boxes_true: npt.NDArray[np.number],
    boxes_detection: npt.NDArray[np.number],
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> npt.NDArray[np.float32]:
    """
    Compute pairwise overlap scores between batches of bounding boxes.

    Supports standard IOU (intersection-over-union) and IOS
    (intersection-over-smaller-area) metrics for all `boxes_true` and
    `boxes_detection` pairs. Returns a matrix of overlap values in range
    `[0, 1]`, matching each box from the first batch to each from the second.

    Args:
        boxes_true: Array of reference boxes in
            shape `(N, 4)` as `(x_min, y_min, x_max, y_max)`.
        boxes_detection: Array of detected boxes in
            shape `(M, 4)` as `(x_min, y_min, x_max, y_max)`.
        overlap_metric: Overlap type.
            Use `OverlapMetric.IOU` for intersection-over-union,
            `OverlapMetric.IOS` for intersection-over-smaller-area.
            Defaults to `OverlapMetric.IOU`.

    Returns:
        Overlap matrix of shape `(N, M)`, where entry
            `[i, j]` is the overlap between `boxes_true[i]` and
            `boxes_detection[j]`.

    Raises:
        ValueError: If `overlap_metric` is not IOU or IOS.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> boxes_true = np.array([
        ...     [100, 100, 200, 200],
        ...     [300, 300, 400, 400]
        ... ])
        >>> boxes_detection = np.array([
        ...     [150, 150, 250, 250],
        ...     [320, 320, 420, 420]
        ... ])
        >>> sv.box_iou_batch(
        ...     boxes_true, boxes_detection, overlap_metric=sv.OverlapMetric.IOU
        ... )
        array([[0.14285..., 0.        ],
               [0.        , 0.47058...]], dtype=float32)
        >>> sv.box_iou_batch(
        ...     boxes_true, boxes_detection, overlap_metric=sv.OverlapMetric.IOS
        ... )
        array([[0.25, 0.  ],
               [0.  , 0.64]], dtype=float32)

        ```
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    # Upcast the corners to float64 right after unpacking so every subtraction
    # and multiplication below runs in float64. This prevents integer-dtype
    # overflow: an int32 box area such as 50000 * 50000 = 2.5e9 wraps to a
    # negative value in int32 before any later cast could run, yielding wrong
    # (often zero) IoU. Upcasting here also gives full float64 precision to
    # float64/int64 callers. It does NOT recover precision already lost when a
    # caller stores coordinates as float32 upstream, because that float32
    # rounding happens before this function is ever called. The final matrix is
    # cast back to float32 to preserve the public return-type contract.
    x_min_true, y_min_true, x_max_true, y_max_true = boxes_true.T.astype(np.float64)
    x_min_det, y_min_det, x_max_det, y_max_det = boxes_detection.T.astype(np.float64)
    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]

    if count_true == 0 or count_det == 0:
        return cast(
            npt.NDArray[np.float32], np.empty((count_true, count_det), dtype=np.float32)
        )

    x_min_inter = np.empty((count_true, count_det), dtype=np.float64)
    x_max_inter = np.empty_like(x_min_inter)
    y_min_inter = np.empty_like(x_min_inter)
    y_max_inter = np.empty_like(x_min_inter)

    np.maximum(x_min_true[:, None], x_min_det[None, :], out=x_min_inter)
    np.minimum(x_max_true[:, None], x_max_det[None, :], out=x_max_inter)
    np.maximum(y_min_true[:, None], y_min_det[None, :], out=y_min_inter)
    np.minimum(y_max_true[:, None], y_max_det[None, :], out=y_max_inter)

    # we reuse x_max_inter and y_max_inter to store inter_w and inter_h
    np.subtract(x_max_inter, x_min_inter, out=x_max_inter)  # inter_w
    np.subtract(y_max_inter, y_min_inter, out=y_max_inter)  # inter_h
    np.clip(x_max_inter, 0.0, None, out=x_max_inter)
    np.clip(y_max_inter, 0.0, None, out=y_max_inter)

    area_inter = x_max_inter * y_max_inter  # inter_w * inter_h

    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    area_det = (x_max_det - x_min_det) * (y_max_det - y_min_det)

    if overlap_metric == OverlapMetric.IOU:
        area_norm = area_true[:, None] + area_det[None, :] - area_inter
    elif overlap_metric == OverlapMetric.IOS:
        area_norm = np.minimum(area_true[:, None], area_det[None, :])
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )

    out: npt.NDArray[np.float32] = np.zeros_like(area_inter, dtype=np.float32)
    np.divide(area_inter, area_norm, out=out, where=area_norm > 0)
    return out


def box_iou_batch_with_jaccard(
    boxes_true: Sequence[Sequence[float]],
    boxes_detection: Sequence[Sequence[float]],
    is_crowd: Sequence[bool],
) -> npt.NDArray[np.float64]:
    """
    Calculate the intersection over union (IoU) between detection bounding boxes (dt)
    and ground-truth bounding boxes (gt).
    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics

    Args:
        boxes_true: Sequence of ground-truth bounding boxes in the
            format [x, y, width, height].
        boxes_detection: Sequence of detection bounding boxes in the
            format [x, y, width, height].
        is_crowd: Sequence indicating if each ground-truth bounding box
            is a crowd region or not.

    Note:
        This function expects bounding boxes in ``[x, y, width, height]`` format
        (COCO convention). All other batch IoU functions in this module use
        ``[x_min, y_min, x_max, y_max]``.

        NaN coordinates propagate silently: if any box value is ``NaN``, the
        corresponding IoU values will be ``NaN``.

    Returns:
        Array of IoU values of shape ``(len(boxes_detection), len(boxes_true))``,
        where row ``i`` contains the IoU of detection ``i`` against all ground-truth
        boxes, and column ``j`` contains the IoU of all detections against ground-truth
        box ``j``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> boxes_true = [
        ...     [10, 20, 30, 40],  # x, y, w, h
        ...     [15, 25, 35, 45]
        ... ]
        >>> boxes_detection = [
        ...     [12, 22, 28, 38],
        ...     [16, 26, 36, 46]
        ... ]
        >>> is_crowd = [False, False]
        >>> ious = sv.box_iou_batch_with_jaccard(
        ...     boxes_true=boxes_true,
        ...     boxes_detection=boxes_detection,
        ...     is_crowd=is_crowd
        ... )
        >>> ious  # doctest: +ELLIPSIS
        array([[0.886..., 0.496...],
               [0.4  ..., 0.862...]])

        ```
    """
    if len(is_crowd) != len(boxes_true):
        raise ValueError(
            f"`is_crowd` length ({len(is_crowd)}) must match "
            f"`boxes_true` length ({len(boxes_true)})."
        )
    if len(boxes_detection) == 0 or len(boxes_true) == 0:
        return np.empty((len(boxes_detection), len(boxes_true)), dtype=np.float64)

    # Smallest number to avoid division by zero.
    eps = np.spacing(1)
    gt = np.asarray(boxes_true, dtype=np.float64)
    dt = np.asarray(boxes_detection, dtype=np.float64)
    crowd = np.asarray(is_crowd, dtype=bool)

    # Boxes are [x, y, w, h]. Build the far corners as `x2 = x + w` (rather than
    # reusing `w`) so that the area/intersection arithmetic is bit-identical to
    # the per-pair reference it replaces.
    gt_x2, gt_y2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    dt_x2, dt_y2 = dt[:, 0] + dt[:, 2], dt[:, 1] + dt[:, 3]

    # Pairwise intersection: rows index detections, columns index ground truth.
    inter_x1 = np.maximum(dt[:, 0][:, None], gt[:, 0][None, :])
    inter_y1 = np.maximum(dt[:, 1][:, None], gt[:, 1][None, :])
    inter_x2 = np.minimum(dt_x2[:, None], gt_x2[None, :])
    inter_y2 = np.minimum(dt_y2[:, None], gt_y2[None, :])
    area_inter = np.maximum(inter_x2 - inter_x1, 0.0) * np.maximum(
        inter_y2 - inter_y1, 0.0
    )

    area_det = np.maximum(dt_x2 - dt[:, 0], 0.0) * np.maximum(dt_y2 - dt[:, 1], 0.0)
    area_gt = np.maximum(gt_x2 - gt[:, 0], 0.0) * np.maximum(gt_y2 - gt[:, 1], 0.0)

    # For a crowd ground truth a detection may match any subregion, so its union
    # collapses to the detection area; otherwise use the standard box union.
    iou: npt.NDArray[np.float64] = np.empty((len(dt), len(gt)), dtype=np.float64)
    if not np.any(crowd):
        area_norm = area_det[:, None] + area_gt[None, :] - area_inter + eps
    else:
        area_norm = np.where(
            crowd[None, :],
            area_det[:, None] + eps,
            area_det[:, None] + area_gt[None, :] - area_inter + eps,
        )
    np.divide(area_inter, area_norm, out=iou)
    return iou


def _polygon_areas(polygons: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Compute the area of each oriented-box polygon using the shoelace formula.

    Args:
        polygons: ``(N, 4, 2)`` array of polygon corners.

    Returns:
        ``(N,)`` array of polygon areas.
    """
    x = polygons[:, :, 0]
    y = polygons[:, :, 1]
    cross = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
    return cast(npt.NDArray[np.floating], 0.5 * np.abs(cross.sum(axis=1)))


def _aabb_envelopes(polygons: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Compute the axis-aligned bounding envelope of each oriented box.

    Args:
        polygons: ``(N, 4, 2)`` array of polygon corners.

    Returns:
        ``(N, 4)`` array of ``(x_min, y_min, x_max, y_max)`` envelopes.
    """
    xs = polygons[:, :, 0]
    ys = polygons[:, :, 1]
    return np.stack(
        [xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1
    )


def _overlapping_envelope_pairs(
    envelopes_true: npt.NDArray[np.floating],
    envelopes_detection: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return index pairs ``(i, j)`` whose axis-aligned envelopes overlap.

    Uses a fused boolean evaluation to halve peak transient memory compared to
    named-intermediate form (4 separate NxM float64 arrays vs 1 boolean array).

    Note:
        This gate is a correctness guarantee, not an approximation: if two
        axis-aligned bounding boxes do not overlap, the convex polygons they
        contain cannot overlap either.

    Args:
        envelopes_true: ``(N, 4)`` array of ``(x_min, y_min, x_max, y_max)``
            envelopes for the ground-truth boxes.
        envelopes_detection: ``(M, 4)`` array of ``(x_min, y_min, x_max, y_max)``
            envelopes for the detection boxes.

    Returns:
        A pair of 1-D index arrays ``(rows, cols)`` identifying the overlapping
        pairs.
    """
    et = envelopes_true[:, None, :]
    ed = envelopes_detection[None, :, :]
    overlap = (
        np.minimum(et[..., 2], ed[..., 2]) > np.maximum(et[..., 0], ed[..., 0])
    ) & (np.minimum(et[..., 3], ed[..., 3]) > np.maximum(et[..., 1], ed[..., 1]))
    return cast(tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], np.nonzero(overlap))


def oriented_box_iou_batch(
    boxes_true: npt.NDArray[np.number],
    boxes_detection: npt.NDArray[np.number],
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """
    Compute pairwise overlap scores between two sets of oriented bounding boxes
    using the configured `overlap_metric`.

    Overlap areas are computed exactly via convex-polygon intersection, gated by
    a cheap axis-aligned envelope pre-filter — no rasterization is involved, so
    the result is exact (free of pixel-quantization error) and independent of the
    coordinate magnitudes.

    `boxes_true` and `boxes_detection` are expected to be in
    `((x1, y1), (x2, y2), (x3, y3), (x4, y4))` format.

    Note:
        Inputs must be **convex** quads with finite coordinates. Self-intersecting
        or non-convex polygons produce undefined results via
        ``cv2.intersectConvexConvex``. NaN or Inf coordinates propagate silently
        as ``0.0`` — validate inputs before calling if needed.

        When ``boxes_true is boxes_detection`` (the same Python object, not just
        equal values), the function computes only the upper triangle of the
        matrix and mirrors it. This optimization is used automatically by the
        NMS/NMM callers that pass the same array twice. A defensive ``.copy()``
        at the call site would disable the optimization silently — see the
        NMS caller comment for context.

    Args:
        boxes_true: A `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4, 2)` where `N` is number of true objects.
            Last axis convention: `[..., 0]` = x-coordinates,
            `[..., 1]` = y-coordinates.
        boxes_detection: A `np.ndarray` representing detection boxes.
            `shape = (M, 4, 2)` where `M` is number of detected objects.
            Last axis convention: `[..., 0]` = x-coordinates,
            `[..., 1]` = y-coordinates.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of oriented boxes (e.g., IoU, IoS).

    Returns:
        Overlap matrix of shape `(N, M)`, where entry `[i, j]` is the overlap
        score between `boxes_true[i]` and `boxes_detection[j]`, in the range
        `[0, 1]` under the configured :attr:`overlap_metric`.

    Raises:
        ValueError: If ``boxes_true`` or ``boxes_detection`` is 3-D with inner
            dimensions other than ``(4, 2)``.
        ValueError: If ``boxes_true`` or ``boxes_detection`` is 2-D with a
            column count other than 8.
        ValueError: If ``boxes_true`` or ``boxes_detection`` is not 2-D or 3-D.
        ValueError: If ``overlap_metric`` is not
            :attr:`~supervision.config.OverlapMetric.IOU` or
            :attr:`~supervision.config.OverlapMetric.IOS`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> a = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32)
        >>> b = np.array([[[1, 0], [3, 0], [3, 2], [1, 2]]], dtype=np.float32)
        >>> sv.oriented_box_iou_batch(a, b)  # doctest: +ELLIPSIS
        array([[0.333...]])

        ```
    """

    for name, arr in (("boxes_true", boxes_true), ("boxes_detection", boxes_detection)):
        if arr.ndim == 3 and arr.shape[1:] != (4, 2):
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 4, 2) "
                f"— each box must have exactly 4 corners with (x, y) coordinates."
            )
        elif arr.ndim == 2 and arr.shape[1] != 8:
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 8) for flat "
                f"YOLO format or (N, 4, 2) for corner format."
            )
        elif arr.ndim not in (2, 3):
            raise ValueError(
                f"`{name}` must be 2-D (N, 8) or 3-D (N, 4, 2), got shape {arr.shape}."
            )

    if overlap_metric == OverlapMetric.IOU:
        normalize_by_union = True
    elif overlap_metric == OverlapMetric.IOS:
        normalize_by_union = False
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )

    # Capture identity before reshape: NMS / NMM pass the same array twice, so
    # the matrix is symmetric and we can compute only its upper triangle.
    is_self_comparison = boxes_true is boxes_detection
    boxes_true = cast(
        npt.NDArray[np.floating], boxes_true.reshape(-1, 4, 2).astype(np.float64)
    )
    boxes_detection = cast(
        npt.NDArray[np.floating],
        boxes_detection.reshape(-1, 4, 2).astype(np.float64),
    )

    n, m = len(boxes_true), len(boxes_detection)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    areas_true = _polygon_areas(boxes_true)
    areas_detection = _polygon_areas(boxes_detection)

    envelopes_true = _aabb_envelopes(boxes_true)
    envelopes_detection = (
        envelopes_true if is_self_comparison else _aabb_envelopes(boxes_detection)
    )
    rows, cols = _overlapping_envelope_pairs(envelopes_true, envelopes_detection)
    if is_self_comparison:
        upper = rows <= cols
        rows, cols = rows[upper], cols[upper]

    polygons_true = [box.astype(np.float32) for box in boxes_true]
    polygons_detection = [box.astype(np.float32) for box in boxes_detection]

    ious: npt.NDArray[np.float64] = np.zeros((n, m), dtype=np.float64)
    for i, j in zip(rows, cols):
        intersection, _ = cv2.intersectConvexConvex(
            polygons_true[i], polygons_detection[j]
        )
        if intersection <= 0:
            continue
        denominator = (
            areas_true[i] + areas_detection[j] - intersection
            if normalize_by_union
            else min(areas_true[i], areas_detection[j])
        )
        if denominator > 0:
            score = intersection / denominator
            ious[i, j] = score
            if is_self_comparison:
                ious[j, i] = score

    # DO NOT remove this clip. cv2.intersectConvexConvex computes in float32
    # internally while polygon areas are computed in float64; the intersection
    # area can exceed the float64 area by ~25 ULP (~1e-7), producing raw IoU
    # or IoS values microscopically above 1.0 for identical boxes. The clip is
    # load-bearing, not defensive duplication.
    return cast(npt.NDArray[np.floating], np.clip(ious, 0.0, 1.0))


def compact_mask_iou_batch(
    masks_true: Any,
    masks_detection: Any,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Compute pairwise overlap between two :class:`CompactMask` collections.

    Avoids materialising full ``(N, H, W)`` arrays by:

    1. Vectorised bounding-box pre-filter — pairs whose boxes do not overlap
       get IoU = 0 without any mask decoding.
    2. Sub-crop decoding — for overlapping pairs, only the intersection region
       of each crop is decoded and compared.
    3. Crop caching — each individual crop is decoded at most once even when it
       participates in many pairs.

    The result is numerically identical to running the dense
    :func:`mask_iou_batch` on ``np.asarray(masks_true)`` /
    ``np.asarray(masks_detection)``.

    Args:
        masks_true: :class:`~supervision.detection.compact_mask.CompactMask`
            holding the ground-truth masks.
        masks_detection: :class:`~supervision.detection.compact_mask.CompactMask`
            holding the detection masks.
        overlap_metric: :class:`OverlapMetric` — ``IOU`` or ``IOS``.

    Returns:
        Float array of shape ``(N1, N2)`` with pairwise overlap values.
    """
    n1: int = len(masks_true)
    n2: int = len(masks_detection)
    result: npt.NDArray[np.floating] = np.zeros((n1, n2), dtype=float)

    if n1 == 0 or n2 == 0:
        return result

    areas_a: npt.NDArray[np.int64] = masks_true.area
    areas_b: npt.NDArray[np.int64] = masks_detection.area

    # Inclusive per-mask bounding boxes obtained from public accessors.
    # bbox_xyxy: (N, 4) → (x1, y1, x2, y2)
    bboxes_a: npt.NDArray[np.int32] = masks_true.bbox_xyxy.astype(np.int32)
    x1a: npt.NDArray[np.int32] = bboxes_a[:, 0]
    y1a: npt.NDArray[np.int32] = bboxes_a[:, 1]
    x2a: npt.NDArray[np.int32] = bboxes_a[:, 2]
    y2a: npt.NDArray[np.int32] = bboxes_a[:, 3]

    bboxes_b: npt.NDArray[np.int32] = masks_detection.bbox_xyxy.astype(np.int32)
    x1b: npt.NDArray[np.int32] = bboxes_b[:, 0]
    y1b: npt.NDArray[np.int32] = bboxes_b[:, 1]
    x2b: npt.NDArray[np.int32] = bboxes_b[:, 2]
    y2b: npt.NDArray[np.int32] = bboxes_b[:, 3]

    # Pairwise intersection bounding box — shape (N1, N2).
    ix1: npt.NDArray[np.int32] = np.maximum(x1a[:, None], x1b[None, :])
    iy1: npt.NDArray[np.int32] = np.maximum(y1a[:, None], y1b[None, :])
    ix2: npt.NDArray[np.int32] = np.minimum(x2a[:, None], x2b[None, :])
    iy2: npt.NDArray[np.int32] = np.minimum(y2a[:, None], y2b[None, :])
    bbox_overlap: npt.NDArray[np.bool_] = (ix1 <= ix2) & (iy1 <= iy2)

    # Decode each crop at most once, even if it participates in many pairs.
    crops_a: dict[int, npt.NDArray[np.bool_]] = {}
    crops_b: dict[int, npt.NDArray[np.bool_]] = {}

    for idx_pair in np.argwhere(bbox_overlap):
        idx_a, idx_b = int(idx_pair[0]), int(idx_pair[1])

        if idx_a not in crops_a:
            crops_a[idx_a] = masks_true.crop(idx_a)
        if idx_b not in crops_b:
            crops_b[idx_b] = masks_detection.crop(idx_b)

        lx1 = int(ix1[idx_a, idx_b])
        ly1 = int(iy1[idx_a, idx_b])
        lx2 = int(ix2[idx_a, idx_b])
        ly2 = int(iy2[idx_a, idx_b])

        ox_a, oy_a = int(x1a[idx_a]), int(y1a[idx_a])
        sub_a = crops_a[idx_a][ly1 - oy_a : ly2 - oy_a + 1, lx1 - ox_a : lx2 - ox_a + 1]

        ox_b, oy_b = int(x1b[idx_b]), int(y1b[idx_b])
        sub_b = crops_b[idx_b][ly1 - oy_b : ly2 - oy_b + 1, lx1 - ox_b : lx2 - ox_b + 1]

        inter = int(np.logical_and(sub_a, sub_b).sum())
        area_a_i = int(areas_a[idx_a])
        area_b_j = int(areas_b[idx_b])

        if overlap_metric == OverlapMetric.IOU:
            union = area_a_i + area_b_j - inter
            result[idx_a, idx_b] = inter / union if union > 0 else 0.0
        elif overlap_metric == OverlapMetric.IOS:
            small = min(area_a_i, area_b_j)
            result[idx_a, idx_b] = inter / small if small > 0 else 0.0
        else:
            raise ValueError(
                f"overlap_metric {overlap_metric} is not supported, "
                "only 'IOU' and 'IOS' are supported"
            )

    return result


def _mask_iou_batch_split(
    masks_true: npt.NDArray[Any],
    masks_detection: npt.NDArray[Any],
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """
    Internal function.
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true: 3D `np.ndarray` representing ground-truth masks.
        masks_detection: 3D `np.ndarray` representing detection masks.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """
    # The overlap of two binary masks is the dot product of their flattened
    # pixels, so the whole (N, M) intersection matrix is a single matmul.
    # float32 counts pixels exactly up to 2**24; for larger masks (beyond
    # ~4096x4096) we promote to float64 so the counts stay exact.
    pixels = int(np.prod(masks_true.shape[1:]))
    count_dtype = np.float32 if pixels <= 2**24 else np.float64
    true_flat = cast(
        npt.NDArray[np.floating],
        masks_true.reshape(masks_true.shape[0], pixels).astype(count_dtype, copy=False),
    )
    detection_flat = cast(
        npt.NDArray[np.floating],
        masks_detection.reshape(masks_detection.shape[0], pixels).astype(
            count_dtype, copy=False
        ),
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        intersection_area: npt.NDArray[np.floating[Any]] = true_flat @ detection_flat.T

    masks_true_area = true_flat.sum(axis=1)
    masks_detection_area = detection_flat.sum(axis=1)

    if overlap_metric == OverlapMetric.IOU:
        union_area = masks_true_area[:, None] + masks_detection_area - intersection_area
        ious = np.divide(
            intersection_area,
            union_area,
            out=np.zeros_like(intersection_area, dtype=float),
            where=union_area != 0,
        )
    elif overlap_metric == OverlapMetric.IOS:
        # ios = intersection_area / min(area1, area2)
        small_area = np.minimum(masks_true_area[:, None], masks_detection_area)
        ious = np.divide(
            intersection_area,
            small_area,
            out=np.zeros_like(intersection_area, dtype=float),
            where=small_area != 0,
        )
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )

    ious = np.nan_to_num(ious)
    return cast(npt.NDArray[np.floating], ious)


def mask_iou_batch(
    masks_true: npt.NDArray[Any] | CompactMask,
    masks_detection: npt.NDArray[Any] | CompactMask,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    memory_limit: int = 1024 * 5,
) -> npt.NDArray[np.floating]:
    """
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Accepts both dense ``(N, H, W)`` boolean arrays and
    :class:`~supervision.detection.compact_mask.CompactMask` objects.
    When both inputs are :class:`~supervision.detection.compact_mask.CompactMask`,
    the computation uses :func:`compact_mask_iou_batch` to avoid materialising
    full ``(N, H, W)`` arrays.

    Args:
        masks_true: 3D `np.ndarray` representing ground-truth masks.
        masks_detection: 3D `np.ndarray` representing detection masks.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        memory_limit: Memory limit in MB, default is 1024 * 5 MB (5GB).
            Controls chunking of ``masks_true`` so that flattened detection
            masks plus each chunk's buffers stay within this limit. A
            ``UserWarning`` is raised when ``masks_detection`` alone
            exceeds the limit, as chunking cannot reduce peak memory
            below that floor. Ignored when both inputs are
            :class:`~supervision.detection.compact_mask.CompactMask`.

    Returns:
        Pairwise IoU of masks from `masks_true` and `masks_detection`.

    Raises:
        ValueError: If ``masks_true`` or ``masks_detection`` are not 3D
            ``(N, H, W)`` arrays, or if they do not share the same
            spatial dimensions ``(H, W)``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> masks_true = np.zeros((1, 4, 4), dtype=bool)
        >>> masks_true[:, :2, :2] = True
        >>> masks_detection = np.zeros((1, 4, 4), dtype=bool)
        >>> masks_detection[:, :3, :3] = True
        >>> sv.mask_iou_batch(masks_true, masks_detection)
        array([[0.44444445]])

        ```
    """

    if isinstance(masks_true, CompactMask) and isinstance(masks_detection, CompactMask):
        return compact_mask_iou_batch(masks_true, masks_detection, overlap_metric)

    # Materialise any CompactMask that was passed alongside a dense array.
    if isinstance(masks_true, CompactMask):
        masks_true = np.asarray(masks_true)
    if isinstance(masks_detection, CompactMask):
        masks_detection = np.asarray(masks_detection)

    if masks_true.ndim != 3 or masks_detection.ndim != 3:
        raise ValueError(
            "masks_true and masks_detection must be 3D (N, H, W); got "
            f"ndim={masks_true.ndim} and ndim={masks_detection.ndim}."
        )
    if masks_true.shape[1:] != masks_detection.shape[1:]:
        raise ValueError(
            "masks_true and masks_detection must share the same (H, W); got "
            f"{masks_true.shape[1:]} and {masks_detection.shape[1:]}."
        )
    # A single pass already handles empty inputs and avoids np.vstack([]) below.
    if masks_true.shape[0] == 0 or masks_detection.shape[0] == 0:
        return _mask_iou_batch_split(masks_true, masks_detection, overlap_metric)

    # Peak memory of a single matmul pass: the flattened detection masks (shared
    # across chunks) plus, per true-mask row, its flattened pixels and the three
    # (N, M) matrices it touches (intersection, denominator and output). The
    # previous (N, M, H, W) estimate overcounted by a factor of M and forced
    # needless chunking now that the intersection is a matmul.
    pixels = masks_true.shape[1] * masks_true.shape[2]
    itemsize = 4 if pixels <= 2**24 else 8
    limit_bytes = memory_limit * 1024 * 1024
    detection_bytes = masks_detection.shape[0] * pixels * itemsize
    per_true_row = pixels * itemsize + 3 * masks_detection.shape[0] * 8
    if detection_bytes > limit_bytes > 0:
        warnings.warn(
            f"detection masks ({detection_bytes // 1024 // 1024} MB) exceed "
            f"memory_limit ({memory_limit} MB); chunking cannot reduce peak "
            "memory below this floor.",
            UserWarning,
            stacklevel=2,
        )
    if detection_bytes + masks_true.shape[0] * per_true_row <= limit_bytes:
        return _mask_iou_batch_split(masks_true, masks_detection, overlap_metric)

    ious = []
    step = max((limit_bytes - detection_bytes) // per_true_row, 1)
    for chunk_start in range(0, masks_true.shape[0], step):
        ious.append(
            _mask_iou_batch_split(
                masks_true[chunk_start : chunk_start + step],
                masks_detection,
                overlap_metric,
            )
        )

    return cast(npt.NDArray[np.floating], np.vstack(ious))


def mask_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> npt.NDArray[np.bool_]:
    """
    Perform Non-Maximum Suppression (NMS) on segmentation predictions.

    IoU is computed exactly on the full-resolution masks for both dense and
    :class:`~supervision.detection.compact_mask.CompactMask` inputs.  The
    ``mask_dimension`` parameter is kept for backward compatibility but is no
    longer used — dense masks are **not** resized before IoU computation.

    Args:
        predictions: A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks: A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        mask_dimension: Deprecated, no longer used. Kept for backward
            compatibility.

    Returns:
        A boolean array indicating which predictions to keep after
            non-maximum suppression.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> masks = np.zeros((2, 4, 4), dtype=bool)
        >>> masks[:, :2, :2] = True
        >>> sv.mask_non_max_suppression(predictions, masks, iou_threshold=0.5)
        array([ True, False])

        ```
    """
    _validate_iou_threshold(iou_threshold)
    rows, columns = predictions.shape

    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    sort_index = predictions[:, 4].argsort()[::-1]
    predictions = predictions[sort_index]
    masks = masks[sort_index]

    ious = mask_iou_batch(masks, masks, overlap_metric)
    categories = predictions[:, 5]

    keep = np.ones(rows, dtype=bool)
    for row_idx in range(rows):
        if keep[row_idx]:
            condition = (ious[row_idx] > iou_threshold) & (
                categories[row_idx] == categories
            )
            keep[row_idx + 1 :] = np.where(
                condition[row_idx + 1 :], False, keep[row_idx + 1 :]
            )

    return cast(npt.NDArray[np.bool_], keep[sort_index.argsort()])


def mask_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> npt.NDArray[np.floating]:
    """
    Perform Soft Non-Maximum Suppression (Soft-NMS) on segmentation predictions.

    Unlike `mask_non_max_suppression`, which discards overlapping masks outright,
    Soft-NMS keeps every detection and instead rescales its confidence by
    `score *= exp(-iou**2 / sigma)` for each higher-scoring, same-category
    overlap — the caller decides whether and where to threshold the result.
    A smaller `sigma` produces a stronger decay.

    The 3rd positional parameter here is `sigma`, not `iou_threshold` as in
    `mask_non_max_suppression` — Soft-NMS has no threshold to suppress at, only
    a decay strength, so the two signatures intentionally diverge at that
    position.

    IoU is computed exactly on the full-resolution masks for both dense and
    :class:`~supervision.detection.compact_mask.CompactMask` inputs. The
    `mask_dimension` parameter is kept for signature parity with
    `mask_non_max_suppression` but is not used — dense masks are **not** resized
    before IoU computation.

    Args:
        predictions: A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks: A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        sigma: Controls the strength of the confidence decay; must be greater
            than `0`. No value of `sigma` reproduces hard
            `mask_non_max_suppression` output — Soft-NMS never drops masks, only
            rescales confidence.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        mask_dimension: Deprecated, unused. Kept for signature parity with
            `mask_non_max_suppression`.

    Returns:
        An array containing the updated (decayed) confidence scores, in the
            same order as the input `predictions`.

    Raises:
        ValueError: If `sigma` is not greater than `0`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> masks = np.zeros((2, 4, 4), dtype=bool)
        >>> masks[:, :2, :2] = True
        >>> sv.mask_soft_non_max_suppression(predictions, masks, sigma=0.5)
        array([0.9       , 0.10826823])

        ```
    """
    _validate_sigma(sigma)
    rows, columns = predictions.shape

    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    sort_index = predictions[:, 4].argsort()[::-1]
    predictions = predictions[sort_index]
    masks = masks[sort_index]

    ious = mask_iou_batch(masks, masks, overlap_metric)
    categories = predictions[:, 5]

    decayed = _soft_nms_decay_from_iou_matrix(
        ious, categories, predictions[:, 4], sigma
    )
    return decayed[sort_index.argsort()]


def _prepare_predictions_for_nms(
    predictions: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Add an agnostic class column when missing, sort by descending score.

    Returns the score-descending sort index, the reordered predictions, and the
    category vector for the loop callers to consume.
    """
    rows, columns = predictions.shape
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]
    categories = predictions[:, 5]
    return sort_index, predictions, categories


def _nms_loop_from_iou_matrix(
    ious: npt.NDArray[np.floating],
    categories: npt.NDArray[np.floating],
    iou_threshold: float,
) -> npt.NDArray[np.bool_]:
    """Greedy NMS suppression loop given a precomputed pairwise IoU matrix.

    Assumes `ious` is square with row/column order matching `categories`.
    Detections sharing a category whose IoU exceeds `iou_threshold` are dropped
    in favour of the higher-confidence entry.
    """
    rows = len(ious)
    ious = ious - np.eye(rows)
    keep: npt.NDArray[np.bool_] = np.ones(rows, dtype=bool)
    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition
    return keep


def _validate_sigma(sigma: float) -> None:
    """Raise `ValueError` when a Soft-NMS `sigma` is not strictly positive."""
    if not sigma > 0:
        raise ValueError(f"Value of `sigma` must be greater than 0, {sigma} given.")


def _soft_nms_decay_from_iou_matrix(
    ious: npt.NDArray[np.floating],
    categories: npt.NDArray[np.floating],
    scores: npt.NDArray[np.floating],
    sigma: float,
) -> npt.NDArray[np.floating]:
    """Vectorized Gaussian Soft-NMS confidence decay given a precomputed IoU matrix.

    Assumes `ious`, `categories`, and `scores` are all sorted by descending score
    (as produced by `_prepare_predictions_for_nms`), and that `ious` is square with
    row/column order matching `categories`. Each detection's score is decayed once
    per higher-scoring, same-category detection that precedes it — equivalent to
    the reference single-pass (no re-sort) Soft-NMS loop, computed as a single
    vectorized closed form: `decayed[j] = scores[j] * exp(-sum_i(ious[i, j]**2) /
    sigma)` summed over same-category `i < j`.
    """
    rows = len(ious)
    same_category = categories[:, None] == categories[None, :]
    precedes = np.triu(np.ones((rows, rows), dtype=bool), k=1)
    weighted_iou_sq = np.where(precedes & same_category, ious**2, 0.0)
    decay_exponent = weighted_iou_sq.sum(axis=0)
    return cast(npt.NDArray[np.floating], scores * np.exp(-decay_exponent / sigma))


def box_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Args:
        predictions: An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        A boolean array indicating which predictions to keep after
            non-maximum suppression.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> sv.box_non_max_suppression(predictions, iou_threshold=0.5)
        array([ True, False])

        ```
    """
    _validate_iou_threshold(iou_threshold)
    sort_index, predictions, categories = _prepare_predictions_for_nms(predictions)
    ious = box_iou_batch(predictions[:, :4], predictions[:, :4], overlap_metric)
    keep = _nms_loop_from_iou_matrix(ious, categories, iou_threshold)
    result: npt.NDArray[np.bool_] = keep[sort_index.argsort()]
    return result


def box_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """
    Perform Soft Non-Maximum Suppression (Soft-NMS) on object detection predictions.

    Unlike `box_non_max_suppression`, which discards overlapping boxes outright,
    Soft-NMS keeps every detection and instead rescales its confidence by
    `score *= exp(-iou**2 / sigma)` for each higher-scoring, same-category
    overlap — the caller decides whether and where to threshold the result.
    A smaller `sigma` produces a stronger decay.

    Args:
        predictions: An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        sigma: Controls the strength of the confidence decay; must be greater
            than `0`. No value of `sigma` reproduces hard
            `box_non_max_suppression` output — Soft-NMS never drops boxes, only
            rescales confidence.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        An array containing the updated (decayed) confidence scores, in the
            same order as the input `predictions`.

    Raises:
        ValueError: If `sigma` is not greater than `0`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> sv.box_soft_non_max_suppression(predictions, sigma=0.5)
        array([0.9       , 0.10826823])

        ```
    """
    _validate_sigma(sigma)
    sort_index, predictions, categories = _prepare_predictions_for_nms(predictions)
    ious = box_iou_batch(predictions[:, :4], predictions[:, :4], overlap_metric)
    decayed = _soft_nms_decay_from_iou_matrix(
        ious, categories, predictions[:, 4], sigma
    )
    result_scores: npt.NDArray[np.floating] = decayed[sort_index.argsort()]
    return result_scores


def _group_overlapping_masks(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[np.bool_],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Apply greedy version of non-maximum merging to avoid detecting too many

    Args:
        predictions: An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        masks: A 3D array of binary masks corresponding to
            the predictions.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged.
            Each group may have 1 or more elements.
    """
    merge_groups: list[list[int]] = []

    scores = predictions[:, 4]
    order = scores.argsort()

    while len(order) > 0:
        idx = int(order[-1])

        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break

        merge_candidate = masks[idx][None, ...]
        candidate_groups = [idx]
        while len(order) > 0:
            ious = mask_iou_batch(masks[order], merge_candidate, overlap_metric)
            above_threshold: npt.NDArray[np.bool_] = ious.flatten() >= iou_threshold
            if not above_threshold.any():
                break
            above_idx = order[above_threshold]
            merge_candidate = np.logical_or.reduce(
                np.concatenate([masks[above_idx], merge_candidate]),
                axis=0,
                keepdims=True,
            )
            candidate_groups.extend(np.flip(above_idx).tolist())
            order = order[~above_threshold]

        merge_groups.append(candidate_groups)
    return merge_groups


def mask_non_max_merge(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    *args: Any,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> list[list[int]]:
    """
    Perform Non-Maximum Merging (NMM) on segmentation predictions.

    Args:
        predictions: A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks: A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum merging.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        mask_dimension: Deprecated in `0.30.0`, removed in `0.33.0`. No longer
            used; the parameter is silently ignored. Passing `mask_dimension`
            positionally emits a deprecation warning.

    Returns:
        A list of groups of prediction indices. Each inner list contains
            the indices of predictions whose masks overlap above `iou_threshold`
            according to the chosen `overlap_metric`, and should be merged or
            kept together as a single detection by non-maximum merging.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.
        TypeError: If more than five positional arguments are passed.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> masks = np.zeros((2, 4, 4), dtype=bool)
        >>> masks[:, :2, :2] = True
        >>> sv.mask_non_max_merge(predictions, masks, iou_threshold=0.5)
        [[0, 1]]

        ```
    """

    _validate_iou_threshold(iou_threshold)
    if len(args) > 2:
        raise TypeError(
            "mask_non_max_merge accepts at most five positional arguments. "
            "Pass overlap_metric and mask_dimension by keyword."
        )
    if args:
        warn_deprecated(
            "Passing `overlap_metric` or `mask_dimension` positionally to "
            "`mask_non_max_merge` is deprecated in `0.30.0` and will be removed "
            "in `0.33.0`. Pass them by keyword instead."
        )
        first = args[0]
        if isinstance(first, OverlapMetric):
            overlap_metric = first
        else:
            mask_dimension = cast(int, first)
        if len(args) == 2:
            second = args[1]
            if isinstance(first, OverlapMetric):
                mask_dimension = cast(int, second)
            else:
                overlap_metric = cast(OverlapMetric, second)

    del mask_dimension

    def group_within(global_indices: npt.NDArray[np.int_]) -> list[list[int]]:
        if isinstance(masks, CompactMask):
            return _group_overlapping_masks_pairwise(
                predictions[global_indices],
                masks[global_indices],
                iou_threshold,
                overlap_metric,
            )
        return _group_overlapping_masks(
            predictions[global_indices],
            masks[global_indices],
            iou_threshold,
            overlap_metric,
        )

    return _non_max_merge_per_category(predictions, group_within)


def _update_mask_candidate(
    masks: npt.NDArray[Any] | CompactMask,
    candidate: npt.NDArray[Any] | CompactMask,
    above_idx: npt.NDArray[np.int_],
) -> npt.NDArray[Any] | CompactMask:
    if isinstance(masks, CompactMask):
        compact_candidate = cast(CompactMask, candidate)
        union_mask = np.logical_or.reduce(
            np.concatenate([masks[above_idx].to_dense(), compact_candidate.to_dense()]),
            axis=0,
            keepdims=True,
        )
        return CompactMask.from_dense(
            masks=union_mask,
            xyxy=mask_to_xyxy(union_mask),
            image_shape=masks.image_shape,
        )
    dense_candidate = cast(npt.NDArray[Any], candidate)
    dense_union: npt.NDArray[Any] = np.logical_or.reduce(
        np.concatenate([masks[above_idx], dense_candidate]),
        axis=0,
        keepdims=True,
    )
    return dense_union


def _greedy_nmm_via_mask_candidate(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Group masks by exact overlap while updating the merged candidate union."""
    merge_groups: list[list[int]] = []
    scores = predictions[:, 4]
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        candidate = masks[[idx]]
        merge_group = [idx]
        while len(order) > 0:
            ious = mask_iou_batch(masks[order], candidate, overlap_metric).flatten()
            above_threshold = ious >= iou_threshold
            if not above_threshold.any():
                break
            above_idx = order[above_threshold]
            candidate = _update_mask_candidate(masks, candidate, above_idx)
            merge_group.extend(np.flip(above_idx).tolist())
            order = order[~above_threshold]
        merge_groups.append(merge_group)
    return merge_groups


def _group_overlapping_masks_pairwise(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    return _greedy_nmm_via_mask_candidate(
        predictions, masks, iou_threshold, overlap_metric
    )


def _non_max_merge_per_category(
    predictions: npt.NDArray[np.floating],
    group_within: Callable[[npt.NDArray[np.int_]], list[list[int]]],
) -> list[list[int]]:
    """Dispatch NMM grouping per class, then translate local indices back to
    the global row positions of ``predictions``.

    ``group_within(global_indices)`` must return merge groups expressed in
    terms of *positions inside `global_indices`*, not absolute row positions.
    When ``predictions`` has no class column, a single pass over all rows is
    performed instead of per-category iteration.
    """
    if predictions.shape[1] == 5:
        global_indices = np.arange(len(predictions), dtype=int)
        return [
            global_indices[group].tolist() for group in group_within(global_indices)
        ]

    category_ids = predictions[:, 5]
    merge_groups: list[list[int]] = []
    for category_id in np.unique(category_ids):
        curr_indices = np.where(category_ids == category_id)[0]
        for local_group in group_within(curr_indices):
            merge_groups.append(curr_indices[local_group].tolist())

    for merge_group in merge_groups:
        if len(merge_group) == 0:
            raise ValueError(
                f"Empty group detected when non-max-merging detections: {merge_groups}"
            )
    return merge_groups


def _group_overlapping_boxes(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.

    Args:
        predictions: An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged.
            Each group may have 1 or more elements.
    """
    merge_groups: list[list[int]] = []
    scores = predictions[:, 4]
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        ious = box_iou_batch(
            predictions[order][:, :4],
            predictions[idx : idx + 1, :4],
            overlap_metric,
        ).flatten()
        above_threshold = ious >= iou_threshold
        merge_group = [idx, *np.flip(order[above_threshold]).tolist()]
        merge_groups.append(merge_group)
        order = order[~above_threshold]
    return merge_groups


def box_non_max_merge(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Apply greedy version of non-maximum merging per category to avoid detecting
    too many overlapping bounding boxes for a given object.

    Args:
        predictions: An array of shape `(n, 5)` or `(n, 6)`
            containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`,
            the confidence scores and class_ids. Omit class_id column to allow
            detections of different classes to be merged.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        list[list[int]]: Groups of prediction indices be merged.
            Each group may have 1 or more elements.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> sv.box_non_max_merge(predictions, iou_threshold=0.5)
        [[0, 1]]

        ```
    """
    _validate_iou_threshold(iou_threshold)

    def group_within(global_indices: npt.NDArray[np.int_]) -> list[list[int]]:
        return _group_overlapping_boxes(
            predictions[global_indices], iou_threshold, overlap_metric
        )

    return _non_max_merge_per_category(predictions, group_within)


def _fuse_box_group(group: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Fuse one group of overlapping boxes into a single confidence-weighted box.

    The fused coordinates are the average of the group's coordinates weighted by
    each box's confidence, so higher-scoring boxes pull the result toward
    themselves; the fused score is the group's mean confidence, and the class id
    (when present) is taken from the highest-scoring member.
    """
    boxes = group[:, :4]
    scores = group[:, 4]

    # Fall back to a uniform average when every score is zero, so the weighted
    # mean stays well defined instead of dividing by a zero weight sum.
    score_sum = scores.sum()
    if score_sum > 0:
        weights = scores / score_sum
    else:
        weights = np.full(len(scores), 1.0 / len(scores))

    fused_box = (boxes * weights[:, np.newaxis]).sum(axis=0)
    fused_score = float(scores.mean())
    fused: npt.NDArray[np.floating] = np.concatenate([fused_box, [fused_score]])

    if group.shape[1] > 5:
        best_class = group[int(scores.argmax()), 5]
        fused = np.concatenate([fused, [best_class]])
    return fused


def box_weighted_box_fusion(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """
    Fuse overlapping object detection boxes with Weighted Box Fusion (WBF).

    Unlike `box_non_max_suppression`, which discards overlapping boxes, and
    `box_non_max_merge`, which only returns groups of overlapping indices, WBF
    replaces each group of overlapping boxes with a single box whose coordinates
    are the confidence-weighted average of the group and whose score is the
    group's mean confidence. This keeps information from every box in a cluster
    instead of dropping all but one, which typically produces better localized
    boxes when combining predictions from several models or augmentations. Based
    on Solovyev et al., "Weighted Boxes Fusion: Ensembling boxes from different
    object detection models" (2019).

    Args:
        predictions: An array of shape `(n, 5)` or `(n, 6)` containing the
            bounding boxes coordinates in format `[x1, y1, x2, y2]`, the
            confidence scores, and optionally the class ids. Omit the class_id
            column to allow boxes of different classes to be fused together.
        iou_threshold: The intersection-over-union threshold used to decide
            whether two boxes belong to the same group. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        An array of fused predictions with the same number of columns as the
            input (`(m, 5)` or `(m, 6)`, with `m <= n`), ordered by descending
            fused confidence.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0.0, 0.0, 10.0, 10.0, 0.9, 0],
        ...     [1.0, 1.0, 11.0, 11.0, 0.8, 0],
        ... ])
        >>> fused = sv.box_weighted_box_fusion(predictions, iou_threshold=0.5)
        >>> fused[:, 4]
        array([0.85])

        ```
    """
    _validate_iou_threshold(iou_threshold)
    if len(predictions) == 0:
        empty_result: npt.NDArray[np.floating] = predictions.copy()
        return empty_result

    merge_groups = box_non_max_merge(predictions, iou_threshold, overlap_metric)
    fused = np.stack([_fuse_box_group(predictions[group]) for group in merge_groups])

    descending_score = np.flip(fused[:, 4].argsort())
    result: npt.NDArray[np.floating] = fused[descending_score]
    return result


def oriented_box_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """
    Perform Non-Maximum Suppression on oriented bounding box predictions.

    Overlap is computed via :func:`oriented_box_iou_batch` on the four
    corners of each box, so detections whose axis-aligned bounding boxes
    overlap heavily but whose oriented bodies do not are kept — unlike
    :func:`box_non_max_suppression`, which would suppress them.

    Args:
        predictions: An array of object detection predictions in the
            format ``(x_min, y_min, x_max, y_max, score)`` or
            ``(x_min, y_min, x_max, y_max, score, class)``. Shape ``(N, 5)``
            or ``(N, 6)``. Only the score (column 4) and optional class
            (column 5) are read; the axis-aligned coordinates are not used.
        oriented_boxes: Array of shape ``(N, 4, 2)`` containing the four
            ``(x, y)`` corners of each oriented box, aligned with
            ``predictions`` row-by-row.
        iou_threshold: The intersection-over-union threshold to use for
            non-maximum suppression.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of oriented boxes (e.g., IoU, IoS).

    Returns:
        A boolean array of shape ``(N,)`` indicating which predictions
            to keep after non-maximum suppression.

    Raises:
        ValueError: If ``iou_threshold`` is not within the closed range
            from 0 to 1.
        ValueError: If ``predictions`` and ``oriented_boxes`` have
            mismatched lengths or invalid shapes.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> oriented_boxes = np.array([
        ...     [[10, 10], [50, 10], [50, 30], [10, 30]],
        ...     [[11, 11], [51, 11], [51, 31], [11, 31]],
        ... ], dtype=np.float32)
        >>> predictions = np.array([
        ...     [10, 10, 50, 30, 0.9, 0],
        ...     [11, 11, 51, 31, 0.8, 0],
        ... ], dtype=np.float32)
        >>> keep = sv.oriented_box_non_max_suppression(
        ...     predictions=predictions,
        ...     oriented_boxes=oriented_boxes,
        ...     iou_threshold=0.5,
        ... )
        >>> keep
        array([ True, False])

        ```
    """
    _validate_iou_threshold(iou_threshold)
    for name, arr in (("predictions", predictions), ("oriented_boxes", oriented_boxes)):
        if name == "predictions":
            if arr.ndim != 2 or arr.shape[1] not in (5, 6):
                raise ValueError(
                    f"`{name}` has shape {arr.shape}; expected (N, 5) or (N, 6)."
                )
            continue
        if arr.ndim == 3 and arr.shape[1:] != (4, 2):
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 4, 2) "
                f"— each box must have exactly 4 corners with (x, y) coordinates."
            )
        elif arr.ndim == 2 and arr.shape[1] != 8:
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 8) for flat "
                f"YOLO format or (N, 4, 2) for corner format."
            )
        elif arr.ndim not in (2, 3):
            raise ValueError(
                f"`{name}` must be 2-D (N, 8) or 3-D (N, 4, 2), got shape {arr.shape}."
            )
    if len(predictions) != len(oriented_boxes):
        raise ValueError(
            f"`predictions` and `oriented_boxes` must have the same length, "
            f"got {len(predictions)} and {len(oriented_boxes)}."
        )
    sort_index, _, categories = _prepare_predictions_for_nms(predictions)
    oriented_boxes = oriented_boxes[sort_index]
    # same object intentional — triggers upper-triangle optimization
    ious = oriented_box_iou_batch(oriented_boxes, oriented_boxes, overlap_metric)
    keep = _nms_loop_from_iou_matrix(ious, categories, iou_threshold)
    result: npt.NDArray[np.bool_] = keep[sort_index.argsort()]
    return result


def _group_overlapping_oriented_boxes(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Greedy non-maximum merging on oriented boxes. Mirrors
    :func:`_group_overlapping_boxes` but uses :func:`oriented_box_iou_batch`.
    """
    merge_groups: list[list[int]] = []
    scores = predictions[:, 4]
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        ious = oriented_box_iou_batch(
            oriented_boxes[order],
            oriented_boxes[idx][None, ...],
            overlap_metric,
        ).flatten()
        above_threshold = ious >= iou_threshold
        merge_group = [idx, *np.flip(order[above_threshold]).tolist()]
        merge_groups.append(merge_group)
        order = order[~above_threshold]
    return merge_groups


def oriented_box_non_max_merge(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Perform Non-Maximum Merging on oriented bounding box predictions,
    grouped per category.

    Mirrors :func:`box_non_max_merge` but uses oriented-box IoU, so groups
    of rotated detections sharing the same body — rather than the same
    axis-aligned bounding box — are merged.

    Args:
        predictions: An array of shape ``(n, 5)`` or ``(n, 6)`` containing
            the axis-aligned coordinates ``[x1, y1, x2, y2]``, confidence
            scores, and optionally class ids. Only the score and optional
            class are used by the grouping logic; overlap is computed on
            ``oriented_boxes``.
        oriented_boxes: Array of shape ``(N, 4, 2)`` containing the four
            ``(x, y)`` corners of each oriented box.
        iou_threshold: The intersection-over-union threshold to use for
            non-maximum merging.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of oriented boxes (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged. Each group may have 1
            or more elements.

    Raises:
        ValueError: If ``iou_threshold`` is not within the closed range
            from 0 to 1.
        ValueError: If ``predictions`` and ``oriented_boxes`` have
            mismatched lengths or invalid shapes.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> oriented_boxes = np.array([
        ...     [[10, 10], [50, 10], [50, 30], [10, 30]],
        ...     [[11, 11], [51, 11], [51, 31], [11, 31]],
        ... ], dtype=np.float32)
        >>> predictions = np.array([
        ...     [10, 10, 50, 30, 0.9, 0],
        ...     [11, 11, 51, 31, 0.8, 0],
        ... ], dtype=np.float32)
        >>> groups = sv.oriented_box_non_max_merge(
        ...     predictions=predictions,
        ...     oriented_boxes=oriented_boxes,
        ...     iou_threshold=0.5,
        ... )
        >>> len(groups)
        1

        ```
    """
    for name, arr in (("predictions", predictions), ("oriented_boxes", oriented_boxes)):
        if name == "predictions":
            if arr.ndim != 2 or arr.shape[1] not in (5, 6):
                raise ValueError(
                    f"`{name}` has shape {arr.shape}; expected (N, 5) or (N, 6)."
                )
            continue
        if arr.ndim == 3 and arr.shape[1:] != (4, 2):
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 4, 2) "
                f"— each box must have exactly 4 corners with (x, y) coordinates."
            )
        elif arr.ndim == 2 and arr.shape[1] != 8:
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 8) for flat "
                f"YOLO format or (N, 4, 2) for corner format."
            )
        elif arr.ndim not in (2, 3):
            raise ValueError(
                f"`{name}` must be 2-D (N, 8) or 3-D (N, 4, 2), got shape {arr.shape}."
            )
    if len(predictions) != len(oriented_boxes):
        raise ValueError(
            f"`predictions` and `oriented_boxes` must have the same length, "
            f"got {len(predictions)} and {len(oriented_boxes)}."
        )
    _validate_iou_threshold(iou_threshold)

    def group_within(global_indices: npt.NDArray[np.int_]) -> list[list[int]]:
        return _group_overlapping_oriented_boxes(
            predictions[global_indices],
            oriented_boxes[global_indices],
            iou_threshold,
            overlap_metric,
        )

    return _non_max_merge_per_category(predictions, group_within)
