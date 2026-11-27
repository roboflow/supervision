"""Overlap measures between boxes, oriented boxes, masks and keypoints."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision._cv2._geometry import _intersect_convex_convex
from supervision.detection.compact_mask import CompactMask

#: ``np.bitwise_count`` (popcount) was added in NumPy 2.0; the packed mask
#: overlap path needs it.
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

#: Largest pixel count whose float32 sum is still exact (``2**24``); masks with
#: more pixels count in float64 instead.
_FLOAT32_EXACT_PIXELS = 2**24


#: Upper bound, in bytes, of the temporary the packed intersection kernel builds
#: per block of true-mask rows (the bitwise AND plus its popcount).
_PACKED_BLOCK_BYTES = 4 * 1024 * 1024


def _count_dtype(pixels: int) -> type[np.floating]:
    """Return the float dtype that counts `pixels` mask pixels exactly.

    Args:
        pixels: Number of pixels in one mask.

    Returns:
        `np.float32` up to `2**24` pixels, `np.float64` beyond.
    """
    return np.float32 if pixels <= _FLOAT32_EXACT_PIXELS else np.float64


def _use_packed(*masks: npt.NDArray[Any]) -> bool:
    """Tell whether the bit-packed overlap path applies to the given masks.

    Args:
        *masks: Mask arrays that would be compared.

    Returns:
        `True` when `np.bitwise_count` exists and every array is boolean.
    """
    return _HAS_BITWISE_COUNT and all(mask.dtype == bool for mask in masks)


class OverlapFilter(Enum):
    """Enum specifying the strategy for filtering overlapping detections.

    Attributes:
        KEEP_ALL: Do not filter detections based on overlap; keep every detection.
            `NONE` and the string `"none"` remain accepted as aliases of this member.
        NON_MAX_SUPPRESSION: Filter detections using non-max suppression. This means,
            detections that overlap by more than a set threshold will be discarded,
            except for the one with the highest confidence.
        NON_MAX_MERGE: Merge detections with non-max merging. This means,
            detections that overlap by more than a set threshold will be merged
            into a single detection.
    """

    KEEP_ALL = "keep_all"
    NON_MAX_SUPPRESSION = "non_max_suppression"
    NON_MAX_MERGE = "non_max_merge"
    # Same value as `KEEP_ALL`, so Enum makes it an alias of that member.
    NONE = "keep_all"

    @classmethod
    def _missing_(cls, value: object) -> OverlapFilter | None:
        """Resolve the legacy string `"none"` to `KEEP_ALL`."""
        if value == "none":
            return cls.KEEP_ALL
        return None

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
    """Enum specifying the metric for measuring overlap between detections.

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


class OverlapOn(Enum):
    """Enum specifying the geometry used to measure overlap between detections.

    Attributes:
        BOX: The detection's own box: its oriented box when
            `data["xyxyxyxy"]` is set, otherwise its axis-aligned `xyxy`.
            Masks are carried through the operation but not used for overlap.
        MASK: The detection's mask. Requires `Detections.mask` on nonempty inputs.

    Example:
        ```pycon
        >>> OverlapOn.from_value("MASK")
        <OverlapOn.MASK: 'mask'>

        ```
    """

    BOX = "box"
    MASK = "mask"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string values of all members."""
        return list(map(lambda member: member.value, cls))

    @classmethod
    def from_value(cls, value: OverlapOn | str) -> OverlapOn:
        """Return the member for an enum member or a case-insensitive string.

        Args:
            value: A member of this enum or its string value.

        Returns:
            The matching member.

        Raises:
            ValueError: If `value` is not a member, a known string or a string.
        """
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


def _validate_iou_threshold(iou_threshold: float) -> None:
    """Raise `ValueError` when an IoU threshold falls outside `[0, 1]`."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError(
            "Value of `iou_threshold` must be in the closed range from 0 to 1, "
            f"{iou_threshold} given."
        )


def _is_mergeable_overlap(
    overlaps: npt.NDArray[np.floating], iou_threshold: float
) -> npt.NDArray[np.bool_]:
    """Flag pairs whose overlap is high enough to be grouped by non-max merging.

    A pair qualifies when its overlap reaches `iou_threshold` and is strictly
    positive. The positivity guard only matters at `iou_threshold=0`, where a
    plain `>=` would group disjoint pairs, whereas non-max suppression, which
    compares with a strict `>`, never suppresses them. For any positive
    threshold the result equals `overlaps >= iou_threshold`.

    Args:
        overlaps: Overlap values (IoU or IoS) of candidates against one
            reference, as a `np.ndarray` of any shape.
        iou_threshold: Minimum overlap, in the closed range from 0 to 1.

    Returns:
        Boolean `np.ndarray` with the shape of `overlaps`.
    """
    mergeable: npt.NDArray[np.bool_] = (overlaps >= iou_threshold) & (overlaps > 0)
    return mergeable


def _coordinate_difference(upper: np.number, lower: np.number) -> float:
    """Subtract box coordinates without overflowing integer NumPy scalars."""
    return float(np.asarray(upper).item() - np.asarray(lower).item())


def box_iou(
    box_true: list[float] | npt.NDArray[np.number],
    box_detection: list[float] | npt.NDArray[np.number],
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> float:
    """Compute overlap metric between two bounding boxes.

    Supports standard IOU (intersection-over-union) and IOS
    (intersection-over-smaller-area) metrics. Returns the overlap value in range
    `[0, 1]`. Integer coordinates are subtracted before conversion to floating
    point so large origins remain precise while area products avoid integer overflow.

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
        TypeError: If either box contains complex coordinates.
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
    box_true_array = np.asarray(box_true)
    box_detection_array = np.asarray(box_detection)
    if np.iscomplexobj(box_true_array) or np.iscomplexobj(box_detection_array):
        raise TypeError("box coordinates must be real-valued")

    x_min_true, y_min_true, x_max_true, y_max_true = box_true_array
    x_min_det, y_min_det, x_max_det, y_max_det = box_detection_array

    x_min_inter = max(x_min_true, x_min_det)
    y_min_inter = max(y_min_true, y_min_det)
    x_max_inter = min(x_max_true, x_max_det)
    y_max_inter = min(y_max_true, y_max_det)

    inter_w = max(0.0, _coordinate_difference(x_max_inter, x_min_inter))
    inter_h = max(0.0, _coordinate_difference(y_max_inter, y_min_inter))

    area_inter = inter_w * inter_h

    area_true = _coordinate_difference(x_max_true, x_min_true) * _coordinate_difference(
        y_max_true, y_min_true
    )
    area_det = _coordinate_difference(x_max_det, x_min_det) * _coordinate_difference(
        y_max_det, y_min_det
    )

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
    """Compute pairwise overlap scores between batches of bounding boxes.

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
    return _box_overlap_float64(boxes_true, boxes_detection, overlap_metric).astype(
        np.float32
    )


_FLOAT64_EXACT_INT_LIMIT = 1 << 53


def _has_coordinates_beyond_float64_exact_int(boxes: npt.NDArray[np.number]) -> bool:
    """Return True if a 64-bit integer array exceeds float64's exact integer range."""
    if not np.issubdtype(boxes.dtype, np.integer) or boxes.dtype.itemsize < 8:
        return False
    if np.issubdtype(boxes.dtype, np.unsignedinteger):
        return int(boxes.max()) > _FLOAT64_EXACT_INT_LIMIT
    return (
        int(boxes.min()) < -_FLOAT64_EXACT_INT_LIMIT
        or int(boxes.max()) > _FLOAT64_EXACT_INT_LIMIT
    )


def _large_integer_box_areas(
    boxes_true: npt.NDArray[np.number],
    boxes_detection: npt.NDArray[np.number],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute exact intersection and box areas for 64-bit integer boxes."""
    true_obj = boxes_true.astype(object)
    det_obj = boxes_detection.astype(object)

    # Subtract as Python integers before casting to float64 so narrow boxes at
    # origins beyond 2**53 keep unit differences and wide spans avoid int64 wrap.
    area_true = np.asarray(
        true_obj[:, 2] - true_obj[:, 0], dtype=np.float64
    ) * np.asarray(true_obj[:, 3] - true_obj[:, 1], dtype=np.float64)
    area_det = np.asarray(det_obj[:, 2] - det_obj[:, 0], dtype=np.float64) * np.asarray(
        det_obj[:, 3] - det_obj[:, 1], dtype=np.float64
    )

    if np.issubdtype(
        np.result_type(boxes_true.dtype, boxes_detection.dtype), np.integer
    ):
        x_min_true, y_min_true, x_max_true, y_max_true = boxes_true.T
        x_min_det, y_min_det, x_max_det, y_max_det = boxes_detection.T
    else:
        # Mixed signed/unsigned 64-bit integers promote to float64 under NumPy
        # result_type, so compare Python ints directly to avoid rounding.
        x_min_true, y_min_true, x_max_true, y_max_true = true_obj.T
        x_min_det, y_min_det, x_max_det, y_max_det = det_obj.T

    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]
    area_inter: npt.NDArray[np.float64] = np.zeros(
        (count_true, count_det), dtype=np.float64
    )
    # Bound object and index temporaries for dense overlap; the output itself
    # remains quadratic because callers require every pairwise score.
    for start in range(0, count_true, 128):
        stop = min(start + 128, count_true)
        x_min_inter = np.maximum(x_min_true[start:stop, None], x_min_det[None, :])
        x_max_inter = np.minimum(x_max_true[start:stop, None], x_max_det[None, :])
        y_min_inter = np.maximum(y_min_true[start:stop, None], y_min_det[None, :])
        y_max_inter = np.minimum(y_max_true[start:stop, None], y_max_det[None, :])
        overlap = (x_max_inter > x_min_inter) & (y_max_inter > y_min_inter)
        rows, cols = np.nonzero(overlap)
        inter_w = np.asarray(
            x_max_inter[rows, cols].astype(object)
            - x_min_inter[rows, cols].astype(object),
            dtype=np.float64,
        )
        inter_h = np.asarray(
            y_max_inter[rows, cols].astype(object)
            - y_min_inter[rows, cols].astype(object),
            dtype=np.float64,
        )
        area_inter[start + rows, cols] = inter_w * inter_h

    return area_inter, area_true, area_det


def _box_overlap_float64(
    boxes_true: npt.NDArray[np.number],
    boxes_detection: npt.NDArray[np.number],
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> npt.NDArray[np.float64]:
    """Compute pairwise IOU or IOS without rounding overlaps to float32."""
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]

    if count_true == 0 or count_det == 0:
        return cast(
            npt.NDArray[np.float64], np.empty((count_true, count_det), dtype=np.float64)
        )

    if (
        np.issubdtype(boxes_true.dtype, np.integer)
        and np.issubdtype(boxes_detection.dtype, np.integer)
        and (
            _has_coordinates_beyond_float64_exact_int(boxes_true)
            or _has_coordinates_beyond_float64_exact_int(boxes_detection)
        )
    ):
        area_inter, area_true, area_det = _large_integer_box_areas(
            boxes_true, boxes_detection
        )
    else:
        # Upcast the corners to float64 right after unpacking so every subtraction
        # and multiplication below runs in float64. This prevents integer-dtype
        # overflow: an int32 box area such as 50000 * 50000 = 2.5e9 wraps to a
        # negative value in int32 before any later cast could run, yielding wrong
        # (often zero) IoU. Upcasting here also gives full float64 precision to
        # float64/int64 callers. It does NOT recover precision already lost when a
        # caller stores coordinates as float32 upstream, because that float32
        # rounding happens before this function is ever called.
        x_min_true, y_min_true, x_max_true, y_max_true = boxes_true.T.astype(np.float64)
        x_min_det, y_min_det, x_max_det, y_max_det = boxes_detection.T.astype(
            np.float64
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

    out: npt.NDArray[np.float64] = np.zeros_like(area_inter, dtype=np.float64)
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


def _polygon_areas(polygons: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
    """Compute the area of each oriented-box polygon using the shoelace formula.

    Each polygon is translated to its own first corner before the shoelace
    products, keeping representable integer coordinates with large origins (e.g.
    geospatial frames) local before conversion to float64. Float inputs that
    were already quantized retain their input precision.

    Args:
        polygons: ``(N, 4, 2)`` array of polygon corners.

    Returns:
        ``(N,)`` array of polygon areas.
    """
    origin = polygons[:, 0:1, :]
    if np.issubdtype(polygons.dtype, np.integer):
        translated = np.asarray(
            polygons.astype(object) - origin.astype(object), dtype=np.float64
        )
    else:
        polygons_float = polygons.astype(np.float64, copy=False)
        translated = polygons_float - origin.astype(np.float64, copy=False)
    x = translated[:, :, 0]
    y = translated[:, :, 1]
    cross = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
    return cast(npt.NDArray[np.float64], 0.5 * np.abs(cross.sum(axis=1)))


def _aabb_envelopes(polygons: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
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
    envelopes_true: npt.NDArray[np.number],
    envelopes_detection: npt.NDArray[np.number],
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


def _oriented_box_pair_intersections(
    boxes_true: npt.NDArray[np.number],
    boxes_detection: npt.NDArray[np.number],
    is_self_comparison: bool = False,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Compute exact intersection areas of the oriented box pairs that may overlap.

    Only pairs whose axis-aligned envelopes overlap are returned; every other
    pair has no intersection. For a self comparison only pairs with
    `row <= col` are returned, since the result is symmetric.

    Args:
        boxes_true: Convex quads shaped `(N, 4, 2)`.
        boxes_detection: Convex quads shaped `(M, 4, 2)`.
        is_self_comparison: Whether both arguments are the same boxes.

    Returns:
        Row indices into `boxes_true`, column indices into `boxes_detection`,
        and the intersection area of each such pair.
    """
    envelopes_true = _aabb_envelopes(boxes_true)
    envelopes_detection = (
        envelopes_true if is_self_comparison else _aabb_envelopes(boxes_detection)
    )
    rows, cols = _overlapping_envelope_pairs(envelopes_true, envelopes_detection)
    if is_self_comparison:
        upper = rows <= cols
        rows, cols = rows[upper], cols[upper]

    local_origins_true = envelopes_true[:, :2]
    local_origins_detection = envelopes_detection[:, :2]
    if np.issubdtype(boxes_true.dtype, np.integer):
        local_polygons_true = np.asarray(
            boxes_true.astype(object) - local_origins_true[:, None, :].astype(object),
            dtype=np.float32,
        )
    else:
        local_polygons_true = (boxes_true - local_origins_true[:, None, :]).astype(
            np.float32
        )
    if np.issubdtype(boxes_detection.dtype, np.integer):
        local_polygons_detection = np.asarray(
            boxes_detection.astype(object)
            - local_origins_detection[:, None, :].astype(object),
            dtype=np.float32,
        )
    else:
        local_polygons_detection = (
            boxes_detection - local_origins_detection[:, None, :]
        ).astype(np.float32)
    # Float32 inputs already have quantized origins, so their envelope minima
    # can be reused without loss. Float64 inputs may retain unit differences at
    # large origins; keep their pair offsets in float64 until each pair's local
    # cast instead of rounding absolute minima to float32 up front.
    origins_true_float32 = (
        local_origins_true if boxes_true.dtype == np.float32 else None
    )
    origins_detection_float32 = (
        local_origins_detection if boxes_detection.dtype == np.float32 else None
    )
    polygon_i_buffer = np.empty((4, 2), dtype=np.float32)
    polygon_j_buffer = np.empty((4, 2), dtype=np.float32)
    intersections = np.zeros(len(rows), dtype=np.float64)
    for index, (i, j) in enumerate(zip(rows, cols)):
        # Rebase precomputed local polygons to the shared overlap origin. This
        # preserves translation-invariant float32 geometry without repeating
        # corner reductions and casts for every candidate pair.
        if origins_true_float32 is not None and origins_detection_float32 is not None:
            origin_i = origins_true_float32[i]
            origin_j = origins_detection_float32[j]
        else:
            origin_i = local_origins_true[i]
            origin_j = local_origins_detection[j]
        pair_origin = np.minimum(origin_i, origin_j)
        offset_i = (origin_i - pair_origin).astype(np.float32, copy=False)
        offset_j = (origin_j - pair_origin).astype(np.float32, copy=False)
        np.add(local_polygons_true[i], offset_i, out=polygon_i_buffer)
        np.add(local_polygons_detection[j], offset_j, out=polygon_j_buffer)
        # Not cv2.intersectConvexConvex: it returns wrong areas for nearly collinear
        # overlapping edges and zero for polygons far below one coordinate unit.
        intersections[index], _ = _intersect_convex_convex(
            polygon_i_buffer, polygon_j_buffer
        )
    return rows, cols, intersections


def oriented_box_iou_batch(
    boxes_true: npt.NDArray[np.number],
    boxes_detection: npt.NDArray[np.number],
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Compute pairwise overlap scores between two sets of oriented bounding boxes using
    the configured `overlap_metric`.

    Overlap areas are computed exactly via convex-polygon intersection, gated by
    a cheap axis-aligned envelope pre-filter — no rasterization is involved, so
    the result is exact (free of pixel-quantization error) and independent of the
    coordinate magnitudes.

    `boxes_true` and `boxes_detection` are expected to be in
    `((x1, y1), (x2, y2), (x3, y3), (x4, y4))` format.

    Note:
        Inputs must be **convex** quads with finite coordinates. Self-intersecting
        or non-convex polygons produce undefined results. NaN or Inf coordinates
        propagate silently as ``0.0`` — validate inputs before calling if needed.

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
        `[0, 1]` under the configured ``overlap_metric``.

    Raises:
        ValueError: If ``boxes_true`` or ``boxes_detection`` is 3-D with inner
            dimensions other than ``(4, 2)``.
        ValueError: If ``boxes_true`` or ``boxes_detection`` is 2-D with a
            column count other than 8.
        ValueError: If ``boxes_true`` or ``boxes_detection`` is not 2-D or 3-D.
        ValueError: If ``overlap_metric`` is not
            :attr:`~supervision.detection.utils.iou.OverlapMetric.IOU` or
            :attr:`~supervision.detection.utils.iou.OverlapMetric.IOS`.

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
    boxes_true = boxes_true.reshape(-1, 4, 2)
    boxes_detection = boxes_detection.reshape(-1, 4, 2)

    n, m = len(boxes_true), len(boxes_detection)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    areas_true = _polygon_areas(boxes_true)
    areas_detection = _polygon_areas(boxes_detection)
    rows, cols, intersections = _oriented_box_pair_intersections(
        boxes_true, boxes_detection, is_self_comparison
    )
    if normalize_by_union:
        denominators = areas_true[rows] + areas_detection[cols] - intersections
    else:
        denominators = np.minimum(areas_true[rows], areas_detection[cols])
    valid = (intersections > 0) & (denominators > 0)
    rows, cols = rows[valid], cols[valid]
    scores = intersections[valid] / denominators[valid]

    ious: npt.NDArray[np.float64] = np.zeros((n, m), dtype=np.float64)
    ious[rows, cols] = scores
    if is_self_comparison:
        ious[cols, rows] = scores

    # DO NOT remove this clip. Intersections are computed on float32-rounded
    # polygons while polygon areas use the original coordinates; the intersection
    # area can exceed the polygon area by ~1e-7 relative, producing raw IoU or
    # IoS values microscopically above 1.0 for identical boxes. The clip is
    # load-bearing, not defensive duplication.
    return cast(npt.NDArray[np.floating], np.clip(ious, 0.0, 1.0))


#: Per-keypoint OKS sigmas of the 17-point COCO person skeleton, in COCO keypoint
#: order (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), as
#: used by `pycocotools`.
_COCO_KEYPOINT_SIGMAS: npt.NDArray[np.float64] = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float64,
    )
    / 10.0
)


def _validate_keypoint_sigmas(sigmas: npt.NDArray[np.float64]) -> None:
    """Check that every per-keypoint OKS sigma is positive and finite.

    The shape is not checked, so callers that do not know the skeleton yet can
    reject bad values early.

    Args:
        sigmas: Per-keypoint OKS sigmas.

    Raises:
        ValueError: If any sigma is zero, negative, NaN or infinite.
    """
    if not np.all(np.isfinite(sigmas) & (sigmas > 0)):
        raise ValueError("`sigmas` must be positive and finite.")


def _resolve_keypoint_sigmas(
    sigmas: npt.ArrayLike | None, num_keypoints: int
) -> npt.NDArray[np.float64]:
    """Return per-keypoint OKS sigmas, defaulting to COCO only for 17 keypoints.

    Raises:
        ValueError: If `sigmas` is `None` for a skeleton that is not 17 points
            long.
        ValueError: If `sigmas` does not hold one positive value per keypoint.
    """
    if sigmas is None:
        if num_keypoints != len(_COCO_KEYPOINT_SIGMAS):
            raise ValueError(
                f"OKS needs one sigma per keypoint, and the COCO preset only "
                f"covers 17-point skeletons; got {num_keypoints} keypoints. "
                f"Pass `sigmas` with shape ({num_keypoints},) for this skeleton."
            )
        return _COCO_KEYPOINT_SIGMAS
    resolved = np.asarray(sigmas, dtype=np.float64)
    if resolved.shape != (num_keypoints,):
        raise ValueError(
            f"`sigmas` must have shape ({num_keypoints},) to match the number of "
            f"keypoints; got {resolved.shape}."
        )
    _validate_keypoint_sigmas(resolved)
    return resolved


def _keypoint_oks_batch(
    keypoints_true: npt.NDArray[np.number],
    keypoints_detection: npt.NDArray[np.number],
    area_true: npt.NDArray[np.number],
    sigmas: npt.ArrayLike | None = None,
    visible_true: npt.NDArray[np.bool_] | None = None,
    xyxy_true: npt.NDArray[np.number] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute pairwise Object Keypoint Similarity (OKS) between keypoint sets.

    OKS is the keypoint counterpart of IoU used by COCO keypoint evaluation:

    `OKS = mean over visible i of exp(-d_i^2 / (2 * area * (2 * sigma_i)^2))`

    where `d_i` is the distance between the `i`-th target and detected keypoint,
    `area` is the target's object area, and only the target's visible keypoints
    contribute. The arithmetic follows `pycocotools` `COCOeval.computeOks`, and
    the result is float64 so that values compared with OKS thresholds match it.

    Args:
        keypoints_true: Target keypoints of shape `(N, K, 2)`.
        keypoints_detection: Detected keypoints of shape `(M, K, 2)`. Detection
            visibility is not used, as in COCO.
        area_true: Object area of each target, shape `(N,)`. COCO uses the
            annotated segmentation area here.
        sigmas: Per-keypoint falloff constants of shape `(K,)`. Defaults to
            `_COCO_KEYPOINT_SIGMAS` when `K == 17` and is required otherwise.
        visible_true: Boolean mask of shape `(N, K)` marking labelled target
            keypoints. When `None`, all target keypoints are visible.
        xyxy_true: Target boxes of shape `(N, 4)` in `(x_min, y_min, x_max,
            y_max)` format, used only for targets with no visible keypoint.
            As in `pycocotools`, such a target's box of width `w` and height
            `h` is expanded to `[x_min - w, x_min + 2w] x [y_min - h, y_min + 2h]`,
            `d_i` becomes the distance from each detected keypoint to that
            expanded box (`0` inside it), and the mean runs over all `K`
            keypoints. When `None`, a target with no visible keypoint has OKS
            `0` with every detection.

    Returns:
        OKS matrix of shape `(N, M)`, where entry `[i, j]` is the similarity
            between `keypoints_true[i]` and `keypoints_detection[j]`.

    Raises:
        ValueError: If the shapes are inconsistent.
        ValueError: If `sigmas` is missing for a skeleton that is not 17 points
            long.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.utils.iou import _keypoint_oks_batch
        >>> keypoints_true = np.array([[[10, 10], [20, 10], [15, 20]]])
        >>> keypoints_detection = np.array([
        ...     [[10, 10], [20, 10], [15, 20]],
        ...     [[12, 10], [20, 13], [15, 20]],
        ... ])
        >>> _keypoint_oks_batch(
        ...     keypoints_true,
        ...     keypoints_detection,
        ...     area_true=np.array([100.0]),
        ...     sigmas=np.array([0.25, 0.25, 0.25]),
        ... )
        array([[1.        , 0.91946...]])

        ```
    """
    keypoints_true = np.asarray(keypoints_true)
    keypoints_detection = np.asarray(keypoints_detection)
    count_true, count_det = len(keypoints_true), len(keypoints_detection)
    if count_true == 0 or count_det == 0:
        return np.zeros((count_true, count_det), dtype=np.float64)

    if (
        keypoints_true.ndim != 3
        or keypoints_true.shape[2] != 2
        or keypoints_detection.shape[1:] != keypoints_true.shape[1:]
    ):
        raise ValueError(
            "`keypoints_true` and `keypoints_detection` must have shapes (N, K, 2) "
            f"and (M, K, 2); got {keypoints_true.shape} and "
            f"{keypoints_detection.shape}."
        )
    num_keypoints = keypoints_true.shape[1]
    area = np.asarray(area_true, dtype=np.float64)
    if area.shape != (count_true,):
        raise ValueError(
            f"`area_true` must have shape ({count_true},); got {area.shape}."
        )
    if visible_true is None:
        visible = np.ones((count_true, num_keypoints), dtype=bool)
    else:
        visible = np.asarray(visible_true, dtype=bool)
        if visible.shape != (count_true, num_keypoints):
            raise ValueError(
                f"`visible_true` must have shape ({count_true}, {num_keypoints}); "
                f"got {visible.shape}."
            )
    boxes = None
    if xyxy_true is not None:
        boxes = np.asarray(xyxy_true, dtype=np.float64)
        if boxes.shape != (count_true, 4):
            raise ValueError(
                f"`xyxy_true` must have shape ({count_true}, 4); got {boxes.shape}."
            )
    variances = (_resolve_keypoint_sigmas(sigmas, num_keypoints) * 2) ** 2

    true_xy = keypoints_true.astype(np.float64)[:, None, :, :]
    detection_xy = keypoints_detection.astype(np.float64)[None, :, :, :]
    dx = detection_xy[..., 0] - true_xy[..., 0]
    dy = detection_xy[..., 1] - true_xy[..., 1]
    num_visible = visible.sum(axis=1)
    use_box = np.zeros(count_true, dtype=bool)
    if boxes is not None:
        use_box = num_visible == 0
        # COCO xywh, then the same expansion and operation order as
        # pycocotools `computeOks`.
        x, y = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2] - x, boxes[:, 3] - y
        x0, x1 = (x - w)[:, None, None], (x + w * 2)[:, None, None]
        y0, y1 = (y - h)[:, None, None], (y + h * 2)[:, None, None]
        xd, yd = detection_xy[..., 0], detection_xy[..., 1]
        box_dx = np.maximum(0, x0 - xd) + np.maximum(0, xd - x1)
        box_dy = np.maximum(0, y0 - yd) + np.maximum(0, yd - y1)
        dx = np.where(use_box[:, None, None], box_dx, dx)
        dy = np.where(use_box[:, None, None], box_dy, dy)
    # Same operation order as pycocotools, so OKS values that land on a
    # threshold round the same way there and here.
    error = (dx**2 + dy**2) / variances / (area[:, None, None] + np.spacing(1)) / 2
    counted = visible | use_box[:, None]
    finite_coordinates = np.isfinite(dx) & np.isfinite(dy)
    similarity = np.where(
        counted[:, None, :] & finite_coordinates, np.exp(-error), 0.0
    ).sum(axis=2)
    num_counted = counted.sum(axis=1)[:, None]
    oks = np.zeros((count_true, count_det), dtype=np.float64)
    np.divide(similarity, num_counted, out=oks, where=num_counted > 0)
    return oks


def compact_mask_iou_batch(
    masks_true: Any,
    masks_detection: Any,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Compute pairwise overlap between two
    :class:`~supervision.detection.compact_mask.CompactMask` collections.

    Avoids materialising full ``(N, H, W)`` arrays by:

    1. Vectorised bounding-box pre-filter — pairs whose boxes do not overlap
       get IoU = 0 without any mask decoding.
    2. Sub-crop decoding — for overlapping pairs, only the intersection region
       of each crop is decoded and compared.
    3. Crop caching — each individual crop is decoded at most once even when it
       participates in many pairs.

    The result is numerically identical to running the dense
    :func:`~supervision.detection.utils.iou.mask_iou_batch` on
    ``np.asarray(masks_true)`` / ``np.asarray(masks_detection)``.

    Args:
        masks_true: :class:`~supervision.detection.compact_mask.CompactMask`
            holding the ground-truth masks.
        masks_detection: :class:`~supervision.detection.compact_mask.CompactMask`
            holding the detection masks.
        overlap_metric: :class:`~supervision.detection.utils.iou.OverlapMetric`
            — ``IOU`` or ``IOS``.

    Returns:
        Float array of shape ``(N1, N2)`` with pairwise overlap values.
    """
    if masks_true.image_shape != masks_detection.image_shape:
        raise ValueError(
            "masks_true and masks_detection must share the same (H, W); got "
            f"{masks_true.image_shape} and {masks_detection.image_shape}."
        )
    if overlap_metric not in (OverlapMetric.IOU, OverlapMetric.IOS):
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )

    n1: int = len(masks_true)
    n2: int = len(masks_detection)
    result: npt.NDArray[np.floating] = np.zeros((n1, n2), dtype=float)

    if n1 == 0 or n2 == 0:
        return result

    areas_a: npt.NDArray[np.int64] = masks_true.area
    symmetric = masks_true is masks_detection
    if symmetric:
        areas_b = areas_a
        nonempty = np.flatnonzero(areas_a > 0)
        result[nonempty, nonempty] = 1.0
    else:
        areas_b = masks_detection.area

    # Use the same count precision as dense mask overlap.
    pixel_count = masks_true.image_shape[0] * masks_true.image_shape[1]
    count_dtype = _count_dtype(pixel_count)
    for idx_a, idx_b, pixel_overlap in _compact_mask_intersections(
        masks_true, masks_detection, upper_triangle=symmetric
    ):
        if pixel_overlap == 0:
            continue
        inter = count_dtype(pixel_overlap)
        area_a_i = count_dtype(areas_a[idx_a])
        area_b_j = count_dtype(areas_b[idx_b])
        if overlap_metric == OverlapMetric.IOU:
            denominator = area_a_i + area_b_j - inter
        else:
            denominator = min(area_a_i, area_b_j)
        overlap = inter / denominator if denominator > 0 else 0.0
        result[idx_a, idx_b] = overlap
        if symmetric:
            result[idx_b, idx_a] = overlap

    return result


def _compact_mask_intersections(
    masks_true: CompactMask,
    masks_detection: CompactMask,
    *,
    upper_triangle: bool = False,
) -> Iterator[tuple[int, int, int]]:
    """Yield mask indices and exact intersection counts for overlapping crops.

    For a self-comparison, callers can request only pairs above the diagonal. Other
    callers receive all overlapping pairs in row order.
    """
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

    # Keep only the active row and the columns needed by later rows.
    crops_b: dict[int, npt.NDArray[np.bool_]] = {}
    for row in np.flatnonzero(bbox_overlap.any(axis=1)):
        idx_a = int(row)
        first_col = idx_a + 1 if upper_triangle else 0
        matching_b = np.flatnonzero(bbox_overlap[idx_a, first_col:])
        crop_a = crops_b.pop(idx_a, None) if upper_triangle else None
        if len(matching_b) == 0:
            continue
        if crop_a is None:
            crop_a = masks_true.crop(idx_a)
        ox_a, oy_a = int(x1a[idx_a]), int(y1a[idx_a])
        for col in matching_b:
            idx_b = first_col + int(col)
            crop_b = crops_b.get(idx_b)
            if crop_b is None:
                crop_b = masks_detection.crop(idx_b)
                crops_b[idx_b] = crop_b

            lx1 = int(ix1[idx_a, idx_b])
            ly1 = int(iy1[idx_a, idx_b])
            lx2 = int(ix2[idx_a, idx_b]) + 1
            ly2 = int(iy2[idx_a, idx_b]) + 1
            sub_a = crop_a[ly1 - oy_a : ly2 - oy_a, lx1 - ox_a : lx2 - ox_a]
            ox_b, oy_b = int(x1b[idx_b]), int(y1b[idx_b])
            sub_b = crop_b[ly1 - oy_b : ly2 - oy_b, lx1 - ox_b : lx2 - ox_b]
            yield idx_a, idx_b, int(np.count_nonzero(sub_a & sub_b))


def _mask_iou_batch_split(
    masks_true: npt.NDArray[Any],
    masks_detection: npt.NDArray[Any],
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Internal function.

    Compute Intersection over Union (IoU) of two sets of masks - `masks_true` and
    `masks_detection`.

    Args:
        masks_true: 3D `np.ndarray` representing ground-truth masks.
        masks_detection: 3D `np.ndarray` representing detection masks.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """
    # float32 counts pixels exactly up to 2**24; for larger masks (beyond
    # ~4096x4096) we promote to float64 so the counts stay exact.
    pixels = int(np.prod(masks_true.shape[1:]))
    count_dtype = _count_dtype(pixels)

    # The overlap of two binary masks is the dot product of their flattened
    # pixels, so the whole (N, M) intersection matrix is a single matmul.
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

    return _overlap_from_areas(
        intersection_area, masks_true_area, masks_detection_area, overlap_metric
    )


def _packed_mask_iou_batch(
    true_words: npt.NDArray[np.uint64],
    detection_words: npt.NDArray[np.uint64],
    count_dtype: type[np.floating],
    overlap_metric: OverlapMetric,
    detection_area: npt.NDArray[np.floating],
    detection_columns: npt.NDArray[np.intp],
    block_rows: int,
) -> npt.NDArray[np.floating]:
    """Internal function.

    Pairwise overlap of two sets of masks packed by
    :func:`~supervision.detection.utils.iou._pack_masks`, counted with
    popcount so the counts are exact integers. Intersections are counted for
    blocks of true-mask rows at once, so the row loop runs `N / block_rows`
    times instead of `N` times.

    Args:
        true_words: 2D `uint64` `np.ndarray` of shape `(N, K)`.
        detection_words: 2D `uint64` `np.ndarray` of shape `(M, K)`, already
            reduced to `detection_columns`.
        count_dtype: Float dtype the pixel counts are cast to.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        detection_area: Precomputed detection mask areas, taken
            before any column was dropped from `detection_words`.
        detection_columns: Indices of the word columns that are
            non-empty in the detections, with `detection_words` already
            reduced to them. Passing them lets a chunked caller reduce the
            detections once instead of once per chunk.
        block_rows: Rows of `true_words` intersected per step.

    Returns:
        Pairwise overlap of shape `(N, M)`.
    """
    masks_true_area = _count_pixels(true_words).astype(count_dtype)
    # Words that are empty in the detections cannot add to any intersection; the
    # areas above were taken before they were dropped.
    true_words = true_words[:, detection_columns]

    intersection_area = np.empty(
        (len(true_words), len(detection_words)), dtype=count_dtype
    )
    for start in range(0, len(true_words), block_rows):
        block = true_words[start : start + block_rows]
        intersection_area[start : start + block_rows] = _count_pixels(
            block[:, None, :] & detection_words[None, :, :]
        )

    return _overlap_from_areas(
        intersection_area, masks_true_area, detection_area, overlap_metric
    )


def _pack_masks(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint64]:
    """Internal function.

    Pack boolean masks into one row of `uint64` words per mask, zero-padded to
    a whole word. Requires NumPy >= 2.0 to count them with
    :func:`~supervision.detection.utils.iou._count_pixels`.

    Args:
        masks: 3D boolean `np.ndarray` of shape `(N, H, W)`.

    Returns:
        2D `uint64` `np.ndarray` of shape `(N, K)`.
    """
    flat = masks.reshape(masks.shape[0], int(np.prod(masks.shape[1:])))
    packed = np.packbits(flat, axis=1)
    # np.pad copies even for a zero width, so pad only when a word is incomplete.
    pad_width = -packed.shape[1] % 8
    if pad_width:
        packed = np.pad(packed, ((0, 0), (0, pad_width)))
    return np.ascontiguousarray(packed).view(np.uint64)


def _count_pixels(words: npt.NDArray[Any]) -> npt.NDArray[np.integer]:
    """Internal function.

    Count the set pixels along the last axis of masks packed by `_pack_masks`.

    Args:
        words: `np.ndarray` returned by `_pack_masks`, or a slice or bitwise
            combination of it.

    Returns:
        Integer `np.ndarray` of pixel counts.
    """
    return cast(npt.NDArray[np.integer], np.bitwise_count(words).sum(axis=-1))


def _overlap_from_areas(
    intersection_area: npt.NDArray[np.floating],
    masks_true_area: npt.NDArray[np.floating],
    masks_detection_area: npt.NDArray[np.floating],
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Internal function.

    Turn pixel counts of two sets of masks into their pairwise overlap.

    Args:
        intersection_area: 2D `np.ndarray` of shape `(N, M)` with the
            intersection pixel counts.
        masks_true_area: 1D `np.ndarray` of shape `(N,)` with mask areas.
        masks_detection_area: 1D `np.ndarray` of shape `(M,)` with mask areas.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        Pairwise overlap of shape `(N, M)`.
    """
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


def _mask_iou_batch_packed(
    masks_true: npt.NDArray[np.bool_],
    masks_detection: npt.NDArray[np.bool_],
    overlap_metric: OverlapMetric,
    memory_limit: int,
) -> npt.NDArray[np.floating]:
    """Internal function.

    Chunked bit-packed overlap of two non-empty boolean mask stacks, sized to
    `memory_limit`. Counts are exact integers, so the result equals the float
    matmul path of :func:`~supervision.detection.utils.iou.mask_iou_batch`.

    Args:
        masks_true: 3D boolean `np.ndarray` of shape `(N, H, W)`.
        masks_detection: 3D boolean `np.ndarray` of shape `(M, H, W)`.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        memory_limit: Memory limit in MB that sizes the chunks of `masks_true`.

    Returns:
        Pairwise overlap of shape `(N, M)`.
    """
    true_count, detection_count = len(masks_true), len(masks_detection)
    pixels = masks_true.shape[1] * masks_true.shape[2]
    limit_bytes = memory_limit * 1024 * 1024
    count_dtype = _count_dtype(pixels)
    word_bytes = ((pixels + 63) // 64) * 8

    # Fixed buffers: packing the detections peaks at the packed words plus their
    # padded copy, and the reduced detection words stay beside the originals. A
    # strided batch only needs a spatial copy when its two spatial strides cannot
    # form a flat view.
    packed_detection_bytes = detection_count * word_bytes
    fixed_bytes = 2 * packed_detection_bytes + detection_count * 8
    height, width = masks_detection.shape[1:]
    if (
        height > 1
        and width > 1
        and masks_detection.strides[1] != width * masks_detection.strides[2]
    ):
        fixed_bytes += masks_detection.size
    # The kernel intersects a block of true rows at once; each row costs the
    # bitwise AND plus its popcount, and one row is the unavoidable minimum.
    row_temp_bytes = packed_detection_bytes + packed_detection_bytes // 8
    detection_floor = fixed_bytes + row_temp_bytes
    if detection_floor > limit_bytes > 0:
        warnings.warn(
            f"detection masks ({detection_floor // 1024 // 1024} MB) exceed "
            f"memory_limit ({memory_limit} MB); chunking cannot reduce peak "
            "memory below this floor.",
            UserWarning,
            stacklevel=3,
        )
    # Give the block temporaries at most half of the headroom, so the chunks of
    # true masks keep room too and a tight limit does not force one-row chunks.
    headroom_bytes = max(limit_bytes - fixed_bytes, 0)
    block_budget = min(_PACKED_BLOCK_BYTES, headroom_bytes // 2)
    block_rows = max(block_budget // max(row_temp_bytes, 1), 1)

    packed_words = _pack_masks(masks_detection)
    detection_area = _count_pixels(packed_words).astype(count_dtype)
    # Reduce the detections to their non-empty words once, not once per chunk.
    detection_columns = np.flatnonzero(packed_words.any(axis=0))
    detection_words = np.ascontiguousarray(packed_words[:, detection_columns])
    # Self-overlap (NMS) compares one stack with itself: slice its packed words
    # instead of packing every chunk again.
    packed_true = packed_words if masks_true is masks_detection else None
    del packed_words

    output_bytes = true_count * detection_count * 8
    per_true_row = 3 * word_bytes + 8 + 3 * detection_count * 8
    if not masks_true.flags.c_contiguous:
        per_true_row += pixels
    detection_bytes = fixed_bytes + block_rows * row_temp_bytes
    step = max((limit_bytes - detection_bytes - output_bytes) // per_true_row, 1)

    result = np.empty((true_count, detection_count), dtype=float)
    for chunk_start in range(0, true_count, step):
        chunk_end = min(chunk_start + step, true_count)
        if packed_true is None:
            true_words = _pack_masks(masks_true[chunk_start:chunk_end])
        else:
            true_words = packed_true[chunk_start:chunk_end]
        result[chunk_start:chunk_end] = _packed_mask_iou_batch(
            true_words,
            detection_words,
            count_dtype,
            overlap_metric,
            detection_area,
            detection_columns,
            block_rows,
        )
    return result


def mask_iou_batch(
    masks_true: npt.NDArray[Any] | CompactMask,
    masks_detection: npt.NDArray[Any] | CompactMask,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    memory_limit: int = 1024 * 5,
) -> npt.NDArray[np.floating]:
    """Compute Intersection over Union (IoU) of two sets of masks - `masks_true` and
    `masks_detection`.

    Accepts both dense ``(N, H, W)`` boolean arrays and
    :class:`~supervision.detection.compact_mask.CompactMask` objects.
    When both inputs are :class:`~supervision.detection.compact_mask.CompactMask`,
    the computation uses
    :func:`~supervision.detection.utils.iou.compact_mask_iou_batch`
    to avoid materialising
    full ``(N, H, W)`` arrays.

    Args:
        masks_true: 3D `np.ndarray` representing ground-truth masks.
        masks_detection: 3D `np.ndarray` representing detection masks.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        memory_limit: Memory limit in MB, default is 1024 * 5 MB (5GB).
            Controls chunking of ``masks_true`` based on estimated working
            buffers. Boolean masks whose float32 copies fit the limit use one
            float matrix product. Boolean masks that do not fit it use
            bit-packed counting on NumPy >= 2.0, which needs 16-32x less memory
            and so warns only when the packed buffers alone exceed the limit;
            other inputs chunk the float copies. A ``UserWarning`` is raised
            when detection buffers alone exceed the limit, as chunking cannot
            reduce peak memory below that floor. The output and a minimum
            one-row chunk may also exceed the limit. Ignored when both inputs
            are :class:`~supervision.detection.compact_mask.CompactMask`.

    Returns:
        Pairwise IoU of masks from `masks_true` and `masks_detection`.

    Raises:
        ValueError: If ``masks_true`` or ``masks_detection`` are not 3D
            ``(N, H, W)`` arrays.
        ValueError: If ``masks_true`` and ``masks_detection`` do not share the same
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
    itemsize = np.dtype(_count_dtype(pixels)).itemsize
    limit_bytes = memory_limit * 1024 * 1024
    detection_bytes = masks_detection.shape[0] * pixels * itemsize
    per_true_row = pixels * itemsize + 3 * masks_detection.shape[0] * 8
    float_fits = detection_bytes + masks_true.shape[0] * per_true_row <= limit_bytes
    # The float matmul is the faster kernel on a fast BLAS, so bit-packed counting
    # is kept for the inputs where its 16-32x smaller working set matters: those
    # whose float copies would not fit `memory_limit`.
    if not float_fits and _use_packed(masks_true, masks_detection):
        return _mask_iou_batch_packed(
            masks_true, masks_detection, overlap_metric, memory_limit
        )
    if detection_bytes > limit_bytes > 0:
        warnings.warn(
            f"detection masks ({detection_bytes // 1024 // 1024} MB) exceed "
            f"memory_limit ({memory_limit} MB); chunking cannot reduce peak "
            "memory below this floor.",
            UserWarning,
            stacklevel=2,
        )
    if float_fits:
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
