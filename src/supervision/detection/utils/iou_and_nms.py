from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.converters import polygon_to_mask
from supervision.detection.utils.masks import resize_masks


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
    x_min_true, y_min_true, x_max_true, y_max_true = boxes_true.T
    x_min_det, y_min_det, x_max_det, y_max_det = boxes_detection.T
    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]

    if count_true == 0 or count_det == 0:
        return cast(
            npt.NDArray[np.float32], np.empty((count_true, count_det), dtype=np.float32)
        )

    x_min_inter = np.empty((count_true, count_det), dtype=np.float32)
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


def _jaccard(box_a: list[float], box_b: list[float], is_crowd: bool) -> float:
    """
    Calculate the Jaccard index (intersection over union) between two bounding boxes.
    If a gt object is marked as "iscrowd", a dt is allowed to match any subregion
    of the gt. Choosing gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
    iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)

    Args:
        box_a: Box coordinates in the format [x, y, width, height].
        box_b: Box coordinates in the format [x, y, width, height].
        iscrowd: Flag indicating if the second box is a crowd region or not.

    Returns:
        Jaccard index between the two bounding boxes.
    """  # noqa: E501
    # Smallest number to avoid division by zero
    EPS = np.spacing(1)

    xa, ya, x2a, y2a = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    xb, yb, x2b, y2b = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    # Innermost left x
    xi = max(xa, xb)
    # Innermost right x
    x2i = min(x2a, x2b)
    # Same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # Calculate areas
    Aa = max(x2a - xa, 0.0) * max(y2a - ya, 0.0)
    Ab = max(x2b - xb, 0.0) * max(y2b - yb, 0.0)
    Ai = max(x2i - xi, 0.0) * max(y2i - yi, 0.0)

    if is_crowd:
        return float(Ai / (Aa + EPS))

    return float(Ai / (Aa + Ab - Ai + EPS))


def box_iou_batch_with_jaccard(
    boxes_true: list[list[float]],
    boxes_detection: list[list[float]],
    is_crowd: list[bool],
) -> npt.NDArray[np.float64]:
    """
    Calculate the intersection over union (IoU) between detection bounding boxes (dt)
    and ground-truth bounding boxes (gt).
    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics

    Args:
        boxes_true: List of ground-truth bounding boxes in the
            format [x, y, width, height].
        boxes_detection: List of detection bounding boxes in the
            format [x, y, width, height].
        is_crowd: List indicating if each ground-truth bounding box
            is a crowd region or not.

    Returns:
        Array of IoU values of shape (len(dt), len(gt)).

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
    assert len(is_crowd) == len(boxes_true), (
        "`is_crowd` must have the same length as `boxes_true`"
    )
    if len(boxes_detection) == 0 or len(boxes_true) == 0:
        return cast(npt.NDArray[np.float64], np.array([]))
    ious: npt.NDArray[np.float64] = np.zeros(
        (len(boxes_detection), len(boxes_true)), dtype=np.float64
    )
    for gt_idx, gt_box in enumerate(boxes_true):
        for det_idx, det_box in enumerate(boxes_detection):
            ious[det_idx, gt_idx] = _jaccard(det_box, gt_box, is_crowd[gt_idx])
    return ious


def oriented_box_iou_batch(
    boxes_true: npt.NDArray[np.number], boxes_detection: npt.NDArray[np.number]
) -> npt.NDArray[np.floating]:
    """
    Compute Intersection over Union (IoU) of two sets of oriented bounding boxes -
    `boxes_true` and `boxes_detection`. Both sets of boxes are expected to be in
    `((x1, y1), (x2, y2), (x3, y3), (x4, y4))` format.

    Args:
        boxes_true: A `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4, 2)` where `N` is number of true objects.
        boxes_detection: A `np.ndarray` representing detection boxes.
            `shape = (M, 4, 2)` where `M` is number of detected objects.

    Returns:
        Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    boxes_true = boxes_true.reshape(-1, 4, 2)
    boxes_detection = boxes_detection.reshape(-1, 4, 2)

    max_height = int(max(boxes_true[:, :, 0].max(), boxes_detection[:, :, 0].max()) + 1)
    # adding 1 because we are 0-indexed
    max_width = int(max(boxes_true[:, :, 1].max(), boxes_detection[:, :, 1].max()) + 1)

    mask_true = np.zeros((boxes_true.shape[0], max_height, max_width), dtype=np.uint8)
    for box_idx, box_true in enumerate(boxes_true):
        mask_true[box_idx] = polygon_to_mask(box_true, (max_width, max_height))

    mask_detection = np.zeros(
        (boxes_detection.shape[0], max_height, max_width), dtype=np.uint8
    )
    for box_idx, box_detection in enumerate(boxes_detection):
        mask_detection[box_idx] = polygon_to_mask(
            box_detection, (max_width, max_height)
        )

    ious = mask_iou_batch(mask_true, mask_detection)
    return ious


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
    intersection_area = np.logical_and(masks_true[:, None], masks_detection).sum(
        axis=(2, 3)
    )

    masks_true_area = masks_true.sum(axis=(1, 2))  # (area1, area2, ...)
    masks_detection_area = masks_detection.sum(axis=(1, 2))  # (area1)

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
    masks_true: npt.NDArray[Any],
    masks_detection: npt.NDArray[Any],
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
            Ignored when both inputs are CompactMask.

    Returns:
        Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """

    if isinstance(masks_true, CompactMask) and isinstance(masks_detection, CompactMask):
        return compact_mask_iou_batch(masks_true, masks_detection, overlap_metric)

    # Materialise any CompactMask that was passed alongside a dense array.
    if isinstance(masks_true, CompactMask):
        masks_true = np.asarray(masks_true)
    if isinstance(masks_detection, CompactMask):
        masks_detection = np.asarray(masks_detection)

    memory = (
        masks_true.shape[0]
        * masks_true.shape[1]
        * masks_true.shape[2]
        * masks_detection.shape[0]
        / 1024
        / 1024
    )
    if memory <= memory_limit:
        return _mask_iou_batch_split(masks_true, masks_detection, overlap_metric)

    ious = []
    step = max(
        memory_limit
        * 1024
        * 1024
        // (
            masks_detection.shape[0]
            * masks_detection.shape[1]
            * masks_detection.shape[2]
        ),
        1,
    )
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
    masks: npt.NDArray[Any],
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
        AssertionError: If `iou_threshold` is not within the closed
            range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, "
        f"{iou_threshold} given."
    )
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
        AssertionError: If `iou_threshold` is not within the
            closed range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, "
        f"{iou_threshold} given."
    )
    rows, columns = predictions.shape

    # add column #5 - category filled with zeros for agnostic nms
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    # sort predictions column #4 - score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes, overlap_metric)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        # drop detections with iou > iou_threshold and
        # same category as current detections
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return cast(npt.NDArray[np.bool_], keep[sort_index.argsort()])


def _group_overlapping_masks(
    predictions: npt.NDArray[np.float64],
    masks: npt.NDArray[np.float64],
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
    masks: npt.NDArray[Any],
    iou_threshold: float = 0.5,
    mask_dimension: int = 640,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
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
            to use for non-maximum suppression.
        mask_dimension: The dimension to which the masks should be
            resized before computing IOU values. Defaults to 640.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        A list of groups of prediction indices. Each inner list contains
            the indices of predictions whose masks overlap above `iou_threshold`
            according to the chosen `overlap_metric`, and should be merged or
            kept together as a single detection by non-maximum merging.

    Raises:
        AssertionError: If `iou_threshold` is not within the closed
            range from `0` to `1`.
    """

    if isinstance(masks, CompactMask):
        # _group_overlapping_masks needs dense arrays for logical_or union merging.
        # Note: np.asarray(masks) first materialises a full-resolution (N, H, W)
        # dense array before downscaling with resize_masks. This reduces the size
        # of the array used for overlap computation but does not avoid the initial
        # full-frame materialisation, which may still be memory-intensive for very
        # large images or object counts.
        masks = resize_masks(np.asarray(masks), mask_dimension)
    else:
        masks = resize_masks(masks, mask_dimension)
    masks_resized = masks

    if predictions.shape[1] == 5:
        return _group_overlapping_masks(
            predictions, masks_resized, iou_threshold, overlap_metric
        )

    category_ids = predictions[:, 5]
    merge_groups = []
    for category_id in np.unique(category_ids):
        curr_indices = np.where(category_ids == category_id)[0]
        merge_class_groups = _group_overlapping_masks(
            predictions[curr_indices],
            masks_resized[curr_indices],
            iou_threshold,
            overlap_metric,
        )

        for merge_class_group in merge_class_groups:
            merge_groups.append(curr_indices[merge_class_group].tolist())

    for merge_group in merge_groups:
        if len(merge_group) == 0:
            raise ValueError(
                f"Empty group detected when non-max-merging detections: {merge_groups}"
            )
    return merge_groups


def _group_overlapping_boxes(
    predictions: npt.NDArray[np.float64],
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

        merge_candidate = np.expand_dims(predictions[idx], axis=0)
        ious = box_iou_batch(
            predictions[order][:, :4], merge_candidate[:, :4], overlap_metric
        )
        ious = ious.flatten()

        above_threshold = ious >= iou_threshold
        merge_group = [idx, *np.flip(order[above_threshold]).tolist()]
        merge_groups.append(merge_group)
        order = order[~above_threshold]
    return merge_groups


def box_non_max_merge(
    predictions: npt.NDArray[np.float64],
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
    """
    if predictions.shape[1] == 5:
        return _group_overlapping_boxes(predictions, iou_threshold, overlap_metric)

    category_ids = predictions[:, 5]
    merge_groups = []
    for category_id in np.unique(category_ids):
        curr_indices = np.where(category_ids == category_id)[0]
        merge_class_groups = _group_overlapping_boxes(
            predictions[curr_indices], iou_threshold, overlap_metric
        )

        for merge_class_group in merge_class_groups:
            merge_groups.append(curr_indices[merge_class_group].tolist())

    for merge_group in merge_groups:
        if len(merge_group) == 0:
            raise ValueError(
                f"Empty group detected when non-max-merging detections: {merge_groups}"
            )
    return merge_groups
