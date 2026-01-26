from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt

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
    def list(cls):
        return list(map(lambda c: c.value, cls))

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
        return list(map(lambda c: c.value, cls))

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
    box_true: list[float] | np.ndarray,
    box_detection: list[float] | np.ndarray,
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> float:
    """
    Compute overlap metric between two bounding boxes.

    Supports standard IOU (intersection-over-union) and IOS
    (intersection-over-smaller-area) metrics. Returns the overlap value in range
    `[0, 1]`.

    Args:
        box_true (`list[float]` or `numpy.array`): Ground truth box in format
          `(x_min, y_min, x_max, y_max)`.
        box_detection (`list[float]` or `numpy.array`): Detected box in format
          `(x_min, y_min, x_max, y_max)`.
        overlap_metric (`OverlapMetric` or `str`): Overlap type.
          Use `OverlapMetric.IOU` for IOU or
          `OverlapMetric.IOS` for IOS. Defaults to `OverlapMetric.IOU`.

    Returns:
        (`float`): Overlap value between boxes in `[0, 1]`.

    Raises:
        ValueError: If `overlap_metric` is not IOU or IOS.

    Examples:
        ```
        import supervision as sv

        box_true = [100, 100, 200, 200]
        box_detection = [150, 150, 250, 250]

        sv.box_iou(box_true, box_detection, overlap_metric=sv.OverlapMetric.IOU)
        # 0.14285714285714285

        sv.box_iou(box_true, box_detection, overlap_metric=sv.OverlapMetric.IOS)
        # 0.25
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
    boxes_true: np.ndarray,
    boxes_detection: np.ndarray,
    overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
) -> np.ndarray:
    """
    Compute pairwise overlap scores between batches of bounding boxes.

    Supports standard IOU (intersection-over-union) and IOS
    (intersection-over-smaller-area) metrics for all `boxes_true` and
    `boxes_detection` pairs. Returns a matrix of overlap values in range
    `[0, 1]`, matching each box from the first batch to each from the second.

    Args:
        boxes_true (`numpy.array`): Array of reference boxes in
            shape `(N, 4)` as `(x_min, y_min, x_max, y_max)`.
        boxes_detection (`numpy.array`): Array of detected boxes in
            shape `(M, 4)` as `(x_min, y_min, x_max, y_max)`.
        overlap_metric (`OverlapMetric` or `str`): Overlap type.
            Use `OverlapMetric.IOU` for intersection-over-union,
            `OverlapMetric.IOS` for intersection-over-smaller-area.
            Defaults to `OverlapMetric.IOU`.

    Returns:
        (`numpy.array`): Overlap matrix of shape `(N, M)`, where entry
            `[i, j]` is the overlap between `boxes_true[i]` and
            `boxes_detection[j]`.

    Raises:
        ValueError: If `overlap_metric` is not IOU or IOS.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        boxes_true = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ])
        boxes_detection = np.array([
            [150, 150, 250, 250],
            [320, 320, 420, 420]
        ])

        sv.box_iou_batch(boxes_true, boxes_detection, overlap_metric=OverlapMetric.IOU)
        # array([[0.14285715, 0.        ],
        #        [0.        , 0.47058824]])

        sv.box_iou_batch(boxes_true, boxes_detection, overlap_metric=OverlapMetric.IOS)
        # array([[0.25, 0.  ],
        #        [0.  , 0.64]])
        ```
    """
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    x_min_true, y_min_true, x_max_true, y_max_true = boxes_true.T
    x_min_det, y_min_det, x_max_det, y_max_det = boxes_detection.T
    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]

    if count_true == 0 or count_det == 0:
        return np.empty((count_true, count_det), dtype=np.float32)

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

    out = np.zeros_like(area_inter, dtype=np.float32)
    np.divide(area_inter, area_norm, out=out, where=area_norm > 0)
    return out


def _jaccard(box_a: list[float], box_b: list[float], is_crowd: bool) -> float:
    """
    Calculate the Jaccard index (intersection over union) between two bounding boxes.
    If a gt object is marked as "iscrowd", a dt is allowed to match any subregion
    of the gt. Choosing gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
    iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)

    Args:
        box_a (List[float]): Box coordinates in the format [x, y, width, height].
        box_b (List[float]): Box coordinates in the format [x, y, width, height].
        iscrowd (bool): Flag indicating if the second box is a crowd region or not.

    Returns:
        float: Jaccard index between the two bounding boxes.
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
        return Ai / (Aa + EPS)

    return Ai / (Aa + Ab - Ai + EPS)


def box_iou_batch_with_jaccard(
    boxes_true: list[list[float]],
    boxes_detection: list[list[float]],
    is_crowd: list[bool],
) -> np.ndarray:
    """
    Calculate the intersection over union (IoU) between detection bounding boxes (dt)
    and ground-truth bounding boxes (gt).
    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics

    Args:
        boxes_true (List[List[float]]): List of ground-truth bounding boxes in the \
            format [x, y, width, height].
        boxes_detection (List[List[float]]): List of detection bounding boxes in the \
            format [x, y, width, height].
        is_crowd (List[bool]): List indicating if each ground-truth bounding box \
            is a crowd region or not.

    Returns:
        np.ndarray: Array of IoU values of shape (len(dt), len(gt)).

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        boxes_true = [
            [10, 20, 30, 40],  # x, y, w, h
            [15, 25, 35, 45]
        ]
        boxes_detection = [
            [12, 22, 28, 38],
            [16, 26, 36, 46]
        ]
        is_crowd = [False, False]

        ious = sv.box_iou_batch_with_jaccard(
            boxes_true=boxes_true,
            boxes_detection=boxes_detection,
            is_crowd=is_crowd
        )
        # array([
        #     [0.8866..., 0.4960...],
        #     [0.4000..., 0.8622...]
        # ])
        ```
    """
    assert len(is_crowd) == len(boxes_true), (
        "`is_crowd` must have the same length as `boxes_true`"
    )
    if len(boxes_detection) == 0 or len(boxes_true) == 0:
        return np.array([])
    ious = np.zeros((len(boxes_detection), len(boxes_true)), dtype=np.float64)
    for g_idx, g in enumerate(boxes_true):
        for d_idx, d in enumerate(boxes_detection):
            ious[d_idx, g_idx] = _jaccard(d, g, is_crowd[g_idx])
    return ious


def oriented_box_iou_batch(
    boxes_true: np.ndarray, boxes_detection: np.ndarray
) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of oriented bounding boxes -
    `boxes_true` and `boxes_detection`. Both sets of boxes are expected to be in
    `((x1, y1), (x2, y2), (x3, y3), (x4, y4))` format.

    Args:
        boxes_true (np.ndarray): a `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4, 2)` where `N` is number of true objects.
        boxes_detection (np.ndarray): a `np.ndarray` representing detection boxes.
            `shape = (M, 4, 2)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    boxes_true = boxes_true.reshape(-1, 4, 2)
    boxes_detection = boxes_detection.reshape(-1, 4, 2)

    max_height = int(max(boxes_true[:, :, 0].max(), boxes_detection[:, :, 0].max()) + 1)
    # adding 1 because we are 0-indexed
    max_width = int(max(boxes_true[:, :, 1].max(), boxes_detection[:, :, 1].max()) + 1)

    mask_true = np.zeros((boxes_true.shape[0], max_height, max_width))
    for i, box_true in enumerate(boxes_true):
        mask_true[i] = polygon_to_mask(box_true, (max_width, max_height))

    mask_detection = np.zeros((boxes_detection.shape[0], max_height, max_width))
    for i, box_detection in enumerate(boxes_detection):
        mask_detection[i] = polygon_to_mask(box_detection, (max_width, max_height))

    ious = mask_iou_batch(mask_true, mask_detection)
    return ious


def _mask_iou_batch_split(
    masks_true: np.ndarray,
    masks_detection: np.ndarray,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> np.ndarray:
    """
    Internal function.
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true (np.ndarray): 3D `np.ndarray` representing ground-truth masks.
        masks_detection (np.ndarray): 3D `np.ndarray` representing detection masks.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        np.ndarray: Pairwise IoU of masks from `masks_true` and `masks_detection`.
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
    return ious


def mask_iou_batch(
    masks_true: np.ndarray,
    masks_detection: np.ndarray,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    memory_limit: int = 1024 * 5,
) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true (np.ndarray): 3D `np.ndarray` representing ground-truth masks.
        masks_detection (np.ndarray): 3D `np.ndarray` representing detection masks.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        memory_limit (int): memory limit in MB, default is 1024 * 5 MB (5GB).

    Returns:
        np.ndarray: Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """
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
    for i in range(0, masks_true.shape[0], step):
        ious.append(
            _mask_iou_batch_split(
                masks_true[i : i + step], masks_detection, overlap_metric
            )
        )

    return np.vstack(ious)


def mask_non_max_suppression(
    predictions: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on segmentation predictions.

    Args:
        predictions (np.ndarray): A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks (np.ndarray): A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        iou_threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        mask_dimension (int): The dimension to which the masks should be
            resized before computing IOU values. Defaults to 640.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after
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
    masks_resized = resize_masks(masks, mask_dimension)
    ious = mask_iou_batch(masks_resized, masks_resized, overlap_metric)
    categories = predictions[:, 5]

    keep = np.ones(rows, dtype=bool)
    for i in range(rows):
        if keep[i]:
            condition = (ious[i] > iou_threshold) & (categories[i] == categories)
            keep[i + 1 :] = np.where(condition[i + 1 :], False, keep[i + 1 :])

    return keep[sort_index.argsort()]


def box_non_max_suppression(
    predictions: np.ndarray,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after n
            on-maximum suppression.

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

    return keep[sort_index.argsort()]


def _group_overlapping_masks(
    predictions: npt.NDArray[np.float64],
    masks: npt.NDArray[np.float64],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Apply greedy version of non-maximum merging to avoid detecting too many

    Args:
        predictions (npt.NDArray[np.float64]): An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        masks (npt.NDArray[np.float64]): A 3D array of binary masks corresponding to
            the predictions.
        iou_threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        list[list[int]]: Groups of prediction indices be merged.
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
            above_threshold: np.ndarray = ious.flatten() >= iou_threshold
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
    predictions: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float = 0.5,
    mask_dimension: int = 640,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """
    Perform Non-Maximum Merging (NMM) on segmentation predictions.

    Args:
        predictions (np.ndarray): A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks (np.ndarray): A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        iou_threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression.
        mask_dimension (int): The dimension to which the masks should be
            resized before computing IOU values. Defaults to 640.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after
            non-maximum suppression.

    Raises:
        AssertionError: If `iou_threshold` is not within the closed
            range from `0` to `1`.
    """
    masks_resized = resize_masks(masks, mask_dimension)
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
        predictions (npt.NDArray[np.float64]): An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        iou_threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        list[list[int]]: Groups of prediction indices be merged.
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
        predictions (npt.NDArray[np.float64]): An array of shape `(n, 5)` or `(n, 6)`
            containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`,
            the confidence scores and class_ids. Omit class_id column to allow
            detections of different classes to be merged.
        iou_threshold (float): The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric (OverlapMetric): Metric used to compute the degree of overlap
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
