from enum import Enum
from typing import List, Union

import numpy as np
import numpy.typing as npt

from supervision.detection.utils import box_iou_batch, mask_iou_batch


def resize_masks(masks: np.ndarray, max_dimension: int = 640) -> np.ndarray:
    """
    Resize all masks in the array to have a maximum dimension of max_dimension,
    maintaining aspect ratio.

    Args:
        masks (np.ndarray): 3D array of binary masks with shape (N, H, W).
        max_dimension (int): The maximum dimension for the resized masks.

    Returns:
        np.ndarray: Array of resized masks.
    """
    max_height = np.max(masks.shape[1])
    max_width = np.max(masks.shape[2])
    scale = min(max_dimension / max_height, max_dimension / max_width)

    new_height = int(scale * max_height)
    new_width = int(scale * max_width)

    x = np.linspace(0, max_width - 1, new_width).astype(int)
    y = np.linspace(0, max_height - 1, new_height).astype(int)
    xv, yv = np.meshgrid(x, y)

    resized_masks = masks[:, yv, xv]

    resized_masks = resized_masks.reshape(masks.shape[0], new_height, new_width)
    return resized_masks


def mask_non_max_suppression(
    predictions: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float = 0.5,
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
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.
        mask_dimension (int, optional): The dimension to which the masks should be
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
    ious = mask_iou_batch(masks_resized, masks_resized)
    categories = predictions[:, 5]

    keep = np.ones(rows, dtype=bool)
    for i in range(rows):
        if keep[i]:
            condition = (ious[i] > iou_threshold) & (categories[i] == categories)
            keep[i + 1 :] = np.where(condition[i + 1 :], False, keep[i + 1 :])

    return keep[sort_index.argsort()]


def box_non_max_suppression(
    predictions: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.

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
    ious = box_iou_batch(boxes, boxes)
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


def group_overlapping_boxes(
    predictions: npt.NDArray[np.float64], iou_threshold: float = 0.5
) -> List[List[int]]:
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.

    Args:
        predictions (npt.NDArray[np.float64]): An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.

    Returns:
        List[List[int]]: Groups of prediction indices be merged.
            Each group may have 1 or more elements.
    """
    merge_groups: List[List[int]] = []

    scores = predictions[:, 4]
    order = scores.argsort()

    while len(order) > 0:
        idx = int(order[-1])

        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break

        merge_candidate = np.expand_dims(predictions[idx], axis=0)
        ious = box_iou_batch(predictions[order][:, :4], merge_candidate[:, :4])
        ious = ious.flatten()

        above_threshold = ious >= iou_threshold
        merge_group = [idx] + np.flip(order[above_threshold]).tolist()
        merge_groups.append(merge_group)
        order = order[~above_threshold]
    return merge_groups


def box_non_max_merge(
    predictions: npt.NDArray[np.float64],
    iou_threshold: float = 0.5,
) -> List[List[int]]:
    """
    Apply greedy version of non-maximum merging per category to avoid detecting
    too many overlapping bounding boxes for a given object.

    Args:
        predictions (npt.NDArray[np.float64]): An array of shape `(n, 5)` or `(n, 6)`
            containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`,
            the confidence scores and class_ids. Omit class_id column to allow
            detections of different classes to be merged.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.

    Returns:
        List[List[int]]: Groups of prediction indices be merged.
            Each group may have 1 or more elements.
    """
    if predictions.shape[1] == 5:
        return group_overlapping_boxes(predictions, iou_threshold)

    category_ids = predictions[:, 5]
    merge_groups = []
    for category_id in np.unique(category_ids):
        curr_indices = np.where(category_ids == category_id)[0]
        merge_class_groups = group_overlapping_boxes(
            predictions[curr_indices], iou_threshold
        )

        for merge_class_group in merge_class_groups:
            merge_groups.append(curr_indices[merge_class_group].tolist())

    for merge_group in merge_groups:
        if len(merge_group) == 0:
            raise ValueError(
                f"Empty group detected when non-max-merging "
                f"detections: {merge_groups}"
            )
    return merge_groups


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


def validate_overlap_filter(
    strategy: Union[OverlapFilter, str],
) -> OverlapFilter:
    if isinstance(strategy, str):
        try:
            strategy = OverlapFilter(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Invalid strategy value: {strategy}. Must be one of "
                f"{[e.value for e in OverlapFilter]}"
            )
    return strategy
