from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.metrics.core import MetricTarget

if TYPE_CHECKING:
    from supervision.detection.core import Detections

SIZE_THRESHOLDS = (32**2, 96**2)


class ObjectSizeCategory(Enum):
    ANY = -1
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


def get_object_size_category(
    data: npt.NDArray, metric_target: MetricTarget
) -> npt.NDArray[np.int_]:
    """
    Get the size category of an object. Distinguish based on the metric target.

    Args:
        data (np.ndarray): The object data, shaped (N, ...).
        metric_target (MetricTarget): Determines whether boxes, masks or
            oriented bounding boxes are used.

    Returns:
        (np.ndarray) The size category of each object, matching
        the enum values of ObjectSizeCategory. Shaped (N,).
    """
    if metric_target == MetricTarget.BOXES:
        return get_bbox_size_category(data)
    if metric_target == MetricTarget.MASKS:
        return get_mask_size_category(data)
    if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
        return get_obb_size_category(data)
    raise ValueError("Invalid metric type")


def get_bbox_size_category(xyxy: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
    """
    Get the size category of a bounding boxes array.

    Args:
        xyxy (np.ndarray): The bounding boxes array shaped (N, 4).

    Returns:
        (np.ndarray) The size category of each bounding box, matching
        the enum values of ObjectSizeCategory. Shaped (N,).
    """
    if len(xyxy.shape) != 2 or xyxy.shape[1] != 4:
        raise ValueError("Bounding boxes must be shaped (N, 4)")

    width = xyxy[:, 2] - xyxy[:, 0]
    height = xyxy[:, 3] - xyxy[:, 1]
    areas = width * height

    result = np.full(areas.shape, ObjectSizeCategory.ANY.value)
    SM, LG = SIZE_THRESHOLDS
    result[areas < SM] = ObjectSizeCategory.SMALL.value
    result[(areas >= SM) & (areas < LG)] = ObjectSizeCategory.MEDIUM.value
    result[areas >= LG] = ObjectSizeCategory.LARGE.value
    return result


def get_mask_size_category(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """
    Get the size category of detection masks.

    Args:
        mask (np.ndarray): The mask array shaped (N, H, W).

    Returns:
        (np.ndarray) The size category of each mask, matching
        the enum values of ObjectSizeCategory. Shaped (N,).
    """
    if len(mask.shape) != 3:
        raise ValueError("Masks must be shaped (N, H, W)")

    areas = np.sum(mask, axis=(1, 2))

    result = np.full(areas.shape, ObjectSizeCategory.ANY.value)
    SM, LG = SIZE_THRESHOLDS
    result[areas < SM] = ObjectSizeCategory.SMALL.value
    result[(areas >= SM) & (areas < LG)] = ObjectSizeCategory.MEDIUM.value
    result[areas >= LG] = ObjectSizeCategory.LARGE.value
    return result


def get_obb_size_category(xyxyxyxy: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
    """
    Get the size category of a oriented bounding boxes array.

    Args:
        xyxyxyxy (np.ndarray): The bounding boxes array shaped (N, 8).

    Returns:
        (np.ndarray) The size category of each bounding box, matching
        the enum values of ObjectSizeCategory. Shaped (N,).
    """
    if len(xyxyxyxy.shape) != 2 or xyxyxyxy.shape[1] != 8:
        raise ValueError("Oriented bounding boxes must be shaped (N, 8)")

    # Shoelace formula
    x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy.T
    areas = 0.5 * np.abs(
        (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1)
        - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4)
    )

    result = np.full(areas.shape, ObjectSizeCategory.ANY.value)
    SM, LG = SIZE_THRESHOLDS
    result[areas < SM] = ObjectSizeCategory.SMALL.value
    result[(areas >= SM) & (areas < LG)] = ObjectSizeCategory.MEDIUM.value
    result[areas >= LG] = ObjectSizeCategory.LARGE.value
    return result


def get_detection_size_category(
    detections: Detections, metric_target: MetricTarget = MetricTarget.BOXES
) -> npt.NDArray[np.int_]:
    """
    Get the size category of a detections object.

    Args:
        xyxyxyxy (np.ndarray): The bounding boxes array shaped (N, 8).
        metric_target (MetricTarget): Determines whether boxes, masks or
            oriented bounding boxes are used.

    Returns:
        (np.ndarray) The size category of each bounding box, matching
        the enum values of ObjectSizeCategory. Shaped (N,).
    """
    if metric_target == MetricTarget.BOXES:
        return get_bbox_size_category(detections.xyxy)
    if metric_target == MetricTarget.MASKS:
        if detections.mask is None:
            raise ValueError("Detections mask is not available")
        return get_mask_size_category(detections.mask)
    if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
        if detections.data.get(ORIENTED_BOX_COORDINATES) is None:
            raise ValueError("Detections oriented bounding boxes are not available")
        return get_obb_size_category(
            np.array(detections.data[ORIENTED_BOX_COORDINATES])
        )
    raise ValueError("Invalid metric type")
