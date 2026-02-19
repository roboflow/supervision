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
    """
    Enum for object size categories based on area in pixels.

    Small: area < 32^2
    Medium: 32^2 <= area < 96^2
    Large: area >= 96^2

    Example:
        ```pycon
        >>> from supervision.metrics.utils.object_size import ObjectSizeCategory
        >>> ObjectSizeCategory.SMALL.value
        1
        >>> ObjectSizeCategory.MEDIUM.value
        2
        >>> ObjectSizeCategory.LARGE.value
        3

        ```
    """

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
        data: The object data, shaped (N, ...).
        metric_target: Determines whether boxes, masks or
            oriented bounding boxes are used.

    Returns:
        The size category of each object, matching
        the enum values of ObjectSizeCategory. Shaped (N,).

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.metrics.core import MetricTarget
        >>> from supervision.metrics.utils.object_size import get_object_size_category
        >>> xyxy = np.array([
        ...     [0, 0, 10, 10],    # 100 (Small)
        ...     [0, 0, 50, 50],    # 2500 (Medium)
        ...     [0, 0, 100, 100]   # 10000 (Large)
        ... ])
        >>> get_object_size_category(xyxy, MetricTarget.BOXES)
        array([1, 2, 3])

        ```
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
        xyxy: The bounding boxes array shaped (N, 4).

    Returns:
        The size category of each bounding box, matching
        the enum values of ObjectSizeCategory. Shaped (N,).

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.metrics.utils.object_size import get_bbox_size_category
        >>> xyxy = np.array([
        ...     [0, 0, 31, 31],    # 961 (Small)
        ...     [0, 0, 32, 32],    # 1024 (Medium)
        ...     [0, 0, 95, 95],    # 9025 (Medium)
        ...     [0, 0, 96, 96]     # 9216 (Large)
        ... ])
        >>> get_bbox_size_category(xyxy)
        array([1, 2, 2, 3])

        ```
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
        mask: The mask array shaped (N, H, W).

    Returns:
        The size category of each mask, matching
        the enum values of ObjectSizeCategory. Shaped (N,).

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.metrics.utils.object_size import get_mask_size_category
        >>> mask = np.zeros((3, 100, 100), dtype=bool)
        >>> mask[0, 0:10, 0:10] = True   # 100 (Small)
        >>> mask[1, 0:50, 0:50] = True   # 2500 (Medium)
        >>> mask[2, 0:100, 0:100] = True # 10000 (Large)
        >>> get_mask_size_category(mask)
        array([1, 2, 3])

        ```
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
        xyxyxyxy: The bounding boxes array shaped (N, 4, 2).

    Returns:
        The size category of each bounding box, matching
        the enum values of ObjectSizeCategory. Shaped (N,).

    Example:
        ```pycon
        >>> import numpy as np
        >>> from supervision.metrics.utils.object_size import get_obb_size_category
        >>> obb = np.array([
        ...     [[0, 0], [10, 0], [10, 10], [0, 10]],   # 100 (Small)
        ...     [[0, 0], [50, 0], [50, 50], [0, 50]],   # 2500 (Medium)
        ...     [[0, 0], [100, 0], [100, 100], [0, 100]] # 10000 (Large)
        ... ])
        >>> get_obb_size_category(obb)
        array([1, 2, 3])

        ```
    """
    if len(xyxyxyxy.shape) != 3 or xyxyxyxy.shape[1] != 4 or xyxyxyxy.shape[2] != 2:
        raise ValueError("Oriented bounding boxes must be shaped (N, 4, 2)")

    # Shoelace formula
    x = xyxyxyxy[:, :, 0]
    y = xyxyxyxy[:, :, 1]
    x1, x2, x3, x4 = x.T
    y1, y2, y3, y4 = y.T
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
        detections: The detections object.
        metric_target: Determines whether boxes, masks or
            oriented bounding boxes are used.

    Returns:
        The size category of each bounding box, matching
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
