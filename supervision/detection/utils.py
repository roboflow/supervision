from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.geometry.core import Vector

MIN_POLYGON_POINT_COUNT = 3


def xyxy_to_polygons(box: np.ndarray) -> np.ndarray:
    """
    Convert an array of boxes to an array of polygons.
    Retains the input datatype.

    Args:
        box (np.ndarray): An array of boxes (N, 4), where each box is represented as a
            list of four coordinates in the format `(x_min, y_min, x_max, y_max)`.

    Returns:
        np.ndarray: An array of polygons (N, 4, 2), where each polygon is
            represented as a list of four coordinates in the format `(x, y)`.
    """
    polygon = np.zeros((box.shape[0], 4, 2), dtype=box.dtype)
    polygon[:, :, 0] = box[:, [0, 2, 2, 0]]
    polygon[:, :, 1] = box[:, [1, 1, 3, 3]]
    return polygon


def polygon_to_mask(polygon: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """Generate a mask from a polygon.

    Args:
        polygon (np.ndarray): The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with
            `1`'s and the rest is filled with `0`'s.
    """
    width, height = map(int, resolution_wh)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)
    return mask


def box_iou(
    box_true: Union[List[float], np.ndarray],
    box_detection: Union[List[float], np.ndarray],
) -> float:
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Both `box_true` and `box_detection` should be in (x_min, y_min, x_max, y_max)
    format.

    Note:
        Use `box_iou` when computing IoU between two individual boxes.
        For comparing multiple boxes (arrays of boxes), use `box_iou_batch` for better
        performance.

    Args:
        box_true (Union[List[float], np.ndarray]): A single bounding box represented as
            [x_min, y_min, x_max, y_max].
        box_detection (Union[List[float], np.ndarray]):
            A single bounding box represented as [x_min, y_min, x_max, y_max].

    Returns:
        IoU (float): IoU score between the two boxes. Ranges from 0.0 (no overlap)
            to 1.0 (perfect overlap).

    """
    box_true = np.array(box_true)
    box_detection = np.array(box_detection)

    inter_x1 = max(box_true[0], box_detection[0])
    inter_y1 = max(box_true[1], box_detection[1])
    inter_x2 = min(box_true[2], box_detection[2])
    inter_y2 = min(box_true[3], box_detection[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_true = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
    area_detection = (box_detection[2] - box_detection[0]) * (
        box_detection[3] - box_detection[1]
    )

    union_area = area_true + area_detection - inter_area

    return inter_area / union_area + 1e-6


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Note:
        Use `box_iou` when computing IoU between two individual boxes.
        For comparing multiple boxes (arrays of boxes), use `box_iou_batch` for better
        performance.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    ious = area_inter / (area_true[:, None] + area_detection - area_inter)
    ious = np.nan_to_num(ious)
    return ious


def _mask_iou_batch_split(
    masks_true: np.ndarray, masks_detection: np.ndarray
) -> np.ndarray:
    """
    Internal function.
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true (np.ndarray): 3D `np.ndarray` representing ground-truth masks.
        masks_detection (np.ndarray): 3D `np.ndarray` representing detection masks.

    Returns:
        np.ndarray: Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """
    intersection_area = np.logical_and(masks_true[:, None], masks_detection).sum(
        axis=(2, 3)
    )

    masks_true_area = masks_true.sum(axis=(1, 2))
    masks_detection_area = masks_detection.sum(axis=(1, 2))
    union_area = masks_true_area[:, None] + masks_detection_area - intersection_area

    return np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(intersection_area, dtype=float),
        where=union_area != 0,
    )


def mask_iou_batch(
    masks_true: np.ndarray,
    masks_detection: np.ndarray,
    memory_limit: int = 1024 * 5,
) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true (np.ndarray): 3D `np.ndarray` representing ground-truth masks.
        masks_detection (np.ndarray): 3D `np.ndarray` representing detection masks.
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
        return _mask_iou_batch_split(masks_true, masks_detection)

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
        ious.append(_mask_iou_batch_split(masks_true[i : i + step], masks_detection))

    return np.vstack(ious)


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


def clip_boxes(xyxy: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """
    Clips bounding boxes coordinates to fit within the frame resolution.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in
            the format `(x_min, y_min, x_max, y_max)`.
        resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
            representing the resolution of the frame.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box with coordinates clipped to fit
            within the frame resolution.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 20, 300, 200],
            [15, 25, 350, 450],
            [-10, -20, 30, 40]
        ])

        sv.clip_boxes(xyxy=xyxy, resolution_wh=(320, 240))
        # array([
        #     [ 10,  20, 300, 200],
        #     [ 15,  25, 320, 240],
        #     [  0,   0,  30,  40]
        # ])
        ```
    """
    result = np.copy(xyxy)
    width, height = resolution_wh
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result


def pad_boxes(xyxy: np.ndarray, px: int, py: Optional[int] = None) -> np.ndarray:
    """
    Pads bounding boxes coordinates with a constant padding.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in the format
            `(x_min, y_min, x_max, y_max)`.
        px (int): The padding value to be added to both the left and right sides of
            each bounding box.
        py (Optional[int]): The padding value to be added to both the top and bottom
            sides of each bounding box. If not provided, `px` will be used for both
            dimensions.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds to a
            bounding box with coordinates padded according to the provided padding
            values.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 20, 30, 40],
            [15, 25, 35, 45]
        ])

        sv.pad_boxes(xyxy=xyxy, px=5, py=10)
        # array([
        #     [ 5, 10, 35, 50],
        #     [10, 15, 40, 55]
        # ])
        ```
    """
    if py is None:
        py = px

    result = xyxy.copy()
    result[:, [0, 1]] -= [px, py]
    result[:, [2, 3]] += [px, py]

    return result


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(x, y, width, height)`
    format to `(x_min, y_min, x_max, y_max)` format.

    Args:
        xywh (np.ndarray): A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(x, y, width, height)`.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xywh = np.array([
            [10, 20, 30, 40],
            [15, 25, 35, 45]
        ])

        sv.xywh_to_xyxy(xywh=xywh)
        # array([
        #     [10, 20, 40, 60],
        #     [15, 25, 50, 70]
        # ])
        ```
    """
    xyxy = xywh.copy()
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
    return xyxy


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    format to `(x, y, width, height)` format.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(x_min, y_min, x_max,
            y_max)`.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x, y, width, height)`.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 20, 40, 60],
            [15, 25, 50, 70]
        ])

        sv.xyxy_to_xywh(xyxy=xyxy)
        # array([
        #     [10, 20, 30, 40],
        #     [15, 25, 35, 45]
        # ])
        ```
    """
    xywh = xyxy.copy()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def xcycwh_to_xyxy(xcycwh: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(center_x, center_y, width, height)`
    format to `(x_min, y_min, x_max, y_max)` format.

    Args:
        xcycwh (np.ndarray): A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(center_x, center_y, width,
            height)`.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xcycwh = np.array([
            [50, 50, 20, 30],
            [30, 40, 10, 15]
        ])

        sv.xcycwh_to_xyxy(xcycwh=xcycwh)
        # array([
        #     [40, 35, 60, 65],
        #     [25, 32.5, 35, 47.5]
        # ])
        ```
    """
    xyxy = xcycwh.copy()
    xyxy[:, 0] = xcycwh[:, 0] - xcycwh[:, 2] / 2
    xyxy[:, 1] = xcycwh[:, 1] - xcycwh[:, 3] / 2
    xyxy[:, 2] = xcycwh[:, 0] + xcycwh[:, 2] / 2
    xyxy[:, 3] = xcycwh[:, 1] + xcycwh[:, 3] / 2
    return xyxy


def xyxy_to_xcycarh(xyxy: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    into measurement space to format `(center x, center y, aspect ratio, height)`,
    where the aspect ratio is `width / height`.

    Args:
        xyxy (np.ndarray): Bounding box in format `(x1, y1, x2, y2)`.
            Expected shape is `(N, 4)`.
    Returns:
        np.ndarray: Bounding box in format
            `(center x, center y, aspect ratio, height)`. Shape `(N, 4)`.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 20, 40, 60],
            [15, 25, 50, 70]
        ])

        sv.xyxy_to_xcycarh(xyxy=xyxy)
        # array([
        #     [25.  , 40.  ,  0.75, 40.  ],
        #     [32.5 , 47.5 ,  0.77777778, 45.  ]
        # ])
        ```

    """
    if xyxy.size == 0:
        return np.empty((0, 4), dtype=float)

    x1, y1, x2, y2 = xyxy.T
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    aspect_ratio = np.divide(
        width,
        height,
        out=np.zeros_like(width, dtype=float),
        where=height != 0,
    )
    result = np.column_stack((center_x, center_y, aspect_ratio, height))
    return result.astype(float)


def mask_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """
    Converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Parameters:
        masks (np.ndarray): A 3D `np.array` of shape `(N, W, H)`
            containing 2D bool masks

    Returns:
        np.ndarray: A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask
    """
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)

        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


def mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    """
    Converts a binary mask to a list of polygons.

    Parameters:
        mask (np.ndarray): A binary mask represented as a 2D NumPy array of
            shape `(H, W)`, where H and W are the height and width of
            the mask, respectively.

    Returns:
        List[np.ndarray]: A list of polygons, where each polygon is represented by a
            NumPy array of shape `(N, 2)`, containing the `x`, `y` coordinates
            of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
        if contour.shape[0] >= MIN_POLYGON_POINT_COUNT
    ]


def filter_polygons_by_area(
    polygons: List[np.ndarray],
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Filters a list of polygons based on their area.

    Parameters:
        polygons (List[np.ndarray]): A list of polygons, where each polygon is
            represented by a NumPy array of shape `(N, 2)`,
            containing the `x`, `y` coordinates of the points.
        min_area (Optional[float]): The minimum area threshold.
            Only polygons with an area greater than or equal to this value
            will be included in the output. If set to None,
            no minimum area constraint will be applied.
        max_area (Optional[float]): The maximum area threshold.
            Only polygons with an area less than or equal to this value
            will be included in the output. If set to None,
            no maximum area constraint will be applied.

    Returns:
        List[np.ndarray]: A new list of polygons containing only those with
            areas within the specified thresholds.
    """
    if min_area is None and max_area is None:
        return polygons
    ares = [cv2.contourArea(polygon) for polygon in polygons]
    return [
        polygon
        for polygon, area in zip(polygons, ares)
        if (min_area is None or area >= min_area)
        and (max_area is None or area <= max_area)
    ]


def polygon_to_xyxy(polygon: np.ndarray) -> np.ndarray:
    """
    Converts a polygon represented by a NumPy array into a bounding box.

    Parameters:
        polygon (np.ndarray): A polygon represented by a NumPy array of shape `(N, 2)`,
            containing the `x`, `y` coordinates of the points.

    Returns:
        np.ndarray: A 1D NumPy array containing the bounding box
            `(x_min, y_min, x_max, y_max)` of the input polygon.
    """
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    return np.array([x_min, y_min, x_max, y_max])


def approximate_polygon(
    polygon: np.ndarray, percentage: float, epsilon_step: float = 0.05
) -> np.ndarray:
    """
    Approximates a given polygon by reducing a certain percentage of points.

    This function uses the Ramer-Douglas-Peucker algorithm to simplify the input
        polygon by reducing the number of points
        while preserving the general shape.

    Parameters:
        polygon (np.ndarray): A 2D NumPy array of shape `(N, 2)` containing
            the `x`, `y` coordinates of the input polygon's points.
        percentage (float): The percentage of points to be removed from the
            input polygon, in the range `[0, 1)`.
        epsilon_step (float): Approximation accuracy step.
            Epsilon is the maximum distance between the original curve
            and its approximation.

    Returns:
        np.ndarray: A new 2D NumPy array of shape `(M, 2)`,
            where `M <= N * (1 - percentage)`, containing
            the `x`, `y` coordinates of the
            approximated polygon's points.
    """

    if percentage < 0 or percentage >= 1:
        raise ValueError("Percentage must be in the range [0, 1).")

    target_points = max(int(len(polygon) * (1 - percentage)), 3)

    if len(polygon) <= target_points:
        return polygon

    epsilon = 0
    approximated_points = polygon
    while True:
        epsilon += epsilon_step
        new_approximated_points = cv2.approxPolyDP(polygon, epsilon, closed=True)
        if len(new_approximated_points) > target_points:
            approximated_points = new_approximated_points
        else:
            break

    return np.squeeze(approximated_points, axis=1)


def extract_ultralytics_masks(yolov8_results) -> Optional[np.ndarray]:
    if not yolov8_results.masks:
        return None

    orig_shape = yolov8_results.orig_shape
    inference_shape = tuple(yolov8_results.masks.data.shape[1:])

    pad = (0, 0)

    if inference_shape != orig_shape:
        gain = min(
            inference_shape[0] / orig_shape[0],
            inference_shape[1] / orig_shape[1],
        )
        pad = (
            (inference_shape[1] - orig_shape[1] * gain) / 2,
            (inference_shape[0] - orig_shape[0] * gain) / 2,
        )

    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(inference_shape[0] - pad[1]), int(inference_shape[1] - pad[0])

    mask_maps = []
    masks = yolov8_results.masks.data.cpu().numpy()
    for i in range(masks.shape[0]):
        mask = masks[i]
        mask = mask[top:bottom, left:right]

        if mask.shape != orig_shape:
            mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))

        mask_maps.append(mask)

    return np.asarray(mask_maps, dtype=bool)


def process_roboflow_result(
    roboflow_result: dict,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, Union[List[np.ndarray], np.ndarray]],
]:
    if not roboflow_result["predictions"]:
        return (
            np.empty((0, 4)),
            np.empty(0),
            np.empty(0),
            None,
            None,
            {CLASS_NAME_DATA_FIELD: np.empty(0)},
        )

    xyxy = []
    confidence = []
    class_id = []
    class_name = []
    masks = []
    tracker_ids = []

    image_width = int(roboflow_result["image"]["width"])
    image_height = int(roboflow_result["image"]["height"])

    for prediction in roboflow_result["predictions"]:
        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        x_min = x - width / 2
        y_min = y - height / 2
        x_max = x_min + width
        y_max = y_min + height

        if "points" not in prediction:
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            if "tracker_id" in prediction:
                tracker_ids.append(prediction["tracker_id"])
        elif len(prediction["points"]) >= 3:
            polygon = np.array(
                [[point["x"], point["y"]] for point in prediction["points"]], dtype=int
            )
            mask = polygon_to_mask(polygon, resolution_wh=(image_width, image_height))
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            masks.append(mask)
            if "tracker_id" in prediction:
                tracker_ids.append(prediction["tracker_id"])

    xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    masks = np.array(masks, dtype=bool) if len(masks) > 0 else None
    tracker_id = np.array(tracker_ids).astype(int) if len(tracker_ids) > 0 else None
    data = {CLASS_NAME_DATA_FIELD: class_name}

    return xyxy, confidence, class_id, masks, tracker_id, data


def move_boxes(
    xyxy: npt.NDArray[np.float64], offset: npt.NDArray[np.int32]
) -> npt.NDArray[np.float64]:
    """
    Parameters:
        xyxy (npt.NDArray[np.float64]): An array of shape `(n, 4)` containing the
            bounding boxes coordinates in format `[x1, y1, x2, y2]`
        offset (np.array): An array of shape `(2,)` containing offset values in format
            is `[dx, dy]`.

    Returns:
        npt.NDArray[np.float64]: Repositioned bounding boxes.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 10, 20, 20],
            [30, 30, 40, 40]
        ])
        offset = np.array([5, 5])

        sv.move_boxes(xyxy=xyxy, offset=offset)
        # array([
        #    [15, 15, 25, 25],
        #    [35, 35, 45, 45]
        # ])
        ```
    """
    return xyxy + np.hstack([offset, offset])


def move_oriented_boxes(
    xyxyxyxy: npt.NDArray[np.float64], offset: npt.NDArray[np.int32]
) -> npt.NDArray[np.float64]:
    """
    Parameters:
    xyxyxyxy (npt.NDArray[np.float64]): An array of shape `(n, 4, 2)` containing the
    oriented bounding boxes coordinates in format
    `[[x1, y1], [x2, y2], [x3, y3], [x3, y3]]`
    offset (np.array): An array of shape `(2,)` containing offset values in format
        is `[dx, dy]`.

    Returns:
    npt.NDArray[np.float64]: Repositioned bounding boxes.

    Examples:
    ```python
    import numpy as np
    import supervision as sv

    xyxyxyxy = np.array([
        [
            [20, 10],
            [10, 20],
            [20, 30],
            [30, 20]
        ],
        [
            [30 ,30],
            [20, 40],
            [30, 50],
            [40, 40]
        ]
    ])
    offset = np.array([5, 5])

    sv.move_oriented_boxes(xyxy=xyxy, offset=offset)
    # array([
    #     [
    #         [25, 15],
    #         [15, 25],
    #         [25, 35],
    #         [35, 25]
    #     ],
    #     [
    #         [35, 35],
    #         [25, 45],
    #         [35, 55],
    #         [45, 45]
    #     ]
    # ])
    ```
    """
    return xyxyxyxy + offset


def move_masks(
    masks: npt.NDArray[np.bool_],
    offset: npt.NDArray[np.int32],
    resolution_wh: Tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """
    Offset the masks in an array by the specified (x, y) amount.

    Args:
        masks (npt.NDArray[np.bool_]): A 3D array of binary masks corresponding to the
            predictions. Shape: `(N, H, W)`, where N is the number of predictions, and
            H, W are the dimensions of each mask.
        offset (npt.NDArray[np.int32]): An array of shape `(2,)` containing int values
            `[dx, dy]`. Supports both positive and negative values for bidirectional
            movement.
        resolution_wh (Tuple[int, int]): The width and height of the desired mask
            resolution.

    Returns:
        (npt.NDArray[np.bool_]) repositioned masks, optionally padded to the specified
            shape.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        mask = np.array([[[False, False, False, False],
                         [False, True,  True,  False],
                         [False, True,  True,  False],
                         [False, False, False, False]]], dtype=bool)

        offset = np.array([1, 1])
        sv.move_masks(mask, offset, resolution_wh=(4, 4))
        # array([[[False, False, False, False],
        #         [False, False, False, False],
        #         [False, False,  True,  True],
        #         [False, False,  True,  True]]], dtype=bool)

        offset = np.array([-2, 2])
        sv.move_masks(mask, offset, resolution_wh=(4, 4))
        # array([[[False, False, False, False],
        #         [False, False, False, False],
        #         [False, False, False, False],
        #         [True,  False, False, False]]], dtype=bool)
        ```
    """
    mask_array = np.full((masks.shape[0], resolution_wh[1], resolution_wh[0]), False)

    if offset[0] < 0:
        source_x_start = -offset[0]
        source_x_end = min(masks.shape[2], resolution_wh[0] - offset[0])
        destination_x_start = 0
        destination_x_end = min(resolution_wh[0], masks.shape[2] + offset[0])
    else:
        source_x_start = 0
        source_x_end = min(masks.shape[2], resolution_wh[0] - offset[0])
        destination_x_start = offset[0]
        destination_x_end = offset[0] + source_x_end - source_x_start

    if offset[1] < 0:
        source_y_start = -offset[1]
        source_y_end = min(masks.shape[1], resolution_wh[1] - offset[1])
        destination_y_start = 0
        destination_y_end = min(resolution_wh[1], masks.shape[1] + offset[1])
    else:
        source_y_start = 0
        source_y_end = min(masks.shape[1], resolution_wh[1] - offset[1])
        destination_y_start = offset[1]
        destination_y_end = offset[1] + source_y_end - source_y_start

    if source_x_end > source_x_start and source_y_end > source_y_start:
        mask_array[
            :,
            destination_y_start:destination_y_end,
            destination_x_start:destination_x_end,
        ] = masks[:, source_y_start:source_y_end, source_x_start:source_x_end]

    return mask_array


def scale_boxes(
    xyxy: npt.NDArray[np.float64], factor: float
) -> npt.NDArray[np.float64]:
    """
    Scale the dimensions of bounding boxes.

    Parameters:
        xyxy (npt.NDArray[np.float64]): An array of shape `(n, 4)` containing the
            bounding boxes coordinates in format `[x1, y1, x2, y2]`
        factor (float): A float value representing the factor by which the box
            dimensions are scaled. A factor greater than 1 enlarges the boxes, while a
            factor less than 1 shrinks them.

    Returns:
        npt.NDArray[np.float64]: Scaled bounding boxes.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 10, 20, 20],
            [30, 30, 40, 40]
        ])

        sv.scale_boxes(xyxy=xyxy, factor=1.5)
        # array([
        #    [ 7.5,  7.5, 22.5, 22.5],
        #    [27.5, 27.5, 42.5, 42.5]
        # ])
        ```
    """
    centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    new_sizes = (xyxy[:, 2:] - xyxy[:, :2]) * factor
    return np.concatenate((centers - new_sizes / 2, centers + new_sizes / 2), axis=1)


def calculate_masks_centroids(masks: np.ndarray) -> np.ndarray:
    """
    Calculate the centroids of binary masks in a tensor.

    Parameters:
        masks (np.ndarray): A 3D NumPy array of shape (num_masks, height, width).
            Each 2D array in the tensor represents a binary mask.

    Returns:
        A 2D NumPy array of shape (num_masks, 2), where each row contains the x and y
            coordinates (in that order) of the centroid of the corresponding mask.
    """
    num_masks, height, width = masks.shape
    total_pixels = masks.sum(axis=(1, 2))

    # offset for 1-based indexing
    vertical_indices, horizontal_indices = np.indices((height, width)) + 0.5
    # avoid division by zero for empty masks
    total_pixels[total_pixels == 0] = 1

    def sum_over_mask(indices: np.ndarray, axis: tuple) -> np.ndarray:
        return np.tensordot(masks, indices, axes=axis)

    aggregation_axis = ([1, 2], [0, 1])
    centroid_x = sum_over_mask(horizontal_indices, aggregation_axis) / total_pixels
    centroid_y = sum_over_mask(vertical_indices, aggregation_axis) / total_pixels

    return np.column_stack((centroid_x, centroid_y)).astype(int)


def is_data_equal(data_a: Dict[str, np.ndarray], data_b: Dict[str, np.ndarray]) -> bool:
    """
    Compares the data payloads of two Detections instances.

    Args:
        data_a, data_b: The data payloads of the instances.

    Returns:
        True if the data payloads are equal, False otherwise.
    """
    return set(data_a.keys()) == set(data_b.keys()) and all(
        np.array_equal(data_a[key], data_b[key]) for key in data_a
    )


def is_metadata_equal(metadata_a: Dict[str, Any], metadata_b: Dict[str, Any]) -> bool:
    """
    Compares the metadata payloads of two Detections instances.

    Args:
        metadata_a, metadata_b: The metadata payloads of the instances.

    Returns:
        True if the metadata payloads are equal, False otherwise.
    """
    return set(metadata_a.keys()) == set(metadata_b.keys()) and all(
        np.array_equal(metadata_a[key], metadata_b[key])
        if (
            isinstance(metadata_a[key], np.ndarray)
            and isinstance(metadata_b[key], np.ndarray)
        )
        else metadata_a[key] == metadata_b[key]
        for key in metadata_a
    )


def merge_data(
    data_list: List[Dict[str, Union[npt.NDArray[np.generic], List]]],
) -> Dict[str, Union[npt.NDArray[np.generic], List]]:
    """
    Merges the data payloads of a list of Detections instances.

    Warning: Assumes that empty detections were filtered-out before passing data to
    this function.

    Args:
        data_list: The data payloads of the Detections instances. Each data payload
            is a dictionary with the same keys, and the values are either lists or
            npt.NDArray[np.generic].

    Returns:
        A single data payload containing the merged data, preserving the original data
            types (list or npt.NDArray[np.generic]).

    Raises:
        ValueError: If data values within a single object have different lengths or if
            dictionaries have different keys.
    """
    if not data_list:
        return {}

    all_keys_sets = [set(data.keys()) for data in data_list]
    if not all(keys_set == all_keys_sets[0] for keys_set in all_keys_sets):
        raise ValueError("All data dictionaries must have the same keys to merge.")

    for data in data_list:
        lengths = [len(value) for value in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                "All data values within a single object must have equal length."
            )

    merged_data = {key: [] for key in all_keys_sets[0]}
    for data in data_list:
        for key in data:
            merged_data[key].append(data[key])

    for key in merged_data:
        if all(isinstance(item, list) for item in merged_data[key]):
            merged_data[key] = list(chain.from_iterable(merged_data[key]))
        elif all(isinstance(item, np.ndarray) for item in merged_data[key]):
            ndim = merged_data[key][0].ndim
            if ndim == 1:
                merged_data[key] = np.hstack(merged_data[key])
            elif ndim > 1:
                merged_data[key] = np.vstack(merged_data[key])
            else:
                raise ValueError(f"Unexpected array dimension for key '{key}'.")
        else:
            raise ValueError(
                f"Inconsistent data types for key '{key}'. Only np.ndarray and list "
                f"types are allowed."
            )

    return merged_data


def merge_metadata(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge metadata from a list of metadata dictionaries.

    This function combines the metadata dictionaries. If a key appears in more than one
    dictionary, the values must be identical for the merge to succeed.

    Warning: Assumes that empty detections were filtered-out before passing metadata to
    this function.

    Args:
        metadata_list (List[Dict[str, Any]]): A list of metadata dictionaries to merge.

    Returns:
        Dict[str, Any]: A single merged metadata dictionary.

    Raises:
        ValueError: If there are conflicting values for the same key or if
        dictionaries have different keys.
    """
    if not metadata_list:
        return {}

    all_keys_sets = [set(metadata.keys()) for metadata in metadata_list]
    if not all(keys_set == all_keys_sets[0] for keys_set in all_keys_sets):
        raise ValueError("All metadata dictionaries must have the same keys to merge.")

    merged_metadata: Dict[str, Any] = {}
    for metadata in metadata_list:
        for key, value in metadata.items():
            if key not in merged_metadata:
                merged_metadata[key] = value
                continue

            other_value = merged_metadata[key]
            if isinstance(value, np.ndarray) and isinstance(other_value, np.ndarray):
                if not np.array_equal(merged_metadata[key], value):
                    raise ValueError(
                        f"Conflicting metadata for key: '{key}': "
                        "{type(value)}, {type(other_value)}."
                    )
            elif isinstance(value, np.ndarray) or isinstance(other_value, np.ndarray):
                # Since [] == np.array([]).
                raise ValueError(
                    f"Conflicting metadata for key: '{key}': "
                    "{type(value)}, {type(other_value)}."
                )
            else:
                print("hm")
                if merged_metadata[key] != value:
                    raise ValueError(f"Conflicting metadata for key: '{key}'.")

    return merged_metadata


def get_data_item(
    data: Dict[str, Union[np.ndarray, List]],
    index: Union[int, slice, List[int], np.ndarray],
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Retrieve a subset of the data dictionary based on the given index.

    Args:
        data: The data dictionary of the Detections object.
        index: The index or indices specifying the subset to retrieve.

    Returns:
        A subset of the data dictionary corresponding to the specified index.
    """
    subset_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            subset_data[key] = value[index]
        elif isinstance(value, list):
            if isinstance(index, slice):
                subset_data[key] = value[index]
            elif isinstance(index, list):
                subset_data[key] = [value[i] for i in index]
            elif isinstance(index, np.ndarray):
                if index.dtype == bool:
                    subset_data[key] = [
                        value[i] for i, index_value in enumerate(index) if index_value
                    ]
                else:
                    subset_data[key] = [value[i] for i in index]
            elif isinstance(index, int):
                subset_data[key] = [value[index]]
            else:
                raise TypeError(f"Unsupported index type: {type(index)}")
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    return subset_data


def contains_holes(mask: npt.NDArray[np.bool_]) -> bool:
    """
    Checks if the binary mask contains holes (background pixels fully enclosed by
    foreground pixels).

    Args:
        mask (npt.NDArray[np.bool_]): 2D binary mask where `True` indicates foreground
            object and `False` indicates background.

    Returns:
        True if holes are detected, False otherwise.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]).astype(bool)

        sv.contains_holes(mask=mask)
        # True

        mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]).astype(bool)

        sv.contains_holes(mask=mask)
        # False
        ```

    ![contains_holes](https://media.roboflow.com/supervision-docs/contains-holes.png){ align=center width="800" }
    """  # noqa E501 // docs
    mask_uint8 = mask.astype(np.uint8)
    _, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        parent_contour_index = 3
        for h in hierarchy[0]:
            if h[parent_contour_index] != -1:
                return True
    return False


def contains_multiple_segments(
    mask: npt.NDArray[np.bool_], connectivity: int = 4
) -> bool:
    """
    Checks if the binary mask contains multiple unconnected foreground segments.

    Args:
        mask (npt.NDArray[np.bool_]): 2D binary mask where `True` indicates foreground
            object and `False` indicates background.
        connectivity (int) : Default: 4 is 4-way connectivity, which means that
            foreground pixels are the part of the same segment/component
            if their edges touch.
            Alternatively: 8 for 8-way connectivity, when foreground pixels are
            connected by their edges or corners touch.

    Returns:
        True when the mask contains multiple not connected components, False otherwise.

    Raises:
        ValueError: If connectivity(int) parameter value is not 4 or 8.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        mask = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0]
        ]).astype(bool)

        sv.contains_multiple_segments(mask=mask, connectivity=4)
        # True

        mask = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0]
        ]).astype(bool)

        sv.contains_multiple_segments(mask=mask, connectivity=4)
        # False
        ```

    ![contains_multiple_segments](https://media.roboflow.com/supervision-docs/contains-multiple-segments.png){ align=center width="800" }
    """  # noqa E501 // docs
    if connectivity != 4 and connectivity != 8:
        raise ValueError(
            "Incorrect connectivity value. Possible connectivity values: 4 or 8."
        )
    mask_uint8 = mask.astype(np.uint8)
    labels = np.zeros_like(mask_uint8, dtype=np.int32)
    number_of_labels, _ = cv2.connectedComponents(
        mask_uint8, labels, connectivity=connectivity
    )
    return number_of_labels > 2


def cross_product(anchors: np.ndarray, vector: Vector) -> np.ndarray:
    """
    Get array of cross products of each anchor with a vector.
    Args:
        anchors: Array of anchors of shape (number of anchors, detections, 2)
        vector: Vector to calculate cross product with

    Returns:
        Array of cross products of shape (number of anchors, detections)
    """
    vector_at_zero = np.array(
        [vector.end.x - vector.start.x, vector.end.y - vector.start.y]
    )
    vector_start = np.array([vector.start.x, vector.start.y])
    return np.cross(vector_at_zero, anchors - vector_start)


def spread_out_boxes(
    xyxy: np.ndarray,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Spread out boxes that overlap with each other.

    Args:
        xyxy: Numpy array of shape (N, 4) where N is the number of boxes.
        max_iterations: Maximum number of iterations to run the algorithm for.
    """
    if len(xyxy) == 0:
        return xyxy

    xyxy_padded = pad_boxes(xyxy, px=1)
    for _ in range(max_iterations):
        # NxN
        iou = box_iou_batch(xyxy_padded, xyxy_padded)
        np.fill_diagonal(iou, 0)
        if np.all(iou == 0):
            break

        overlap_mask = iou > 0

        # Nx2
        centers = (xyxy_padded[:, :2] + xyxy_padded[:, 2:]) / 2

        # NxNx2
        delta_centers = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        delta_centers *= overlap_mask[:, :, np.newaxis]

        # Nx2
        delta_sum = np.sum(delta_centers, axis=1)
        delta_magnitude = np.linalg.norm(delta_sum, axis=1, keepdims=True)
        direction_vectors = np.divide(
            delta_sum,
            delta_magnitude,
            out=np.zeros_like(delta_sum),
            where=delta_magnitude != 0,
        )

        force_vectors = np.sum(iou, axis=1)
        force_vectors = force_vectors[:, np.newaxis] * direction_vectors

        force_vectors *= 10
        force_vectors[(force_vectors > 0) & (force_vectors < 2)] = 2
        force_vectors[(force_vectors < 0) & (force_vectors > -2)] = -2

        force_vectors = force_vectors.astype(int)

        xyxy_padded[:, [0, 1]] += force_vectors
        xyxy_padded[:, [2, 3]] += force_vectors

    return pad_boxes(xyxy_padded, px=-1)


def _jaccard(box_a: List[float], box_b: List[float], is_crowd: bool) -> float:
    """
    Calculate the Jaccard index (intersection over union) between two bounding boxes.
    If a gt object is marked as "iscrowd", a dt is allowed to match any subregion
    of the gt. Choosing gt' in the crowd gt that best matches the dt can be done using
    gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
    iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)

    Args:
        box_a (List[float]): Box coordinates in the format [x, y, width, height].
        box_b (List[float]): Box coordinates in the format [x, y, width, height].
        iscrowd (bool): Flag indicating if the second box is a crowd region or not.

    Returns:
        float: Jaccard index between the two bounding boxes.
    """
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
    boxes_true: List[List[float]],
    boxes_detection: List[List[float]],
    is_crowd: List[bool],
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
