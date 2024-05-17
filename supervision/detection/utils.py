from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from supervision.config import CLASS_NAME_DATA_FIELD

MIN_POLYGON_POINT_COUNT = 3


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
    width, height = resolution_wh
    mask = np.zeros((height, width))

    cv2.fillPoly(mask, [polygon], color=1)
    return mask


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

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
    return area_inter / (area_true[:, None] + area_detection - area_inter)


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
        memory_limit (int, optional): memory limit in MB, default is 1024 * 5 MB (5GB).

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
    """
    if py is None:
        py = px

    result = xyxy.copy()
    result[:, [0, 1]] -= [px, py]
    result[:, [2, 3]] += [px, py]

    return result


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy


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
    bboxes = np.zeros((n, 4), dtype=int)

    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)

        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            bboxes[i, :] = [x_min, y_min, x_max, y_max]

    return bboxes


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


def move_boxes(xyxy: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Parameters:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes
            coordinates in format `[x1, y1, x2, y2]`
        offset (np.array): An array of shape `(2,)` containing offset values in format
            is `[dx, dy]`.

    Returns:
        np.ndarray: Repositioned bounding boxes.

    Example:
        ```python
        import numpy as np
        import supervision as sv

        boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
        offset = np.array([5, 5])
        moved_box = sv.move_boxes(boxes, offset)
        print(moved_box)
        # np.array([
        #    [15, 15, 25, 25],
        #     [35, 35, 45, 45]
        # ])
        ```
    """
    return xyxy + np.hstack([offset, offset])


def move_masks(
    masks: np.ndarray,
    offset: np.ndarray,
    resolution_wh: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Offset the masks in an array by the specified (x, y) amount.

    Args:
        masks (np.ndarray): A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        offset (np.ndarray): An array of shape `(2,)` containing non-negative int values
            `[dx, dy]`.
        resolution_wh (Tuple[int, int]): The width and height of the desired mask
            resolution.

    Returns:
        (np.ndarray) repositioned masks, optionally padded to the specified shape.
    """

    if offset[0] < 0 or offset[1] < 0:
        raise ValueError(f"Offset values must be non-negative integers. Got: {offset}")

    mask_array = np.full((masks.shape[0], resolution_wh[1], resolution_wh[0]), False)
    mask_array[
        :,
        offset[1] : masks.shape[1] + offset[1],
        offset[0] : masks.shape[2] + offset[0],
    ] = masks

    return mask_array


def scale_boxes(xyxy: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale the dimensions of bounding boxes.

    Parameters:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes
            coordinates in format `[x1, y1, x2, y2]`
        factor (float): A float value representing the factor by which the box
            dimensions are scaled. A factor greater than 1 enlarges the boxes, while a
            factor less than 1 shrinks them.

    Returns:
        np.ndarray: Scaled bounding boxes.

    Example:
        ```python
        import numpy as np
        import supervision as sv

        boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
        factor = 1.5
        scaled_bb = sv.scale_boxes(boxes, factor)
        print(scaled_bb)
        # np.array([
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


def merge_data(
    data_list: List[Dict[str, Union[np.ndarray, List]]],
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Merges the data payloads of a list of Detections instances.

    Args:
        data_list: The data payloads of the Detections instances. Each data payload
            is a dictionary with the same keys, and the values are either lists or
            np.ndarray.

    Returns:
        A single data payload containing the merged data, preserving the original data
            types (list or np.ndarray).

    Raises:
        ValueError: If data values within a single object have different lengths or if
            dictionaries have different keys.
    """
    if not data_list:
        return {}

    for data in data_list:
        lengths = [len(value) for value in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                "All data values within a single object must have equal length."
            )

    keys_by_data = [set(data.keys()) for data in data_list]
    keys_by_data = [keys for keys in keys_by_data if len(keys) > 0]
    if not keys_by_data:
        return {}

    common_keys = set.intersection(*keys_by_data)
    all_keys = set.union(*keys_by_data)
    if common_keys != all_keys:
        raise ValueError(
            f"All sv.Detections.data dictionaries must have the same keys. Common "
            f"keys: {common_keys}, but some dictionaries have additional keys: "
            f"{all_keys.difference(common_keys)}."
        )

    merged_data = {key: [] for key in all_keys}
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
