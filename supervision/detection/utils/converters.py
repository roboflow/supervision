import cv2
import numpy as np

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


def polygon_to_mask(polygon: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
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


def xyxy_to_mask(boxes: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
    """
    Converts a 2D `np.ndarray` of bounding boxes into a 3D `np.ndarray` of bool masks.

    Parameters:
        boxes (np.ndarray): A 2D `np.ndarray` of shape `(N, 4)`
            containing bounding boxes `(x_min, y_min, x_max, y_max)`
        resolution_wh (Tuple[int, int]): A tuple `(width, height)` specifying
            the resolution of the output masks

    Returns:
        np.ndarray: A 3D `np.ndarray` of shape `(N, height, width)`
            containing 2D bool masks for each bounding box
    """
    width, height = resolution_wh
    n = boxes.shape[0]
    masks = np.zeros((n, height, width), dtype=bool)

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(width - 1, int(x_max))
        y_max = min(height - 1, int(y_max))

        if x_max >= x_min and y_max >= y_min:
            masks[i, y_min:y_max + 1, x_min:x_max + 1] = True

    return masks


def mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
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
