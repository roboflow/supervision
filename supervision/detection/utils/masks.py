from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def move_masks(
    masks: npt.NDArray[np.bool_],
    offset: npt.NDArray[np.int32],
    resolution_wh: tuple[int, int],
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

    return resized_masks.reshape(masks.shape[0], new_height, new_width)
