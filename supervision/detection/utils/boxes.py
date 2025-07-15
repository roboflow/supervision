from __future__ import annotations

import numpy as np
import numpy.typing as npt

from supervision.detection.utils.iou_and_nms import box_iou_batch


def clip_boxes(xyxy: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
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


def pad_boxes(xyxy: np.ndarray, px: int, py: int | None = None) -> np.ndarray:
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


def denormalize_boxes(
    normalized_xyxy: np.ndarray,
    resolution_wh: tuple[int, int],
    normalization_factor: float = 1.0,
) -> np.ndarray:
    """
    Converts normalized bounding box coordinates to absolute pixel values.

    Args:
        normalized_xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each row
            contains normalized coordinates in the format `(x_min, y_min, x_max, y_max)`,
            with values between 0 and `normalization_factor`.
        resolution_wh (Tuple[int, int]): A tuple `(width, height)` representing the
            target image resolution.
        normalization_factor (float, optional): The normalization range of the input
            coordinates. Defaults to 1.0.

    Returns:
        np.ndarray: An array of shape `(N, 4)` with absolute coordinates in
            `(x_min, y_min, x_max, y_max)` format.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        # Default normalization (0-1)
        normalized_xyxy = np.array([
            [0.1, 0.2, 0.5, 0.6],
            [0.3, 0.4, 0.7, 0.8]
        ])
        resolution_wh = (100, 200)
        sv.denormalize_boxes(normalized_xyxy, resolution_wh)
        # array([
        #     [ 10.,  40.,  50., 120.],
        #     [ 30.,  80.,  70., 160.]
        # ])

        # Custom normalization (0-100)
        normalized_xyxy = np.array([
            [10., 20., 50., 60.],
            [30., 40., 70., 80.]
        ])
        sv.denormalize_boxes(normalized_xyxy, resolution_wh, normalization_factor=100.0)
        # array([
        #     [ 10.,  40.,  50., 120.],
        #     [ 30.,  80.,  70., 160.]
        # ])
        ```
    """  # noqa E501 // docs
    width, height = resolution_wh
    result = normalized_xyxy.copy()

    result[[0, 2]] = (result[[0, 2]] * width) / normalization_factor
    result[[1, 3]] = (result[[1, 3]] * height) / normalization_factor

    return result


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
