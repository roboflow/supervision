from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from supervision.detection.compact_mask import CompactMask

_MASK_DIMENSIONS_ERROR = (
    "Mask must be a 2D array or a single-mask batch with shape (1, H, W)."
)
_COMPACT_MASK_LENGTH_ERROR = "CompactMask inputs must contain exactly one mask."


def _coerce_single_mask(
    mask: npt.NDArray[Any] | CompactMask, mask_name: str
) -> npt.NDArray[np.bool_]:
    if isinstance(mask, CompactMask):
        if len(mask) != 1:
            raise ValueError(f"{mask_name} {_COMPACT_MASK_LENGTH_ERROR}")
        return cast(
            npt.NDArray[np.bool_], np.asarray(mask[0], dtype=bool)
        )

    mask_array = np.asarray(mask)
    if mask_array.ndim == 2:
        return cast(npt.NDArray[np.bool_], mask_array.astype(bool, copy=False))
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        return cast(
            npt.NDArray[np.bool_], mask_array[0].astype(bool, copy=False)
        )
    raise ValueError(f"{mask_name} {_MASK_DIMENSIONS_ERROR}")


def _validate_mask_pair(
    prediction: npt.NDArray[Any] | CompactMask,
    target: npt.NDArray[Any] | CompactMask,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    prediction_mask = _coerce_single_mask(prediction, "prediction")
    target_mask = _coerce_single_mask(target, "target")

    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "prediction and target must have the same shape. "
            f"Got {prediction_mask.shape} and {target_mask.shape}."
        )
    return prediction_mask, target_mask


def _validate_tolerance(tolerance: int) -> int:
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(
        tolerance, (int, np.integer)
    ):
        raise ValueError("tolerance must be a non-negative integer.")
    if tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer.")
    return int(tolerance)


def _resolve_empty_mask_score(
    prediction_mask: npt.NDArray[np.bool_], target_mask: npt.NDArray[np.bool_]
) -> float | None:
    prediction_empty = not prediction_mask.any()
    target_empty = not target_mask.any()

    if prediction_empty and target_empty:
        return 1.0
    if prediction_empty or target_empty:
        return 0.0
    return None


def _extract_boundary(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    mask_uint8 = mask.astype(np.uint8, copy=False)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, borderType=cv2.BORDER_CONSTANT)
    return cast(
        npt.NDArray[np.bool_],
        np.logical_and(mask, np.logical_not(eroded_mask.astype(bool, copy=False))),
    )


def _build_tolerance_kernel(tolerance: int) -> npt.NDArray[np.uint8]:
    kernel_size = 2 * tolerance + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    ).astype(np.uint8, copy=False)


def _expand_boundary(
    boundary_mask: npt.NDArray[np.bool_], tolerance: int
) -> npt.NDArray[np.bool_]:
    if tolerance == 0:
        return boundary_mask
    kernel = _build_tolerance_kernel(tolerance)
    dilated_boundary = cv2.dilate(
        boundary_mask.astype(np.uint8, copy=False),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return cast(
        npt.NDArray[np.bool_], np.asarray(dilated_boundary, dtype=bool)
    )


def mask_iou(
    prediction: npt.NDArray[Any] | CompactMask,
    target: npt.NDArray[Any] | CompactMask,
) -> float:
    """
    Compute Intersection over Union (IoU) for a single pair of segmentation masks.

    Args:
        prediction: Predicted binary mask. Accepts a 2D array, a
            single-mask batch with shape `(1, H, W)`, or a
            :class:`~supervision.detection.compact_mask.CompactMask`
            containing exactly one mask.
        target: Target binary mask. Accepts the same input forms as
            `prediction`.

    Returns:
        IoU score as a Python float.

    Raises:
        ValueError: If shapes differ or the inputs do not represent exactly one
            mask each.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        >>> target = np.array([[1, 0], [0, 0]], dtype=bool)
        >>> sv.mask_iou(prediction, target)
        0.5

        ```
    """
    prediction_mask, target_mask = _validate_mask_pair(prediction, target)
    empty_mask_score = _resolve_empty_mask_score(prediction_mask, target_mask)
    if empty_mask_score is not None:
        return empty_mask_score

    intersection = np.logical_and(prediction_mask, target_mask).sum()
    union = np.logical_or(prediction_mask, target_mask).sum()
    return float(intersection / union)


def dice_coefficient(
    prediction: npt.NDArray[Any] | CompactMask,
    target: npt.NDArray[Any] | CompactMask,
) -> float:
    """
    Compute Dice coefficient for a single pair of segmentation masks.

    Args:
        prediction: Predicted binary mask. Accepts a 2D array, a
            single-mask batch with shape `(1, H, W)`, or a
            :class:`~supervision.detection.compact_mask.CompactMask`
            containing exactly one mask.
        target: Target binary mask. Accepts the same input forms as
            `prediction`.

    Returns:
        Dice coefficient as a Python float.

    Raises:
        ValueError: If shapes differ or the inputs do not represent exactly one
            mask each.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        >>> target = np.array([[1, 0], [0, 0]], dtype=bool)
        >>> round(sv.dice_coefficient(prediction, target), 2)
        0.67

        ```
    """
    prediction_mask, target_mask = _validate_mask_pair(prediction, target)
    empty_mask_score = _resolve_empty_mask_score(prediction_mask, target_mask)
    if empty_mask_score is not None:
        return empty_mask_score

    intersection = np.logical_and(prediction_mask, target_mask).sum()
    denominator = prediction_mask.sum() + target_mask.sum()
    return float((2 * intersection) / denominator)


def boundary_iou(
    prediction: npt.NDArray[Any] | CompactMask,
    target: npt.NDArray[Any] | CompactMask,
    tolerance: int = 2,
) -> float:
    """
    Compute boundary IoU for a single pair of segmentation masks.

    Boundary IoU dilates the foreground contours of both masks by `tolerance`
    pixels before measuring IoU. This makes the score more forgiving to small
    contour shifts than standard region IoU.

    Args:
        prediction: Predicted binary mask. Accepts a 2D array, a
            single-mask batch with shape `(1, H, W)`, or a
            :class:`~supervision.detection.compact_mask.CompactMask`
            containing exactly one mask.
        target: Target binary mask. Accepts the same input forms as
            `prediction`.
        tolerance: Integer pixel distance used to dilate both
            boundaries before comparison. `0` requires exact boundary-pixel
            agreement.

    Returns:
        Boundary IoU score as a Python float.

    Raises:
        ValueError: If shapes differ, the inputs do not represent exactly one
            mask each, or `tolerance` is invalid.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> prediction = np.zeros((5, 5), dtype=bool)
        >>> target = np.zeros((5, 5), dtype=bool)
        >>> prediction[1:4, 1:4] = True
        >>> target[1:4, 2:5] = True
        >>> round(sv.boundary_iou(prediction, target, tolerance=1), 2)
        0.7

        ```
    """
    tolerance = _validate_tolerance(tolerance)
    prediction_mask, target_mask = _validate_mask_pair(prediction, target)
    empty_mask_score = _resolve_empty_mask_score(prediction_mask, target_mask)
    if empty_mask_score is not None:
        return empty_mask_score

    prediction_boundary = _expand_boundary(
        _extract_boundary(prediction_mask), tolerance
    )
    target_boundary = _expand_boundary(_extract_boundary(target_mask), tolerance)

    intersection = np.logical_and(prediction_boundary, target_boundary).sum()
    union = np.logical_or(prediction_boundary, target_boundary).sum()
    return float(intersection / union) if union > 0 else 0.0


def boundary_f_score(
    prediction: npt.NDArray[Any] | CompactMask,
    target: npt.NDArray[Any] | CompactMask,
    tolerance: int = 2,
) -> float:
    """
    Compute boundary F-score for a single pair of segmentation masks.

    Boundary F-score measures contour agreement by matching predicted boundary
    pixels to target boundary pixels within `tolerance` pixels, then combining
    boundary precision and boundary recall into a single score.

    Args:
        prediction: Predicted binary mask. Accepts a 2D array, a
            single-mask batch with shape `(1, H, W)`, or a
            :class:`~supervision.detection.compact_mask.CompactMask`
            containing exactly one mask.
        target: Target binary mask. Accepts the same input forms as
            `prediction`.
        tolerance: Integer pixel distance used for boundary
            matching. `0` requires exact boundary-pixel agreement.

    Returns:
        Boundary F-score as a Python float.

    Raises:
        ValueError: If shapes differ, the inputs do not represent exactly one
            mask each, or `tolerance` is invalid.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> prediction = np.zeros((5, 5), dtype=bool)
        >>> target = np.zeros((5, 5), dtype=bool)
        >>> prediction[1:4, 1:4] = True
        >>> target[1:4, 2:5] = True
        >>> round(sv.boundary_f_score(prediction, target, tolerance=1), 2)
        1.0

        ```
    """
    tolerance = _validate_tolerance(tolerance)
    prediction_mask, target_mask = _validate_mask_pair(prediction, target)
    empty_mask_score = _resolve_empty_mask_score(prediction_mask, target_mask)
    if empty_mask_score is not None:
        return empty_mask_score

    prediction_boundary = _extract_boundary(prediction_mask)
    target_boundary = _extract_boundary(target_mask)

    expanded_prediction_boundary = _expand_boundary(prediction_boundary, tolerance)
    expanded_target_boundary = _expand_boundary(target_boundary, tolerance)

    matched_prediction = np.logical_and(
        prediction_boundary, expanded_target_boundary
    ).sum()
    matched_target = np.logical_and(target_boundary, expanded_prediction_boundary).sum()

    prediction_boundary_area = prediction_boundary.sum()
    target_boundary_area = target_boundary.sum()

    precision = (
        float(matched_prediction / prediction_boundary_area)
        if prediction_boundary_area > 0
        else 0.0
    )
    recall = (
        float(matched_target / target_boundary_area)
        if target_boundary_area > 0
        else 0.0
    )

    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
