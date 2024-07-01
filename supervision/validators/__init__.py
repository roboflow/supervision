from typing import Any, Dict

import numpy as np


def validate_xyxy(xyxy: Any) -> None:
    expected_shape = "(_, 4)"
    actual_shape = str(getattr(xyxy, "shape", None))
    is_valid = isinstance(xyxy, np.ndarray) and xyxy.ndim == 2 and xyxy.shape[1] == 4
    if not is_valid:
        raise ValueError(
            f"xyxy must be a 2D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )


def validate_mask(mask: Any, n: int) -> None:
    expected_shape = f"({n}, H, W)"
    actual_shape = str(getattr(mask, "shape", None))
    is_valid = mask is None or (
        isinstance(mask, np.ndarray) and len(mask.shape) == 3 and mask.shape[0] == n
    )
    if not is_valid:
        raise ValueError(
            f"mask must be a 3D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )


def validate_class_id(class_id: Any, n: int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(class_id, "shape", None))
    is_valid = class_id is None or (
        isinstance(class_id, np.ndarray) and class_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"class_id must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_confidence(confidence: Any, n: int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(confidence, "shape", None))
    is_valid = confidence is None or (
        isinstance(confidence, np.ndarray) and confidence.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"confidence must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_keypoint_confidence(confidence: Any, n: int, m: int) -> None:
    expected_shape = f"({n,m})"
    actual_shape = str(getattr(confidence, "shape", None))

    if confidence is not None:
        is_valid = isinstance(confidence, np.ndarray) and confidence.shape == (n, m)
        if not is_valid:
            raise ValueError(
                f"confidence must be a 1D np.ndarray with shape {expected_shape}, but "
                f"got shape {actual_shape}"
            )


def validate_tracker_id(tracker_id: Any, n: int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(tracker_id, "shape", None))
    is_valid = tracker_id is None or (
        isinstance(tracker_id, np.ndarray) and tracker_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"tracker_id must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_data(data: Dict[str, Any], n: int) -> None:
    for key, value in data.items():
        if isinstance(value, list):
            if len(value) != n:
                raise ValueError(f"Length of list for key '{key}' must be {n}")
        elif isinstance(value, np.ndarray):
            if value.ndim == 1 and value.shape[0] != n:
                raise ValueError(f"Shape of np.ndarray for key '{key}' must be ({n},)")
            elif value.ndim > 1 and value.shape[0] != n:
                raise ValueError(
                    f"First dimension of np.ndarray for key '{key}' must have size {n}"
                )
        else:
            raise ValueError(f"Value for key '{key}' must be a list or np.ndarray")


def validate_xy(xy: Any, n: int, m: int) -> None:
    expected_shape = f"({n, m},)"
    actual_shape = str(getattr(xy, "shape", None))

    is_valid = isinstance(xy, np.ndarray) and (
        xy.shape == (n, m, 2) or xy.shape == (n, m, 3)
    )
    if not is_valid:
        raise ValueError(
            f"xy must be a 2D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )


def validate_detections_fields(
    xyxy: Any,
    mask: Any,
    class_id: Any,
    confidence: Any,
    tracker_id: Any,
    data: Dict[str, Any],
) -> None:
    validate_xyxy(xyxy)
    n = len(xyxy)
    validate_mask(mask, n)
    validate_class_id(class_id, n)
    validate_confidence(confidence, n)
    validate_tracker_id(tracker_id, n)
    validate_data(data, n)


def validate_keypoints_fields(
    xy: Any,
    class_id: Any,
    confidence: Any,
    data: Dict[str, Any],
) -> None:
    n = len(xy)
    m = len(xy[0]) if len(xy) > 0 else 0
    validate_xy(xy, n, m)
    validate_class_id(class_id, n)
    validate_keypoint_confidence(confidence, n, m)
    validate_data(data, n)
