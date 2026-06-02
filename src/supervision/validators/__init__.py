from typing import Any

import numpy as np

from supervision.detection.compact_mask import CompactMask
from supervision.utils.internal import warn_deprecated


def validate_xyxy(xyxy: Any) -> None:
    """Validate that xyxy is a 2D np.ndarray with shape (N, 4).

    ```pycon
    >>> validate_xyxy(np.array([[0, 0, 1, 1], [1, 1, 2, 2]]))

    ```
    """
    expected_shape = "(_, 4)"
    actual_shape = str(getattr(xyxy, "shape", None))
    is_valid = isinstance(xyxy, np.ndarray) and xyxy.ndim == 2 and xyxy.shape[1] == 4
    if not is_valid:
        raise ValueError(
            f"xyxy must be a 2D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )


def validate_mask(mask: Any, n: int) -> None:
    if mask is None:
        return

    # Fast path: CompactMask only needs a length check.

    if isinstance(mask, CompactMask):
        if len(mask) != n:
            raise ValueError(f"mask must contain {n} masks, but got {len(mask)}")
        return

    expected_shape = f"({n}, H, W)"
    actual_shape = str(getattr(mask, "shape", None))
    actual_dtype = getattr(mask, "dtype", None)

    is_valid_shape = (
        isinstance(mask, np.ndarray) and len(mask.shape) == 3 and mask.shape[0] == n
    )
    if not is_valid_shape:
        raise ValueError(
            "mask must be a 3D np.ndarray with shape "
            + f"{expected_shape}, but got shape {actual_shape}"
        )
    if not np.issubdtype(actual_dtype, bool):
        warn_deprecated(
            f"A `Detections` object was created with a mask of type {actual_dtype}."
            " Masks of type other than `bool` are deprecated and may produce unexpected"
            " behavior. Starting from `supervision-0.28.0`, passing a mask with"
            " `dtype` different from `bool` to `Detections` will raise a `ValueError`"
            " during validation instead of being accepted with a warning. To migrate,"
            " please ensure your masks are boolean, for example by using"
            " `mask = np.array(..., dtype=bool)` or by converting existing masks with"
            " `mask = mask.astype(bool)` before creating the `Detections` object. If"
            " you did not create the mask manually, please report the issue to the"
            " `supervision` team."
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


def validate_detection_confidence(confidence: Any, n: int) -> None:
    """Validate detection-level confidence: 1D ``np.ndarray`` with shape ``(n,)``."""
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


def validate_keypoint_confidence(keypoint_confidence: Any, n: int, m: int) -> None:
    """Validate per-keypoint confidence: 2D ``np.ndarray`` with shape ``(n, m)``."""
    actual_shape = str(getattr(keypoint_confidence, "shape", None))

    if keypoint_confidence is not None:
        if (
            not isinstance(keypoint_confidence, np.ndarray)
            or keypoint_confidence.ndim != 2
        ):
            raise ValueError(
                f"keypoint_confidence must be a 2D np.ndarray with shape (n, m), but "
                f"got shape {actual_shape}"
            )
        if keypoint_confidence.shape[0] != n:
            raise ValueError(
                f"keypoint_confidence first dimension must be {n}, but got shape {actual_shape}"
            )
        if n > 0 and keypoint_confidence.shape[1] != m:
            raise ValueError(
                f"keypoint_confidence second dimension must be {m}, but "
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


def validate_data(data: dict[str, Any], n: int) -> None:
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
    actual_shape = str(getattr(xy, "shape", None))

    if not isinstance(xy, np.ndarray) or xy.ndim != 3 or xy.shape[2] not in (2, 3):
        raise ValueError(
            f"xy must be a 3D np.ndarray with shape (n, m, 2) or (n, m, 3), "
            f"but got shape {actual_shape}"
        )
    if xy.shape[0] != n:
        raise ValueError(
            f"xy first dimension must be {n}, but got shape {actual_shape}"
        )
    if n > 0 and xy.shape[1] != m:
        raise ValueError(
            f"xy second dimension must be {m}, but got shape {actual_shape}"
        )


def validate_detections_fields(
    xyxy: Any,
    mask: Any,
    class_id: Any,
    confidence: Any,
    tracker_id: Any,
    data: dict[str, Any],
) -> None:
    validate_xyxy(xyxy)
    n = len(xyxy)
    validate_mask(mask, n)
    validate_class_id(class_id, n)
    validate_detection_confidence(confidence, n)
    validate_tracker_id(tracker_id, n)
    validate_data(data, n)


def validate_visible(visible: Any, n: int, m: int) -> None:
    """Validate per-keypoint visibility mask: 2D bool ``np.ndarray`` with shape ``(n, m)``."""
    if visible is None:
        return
    actual_shape = str(getattr(visible, "shape", None))
    if not isinstance(visible, np.ndarray) or visible.ndim != 2:
        raise ValueError(
            f"visible must be a 2D np.ndarray with shape (n, m), but "
            f"got shape {actual_shape}"
        )
    if visible.shape[0] != n:
        raise ValueError(
            f"visible first dimension must be {n}, but got shape {actual_shape}"
        )
    if n > 0 and visible.shape[1] != m:
        raise ValueError(
            f"visible second dimension must be {m}, but got shape {actual_shape}"
        )


def validate_key_points_fields(
    xy: Any,
    class_id: Any,
    keypoint_confidence: Any,
    detection_confidence: Any = None,
    visible: Any = None,
    data: dict[str, Any] | None = None,
) -> None:
    n = len(xy)
    m = len(xy[0]) if len(xy) > 0 else 0
    validate_xy(xy, n, m)
    validate_class_id(class_id, n)
    validate_keypoint_confidence(keypoint_confidence, n, m)
    if detection_confidence is not None:
        validate_detection_confidence(detection_confidence, n)
    validate_visible(visible, n, m)
    if data is not None:
        validate_data(data, n)


def validate_resolution(resolution: Any) -> tuple[int, int]:
    if not (isinstance(resolution, tuple) and len(resolution) == 2):
        raise ValueError(
            f"""
            resolution must be a tuple of two integers, got
            {type(resolution)} with value {resolution}
            """
        )
    w, h = resolution
    if not (isinstance(w, int) and isinstance(h, int)):
        raise ValueError(
            f"""
            Both elements in resolution must be integers.
            Got types ({type(w)}, {type(h)})
            """
        )
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Both dimensions in resolution must be positive. Got ({w}, {h})."
        )
    return w, h
