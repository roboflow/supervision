from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.keypoint.core import KeyPoints


def mock_detections(
    xyxy: npt.NDArray[np.float32],
    mask: Optional[list[np.ndarray]] = None,
    confidence: Optional[list[float]] = None,
    class_id: Optional[list[int]] = None,
    tracker_id: Optional[list[int]] = None,
    data: Optional[dict[str, list[Any]]] = None,
) -> Detections:
    def convert_data(data: dict[str, list[Any]]):
        return {k: np.array(v) for k, v in data.items()}

    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=(mask if mask is None else np.array(mask, dtype=bool)),
        confidence=(
            confidence if confidence is None else np.array(confidence, dtype=np.float32)
        ),
        class_id=(class_id if class_id is None else np.array(class_id, dtype=int)),
        tracker_id=(
            tracker_id if tracker_id is None else np.array(tracker_id, dtype=int)
        ),
        data=convert_data(data) if data else {},
    )


def mock_keypoints(
    xy: npt.NDArray[np.float32],
    confidence: Optional[list[float]] = None,
    class_id: Optional[list[int]] = None,
    data: Optional[dict[str, list[Any]]] = None,
) -> KeyPoints:
    def convert_data(data: dict[str, list[Any]]):
        return {k: np.array(v) for k, v in data.items()}

    return KeyPoints(
        xy=np.array(xy, dtype=np.float32),
        confidence=(
            confidence if confidence is None else np.array(confidence, dtype=np.float32)
        ),
        class_id=(class_id if class_id is None else np.array(class_id, dtype=int)),
        data=convert_data(data) if data else {},
    )


def assert_almost_equal(actual, expected, tolerance=1e-5):
    assert abs(actual - expected) < tolerance, f"Expected {expected}, but got {actual}."
