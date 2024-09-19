import numpy as np
import pytest

import supervision as sv


@pytest.mark.parametrize(
    "detections, expected_results",
    [
        (
            [
                sv.Detections(
                    xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                    class_id=np.array([1, 1]),
                    confidence=np.array([1, 1]),
                ),
                sv.Detections(
                    xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                    class_id=np.array([1, 1]),
                    confidence=np.array([1, 1]),
                ),
            ],
            sv.Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                class_id=np.array([1, 1]),
                confidence=np.array([1, 1]),
                tracker_id=np.array([1, 2]),
            ),
        ),
    ],
)
def test_byte_tracker(
    detections: list[sv.Detections],
    expected_results: sv.Detections,
) -> None:
    byte_tracker = sv.ByteTrack()
    tracked_detections = [byte_tracker.update_with_detections(d) for d in detections]
    assert tracked_detections[-1] == expected_results
