import numpy as np
import pytest

import supervision as sv


@pytest.mark.parametrize(
    ("detections", "expected_results"),
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


def test_byte_tracker_does_not_skip_external_ids_for_short_lived_tracks() -> None:
    def detections_from_boxes(boxes: list[list[float]]) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
            confidence=np.ones(len(boxes), dtype=np.float32),
        )

    # A transient false-positive appears and disappears before becoming confirmed.
    # It should not consume an external tracker id.
    frames = [
        detections_from_boxes([[0, 0, 10, 10]]),
        detections_from_boxes([[0, 0, 10, 10], [100, 100, 110, 110]]),
        detections_from_boxes([[0, 0, 10, 10]]),
        detections_from_boxes([[0, 0, 10, 10], [200, 200, 210, 210]]),
        detections_from_boxes([[0, 0, 10, 10], [200, 200, 210, 210]]),
    ]

    byte_tracker = sv.ByteTrack(minimum_consecutive_frames=1)

    tracked = [byte_tracker.update_with_detections(frame) for frame in frames]
    assert tracked[-1].tracker_id is not None
    assert np.array_equal(np.sort(tracked[-1].tracker_id), np.array([1, 2]))
