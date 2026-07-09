import numpy as np
import pytest

import supervision as sv
from supervision.tracker.byte_tracker import matching


def _detections_from_boxes(
    boxes: list[list[float]], confidence: list[float] | None = None
) -> sv.Detections:
    """Create detections with class ids and confidence for tracker regressions."""
    if confidence is None:
        confidence = [1.0] * len(boxes)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        confidence=np.array(confidence, dtype=np.float32),
    )


def test_top_level_bytetrack_access_returns_class() -> None:
    """Top-level ByteTrack access should still resolve to the class object."""
    sv.__dict__.pop("ByteTrack", None)
    tracker_cls = sv.ByteTrack
    assert tracker_cls is not None


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
    """ByteTrack should preserve stable tracker ids for repeated detections."""
    byte_tracker = sv.ByteTrack()
    tracked_detections = [byte_tracker.update_with_detections(d) for d in detections]
    assert tracked_detections[-1] == expected_results


def test_byte_tracker_does_not_skip_external_ids_for_short_lived_tracks() -> None:
    """Unconfirmed short-lived tracks should not consume external ids."""
    # A transient false-positive appears and disappears before becoming confirmed.
    # It should not consume an external tracker id.
    frames = [
        _detections_from_boxes([[0, 0, 10, 10]]),
        _detections_from_boxes([[0, 0, 10, 10], [100, 100, 110, 110]]),
        _detections_from_boxes([[0, 0, 10, 10]]),
        _detections_from_boxes([[0, 0, 10, 10], [200, 200, 210, 210]]),
        _detections_from_boxes([[0, 0, 10, 10], [200, 200, 210, 210]]),
    ]

    byte_tracker = sv.ByteTrack(minimum_consecutive_frames=1)

    tracked = [byte_tracker.update_with_detections(frame) for frame in frames]
    assert tracked[-1].tracker_id is not None
    assert np.array_equal(np.sort(tracked[-1].tracker_id), np.array([1, 2]))


def test_high_activation_threshold_can_start_track_at_score_one() -> None:
    """A high activation threshold must still allow maximum-confidence tracks."""
    byte_tracker = sv.ByteTrack(track_activation_threshold=0.95)
    detections = _detections_from_boxes([[0, 0, 10, 10]], confidence=[1.0])

    tracked = byte_tracker.update_with_detections(detections)

    assert tracked.tracker_id is not None
    assert np.array_equal(tracked.tracker_id, np.array([1]))


def test_update_with_detections_does_not_mutate_input() -> None:
    """Tracking should return a copy instead of writing ids into the input."""
    byte_tracker = sv.ByteTrack()
    detections = _detections_from_boxes([[0, 0, 10, 10]])

    _ = byte_tracker.update_with_detections(detections)

    assert detections.tracker_id is None


def test_update_with_tensors_activates_on_second_consecutive_frame() -> None:
    """A second consecutive tensor frame should activate the delayed track."""
    byte_tracker = sv.ByteTrack(minimum_consecutive_frames=2)
    tensors = np.array([[0, 0, 10, 10, 0.9]], dtype=np.float32)

    first_frame = byte_tracker.update_with_tensors(tensors)
    second_frame = byte_tracker.update_with_tensors(tensors)

    assert first_frame == []
    assert len(second_frame) == 1
    assert second_frame[0].is_activated
    assert second_frame[0].external_track_id == 1


def test_score_equal_to_activation_threshold_keeps_existing_track() -> None:
    """A detection exactly at the threshold should remain eligible to match."""
    byte_tracker = sv.ByteTrack(track_activation_threshold=0.5)
    _ = byte_tracker.update_with_detections(
        _detections_from_boxes([[0, 0, 10, 10]], confidence=[1.0])
    )

    tracked = byte_tracker.update_with_detections(
        _detections_from_boxes([[0, 0, 10, 10]], confidence=[0.5])
    )

    assert tracked.tracker_id is not None
    assert np.array_equal(tracked.tracker_id, np.array([1]))


def test_linear_assignment_does_not_mutate_cost_matrix() -> None:
    """Assignment should not rewrite the caller-owned cost matrix."""
    cost_matrix = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    original = cost_matrix.copy()

    _ = matching.linear_assignment(cost_matrix, thresh=0.5)

    assert np.array_equal(cost_matrix, original)


@pytest.mark.parametrize(
    "tensors",
    [
        pytest.param(
            np.array([[np.nan, 0, 10, 10, 0.9]], dtype=np.float32),
            id="nan",
        ),
        pytest.param(
            np.array([[0, np.inf, 10, 10, 0.9]], dtype=np.float32),
            id="inf",
        ),
        pytest.param(
            np.array([[0, 0, 0, 10, 0.9]], dtype=np.float32),
            id="zero-width",
        ),
        pytest.param(
            np.array([[10, 0, 0, 10, 0.9]], dtype=np.float32),
            id="negative-width",
        ),
        pytest.param(
            np.array([[0, 0, 10, 0, 0.9]], dtype=np.float32),
            id="zero-height",
        ),
        pytest.param(
            np.array([[0, 10, 10, 0, 0.9]], dtype=np.float32),
            id="negative-height",
        ),
        pytest.param(np.empty((0, 5), dtype=np.float32), id="empty"),
    ],
)
def test_update_with_tensors_ignores_invalid_boxes(
    tensors: np.ndarray,
) -> None:
    """Invalid tensors should be dropped before track creation."""
    byte_tracker = sv.ByteTrack()

    tracks = byte_tracker.update_with_tensors(tensors)

    assert tracks == []


def test_update_with_tensors_respects_min_consecutive_frames_on_first_frame() -> None:
    """First-frame tensor updates should not emit unconfirmed track id -1."""
    byte_tracker = sv.ByteTrack(minimum_consecutive_frames=2)
    tensors = np.array([[0, 0, 10, 10, 0.9]], dtype=np.float32)

    tracks = byte_tracker.update_with_tensors(tensors)

    assert tracks == []
