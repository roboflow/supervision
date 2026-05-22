from __future__ import annotations

import numpy as np

from supervision.detection.core import Detections
from supervision.detection.tools.smoother import DetectionsSmoother


def test_smoother_confidence_averaged_when_all_frames_have_confidence() -> None:
    smoother = DetectionsSmoother()

    first = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.2], dtype=float),
        tracker_id=np.array([1]),
    )
    second = Detections(
        xyxy=np.array([[2, 2, 12, 12]], dtype=float),
        confidence=np.array([0.8], dtype=float),
        tracker_id=np.array([1]),
    )

    smoother.update_with_detections(first)
    smoothed = smoother.update_with_detections(second)

    assert smoothed.confidence is not None
    np.testing.assert_allclose(smoothed.confidence, np.array([0.5], dtype=float))


def test_detections_smoother_confidence_is_none_if_any_frame_missing() -> None:
    smoother = DetectionsSmoother()

    with_confidence = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.2], dtype=float),
        tracker_id=np.array([1]),
    )
    without_confidence = Detections(
        xyxy=np.array([[2, 2, 12, 12]], dtype=float),
        confidence=None,
        tracker_id=np.array([1]),
    )

    smoother.update_with_detections(with_confidence)
    smoothed = smoother.update_with_detections(without_confidence)

    assert smoothed.confidence is None


def test_detections_smoother_averages_xyxy_over_two_frames() -> None:
    smoother = DetectionsSmoother()

    first = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.8], dtype=float),
        tracker_id=np.array([1]),
    )
    second = Detections(
        xyxy=np.array([[2, 2, 12, 12]], dtype=float),
        confidence=np.array([0.8], dtype=float),
        tracker_id=np.array([1]),
    )

    smoother.update_with_detections(first)
    smoothed = smoother.update_with_detections(second)

    expected_xyxy = np.array([[1, 1, 11, 11]], dtype=float)
    np.testing.assert_allclose(smoothed.xyxy, expected_xyxy)


def test_detections_smoother_empty_detections_does_not_raise() -> None:
    smoother = DetectionsSmoother()

    smoothed = smoother.update_with_detections(Detections.empty())

    assert len(smoothed) == 0
