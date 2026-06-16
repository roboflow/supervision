from __future__ import annotations

import numpy as np

from supervision.detection.core import Detections
from supervision.detection.tools.smoother import DetectionsSmoother


def test_smoother_averages_xyxy_and_confidence() -> None:
    """Boxes and confidence are averaged over the window."""
    smoother = DetectionsSmoother(length=3)
    smoother.update_with_detections(
        Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.5]),
            tracker_id=np.array([1]),
        )
    )
    smoothed = smoother.update_with_detections(
        Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
            confidence=np.array([0.7]),
            tracker_id=np.array([1]),
        )
    )

    np.testing.assert_array_equal(smoothed.xyxy, np.array([[1, 1, 11, 11]]))
    np.testing.assert_allclose(smoothed.confidence, np.array([0.6]))


def test_smoother_without_confidence_does_not_crash() -> None:
    """Detections without confidence smooth boxes and keep confidence as None."""
    smoother = DetectionsSmoother(length=3)
    smoother.update_with_detections(
        Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            tracker_id=np.array([1]),
        )
    )
    smoothed = smoother.update_with_detections(
        Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
            tracker_id=np.array([1]),
        )
    )

    np.testing.assert_array_equal(smoothed.xyxy, np.array([[1, 1, 11, 11]]))
    assert smoothed.confidence is None


def test_smoother_mixed_confidence_window_yields_none() -> None:
    """A window mixing frames with and without confidence keeps confidence as None."""
    smoother = DetectionsSmoother(length=3)
    smoother.update_with_detections(
        Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.5]),
            tracker_id=np.array([1]),
        )
    )
    smoothed = smoother.update_with_detections(
        Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
            tracker_id=np.array([1]),
        )
    )

    np.testing.assert_array_equal(smoothed.xyxy, np.array([[1, 1, 11, 11]]))
    assert smoothed.confidence is None
