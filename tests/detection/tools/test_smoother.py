"""Tests for DetectionsSmoother bounding-box and confidence smoothing."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from supervision.detection.core import Detections
from supervision.detection.tools.smoother import DetectionsSmoother
from supervision.utils.internal import SupervisionWarnings


class TestDetectionsSmoother:
    @pytest.mark.parametrize(
        ("conf1", "conf2", "expected_confidence"),
        [
            pytest.param(
                np.array([0.5]),
                np.array([0.7]),
                np.array([0.6]),
                id="with_confidence",
            ),
            pytest.param(
                None,
                None,
                None,
                id="no_confidence",
            ),
            pytest.param(
                np.array([0.5]),
                None,
                np.array([0.5]),
                id="mixed_window_averages_present",
            ),
        ],
    )
    def test_smoother_confidence_scenarios(
        self,
        conf1: np.ndarray | None,
        conf2: np.ndarray | None,
        expected_confidence: np.ndarray | None,
    ) -> None:
        """Boxes average over window; confidence averages present values or None."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                confidence=conf1,
                tracker_id=np.array([1]),
            )
        )
        smoothed = smoother.update_with_detections(
            Detections(
                xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
                confidence=conf2,
                tracker_id=np.array([1]),
            )
        )

        assert_allclose(smoothed.xyxy, np.array([[1, 1, 11, 11]]), atol=1e-5)
        if expected_confidence is None:
            assert smoothed.confidence is None
        else:
            assert smoothed.confidence is not None
            assert_allclose(smoothed.confidence, expected_confidence, atol=1e-5)

    def test_smoother_reappearing_track_keeps_history(self) -> None:
        """Missing tracks stay silent but still contribute when they return."""
        smoother = DetectionsSmoother(length=3)
        first = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.5]),
            tracker_id=np.array([1]),
        )
        missing = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            tracker_id=np.array([], dtype=int),
        )
        returned = Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
            confidence=np.array([0.7]),
            tracker_id=np.array([1]),
        )

        smoother.update_with_detections(first)
        smoothed_missing = smoother.update_with_detections(missing)
        smoothed_returned = smoother.update_with_detections(returned)

        assert len(smoothed_missing) == 0
        assert len(smoothed_returned) == 1
        assert smoothed_returned.confidence is not None
        assert_allclose(smoothed_returned.xyxy, np.array([[1, 1, 11, 11]]), atol=1e-5)
        assert_allclose(smoothed_returned.confidence, np.array([0.6]), atol=1e-5)

    def test_smoother_tracker_id_none_warns_and_returns_unchanged(self) -> None:
        """update_with_detections warns and returns input when tracker_id is None."""
        smoother = DetectionsSmoother(length=3)
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            tracker_id=None,
        )

        with pytest.warns(SupervisionWarnings):
            result = smoother.update_with_detections(detections)

        assert result is detections

    def test_smoother_window_full_averages_all_frames(self) -> None:
        """Full window (length=3) averages all 3 frames, not just the last two."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                confidence=np.array([0.3]),
                tracker_id=np.array([1]),
            )
        )
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[3, 3, 13, 13]], dtype=np.float32),
                confidence=np.array([0.6]),
                tracker_id=np.array([1]),
            )
        )
        smoothed = smoother.update_with_detections(
            Detections(
                xyxy=np.array([[6, 6, 16, 16]], dtype=np.float32),
                confidence=np.array([0.9]),
                tracker_id=np.array([1]),
            )
        )

        assert_allclose(smoothed.xyxy, np.array([[3, 3, 13, 13]]), atol=1e-5)
        assert smoothed.confidence is not None
        assert_allclose(smoothed.confidence, np.array([0.6]), atol=1e-5)

    def test_smoother_does_not_emit_missing_tracks(self) -> None:
        """A missing track should keep history but stop emitting ghost boxes."""
        smoother = DetectionsSmoother(length=3)
        first = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.3]),
            tracker_id=np.array([1]),
        )
        missing = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            tracker_id=np.array([], dtype=int),
        )
        second = Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
            confidence=np.array([0.9]),
            tracker_id=np.array([1]),
        )

        smoother.update_with_detections(first)
        smoothed_missing = smoother.update_with_detections(missing)
        smoothed_returned = smoother.update_with_detections(second)

        assert len(smoothed_missing) == 0
        assert smoothed_returned.confidence is not None
        assert_allclose(smoothed_returned.xyxy, np.array([[1, 1, 11, 11]]), atol=1e-5)

    def test_reset_clears_track_history(self) -> None:
        """reset() must drop cached frames so post-reset output ignores prior boxes."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                confidence=np.array([0.5]),
                tracker_id=np.array([1]),
            )
        )

        smoother.reset()
        smoothed = smoother.update_with_detections(
            Detections(
                xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32),
                confidence=np.array([0.7]),
                tracker_id=np.array([1]),
            )
        )

        assert len(smoother.tracks) == 1
        assert_allclose(smoothed.xyxy, np.array([[2, 2, 12, 12]]), atol=1e-5)

    def test_reset_preserves_window_length(self) -> None:
        """reset() must keep the configured window so maxlen still bounds new tracks."""
        smoother = DetectionsSmoother(length=2)
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                tracker_id=np.array([1]),
            )
        )

        smoother.reset()
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                tracker_id=np.array([9]),
            )
        )

        assert smoother.tracks[9].maxlen == 2
