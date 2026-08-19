"""Tests for DetectionsSmoother bounding-box and confidence smoothing."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from supervision.config import ORIENTED_BOX_COORDINATES
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


class TestDetectionsSmootherOrientedBoxes:
    """Oriented corners must be smoothed alongside `xyxy` (issue #2318).

    Everything on the returned detection other than `xyxy` and `confidence` is
    copied from the oldest frame in the window, so the oriented corners used to
    describe a different position from the smoothed axis-aligned box beside
    them. The geometry helpers that read `xyxyxyxy` then disagree with `xyxy`.
    """

    @staticmethod
    def _obb(cx: float, cy: float, half: float = 10.0) -> Detections:
        """A square OBB centred at `(cx, cy)`, with a matching `xyxy`."""
        corners = np.array(
            [
                [
                    [cx - half, cy - half],
                    [cx + half, cy - half],
                    [cx + half, cy + half],
                    [cx - half, cy + half],
                ]
            ],
            dtype=np.float32,
        )
        return Detections(
            xyxy=np.array(
                [[cx - half, cy - half, cx + half, cy + half]], dtype=np.float32
            ),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
            tracker_id=np.array([1]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

    @staticmethod
    def _rectangle(angle: float) -> Detections:
        """Create a rotated non-square OBB centered at the origin."""
        base = np.array(
            [[[-20.0, -5.0], [20.0, -5.0], [20.0, 5.0], [-20.0, 5.0]]],
            dtype=np.float32,
        )
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        corners = base @ rotation.T
        return Detections(
            xyxy=np.array(
                [
                    [
                        corners[..., 0].min(),
                        corners[..., 1].min(),
                        corners[..., 0].max(),
                        corners[..., 1].max(),
                    ]
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
            tracker_id=np.array([1]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

    def test_corners_are_smoothed_with_the_box(self) -> None:
        """Translation smoothing moves OBB corners with the axis-aligned box."""
        smoother = DetectionsSmoother(length=3)
        for cx in (0.0, 100.0, 200.0):
            result = smoother.update_with_detections(self._obb(cx, 0.0))

        # Mean of the three centres.
        assert_allclose(result.xyxy[0], np.array([90.0, -10.0, 110.0, 10.0]))
        assert_allclose(
            result.data[ORIENTED_BOX_COORDINATES][0],
            np.array([[90.0, -10.0], [110.0, -10.0], [110.0, 10.0], [90.0, 10.0]]),
        )

    def test_corners_agree_with_the_smoothed_box(self):
        """The invariant that matters: the two must describe the same position."""
        smoother = DetectionsSmoother(length=3)
        for cx in (0.0, 100.0, 200.0):
            result = smoother.update_with_detections(self._obb(cx, 0.0))

        box_centre_x = (result.xyxy[0][0] + result.xyxy[0][2]) / 2
        corner_centre_x = result.data[ORIENTED_BOX_COORDINATES][0][:, 0].mean()
        assert_allclose(corner_centre_x, box_centre_x)

    def test_corner_order_is_aligned_before_averaging(self) -> None:
        """Cyclically shifted equivalent corners preserve the square geometry."""
        smoother = DetectionsSmoother(length=2)

        upright = np.array(
            [[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]],
            dtype=np.float32,
        )
        shifted = np.roll(upright, -1, axis=1)

        for corners in (upright, shifted):
            detections = Detections(
                xyxy=np.array([[-10.0, -10.0, 10.0, 10.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0]),
                tracker_id=np.array([1]),
                data={ORIENTED_BOX_COORDINATES: corners},
            )
            result = smoother.update_with_detections(detections)

        assert_allclose(result.data[ORIENTED_BOX_COORDINATES], upright)
        assert_allclose(
            result.xyxy,
            np.array([[-10.0, -10.0, 10.0, 10.0]], dtype=np.float32),
        )

    def test_rotated_rectangle_keeps_xyxy_and_obb_envelopes_consistent(self) -> None:
        """Averaged rotated corners derive the matching axis-aligned envelope."""
        smoother = DetectionsSmoother(length=2)
        for angle in (np.pi / 4, -np.pi / 4):
            result = smoother.update_with_detections(self._rectangle(angle))

        corners = result.data[ORIENTED_BOX_COORDINATES]
        assert_allclose(
            result.xyxy,
            np.array(
                [
                    [
                        corners[..., 0].min(),
                        corners[..., 1].min(),
                        corners[..., 0].max(),
                        corners[..., 1].max(),
                    ]
                ]
            ),
        )
        x = corners[0, :, 0]
        y = corners[0, :, 1]
        assert 0.0 < 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def test_mixed_oriented_box_metadata_is_dropped(self) -> None:
        """Mixed OBB and axis-aligned history must not retain stale OBB data."""
        smoother = DetectionsSmoother(length=2)
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[-10.0, -10.0, 10.0, 10.0]], dtype=np.float32),
                tracker_id=np.array([1]),
            )
        )
        result = smoother.update_with_detections(self._obb(100.0, 0.0))

        assert ORIENTED_BOX_COORDINATES not in result.data
        assert_allclose(result.xyxy, np.array([[40.0, -10.0, 60.0, 10.0]]))

    def test_detections_without_oriented_boxes_are_unaffected(self) -> None:
        """The common axis-aligned case must not gain the key."""
        smoother = DetectionsSmoother(length=2)
        for x in (0.0, 100.0):
            detections = Detections(
                xyxy=np.array([[x, 0.0, x + 20.0, 20.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0]),
                tracker_id=np.array([1]),
            )
            result = smoother.update_with_detections(detections)

        assert ORIENTED_BOX_COORDINATES not in result.data
        assert_allclose(result.xyxy[0], np.array([50.0, 0.0, 70.0, 20.0]))
