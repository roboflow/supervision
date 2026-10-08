"""Tests for DetectionsSmoother bounding-box and confidence smoothing."""

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.tools.smoother import DetectionsSmoother
from supervision.utils.internal import SupervisionWarnings


def _track(x: float, confidence: float, tracker_id: int = 7) -> Detections:
    """Build one 10x10 tracked box whose left edge sits at `x`."""
    return Detections(
        xyxy=np.array([[x, 0, x + 10, 10]], dtype=np.float32),
        confidence=np.array([confidence]),
        tracker_id=np.array([tracker_id]),
    )


def _both_tracks() -> Detections:
    """Build one frame holding tracks 1 and 2, without confidence scores."""
    return Detections(
        xyxy=np.array([[0, 0, 10, 10], [50, 0, 60, 10]], dtype=np.float32),
        tracker_id=np.array([1, 2]),
    )


def _untracked_empty() -> Detections:
    """Build the frame `Detections.empty()` returns: no boxes, no tracker IDs."""
    return Detections.empty()


def _untracked_nonempty() -> Detections:
    """Build a frame with one box but no tracker IDs."""
    return Detections(xyxy=np.array([[200, 0, 210, 10]], dtype=float))


def _tracked_empty() -> Detections:
    """Build a frame with no boxes and an empty tracker-ID array."""
    return Detections(
        xyxy=np.empty((0, 4), dtype=np.float32), tracker_id=np.array([], dtype=int)
    )


def _feed(
    smoother: DetectionsSmoother, frame_factories: list[Callable[[], Detections]]
) -> None:
    """Send one frame per factory to `smoother`, discarding the results."""
    for make_frame in frame_factories:
        smoother.update_with_detections(make_frame())


# Frames without tracker IDs warn on every call; the tests below check their effect on
# history, so the warning itself is asserted only where it is the behavior under test.
@pytest.mark.filterwarnings("ignore::supervision.utils.internal.SupervisionWarnings")
class TestDetectionsSmoother:
    @pytest.mark.parametrize(
        "make_missing_frame",
        [
            pytest.param(_untracked_empty, id="untracked-empty"),
            pytest.param(_untracked_nonempty, id="untracked-nonempty"),
            pytest.param(_tracked_empty, id="tracked-empty"),
        ],
    )
    @pytest.mark.parametrize(
        ("gap_frames", "expected_xyxy", "expected_confidence"),
        [
            pytest.param(0, [[50, 0, 60, 10]], [0.7], id="no-gap"),
            pytest.param(1, [[50, 0, 60, 10]], [0.7], id="gap-inside-window"),
            pytest.param(2, [[100, 0, 110, 10]], [0.9], id="gap-evicts-first-frame"),
            pytest.param(3, [[100, 0, 110, 10]], [0.9], id="gap-fills-window"),
        ],
    )
    def test_returning_track_averages_only_frames_inside_window(
        self,
        make_missing_frame: Callable[[], Detections],
        gap_frames: int,
        expected_xyxy: list[list[int]],
        expected_confidence: list[float],
    ) -> None:
        """Frames missing tracker IDs or boxes age history like any other frame."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))
        _feed(smoother, [make_missing_frame] * gap_frames)

        result = smoother.update_with_detections(_track(100, 0.9))

        assert_allclose(result.xyxy, expected_xyxy)
        assert_allclose(result.confidence, expected_confidence)

    @pytest.mark.parametrize(
        ("smoother_kwargs", "gap_frames", "track_survives"),
        [
            pytest.param({"length": 1}, 1, False, id="length-1-expires-after-1"),
            pytest.param({"length": 2}, 1, True, id="length-2-survives-1"),
            pytest.param({"length": 2}, 2, False, id="length-2-expires-after-2"),
            pytest.param({"length": 3}, 2, True, id="length-3-survives-2"),
            pytest.param({"length": 3}, 3, False, id="length-3-expires-after-3"),
            pytest.param({}, 4, True, id="default-length-survives-4"),
            pytest.param({}, 5, False, id="default-length-expires-after-5"),
        ],
    )
    def test_track_expires_once_untracked_frames_fill_the_window(
        self, smoother_kwargs: dict[str, int], gap_frames: int, track_survives: bool
    ) -> None:
        """A track survives fewer untracked frames than the window length."""
        smoother = DetectionsSmoother(**smoother_kwargs)
        smoother.update_with_detections(_track(0, 0.5))

        _feed(smoother, [_untracked_empty] * gap_frames)

        assert (smoother.get_track(7) is not None) == track_survives

    def test_alternating_untracked_and_tracked_empty_frames_age_history(self) -> None:
        """Both kinds of missing-ID frame count toward the same window."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))

        _feed(smoother, [_untracked_empty, _tracked_empty, _untracked_nonempty])

        assert smoother.get_track(7) is None

    def test_untracked_frame_returns_input_unchanged_with_cached_history(self) -> None:
        """A frame without tracker IDs is passed through even when tracks are cached."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))
        untracked = _untracked_nonempty()

        with pytest.warns(SupervisionWarnings, match="requires tracker_id"):
            result = smoother.update_with_detections(untracked)

        assert result is untracked

    def test_untracked_frames_expire_every_cached_track_together(self) -> None:
        """Two tracks seen on the same frame both expire when the window fills."""
        smoother = DetectionsSmoother(length=2)
        smoother.update_with_detections(_both_tracks())

        _feed(smoother, [_untracked_empty] * 2)

        assert smoother.get_track(1) is None
        assert smoother.get_track(2) is None

    def test_untracked_frame_expires_the_older_track_first(self) -> None:
        """Tracks first seen on different frames age out on different frames."""
        smoother = DetectionsSmoother(length=2)
        smoother.update_with_detections(_track(0, 0.5, tracker_id=1))
        smoother.update_with_detections(_track(50, 0.5, tracker_id=2))

        _feed(smoother, [_untracked_empty])

        assert smoother.get_track(1) is None
        assert smoother.get_track(2) is not None

    def test_returning_track_does_not_revive_an_absent_track(self) -> None:
        """Only the returning track is emitted; the absent one keeps aging silently."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_both_tracks())
        _feed(smoother, [_untracked_empty])

        result = smoother.update_with_detections(_track(100, 0.9, tracker_id=1))

        assert result.tracker_id is not None
        assert result.tracker_id.tolist() == [1]
        assert_allclose(result.xyxy, [[50, 0, 60, 10]])
        assert smoother.get_track(2) is not None

    def test_returning_oriented_track_ignores_expired_corners(self) -> None:
        """Oriented corners age out together with the axis-aligned box."""
        smoother = DetectionsSmoother(length=2)
        corners = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=float)
        first = _track(0, 0.5)
        first.data[ORIENTED_BOX_COORDINATES] = corners
        returned = _track(100, 0.9)
        returned.data[ORIENTED_BOX_COORDINATES] = corners + np.array([100, 0])
        smoother.update_with_detections(first)
        _feed(smoother, [_untracked_empty] * 2)

        result = smoother.update_with_detections(returned)

        assert_allclose(
            result.data[ORIENTED_BOX_COORDINATES], corners + np.array([100, 0])
        )

    def test_partially_aged_track_returns_its_remaining_box(self) -> None:
        """A window holding one sample and two empty slots still yields that sample."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))
        _feed(smoother, [_untracked_empty] * 2)

        track = smoother.get_track(7)

        assert track is not None
        assert_allclose(track.xyxy, [[0, 0, 10, 10]])
        assert_allclose(track.confidence, [0.5])

    def test_get_smoothed_detections_includes_aged_tracks(self) -> None:
        """Without `track_ids`, a track with empty slots in its window is emitted."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))
        _feed(smoother, [_untracked_empty])

        result = smoother.get_smoothed_detections()

        assert len(result) == 1
        assert_allclose(result.xyxy, [[0, 0, 10, 10]])

    def test_untracked_frame_after_reset_leaves_no_tracks(self) -> None:
        """An untracked frame after `reset` does not resurrect any cached track."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))
        smoother.reset()

        _feed(smoother, [_untracked_empty])

        assert len(smoother.get_smoothed_detections()) == 0

    def test_returning_frame_without_confidence_averages_present_frames(self) -> None:
        """Confidence is averaged over the frames that carry it, skipping the gap."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5))
        _feed(smoother, [_untracked_empty])
        returned = Detections(
            xyxy=np.array([[100, 0, 110, 10]], dtype=np.float32),
            tracker_id=np.array([7]),
        )

        result = smoother.update_with_detections(returned)

        assert_allclose(result.confidence, [0.5])

    def test_tracks_disagreeing_on_confidence_drop_it_for_all(self) -> None:
        """One track without confidence in its window removes confidence for all."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(_track(0, 0.5, tracker_id=1))
        _feed(smoother, [_untracked_empty])

        result = smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [50, 0, 60, 10]], dtype=np.float32),
                tracker_id=np.array([1, 2]),
            )
        )

        assert len(result) == 2
        assert result.confidence is None

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

    def test_smoother_keeps_current_frame_metadata_across_tracks(self) -> None:
        """Tracks first seen on different frames merge with this frame's metadata."""
        smoother = DetectionsSmoother(length=3)
        smoother.update_with_detections(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                tracker_id=np.array([1]),
                metadata={"frame_index": 0},
            )
        )

        smoothed = smoother.update_with_detections(
            Detections(
                xyxy=np.array([[2, 2, 12, 12], [30, 30, 40, 40]], dtype=np.float32),
                tracker_id=np.array([1, 2]),
                metadata={"frame_index": 1},
            )
        )

        assert smoothed.metadata == {"frame_index": 1}
        assert_allclose(
            smoothed.xyxy, np.array([[1, 1, 11, 11], [30, 30, 40, 40]]), atol=1e-5
        )

    def test_smoother_reports_current_frame_class(self) -> None:
        """A track that changes class reports its latest class, not the oldest."""
        smoother = DetectionsSmoother(length=3)
        for class_id, class_name in ((0, "car"), (1, "truck")):
            smoothed = smoother.update_with_detections(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([class_id]),
                    tracker_id=np.array([1]),
                    data={"class_name": np.array([class_name])},
                )
            )

        assert smoothed.class_id is not None
        assert smoothed.class_id.tolist() == [1]
        assert smoothed["class_name"].tolist() == ["truck"]

    def test_reset_clears_track_history(self) -> None:
        """Reset() must drop cached frames so post-reset output ignores prior boxes."""
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
        """Reset() must keep the configured window so maxlen still bounds new tracks."""
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

    def test_warning_raised_as_error_leaves_history_unchanged(self) -> None:
        """A raised warning aborts the update before history ages, keeping it atomic."""
        smoother = DetectionsSmoother(length=2)
        smoother.update_with_detections(_track(0, 0.5))
        with warnings.catch_warnings():
            warnings.simplefilter("error", SupervisionWarnings)
            with pytest.raises(SupervisionWarnings):
                smoother.update_with_detections(_untracked_nonempty())

        result = smoother.update_with_detections(_track(100, 0.9))

        assert_allclose(result.xyxy, [[50, 0, 60, 10]])


class TestDetectionsSmootherOrientedBoxes:
    """Oriented corners must be smoothed alongside `xyxy` (issue #2318).

    Everything on the returned detection other than `xyxy` and `confidence` is copied
    from the oldest frame in the window, so the oriented corners used to describe a
    different position from the smoothed axis-aligned box beside them. The geometry
    helpers that read `xyxyxyxy` then disagree with `xyxy`.
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
