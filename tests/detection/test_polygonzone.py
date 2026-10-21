from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES
from tests.helpers import (
    _ConstantTransform,
    _create_detections,
    _ShiftTransform,
    _WrongShapeTransform,
)

DETECTION_BOXES = np.array(
    [
        [35.0, 35.0, 65.0, 65.0],
        [60.0, 60.0, 90.0, 90.0],
        [85.0, 85.0, 115.0, 115.0],
        [110.0, 110.0, 140.0, 140.0],
        [135.0, 135.0, 165.0, 165.0],
        [160.0, 160.0, 190.0, 190.0],
        [185.0, 185.0, 215.0, 215.0],
        [210.0, 210.0, 240.0, 240.0],
        [235.0, 235.0, 265.0, 265.0],
    ],
    dtype=np.float32,
)

DETECTIONS = _create_detections(
    xyxy=DETECTION_BOXES, class_id=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
)

POLYGON = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])


class TestPolygonZoneInit:
    @pytest.mark.parametrize(
        ("polygon", "triggering_anchors", "exception"),
        [
            (POLYGON, [sv.Position.CENTER], DoesNotRaise()),
            (
                POLYGON,
                [],
                pytest.raises(ValueError, match="Triggering anchors cannot be empty"),
            ),
        ],
    )
    def test_empty_anchors_raises(self, polygon, triggering_anchors, exception) -> None:
        with exception:
            sv.PolygonZone(polygon, triggering_anchors=triggering_anchors)

    def test_generator_triggering_anchors_is_materialized(self):
        """A generator passed for triggering_anchors must not be silently exhausted.

        Calls trigger() twice on the same zone: a materialized list keeps returning
        results on repeated calls, whereas an un-materialized generator would be
        exhausted after the first trigger() and silently yield no anchors on the second.
        """
        zone = sv.PolygonZone(
            POLYGON, triggering_anchors=(p for p in [sv.Position.CENTER])
        )
        detections = _create_detections(
            xyxy=[[140.0, 140.0, 160.0, 160.0]], class_id=[0]
        )
        assert zone.trigger(detections)[0]
        assert zone.trigger(detections)[0]

    @pytest.mark.parametrize(
        ("polygon", "expected_vertex_count"),
        [
            (np.empty((0, 2), dtype=int), 0),
            (np.array([[20, 20]]), 1),
            (np.array([[0, 0], [40, 40]]), 2),
        ],
    )
    def test_degenerate_polygon_raises(
        self, polygon: np.ndarray, expected_vertex_count: int
    ) -> None:
        """A polygon enclosing no area cannot be a zone and must not be accepted."""
        with pytest.raises(
            ValueError, match=f"at least 3 vertices.*got {expected_vertex_count}"
        ):
            sv.PolygonZone(polygon)

    def test_triangle_is_accepted(self) -> None:
        """Three vertices is the smallest polygon that encloses an area."""
        zone = sv.PolygonZone(np.array([[0, 0], [100, 0], [100, 100]]))

        assert zone.mask.sum() > 0

    def test_wrongly_shaped_polygon_raises(self) -> None:
        """A polygon must be (N, 2) coordinate pairs, not a flat or 3-D array."""
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            sv.PolygonZone(np.array([0, 0, 40, 0, 40, 40]))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(
                {"minimum_crossing_threshold": 0},
                "minimum_crossing_threshold",
                id="zero-crossing-threshold",
            ),
            pytest.param(
                {"max_absent_frames": 0}, "max_absent_frames", id="zero-absent-frames"
            ),
        ],
    )
    def test_non_positive_tracking_parameter_raises(
        self, kwargs: dict[str, int], match: str
    ) -> None:
        """Debounce and absence windows must allow at least one frame."""
        with pytest.raises(ValueError, match=match):
            sv.PolygonZone(POLYGON, **kwargs)


class TestPolygonZoneTrigger:
    @pytest.mark.parametrize(
        ("detections", "polygon_zone", "expected_results", "exception"),
        [
            (
                DETECTIONS,
                sv.PolygonZone(
                    POLYGON,
                    triggering_anchors=(
                        sv.Position.TOP_LEFT,
                        sv.Position.TOP_RIGHT,
                        sv.Position.BOTTOM_LEFT,
                        sv.Position.BOTTOM_RIGHT,
                    ),
                ),
                np.array(
                    [False, False, False, True, True, True, False, False, False],
                    dtype=bool,
                ),
                DoesNotRaise(),
            ),  # all four corners must be inside
            (
                DETECTIONS,
                sv.PolygonZone(POLYGON),
                np.array(
                    [False, False, True, True, True, True, False, False, False],
                    dtype=bool,
                ),
                DoesNotRaise(),
            ),  # default anchor (BOTTOM_CENTER)
            (
                DETECTIONS,
                sv.PolygonZone(
                    POLYGON,
                    triggering_anchors=[sv.Position.BOTTOM_CENTER],
                ),
                np.array(
                    [False, False, True, True, True, True, False, False, False],
                    dtype=bool,
                ),
                DoesNotRaise(),
            ),  # explicit BOTTOM_CENTER matches default
            (
                sv.Detections.empty(),
                sv.PolygonZone(POLYGON),
                np.array([], dtype=bool),
                DoesNotRaise(),
            ),  # empty detections return empty array
        ],
    )
    def test_anchor_configurations(
        self,
        detections: sv.Detections,
        polygon_zone: sv.PolygonZone,
        expected_results: np.ndarray,
        exception: Exception,
    ) -> None:
        with exception:
            in_zone = polygon_zone.trigger(detections)
            assert np.all(in_zone == expected_results)
            assert polygon_zone.current_count == int(np.sum(expected_results))

    def test_straddling_detection_assigned_to_one_zone(self) -> None:
        """Detection straddling two adjacent zones is counted in exactly one zone.

        The old implementation clipped each detection box to the zone's bounding box
        before computing anchors, so a box straddling two adjacent zones got a
        different anchor per zone and was double-counted.  The fix computes anchors
        from the original unclipped box, giving a single consistent position.

        Setup: zone_left covers x 0-99, zone_right covers x 100-200.
        Detection [60, 80, 140, 120] has BOTTOM_CENTER = round((60+140)/2), round(120)
        = (100, 120), which falls in zone_right only.
        """
        zone_left = sv.PolygonZone(
            np.array([[0, 0], [99, 0], [99, 200], [0, 200]], dtype=np.int32)
        )
        zone_right = sv.PolygonZone(
            np.array([[100, 0], [200, 0], [200, 200], [100, 200]], dtype=np.int32)
        )
        detections = _create_detections(
            xyxy=[[60.0, 80.0, 140.0, 120.0]],
            class_id=[0],
        )

        results = np.array(
            [zone.trigger(detections)[0] for zone in [zone_left, zone_right]],
            dtype=bool,
        )
        assert np.sum(results) == 1, (
            "Detection should appear in exactly one zone, not zero or multiple"
        )
        assert results[1], "BOTTOM_CENTER (100, 120) should be in zone_right"

    def test_out_of_bounds_anchor_excluded(self) -> None:
        """Anchors with negative coordinates are excluded, not wrapped or clamped."""
        zone = sv.PolygonZone(
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            triggering_anchors=[sv.Position.CENTER],
        )
        # CENTER = (round((-50+0)/2), round((25+75)/2)) = (-25, 50) — x < 0.
        detections = _create_detections(
            xyxy=[[-50.0, 25.0, 0.0, 75.0]],
            class_id=[0],
        )
        result = zone.trigger(detections)
        assert not result[0]

    def test_anchor_on_polygon_boundary_included(self) -> None:
        """An anchor landing exactly on a polygon corner is counted as inside."""
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        zone = sv.PolygonZone(polygon, triggering_anchors=[sv.Position.BOTTOM_RIGHT])
        # BOTTOM_RIGHT = (round(x2), round(y2)) = (100, 100) — the polygon corner.
        detections = _create_detections(
            xyxy=[[50.0, 50.0, 100.0, 100.0]],
            class_id=[0],
        )
        result = zone.trigger(detections)
        assert result[0]

    def test_anchor_on_polygon_boundary_included_any_mode(self) -> None:
        """With require_all_anchors=False and multiple anchors, an anchor landing
        exactly on the polygon boundary is enough to trigger, even though the other
        anchors of the same detection fall outside the polygon."""
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        anchors = [sv.Position.TOP_LEFT, sv.Position.BOTTOM_RIGHT]
        # TOP_LEFT = (-50, -50) is outside; BOTTOM_RIGHT = (100, 100) is exactly
        # the polygon corner (boundary), which counts as inside.
        detections = _create_detections(
            xyxy=[[-50.0, -50.0, 100.0, 100.0]],
            class_id=[0],
        )
        any_anchor_zone = sv.PolygonZone(
            polygon, triggering_anchors=anchors, require_all_anchors=False
        )
        all_anchors_zone = sv.PolygonZone(
            polygon, triggering_anchors=anchors, require_all_anchors=True
        )
        assert any_anchor_zone.trigger(detections)[0]
        assert not all_anchors_zone.trigger(detections)[0]

    def test_require_all_anchors_false_triggers_on_any_anchor(self) -> None:
        """With require_all_anchors=False, any anchor inside triggers."""
        # Box [85, 85, 115, 115] has only BOTTOM_RIGHT (115, 115) inside POLYGON
        # ([100, 100]..[200, 200]); the other three corners are outside.
        detections = _create_detections(xyxy=[[85.0, 85.0, 115.0, 115.0]], class_id=[0])
        anchors = (
            sv.Position.TOP_LEFT,
            sv.Position.TOP_RIGHT,
            sv.Position.BOTTOM_LEFT,
            sv.Position.BOTTOM_RIGHT,
        )
        all_required = sv.PolygonZone(POLYGON, triggering_anchors=anchors)
        any_anchor = sv.PolygonZone(
            POLYGON, triggering_anchors=anchors, require_all_anchors=False
        )
        assert not all_required.trigger(detections)[0]
        result = any_anchor.trigger(detections)
        assert result[0]
        assert any_anchor.current_count == 1

    def test_require_all_anchors_false_all_outside_does_not_trigger(self) -> None:
        """With require_all_anchors=False, a detection with every anchor outside the
        zone still does not trigger (exercises the np.any all-False branch)."""
        # Box [0, 0, 20, 20] has all four corners well outside POLYGON
        # ([100, 100]..[200, 200]).
        detections = _create_detections(xyxy=[[0.0, 0.0, 20.0, 20.0]], class_id=[0])
        anchors = (
            sv.Position.TOP_LEFT,
            sv.Position.TOP_RIGHT,
            sv.Position.BOTTOM_LEFT,
            sv.Position.BOTTOM_RIGHT,
        )
        any_anchor = sv.PolygonZone(
            POLYGON, triggering_anchors=anchors, require_all_anchors=False
        )
        result = any_anchor.trigger(detections)
        assert not result[0]
        assert any_anchor.current_count == 0

    def test_require_all_anchors_default_matches_explicit_true(self) -> None:
        """Omitting require_all_anchors and passing require_all_anchors=True explicitly
        must produce identical trigger() results, pinning the documented default
        (True)."""
        anchors = (
            sv.Position.TOP_LEFT,
            sv.Position.TOP_RIGHT,
            sv.Position.BOTTOM_LEFT,
            sv.Position.BOTTOM_RIGHT,
        )
        default_zone = sv.PolygonZone(POLYGON, triggering_anchors=anchors)
        explicit_true_zone = sv.PolygonZone(
            POLYGON, triggering_anchors=anchors, require_all_anchors=True
        )
        assert np.array_equal(
            default_zone.trigger(DETECTIONS), explicit_true_zone.trigger(DETECTIONS)
        )

    def test_require_all_anchors_has_no_effect_with_single_anchor(self) -> None:
        """With a single triggering anchor, require_all_anchors is a no-op: both
        settings must produce identical trigger() results (per the docstring:

        "Has no effect when triggering_anchors has a single entry").
        """
        # Box [140, 140, 160, 160] has its BOTTOM_CENTER inside POLYGON.
        detections = _create_detections(
            xyxy=[[140.0, 140.0, 160.0, 160.0]], class_id=[0]
        )
        require_all = sv.PolygonZone(
            POLYGON,
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
            require_all_anchors=True,
        )
        require_any = sv.PolygonZone(
            POLYGON,
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
            require_all_anchors=False,
        )
        assert np.array_equal(
            require_all.trigger(detections), require_any.trigger(detections)
        )

    def test_half_pixel_anchor_uses_nearest_pixel(self) -> None:
        """Half-pixel anchors should not be biased toward the larger x and y."""
        zone_left = sv.PolygonZone(
            np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.int32)
        )
        zone_right = sv.PolygonZone(
            np.array([[101, 0], [200, 0], [200, 200], [101, 200]], dtype=np.int32)
        )
        detections = _create_detections(
            xyxy=[[60.0, 80.0, 141.0, 120.0]],
            class_id=[0],
        )

        left_result = zone_left.trigger(detections)[0]
        right_result = zone_right.trigger(detections)[0]

        assert left_result
        assert not right_result


FOUR_CORNERS = (
    sv.Position.TOP_LEFT,
    sv.Position.TOP_RIGHT,
    sv.Position.BOTTOM_LEFT,
    sv.Position.BOTTOM_RIGHT,
)
# The camera moved by (30, -20): DETECTION_BOXES as seen in the current frame.
SHIFTED_DETECTIONS = _create_detections(
    xyxy=DETECTION_BOXES + np.array([30.0, -20.0, 30.0, -20.0]),
    class_id=[0] * len(DETECTION_BOXES),
)


class TestPolygonZoneTriggerWithCoordTransform:
    """`PolygonZone.trigger` with a per-frame `coord_transform`."""

    @pytest.mark.parametrize(
        ("triggering_anchors", "require_all_anchors", "transform", "expected"),
        [
            pytest.param(
                FOUR_CORNERS,
                True,
                _ShiftTransform(30, -20),
                [False, False, False, True, True, True, False, False, False],
                id="all-four-corners",
            ),
            pytest.param(
                FOUR_CORNERS,
                False,
                _ShiftTransform(30, -20),
                [False, False, True, True, True, True, True, False, False],
                id="any-of-four-corners",
            ),
            pytest.param(
                (sv.Position.BOTTOM_CENTER,),
                True,
                _ShiftTransform(30, -20),
                [False, False, True, True, True, True, False, False, False],
                id="bottom-center",
            ),
            pytest.param(
                (sv.Position.BOTTOM_CENTER,),
                True,
                None,
                [False, False, False, True, True, False, False, False, False],
                id="bottom-center-without-transform",
            ),
        ],
    )
    def test_shifted_detections_are_tested_in_reference_coordinates(
        self,
        triggering_anchors: tuple[sv.Position, ...],
        require_all_anchors: bool,
        transform: _ShiftTransform | None,
        expected: list[bool],
    ) -> None:
        """Anchors are mapped back by the camera shift before the inside test."""
        zone = sv.PolygonZone(POLYGON, triggering_anchors, require_all_anchors)

        in_zone = zone.trigger(SHIFTED_DETECTIONS, coord_transform=transform)

        np.testing.assert_array_equal(in_zone, expected)
        assert zone.current_count == sum(expected)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize(
        "mapped_point",
        [
            pytest.param((-0.4, 50.0), id="fractional-negative-x"),
            pytest.param((-60.0, 45.0), id="negative-x"),
            pytest.param((160.0, 45.0), id="beyond-mask-width"),
            pytest.param((45.0, -1e30), id="far-beyond-int-range"),
            pytest.param((np.nan, np.nan), id="nan"),
            pytest.param((np.inf, 45.0), id="inf"),
        ],
    )
    def test_anchor_mapped_out_of_bounds_is_not_counted(
        self, mapped_point: tuple[float, float]
    ) -> None:
        """An anchor inside the zone in frame but mapped outside its mask is out."""
        zone = sv.PolygonZone(np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        detections = _create_detections(xyxy=[[40.0, 40.0, 60.0, 50.0]], class_id=[0])

        in_zone = zone.trigger(
            detections, coord_transform=_ConstantTransform(*mapped_point)
        )

        np.testing.assert_array_equal(in_zone, [False])
        assert zone.current_count == 0

    def test_rejects_transform_returning_wrong_shape(self) -> None:
        """A transform that does not return one point per input point raises."""
        zone = sv.PolygonZone(POLYGON)

        with pytest.raises(ValueError, match="coord_transform must return"):
            zone.trigger(DETECTIONS, coord_transform=_WrongShapeTransform())


INSIDE_BOX = [140.0, 140.0, 160.0, 160.0]
OUTSIDE_BOX = [10.0, 10.0, 30.0, 30.0]


def _tracked_frame(
    positions: list[str], class_ids: list[int] | None = None
) -> sv.Detections:
    """Build one frame where object `i` has tracker id `i` at an 'in'/'out' box."""
    if not positions:
        return sv.Detections.empty()
    boxes = [INSIDE_BOX if position == "in" else OUTSIDE_BOX for position in positions]
    return _create_detections(
        xyxy=boxes,
        class_id=class_ids,
        tracker_id=list(range(len(positions))),
    )


def _run_sequence(zone: sv.PolygonZone, sequence: list[str | None]) -> None:
    """Trigger `zone` once per entry: 'in'/'out' for tracker 0, `None` for absent."""
    for position in sequence:
        zone.trigger(_tracked_frame([] if position is None else [position], [0]))


class TestPolygonZoneCurrentCountPerClass:
    @pytest.mark.parametrize(
        ("positions", "class_ids", "expected"),
        [
            pytest.param(
                ["in", "in", "in", "out"], [0, 1, 1, 0], {0: 1, 1: 2}, id="mixed"
            ),
            pytest.param(["in", "out"], [0, 3], {0: 1}, id="omits-absent-class"),
            pytest.param(["in", "in"], None, {None: 2}, id="unclassified"),
            pytest.param(["out"], None, {}, id="unclassified-none-inside"),
            pytest.param([], None, {}, id="empty-frame"),
        ],
    )
    def test_counts_objects_inside_by_class(
        self,
        positions: list[str],
        class_ids: list[int] | None,
        expected: dict[int | None, int],
    ) -> None:
        """Only detections inside the zone are counted, grouped by class id."""
        zone = sv.PolygonZone(POLYGON)

        zone.trigger(_tracked_frame(positions, class_ids))

        assert zone.current_count_per_class == expected

    def test_reflects_only_latest_frame(self) -> None:
        """Per-class occupancy is replaced, not accumulated, on each frame."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(_tracked_frame(["in", "in"], [0, 1]))

        zone.trigger(_tracked_frame(["in"], [1]))

        assert zone.current_count_per_class == {1: 1}


class TestPolygonZoneEntryExit:
    @pytest.mark.parametrize(
        ("sequence", "zone_kwargs", "expected_in_out"),
        [
            pytest.param(["out", "in"], {}, (1, 0), id="enters"),
            pytest.param(["out", "in", "out"], {}, (1, 1), id="enters-and-exits"),
            pytest.param(["in", "out"], {}, (0, 1), id="first-seen-inside"),
            pytest.param(["in", "in"], {}, (0, 0), id="stays-inside"),
            pytest.param(
                ["out", "in", "out", "in"], {}, (2, 1), id="re-enters-after-exit"
            ),
            pytest.param(
                ["out", "in", "out"],
                {"minimum_crossing_threshold": 2},
                (0, 0),
                id="flicker-below-threshold",
            ),
            pytest.param(
                ["out", "in", "in"],
                {"minimum_crossing_threshold": 2},
                (1, 0),
                id="sustained-entry-at-threshold",
            ),
            pytest.param(
                ["out", "in", None, "in"],
                {"minimum_crossing_threshold": 2},
                (1, 0),
                id="missing-frame-does-not-reset-run",
            ),
            pytest.param(
                ["out", None, None, "in"],
                {"max_absent_frames": 2},
                (1, 0),
                id="state-kept-through-short-absence",
            ),
            pytest.param(
                ["out", None, None, "in"],
                {"max_absent_frames": 1},
                (0, 0),
                id="state-dropped-after-long-absence",
            ),
            pytest.param(
                ["in", "out", "in"],
                {"minimum_crossing_threshold": 2},
                (0, 0),
                id="exit-flicker-below-threshold",
            ),
            pytest.param(
                ["in", "out", "out"],
                {"minimum_crossing_threshold": 2},
                (0, 1),
                id="sustained-exit-at-threshold",
            ),
            pytest.param(
                ["out", "in", None, None, "out"],
                {"max_absent_frames": 1},
                (1, 0),
                id="dropped-while-inside-is-not-an-exit",
            ),
        ],
    )
    def test_counts_confirmed_boundary_crossings(
        self,
        sequence: list[str | None],
        zone_kwargs: dict[str, int],
        expected_in_out: tuple[int, int],
    ) -> None:
        """A tracked object's entries and exits follow its confirmed side changes."""
        zone = sv.PolygonZone(POLYGON, **zone_kwargs)

        _run_sequence(zone, sequence)

        assert (zone.in_count, zone.out_count) == expected_in_out

    def test_repeated_tracker_id_is_one_observation(self) -> None:
        """Copies of one id in a frame cannot satisfy the debounce on their own."""
        zone = sv.PolygonZone(POLYGON, minimum_crossing_threshold=3)
        zone.trigger(_tracked_frame(["out"], [0]))
        duplicates = _create_detections(
            xyxy=[INSIDE_BOX] * 3, class_id=[0, 0, 0], tracker_id=[0, 0, 0]
        )

        zone.trigger(duplicates)

        assert zone.in_count == 0

    def test_counts_crossings_in_the_reference_frame(self) -> None:
        """With camera motion, crossings are judged where the zone was drawn."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(_tracked_frame(["out"], [0]))
        shifted_inside = _create_detections(
            xyxy=[[90.0, 140.0, 110.0, 160.0]], class_id=[0], tracker_id=[0]
        )

        zone.trigger(shifted_inside, coord_transform=_ShiftTransform(-50, 0))

        assert zone.in_count == 1

    def test_counts_entries_and_exits_per_class(self) -> None:
        """Each crossing is attributed to the class of the crossing detection."""
        zone = sv.PolygonZone(POLYGON)
        frames = [
            _tracked_frame(["out", "out", "out"], [0, 1, 1]),
            _tracked_frame(["in", "in", "in"], [0, 1, 1]),
            _tracked_frame(["in", "out", "in"], [0, 1, 1]),
        ]

        for frame in frames:
            zone.trigger(frame)

        assert zone.in_count_per_class == {0: 1, 1: 2}
        assert zone.out_count_per_class == {1: 1}

    def test_unclassified_crossings_are_keyed_by_none(self) -> None:
        """Detections without class ids are counted under the `None` key."""
        zone = sv.PolygonZone(POLYGON)

        for position in ["out", "in"]:
            zone.trigger(_tracked_frame([position]))

        assert zone.in_count_per_class == {None: 1}

    @pytest.mark.parametrize(
        "tracker_id",
        [
            pytest.param(-1.0, id="unconfirmed"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="positive-infinity"),
            pytest.param(-np.inf, id="negative-infinity"),
        ],
    )
    def test_unusable_tracker_ids_are_ignored(self, tracker_id: float) -> None:
        """Negative or non-finite tracker ids cannot be followed and never count."""
        zone = sv.PolygonZone(POLYGON)
        frames = [
            sv.Detections(
                xyxy=np.array([box], dtype=float), tracker_id=np.array([tracker_id])
            )
            for box in (OUTSIDE_BOX, INSIDE_BOX)
        ]

        for frame in frames:
            zone.trigger(frame)

        assert (zone.in_count, zone.out_count) == (0, 0)

    def test_untracked_detections_only_update_occupancy(self) -> None:
        """Without tracker ids, occupancy is counted but no entry is recorded."""
        zone = sv.PolygonZone(POLYGON)
        frames = [
            _create_detections(xyxy=[OUTSIDE_BOX], class_id=[0]),
            _create_detections(xyxy=[INSIDE_BOX], class_id=[0]),
        ]

        for frame in frames:
            zone.trigger(frame)

        assert (zone.current_count, zone.in_count) == (1, 0)


class TestPolygonZoneReset:
    def test_clears_counts_and_track_state(self) -> None:
        """After reset, a track already inside is re-seeded rather than counted."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(_tracked_frame(["out"], [0]))
        zone.trigger(_tracked_frame(["in"], [0]))

        zone.reset()
        zone.trigger(_tracked_frame(["out"], [0]))

        assert (zone.in_count, zone.out_count, zone.current_count) == (0, 0, 0)
        assert zone.in_count_per_class == {}


ZONE_PIXELS = 101 * 101
LEFT_HALF_PIXELS = 51 * 101


def _left_half_mask(image_wh: tuple[int, int] = (250, 250)) -> np.ndarray:
    """Return one `(1, H, W)` mask covering zone columns 100-150 on every zone row."""
    width, height = image_wh
    mask = np.zeros((1, height, width), dtype=bool)
    mask[0, 100:201, 100:151] = True
    return mask


class TestPolygonZoneOccupancy:
    @pytest.mark.parametrize(
        ("xyxy", "expected_pixels"),
        [
            pytest.param([[100, 100, 200, 200]], ZONE_PIXELS, id="zone-box"),
            pytest.param([[100, 100, 150, 200]], LEFT_HALF_PIXELS, id="left-half"),
            pytest.param(
                [[100, 100, 150, 200], [125, 100, 175, 200]],
                76 * 101,
                id="overlap-counted-once",
            ),
            pytest.param(
                [[100, 100, 150, 200], [100, 100, 150, 200]],
                LEFT_HALF_PIXELS,
                id="duplicate-boxes",
            ),
            pytest.param([[-50, -50, 150, 150]], 51 * 51, id="partly-outside"),
            pytest.param([[0, 0, 50, 50]], 0, id="outside"),
            pytest.param([[150, 200, 100, 100]], LEFT_HALF_PIXELS, id="reversed"),
            pytest.param(
                [[100.9, 100, 150.9, 200]], LEFT_HALF_PIXELS, id="fractional-edges"
            ),
        ],
    )
    def test_box_coverage(self, xyxy: list[list[float]], expected_pixels: int) -> None:
        """Covered zone pixels are the union of the boxes, edges inclusive."""
        zone = sv.PolygonZone(POLYGON)
        detections = sv.Detections(xyxy=np.array(xyxy, dtype=np.float64))

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(expected_pixels / ZONE_PIXELS)

    def test_empty_detections_are_zero(self) -> None:
        """No detections cover nothing."""
        zone = sv.PolygonZone(POLYGON)

        occupancy = zone.occupancy(sv.Detections.empty())

        assert occupancy == 0.0

    def test_mask_takes_precedence_over_box(self) -> None:
        """A mask narrower than its box is measured by its own pixels."""
        zone = sv.PolygonZone(POLYGON)
        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 200.0, 200.0]]), mask=_left_half_mask()
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(LEFT_HALF_PIXELS / ZONE_PIXELS)

    def test_compact_mask_matches_dense_mask(self) -> None:
        """Compact and dense masks of the same objects give the same occupancy."""
        zone = sv.PolygonZone(POLYGON)
        masks = np.concatenate([_left_half_mask(), np.zeros((1, 250, 250), bool)])
        masks[1, 150:240, 180:245] = True
        xyxy = sv.mask_to_xyxy(masks).astype(np.float32)
        compact = sv.CompactMask.from_dense(masks, xyxy, image_shape=(250, 250))
        expected = zone.occupancy(sv.Detections(xyxy=xyxy, mask=masks))
        compact_detections = sv.Detections(xyxy=xyxy, mask=compact)

        occupancy = zone.occupancy(compact_detections)

        assert occupancy == pytest.approx(expected)

    def test_compact_masks_outside_zone_are_zero(self) -> None:
        """Compact masks that never touch the zone cover none of it."""
        zone = sv.PolygonZone(POLYGON)
        masks = np.zeros((2, 250, 250), dtype=bool)
        masks[0, 10:40, 10:40] = True
        masks[1, 210:240, 10:240] = True
        xyxy = sv.mask_to_xyxy(masks).astype(np.float32)
        compact = sv.CompactMask.from_dense(masks, xyxy, image_shape=(250, 250))

        occupancy = zone.occupancy(sv.Detections(xyxy=xyxy, mask=compact))

        assert occupancy == 0.0

    def test_zone_with_negative_vertices(self) -> None:
        """A zone reaching past the frame's top-left corner is measured where drawn."""
        zone = sv.PolygonZone(
            np.array([[-50, -50], [100, -50], [100, 100], [-50, 100]])
        )
        detections = sv.Detections(xyxy=np.array([[-50.0, -50.0, 100.0, 100.0]]))

        occupancy = zone.occupancy(detections)

        assert occupancy == 1.0

    def test_mask_image_smaller_than_zone(self) -> None:
        """Zone pixels beyond the mask's image are counted as uncovered."""
        zone = sv.PolygonZone(POLYGON)
        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 149.0, 149.0]]),
            mask=np.ones((1, 150, 150), dtype=bool),
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(50 * 50 / ZONE_PIXELS)

    def test_oriented_boxes_take_precedence_over_box(self) -> None:
        """Oriented-box corners, not their axis-aligned envelope, are measured."""
        zone = sv.PolygonZone(POLYGON)
        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 200.0, 200.0]]),
            data={
                ORIENTED_BOX_COORDINATES: np.array(
                    [[[100, 100], [150, 100], [150, 200], [100, 200]]], dtype=float
                )
            },
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(LEFT_HALF_PIXELS / ZONE_PIXELS)

    def test_overlapping_oriented_boxes_are_counted_once(self) -> None:
        """The overlap of two oriented boxes stays covered instead of cancelling."""
        zone = sv.PolygonZone(POLYGON)
        corners = np.array(
            [
                [[100, 100], [150, 100], [150, 200], [100, 200]],
                [[125, 100], [175, 100], [175, 200], [125, 200]],
            ],
            dtype=float,
        )
        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 150.0, 200.0], [125.0, 100.0, 175.0, 200.0]]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(76 * 101 / ZONE_PIXELS)

    def test_rotated_box_crossing_the_zone_edge_is_exact(self) -> None:
        """A rotated box cut by the zone covers the pixels polygon_to_mask paints."""
        zone = sv.PolygonZone(np.array([[39, 23], [98, 23], [98, 82], [39, 82]]))
        corners = np.array([[[81, 1], [74, 37], [-6, 21], [2, -15]]], dtype=float)
        detections = sv.Detections(
            xyxy=np.array([[-6.0, -15.0, 81.0, 37.0]]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(435 / (60 * 60))

    def test_oriented_box_outside_zone_adds_nothing(self) -> None:
        """An oriented box away from the zone leaves the coverage unchanged."""
        zone = sv.PolygonZone(POLYGON)
        corners = np.array(
            [
                [[100, 100], [150, 100], [150, 200], [100, 200]],
                [[300, 300], [350, 300], [350, 350], [300, 350]],
            ],
            dtype=float,
        )
        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 150.0, 200.0], [300.0, 300.0, 350.0, 350.0]]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(51 * 101 / ZONE_PIXELS)

    def test_oriented_box_larger_than_canvas_limit_covers_zone(self) -> None:
        """A huge oriented box enclosing the zone is drawn on the zone's window."""
        zone = sv.PolygonZone(POLYGON)
        corners = np.array(
            [[[-5000, -5000], [5000, -5000], [5000, 5000], [-5000, 5000]]], dtype=float
        )
        detections = sv.Detections(
            xyxy=np.array([[-5000.0, -5000.0, 5000.0, 5000.0]]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        occupancy = zone.occupancy(detections)

        assert occupancy == pytest.approx(1.0)

    def test_rejects_misshaped_oriented_boxes(self) -> None:
        """Oriented-box data must be `(N, K, 2)` corner arrays."""
        zone = sv.PolygonZone(POLYGON)
        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 150.0, 150.0]]),
            data={ORIENTED_BOX_COORDINATES: np.zeros((1, 8))},
        )

        with pytest.raises(ValueError, match=r"\(N, K, 2\)"):
            zone.occupancy(detections)


class TestPolygonZoneTimeInZone:
    @pytest.mark.parametrize(
        ("sequence", "zone_kwargs", "expected"),
        [
            pytest.param(["out", "in", "in", "in"], {}, 2.0, id="since-entry"),
            pytest.param(["in", "in"], {}, 1.0, id="first-seen-inside"),
            pytest.param(["out", "in", "out"], {}, np.nan, id="after-exit"),
            pytest.param(
                ["out", "in", "out", "in", "in"], {}, 1.0, id="re-entry-restarts"
            ),
            pytest.param(["out"], {}, np.nan, id="never-inside"),
            pytest.param(
                ["out", "in", "in"],
                {"minimum_crossing_threshold": 2},
                1.0,
                id="entry-dated-to-run-start",
            ),
            pytest.param(
                ["in", "out", "in"],
                {"minimum_crossing_threshold": 2},
                2.0,
                id="flicker-keeps-stay",
            ),
            pytest.param(
                ["out", "in", None, "in"], {}, 2.0, id="short-absence-keeps-stay"
            ),
            pytest.param(
                ["in", None, None, "in"],
                {"max_absent_frames": 1},
                0.0,
                id="long-absence-restarts",
            ),
            pytest.param(
                ["in", "in", "out"],
                {"minimum_crossing_threshold": 2},
                2.0,
                id="unconfirmed-exit-keeps-stay",
            ),
        ],
    )
    def test_counts_frames_of_current_stay(
        self,
        sequence: list[str | None],
        zone_kwargs: dict[str, int],
        expected: float,
    ) -> None:
        """Without timestamps, dwell is the frames since the current stay began."""
        zone = sv.PolygonZone(POLYGON, **zone_kwargs)
        _run_sequence(zone, sequence)

        dwell = zone.time_in_zone(_tracked_frame(["in"], [0]))

        np.testing.assert_array_equal(dwell, [expected])

    def test_uses_given_timestamps(self) -> None:
        """Explicit timestamps set the unit, such as seconds."""
        zone = sv.PolygonZone(POLYGON)
        for position, timestamp in [("out", 10.0), ("in", 10.5), ("in", 12.0)]:
            zone.trigger(_tracked_frame([position], [0]), timestamp=timestamp)

        dwell = zone.time_in_zone(_tracked_frame(["in"], [0]))

        np.testing.assert_array_equal(dwell, [1.5])

    def test_aligns_with_the_detections_passed(self) -> None:
        """Each object gets its own dwell, in the order of the detections given."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(_tracked_frame(["in", "out"], [0, 0]))
        zone.trigger(_tracked_frame(["in", "in"], [0, 0]))
        detections = _create_detections(
            xyxy=[INSIDE_BOX, INSIDE_BOX, INSIDE_BOX], tracker_id=[1, 0, 9]
        )

        dwell = zone.time_in_zone(detections)

        np.testing.assert_array_equal(dwell, [0.0, 1.0, np.nan])

    @pytest.mark.parametrize(
        "detections",
        [
            pytest.param(
                _create_detections(xyxy=[INSIDE_BOX], tracker_id=[-1]),
                id="unconfirmed-tracker",
            ),
            pytest.param(_create_detections(xyxy=[INSIDE_BOX]), id="untracked"),
            pytest.param(
                sv.Detections(
                    xyxy=np.array([INSIDE_BOX]), tracker_id=np.array([np.nan])
                ),
                id="non-finite-tracker",
            ),
            pytest.param(
                sv.Detections(
                    xyxy=np.array([INSIDE_BOX]), tracker_id=np.array([np.inf])
                ),
                id="infinite-tracker",
            ),
        ],
    )
    def test_is_nan_without_a_confirmed_tracker_id(
        self, detections: sv.Detections
    ) -> None:
        """Objects the zone cannot follow have no dwell."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(detections)

        dwell = zone.time_in_zone(detections)

        np.testing.assert_array_equal(dwell, [np.nan])

    def test_is_nan_before_any_trigger(self) -> None:
        """A zone that has seen no frames reports no dwell."""
        zone = sv.PolygonZone(POLYGON)

        dwell = zone.time_in_zone(_tracked_frame(["in"], [0]))

        np.testing.assert_array_equal(dwell, [np.nan])

    @pytest.mark.parametrize(
        ("timestamps", "match"),
        [
            pytest.param([5.0, 4.0], "must not decrease", id="decreasing"),
            pytest.param(
                [5.0, None], "on every trigger call", id="seconds-then-frames"
            ),
            pytest.param(
                [None, 100.0], "on every trigger call", id="frames-then-seconds"
            ),
            pytest.param([np.nan], "must be finite", id="nan"),
            pytest.param([np.inf], "must be finite", id="inf"),
        ],
    )
    def test_rejects_invalid_timestamps(
        self, timestamps: list[float | None], match: str
    ) -> None:
        """A timestamp must be finite and never earlier than the previous one."""
        zone = sv.PolygonZone(POLYGON)
        *earlier, last = timestamps
        for timestamp in earlier:
            zone.trigger(_tracked_frame(["in"], [0]), timestamp=timestamp)

        with pytest.raises(ValueError, match=match):
            zone.trigger(_tracked_frame(["in"], [0]), timestamp=last)

    def test_reset_restarts_the_clock(self) -> None:
        """After reset, frame counting and stays start over."""
        zone = sv.PolygonZone(POLYGON)
        _run_sequence(zone, ["in", "in", "in"])

        zone.reset()
        zone.trigger(_tracked_frame(["in"], [0]))

        dwell = zone.time_in_zone(_tracked_frame(["in"], [0]))

        np.testing.assert_array_equal(dwell, [0.0])

    def test_reset_allows_switching_timestamp_mode(self) -> None:
        """After reset, a zone counted in frames can be given timestamps."""
        zone = sv.PolygonZone(POLYGON)
        _run_sequence(zone, ["in", "in"])
        zone.reset()

        zone.trigger(_tracked_frame(["in"], [0]), timestamp=10.0)

        np.testing.assert_array_equal(
            zone.time_in_zone(_tracked_frame(["in"], [0])), [0.0]
        )

    def test_equal_timestamps_are_allowed(self) -> None:
        """Two frames at the same time are not a decreasing clock."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(_tracked_frame(["in"], [0]), timestamp=3.0)

        zone.trigger(_tracked_frame(["in"], [0]), timestamp=3.0)

        np.testing.assert_array_equal(
            zone.time_in_zone(_tracked_frame(["in"], [0])), [0.0]
        )

    def test_failed_trigger_leaves_the_clock_unchanged(self) -> None:
        """A frame whose coord_transform raises does not advance dwell."""
        zone = sv.PolygonZone(POLYGON)
        zone.trigger(_tracked_frame(["in"], [0]), timestamp=0.0)
        with pytest.raises(ValueError, match="coord_transform must return"):
            zone.trigger(
                _tracked_frame(["in", "in"], [0, 0]),
                coord_transform=_WrongShapeTransform(),
                timestamp=100.0,
            )

        dwell = zone.time_in_zone(_tracked_frame(["in"], [0]))

        np.testing.assert_array_equal(dwell, [0.0])
