from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.tools.polygon_zone import _boxes_union_mask
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
    @pytest.mark.parametrize("dtype_name", ["int8", "uint8", "int16", "uint16"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_retains_zone_at_integer_coordinate_limit(
        self, dtype_name: str, axis: int
    ) -> None:
        """The mask margin cannot wrap at a polygon coordinate's integer limit."""
        limit = int(np.iinfo(np.dtype(dtype_name)).max)
        width, height = (limit, 3) if axis == 0 else (3, limit)
        polygon = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=dtype_name
        )

        zone = sv.PolygonZone(polygon)

        expected = np.zeros((height + 2, width + 2), dtype=np.uint8)
        expected[: height + 1, : width + 1] = 1
        np.testing.assert_array_equal(zone.mask, expected)
        np.testing.assert_array_equal(zone.polygon, polygon)

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
    def test_empty_anchors_raises(
        self,
        polygon: np.ndarray,
        triggering_anchors: list[sv.Position],
        exception: AbstractContextManager[object],
    ) -> None:
        """An empty list of triggering anchors is rejected."""
        with exception:
            sv.PolygonZone(polygon, triggering_anchors=triggering_anchors)

    def test_generator_triggering_anchors_is_materialized(self) -> None:
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
    @pytest.mark.parametrize("dtype_name", ["int8", "uint8", "int16", "uint16"])
    def test_counts_inside_narrow_integer_polygon(self, dtype_name: str) -> None:
        """Inside detections still trigger when the polygon reaches its dtype limit."""
        width = int(np.iinfo(np.dtype(dtype_name)).max)
        polygon = np.array([[0, 0], [width, 0], [width, 3], [0, 3]], dtype=dtype_name)
        zone = sv.PolygonZone(polygon, triggering_anchors=[sv.Position.TOP_LEFT])
        detections = sv.Detections(
            xyxy=np.array(
                [[1, 1, 2, 2], [width, 2, width + 1, 3], [width + 1, 1, width + 2, 2]]
            )
        )

        in_zone = zone.trigger(detections)

        np.testing.assert_array_equal(in_zone, [True, True, False])
        assert zone.current_count == 2

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
        exception: AbstractContextManager[object],
    ) -> None:
        """Each anchor configuration triggers the expected detections."""
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

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(None, id="direct"),
            pytest.param(_ShiftTransform(0, 0), id="identity"),
        ],
    )
    def test_rejects_negative_subpixel_anchors(
        self, transform: _ShiftTransform | None
    ) -> None:
        """A negative subpixel anchor stays outside when its nearest pixel is zero."""
        zone = sv.PolygonZone(
            np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
            triggering_anchors=[sv.Position.TOP_LEFT],
        )
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [-0.4, 5.0, 4.0, 8.0],
                    [5.0, -0.4, 8.0, 4.0],
                    [0.0, 5.0, 4.0, 8.0],
                    [5.0, 0.0, 8.0, 4.0],
                    [0.4, 5.0, 4.0, 8.0],
                    [20.0, 5.0, 24.0, 8.0],
                    [22.0, 5.0, 24.0, 8.0],
                ]
            )
        )

        in_zone = zone.trigger(detections, coord_transform=transform)

        np.testing.assert_array_equal(
            in_zone, [False, False, True, True, True, True, False]
        )
        assert zone.current_count == 4

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_oversized_anchors_are_excluded_without_warning(self) -> None:
        """Finite coordinates outside the integer range are excluded without warning."""
        zone = sv.PolygonZone(
            np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
            triggering_anchors=[sv.Position.TOP_LEFT],
        )
        detections = sv.Detections(
            xyxy=np.array([[1e30, 5.0, 1e30, 8.0], [5.0, -1e30, 8.0, 5.0]])
        )

        in_zone = zone.trigger(detections)

        np.testing.assert_array_equal(in_zone, [False, False])
        assert zone.current_count == 0

    def test_float16_anchor_on_wide_polygon_boundary_is_included(self) -> None:
        """The mask width keeps its integer precision for float16 anchors."""
        zone = sv.PolygonZone(
            np.array([[0, 0], [8192, 0], [8192, 8], [0, 8]]),
            triggering_anchors=[sv.Position.BOTTOM_RIGHT],
        )
        detections = sv.Detections(
            xyxy=np.array([[8190, 1, 8192, 5]], dtype=np.float16)
        )

        in_zone = zone.trigger(detections)

        np.testing.assert_array_equal(in_zone, [True])
        assert zone.current_count == 1

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
            pytest.param(
                (np.uint64(0x7FF0000000000001).view(np.float64), 45.0),
                id="signaling-nan",
            ),
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


#: Square zone whose vertices span pixels 0..99, so it holds 100 x 100 pixels.
OCCUPANCY_ZONE = np.array([[0, 0], [99, 0], [99, 99], [0, 99]])

#: Detections a zero-area zone must reject, with or without anything to measure.
ZERO_AREA_ZONE_DETECTIONS = [
    pytest.param(sv.Detections.empty(), id="empty-detections"),
    pytest.param(
        sv.Detections(xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32)),
        id="one-box",
    ),
]


def _masks_detections(
    frame_hw: tuple[int, int], regions: list[tuple[slice, slice]]
) -> sv.Detections:
    """Build detections with one dense boolean mask per `(rows, cols)` region."""
    masks = np.zeros((len(regions), *frame_hw), dtype=bool)
    for index, (rows, cols) in enumerate(regions):
        masks[index, rows, cols] = True
    return sv.Detections(xyxy=sv.mask_to_xyxy(masks).astype(np.float32), mask=masks)


class TestBoxesUnionMask:
    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param([[1.7, 2.2, 6.9, 7.5]], id="fractional-truncated"),
            pytest.param([[-3.5, -0.5, 4.0, 3.0]], id="negative-clipped"),
            pytest.param([[8.0, 5.0, 20.0, 30.0]], id="edge-touching-and-beyond"),
            pytest.param([[12.0, 0.0, 15.0, 4.0]], id="outside-canvas"),
            pytest.param([[2.0, 2.0, 2.0, 6.0]], id="zero-width"),
            pytest.param(
                [[1.0, 1.0, 5.0, 5.0], [1.0, 1.0, 5.0, 5.0]], id="duplicate-boxes"
            ),
            pytest.param([[0.0, 0.0, 9.0, 8.0], [2.0, 2.0, 4.0, 4.0]], id="nested"),
            pytest.param(
                [[0.0, 0.0, 3.0, 8.0], [3.0, 0.0, 6.0, 8.0]], id="shared-edge"
            ),
        ],
    )
    def test_matches_exclusive_xyxy_to_mask_union(
        self, xyxy: list[list[float]]
    ) -> None:
        """One-canvas painting covers the pixels of per-box half-open masks."""
        boxes = np.array(xyxy, dtype=np.float32)
        expected = sv.xyxy_to_mask(boxes, (10, 8), "exclusive").any(axis=0)

        union = _boxes_union_mask(boxes, (10, 8))

        np.testing.assert_array_equal(union, expected)

    def test_non_finite_box_covers_nothing(self) -> None:
        """A box with a NaN or infinite edge adds no pixels instead of failing."""
        boxes = np.array([[0, 0, 5, 8], [np.nan, 0, 9, 8], [0, 0, np.inf, 8]])
        expected = sv.xyxy_to_mask(boxes[:1], (10, 8), "exclusive")[0]

        union = _boxes_union_mask(boxes, (10, 8))

        np.testing.assert_array_equal(union, expected)


class TestPolygonZoneOccupancyRatio:
    def test_sparse_compact_masks_do_not_decode_a_huge_union(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Distant sparse masks do not require decoding their full union rectangle."""
        compact = CompactMask(
            rles=[np.array([0, 1], dtype=np.int32)] * 2,
            crop_shapes=np.ones((2, 2), dtype=np.int32),
            offsets=np.array([[0, 0], [99_999, 99_999]], dtype=np.int32),
            image_shape=(100_000, 100_000),
        )
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 1, 1], [99_999, 99_999, 100_000, 100_000]]),
            mask=compact,
        )
        zone = sv.PolygonZone(np.array([[0, 0], [9, 0], [9, 9], [0, 9]]))
        original_crop = CompactMask.crop

        def bounded_crop(mask: CompactMask, index: int) -> npt.NDArray[np.bool_]:
            """Prevent the regression from allocating the huge union crop."""
            assert np.prod(mask._crop_shapes[index], dtype=np.int64) <= 10_000
            return original_crop(mask, index)

        monkeypatch.setattr(CompactMask, "crop", bounded_crop)

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(0.01)

    def test_compact_long_run_is_clipped_before_expansion(self) -> None:
        """A long foreground run only paints the part inside the zone."""
        compact = CompactMask(
            rles=[np.array([0, 100_000_000], dtype=np.int32)],
            crop_shapes=np.array([[1, 100_000_000]], dtype=np.int32),
            offsets=np.array([[0, 0]], dtype=np.int32),
            image_shape=(1, 100_000_000),
        )
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 100_000_000, 1]]), mask=compact
        )
        zone = sv.PolygonZone(np.array([[0, 0], [9, 0], [9, 9], [0, 9]]))

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(0.1)

    @pytest.mark.parametrize(
        ("corner", "dtype"),
        [
            pytest.param(2**32, np.int64, id="int64-wraps-to-zone"),
            pytest.param(2**31, np.float32, id="float32-rounded-limit"),
        ],
    )
    def test_rejects_oriented_coordinates_outside_int32(
        self, corner: int, dtype: type
    ) -> None:
        """Distant oriented vertices cannot wrap into the zone during rasterization."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        polygons: npt.NDArray[Any] = np.array(
            [
                [
                    [corner, corner],
                    [corner + 99, corner],
                    [corner + 99, corner + 99],
                    [corner, corner + 99],
                ]
            ],
            dtype=dtype,
        )
        detections = sv.Detections(
            xyxy=np.array([[corner, corner, corner + 99, corner + 99]]),
            data={ORIENTED_BOX_COORDINATES: polygons},
        )

        with pytest.raises(ValueError, match="int32"):
            zone.occupancy_ratio(detections)

    @pytest.mark.parametrize(
        ("xyxy", "expected"),
        [
            pytest.param([[0, 0, 49, 49]], 0.2401, id="half-open-box-pixels"),
            pytest.param(
                [[0, 0, 50, 100], [25, 0, 75, 100]], 0.75, id="overlap-counted-once"
            ),
            pytest.param([[-50, -50, 500, 500]], 1.0, id="oversized-box-clipped"),
            pytest.param([[200, 200, 300, 300]], 0.0, id="box-outside-zone"),
        ],
    )
    def test_boxes_cover_union_share_of_zone(
        self, xyxy: list[list[float]], expected: float
    ) -> None:
        """Axis-aligned boxes cover their half-open pixels, overlaps counted once."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        detections = sv.Detections(xyxy=np.array(xyxy, dtype=np.float32))

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(expected)

    def test_empty_detections_occupy_nothing(self) -> None:
        """No detections leave the zone empty instead of failing to build a union."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)

        ratio = zone.occupancy_ratio(sv.Detections.empty())

        assert ratio == 0.0

    @pytest.mark.parametrize(
        ("frame_hw", "regions", "expected"),
        [
            pytest.param(
                (300, 400),
                [
                    (slice(0, 100), slice(0, 50)),
                    (slice(0, 100), slice(25, 75)),
                    (slice(200, 250), slice(300, 350)),
                ],
                0.75,
                id="larger-frame-cropped-to-zone",
            ),
            pytest.param(
                (50, 50),
                [(slice(0, 50), slice(0, 50))],
                0.25,
                id="smaller-frame-padded",
            ),
        ],
    )
    def test_dense_masks_cover_union_share_of_zone(
        self,
        frame_hw: tuple[int, int],
        regions: list[tuple[slice, slice]],
        expected: float,
    ) -> None:
        """Dense masks of any frame size are unioned over the zone's own extent."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        detections = _masks_detections(frame_hw, regions)

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(expected)

    def test_compact_mask_is_pasted_at_its_offset(self) -> None:
        """A compact mask lands at its `(x, y)` offset, not at a swapped or zero one.

        The 800-pixel mask sits right of a wide zone's middle with a different `x` and
        `y` offset: swapping them would move it below the zone and report `0.0`.
        """
        zone = sv.PolygonZone(np.array([[0, 0], [199, 0], [199, 99], [0, 99]]))
        masks = np.zeros((1, 120, 240), dtype=bool)
        masks[0, 10:30, 150:190] = True
        xyxy = np.array([[140, 5, 200, 40]], dtype=np.float32)
        compact = CompactMask.from_dense(masks, xyxy, image_shape=(120, 240))
        detections = sv.Detections(xyxy=xyxy, mask=compact)

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(800 / 20_000)

    def test_overlapping_oriented_boxes_are_unioned(self) -> None:
        """Overlapping oriented boxes count their shared pixels once, not zero times.

        A single even-odd fill of both outlines would leave their intersection empty.
        """
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        corners = np.array(
            [
                [[0, 0], [49, 0], [49, 99], [0, 99]],
                [[25, 0], [74, 0], [74, 99], [25, 99]],
            ],
            dtype=np.float32,
        )
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 50, 100], [25, 0, 75, 100]], dtype=np.float32),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(0.75)

    def test_zone_mask_scaled_to_255_counts_pixels(self) -> None:
        """A 0/255 zone mask is measured in pixels, not in summed mask values."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        zone.mask = zone.mask.astype(np.uint8) * 255
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 50, 100], [25, 0, 75, 100]], dtype=np.float32)
        )

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(0.75)

    def test_leaves_current_count_unchanged(self) -> None:
        """Measuring occupancy does not overwrite the count of the last trigger."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        zone.trigger(sv.Detections(xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32)))

        zone.occupancy_ratio(sv.Detections.empty())

        assert zone.current_count == 1

    @pytest.mark.parametrize(
        ("polygon", "expected"),
        [
            pytest.param(OCCUPANCY_ZONE, 1.0, id="pixel-equal-zone-fully-covered"),
            pytest.param(
                np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
                10_000 / 10_201,
                id="vertex-equal-zone-misses-far-edges",
            ),
        ],
    )
    def test_box_convention_against_inclusive_zone(
        self, polygon: np.ndarray, expected: float
    ) -> None:
        """Half-open box `[0, 0, 100, 100]` fills exactly the 100 x 100 pixel zone.

        The zone includes its vertex pixels while boxes stay half-open, so a zone drawn
        on vertices 0..100 keeps its last row and column uncovered.
        """
        zone = sv.PolygonZone(polygon)
        detections = sv.Detections(xyxy=np.array([[0, 0, 100, 100]], dtype=float))

        ratio = zone.occupancy_ratio(detections)

        assert ratio == expected

    def test_non_boolean_mask_counts_nonzero_pixels(self) -> None:
        """A 0/255 detection mask covers the same pixels as its boolean twin."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        masks: npt.NDArray[Any] = np.zeros((2, 100, 100), dtype=np.uint8)
        masks[0, :, :50] = 255
        masks[1, :, 25:75] = 255
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 50, 100], [25, 0, 75, 100]], dtype=np.float32),
            mask=masks.astype(bool),
        )
        detections.mask = masks

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(0.75)

    def test_rejects_flat_oriented_box_corners(self) -> None:
        """Corners stored flat as `(N, 8)` raise a ValueError naming both shapes."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 50, 100]], dtype=np.float32),
            data={ORIENTED_BOX_COORDINATES: np.zeros((1, 8), dtype=np.float32)},
        )

        with pytest.raises(ValueError, match=r"shape \(1, K, 2\); got shape \(1, 8\)"):
            zone.occupancy_ratio(detections)

    def test_rejects_mask_reassigned_with_wrong_shape(self) -> None:
        """A mask swapped in after construction is re-checked before reduction."""
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        detections = _masks_detections((100, 100), [(slice(0, 100), slice(0, 50))])
        detections.mask = np.ones((100, 100), dtype=bool)

        with pytest.raises(
            ValueError,
            match=r"mask must have shape \(1, H, W\); got shape \(100, 100\)",
        ):
            zone.occupancy_ratio(detections)

    @pytest.mark.parametrize("detections", ZERO_AREA_ZONE_DETECTIONS)
    def test_rejects_zone_polygon_off_frame(self, detections: sv.Detections) -> None:
        """A polygon at negative coordinates leaves no zone pixels and raises.

        The zone mask still has a pixel, but the polygon misses it, so the ratio has no
        denominator; an empty frame must not hide that behind a `0.0`.
        """
        zone = sv.PolygonZone(np.array([[-10, -10], [-1, -10], [-1, -1], [-10, -1]]))

        with pytest.raises(ValueError, match="Zone mask covers no pixels"):
            zone.occupancy_ratio(detections)

    @pytest.mark.parametrize("detections", ZERO_AREA_ZONE_DETECTIONS)
    def test_rejects_zone_mask_without_pixels(self, detections: sv.Detections) -> None:
        """A frame-sized zone mask with no nonzero pixels raises instead of dividing.

        Reassigning an all-zero mask isolates the guard from polygon rasterization.
        """
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        zone.mask = np.zeros_like(zone.mask)

        with pytest.raises(ValueError, match=r"shape \(101, 101\) with no nonzero"):
            zone.occupancy_ratio(detections)

    def test_oriented_box_takes_precedence_over_enclosing_box(self) -> None:
        """A rotated box covers only its own body, not the axis-aligned box around it.

        The diamond's enclosing `xyxy` box covers the whole zone, so falling back to it
        would report full occupancy; the oriented corners must be rasterized instead,
        with the same rule as `polygon_to_mask`.
        """
        zone = sv.PolygonZone(OCCUPANCY_ZONE)
        diamond = np.array([[50, 0], [99, 50], [50, 99], [0, 50]], dtype=np.float32)
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
            data={ORIENTED_BOX_COORDINATES: diamond[np.newaxis]},
        )
        expected = np.count_nonzero(sv.polygon_to_mask(diamond, (101, 101))) / 10_000

        ratio = zone.occupancy_ratio(detections)

        assert ratio == pytest.approx(expected)
        assert ratio < 0.6
