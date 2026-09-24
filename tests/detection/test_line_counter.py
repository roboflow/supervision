from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision import (
    Detections,
    LineZone,
    LineZoneAnnotator,
    LineZoneAnnotatorMulticlass,
)
from supervision.draw.color import Color
from supervision.geometry.core import Point, Position, Vector
from tests.helpers import _create_detections


@pytest.mark.parametrize(
    ("vector", "expected_result", "exception"),
    [
        (
            Vector(start=Point(x=0.0, y=0.0), end=Point(x=0.0, y=0.0)),
            None,
            pytest.raises(ValueError, match="magnitude of the vector"),
        ),
        (
            Vector(start=Point(x=1.0, y=1.0), end=Point(x=1.0, y=1.0)),
            None,
            pytest.raises(ValueError, match="cannot be zero"),
        ),
        (
            Vector(start=Point(x=0.0, y=0.0), end=Point(x=0.0, y=4.0)),
            (
                Vector(start=Point(x=0.0, y=0.0), end=Point(x=-1.0, y=0.0)),
                Vector(start=Point(x=0.0, y=4.0), end=Point(x=1.0, y=4.0)),
            ),
            DoesNotRaise(),
        ),
        (
            Vector(Point(0.0, 0.0), Point(4.0, 0.0)),
            (
                Vector(start=Point(x=0.0, y=0.0), end=Point(x=0.0, y=1.0)),
                Vector(start=Point(x=4.0, y=0.0), end=Point(x=4.0, y=-1.0)),
            ),
            DoesNotRaise(),
        ),
        (
            Vector(Point(0.0, 0.0), Point(3.0, 4.0)),
            (
                Vector(start=Point(x=0, y=0), end=Point(x=-0.8, y=0.6)),
                Vector(start=Point(x=3, y=4), end=Point(x=3.8, y=3.4)),
            ),
            DoesNotRaise(),
        ),
        (
            Vector(Point(0.0, 0.0), Point(4.0, 3.0)),
            (
                Vector(start=Point(x=0, y=0), end=Point(x=-0.6, y=0.8)),
                Vector(start=Point(x=4, y=3), end=Point(x=4.6, y=2.2)),
            ),
            DoesNotRaise(),
        ),
        (
            Vector(Point(0.0, 0.0), Point(3.0, -4.0)),
            (
                Vector(start=Point(x=0, y=0), end=Point(x=0.8, y=0.6)),
                Vector(start=Point(x=3, y=-4), end=Point(x=2.2, y=-4.6)),
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_calculate_region_of_interest_limits(
    vector: Vector,
    expected_result: tuple[Vector, Vector] | None,
    exception: Exception,
) -> None:
    with exception:
        result = LineZone._calculate_region_of_interest_limits(vector=vector)
        assert result == expected_result


@pytest.mark.parametrize(
    ("vector", "xyxy_sequence", "expected_crossed_in", "expected_crossed_out"),
    [
        (  # Vertical line, simple crossing
            Vector(Point(0, 0), Point(0, 10)),
            [
                [4, 4, 6, 6],
                [4 - 10, 4, 6 - 10, 6],
                [4, 4, 6, 6],
                [4 - 10, 4, 6 - 10, 6],
            ],
            [False, False, True, False],
            [False, True, False, True],
        ),
        (  # Vertical line reversed, simple crossing
            Vector(Point(0, 10), Point(0, 0)),
            [
                [4, 4, 6, 6],
                [4 - 10, 4, 6 - 10, 6],
                [4, 4, 6, 6],
                [4 - 10, 4, 6 - 10, 6],
            ],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Horizontal line, simple crossing
            Vector(Point(0, 0), Point(10, 0)),
            [
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
            ],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Horizontal line reversed, simple crossing
            Vector(Point(10, 0), Point(0, 0)),
            [
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
            ],
            [False, False, True, False],
            [False, True, False, True],
        ),
        (  # Diagonal line, simple crossing
            Vector(Point(5, 0), Point(0, 5)),
            [
                [0, 0, 2, 2],
                [0 + 10, 0 + 10, 2 + 10, 2 + 10],
                [0, 0, 2, 2],
                [0 + 10, 0 + 10, 2 + 10, 2 + 10],
            ],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Crossing beside - right side
            Vector(Point(0, 0), Point(10, 0)),
            [
                [20, 4, 24, 6],
                [20, 4 - 10, 24, 6 - 10],
                [20, 4, 24, 6],
                [20, 4 - 10, 24, 6 - 10],
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # Horizontal line, simple crossing, far away
            Vector(Point(0, 0), Point(10, 0)),
            [
                [4, 1e32, 6, 1e32 + 2],
                [4, -1e32, 6, -1e32 + 2],
                [4, 1e32, 6, 1e32 + 2],
                [4, -1e32, 6, -1e32 + 2],
            ],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Crossing beside - left side
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-20, 4, -24, 6],
                [-20, 4 - 10, -24, 6 - 10],
                [-20, 4, -24, 6],
                [-20, 4 - 10, -24, 6 - 10],
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # Move above
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-4, 4, -2, 6],
                [-4 + 20, 4, -2 + 20, 6],
                [-4, 4, -2, 6],
                [-4 + 20, 4, -2 + 20, 6],
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # Move below
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-4, -6, -2, -4],
                [-4 + 20, -6, -2 + 20, -4],
                [-4, -6, -2, -4],
                [-4 + 20, -6, -2 + 20, -4],
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # Move into line partway
            Vector(Point(0, 0), Point(10, 0)),
            [
                [4, 4, 6, 6],
                [4 + 5, 4, 6 + 5, 6],
                [4, 4, 6, 6],
                [4 + 5, 4, 6 + 5, 6],
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # V-shaped crossing from outside limits - not supported.
            Vector(Point(0, 0), Point(10, 0)),
            [[-3, 6, -1, 8], [4, -6, 6, -4], [11, 6, 13, 8]],
            [False, False, False],
            [False, False, False],
        ),
        (  # Diagonal movement, from within limits to outside - not supported
            Vector(Point(0, 0), Point(10, 0)),
            [[4, 1, 6, 3], [11, 1 - 20, 13, 3 - 20]],
            [False, False],
            [False, False],
        ),
        (  # Diagonal movement, from outside limits to within - not supported
            Vector(Point(0, 0), Point(10, 0)),
            [
                [11, 21, 13, 23],
                [4, -3, 6, -1],
            ],
            [False, False],
            [False, False],
        ),
        (  # Diagonal crossing, from outside to outside limits - not supported.
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-4, 4, -2, 8],
                [-4 + 16, -4, -2 + 16, -6],
                [-4, 4, -2, 8],
                [-4 + 16, -4, -2 + 16, -6],
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
    ],
)
def test_line_zone_one_detection_default_anchors(
    vector: Vector,
    xyxy_sequence: list[list[float]],
    expected_crossed_in: list[bool],
    expected_crossed_out: list[bool],
) -> None:
    line_zone = LineZone(start=vector.start, end=vector.end)

    crossed_in_list = []
    crossed_out_list = []
    for i, bbox in enumerate(xyxy_sequence):
        detections = _create_detections(
            xyxy=[bbox],
            tracker_id=[0],
        )
        crossed_in, crossed_out = line_zone.trigger(detections)
        crossed_in_list.append(crossed_in[0])
        crossed_out_list.append(crossed_out[0])

    assert crossed_in_list == expected_crossed_in, (
        f"expected {expected_crossed_in}, got {crossed_in_list}"
    )
    assert crossed_out_list == expected_crossed_out, (
        f"expected {expected_crossed_out}, got {crossed_out_list}"
    )


@pytest.mark.parametrize(
    (
        "vector",
        "xyxy_sequence",
        "triggering_anchors",
        "expected_crossed_in",
        "expected_crossed_out",
    ),
    [
        (  # Scrape line, left side, corner anchors
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
            ],
            [
                Position.TOP_LEFT,
                Position.BOTTOM_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_RIGHT,
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # Scrape line, left side, right anchors
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
            ],
            [Position.TOP_RIGHT, Position.BOTTOM_RIGHT],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Scrape line, left side, center anchor (along line point)
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
            ],
            [Position.CENTER],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Scrape line, left side, center anchor (along line point)
            Vector(Point(0, 0), Point(10, 0)),
            [
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
                [-2, 4, 2, 6],
                [-2, 4 - 10, 2, 6 - 10],
                [-2, 4 - 10, 2, 6 - 10],
            ],
            [Position.CENTER],
            [False, True, False, True, False],
            [False, False, True, False, False],
        ),
        (  # Scrape line, right side, corner anchors
            Vector(Point(0, 0), Point(10, 0)),
            [
                [8, 4, 12, 6],
                [8, 4 - 10, 12, 6 - 10],
                [8, 4, 12, 6],
                [8, 4 - 10, 12, 6 - 10],
            ],
            [
                Position.TOP_LEFT,
                Position.BOTTOM_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_RIGHT,
            ],
            [False, False, False, False],
            [False, False, False, False],
        ),
        (  # Scrape line, right side, left anchors
            Vector(Point(0, 0), Point(10, 0)),
            [
                [8, 4, 12, 6],
                [8, 4 - 10, 12, 6 - 10],
                [8, 4, 12, 6],
                [8, 4 - 10, 12, 6 - 10],
            ],
            [Position.TOP_LEFT, Position.BOTTOM_LEFT],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Scrape line, right side, center anchor (along line point)
            Vector(Point(0, 0), Point(10, 0)),
            [
                [8, 4, 12, 6],
                [8, 4 - 10, 12, 6 - 10],
                [8, 4, 12, 6],
                [8, 4 - 10, 12, 6 - 10],
            ],
            [Position.CENTER],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Simple crossing, one anchor
            Vector(Point(0, 0), Point(10, 0)),
            [
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
            ],
            [Position.CENTER],
            [False, True, False, True],
            [False, False, True, False],
        ),
        (  # Simple crossing, all box anchors
            Vector(Point(0, 0), Point(10, 0)),
            [
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
                [4, 4, 6, 6],
                [4, 4 - 10, 6, 6 - 10],
            ],
            [
                Position.CENTER,
                Position.CENTER_LEFT,
                Position.CENTER_RIGHT,
                Position.TOP_CENTER,
                Position.TOP_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_LEFT,
                Position.BOTTOM_CENTER,
                Position.BOTTOM_RIGHT,
            ],
            [False, True, False, True],
            [False, False, True, False],
        ),
    ],
)
def test_line_zone_one_detection(
    vector: Vector,
    xyxy_sequence: list[list[float]],
    triggering_anchors: list[Position],
    expected_crossed_in: list[bool],
    expected_crossed_out: list[bool],
) -> None:
    line_zone = LineZone(
        start=vector.start, end=vector.end, triggering_anchors=triggering_anchors
    )

    crossed_in_list = []
    crossed_out_list = []
    for i, bbox in enumerate(xyxy_sequence):
        detections = _create_detections(
            xyxy=[bbox],
            tracker_id=[0],
        )
        crossed_in, crossed_out = line_zone.trigger(detections)
        crossed_in_list.append(crossed_in[0])
        crossed_out_list.append(crossed_out[0])

    assert crossed_in_list == expected_crossed_in, (
        f"expected {expected_crossed_in}, got {crossed_in_list}"
    )
    assert crossed_out_list == expected_crossed_out, (
        f"expected {expected_crossed_out}, got {crossed_out_list}"
    )


@pytest.mark.parametrize(
    (
        "vector",
        "xyxy_sequence",
        "anchors",
        "expected_crossed_in",
        "expected_crossed_out",
        "exception",
    ),
    [
        (  # One stays, one crosses
            Vector(Point(0, 0), Point(10, 0)),
            [
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
            ],
            [
                Position.TOP_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_LEFT,
                Position.BOTTOM_RIGHT,
            ],
            [[False, False], [False, True], [False, False], [False, True]],
            [[False, False], [False, False], [False, True], [False, False]],
            DoesNotRaise(),
        ),
        (  # Both cross at the same time
            Vector(Point(0, 0), Point(10, 0)),
            [
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4 - 10, 6, 6 - 10], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4 - 10, 6, 6 - 10], [4, 4 - 10, 6, 6 - 10]],
            ],
            [
                Position.TOP_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_LEFT,
                Position.BOTTOM_RIGHT,
            ],
            [[False, False], [True, True], [False, False], [True, True]],
            [[False, False], [False, False], [True, True], [False, False]],
            DoesNotRaise(),
        ),
    ],
)
def test_line_zone_multiple_detections(
    vector: Vector,
    xyxy_sequence: list[list[list[float]]],
    anchors: list[Position],
    expected_crossed_in: list[list[bool]],
    expected_crossed_out: list[list[bool]],
    exception: Exception,
) -> None:
    with exception:
        line_zone = LineZone(
            start=vector.start, end=vector.end, triggering_anchors=anchors
        )
        crossed_in_list = []
        crossed_out_list = []
        for bboxes in xyxy_sequence:
            detections = _create_detections(
                xyxy=bboxes,
                tracker_id=[i for i in range(0, len(bboxes))],
            )
            crossed_in, crossed_out = line_zone.trigger(detections)
            crossed_in_list.append(list(crossed_in))
            crossed_out_list.append(list(crossed_out))

        assert crossed_in_list == expected_crossed_in
        assert crossed_out_list == expected_crossed_out


@pytest.mark.parametrize(
    (
        "vector",
        "xyxy_sequence",
        "triggering_anchors",
        "minimum_crossing_threshold",
        "expected_crossed_in",
        "expected_crossed_out",
    ),
    [
        (  # Detection lingers around line, all crosses counted
            Vector(Point(0, 0), Point(10, 0)),
            [
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4 - 10, 3, 6],
            ],
            [
                Position.TOP_LEFT,
            ],
            1,
            [False, True, False, True, False],
            [False, False, True, False, False],
        ),
        (  # Detection lingers around line, only final cross counted
            Vector(Point(0, 0), Point(10, 0)),
            [
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4 - 10, 3, 6],
            ],
            [
                Position.TOP_LEFT,
            ],
            2,
            [False, False, False, False, True],
            [False, False, False, False, False],
        ),
        (  # Detection lingers around line for a long time
            Vector(Point(0, 0), Point(10, 0)),
            [
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4 - 10, 3, 6],
            ],
            [
                Position.TOP_LEFT,
            ],
            2,
            [False] * 12 + [True],
            [False] * 13,
        ),
        (  # Detection lingers around line, longer cycle
            Vector(Point(0, 0), Point(10, 0)),
            [
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4, 3, 6],
                [2, 4, 3, 6],
                [2, 4, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4 - 10, 3, 6],
                [2, 4 - 10, 3, 6],
            ],
            [
                Position.TOP_LEFT,
            ],
            4,
            [False] * 8 + [True],
            [False] * 9,
        ),
    ],
)
def test_line_zone_one_detection_long_horizon(
    vector: Vector,
    xyxy_sequence: list[list[float]],
    triggering_anchors: list[Position],
    minimum_crossing_threshold: int,
    expected_crossed_in: list[bool],
    expected_crossed_out: list[bool],
) -> None:
    line_zone = LineZone(
        start=vector.start,
        end=vector.end,
        triggering_anchors=triggering_anchors,
        minimum_crossing_threshold=minimum_crossing_threshold,
    )

    crossed_in_list = []
    crossed_out_list = []
    for i, bbox in enumerate(xyxy_sequence):
        detections = _create_detections(
            xyxy=[bbox],
            tracker_id=[0],
        )
        crossed_in, crossed_out = line_zone.trigger(detections)
        crossed_in_list.append(crossed_in[0])
        crossed_out_list.append(crossed_out[0])

    assert crossed_in_list == expected_crossed_in, (
        f"expected {expected_crossed_in}, got {crossed_in_list}"
    )
    assert crossed_out_list == expected_crossed_out, (
        f"expected {expected_crossed_out}, got {crossed_out_list}"
    )


_ABOVE_LINE_BOX = [2.0, -6.0, 3.0, -4.0]
_BELOW_LINE_BOX = [2.0, 4.0, 3.0, 6.0]
_STRADDLING_LINE_BOX = [2.0, -1.0, 3.0, 1.0]
_SIDE_TO_BOX = {"A": _ABOVE_LINE_BOX, "B": _BELOW_LINE_BOX, "S": _STRADDLING_LINE_BOX}


def _boxes_from_sides(sides: str) -> list[list[float]]:
    """Map a side string like `"AABSAA"` to boxes above, below, or straddling `y=0`.

    `"A"` is above the line, `"B"` is below it, and `"S"` straddles both sides at once
    (top corners above, bottom corners below) — one character encodes a single frame of
    a tracker's trajectory and the whole string encodes the sequence.
    """
    return [_SIDE_TO_BOX[side] for side in sides]


class TestLineZoneSubThresholdFlicker:
    """Sub-threshold side changes must not be counted as crossings (#2598)."""

    @staticmethod
    def _line_zone(minimum_crossing_threshold: int) -> LineZone:
        """Build a horizontal line zone driven by the TOP_LEFT anchor alone."""
        return LineZone(
            start=Point(0, 0),
            end=Point(10, 0),
            triggering_anchors=[Position.TOP_LEFT],
            minimum_crossing_threshold=minimum_crossing_threshold,
        )

    @pytest.mark.parametrize(
        ("sides", "minimum_crossing_threshold", "expected_counts"),
        [
            pytest.param("AAAAAA", 2, (0, 0), id="never-leaves-side"),
            pytest.param("AAABAAA", 2, (0, 0), id="one-frame-flicker"),
            pytest.param("AAABAAABAAABAAA", 2, (0, 0), id="three-separated-flickers"),
            pytest.param("AAABBAAA", 3, (0, 0), id="two-frame-flicker-under-threshold"),
            pytest.param("AAABBB", 2, (0, 1), id="sustained-crossing"),
            pytest.param("AAABBBAAA", 2, (1, 1), id="crossing-then-return"),
            pytest.param("AAABAAABBB", 2, (0, 1), id="crossing-after-flicker"),
            pytest.param("AB", 1, (0, 1), id="threshold-one-unchanged"),
            pytest.param("BBBABBB", 2, (0, 0), id="one-frame-flicker-from-other-side"),
            pytest.param("AAABBB", 3, (0, 1), id="sustained-crossing-threshold-three"),
        ],
    )
    def test_counts_only_sustained_side_changes(
        self,
        sides: str,
        minimum_crossing_threshold: int,
        expected_counts: tuple[int, int],
    ) -> None:
        """An excursion shorter than the threshold is noise, not a crossing."""
        line_zone = self._line_zone(minimum_crossing_threshold)

        for box in _boxes_from_sides(sides):
            line_zone.trigger(_create_detections(xyxy=[box], tracker_id=[0]))

        assert (line_zone.in_count, line_zone.out_count) == expected_counts

    def test_straddling_frame_is_skipped_with_default_anchors(self) -> None:
        """A straddling box is invisible to the sustain-window gate, not a flicker.

        With the default four-corner anchors, a box that straddles the line makes
        `_compute_anchor_sides` report both `has_any_left_trigger` and
        `has_any_right_trigger` as True for that frame, so it is skipped before ever
        reaching the crossing history (the ambiguous-straddle guard). A straddle
        sandwiched inside a sustained crossing must therefore count exactly as if
        that frame were omitted from the sequence.
        """
        line_zone = LineZone(
            start=Point(0, 0), end=Point(10, 0), minimum_crossing_threshold=2
        )

        for box in _boxes_from_sides("AAASBBB"):
            line_zone.trigger(_create_detections(xyxy=[box], tracker_id=[0]))

        assert (line_zone.in_count, line_zone.out_count) == (0, 1)

    def test_flicker_does_not_leak_between_trackers(self) -> None:
        """One tracker's flicker must not disturb another tracker's real crossing."""
        line_zone = self._line_zone(minimum_crossing_threshold=2)
        flickering_boxes = _boxes_from_sides("AAABAAA")
        crossing_boxes = _boxes_from_sides("AAABBBB")

        for flickering_box, crossing_box in zip(
            flickering_boxes, crossing_boxes, strict=True
        ):
            line_zone.trigger(
                _create_detections(
                    xyxy=[flickering_box, crossing_box], tracker_id=[1, 2]
                )
            )

        assert (line_zone.in_count, line_zone.out_count) == (0, 1)

    def test_dropped_frame_inside_excursion_is_coasted_through(self) -> None:
        """A single dropped frame mid-excursion is coasted through, not reset.

        The sustain-window deque records only observed sides, with no notion of a gap,
        so a flicker excursion that straddles one dropped frame (the tracker entirely
        absent from that frame's detections, per the coasting-tolerance eviction design)
        is judged purely on the sides recorded either side of the gap. This documents
        the resulting current behavior — one still-spurious crossing from the flicker,
        plus the eventual genuine crossing — rather than asserting it is the intended
        fix.
        """
        line_zone = self._line_zone(minimum_crossing_threshold=2)

        for box in _boxes_from_sides("AAAB"):
            line_zone.trigger(_create_detections(xyxy=[box], tracker_id=[0]))
        line_zone.trigger(Detections.empty())
        for box in _boxes_from_sides("BAAA"):
            line_zone.trigger(_create_detections(xyxy=[box], tracker_id=[0]))

        assert (line_zone.in_count, line_zone.out_count) == (1, 1)

    def test_concurrent_flicker_does_not_leak_between_trackers(self) -> None:
        """Two independently flickering trackers must not cross-contaminate counts.

        Companion to `test_flicker_does_not_leak_between_trackers`, which pairs a
        flicker with a real crossing; here both trackers are noisy at once, with
        different flicker shapes (one single-frame excursion vs. two), to confirm
        per-tracker isolation holds under concurrent noise, not only under a
        concurrent real crossing.
        """
        line_zone = self._line_zone(minimum_crossing_threshold=2)
        single_flicker_boxes = _boxes_from_sides("AAABAAA")
        double_flicker_boxes = _boxes_from_sides("AABAABA")

        for single_box, double_box in zip(
            single_flicker_boxes, double_flicker_boxes, strict=True
        ):
            line_zone.trigger(
                _create_detections(xyxy=[single_box, double_box], tracker_id=[1, 2])
            )

        assert (line_zone.in_count, line_zone.out_count) == (0, 0)

    def test_evicted_tracker_does_not_inherit_confirmed_side(self) -> None:
        """Eviction clears the reference side, so a reused tracker ID starts fresh."""
        line_zone = self._line_zone(minimum_crossing_threshold=2)
        for box in _boxes_from_sides("BBB"):
            line_zone.trigger(_create_detections(xyxy=[box], tracker_id=[0]))
        for _ in range(line_zone.crossing_history_length):
            line_zone.trigger(Detections.empty())

        for box in _boxes_from_sides("AAA"):
            line_zone.trigger(_create_detections(xyxy=[box], tracker_id=[0]))

        assert (line_zone.in_count, line_zone.out_count) == (0, 0)


@pytest.mark.parametrize(
    (
        "vector",
        "xyxy_sequence",
        "anchors",
        "minimum_crossing_threshold",
        "expected_crossed_in",
        "expected_crossed_out",
        "expected_count_in",
        "expected_count_out",
        "exception",
    ),
    [
        (  # One stays, one crosses, one disappears before crossing
            Vector(Point(0, 0), Point(10, 0)),
            [
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
            ],
            [
                Position.TOP_LEFT,
            ],
            1,
            [
                [False, False, False],
                [False, True, False],
                [False, False],
                [False, True],
                [False, False],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, True],
                [False, False],
                [False, False],
            ],
            [0, 1, 1, 2, 2],
            [0, 0, 1, 1, 1],
            DoesNotRaise(),
        ),
        (  # One stays, one crosses, one disappears immediately after crossing
            Vector(Point(0, 0), Point(10, 0)),
            [
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
            ],
            [
                Position.TOP_LEFT,
            ],
            1,
            [
                [False, False, False],
                [False, True, False],
                [False, False, True],
                [False, True],
                [False, False],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, True, False],
                [False, False],
                [False, False],
            ],
            [0, 1, 2, 3, 3],
            [0, 0, 1, 1, 1],
            DoesNotRaise(),
        ),
        (  # One stays, one crosses, one disappears before crossing
            Vector(Point(0, 0), Point(10, 0)),
            [
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
            ],
            [
                Position.TOP_LEFT,
            ],
            2,
            [
                [False, False, False],
                [False, False, False],
                [False, False],
                [False, False],
                [False, True],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False],
                [False, False],
                [False, False],
            ],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            DoesNotRaise(),
        ),
        (  # One stays, one crosses, one disappears immediately after crossing
            Vector(Point(0, 0), Point(10, 0)),
            [
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10], [4, 4, 6, 6]],
                [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
                [[4, 4, 6, 6], [4, 4 - 10, 6, 6 - 10]],
            ],
            [
                Position.TOP_LEFT,
            ],
            2,
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False],
                [False, True],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False],
                [False, False],
            ],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            DoesNotRaise(),
        ),
    ],
)
def test_line_zone_long_horizon_disappearing_detections(
    vector: Vector,
    xyxy_sequence: list[list[list[float] | None]],
    anchors: list[Position],
    minimum_crossing_threshold: int,
    expected_crossed_in: list[list[bool]],
    expected_crossed_out: list[list[bool]],
    expected_count_in: list[int],
    expected_count_out: list[int],
    exception: Exception,
) -> None:
    with exception:
        line_zone = LineZone(
            start=vector.start,
            end=vector.end,
            triggering_anchors=anchors,
            minimum_crossing_threshold=minimum_crossing_threshold,
        )
        crossed_in_list = []
        crossed_out_list = []
        count_in_list = []
        count_out_list = []
        for bboxes in xyxy_sequence:
            detections = _create_detections(
                xyxy=bboxes,
                tracker_id=[i for i in range(0, len(bboxes))],
            )
            crossed_in, crossed_out = line_zone.trigger(detections)
            crossed_in_list.append(list(crossed_in))
            crossed_out_list.append(list(crossed_out))
            count_in_list.append(line_zone.in_count)
            count_out_list.append(line_zone.out_count)

        assert crossed_in_list == expected_crossed_in
        assert crossed_out_list == expected_crossed_out
        assert count_in_list == expected_count_in
        assert count_out_list == expected_count_out


@pytest.mark.parametrize(
    (
        "xyxy_sequence",
        "tracker_id_sequence",
        "class_id_sequence",
        "expected_in_count_per_class",
        "expected_out_count_per_class",
    ),
    [
        pytest.param(
            [
                [[4, 4, 6, 6]],  # frame 0: object 0, class 0, position 4,4,6,6
                [[4, -6, 6, -4]],  # frame 1: object 0, class 0, position 4,-6,6,-4
                [[4, 4, 6, 6]],  # frame 2: object 0, class 0, position 4,4,6,6
                [[4, 4, 6, 6]],  # frame 3: object 0, class 1, position 4,4,6,6
                [[4, -6, 6, -4]],  # frame 4: object 0, class 1, position 4,-6,6,-4
                [[4, 4, 6, 6]],  # frame 5: object 0, class 1, position 4,4,6,6
            ],
            # tracker_id_sequence
            [[0], [0], [0], [0], [0], [0]],
            # class_id_sequence
            [[0], [0], [0], [1], [1], [1]],
            # expected_in_count_per_class
            {0: 1, 1: 1},
            # expected_out_count_per_class
            {0: 1, 1: 1},
            id="single_object_tracker_id_reuse_with_different_classes",
        ),
        pytest.param(
            [
                # frame 0: objects 0&1 cross IN
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                # frame 1
                [[4, -6, 6, -4], [4, -6, 6, -4]],
                # frame 2: objects 0&1 cross OUT
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                # frame 3: objects 2&3 cross IN
                [[4, 4, 6, 6], [4, 4, 6, 6]],
                # frame 4
                [[4, -6, 6, -4], [4, -6, 6, -4]],
                # frame 5: objects 2&3 cross OUT
                [[4, 4, 6, 6], [4, 4, 6, 6]],
            ],
            # tracker_id_sequence
            [[0, 1], [0, 1], [0, 1], [2, 3], [2, 3], [2, 3]],
            # class_id_sequence
            [[0, 1], [0, 1], [0, 1], [4, 5], [4, 5], [4, 5]],
            # expected_in_count_per_class
            {0: 1, 1: 1, 4: 1, 5: 1},
            # expected_out_count_per_class
            {0: 1, 1: 1, 4: 1, 5: 1},
            id="multiple_objects_tracker_id_reuse_with_different_classes",
        ),
    ],
)
def test_line_zone_tracker_id_reuse_with_different_classes(
    xyxy_sequence: list[list[list[float]]],
    tracker_id_sequence: list[list[int]],
    class_id_sequence: list[list[int]],
    expected_in_count_per_class: dict[int, int],
    expected_out_count_per_class: dict[int, int],
) -> None:
    line_zone = LineZone(start=Point(0, 0), end=Point(10, 0))

    for xyxy, tracker_id, class_id in zip(
        xyxy_sequence, tracker_id_sequence, class_id_sequence
    ):
        detections = _create_detections(
            xyxy=xyxy, tracker_id=tracker_id, class_id=class_id
        )
        line_zone.trigger(detections)

    assert line_zone.in_count_per_class == expected_in_count_per_class
    assert line_zone.out_count_per_class == expected_out_count_per_class


def test_line_zone_trigger_does_not_call_np_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against reintroducing np.cross, deprecated for 2-D input in NumPy 2.0."""

    def _raise(*args, **kwargs):
        raise AssertionError("np.cross must not be called on 2-D vectors")

    monkeypatch.setattr(np, "cross", _raise)

    line_zone = LineZone(start=Point(0, 0), end=Point(0, 10))
    for xyxy in [[4, 4, 6, 6], [-6, 4, -4, 6]]:
        detections = _create_detections(xyxy=[xyxy], tracker_id=[0])
        crossed_in, crossed_out = line_zone.trigger(detections)

    assert not crossed_in[0]
    assert crossed_out[0]
    assert line_zone.out_count == 1


def test_line_zone_skips_unconfirmed_tracks() -> None:
    """Detections with a negative tracker_id (unconfirmed tracks) are ignored.

    Regression guard for #2578: ByteTrack-style trackers return -1 for tracks that have
    not yet been confirmed, and every -1 was previously collapsed into one shared track,
    silently inflating crossing counts.
    """
    line_zone = LineZone(start=Point(0, 100), end=Point(200, 100))

    for y in (20, 160, 20, 160, 20, 160):
        detections = _create_detections(
            xyxy=[[80.0, y, 120.0, y + 40]], tracker_id=[-1]
        )
        line_zone.trigger(detections)

    assert (line_zone.in_count, line_zone.out_count) == (0, 0)


@pytest.mark.parametrize(
    "make_anchors",
    [
        pytest.param(
            lambda: (
                anchor
                for anchor in (
                    Position.TOP_LEFT,
                    Position.TOP_RIGHT,
                    Position.BOTTOM_LEFT,
                    Position.BOTTOM_RIGHT,
                )
            ),
            id="generator",
        ),
        pytest.param(
            lambda: map(
                lambda name: Position[name],
                ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"],
            ),
            id="map",
        ),
        pytest.param(
            lambda: filter(
                lambda _: True,
                (
                    Position.TOP_LEFT,
                    Position.TOP_RIGHT,
                    Position.BOTTOM_LEFT,
                    Position.BOTTOM_RIGHT,
                ),
            ),
            id="filter",
        ),
    ],
)
def test_line_zone_counts_crossing_with_single_pass_triggering_anchors(
    make_anchors,
) -> None:
    """A single-pass anchors iterable survives construction and counts a crossing.

    Covers the three single-pass iterable shapes named in the PR #2561 bug report
    — generator expression, `map`, and `filter` — none of which must be exhausted
    by the constructor's emptiness check before `trigger()` iterates them.
    """
    line_zone = LineZone(
        start=Point(0, 100), end=Point(200, 100), triggering_anchors=make_anchors()
    )

    line_zone.trigger(_create_detections(xyxy=[[10, 110, 20, 150]], tracker_id=[1]))
    line_zone.trigger(_create_detections(xyxy=[[10, 50, 20, 90]], tracker_id=[1]))

    assert (line_zone.in_count, line_zone.out_count) == (1, 0)


def test_line_zone_trigger_evicts_stale_crossing_history() -> None:
    """History for tracker IDs absent from the current frame is evicted."""
    line_zone = LineZone(start=Point(0, 0), end=Point(10, 0))
    first_detections = _create_detections(
        xyxy=[[4, 4, 6, 6]], tracker_id=[0], class_id=[1]
    )
    second_detections = _create_detections(
        xyxy=[[4, 4, 6, 6]], tracker_id=[1], class_id=[2]
    )

    line_zone.trigger(first_detections)
    # Trigger twice with second_detections so tracker_id=0 accumulates
    # crossing_history_length absent frames (default=2) and is evicted.
    line_zone.trigger(second_detections)
    line_zone.trigger(second_detections)

    assert set(line_zone.crossing_state_history) == {1}


def test_line_zone_trigger_evicts_stale_crossing_history_on_empty_frames() -> None:
    """Empty frames age out tracker crossing history."""
    line_zone = LineZone(start=Point(0, 0), end=Point(10, 0))
    detections = _create_detections(xyxy=[[4, 4, 6, 6]], tracker_id=[0], class_id=[1])

    line_zone.trigger(detections)
    for _ in range(line_zone.crossing_history_length):
        line_zone.trigger(Detections.empty())

    assert not line_zone.crossing_state_history


def test_line_zone_trigger_evicts_stale_crossing_history_on_class_change() -> None:
    """Class changes must not split a tracker crossing history."""
    line_zone = LineZone(start=Point(0, 0), end=Point(10, 0))
    first_detections = _create_detections(
        xyxy=[[4, 4, 6, 6]], tracker_id=[0], class_id=[1]
    )
    second_detections = _create_detections(
        xyxy=[[4, 4, 6, 6]], tracker_id=[0], class_id=[2]
    )

    line_zone.trigger(first_detections)
    for _ in range(line_zone.crossing_history_length):
        line_zone.trigger(second_detections)

    assert set(line_zone.crossing_state_history) == {0}


def test_line_zone_class_flicker_keeps_crossing_counts_continuous() -> None:
    """A tracker's class change must not suppress a real crossing."""
    line_zone = LineZone(start=Point(0, 0), end=Point(10, 0))
    detections_sequence = [
        _create_detections(xyxy=[[4, 4, 6, 6]], tracker_id=[0], class_id=[0]),
        _create_detections(xyxy=[[4, -6, 6, -4]], tracker_id=[0], class_id=[1]),
        _create_detections(xyxy=[[4, -6, 6, -4]], tracker_id=[0], class_id=[1]),
        _create_detections(xyxy=[[4, 4, 6, 6]], tracker_id=[0], class_id=[1]),
    ]

    crossed_in = []
    crossed_out = []
    for detections in detections_sequence:
        crossed_in_frame, crossed_out_frame = line_zone.trigger(detections)
        crossed_in.append(bool(crossed_in_frame[0]))
        crossed_out.append(bool(crossed_out_frame[0]))

    assert crossed_in == [False, True, False, False]
    assert crossed_out == [False, False, False, True]
    assert line_zone.in_count_per_class == {1: 1}
    assert line_zone.out_count_per_class == {1: 1}
    assert set(line_zone.crossing_state_history) == {0}


def test_line_zone_annotator_multiclass_supports_none_class_id() -> None:
    line_zone = LineZone(start=Point(0, 0), end=Point(0, 10))
    for xyxy in [[4, 4, 6, 6], [-6, 4, -4, 6]]:
        detections = _create_detections(xyxy=[xyxy], tracker_id=[0])
        line_zone.trigger(detections)

    assert line_zone.out_count_per_class == {None: 1}

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotator = LineZoneAnnotatorMulticlass(force_draw_class_ids=False)
    annotated_frame = annotator.annotate(frame=frame.copy(), line_zones=[line_zone])

    assert annotated_frame.shape == frame.shape
    assert not np.array_equal(annotated_frame, frame)


class TestLineZoneInit:
    @pytest.mark.parametrize(
        ("triggering_anchors", "exception"),
        [
            pytest.param([Position.CENTER], DoesNotRaise(), id="non-empty-list"),
            pytest.param(
                [],
                pytest.raises(ValueError, match="Triggering anchors cannot be empty"),
                id="empty-list",
            ),
            pytest.param(
                (anchor for anchor in []),
                pytest.raises(ValueError, match="Triggering anchors cannot be empty"),
                id="empty-generator",
            ),
        ],
    )
    def test_empty_anchors_raises(self, triggering_anchors, exception) -> None:
        """LineZone rejects an anchors iterable that is empty, generator included.

        Mirrors `TestPolygonZoneInit::test_empty_anchors_raises`. The empty-generator
        case is the regression guard for #2561: the constructor must materialize a
        single-pass iterable before checking it, or an empty generator would reach
        the check unconsumed and slip past it.
        """
        with exception:
            LineZone(
                start=Point(0, 0),
                end=Point(10, 0),
                triggering_anchors=triggering_anchors,
            )


def test_line_zone_label_rotation_uses_pillow_canvas() -> None:
    """Render a rotated BGRA count label through the domain-specific Pillow path."""
    upright = LineZoneAnnotator._make_label_image(
        "out: 7",
        text_scale=0.75,
        text_thickness=1,
        text_padding=4,
        text_color=Color.WHITE,
        text_box_show=True,
        text_box_color=Color.BLACK,
        line_angle_degrees=0,
    )
    label = LineZoneAnnotator._make_label_image(
        "out: 7",
        text_scale=0.75,
        text_thickness=1,
        text_padding=4,
        text_color=Color.WHITE,
        text_box_show=True,
        text_box_color=Color.BLACK,
        line_angle_degrees=37,
    )

    assert label.ndim == 3
    assert label.shape[2] == 4
    assert np.any(label[..., 3])
    assert not np.array_equal(label[..., 3], upright[..., 3])
