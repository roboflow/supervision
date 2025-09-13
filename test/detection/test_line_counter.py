from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import pytest

from supervision import LineZone
from supervision.geometry.core import Point, Position, Vector
from test.test_utils import mock_detections


@pytest.mark.parametrize(
    "vector, expected_result, exception",
    [
        (
            Vector(start=Point(x=0.0, y=0.0), end=Point(x=0.0, y=0.0)),
            None,
            pytest.raises(ValueError),
        ),
        (
            Vector(start=Point(x=1.0, y=1.0), end=Point(x=1.0, y=1.0)),
            None,
            pytest.raises(ValueError),
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
    "vector, xyxy_sequence, expected_crossed_in, expected_crossed_out",
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
        detections = mock_detections(
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
    "vector, xyxy_sequence, triggering_anchors, expected_crossed_in, "
    "expected_crossed_out",
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
        detections = mock_detections(
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
    "vector, xyxy_sequence, anchors, expected_crossed_in, "
    "expected_crossed_out, exception",
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
            detections = mock_detections(
                xyxy=bboxes,
                tracker_id=[i for i in range(0, len(bboxes))],
            )
            crossed_in, crossed_out = line_zone.trigger(detections)
            crossed_in_list.append(list(crossed_in))
            crossed_out_list.append(list(crossed_out))

        assert crossed_in_list == expected_crossed_in
        assert crossed_out_list == expected_crossed_out


@pytest.mark.parametrize(
    "vector, xyxy_sequence, triggering_anchors, minimum_crossing_threshold, "
    "expected_crossed_in, expected_crossed_out",
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
        detections = mock_detections(
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
    "vector, xyxy_sequence, anchors, minimum_crossing_threshold, "
    "expected_crossed_in, expected_crossed_out, expected_count_in, "
    "expected_count_out, exception",
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
            detections = mock_detections(
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
    "vector, test_sequence, expected_crossed_in, expected_crossed_out, "
    "expected_in_count_per_class, expected_out_count_per_class",
    [
        (  # Object changes class during tracking
            Vector(Point(0, 0), Point(10, 0)),
            [
                # Frame 1: Object on right side, class 0
                {"xyxy": [[4, 4, 6, 6]], "tracker_id": [1], "class_id": [0]},
                # Frame 2: Object crosses to left side, still class 0
                {"xyxy": [[4, -6, 6, -4]], "tracker_id": [1], "class_id": [0]},
                # Frame 3: Object stays on left side, class changes to 1
                {"xyxy": [[4, -6, 6, -4]], "tracker_id": [1], "class_id": [1]},
                # Frame 4: Object crosses back to right side, still class 1
                {"xyxy": [[4, 4, 6, 6]], "tracker_id": [1], "class_id": [1]},
            ],
            [False, True, False, False],
            [False, False, False, True],
            {0: 1, 1: 0},  # Class 0 crossed in once, class 1 never crossed in
            {0: 0, 1: 1},  # Class 0 never crossed out, class 1 crossed out once
        ),
        (  # Multiple objects with different classes
            Vector(Point(0, 0), Point(10, 0)),
            [
                # Frame 1: Both objects on right side
                {
                    "xyxy": [[2, 4, 4, 6], [6, 4, 8, 6]], 
                    "tracker_id": [1, 2], 
                    "class_id": [0, 1]
                },
                # Frame 2: Both objects cross to left side
                {
                    "xyxy": [[2, -6, 4, -4], [6, -6, 8, -4]], 
                    "tracker_id": [1, 2], 
                    "class_id": [0, 1]
                },
            ],
            [[False, False], [True, True]],
            [[False, False], [False, False]],
            {0: 1, 1: 1},  # Both classes crossed in once
            {0: 0, 1: 0},  # Neither class crossed out
        ),
    ],
)
def test_line_zone_class_id_tracking(
    vector: Vector,
    test_sequence: list[dict],
    expected_crossed_in: list[list[bool]],
    expected_crossed_out: list[list[bool]],
    expected_in_count_per_class: dict[int, int],
    expected_out_count_per_class: dict[int, int],
) -> None:
    """Test that LineZone properly handles class_id changes during tracking."""
    line_zone = LineZone(
        start=vector.start, 
        end=vector.end,
        triggering_anchors=[Position.CENTER]
    )
    
    crossed_in_results = []
    crossed_out_results = []
    
    for frame in test_sequence:
        detections = mock_detections(**frame)
        crossed_in, crossed_out = line_zone.trigger(detections)
        crossed_in_results.append(list(crossed_in))
        crossed_out_results.append(list(crossed_out))
    
    assert crossed_in_results == expected_crossed_in
    assert crossed_out_results == expected_crossed_out
    
    # Check per-class counts
    for class_id, expected_count in expected_in_count_per_class.items():
        assert line_zone.in_count_per_class.get(class_id, 0) == expected_count
    
    for class_id, expected_count in expected_out_count_per_class.items():
        assert line_zone.out_count_per_class.get(class_id, 0) == expected_count


@pytest.mark.parametrize(
    "vector, test_sequence, minimum_crossing_threshold, expected_final_counts",
    [
        (  # Class change with minimum threshold
            Vector(Point(0, 0), Point(10, 0)),
            [
                # Frames 1-3: Object oscillates on right side, class 0
                {"xyxy": [[4, 4, 6, 6]], "tracker_id": [1], "class_id": [0]},
                {"xyxy": [[4, -2, 6, 0]], "tracker_id": [1], "class_id": [0]},
                {"xyxy": [[4, 4, 6, 6]], "tracker_id": [1], "class_id": [0]},
                # Frame 4: Class changes to 1, still on right side
                {"xyxy": [[4, 4, 6, 6]], "tracker_id": [1], "class_id": [1]},
                # Frames 5-7: Object crosses to left side as class 1
                {"xyxy": [[4, -6, 6, -4]], "tracker_id": [1], "class_id": [1]},
                {"xyxy": [[4, -6, 6, -4]], "tracker_id": [1], "class_id": [1]},
                {"xyxy": [[4, -6, 6, -4]], "tracker_id": [1], "class_id": [1]},
            ],
            2,
            {0: 0, 1: 1},  # Only class 1 should have crossed after threshold met
        ),
    ],
)
def test_line_zone_class_id_with_minimum_threshold(
    vector: Vector,
    test_sequence: list[dict],
    minimum_crossing_threshold: int,
    expected_final_counts: dict[int, int],
) -> None:
    """Test class_id changes work correctly with minimum crossing threshold."""
    line_zone = LineZone(
        start=vector.start, 
        end=vector.end,
        triggering_anchors=[Position.CENTER],
        minimum_crossing_threshold=minimum_crossing_threshold
    )
    
    for frame in test_sequence:
        detections = mock_detections(**frame)
        line_zone.trigger(detections)
    
    # Check final counts
    for class_id, expected_count in expected_final_counts.items():
        assert line_zone.in_count_per_class.get(class_id, 0) == expected_count


def test_line_zone_class_id_none_handling() -> None:
    """Test LineZone handles None class_id values correctly."""
    line_zone = LineZone(
        start=Point(0, 0), 
        end=Point(10, 0),
        triggering_anchors=[Position.CENTER]
    )
    
    # Object with None class_id crosses the line
    test_sequence = [
        # Frame 1: Object on right side, no class_id
        {"xyxy": [[4, 4, 6, 6]], "tracker_id": [1], "class_id": None},
        # Frame 2: Object crosses to left side, still no class_id
        {"xyxy": [[4, -6, 6, -4]], "tracker_id": [1], "class_id": None},
    ]
    
    for frame in test_sequence:
        detections = mock_detections(**frame)
        line_zone.trigger(detections)
    
    # Should work with None class_id
    assert line_zone.in_count_per_class[None] == 1
    assert line_zone.in_count == 1