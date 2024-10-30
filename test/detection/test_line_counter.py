from contextlib import ExitStack as DoesNotRaise
from typing import List, Optional, Tuple

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
    expected_result: Optional[Tuple[Vector, Vector]],
    exception: Exception,
) -> None:
    with exception:
        result = LineZone.calculate_region_of_interest_limits(vector=vector)
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
    xyxy_sequence: List[List[float]],
    expected_crossed_in: List[bool],
    expected_crossed_out: List[bool],
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

    assert (
        crossed_in_list == expected_crossed_in
    ), f"expected {expected_crossed_in}, got {crossed_in_list}"
    assert (
        crossed_out_list == expected_crossed_out
    ), f"expected {expected_crossed_out}, got {crossed_out_list}"


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
    xyxy_sequence: List[List[float]],
    triggering_anchors: List[Position],
    expected_crossed_in: List[bool],
    expected_crossed_out: List[bool],
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

    assert (
        crossed_in_list == expected_crossed_in
    ), f"expected {expected_crossed_in}, got {crossed_in_list}"
    assert (
        crossed_out_list == expected_crossed_out
    ), f"expected {expected_crossed_out}, got {crossed_out_list}"


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
    xyxy_sequence: List[List[List[float]]],
    anchors: List[Position],
    expected_crossed_in: List[List[bool]],
    expected_crossed_out: List[List[bool]],
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
