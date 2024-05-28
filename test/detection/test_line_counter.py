from contextlib import ExitStack as DoesNotRaise
from itertools import chain, combinations
from test.test_utils import mock_detections
from typing import List, Optional, Tuple

import numpy as np
import pytest

from supervision import LineZone
from supervision.geometry.core import Point, Position, Vector


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
        (
            Vector(
                Point(0, 0),
                Point(0, 100),
            ),
            [
                [100, 50, 120, 70],
                [-100, 50, -80, 70],
            ],
            [False, False],
            [False, True],
        ),
        (
            Vector(
                Point(0, 0),
                Point(0, 100),
            ),
            [
                [-100, 50, -80, 70],
                [100, 50, 120, 70],
            ],
            [False, True],
            [False, False],
        ),
        (
            Vector(
                Point(0, 0),
                Point(0, 100),
            ),
            [
                [-100, 50, -80, 70],
                [-10, 50, 20, 70],
                [100, 50, 120, 70],
            ],
            [False, False, True],
            [False, False, False],
        ),
        (
            Vector(
                Point(0, 0),
                Point(100, 100),
            ),
            [
                [50, 45, 70, 30],
                [40, 50, 50, 40],
                [0, 50, 10, 40],
            ],
            [False, False, False],
            [False, False, True],
        ),
        (
            Vector(
                Point(0, 0),
                Point(100, 0),
            ),
            [
                [50, -45, 70, -30],
                [40, 50, 50, 40],
            ],
            [False, False],
            [False, True],
        ),
        (
            Vector(
                Point(0, 0),
                Point(0, -100),
            ),
            [
                [100, -50, 120, -70],
                [-100, -50, -80, -70],
            ],
            [False, True],
            [False, False],
        ),
        (
            Vector(
                Point(0, 0),
                Point(50, 100),
            ),
            [
                [50, 50, 70, 30],
                [40, 50, 50, 40],
                [0, 50, 10, 40],
            ],
            [False, False, False],
            [False, False, True],
        ),
        (
            Vector(
                Point(0, 0),
                Point(0, 100),
            ),
            [
                [100, 50, 120, 70],
                [-100, 50, -80, 70],
                [100, 50, 120, 70],
                [-100, 50, -80, 70],
                [100, 50, 120, 70],
                [-100, 50, -80, 70],
                [100, 50, 120, 70],
                [-100, 50, -80, 70],
            ],
            [False, False, True, False, True, False, True, False],
            [False, True, False, True, False, True, False, True],
        ),
        (
            Vector(
                Point(0, 0),
                Point(-100, 0),
            ),
            [
                [-50, 70, -40, 50],
                [-50, -70, -40, -50],
                [-50, 70, -40, 50],
                [-50, -70, -40, -50],
                [-50, 70, -40, 50],
                [-50, -70, -40, -50],
                [-50, 70, -40, 50],
                [-50, -70, -40, -50],
            ],
            [False, False, True, False, True, False, True, False],
            [False, True, False, True, False, True, False, True],
        ),
        (
            Vector(
                Point(0, 100),
                Point(0, 200),
            ),
            [
                [-100, 150, -80, 170],
                [-100, 50, -80, 70],
                [-10, 50, 20, 70],
                [100, 50, 120, 70],
            ],  # detection goes "around" line start and hence never crosses it
            [False, False, False, False],
            [False, False, False, False],
        ),
        (
            Vector(
                Point(0, 100),
                Point(0, 200),
            ),
            [
                [-100, 150, -80, 170],
                [-100, 250, -80, 270],
                [-10, 250, 20, 270],
                [100, 250, 120, 270],
            ],  # detection goes "around" line end and hence never crosses it
            [False, False, False, False],
            [False, False, False, False],
        ),
        (
            Vector(
                Point(-50, -50),
                Point(-100, -150),
            ),
            [
                [-30, -80, -20, -100],
                [-150, -60, -110, -70],
                [-10, -100, 20, -130],
            ],
            [False, True, False],
            [False, False, True],
        ),
    ],
)
def test_line_zone_single_detection(
    vector: Vector,
    xyxy_sequence: List[List[int]],
    expected_crossed_in: List[bool],
    expected_crossed_out: List[bool],
) -> None:
    line_zone = LineZone(start=vector.start, end=vector.end)
    for i, bbox in enumerate(xyxy_sequence):
        detections = mock_detections(
            xyxy=[bbox],
            tracker_id=[0],
        )
        crossed_in, crossed_out = line_zone.trigger(detections)
        assert crossed_in[0] == expected_crossed_in[i]
        assert crossed_out[0] == expected_crossed_out[i]
        assert line_zone.in_count == sum(expected_crossed_in[: (i + 1)])
        assert line_zone.out_count == sum(expected_crossed_out[: (i + 1)])


@pytest.mark.parametrize(
    "vector,"
    "xyxy_sequence,"
    "expected_crossed_in,"
    "expected_crossed_out,"
    "crossing_anchors",
    [
        (
            Vector(
                Point(0, 0),
                Point(100, 100),
            ),
            [
                [50, 30, 60, 20],
                [20, 50, 40, 30],
            ],
            [False, False],
            [False, True],
            [Position.TOP_LEFT, Position.TOP_RIGHT, Position.BOTTOM_LEFT],
        ),
        (
            Vector(
                Point(0, 0),
                Point(0, 100),
            ),
            [
                [-100, 50, -80, 70],
                [-100, 50, 120, 70],
            ],
            [False, True],
            [False, False],
            [Position.TOP_RIGHT, Position.BOTTOM_RIGHT],
        ),
    ],
)
def test_line_zone_single_detection_on_subset_of_anchors(
    vector: Vector,
    xyxy_sequence: List[List[int]],
    expected_crossed_in: List[bool],
    expected_crossed_out: List[bool],
    crossing_anchors: List[Position],
) -> None:
    def powerset(s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    for anchors in powerset(
        [
            Position.TOP_LEFT,
            Position.TOP_RIGHT,
            Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT,
        ]
    ):
        if not anchors:
            continue
        line_zone = LineZone(
            start=vector.start, end=vector.end, triggering_anchors=anchors
        )
        for i, bbox in enumerate(xyxy_sequence):
            detections = mock_detections(
                xyxy=[bbox],
                tracker_id=[0],
            )
            crossed_in, crossed_out = line_zone.trigger(detections)
            if all(anchor in crossing_anchors for anchor in anchors):
                assert crossed_in == expected_crossed_in[i]
                assert crossed_out == expected_crossed_out[i]
            else:
                assert np.all(not crossed_in)
                assert np.all(not crossed_out)


@pytest.mark.parametrize(
    "vector,"
    "xyxy_sequence,"
    "expected_crossed_in,"
    "expected_crossed_out,"
    "anchors, exception",
    [
        (
            Vector(
                Point(0, 0),
                Point(0, 100),
            ),
            [
                [[100, 50, 120, 70], [100, 50, 120, 70]],
                [[-100, 50, -80, 70], [100, 50, 120, 70]],
                [[100, 50, 120, 70], [100, 50, 120, 70]],
            ],
            [[False, False], [False, False], [True, False]],
            [[False, False], [True, False], [False, False]],
            [
                Position.TOP_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_LEFT,
                Position.BOTTOM_RIGHT,
            ],
            DoesNotRaise(),
        ),
        (
            Vector(
                Point(0, 0),
                Point(-100, 0),
            ),
            [
                [[-50, 70, -40, 50], [-80, -50, -70, -40]],
                [[-50, -70, -40, -50], [-80, 50, -70, 40]],
                [[-50, 70, -40, 50], [-80, 50, -70, 40]],
                [[-50, -70, -40, -50], [-80, 50, -70, 40]],
                [[-50, 70, -40, 50], [-80, 50, -70, 40]],
                [[-50, -70, -40, -50], [-80, 50, -70, 40]],
                [[-50, 70, -40, 50], [-80, 50, -70, 40]],
                [[-50, -70, -40, -50], [-80, -50, -70, -40]],
            ],
            [
                (False, False),
                (False, True),
                (True, False),
                (False, False),
                (True, False),
                (False, False),
                (True, False),
                (False, False),
            ],
            [
                (False, False),
                (True, False),
                (False, False),
                (True, False),
                (False, False),
                (True, False),
                (False, False),
                (True, True),
            ],
            [
                Position.TOP_LEFT,
                Position.TOP_RIGHT,
                Position.BOTTOM_LEFT,
                Position.BOTTOM_RIGHT,
            ],
            DoesNotRaise(),
        ),
        (
            Vector(
                Point(-50, -50),
                Point(-100, -150),
            ),
            [
                [[-30, -80, -20, -100], [100, 50, 120, 70]],
                [[-100, -80, -20, -100], [100, 50, 120, 70]],
            ],
            [[False, False], [True, False]],
            [[False, False], [False, False]],
            [Position.TOP_LEFT],
            DoesNotRaise(),
        ),
        (
            Vector(
                Point(0, 0),
                Point(-100, 0),
            ),
            [[[-50, 70, -40, 50], [-80, -50, -70, -40]]],
            [(False, False)],
            [(False, False)],
            [],  # raise because of empty anchors
            pytest.raises(ValueError),
        ),
    ],
)
def test_line_zone_multiple_detections(
    vector: Vector,
    xyxy_sequence: List[List[List[int]]],
    expected_crossed_in: List[bool],
    expected_crossed_out: List[bool],
    anchors: List[Position],
    exception: Exception,
) -> None:
    with exception:
        line_zone = LineZone(
            start=vector.start, end=vector.end, triggering_anchors=anchors
        )
        for i, bboxes in enumerate(xyxy_sequence):
            detections = mock_detections(
                xyxy=bboxes,
                tracker_id=[i for i in range(0, len(bboxes))],
            )
            crossed_in, crossed_out = line_zone.trigger(detections)
            assert np.all(crossed_in == expected_crossed_in[i])
            assert np.all(crossed_out == expected_crossed_out[i])
