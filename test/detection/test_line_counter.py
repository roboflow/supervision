from contextlib import ExitStack as DoesNotRaise
from itertools import chain, combinations
from test.test_utils import mock_detections
from typing import Optional, Tuple

import numpy as np
import pytest

from supervision import Detections, LineZone
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
    "vector, bbox_sequence, expected_count_in, expected_count_out",
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
    ],
)
def test_line_zone_single_detection(
    vector, bbox_sequence, expected_count_in: list[bool], expected_count_out: list[bool]
) -> None:
    line_zone = LineZone(start=vector.start, end=vector.end)
    for i, bbox in enumerate(bbox_sequence):
        detections = mock_detections(
            xyxy=[bbox],
            tracker_id=[i for i in range(0, 1)],
        )
        count_in, count_out = line_zone.trigger(detections)
        assert count_in[0] == expected_count_in[i]
        assert count_out[0] == expected_count_out[i]
        assert line_zone.in_count == sum(expected_count_in[: (i + 1)])
        assert line_zone.out_count == sum(expected_count_out[: (i + 1)])


@pytest.mark.parametrize(
    "vector, bbox_sequence, expected_count_in, expected_count_out, crossing_anchors",
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
    vector,
    bbox_sequence,
    expected_count_in: list[bool],
    expected_count_out: list[bool],
    crossing_anchors,
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
        for i, bbox in enumerate(bbox_sequence):
            detections = mock_detections(
                xyxy=[bbox],
                tracker_id=[i for i in range(0, 1)],
            )
            count_in, count_out = line_zone.trigger(detections)
            if all(anchor in crossing_anchors for anchor in anchors):
                assert count_in == expected_count_in[i]
                assert count_out == expected_count_out[i]
            else:
                assert np.all(not count_in)
                assert np.all(not count_out)


@pytest.mark.parametrize(
    "vector, bbox_sequence, expected_count_in, expected_count_out",
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
        ),
    ],
)
def test_line_zone_multiple_detections(
    vector, bbox_sequence, expected_count_in: list[bool], expected_count_out: list[bool]
) -> None:
    line_zone = LineZone(start=vector.start, end=vector.end)
    for i, bboxes in enumerate(bbox_sequence):
        detections = mock_detections(
            xyxy=bboxes,
            tracker_id=[i for i in range(0, len(bboxes))],
        )
        count_in, count_out = line_zone.trigger(detections)
        assert np.all(count_in == expected_count_in[i])
        assert np.all(count_out == expected_count_out[i])


@pytest.mark.parametrize(
    "vector, bbox_sequence",
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
        ),
    ],
)
def test_line_zone_does_not_count_detections_without_tracker_id(vector, bbox_sequence):
    line_zone = LineZone(start=vector.start, end=vector.end)
    for bbox in bbox_sequence:
        detections = Detections(
            xyxy=np.array([bbox]).reshape((-1, 4)),
            tracker_id=np.array([None for _ in range(0, 1)]),
        )
        count_in, count_out = line_zone.trigger(detections)
        assert np.all(not count_in)
        assert np.all(not count_out)
