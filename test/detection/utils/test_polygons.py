from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils.polygons import filter_polygons_by_area


@pytest.mark.parametrize(
    "polygons, min_area, max_area, expected_result, exception",
    [
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # single polygon without area constraints
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            50,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # single polygon with min_area constraint
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            50,
            [],
            DoesNotRaise(),
        ),  # single polygon with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            200,
            None,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            None,
            200,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # two polygons with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            200,
            200,
            [],
            DoesNotRaise(),
        ),  # two polygons with both area constraints
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            100,
            100,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area and
        # max_area equal to the area of the first polygon
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            400,
            400,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area and
        # max_area equal to the area of the second polygon
    ],
)
def test_filter_polygons_by_area(
    polygons: list[np.ndarray],
    min_area: float | None,
    max_area: float | None,
    expected_result: list[np.ndarray],
    exception: Exception,
) -> None:
    with exception:
        result = filter_polygons_by_area(
            polygons=polygons, min_area=min_area, max_area=max_area
        )
        assert len(result) == len(expected_result)
        for result_polygon, expected_result_polygon in zip(result, expected_result):
            assert np.array_equal(result_polygon, expected_result_polygon)
