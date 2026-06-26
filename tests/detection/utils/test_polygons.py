from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils.polygons import (
    approximate_polygon,
    filter_polygons_by_area,
)


@pytest.mark.parametrize(
    ("polygons", "min_area", "max_area", "expected_result", "exception"),
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


def _regular_polygon(num_points: int, radius: float = 40.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    return np.stack(
        [50 + radius * np.cos(angles), 50 + radius * np.sin(angles)], axis=1
    ).astype(np.float32)


class TestApproximatePolygon:
    @pytest.mark.parametrize("num_points", [20, 50, 100, 200])
    @pytest.mark.parametrize("percentage", [0.1, 0.5, 0.75, 0.9])
    def test_within_budget_and_valid(self, num_points: int, percentage: float) -> None:
        """The result stays a valid polygon and respects the point budget.

        The exception is the 3-point floor: keeping a valid polygon (at least 3
        points) can leave more points than the budget by an arbitrary margin,
        since a single epsilon step may jump straight from above budget to below
        3 points.
        """
        polygon = _regular_polygon(num_points)
        target_points = max(int(num_points * (1 - percentage)), 3)

        result = approximate_polygon(polygon, percentage=percentage)

        assert result.ndim == 2
        assert result.shape[1] == 2
        assert 3 <= len(result) <= num_points
        if target_points > 3:
            assert len(result) <= target_points

    def test_zero_percentage_keeps_polygon(self) -> None:
        """A percentage of 0 removes no points."""
        polygon = _regular_polygon(40)

        result = approximate_polygon(polygon, percentage=0.0)

        assert len(result) == len(polygon)

    @pytest.mark.parametrize(
        "percentage",
        [
            pytest.param(-0.001, id="just-below-zero"),
            pytest.param(-1.0, id="negative-one"),
            pytest.param(1.0, id="exactly-one"),
            pytest.param(1.5, id="above-one"),
        ],
    )
    def test_raises_on_out_of_range_percentage(self, percentage: float) -> None:
        """Percentage outside [0, 1) must raise ValueError."""
        polygon = _regular_polygon(20)
        with pytest.raises(ValueError, match="Percentage must be in the range"):
            approximate_polygon(polygon, percentage=percentage)

    @pytest.mark.parametrize(
        "epsilon_step",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-0.05, id="negative"),
        ],
    )
    def test_raises_on_non_positive_epsilon_step(self, epsilon_step: float) -> None:
        """Non-positive epsilon_step must raise ValueError."""
        polygon = _regular_polygon(20)
        with pytest.raises(ValueError, match="epsilon_step must be positive"):
            approximate_polygon(polygon, percentage=0.5, epsilon_step=epsilon_step)
