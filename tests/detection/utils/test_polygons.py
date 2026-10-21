from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils.polygons import (
    approximate_polygon,
    filter_polygons_by_area,
    merge_multi_segment,
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
        (
            [np.empty((0, 2))],
            50,
            None,
            [],
            DoesNotRaise(),
        ),  # empty polygon with min_area constraint
        (
            [np.empty((0, 2))],
            None,
            50,
            [np.empty((0, 2))],
            DoesNotRaise(),
        ),  # empty polygon with max_area constraint
        (
            [
                np.empty((0, 2)),
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
            ],
            50,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # mixed empty and valid polygons with min_area
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

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.float64, id="float64"),
            pytest.param(np.float32, id="float32"),
            pytest.param(np.int64, id="int64"),
            pytest.param(np.int32, id="int32"),
            pytest.param(np.int16, id="int16"),
            pytest.param(np.uint8, id="uint8"),
        ],
    )
    def test_approximate_polygon_preserves_dtype(self, dtype: type) -> None:
        """Ensure approximate_polygon supports float64/int64 and preserves dtype."""
        polygon = np.array(
            [[0, 0], [10, 0], [10, 10], [0, 10], [5, 10], [5, 5], [3, 7], [1, 9]],
            dtype=dtype,
        )
        result = approximate_polygon(polygon, percentage=0.5)
        assert result.dtype == dtype
        assert 3 <= len(result) <= len(polygon)

    @pytest.mark.parametrize(
        ("dtype", "origin"),
        [
            pytest.param(np.int64, 1_000_000_000, id="int64-large-origin"),
            pytest.param(np.float64, 1_000_000_000.0, id="float64-large-origin"),
        ],
    )
    def test_approximate_polygon_preserves_large_origin_geometry(
        self, dtype: type, origin: float
    ) -> None:
        """Approximation keeps large-origin local geometry instead of collapsing it."""
        polygon = np.array(
            [
                [origin, origin],
                [origin + 10, origin],
                [origin + 10, origin + 10],
                [origin, origin + 10],
                [origin + 5, origin + 10],
                [origin + 5, origin + 5],
                [origin + 3, origin + 7],
                [origin + 1, origin + 9],
            ],
            dtype=dtype,
        )

        result = approximate_polygon(polygon, percentage=0.5)

        assert result.dtype == dtype
        assert 3 <= len(result) <= 4
        assert np.all(result >= origin)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float64, id="float64"),
        pytest.param(np.float32, id="float32"),
        pytest.param(np.int64, id="int64"),
        pytest.param(np.int32, id="int32"),
        pytest.param(np.int16, id="int16"),
        pytest.param(np.uint8, id="uint8"),
    ],
)
def test_filter_polygons_by_area_dtypes(dtype: type) -> None:
    """Ensure filter_polygons_by_area supports float64/int64 dtypes."""
    small = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=dtype)
    big = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=dtype)
    result = filter_polygons_by_area([small, big], min_area=50)
    assert len(result) == 1
    assert result[0].dtype == dtype
    assert np.array_equal(result[0], big)


@pytest.mark.parametrize(
    ("dtype", "origin"),
    [
        pytest.param(np.int64, 1_000_000_000, id="int64-large-origin"),
        pytest.param(np.float64, 1_000_000_000.0, id="float64-large-origin"),
    ],
)
def test_filter_polygons_by_area_preserves_large_origin_geometry(
    dtype: type, origin: float
) -> None:
    """Area filtering measures large-origin polygons in their local coordinates."""
    square = np.array(
        [
            [origin, origin],
            [origin + 10, origin],
            [origin + 10, origin + 10],
            [origin, origin + 10],
        ],
        dtype=dtype,
    )

    result = filter_polygons_by_area([square], min_area=50)

    assert len(result) == 1
    assert np.array_equal(result[0], square)


class TestMergeMultiSegment:
    """Tests for merging multiple disconnected polygon segments."""

    def test_empty_segments_returns_empty_array(self) -> None:
        """Empty input yields a 2D empty array."""
        result = merge_multi_segment([])

        assert result.shape == (0, 2)

    def test_single_segment_returns_unchanged(self) -> None:
        """Single polygon segment is returned as-is without modification."""
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)

        result = merge_multi_segment([poly])

        np.testing.assert_array_equal(result, poly)

    def test_merge_two_disconnected_segments(self) -> None:
        """Two separated square segments are merged into a single closed contour."""
        p1 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.int32)
        p2 = np.array([[35, 5], [45, 5], [45, 15], [35, 15]], dtype=np.int32)

        result = merge_multi_segment([p1, p2])

        assert result.ndim == 2
        assert result.shape[-1] == 2
        assert len(result) == 10
        # The first and last points of p1 at the seam are connected to p2
        assert [15, 5] in result.tolist()
        assert [35, 5] in result.tolist()

    def test_merge_three_disconnected_segments(self) -> None:
        """Three collinear square segments are merged in order."""
        p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        p2 = np.array([[20, 0], [30, 0], [30, 10], [20, 10]], dtype=np.float32)
        p3 = np.array([[40, 0], [50, 0], [50, 10], [40, 10]], dtype=np.float32)

        result = merge_multi_segment([p1, p2, p3])

        assert result.ndim == 2
        assert result.shape[-1] == 2
        assert len(result) == 16

    def test_merge_intermediate_segment_reversed_indices(self) -> None:
        """Intermediate polygon with idx[0] > idx[1] connects closest vertices."""
        p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        # 5 points; closest to p1 is index 4 ([20, 0]);
        # closest to p3 is index 2 ([30, 0])
        p2 = np.array(
            [
                [25, -5],
                [27, -5],
                [30, 0],
                [30, 10],
                [20, 0],
            ],
            dtype=np.float32,
        )
        p3 = np.array([[40, 0], [50, 0], [50, 10], [40, 10]], dtype=np.float32)

        result = merge_multi_segment([p1, p2, p3])

        assert result.ndim == 2
        assert result.shape[-1] == 2
        # p1 (5 points) + p2 forward (3 points) + p3 (5 points) + p2 return (3 pts) + 1
        assert len(result) == 17
        # Ensure the transition from p1 connects to p2[4] ([20, 0]), not [30, 0]
        # In result, after p1 ends at [10, 0], the next point should be [20, 0]
        p1_end_idx = 4
        np.testing.assert_array_equal(result[p1_end_idx], [10.0, 0.0])
        np.testing.assert_array_equal(result[p1_end_idx + 1], [20.0, 0.0])

    def test_invalid_shape_raises_value_error(self) -> None:
        """A segment with incorrect dimension or coordinates raises ValueError."""
        invalid_1d = np.array([0, 0, 10, 10])

        with pytest.raises(ValueError, match="Each segment must have shape \\(N, 2\\)"):
            merge_multi_segment([invalid_1d])

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.uint8, id="uint8"),
            pytest.param(np.int32, id="int32"),
            pytest.param(np.int64, id="int64"),
            pytest.param(np.float32, id="float32"),
            pytest.param(np.float64, id="float64"),
        ],
    )
    def test_preserves_input_dtype(self, dtype: type) -> None:
        """The merged polygon retains the dtype of the input segments."""
        p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=dtype)
        p2 = np.array([[20, 0], [30, 0], [30, 10], [20, 10]], dtype=dtype)

        result = merge_multi_segment([p1, p2])

        assert result.dtype == dtype

    def test_no_overflow_uint8_and_large_int32(self) -> None:
        """Nearest pair calculation does not overflow with uint8 or large int32."""
        # uint8: squared distance of 20 or 40 would overflow in uint8
        p1_uint8 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.uint8)
        p2_uint8 = np.array([[35, 5], [45, 5], [45, 15], [35, 15]], dtype=np.uint8)
        res_uint8 = merge_multi_segment([p1_uint8, p2_uint8])
        assert res_uint8.dtype == np.uint8
        # Closest vertices are [15, 5] and [35, 5] (dx=20), not [5, 5] and [45, 5]
        # In the result, the seam must connect p1's [15, 5] to p2's [35, 5]
        p1_seam_idx = 4
        np.testing.assert_array_equal(res_uint8[p1_seam_idx], [15, 5])
        np.testing.assert_array_equal(res_uint8[p1_seam_idx + 1], [35, 5])

        # int32: coordinate difference > 46,341 squares to > 2**31 - 1
        p1_int32 = np.array(
            [[0, 0], [100000, 0], [100000, 10], [0, 10]], dtype=np.int32
        )
        p2_int32 = np.array(
            [[160000, 0], [200000, 0], [200000, 10], [160000, 10]],
            dtype=np.int32,
        )
        res_int32 = merge_multi_segment([p1_int32, p2_int32])
        assert res_int32.dtype == np.int32
        # Closest vertices are [100000, 0] and [160000, 0]
        np.testing.assert_array_equal(res_int32[4], [100000, 0])
        np.testing.assert_array_equal(res_int32[5], [160000, 0])
