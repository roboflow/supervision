"""Tests for private geometry fallbacks."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from supervision._cv2._geometry import (
    _approx_poly_dp,
    _contour_area,
    _farthest_from_segment,
    _intersect_convex_convex,
)

cv2: Any
try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    cv2 = None

# Tests with literal expectations need no native OpenCV and must also run in the
# jobs without it; only tests that call or compare against it are gated.
_requires_opencv = pytest.mark.skipif(
    cv2 is None, reason="OpenCV is required as the reference implementation"
)
# OpenCV 4.13 started measuring approxPolyDP distance to the finite segment
# (opencv#28119); older releases keep the infinite-line behaviour.
_requires_opencv_segment_distance = pytest.mark.skipif(
    cv2 is None or tuple(int(p) for p in cv2.__version__.split(".")[:2]) < (4, 13),
    reason="OpenCV 4.13+ is required: it measures approxPolyDP to the segment",
)


def _sort_points_lexicographically(points: np.ndarray) -> np.ndarray:
    """Sort 2D points row-wise so point sets can be compared order-independently."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


def _max_distance_to_closed_polyline(points: np.ndarray, vertices: np.ndarray) -> float:
    """Return the largest distance from points to a closed polyline."""
    starts = vertices
    segments = np.roll(vertices, -1, axis=0) - starts
    offsets = points[:, np.newaxis, :] - starts[np.newaxis, :, :]
    lengths_squared = np.sum(segments**2, axis=1)
    projections = np.divide(
        np.sum(offsets * segments[np.newaxis, :, :], axis=2),
        lengths_squared[np.newaxis, :],
        out=np.zeros((len(points), len(vertices))),
        where=lengths_squared[np.newaxis, :] != 0,
    )
    closest = (
        starts[np.newaxis, :, :]
        + np.clip(projections, 0, 1)[:, :, np.newaxis] * segments[np.newaxis, :, :]
    )
    distances = np.linalg.norm(points[:, np.newaxis, :] - closest, axis=2)
    return float(np.max(np.min(distances, axis=1)))


@_requires_opencv
@pytest.mark.parametrize(
    ("contour", "oriented"),
    [
        pytest.param(
            np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.int32),
            False,
            id="counter-clockwise-absolute",
        ),
        pytest.param(
            np.array([[0, 0], [0, 4], [4, 4], [4, 0]], dtype=np.int32),
            True,
            id="clockwise-oriented",
        ),
        pytest.param(
            np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32),
            False,
            id="degenerate",
        ),
    ],
)
def test_contour_area_matches_opencv(contour: np.ndarray, oriented: bool) -> None:
    """Match OpenCV's signed and absolute shoelace areas."""
    actual = _contour_area(contour, oriented=oriented)
    expected = cv2.contourArea(contour, oriented=oriented)

    assert actual == expected


@_requires_opencv
@pytest.mark.parametrize(
    ("contour", "epsilon"),
    [
        pytest.param(
            np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.int32),
            0.5,
            id="rectangle",
        ),
        pytest.param(
            np.array(
                [[0, 0], [2, 0], [4, 0], [4, 4], [2, 4], [0, 4]],
                dtype=np.int32,
            ),
            0.5,
            id="collinear-runs",
        ),
        pytest.param(
            np.array(
                [[4, 0], [8, 0], [12, 4], [12, 8], [8, 12], [4, 12], [0, 8], [0, 4]],
                dtype=np.int32,
            ),
            0.5,
            id="octagon",
        ),
    ],
)
def test_approx_poly_dp_matches_opencv(contour: np.ndarray, epsilon: float) -> None:
    """Match OpenCV's closed Douglas-Peucker output."""
    actual = _approx_poly_dp(contour, epsilon, closed=True)
    expected = cv2.approxPolyDP(contour, epsilon, closed=True)

    np.testing.assert_array_equal(actual, expected)


def test_approx_poly_dp_approximates_irregular_closed_contours() -> None:
    """Keep irregular closed contours within the requested approximation error."""
    rng = np.random.default_rng(20260717)

    for _ in range(100):
        count = int(rng.integers(4, 40))
        angles = np.sort(rng.uniform(0, 2 * np.pi, count))
        radii = rng.uniform(10, 100, count)
        contour = np.rint(
            np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
        ).astype(np.int32)
        epsilon = float(rng.uniform(0, 10))

        actual = _approx_poly_dp(contour, epsilon, closed=True)
        vertices = actual.reshape(-1, 2)
        is_input_vertex = np.any(
            np.all(vertices[:, np.newaxis, :] == contour[np.newaxis, :, :], axis=2),
            axis=1,
        )

        assert actual.dtype == contour.dtype
        assert 3 <= len(vertices) <= len(contour)
        assert np.all(is_input_vertex)
        assert _max_distance_to_closed_polyline(contour, vertices) <= epsilon


@_requires_opencv
def test_approx_poly_dp_preserves_explicitly_closed_contour_anchors() -> None:
    """Match OpenCV anchors when the first contour point is repeated at the end."""
    contour = np.array(
        [[48, 68], [63, 62], [-39, 73], [44, -81], [48, 68]], dtype=np.int32
    )
    epsilon = 12.301107

    actual = _approx_poly_dp(contour, epsilon, closed=True)
    expected = cv2.approxPolyDP(contour, epsilon, closed=True)

    np.testing.assert_array_equal(actual, expected)


@_requires_opencv
@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32),
            np.array([[2, 0], [6, 0], [6, 4], [2, 4]], dtype=np.float32),
            id="partial-overlap",
        ),
        pytest.param(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=np.float32),
            id="nested",
        ),
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            np.array([[3, 3], [4, 3], [4, 4], [3, 4]], dtype=np.float32),
            id="disjoint",
        ),
    ],
)
def test_intersect_convex_convex_matches_opencv(
    first: np.ndarray, second: np.ndarray
) -> None:
    """Match OpenCV's convex intersection area and vertices."""
    actual_area, actual_polygon = _intersect_convex_convex(first, second)
    expected_area, expected_polygon = cv2.intersectConvexConvex(first, second)

    assert actual_area == pytest.approx(expected_area, abs=1e-9)
    if expected_area == 0:
        assert actual_polygon.size == 0
        return

    assert actual_polygon.shape == expected_polygon.shape
    np.testing.assert_allclose(
        _sort_points_lexicographically(actual_polygon.reshape(-1, 2)),
        _sort_points_lexicographically(expected_polygon.reshape(-1, 2)),
        atol=1e-6,
        rtol=0,
    )


@_requires_opencv
def test_intersect_convex_convex_bounds_float32_roundoff() -> None:
    """Bound OpenCV float32 area drift across rotated-rectangle intersections."""
    rng = np.random.default_rng(20260717)

    for _ in range(200):
        centers = rng.uniform(-100, 100, (2, 2))
        sizes = rng.uniform(1, 100, (2, 2))
        angles = rng.uniform(0, 180, 2)
        polygons = [
            cv2.boxPoints((tuple(center), tuple(size), float(angle)))
            for center, size, angle in zip(centers, sizes, angles)
        ]

        actual_area, _ = _intersect_convex_convex(polygons[0], polygons[1])
        expected_area, _ = cv2.intersectConvexConvex(polygons[0], polygons[1])

        assert actual_area == pytest.approx(expected_area, abs=5e-4)


def test_approx_poly_dp_keeps_vertex_beyond_segment_endpoint_closed() -> None:
    """Keep a vertex far beyond a segment end that sits near the infinite line."""
    contour = np.array([[2, 3], [10, 13], [11, 3], [9, 1], [13, 6]], dtype=np.int32)
    actual = _approx_poly_dp(contour, 1.0, closed=True).reshape(-1, 2)
    assert len(actual) == 5
    np.testing.assert_array_equal(actual, contour)


def test_approx_poly_dp_keeps_vertex_beyond_segment_endpoint_open() -> None:
    """Keep an open-polyline vertex whose projection falls outside the segment."""
    contour = np.array(
        [[86, 24], [97, 37], [15, 41], [97, 11], [69, 13], [68, 15]], dtype=np.int32
    )
    actual = _approx_poly_dp(contour, 10.0, closed=False)

    np.testing.assert_array_equal(
        actual.reshape(-1, 2), [[86, 24], [97, 37], [15, 41], [97, 11], [68, 15]]
    )


def test_approx_poly_dp_seeds_open_contour_repeating_first_point_once() -> None:
    """Seed an open contour whose last point repeats its first with one pass."""
    contour = np.array([[3, 11], [2, 6], [13, 14], [3, 11]], dtype=np.int32)

    actual = _approx_poly_dp(contour, 2.5, closed=False)

    np.testing.assert_array_equal(actual.reshape(-1, 2), [[3, 11], [2, 6], [13, 14]])


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), -1.0, 1e30])
def test_approx_poly_dp_rejects_invalid_epsilon(epsilon: float) -> None:
    """Reject NaN, infinite, negative and overflowing epsilon as OpenCV does."""
    contour = np.array([[0, 0], [1, 0], [2, 0], [2, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="epsilon must be"):
        _approx_poly_dp(contour, epsilon, closed=False)


class TestFarthestFromSegment:
    """Scaled farthest-point search used by the Douglas-Peucker traversal."""

    @pytest.mark.parametrize(
        ("xs", "ys", "start", "end", "expected"),
        [
            pytest.param(
                [0, 2, 4], [0, 3, 0], 0, 2, (144, 1, 16), id="projects-inside-segment"
            ),
            pytest.param(
                [0, -3, 4], [0, 4, 0], 0, 2, (400, 1, 16), id="nearest-to-start"
            ),
            pytest.param([0, 7, 4], [0, 4, 0], 0, 2, (400, 1, 16), id="nearest-to-end"),
            pytest.param(
                [0, 0, 4], [0, 0, 0], 0, 2, (0, 0, 16), id="coincides-with-start"
            ),
            pytest.param(
                [0, 4, 4], [0, 0, 0], 0, 2, (0, 0, 16), id="coincides-with-end"
            ),
            pytest.param([0, 4], [0, 0], 0, 1, (0, 0, 16), id="no-point-between-ends"),
            pytest.param(
                [0, 2, 4, 2], [0, 0, 0, -3], 2, 0, (144, 3, 16), id="wraps-at-count"
            ),
            pytest.param(
                [1, 3, 1], [1, 5, 1], 0, 2, (20, 1, 1), id="zero-length-segment"
            ),
        ],
    )
    def test_returns_scaled_maximum_split_and_scale(
        self,
        xs: list[float],
        ys: list[float],
        start: int,
        end: int,
        expected: tuple[float, int, float],
    ) -> None:
        """Return the scaled maximum, its index and the squared segment length."""
        actual = _farthest_from_segment(xs, ys, start, end, len(xs))

        assert actual == expected


def _random_integer_contours(dtype: type) -> tuple[list[np.ndarray], list[float]]:
    """Build seeded integer-valued contours with mixed integer and float epsilons."""
    rng = np.random.default_rng(20261006)
    contours = [
        rng.integers(-50, 100, (int(rng.integers(2, 40)), 2)).astype(dtype)
        for _ in range(300)
    ]
    raw = rng.uniform(0, 15, len(contours))
    epsilons = np.where(np.arange(len(contours)) % 2 == 1, np.rint(raw), raw)
    return contours, epsilons.tolist()


def _approx_poly_dp_mismatches(
    contours: list[np.ndarray],
    epsilons: list[float],
    closed: bool,
    repeat_first: bool,
) -> list[str]:
    """Describe every contour whose fallback result differs from OpenCV's."""
    mismatches = []
    for contour, epsilon in zip(contours, epsilons):
        candidate = np.vstack([contour, contour[:1]]) if repeat_first else contour
        actual = _approx_poly_dp(candidate, epsilon, closed)
        expected = cv2.approxPolyDP(candidate, epsilon, closed)
        if not np.array_equal(actual, expected):
            mismatches.append(
                f"{candidate.dtype} closed={closed} eps={epsilon}: "
                f"{candidate.tolist()} -> {actual.reshape(-1, 2).tolist()} "
                f"!= {expected.reshape(-1, 2).tolist()}"
            )
    return mismatches


@_requires_opencv_segment_distance
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(np.int32, id="int32"), pytest.param(np.float32, id="float32")],
)
@pytest.mark.parametrize(
    ("closed", "repeat_first"),
    [
        pytest.param(True, False, id="closed"),
        pytest.param(False, False, id="open"),
        pytest.param(False, True, id="open-first-equals-last"),
    ],
)
def test_approx_poly_dp_matches_opencv_on_random_contours(
    dtype: type, closed: bool, repeat_first: bool
) -> None:
    """Match OpenCV 4.13+ on seeded integer contours, ties and boundaries included."""
    contours, epsilons = _random_integer_contours(dtype)

    mismatches = _approx_poly_dp_mismatches(contours, epsilons, closed, repeat_first)

    assert not mismatches, "\n".join(mismatches[:5])
