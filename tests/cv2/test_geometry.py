"""Tests for private geometry fallbacks."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from supervision._cv2._geometry import (
    _approx_poly_dp,
    _contour_area,
    _intersect_convex_convex,
)

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


def _sort_points_lexicographically(points: np.ndarray) -> np.ndarray:
    """Sort 2D points row-wise so point sets can be compared order-independently."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


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


def test_approx_poly_dp_matches_opencv_for_irregular_closed_contours() -> None:
    """Match OpenCV anchor selection and cleanup on irregular closed contours."""
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
        expected = cv2.approxPolyDP(contour, epsilon, closed=True)

        np.testing.assert_array_equal(actual, expected)


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
