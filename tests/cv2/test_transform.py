"""Tests for private transform and filter fallbacks."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from supervision._cv2._transform import (
    _blur,
    _get_perspective_transform,
    _perspective_transform,
)

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


SPEED_SOURCE = np.array(
    [[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]], dtype=np.float32
)
SPEED_TARGET = np.array([[0, 0], [24, 0], [24, 249], [0, 249]], dtype=np.float32)
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
DOUBLE = UNIT * 2
IDENTICAL = np.ones((4, 2), dtype=np.float32)
NAN_CORNER = np.array([[np.nan, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
# Thin but non-degenerate quads with ill-conditioned perspective systems.
THIN_8 = np.array([[0, 0], [1, 0], [1, 1e-8], [0, 1e-8]], dtype=np.float32)
THIN_6 = np.array([[0, 0], [1, 0], [1, 1e-6], [0, 1e-6]], dtype=np.float32)
THIN_JITTERED = np.array(
    [[0, 0], [1, 0], [1.0000001, 1.1e-8], [0.05, 0.9e-8]], dtype=np.float32
)
THIN_OFFSET = np.array(
    [
        [44247.203125, 10406.745],
        [44890.848, 10406.714],
        [44821.34, 10407.404],
        [44244.52, 10407.388],
    ],
    dtype=np.float32,
)
POINTS = np.zeros((4, 1, 2), dtype=np.float32)
# Unit-square corners jittered by at most 0.2 stay strictly convex at any scale.
_RNG = np.random.default_rng(20261001)
RANDOM_QUADS = (
    (UNIT + _RNG.uniform(-0.2, 0.2, (3, 2, 4, 2)))
    * _RNG.uniform(10, 2000, (3, 2, 1, 1))
    + _RNG.uniform(-500, 500, (3, 2, 1, 2))
).astype(np.float32)
SPEED = pytest.param(SPEED_SOURCE, SPEED_TARGET, id="speed-example")
SCALE = pytest.param(UNIT, DOUBLE, id="uniform-scale")
RANDOM = [pytest.param(*pair, id=f"random-{i}") for i, pair in enumerate(RANDOM_QUADS)]


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (4, 2) points through a 3x3 matrix by plain float64 homogeneous division."""
    homogeneous = points.astype(np.float64) @ matrix[:, :2].T + matrix[:, 2]
    return homogeneous[:, :2] / homogeneous[:, 2:]


def test_fallback_blur_preserves_shape_and_dtype() -> None:
    """Preserve source shape and dtype during blurring."""
    source = np.arange(25, dtype=np.uint8).reshape(5, 5)
    blurred = _blur(source, (3, 3))

    assert blurred.shape == source.shape
    assert blurred.dtype == source.dtype


class TestGetPerspectiveTransform:
    @pytest.mark.parametrize(("source", "target"), [*RANDOM, SPEED, SCALE])
    def test_maps_source_corners_onto_target(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """Return a float64 (3, 3) matrix with m[2, 2] == 1 that maps src onto dst."""
        actual = _get_perspective_transform(source, target)

        assert actual.shape == (3, 3)
        assert actual.dtype == np.float64
        assert actual[2, 2] == 1.0
        # The float64 solve lands corners within ~1e-14 of the largest target
        # coordinate (measured over 50k random quads); 1e-12 leaves wide margin.
        np.testing.assert_allclose(
            _project(actual, source), target, rtol=0, atol=1e-12 * np.abs(target).max()
        )

    @pytest.mark.parametrize(("source", "target"), [*RANDOM, SPEED, SCALE])
    def test_matches_opencv_within_float32_rounding(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """Project the source corners where OpenCV does, up to its float32 rounding.

        OpenCV forms the product terms of its system from float32 products, so the
        matrices differ elementwise by more than any fixed rtol on unseen quads; the
        corners they map are the contract, and OpenCV's own corner error bounds the gap.
        """
        expected = cv2.getPerspectiveTransform(source, target)

        actual = _get_perspective_transform(source, target)

        # OpenCV's corner error reaches ~18 float32 ulps of the largest target
        # coordinate (measured over 50k random quads); 32 ulps leaves ~1.8x margin.
        tolerance = 32 * np.finfo(np.float32).eps * np.abs(target).max()
        np.testing.assert_allclose(
            _project(actual, source),
            _project(expected, source),
            rtol=0,
            atol=tolerance,
        )

    @pytest.mark.parametrize("shape", [(4, 1, 2), (1, 4, 2)])
    def test_accepts_opencv_point_vector_layouts(self, shape: tuple[int, ...]) -> None:
        """Accept the point vector layouts OpenCV accepts."""
        expected = _get_perspective_transform(UNIT, DOUBLE)

        actual = _get_perspective_transform(UNIT.reshape(shape), DOUBLE.reshape(shape))

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            pytest.param(THIN_8, THIN_8, id="1x1e-8-identity"),
            pytest.param(THIN_6, UNIT, id="1x1e-6-to-unit"),
            pytest.param(THIN_8, UNIT, id="1x1e-8-to-unit"),
            pytest.param(THIN_JITTERED, THIN_JITTERED, id="jittered-thin-identity"),
            pytest.param(THIN_OFFSET, THIN_OFFSET, id="offset-thin-identity"),
        ],
    )
    def test_projects_thin_quads_like_opencv(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """Solve thin quads without raising and land their corners where OpenCV does."""
        reference = cv2.getPerspectiveTransform(source, target)

        actual = _get_perspective_transform(source, target)

        assert actual[2, 2] == 1.0
        # Thin quads give ill-conditioned matrices, so compare where corners land
        # rather than matrix entries. Measured gaps stay below 0.05 float32 ulps of
        # the largest target coordinate, so 4 ulps leaves margin for other LAPACKs.
        tolerance = 4 * np.finfo(np.float32).eps * np.abs(target).max()
        np.testing.assert_allclose(
            _project(actual, source), _project(reference, source), atol=tolerance
        )

    @pytest.mark.parametrize(
        ("source", "target", "error", "match"),
        [
            pytest.param(
                UNIT.astype(float), DOUBLE, TypeError, "float32", id="float64"
            ),
            pytest.param(UNIT.tolist(), DOUBLE, TypeError, "float32", id="list"),
            pytest.param(UNIT[:3], DOUBLE[:3], ValueError, "shape", id="three-points"),
            pytest.param(UNIT.T.copy(), DOUBLE.T.copy(), ValueError, "shape", id="2x4"),
            pytest.param(NAN_CORNER, DOUBLE, ValueError, "finite", id="non-finite"),
            pytest.param(
                IDENTICAL, DOUBLE, ValueError, "singular", id="identical-src-points"
            ),
        ],
    )
    def test_raises_for_invalid_quads(
        self, source: Any, target: Any, error: type[Exception], match: str
    ) -> None:
        """Raise for unsupported inputs and for a singular perspective system."""
        with pytest.raises(error, match=match):
            _get_perspective_transform(source, target)


class TestPerspectiveTransform:
    @pytest.mark.parametrize(("source", "target"), [SPEED, SCALE, RANDOM[0]])
    @pytest.mark.parametrize("shape", [(16, 1, 2), (1, 16, 2)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matches_opencv(
        self,
        source: np.ndarray,
        target: np.ndarray,
        shape: tuple[int, ...],
        dtype: type[np.floating[Any]],
    ) -> None:
        """Project points like OpenCV, preserving point layout and dtype."""
        matrix = cv2.getPerspectiveTransform(source, target)
        # Convex combinations of the corners stay inside the source quad, away from
        # the horizon line where w -> 0 would amplify rounding without bound.
        weights = np.random.default_rng(7).dirichlet(np.ones(4), size=16)
        points = (weights @ source.astype(np.float64)).astype(dtype).reshape(shape)
        expected = cv2.perspectiveTransform(points, matrix)

        actual = _perspective_transform(points, matrix)

        assert actual.shape == shape
        assert actual.dtype == dtype
        # Both compute in float64, but OpenCV's build may fuse its sums into FMAs;
        # rtol=1e-9 absorbs that platform-dependent rounding with wide margin.
        tolerance = 4 * np.finfo(np.float32).eps if dtype == np.float32 else 1e-9
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_zeroes_points_whose_w_is_within_float32_epsilon(
        self, dtype: type[np.floating[Any]]
    ) -> None:
        """Map points with |w| <= FLT_EPSILON to the origin like OpenCV."""
        eps = np.finfo(np.float32).eps
        above = np.nextafter(eps, np.float32(np.inf))
        # The matrix sets w = x, so each point (x, x) maps to (1, 1) unless its x
        # falls within OpenCV's |w| threshold.
        matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
        x = np.array([0, eps, -eps, eps / 2, above, -above, 2], dtype=dtype)
        points = np.column_stack((x, x))[:, None, :]
        expected = np.array(
            [[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]], [[1, 1]], [[1, 1]], [[1, 1]]],
            dtype=dtype,
        )

        actual = _perspective_transform(points, matrix)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_identity_matrix_zeroes_non_finite_points(
        self, dtype: type[np.floating[Any]]
    ) -> None:
        """Map NaN and infinite points to the origin under the affine identity."""
        points = np.array([[[np.nan, 1]], [[np.inf, 1]]], dtype=dtype)

        projected = _perspective_transform(points, np.eye(3))

        np.testing.assert_array_equal(projected, np.zeros_like(points))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_projective_matrix_keeps_ieee_result_for_infinite_w(
        self, dtype: type[np.floating[Any]]
    ) -> None:
        """Zero points with NaN w but keep IEEE NaN for infinite w, like OpenCV."""
        # With w = x + 1, (nan, 1) and (1, inf) give NaN w, while (inf, 1) gives
        # infinite w, so inf * (1 / inf) evaluates to NaN.
        matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.float64)
        points = np.array([[[np.nan, 1]], [[1, np.inf]], [[np.inf, 1]]], dtype=dtype)
        expected = np.array([[[0, 0]], [[0, 0]], [[np.nan, np.nan]]], dtype=dtype)

        projected = _perspective_transform(points, matrix)

        np.testing.assert_array_equal(projected, expected)

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize(
        ("points", "matrix"),
        [
            pytest.param(
                np.array([[[1e308, 1.0]]], dtype=np.float64),
                np.diag([10.0, 1.0, 1.0]),
                id="float64-sum-overflow",
            ),
            pytest.param(
                np.array([[[2e38, 1.0]]], dtype=np.float32),
                np.diag([10.0, 1.0, 1.0]),
                id="float32-cast-overflow",
            ),
        ],
    )
    def test_overflow_gives_infinity_without_raising_or_warning(
        self, points: np.ndarray, matrix: np.ndarray
    ) -> None:
        """Return IEEE infinity for overflowing coordinates, silently like OpenCV.

        A caller running under ``np.seterr(over="raise")`` or ``-W error`` must not
        see the fallback fail where OpenCV returns infinity: the sums can overflow in
        float64, and the float64 result can overflow again when cast to float32.
        """
        expected = np.array([[[np.inf, 1.0]]], dtype=points.dtype)

        with np.errstate(over="raise", invalid="raise"):
            projected = _perspective_transform(points, matrix)

        np.testing.assert_array_equal(projected, expected)

    @pytest.mark.parametrize("matrix_dtype", [np.float32, np.int32])
    def test_accepts_non_float64_matrices(self, matrix_dtype: type[Any]) -> None:
        """Accept float32 and int32 matrices; the identity maps points to themselves."""
        points = np.array([[[1.5, -2.0]], [[300.25, 40.0]]], dtype=np.float32)

        projected = _perspective_transform(points, np.eye(3, dtype=matrix_dtype))

        np.testing.assert_array_equal(projected, points)

    @pytest.mark.parametrize("shape", [(0, 1, 2), (1, 0, 2)])
    def test_returns_none_for_empty_points(self, shape: tuple[int, ...]) -> None:
        """Return None for an empty point vector, as OpenCV's bindings do."""
        points = np.empty(shape, dtype=np.float32)

        projected = _perspective_transform(points, np.eye(3))

        assert projected is None

    @pytest.mark.parametrize(
        ("points", "matrix", "error"),
        [
            pytest.param(POINTS[:, 0], np.eye(3), ValueError, id="n-by-2"),
            pytest.param(POINTS.astype(np.int32), np.eye(3), TypeError, id="int32"),
            pytest.param(POINTS.tolist(), np.eye(3), TypeError, id="list-points"),
            pytest.param(POINTS, np.eye(3)[:2], ValueError, id="2-by-3-matrix"),
        ],
    )
    def test_raises_for_unsupported_inputs(
        self, points: Any, matrix: Any, error: type[Exception]
    ) -> None:
        """Reject point layouts, dtypes and matrices outside the 2-D contract."""
        with pytest.raises(error, match=r"float32|float64|shape|array"):
            _perspective_transform(points, matrix)
