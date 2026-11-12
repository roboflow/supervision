"""Tests for private drawing fallbacks."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from supervision._cv2._drawing import (
    _circle,
    _clip_line,
    _draw_contours,
    _ellipse,
    _fill_poly,
    _line,
    _polylines,
    _rectangle,
)

cv2: Any
try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    cv2 = None

requires_cv2 = pytest.mark.skipif(
    cv2 is None,
    reason="OpenCV is required as the reference implementation for this test",
)


def _assert_draws_on_canvas(result: np.ndarray, image: np.ndarray) -> None:
    """Assert that a fallback drawing operation mutates and returns its image."""
    assert result is image
    assert np.any(image)


@requires_cv2
def test_fallback_line_draws_and_matches_opencv_extent() -> None:
    """Draw a line in place with the same pixels as OpenCV."""
    actual = np.zeros((20, 20, 3), dtype=np.uint8)
    expected = np.zeros_like(actual)

    result = _line(actual, (2, 3), (15, 12), (10, 20, 30), thickness=2)
    cv2.line(expected, (2, 3), (15, 12), (10, 20, 30), thickness=2)

    _assert_draws_on_canvas(result, actual)
    np.testing.assert_array_equal(actual, expected)


class TestFallbackThickLine:
    """Thick lines, polylines and contour outlines painted pixel for pixel like
    OpenCV."""

    @pytest.mark.parametrize(
        "points",
        [
            pytest.param([], id="no-polylines"),
            pytest.param([np.empty((0, 2), dtype=np.int32)], id="empty-polyline"),
        ],
    )
    def test_empty_polylines_leave_image_unchanged(
        self, points: list[np.ndarray]
    ) -> None:
        """Treat absent and empty thick outlines as in-place no-ops."""
        image = np.full((8, 8, 3), 17, dtype=np.uint8)
        expected = image.copy()

        result = _polylines(image, points, True, (1, 2, 3), thickness=3)

        assert result is image
        np.testing.assert_array_equal(image, expected)

    @requires_cv2
    @pytest.mark.parametrize(
        ("pt1", "pt2", "thickness"),
        [
            pytest.param((2, 3), (15, 12), 2, id="even-thickness-diagonal"),
            pytest.param((3, 20), (36, 20), 5, id="odd-thickness-horizontal"),
            pytest.param((20, 2), (20, 37), 4, id="vertical"),
            pytest.param((5, 30), (34, 7), 9, id="steep-and-wide"),
            pytest.param((18, 18), (18, 18), 6, id="single-point-is-a-disc"),
            pytest.param((-11, 34), (63, 2), 18, id="clipped-at-canvas-edges"),
            pytest.param((-30, -5), (70, -2), 12, id="outside-canvas-cap-reaches-in"),
        ],
    )
    def test_line_matches_opencv(
        self, pt1: tuple[int, int], pt2: tuple[int, int], thickness: int
    ) -> None:
        """Paint the quadrilateral and both round caps where OpenCV does."""
        actual = np.zeros((40, 40), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _line(actual, pt1, pt2, 1, thickness=thickness)
        cv2.line(expected, pt1, pt2, 1, thickness=thickness)

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    @pytest.mark.parametrize("is_closed", [True, False])
    @pytest.mark.parametrize("thickness", [2, 3, 7])
    def test_polylines_matches_opencv(self, is_closed: bool, thickness: int) -> None:
        """Join segments and cap open ends exactly as OpenCV's `PolyLine`."""
        points = [np.array([[4, 5], [30, 8], [26, 33], [9, 24]], dtype=np.int32)]
        actual = np.zeros((40, 40, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _polylines(actual, points, is_closed, (1, 2, 3), thickness=thickness)
        cv2.polylines(expected, points, is_closed, (1, 2, 3), thickness=thickness)

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    @pytest.mark.parametrize(
        ("points", "is_closed"),
        [
            pytest.param([[12, 12]], True, id="closed-single-point"),
            pytest.param([[12, 12]], False, id="open-single-point-draws-nothing"),
            pytest.param([[6, 6], [6, 6], [20, 9]], False, id="repeated-point"),
        ],
    )
    def test_polylines_degenerate_input_matches_opencv(
        self, points: list[list[int]], is_closed: bool
    ) -> None:
        """Handle one-point and repeated-point polylines like OpenCV."""
        polygon = [np.array(points, dtype=np.int32)]
        actual = np.zeros((25, 25), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _polylines(actual, polygon, is_closed, 1, thickness=4)
        cv2.polylines(expected, polygon, is_closed, 1, thickness=4)

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    def test_draw_contours_outline_matches_opencv(self) -> None:
        """Stroke offset contours with the same thick closed outline as OpenCV."""
        contours = [
            np.array([[[2, 2]], [[17, 3]], [[15, 16]], [[3, 14]]], dtype=np.int32),
            np.array([[[22, 20]], [[30, 34]], [[18, 30]]], dtype=np.int32),
        ]
        actual = np.zeros((40, 40, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _draw_contours(actual, contours, -1, (5, 6, 7), thickness=3, offset=(2, 1))
        cv2.drawContours(expected, contours, -1, (5, 6, 7), thickness=3, offset=(2, 1))

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    def test_random_lines_match_opencv(self) -> None:
        """Match OpenCV on random thick lines, many of them leaving the canvas."""
        rng = np.random.default_rng(2026)

        for _ in range(200):
            height, width = (int(value) for value in rng.integers(5, 60, size=2))
            pt1 = tuple(int(value) for value in rng.integers(-20, 80, size=2))
            pt2 = tuple(int(value) for value in rng.integers(-20, 80, size=2))
            thickness = int(rng.integers(2, 16))
            actual = np.zeros((height, width), dtype=np.uint8)
            expected = np.zeros_like(actual)

            _line(actual, pt1, pt2, 1, thickness=thickness)
            cv2.line(expected, pt1, pt2, 1, thickness=thickness)

            np.testing.assert_array_equal(actual, expected)

    def test_line_leaves_far_pixels_untouched(self) -> None:
        """Keep a thick line's paint within its caps without OpenCV installed."""
        image = np.zeros((30, 30), dtype=np.uint8)

        _line(image, (10, 15), (20, 15), 1, thickness=4)

        assert image[15, 15] == 1
        assert image[15, 8] == 1
        assert not image[:, :7].any()
        assert not image[:, 24:].any()
        assert not image[:12].any()


class TestClipLine:
    """Clipping against bounds that cannot contain a pixel."""

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            pytest.param(0, 8, id="zero-width"),
            pytest.param(8, 0, id="zero-height"),
        ],
    )
    def test_empty_bounds_reject_segment(self, width: int, height: int) -> None:
        """Reject an empty clipping rectangle without changing either endpoint."""
        first, second = (1, 2), (5, 6)

        result = _clip_line(width, height, first, second)

        assert result == (False, first, second)


def test_fallback_rectangle_fills_the_requested_region() -> None:
    """Fill a rectangle using OpenCV's inclusive corner convention."""
    image = np.zeros((12, 14, 3), dtype=np.uint8)

    result = _rectangle(image, (2, 3), (7, 8), (1, 2, 3), thickness=-1)

    _assert_draws_on_canvas(result, image)
    np.testing.assert_array_equal(
        image[3:9, 2:8], np.full((6, 6, 3), (1, 2, 3), dtype=np.uint8)
    )


@requires_cv2
@pytest.mark.parametrize(
    ("shape", "point_1", "point_2", "thickness"),
    [
        pytest.param((24, 28, 3), (6, 5), (20, 17), 0, id="thickness-0"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 1, id="thickness-1"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 2, id="thickness-2"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 3, id="thickness-3"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 4, id="thickness-4"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 6, id="thickness-6"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 7, id="thickness-7"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 8, id="thickness-8"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), 25, id="thickness-over-box"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), -2, id="negative-fills"),
        pytest.param((24, 28, 3), (6, 5), (20, 17), -5, id="negative-fills-more"),
        pytest.param((24, 28, 3), (6, 9), (20, 9), 3, id="zero-height"),
        pytest.param((24, 28, 3), (9, 5), (9, 17), 3, id="zero-width"),
        pytest.param((24, 28, 3), (20, 17), (6, 5), 3, id="reversed-corners"),
        pytest.param((24, 28), (6, 5), (20, 17), 3, id="two-dimensional-image"),
        pytest.param((1, 1, 3), (0, 0), (0, 0), 3, id="one-pixel-canvas"),
        pytest.param((24, 28, 3), (40, 40), (60, 60), 3, id="off-canvas"),
        pytest.param((16, 16, 3), (-4, 2), (13, 20), 5, id="clipped-left-bottom"),
    ],
)
def test_fallback_rectangle_matches_opencv(
    shape: tuple[int, ...],
    point_1: tuple[int, int],
    point_2: tuple[int, int],
    thickness: int,
) -> None:
    """Match OpenCV's rectangle raster for filled, thick, clipped, and edge cases."""
    actual = np.zeros(shape, dtype=np.uint8)
    expected = np.zeros_like(actual)

    _rectangle(actual, point_1, point_2, (1, 2, 3), thickness=thickness)
    cv2.rectangle(expected, point_1, point_2, (1, 2, 3), thickness=thickness)

    np.testing.assert_array_equal(actual, expected)


def test_fallback_rectangle_border_is_centered_on_the_edges() -> None:
    """Spread a `thickness=2` border one pixel outside and inside each edge."""
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    _rectangle(image, (20, 20), (80, 60), (1, 2, 3), thickness=2)

    painted = np.asarray(image.any(axis=2))
    rows, columns = np.where(painted)
    assert int(painted.sum()) == 596
    assert (columns.min(), columns.max(), rows.min(), rows.max()) == (19, 81, 19, 61)
    assert painted[40, 19]
    assert not painted[40, 22]


def test_fallback_rectangle_numpy_integer_thickness_matches_python_int() -> None:
    """Keep a NumPy integer `thickness` from wrapping around in the corner radius."""
    actual = np.zeros((120, 140, 3), dtype=np.uint8)
    expected = np.zeros_like(actual)

    _rectangle(actual, (50, 50), (90, 80), (1, 2, 3), thickness=np.uint8(40))
    _rectangle(expected, (50, 50), (90, 80), (1, 2, 3), thickness=40)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("thickness", [10_000, 32_767])
def test_fallback_rectangle_huge_thickness_covers_a_small_canvas(
    thickness: int,
) -> None:
    """Paint a small canvas without allocating arrays sized by the border radius."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    _rectangle(image, (2, 2), (5, 5), (1, 2, 3), thickness=thickness)

    assert image.any(axis=2).all()


@pytest.mark.parametrize(
    ("thickness", "error"),
    [
        pytest.param(2.5, TypeError, id="float"),
        pytest.param(32_768, ValueError, id="above-opencv-maximum"),
    ],
)
def test_fallback_rectangle_rejects_thickness_opencv_rejects(
    thickness: float, error: type[Exception]
) -> None:
    """Raise for a non-integer or oversized `thickness` instead of drawing."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(error):
        _rectangle(image, (2, 2), (5, 5), (1, 2, 3), thickness=thickness)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(
            lambda image, thickness: _line(
                image, (2, 2), (5, 5), (1, 2, 3), thickness=thickness
            ),
            id="line",
        ),
        pytest.param(
            lambda image, thickness: _polylines(
                image,
                [np.array([[1, 1], [6, 2], [4, 6]], dtype=np.int32)],
                True,
                (1, 2, 3),
                thickness=thickness,
            ),
            id="polylines",
        ),
        pytest.param(
            lambda image, thickness: _draw_contours(
                image,
                [np.array([[[1, 1]], [[6, 2]], [[4, 6]]], dtype=np.int32)],
                -1,
                (1, 2, 3),
                thickness=thickness,
            ),
            id="draw-contours",
        ),
    ],
)
@pytest.mark.parametrize(
    ("thickness", "error"),
    [
        pytest.param(2.5, TypeError, id="float"),
        pytest.param(32_768, ValueError, id="above-opencv-maximum"),
        pytest.param(10**9, ValueError, id="far-above-opencv-maximum"),
    ],
)
def test_fallback_thick_strokes_reject_thickness_opencv_rejects(
    draw: Any, thickness: float, error: type[Exception]
) -> None:
    """Raise before rasterizing a stroke whose `thickness` OpenCV would reject."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(error):
        draw(image, thickness)

    assert not image.any()


@pytest.mark.parametrize("thickness", [np.uint8(4), np.int64(4)])
def test_fallback_line_numpy_integer_thickness_matches_python_int(
    thickness: np.integer,
) -> None:
    """Draw a NumPy integer `thickness` exactly like the equal Python int."""
    actual = np.zeros((20, 20), dtype=np.uint8)
    expected = np.zeros_like(actual)

    _line(actual, (3, 4), (15, 12), 1, thickness=thickness)
    _line(expected, (3, 4), (15, 12), 1, thickness=4)

    np.testing.assert_array_equal(actual, expected)


@requires_cv2
def test_fallback_scalar_color_matches_opencv_channel_padding() -> None:
    """Pad short scalar colors in BGR channel order like OpenCV."""
    actual = np.zeros((8, 8, 3), dtype=np.uint8)
    expected = np.zeros_like(actual)

    _rectangle(actual, (1, 1), (3, 3), 7, thickness=-1)
    cv2.rectangle(expected, (1, 1), (3, 3), 7, thickness=-1)

    np.testing.assert_array_equal(actual, expected)


def test_fallback_circle_draws_a_filled_disk() -> None:
    """Fill a circle while keeping pixels outside its radius unchanged."""
    image = np.zeros((15, 15, 3), dtype=np.uint8)

    result = _circle(image, (7, 7), 4, (1, 2, 3), thickness=-1)

    _assert_draws_on_canvas(result, image)
    assert tuple(image[7, 7]) == (1, 2, 3)
    assert tuple(image[0, 0]) == (0, 0, 0)


def test_fallback_thick_circle_is_centred_on_its_radius() -> None:
    """Spread a thick circle outline across its radius, as OpenCV does."""
    image = np.zeros((31, 31, 3), dtype=np.uint8)

    _circle(image, (15, 15), 10, (1, 2, 3), thickness=5)

    # The stroke covers roughly radius 7.5 to 12.5 around the centre.
    assert tuple(image[15, 15 + 12]) == (1, 2, 3)
    assert tuple(image[15 + 12, 15]) == (1, 2, 3)
    assert tuple(image[15, 15 + 6]) == (0, 0, 0)


@requires_cv2
@pytest.mark.parametrize("thickness", [2, 3, 5])
@pytest.mark.parametrize("radius", [1, 4, 9])
def test_fallback_thick_circle_matches_opencv(thickness: int, radius: int) -> None:
    """Draw thick circle outlines pixel-for-pixel like OpenCV."""
    assert cv2 is not None
    actual = np.zeros((30, 30, 3), dtype=np.uint8)
    expected = np.zeros_like(actual)

    _circle(actual, (14, 15), radius, (255, 0, 0), thickness)
    cv2.circle(expected, (14, 15), radius, (255, 0, 0), thickness)

    np.testing.assert_array_equal(actual, expected)


class TestFallbackThinAndFilledCircle:
    """One-pixel and filled circles painted pixel for pixel like OpenCV's `Circle`."""

    @requires_cv2
    @pytest.mark.parametrize("thickness", [-1, 0, 1])
    @pytest.mark.parametrize(
        ("center", "radius"),
        [
            pytest.param((14, 15), 0, id="zero-radius"),
            pytest.param((14, 15), 1, id="radius-one"),
            pytest.param((14, 15), 6, id="inside"),
            pytest.param((14, 15), 13, id="touching-edges"),
            pytest.param((3, 26), 9, id="clipped-corner"),
            pytest.param((-4, 12), 7, id="center-off-canvas"),
            pytest.param((60, 60), 5, id="fully-off-canvas"),
        ],
    )
    def test_matches_opencv(
        self, center: tuple[int, int], radius: int, thickness: int
    ) -> None:
        """Paint the same pixels as OpenCV for outlines and filled disks."""
        actual = np.zeros((30, 30, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _circle(actual, center, radius, (255, 0, 9), thickness)
        cv2.circle(expected, center, radius, (255, 0, 9), thickness)

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    def test_random_circles_match_opencv(self) -> None:
        """Match OpenCV on random thin and filled circles, many leaving the canvas."""
        rng = np.random.default_rng(2026)

        for _ in range(200):
            height, width = (int(value) for value in rng.integers(5, 60, size=2))
            center = tuple(int(value) for value in rng.integers(-30, 90, size=2))
            radius = int(rng.integers(0, 50))
            thickness = int(rng.choice([-1, 0, 1]))
            actual = np.zeros((height, width), dtype=np.uint8)
            expected = np.zeros_like(actual)

            _circle(actual, center, radius, 1, thickness)
            cv2.circle(expected, center, radius, 1, thickness)

            np.testing.assert_array_equal(actual, expected)

    def test_outline_follows_midpoint_steps(self) -> None:
        """Draw a radius-2 outline as OpenCV's midpoint diamond, no OpenCV needed."""
        image = np.zeros((5, 5), dtype=np.uint8)

        _circle(image, (2, 2), 2, 1, thickness=1)

        expected = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(image, expected)

    @pytest.mark.parametrize("thickness", [-1, 1])
    def test_huge_circle_off_canvas_draws_nothing(self, thickness: int) -> None:
        """Skip a circle whose bounds miss the canvas without walking its radius."""
        image = np.zeros((32, 32), dtype=np.uint8)

        _circle(image, (2_000_000, 0), 1_000_000, 1, thickness=thickness)

        assert not np.any(image)

    def test_rejects_negative_radius(self) -> None:
        """Reject a negative radius before drawing any pixels, as OpenCV does."""
        image = np.zeros((15, 15), dtype=np.uint8)

        with pytest.raises(ValueError, match="Circle radius must be non-negative"):
            _circle(image, (7, 7), -1, 1)

        assert not np.any(image)


class TestFallbackThinStroke:
    """One-pixel lines, polylines and contour outlines painted like OpenCV's 8-connected
    `Line`."""

    @requires_cv2
    @pytest.mark.parametrize(
        ("pt1", "pt2"),
        [
            pytest.param((25, 11), (33, 4), id="shallow-diagonal"),
            pytest.param((47, 46), (62, 12), id="steep"),
            pytest.param((3, 20), (36, 20), id="horizontal"),
            pytest.param((18, 18), (18, 18), id="single-point"),
            pytest.param((36, 76), (-20, 24), id="clipped-at-two-edges"),
            pytest.param((-30, -5), (70, -2), id="fully-off-canvas"),
            pytest.param(
                (-1_000_000_000, -1_000_000_000),
                (2, 1),
                id="clipped-with-float-rounding",
            ),
        ],
    )
    def test_line_matches_opencv(
        self, pt1: tuple[int, int], pt2: tuple[int, int]
    ) -> None:
        """Step the clipped segment exactly as OpenCV's `LineIterator`."""
        actual = np.zeros((60, 70, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _line(actual, pt1, pt2, (1, 2, 3), thickness=1)
        cv2.line(expected, pt1, pt2, (1, 2, 3), thickness=1)

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    def test_random_lines_match_opencv(self) -> None:
        """Match OpenCV on random thin lines, many of them leaving the canvas."""
        rng = np.random.default_rng(2026)

        for _ in range(200):
            height, width = (int(value) for value in rng.integers(5, 60, size=2))
            pt1 = tuple(int(value) for value in rng.integers(-40, 100, size=2))
            pt2 = tuple(int(value) for value in rng.integers(-40, 100, size=2))
            actual = np.zeros((height, width), dtype=np.uint8)
            expected = np.zeros_like(actual)

            _line(actual, pt1, pt2, 1, thickness=1)
            cv2.line(expected, pt1, pt2, 1, thickness=1)

            np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    @pytest.mark.parametrize("is_closed", [True, False])
    @pytest.mark.parametrize(
        "points",
        [
            pytest.param([[4, 5], [30, 8], [26, 33], [9, 24]], id="quadrilateral"),
            pytest.param([[-6, 3], [45, 17], [12, 50]], id="clipped"),
            pytest.param([[12, 12]], id="single-point"),
            pytest.param([[6, 6], [6, 6], [20, 9]], id="repeated-point"),
        ],
    )
    def test_polylines_matches_opencv(
        self, points: list[list[int]], is_closed: bool
    ) -> None:
        """Draw every polyline edge as OpenCV's thin line."""
        polygon = [np.array(points, dtype=np.int32)]
        actual = np.zeros((40, 40), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _polylines(actual, polygon, is_closed, 1, thickness=1)
        cv2.polylines(expected, polygon, is_closed, 1, thickness=1)

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    @pytest.mark.parametrize("thickness", [0, 1])
    def test_draw_contours_outline_matches_opencv(self, thickness: int) -> None:
        """Stroke offset contours, including a single point, like OpenCV."""
        contours = [
            np.array([[[2, 2]], [[17, 3]], [[15, 16]], [[3, 14]]], dtype=np.int32),
            np.array([[[22, 20]], [[45, 34]], [[18, 30]]], dtype=np.int32),
            np.array([[[30, 5]]], dtype=np.int32),
        ]
        actual = np.zeros((40, 40, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _draw_contours(
            actual, contours, -1, (5, 6, 7), thickness=thickness, offset=(2, 1)
        )
        cv2.drawContours(
            expected, contours, -1, (5, 6, 7), thickness=thickness, offset=(2, 1)
        )

        np.testing.assert_array_equal(actual, expected)

    def test_line_follows_bresenham_steps(self) -> None:
        """Step a shallow line with OpenCV's tie-breaking, no OpenCV needed."""
        image = np.zeros((3, 5), dtype=np.uint8)

        _line(image, (0, 0), (4, 2), 1, thickness=1)

        expected = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(image, expected)


def test_fallback_ellipse_draws_a_rotated_outline() -> None:
    """Draw a rotated ellipse outline on a multi-channel image."""
    image = np.zeros((24, 24, 3), dtype=np.uint8)

    result = _ellipse(
        image,
        center=(12, 12),
        axes=(7, 4),
        angle=30,
        startAngle=0,
        endAngle=360,
        color=(1, 2, 3),
        thickness=2,
    )

    _assert_draws_on_canvas(result, image)


@pytest.mark.parametrize(
    "axes",
    [
        pytest.param((-1, 4), id="negative-horizontal-axis"),
        pytest.param((4, -1), id="negative-vertical-axis"),
    ],
)
def test_fallback_ellipse_rejects_negative_axes(axes: tuple[int, int]) -> None:
    """Reject a negative semi-axis before drawing any pixels."""
    image = np.zeros((15, 15), dtype=np.uint8)

    with pytest.raises(ValueError, match="Ellipse axes must be non-negative"):
        _ellipse(image, (7, 7), axes, 0, 0, 360, 1)

    assert not np.any(image)


def test_fallback_ellipse_fills_partial_arc_as_pie_slice() -> None:
    """Include the center in a partial filled arc and leave the opposite side empty."""
    image = np.zeros((21, 21), dtype=np.uint8)

    result = _ellipse(image, (10, 10), (7, 5), 0, 0, 90, 1, thickness=-1)

    assert result is image
    assert image[10, 10] == 1
    assert image[10, 15] == 1
    assert image[5, 10] == 0
    assert np.count_nonzero(image) > 1


class TestFallbackEllipse:
    """Ellipse arcs and fills painted pixel for pixel like OpenCV."""

    @requires_cv2
    @pytest.mark.parametrize(
        ("center", "axes", "angle", "start_angle", "end_angle", "thickness"),
        [
            pytest.param((50, 40), (30, 20), 0, 0, 360, 1, id="thin-full"),
            pytest.param((50, 40), (30, 20), 0, 0, 360, 2, id="thick-full"),
            pytest.param((50, 40), (30, 20), 0, 0, 360, 7, id="very-thick-full"),
            pytest.param((60, 60), (40, 25), 30, 0, 360, 3, id="rotated"),
            pytest.param((50, 50), (35, 20), 20, 0, 90, 3, id="quarter-arc"),
            pytest.param((50, 50), (35, 20), 20, 30, 200, 5, id="partial-arc"),
            pytest.param((50, 50), (35, 20), 20, 200, 400, 4, id="arc-past-full-turn"),
            pytest.param((50, 40), (30, 20), 0, 0, 360, -1, id="filled-full"),
            pytest.param((50, 50), (35, 20), 20, 30, 200, -1, id="filled-pie"),
            pytest.param(
                (50, 40),
                (30, 20),
                0,
                0,
                359.6,
                -1,
                id="filled-fractional-arc-rounds-to-full-turn",
            ),
            pytest.param(
                (50, 40),
                (30, 20),
                0,
                0.4,
                360,
                -1,
                id="filled-fractional-start-rounds-to-full-turn",
            ),
            pytest.param(
                (50, 50),
                (35, 20),
                20,
                0.6,
                360.4,
                -1,
                id="filled-fractional-arc-rounds-to-pie",
            ),
            pytest.param((5, 5), (20, 12), 25, 0, 360, 3, id="center-outside-top-left"),
            pytest.param((150, 40), (25, 15), 25, 0, 360, 3, id="center-past-right"),
            pytest.param((40, 40), (0, 20), 25, 0, 360, 3, id="zero-semi-axis"),
            pytest.param((40, 40), (0, 0), 25, 0, 360, 3, id="zero-both-axes"),
            pytest.param((40, 40), (30, 20), 12.5, 45.5, 300.5, 2, id="half-degrees"),
        ],
    )
    def test_ellipse_matches_opencv(
        self,
        center: tuple[int, int],
        axes: tuple[int, int],
        angle: float,
        start_angle: float,
        end_angle: float,
        thickness: int,
    ) -> None:
        """Match OpenCV's `Ellipse2Poly` outline and fill paths."""
        actual = np.zeros((80, 100, 3), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _ellipse(
            actual,
            center,
            axes,
            angle,
            start_angle,
            end_angle,
            (1, 2, 3),
            thickness=thickness,
        )
        cv2.ellipse(
            expected,
            center,
            axes,
            angle,
            start_angle,
            end_angle,
            (1, 2, 3),
            thickness=thickness,
        )

        np.testing.assert_array_equal(actual, expected)

    @requires_cv2
    def test_random_ellipses_match_opencv(self) -> None:
        """Match OpenCV on random arcs, including partly off-canvas ellipses."""
        rng = np.random.default_rng(20261016)

        for _ in range(200):
            height, width = (int(value) for value in rng.integers(5, 90, size=2))
            center = (
                int(rng.integers(-20, width + 20)),
                int(rng.integers(-20, height + 20)),
            )
            axes = (
                int(rng.integers(0, 60)),
                int(rng.integers(0, 60)),
            )
            angle = int(rng.integers(-720, 720))
            start_angle = int(rng.integers(-360, 721))
            end_angle = int(rng.integers(-360, 721))
            thickness = int(rng.integers(-3, 16)) or 1
            actual = np.zeros((height, width), dtype=np.uint8)
            expected = np.zeros_like(actual)

            _ellipse(
                actual,
                center,
                axes,
                angle,
                start_angle,
                end_angle,
                1,
                thickness=thickness,
            )
            cv2.ellipse(
                expected,
                center,
                axes,
                angle,
                start_angle,
                end_angle,
                1,
                thickness=thickness,
            )

            np.testing.assert_array_equal(actual, expected)


def test_fallback_polylines_respects_closed_flag() -> None:
    """Close a polyline only when the caller requests a closed path."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    points = [np.array([[2, 2], [8, 2], [8, 8]], dtype=np.int32)]

    result = _polylines(image, points, isClosed=True, color=(1, 2, 3), thickness=1)

    _assert_draws_on_canvas(result, image)
    assert tuple(image[5, 5]) == (1, 2, 3)


def test_fallback_fill_poly_fills_multiple_channels() -> None:
    """Fill a polygon with a BGR color in place."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    polygon = np.array([[2, 2], [9, 2], [6, 9]], dtype=np.int32)

    result = _fill_poly(image, [polygon], color=(4, 5, 6))

    _assert_draws_on_canvas(result, image)
    assert tuple(image[4, 5]) == (4, 5, 6)


@requires_cv2
def test_fallback_fill_poly_matches_opencv_for_mask_conversion() -> None:
    """Fill an integer polygon with the same inclusive mask boundary as OpenCV."""
    polygon = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.int32)
    actual = np.zeros((10, 10), dtype=np.uint8)
    expected = np.zeros((10, 10), dtype=np.uint8)

    _fill_poly(actual, [polygon], color=(1,))
    cv2.fillPoly(expected, [polygon], color=(1,))

    np.testing.assert_array_equal(actual, expected)


@requires_cv2
def test_fallback_fill_poly_matches_opencv_boundary_pixels() -> None:
    """Include the boundary pixel OpenCV paints for a narrow triangle."""
    polygon = np.array([[10, 9], [9, 9], [7, 11]], dtype=np.int32)
    actual = np.zeros((12, 12), dtype=np.uint8)
    expected = np.zeros((12, 12), dtype=np.uint8)

    _fill_poly(actual, [polygon], color=(1,))
    cv2.fillPoly(expected, [polygon], color=(1,))

    np.testing.assert_array_equal(actual, expected)


@requires_cv2
def test_fallback_fill_poly_matches_opencv_for_centered_polygons() -> None:
    """Match OpenCV for polygons confined to the image bounds."""
    rng = np.random.default_rng(1000)
    height = width = 200
    center = np.array([width / 2, height / 2])

    for _ in range(300):
        vertex_count = int(rng.integers(3, 12))
        angles = np.sort(rng.uniform(0, 2 * np.pi, vertex_count))
        radii = rng.uniform(30, 90, vertex_count)
        polygon = np.column_stack(
            (
                center[0] + radii * np.cos(angles),
                center[1] + radii * np.sin(angles),
            )
        ).round()
        polygon = polygon.astype(np.int32)
        actual = np.zeros((height, width), dtype=np.uint8)
        expected = np.zeros((height, width), dtype=np.uint8)

        _fill_poly(actual, [polygon], color=(1,))
        cv2.fillPoly(expected, [polygon], color=(1,))

        np.testing.assert_array_equal(actual, expected)


@requires_cv2
def test_fallback_fill_poly_matches_opencv_for_nested_contours() -> None:
    """Apply even-odd filling across multiple contours like OpenCV."""
    outer = np.array([[1, 1], [9, 1], [9, 9], [1, 9]], dtype=np.int32)
    inner = np.array([[3, 3], [7, 3], [7, 7], [3, 7]], dtype=np.int32)
    actual = np.zeros((11, 11), dtype=np.uint8)
    expected = np.zeros((11, 11), dtype=np.uint8)

    _fill_poly(actual, [outer, inner], color=(1,))
    cv2.fillPoly(expected, [outer, inner], color=(1,))

    np.testing.assert_array_equal(actual, expected)


@requires_cv2
@pytest.mark.parametrize("entry_point", ["fill-poly", "draw-contours"])
@pytest.mark.parametrize(
    ("points", "offset"),
    [
        pytest.param(
            ((-5, -4), (18, -4), (18, 16), (-5, 16)),
            (2, 1),
            id="crosses-all-canvas-edges-with-positive-offset",
        ),
        pytest.param(
            ((1, 1), (18, 2), (15, 16), (3, 12)),
            (-2, -1),
            id="crosses-right-and-bottom-with-negative-offset",
        ),
    ],
)
def test_fallback_filled_polygons_match_opencv_when_clipped_and_offset(
    entry_point: str,
    points: tuple[tuple[int, int], ...],
    offset: tuple[int, int],
) -> None:
    """Match OpenCV for filled polygons crossing canvas edges with offsets."""
    polygon = np.asarray(points, dtype=np.int32)
    contour = polygon.reshape(-1, 1, 2)
    actual = np.zeros((12, 14), dtype=np.uint8)
    expected = np.zeros_like(actual)

    if entry_point == "fill-poly":
        _fill_poly(actual, [polygon], color=(1,), offset=offset)
        cv2.fillPoly(expected, [polygon], color=(1,), offset=offset)
    else:
        _draw_contours(actual, [contour], -1, (1,), thickness=-1, offset=offset)
        cv2.drawContours(expected, [contour], -1, (1,), thickness=-1, offset=offset)

    np.testing.assert_array_equal(actual, expected)


@requires_cv2
@pytest.mark.parametrize(
    "points",
    [
        pytest.param(
            ((70, 34), (28, 68), (0, -21), (67, -23)),
            id="slanted-edges-leave-three-sides",
        ),
        pytest.param(
            ((22, 89), (16, 1), (-2, 71), (-5, 42)),
            id="edge-clipped-to-a-single-point",
        ),
        pytest.param(
            ((73, -5), (79, -14), (50, 88)),
            id="edge-rejected-beyond-a-corner",
        ),
        pytest.param(
            ((8, 67), (-6, -30), (-21, 56), (-26, 83), (20, 36)),
            id="self-intersecting-off-canvas",
        ),
    ],
)
def test_fallback_fill_poly_matches_opencv_for_edges_leaving_the_canvas(
    points: tuple[tuple[int, int], ...],
) -> None:
    """Fill polygons whose slanted edges leave the canvas exactly like OpenCV."""
    polygon = np.asarray(points, dtype=np.int32)
    actual = np.zeros((50, 60), dtype=np.uint8)
    expected = np.zeros_like(actual)

    _fill_poly(actual, [polygon], color=(1,))
    cv2.fillPoly(expected, [polygon], color=(1,))

    np.testing.assert_array_equal(actual, expected)


@requires_cv2
def test_fallback_random_filled_polygons_match_opencv() -> None:
    """Match OpenCV on random filled polygons, most of them leaving the canvas."""
    rng = np.random.default_rng(2026)

    for _ in range(200):
        height, width = (int(value) for value in rng.integers(1, 60, size=2))
        polygons = [
            rng.integers(-40, 100, size=(int(rng.integers(1, 9)), 2)).astype(np.int32)
            for _ in range(int(rng.integers(1, 3)))
        ]
        actual = np.zeros((height, width), dtype=np.uint8)
        expected = np.zeros_like(actual)

        _fill_poly(actual, polygons, color=(1,))
        cv2.fillPoly(expected, polygons, color=(1,))

        np.testing.assert_array_equal(actual, expected)


def test_fallback_draw_contours_fills_selected_contour() -> None:
    """Fill the selected contour when drawContours receives a negative thickness."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    contours = [np.array([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)]

    result = _draw_contours(
        image, contours, contourIdx=0, color=(7, 8, 9), thickness=-1
    )

    _assert_draws_on_canvas(result, image)
    assert tuple(image[5, 5]) == (7, 8, 9)


@requires_cv2
def test_fallback_draw_contours_filled_matches_opencv() -> None:
    """Match OpenCV's filled-contour boundary pixels and nested contours."""
    outer = np.array([[[1, 1]], [[9, 1]], [[9, 9]], [[1, 9]]], dtype=np.int32)
    inner = np.array([[[3, 3]], [[7, 3]], [[7, 7]], [[3, 7]]], dtype=np.int32)
    actual = np.zeros((11, 11), dtype=np.uint8)
    expected = np.zeros((11, 11), dtype=np.uint8)

    _draw_contours(actual, [outer, inner], contourIdx=-1, color=(1,), thickness=-1)
    cv2.drawContours(expected, [outer, inner], -1, (1,), thickness=-1)

    np.testing.assert_array_equal(actual, expected)


def test_fallback_draw_contours_rejects_hierarchy() -> None:
    """Reject hierarchy-guided drawing the flat fallback cannot reproduce."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    contours = [np.array([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)]
    hierarchy = np.array([[[-1, -1, -1, -1]]], dtype=np.int32)

    with pytest.raises(ValueError, match="hierarchy"):
        _draw_contours(
            image, contours, contourIdx=0, color=(7, 8, 9), hierarchy=hierarchy
        )


def test_draw_line_consumer_uses_fallback_without_opencv() -> None:
    """Exercise a Supervision drawing consumer in a cv2-blocked process."""
    source = """
import numpy as np
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())

from supervision.draw.color import Color
from supervision.draw.utils import draw_line
from supervision.geometry.core import Point

image = np.zeros((12, 12, 3), dtype=np.uint8)
result = draw_line(image, Point(2, 2), Point(8, 8), color=Color.RED)
assert result is image
assert np.any(result)
"""
    environment = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


@requires_cv2
@pytest.mark.parametrize(
    ("dtype", "color"),
    [
        pytest.param(np.uint8, (-10, 300, 50), id="uint8-out-of-bounds-multichannel"),
        pytest.param(np.uint8, 300, id="uint8-out-of-bounds-scalar"),
        pytest.param(np.uint8, -10, id="uint8-negative-scalar"),
        pytest.param(np.uint8, (128.6, 200.2, 0.0), id="uint8-float-multichannel"),
        pytest.param(np.int8, (-150, 200, 50), id="int8-out-of-bounds"),
        pytest.param(np.uint16, (-10, 70000, 500), id="uint16-out-of-bounds"),
        pytest.param(np.float32, (-10.4, 300.6, 50.2), id="float32-values"),
    ],
)
@pytest.mark.parametrize("shape", [(20, 20), (20, 20, 3)])
def test_fallback_drawing_colors_saturate_and_round_like_opencv(
    dtype: np.dtype[Any], color: Any, shape: tuple[int, ...]
) -> None:
    """Saturate out-of-bounds drawing colors and round floats matching OpenCV."""
    actual_rect = np.zeros(shape, dtype=dtype)
    expected_rect = np.zeros(shape, dtype=dtype)
    _rectangle(actual_rect, (2, 2), (8, 8), color, -1)
    cv2.rectangle(expected_rect, (2, 2), (8, 8), color, -1)
    np.testing.assert_array_equal(actual_rect, expected_rect)

    actual_circ = np.zeros(shape, dtype=dtype)
    expected_circ = np.zeros(shape, dtype=dtype)
    _circle(actual_circ, (10, 10), 4, color, -1)
    cv2.circle(expected_circ, (10, 10), 4, color, -1)
    np.testing.assert_array_equal(actual_circ, expected_circ)

    actual_line = np.zeros(shape, dtype=dtype)
    expected_line = np.zeros(shape, dtype=dtype)
    _line(actual_line, (2, 2), (15, 15), color, 1)
    cv2.line(expected_line, (2, 2), (15, 15), color, 1)
    np.testing.assert_array_equal(actual_line, expected_line)
