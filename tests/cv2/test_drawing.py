"""Tests for private drawing fallbacks."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supervision._cv2._drawing import (
    _circle,
    _draw_contours,
    _ellipse,
    _fill_poly,
    _line,
    _polylines,
    _rectangle,
)

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
        _rectangle(image, (2, 2), (5, 5), (1, 2, 3), thickness=thickness)


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


def test_fallback_draw_contours_fills_selected_contour() -> None:
    """Fill the selected contour when drawContours receives a negative thickness."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    contours = [np.array([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)]

    result = _draw_contours(
        image, contours, contourIdx=0, color=(7, 8, 9), thickness=-1
    )

    _assert_draws_on_canvas(result, image)
    assert tuple(image[5, 5]) == (7, 8, 9)


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
