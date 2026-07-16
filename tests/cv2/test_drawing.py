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
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


def _assert_draws_on_canvas(result: np.ndarray, image: np.ndarray) -> None:
    """Assert that a fallback drawing operation mutates and returns its image."""
    assert result is image
    assert np.any(image)


def test_fallback_line_draws_and_matches_opencv_extent() -> None:
    """Draw a line in place with the same raster extent as OpenCV."""
    actual = np.zeros((20, 20, 3), dtype=np.uint8)
    expected = np.zeros_like(actual)

    result = _line(actual, (2, 3), (15, 12), (10, 20, 30), thickness=2)
    cv2.line(expected, (2, 3), (15, 12), (10, 20, 30), thickness=2)

    _assert_draws_on_canvas(result, actual)
    actual_points = np.argwhere(np.any(actual != 0, axis=2))
    expected_points = np.argwhere(np.any(expected != 0, axis=2))
    np.testing.assert_allclose(
        actual_points.min(axis=0), expected_points.min(axis=0), atol=1
    )
    np.testing.assert_allclose(
        actual_points.max(axis=0), expected_points.max(axis=0), atol=1
    )


def test_fallback_rectangle_fills_the_requested_region() -> None:
    """Fill a rectangle using OpenCV's inclusive corner convention."""
    image = np.zeros((12, 14, 3), dtype=np.uint8)

    result = _rectangle(image, (2, 3), (7, 8), (1, 2, 3), thickness=-1)

    _assert_draws_on_canvas(result, image)
    np.testing.assert_array_equal(
        image[3:9, 2:8], np.full((6, 6, 3), (1, 2, 3), dtype=np.uint8)
    )


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


def test_fallback_draw_contours_fills_selected_contour() -> None:
    """Fill the selected contour when drawContours receives a negative thickness."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    contours = [np.array([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)]

    result = _draw_contours(
        image, contours, contourIdx=0, color=(7, 8, 9), thickness=-1
    )

    _assert_draws_on_canvas(result, image)
    assert tuple(image[5, 5]) == (7, 8, 9)


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
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
