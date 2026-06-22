import numpy as np
import pytest

from supervision.draw.color import Color
from supervision.draw.utils import draw_rounded_rectangle
from supervision.geometry.core import Rect


@pytest.mark.parametrize(
    "border_radius",
    [
        pytest.param(0, id="radius-zero"),
        pytest.param(-5, id="radius-negative"),
    ],
)
def test_draw_rounded_rectangle_square_matches_plain_rectangle(
    border_radius: int,
) -> None:
    """Non-positive border_radius fills exactly the same pixels as a plain box.

    For border_radius < 0: previously raised cv2.error: radius >= 0 in
    function 'circle'; fast path now silently draws square corners instead.
    """
    rect = Rect(x=20, y=30, width=120, height=80)
    scene = np.full((150, 200, 3), 17, dtype=np.uint8)

    result = draw_rounded_rectangle(scene.copy(), rect, Color.RED, border_radius)

    expected = scene.copy()
    expected[30:111, 20:141] = Color.RED.as_bgr()
    assert np.array_equal(result, expected)


def test_draw_rounded_rectangle_clamped_to_zero_acts_as_square() -> None:
    """A positive border_radius clamped to 0 by a degenerate box draws square corners.

    1px-wide box: min(10, 1 // 2) = min(10, 0) = 0 → fast path fires even
    though the caller passed a positive radius.
    """
    rect = Rect(x=10, y=10, width=1, height=20)
    scene = np.full((50, 50, 3), 17, dtype=np.uint8)

    result = draw_rounded_rectangle(scene.copy(), rect, Color.RED, border_radius=10)

    expected = scene.copy()
    expected[10:31, 10:12] = Color.RED.as_bgr()
    assert np.array_equal(result, expected)


def test_draw_rounded_rectangle_positive_radius_rounds_corners() -> None:
    """A positive border radius leaves the extreme corners unpainted."""
    rect = Rect(x=20, y=30, width=120, height=80)
    scene = np.zeros((150, 200, 3), dtype=np.uint8)

    result = draw_rounded_rectangle(scene.copy(), rect, Color.RED, border_radius=15)

    red = np.array(Color.RED.as_bgr(), dtype=np.uint8)
    bg = np.zeros(3, dtype=np.uint8)

    # center row is fully filled between the inner rectangle bounds
    center_y = (30 + 110) // 2  # 70; 40px from each y edge, well past border_radius=15
    assert np.all(result[center_y, 35:126] == red)

    # all four extreme corners stay background (clipped by border_radius=15)
    assert np.array_equal(result[30, 20], bg)  # top-left
    assert np.array_equal(result[30, 140], bg)  # top-right
    assert np.array_equal(result[110, 20], bg)  # bottom-left
    assert np.array_equal(result[110, 140], bg)  # bottom-right
