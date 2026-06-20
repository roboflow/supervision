import numpy as np
import pytest

from supervision.draw.color import Color
from supervision.draw.utils import draw_rounded_rectangle
from supervision.geometry.core import Rect


@pytest.mark.parametrize("border_radius", [0, -5])
def test_draw_rounded_rectangle_square_matches_plain_rectangle(
    border_radius: int,
) -> None:
    """A non-positive border radius fills exactly the same pixels as a plain box."""
    rect = Rect(x=20, y=30, width=120, height=80)
    scene = np.full((150, 200, 3), 17, dtype=np.uint8)

    result = draw_rounded_rectangle(scene.copy(), rect, Color.RED, border_radius)

    expected = scene.copy()
    expected[30:111, 20:141] = Color.RED.as_bgr()
    assert np.array_equal(result, expected)


def test_draw_rounded_rectangle_positive_radius_rounds_corners() -> None:
    """A positive border radius leaves the extreme corners unpainted."""
    rect = Rect(x=20, y=30, width=120, height=80)
    scene = np.zeros((150, 200, 3), dtype=np.uint8)

    result = draw_rounded_rectangle(scene.copy(), rect, Color.RED, border_radius=15)

    # the very top-left corner pixel stays background, the body is filled
    assert np.array_equal(result[30, 20], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(result[70, 80], np.array(Color.RED.as_bgr(), dtype=np.uint8))
