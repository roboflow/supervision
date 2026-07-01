import cv2
import numpy as np
import pytest

from supervision.draw.color import Color
from supervision.draw.utils import draw_image, draw_rounded_rectangle
from supervision.geometry.core import Rect


def test_draw_image_invalid_path_raises_oserror(tmp_path) -> None:
    """Existing but undecodable image files raise OSError."""
    invalid_image_path = tmp_path / "invalid_image.dat"
    invalid_image_path.write_bytes(b"not an image")
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    with pytest.raises(OSError, match="Could not decode image path"):
        draw_image(
            scene=scene,
            image=str(invalid_image_path),
            opacity=1.0,
            rect=rect,
        )


def test_draw_image_valid_image(tmp_path) -> None:
    """Valid image files are decoded and drawn onto the scene."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), image)

    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    result = draw_image(
        scene=scene,
        image=str(image_path),
        opacity=1.0,
        rect=rect,
    )

    assert isinstance(result, np.ndarray)


def test_draw_image_grayscale_file_raises_value_error(tmp_path) -> None:
    """Grayscale image files raise ValueError before channel access."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image_path = tmp_path / "grayscale.png"
    cv2.imwrite(str(image_path), image)

    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    with pytest.raises(ValueError, match="3 or 4 channels"):
        draw_image(
            scene=scene,
            image=str(image_path),
            opacity=1.0,
            rect=rect,
        )


def test_draw_image_grayscale_array_raises_value_error() -> None:
    """Grayscale image arrays raise ValueError before channel access."""
    image = np.zeros((100, 100), dtype=np.uint8)
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = Rect(x=0, y=0, width=100, height=100)

    with pytest.raises(ValueError, match="3 or 4 channels"):
        draw_image(
            scene=scene,
            image=image,
            opacity=1.0,
            rect=rect,
        )


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
