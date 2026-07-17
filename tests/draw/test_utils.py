import numpy as np
import pytest

from supervision import _cv2 as cv2
from supervision.draw.color import Color
from supervision.draw.utils import (
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_rectangle,
    draw_rounded_rectangle,
    draw_text,
)
from supervision.geometry.core import Point, Rect
from supervision.utils.image import grayscale_image


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


# ---------------------------------------------------------------------------
# draw_line
# ---------------------------------------------------------------------------


def test_draw_line_modifies_scene() -> None:
    """draw_line changes at least one pixel along the drawn path."""
    scene = np.zeros((50, 50, 3), dtype=np.uint8)
    before = scene.copy()
    result = draw_line(
        scene=scene,
        start=Point(x=0, y=25),
        end=Point(x=49, y=25),
        color=Color.WHITE,
        thickness=1,
    )
    assert result is scene
    assert not np.array_equal(result, before)


def test_draw_line_preserves_shape_and_dtype() -> None:
    """draw_line returns the same shape and dtype as the input scene."""
    scene = np.zeros((30, 30, 3), dtype=np.uint8)
    result = draw_line(
        scene=scene,
        start=Point(x=0, y=0),
        end=Point(x=29, y=29),
        color=Color.WHITE,
    )
    assert result.shape == (30, 30, 3)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# draw_rectangle
# ---------------------------------------------------------------------------


def test_draw_rectangle_modifies_border_pixels() -> None:
    """draw_rectangle draws on the border of the specified rect."""
    scene = np.zeros((50, 50, 3), dtype=np.uint8)
    before = scene.copy()
    result = draw_rectangle(
        scene=scene,
        rect=Rect(x=5, y=5, width=20, height=20),
        color=Color.WHITE,
        thickness=1,
    )
    assert result is scene
    assert not np.array_equal(result, before)


def test_draw_rectangle_interior_unchanged_for_thin_border() -> None:
    """A 1-pixel border rect leaves the interior pixel untouched."""
    scene = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_rectangle(
        scene=scene,
        rect=Rect(x=10, y=10, width=20, height=20),
        color=Color.WHITE,
        thickness=1,
    )
    # centre of the rect interior — should remain black
    assert scene[20, 20].tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# draw_filled_rectangle
# ---------------------------------------------------------------------------


def test_draw_filled_rectangle_fills_interior() -> None:
    """draw_filled_rectangle sets interior pixels to the given colour."""
    scene = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_filled_rectangle(
        scene=scene,
        rect=Rect(x=10, y=10, width=20, height=20),
        color=Color.WHITE,
        opacity=1.0,
    )
    # centre should be white (BGR 255, 255, 255)
    assert scene[20, 20].tolist() == [255, 255, 255]


def test_draw_filled_rectangle_opacity_blends() -> None:
    """opacity<1 blends the fill colour with the original background."""
    scene = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_filled_rectangle(
        scene=scene,
        rect=Rect(x=10, y=10, width=20, height=20),
        color=Color.WHITE,
        opacity=0.5,
    )
    # blended pixel should be neither black nor white
    val = scene[20, 20, 0]
    assert 0 < val < 255


# ---------------------------------------------------------------------------
# draw_text
# ---------------------------------------------------------------------------


def test_draw_text_modifies_scene() -> None:
    """draw_text changes at least one pixel in the scene."""
    scene = np.zeros((100, 200, 3), dtype=np.uint8)
    before = scene.copy()
    result = draw_text(
        scene=scene,
        text="Hi",
        text_anchor=Point(x=100, y=50),
        text_color=Color.WHITE,
    )
    assert result is scene
    assert not np.array_equal(result, before)


def test_draw_text_preserves_shape() -> None:
    """draw_text does not change the scene dimensions."""
    scene = np.zeros((100, 200, 3), dtype=np.uint8)
    result = draw_text(
        scene=scene,
        text="Test",
        text_anchor=Point(x=100, y=50),
    )
    assert result.shape == (100, 200, 3)


# ---------------------------------------------------------------------------
# grayscale_image
# ---------------------------------------------------------------------------


def test_grayscale_image_preserves_shape() -> None:
    """grayscale_image returns a 3-channel image with the same HxW."""
    image = np.random.default_rng(0).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    result = grayscale_image(image=image)
    assert result.shape == (60, 80, 3)
    assert result.dtype == np.uint8


def test_grayscale_image_channels_are_equal() -> None:
    """All three output channels contain identical luminance values."""
    image = np.random.default_rng(1).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    result = grayscale_image(image=image)
    np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
    np.testing.assert_array_equal(result[:, :, 0], result[:, :, 2])


def test_grayscale_image_uniform_color_stays_gray() -> None:
    """A uniform-colour input maps to a single uniform gray value."""
    image = np.full((20, 20, 3), fill_value=[100, 150, 200], dtype=np.uint8)
    result = grayscale_image(image=image)
    assert result[:, :, 0].min() == result[:, :, 0].max()
