from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from supervision import _cv2 as cv2
from supervision.draw.color import Color
from supervision.draw.utils import (
    _map_to_drawing_coordinates,
    draw_filled_polygon,
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_rounded_rectangle,
    draw_text,
)
from supervision.geometry.core import Point, Rect
from supervision.utils.image import grayscale_image
from tests.helpers import _MoveBeyondX, _WrongShapeTransform

DRAWING_LIMIT = 2**20


def test_draw_image_invalid_path_raises_oserror(tmp_path: Path) -> None:
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


def test_draw_image_valid_image(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("channels", [3, 4])
def test_draw_image_16_bit_file_draws_its_colors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channels: int
) -> None:
    """A 16-bit image file is scaled down to 8 bits, not saturated to white."""
    image_path = tmp_path / "image16.png"
    image_path.touch()
    image = np.zeros((10, 10, channels), dtype=np.uint16)
    image[...] = (51400, 50 * 256, 200 * 256, 128 * 256)[:channels]
    monkeypatch.setattr(cv2, "imread", lambda path, flags: image)
    scene = np.zeros((20, 20, 3), dtype=np.uint8)
    expected = scene.copy()
    expected[5:15, 5:15] = (101, 25, 100) if channels == 4 else (201, 50, 200)

    result = draw_image(
        scene=scene,
        image=str(image_path),
        opacity=1.0,
        rect=Rect(x=5, y=5, width=10, height=10),
    )

    np.testing.assert_array_equal(result, expected)


def test_draw_image_grayscale_file_raises_value_error(tmp_path: Path) -> None:
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

    For border_radius < 0: previously raised cv2.error: radius >= 0 in function
    'circle'; fast path now silently draws square corners instead.
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
    """Opacity<1 blends the fill colour with the original background."""
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


class TestMapToDrawingCoordinates:
    """`_map_to_drawing_coordinates` maps points or rejects undrawable ones."""

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize(
        ("mapped_x", "expected"),
        [
            pytest.param(np.nan, None, id="nan"),
            pytest.param(np.inf, None, id="inf"),
            pytest.param(-np.inf, None, id="negative-inf"),
            pytest.param(1e30, None, id="far-beyond-limit"),
            pytest.param(DRAWING_LIMIT + 0.5, None, id="just-beyond-limit"),
            pytest.param(-DRAWING_LIMIT - 0.5, None, id="just-beyond-negative-limit"),
            pytest.param(
                DRAWING_LIMIT,
                np.array([[0.0, 5.0], [DRAWING_LIMIT, 2.0]]),
                id="at-limit",
            ),
            pytest.param(
                -DRAWING_LIMIT,
                np.array([[0.0, 5.0], [-DRAWING_LIMIT, 2.0]]),
                id="at-negative-limit",
            ),
            pytest.param(
                2.5, np.array([[0.0, 5.0], [2.5, 2.0]]), id="fraction-not-rounded"
            ),
        ],
    )
    def test_returns_mapped_points_or_none_when_undrawable(
        self, mapped_x: float, expected: np.ndarray | None
    ) -> None:
        """One non-finite point, or one beyond 2**20 px before rounding, gives None."""
        points = np.array([[0, 5], [10, 2]])

        mapped = _map_to_drawing_coordinates(points, _MoveBeyondX(1, mapped_x))

        np.testing.assert_equal(mapped, expected)

    def test_rejects_transform_returning_wrong_shape(self) -> None:
        """A transform that does not return one point per input point raises."""
        points = np.array([[0, 5], [10, 2]])

        with pytest.raises(ValueError, match="coord_transform must return"):
            _map_to_drawing_coordinates(points, _WrongShapeTransform())


class TestDrawImageSceneLayouts:
    @pytest.mark.parametrize(
        "scene_shape", [pytest.param((6, 8), id="2d"), pytest.param((6, 8, 1), id="3d")]
    )
    def test_grayscale_scene_receives_image_luminance(
        self, scene_shape: tuple[int, ...]
    ) -> None:
        """Draw an opaque BGR image on a grayscale scene as its gray luminance."""
        scene = np.full(scene_shape, 7, dtype=np.uint8)
        image = np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8)
        expected = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0, 0]

        result = draw_image(scene, image, opacity=1.0, rect=Rect(1, 1, 4, 4))

        assert result.shape == scene_shape
        np.testing.assert_array_equal(
            result[1:5, 1:5], np.full_like(result[1:5, 1:5], expected)
        )
        assert result[0, 0].item() == 7

    def test_grayscale_scene_blends_with_opacity(self) -> None:
        """Blend the image luminance into a grayscale scene at the given opacity."""
        scene = np.zeros((4, 4), dtype=np.uint8)
        image = np.full((4, 4, 3), 200, dtype=np.uint8)

        result = draw_image(scene, image, opacity=0.5, rect=Rect(0, 0, 4, 4))

        np.testing.assert_array_equal(result, np.full((4, 4), 100, dtype=np.uint8))

    @pytest.mark.parametrize(
        ("scene_pixel", "expected"),
        [
            pytest.param((0, 0, 0, 0), (255, 0, 0, 128), id="transparent-scene"),
            pytest.param((0, 0, 255, 128), (170, 0, 85, 192), id="translucent-scene"),
            pytest.param((0, 0, 255, 255), (128, 0, 127, 255), id="opaque-scene"),
        ],
    )
    def test_translucent_image_on_bgra_scene_composites_straight_alpha(
        self, scene_pixel: tuple[int, ...], expected: tuple[int, ...]
    ) -> None:
        """A translucent draw weights colors by alpha rather than darkening them."""
        scene = np.full((4, 4, 4), scene_pixel, dtype=np.uint8)
        image = np.full((4, 4, 4), (255, 0, 0, 128), dtype=np.uint8)

        result = draw_image(scene, image, opacity=1.0, rect=Rect(0, 0, 4, 4))

        assert tuple(result[0, 0].tolist()) == expected

    def test_invisible_image_keeps_transparent_bgra_scene(self) -> None:
        """A fully transparent draw leaves even a transparent scene's color alone."""
        scene = np.full((4, 4, 4), (9, 8, 7, 0), dtype=np.uint8)
        image = np.full((4, 4, 4), (255, 0, 0, 0), dtype=np.uint8)

        result = draw_image(scene, image, opacity=1.0, rect=Rect(0, 0, 4, 4))

        assert tuple(result[0, 0].tolist()) == (9, 8, 7, 0)

    @pytest.mark.parametrize(
        ("opacity", "expected_alpha"),
        [pytest.param(1.0, 255, id="opaque"), pytest.param(0.0, 40, id="invisible")],
    )
    def test_bgra_scene_keeps_alpha_channel(
        self, opacity: float, expected_alpha: int
    ) -> None:
        """Draw on a BGRA scene and make drawn pixels opaque with the image."""
        scene = np.full((4, 4, 4), (1, 2, 3, 40), dtype=np.uint8)
        image = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
        expected_bgr = (10, 20, 30) if opacity else (1, 2, 3)

        result = draw_image(scene, image, opacity=opacity, rect=Rect(0, 0, 4, 4))

        assert result.shape == (4, 4, 4)
        np.testing.assert_array_equal(result[..., :3], np.full((4, 4, 3), expected_bgr))
        np.testing.assert_array_equal(result[..., 3], np.full((4, 4), expected_alpha))


_POLYGON = np.array([[5, 5], [40, 8], [35, 40], [8, 30]])
_RECT = Rect(x=10, y=10, width=20, height=20)


class TestDrawOnBgraScene:
    """Verify the shape and text helpers on BGRA scenes."""

    @pytest.mark.parametrize(
        "draw",
        [
            pytest.param(
                lambda scene: draw_line(scene, Point(2, 2), Point(40, 30), Color.RED),
                id="line",
            ),
            pytest.param(
                lambda scene: draw_rectangle(scene, _RECT, Color.RED), id="rectangle"
            ),
            pytest.param(
                lambda scene: draw_filled_rectangle(scene, _RECT, Color.RED),
                id="filled-rectangle",
            ),
            pytest.param(
                lambda scene: draw_filled_rectangle(scene, _RECT, Color.RED, 0.5),
                id="translucent-filled-rectangle",
            ),
            pytest.param(
                lambda scene: draw_rounded_rectangle(scene, _RECT, Color.RED, 4),
                id="rounded-rectangle",
            ),
            pytest.param(
                lambda scene: draw_polygon(scene, _POLYGON, Color.RED), id="polygon"
            ),
            pytest.param(
                lambda scene: draw_filled_polygon(scene, _POLYGON, Color.RED),
                id="filled-polygon",
            ),
            pytest.param(
                lambda scene: draw_filled_polygon(scene, _POLYGON, Color.RED, 0.5),
                id="translucent-filled-polygon",
            ),
            pytest.param(
                lambda scene: draw_text(
                    scene, "a", Point(25, 25), Color.RED, background_color=Color.WHITE
                ),
                id="text",
            ),
        ],
    )
    def test_opaque_bgra_scene_matches_bgr_scene(
        self, draw: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]]
    ) -> None:
        """An opaque BGRA scene gets the colors of its BGR copy and stays opaque."""
        bgr = np.full((50, 50, 3), 40, dtype=np.uint8)
        scene = np.dstack([bgr, np.full((50, 50), 255, dtype=np.uint8)])
        expected = draw(bgr.copy())

        result = draw(scene)

        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        assert (result[..., 3] == 255).all()

    @pytest.mark.parametrize(
        ("scene_alpha", "expected_pixel"),
        [
            pytest.param(0, (0, 0, 255, 128), id="transparent-scene"),
            pytest.param(128, (13, 13, 183, 192), id="translucent-scene"),
        ],
    )
    @pytest.mark.parametrize(
        "draw",
        [
            pytest.param(
                lambda scene: draw_filled_rectangle(scene, _RECT, Color.RED, 0.5),
                id="filled-rectangle",
            ),
            pytest.param(
                lambda scene: draw_filled_polygon(scene, _POLYGON, Color.RED, 0.5),
                id="filled-polygon",
            ),
        ],
    )
    def test_translucent_fill_is_composited_over_scene_alpha(
        self,
        draw: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
        scene_alpha: int,
        expected_pixel: tuple[int, int, int, int],
    ) -> None:
        """A translucent fill keeps its color on a scene that is not opaque."""
        scene = np.full((50, 50, 4), 40, dtype=np.uint8)
        scene[..., 3] = scene_alpha

        result = draw(scene)

        np.testing.assert_allclose(result[20, 20], expected_pixel, atol=1)
        np.testing.assert_array_equal(result[45, 45], (40, 40, 40, scene_alpha))

    @pytest.mark.parametrize("scene_alpha", [0, 128])
    def test_text_keeps_its_color_on_see_through_scene(self, scene_alpha: int) -> None:
        """Text on a scene that is not opaque keeps the text color at glyph edges."""
        scene = np.full((60, 120, 4), 40, dtype=np.uint8)
        scene[..., 3] = scene_alpha
        before = scene.copy()

        result = draw_text(scene, "Agy", Point(60, 30), Color.RED, text_scale=1.0)

        drawn = (result != before).any(axis=-1)
        assert drawn.any()
        assert (result[drawn][:, 3] >= scene_alpha).all()
        assert result[..., 3].max() == 255
        if scene_alpha == 0:
            np.testing.assert_allclose(
                result[drawn][:, :3],
                np.broadcast_to(Color.RED.as_bgr(), (int(drawn.sum()), 3)),
                atol=1,
            )

    def test_text_outside_bgra_scene_leaves_it_unchanged(self) -> None:
        """Text anchored far outside a BGRA scene draws nothing."""
        scene = np.full((40, 40, 4), 40, dtype=np.uint8)
        before = scene.copy()

        result = draw_text(scene, "a", Point(500, 500), Color.RED)

        np.testing.assert_array_equal(result, before)
