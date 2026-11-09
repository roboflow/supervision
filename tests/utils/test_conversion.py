from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageChops

from supervision.draw.color import Color
from supervision.utils.conversion import (
    _palette_index_patch,
    cv2_to_pillow,
    ensure_cv2_image_for_standalone_function,
    images_to_cv2,
    pillow_to_cv2,
)


def test_ensure_cv2_image_for_processing_when_pillow_image_submitted(
    empty_cv2_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    """Pillow input is processed as an array and returned without content changes."""
    # given
    param_a_value = 3
    param_b_value = "some"

    @ensure_cv2_image_for_standalone_function
    def my_custom_processing_function(
        image: npt.NDArray[np.uint8] | Image.Image,
        param_a: int,
        param_b: str,
    ) -> npt.NDArray[np.uint8] | Image.Image:
        """Verify conversion and argument forwarding before returning the array."""
        assert isinstance(image, np.ndarray)
        assert np.allclose(image, empty_cv2_image), (
            "Expected conversion to OpenCV image to happen"
        )
        assert param_a == param_a_value, (
            f"Parameter a expected to be {param_a_value} in target function"
        )
        assert param_b == param_b_value, (
            f"Parameter b expected to be {param_b_value} in target function"
        )
        return image

    # when
    result = my_custom_processing_function(
        empty_pillow_image,
        param_a_value,
        param_b=param_b_value,
    )

    # then
    assert isinstance(result, Image.Image)
    difference = ImageChops.difference(result, empty_pillow_image)
    assert difference.getbbox() is None, (
        "Wrapper is expected to convert-back the OpenCV image "
        "into Pillow format without changes to content"
    )


def test_ensure_cv2_image_for_processing_when_cv2_image_submitted(
    empty_cv2_image: np.ndarray,
) -> None:
    # given
    param_a_value = 3
    param_b_value = "some"

    @ensure_cv2_image_for_standalone_function
    def my_custom_processing_function(
        image: np.ndarray,
        param_a: int,
        param_b: str,
    ) -> np.ndarray:
        assert np.allclose(image, empty_cv2_image), (
            "Expected conversion to OpenCV image to happen"
        )
        assert param_a == param_a_value, (
            f"Parameter a expected to be {param_a_value} in target function"
        )
        assert param_b == param_b_value, (
            f"Parameter b expected to be {param_b_value} in target function"
        )
        return image

    # when
    result = my_custom_processing_function(
        empty_cv2_image,
        param_a_value,
        param_b=param_b_value,
    )

    # then
    assert result is empty_cv2_image, "Expected to return OpenCV image without changes"


def test_cv2_to_pillow(
    empty_cv2_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # when
    result = cv2_to_pillow(image=empty_cv2_image)

    # then
    difference = ImageChops.difference(result, empty_pillow_image)
    assert difference.getbbox() is None, (
        "Conversion to PIL.Image expected not to change the content of image"
    )


def test_cv2_to_pillow_bgr_reorders_channels_to_rgb() -> None:
    """A BGR array is converted to an RGB-mode image with swapped channels."""
    # given
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R

    # when
    result = cv2_to_pillow(image)

    # then
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (30, 20, 10)


def test_cv2_to_pillow_grayscale_passes_through() -> None:
    """A 2-D grayscale array becomes an L-mode image of the same size."""
    # given
    image = np.zeros((4, 5), dtype=np.uint8)

    # when
    result = cv2_to_pillow(image)

    # then
    assert result.mode == "L"
    assert result.size == (5, 4)


def test_cv2_to_pillow_bgra_reorders_channels_to_rgba() -> None:
    """A BGRA array is converted to an RGBA-mode image with swapped channels."""
    # given
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R
    image[:, :, 3] = 255  # A

    # when
    result = cv2_to_pillow(image)

    # then
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (30, 20, 10, 255)


def test_cv2_to_pillow_invalid_shape_raises() -> None:
    """An unsupported channel count raises ValueError."""
    # given
    image = np.zeros((2, 2, 2), dtype=np.uint8)

    # when / then
    with pytest.raises(ValueError, match="Expected shape"):
        cv2_to_pillow(image)


def test_pillow_to_cv2(
    empty_cv2_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # when
    result = pillow_to_cv2(image=empty_pillow_image)

    # then
    assert np.allclose(result, empty_cv2_image), (
        "Conversion to OpenCV image expected not to change the content of image"
    )


def test_pillow_to_cv2_handles_palette_images() -> None:
    """Palette images must resolve their palette colors before BGR conversion."""
    image = Image.new("P", (1, 1))
    image.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
    image.putdata([1])

    result = pillow_to_cv2(image=image)

    np.testing.assert_array_equal(result, np.array([[[0, 0, 255]]], dtype=np.uint8))


def test_pillow_to_cv2_handles_rgba_images() -> None:
    """RGBA images drop alpha and reorder color channels like OpenCV."""
    image = Image.new("RGBA", (1, 1), color=(10, 20, 30, 40))

    result = pillow_to_cv2(image=image)

    np.testing.assert_array_equal(result, np.array([[[30, 20, 10]]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(
            Image.new("1", (2, 1)).point(lambda _: 0),
            np.array([[0, 0]], dtype=np.uint8),
            id="1-bit-black",
        ),
        pytest.param(
            Image.new("1", (2, 1), color=1),
            np.array([[255, 255]], dtype=np.uint8),
            id="1-bit-white",
        ),
        pytest.param(
            Image.new("LA", (1, 1), color=(200, 7)),
            np.array([[200]], dtype=np.uint8),
            id="grayscale-alpha-drops-alpha",
        ),
        pytest.param(
            Image.fromarray(np.array([[0, 255, 256, 30000, 65535]], dtype=np.uint16)),
            np.array([[0, 0, 1, 117, 255]], dtype=np.uint8),
            id="16-bit-high-byte-not-wrapped",
        ),
        pytest.param(
            Image.frombytes(
                "I;16B",
                (5, 1),
                np.array([[0, 255, 256, 30000, 65535]], dtype=">u2").tobytes(),
            ),
            np.array([[0, 0, 1, 117, 255]], dtype=np.uint8),
            id="16-bit-big-endian-high-byte",
        ),
        pytest.param(
            Image.fromarray(np.array([[-5, 0, 300, 70000, 2**31 - 1]], dtype=np.int32)),
            np.array([[0, 0, 1, 255, 255]], dtype=np.uint8),
            id="32-bit-int-scaled-and-clipped",
        ),
        pytest.param(
            Image.fromarray(np.array([[-1.0, 0.0, 12.7, 300.0]], dtype=np.float32)),
            np.array([[0, 0, 12, 255]], dtype=np.uint8),
            id="float-clipped",
        ),
        pytest.param(
            Image.new("RGB", (1, 1), color=(10, 20, 30)).convert("CMYK"),
            np.array([[[30, 20, 10]]], dtype=np.uint8),
            id="cmyk-converted-to-color",
        ),
        pytest.param(
            Image.new("RGB", (1, 1), color=(10, 20, 30)).convert("HSV"),
            np.array([[[30, 20, 10]]], dtype=np.uint8),
            id="hsv-converted-to-color",
        ),
        pytest.param(
            Image.new("RGB", (1, 1), color=(10, 20, 30)).convert("RGBX"),
            np.array([[[30, 20, 10]]], dtype=np.uint8),
            id="rgbx-drops-padding",
        ),
    ],
)
def test_pillow_to_cv2_reduces_every_mode_to_8_bit(
    image: Image.Image, expected: np.ndarray
) -> None:
    """Modes beyond RGB/RGBA/L/P come back as the 8-bit grayscale or BGR array OpenCV
    would produce, instead of raw mode bytes, wrapped values or an error."""
    result = pillow_to_cv2(image=image)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected)


def test_pillow_to_cv2_palette_with_alpha_resolves_colors() -> None:
    """A PA image resolves its palette like P and drops the alpha plane."""
    image = Image.new("P", (1, 1))
    image.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
    image.putdata([1])
    image = image.convert("PA")

    result = pillow_to_cv2(image=image)

    np.testing.assert_array_equal(result, np.array([[[0, 0, 255]]], dtype=np.uint8))


def test_pillow_to_cv2_ycbcr_converted_to_color() -> None:
    """YCbCr is converted through Pillow, whose round trip is lossy by a level or two,
    rather than having its luma and chroma planes drawn as BGR."""
    image = Image.new("RGB", (1, 1), color=(10, 20, 30)).convert("YCbCr")

    result = pillow_to_cv2(image=image)

    assert result.shape == (1, 1, 3)
    np.testing.assert_allclose(result, [[[30, 20, 10]]], atol=2)


def test_pillow_to_cv2_matches_opencv_reads_of_the_same_file(tmp_path: Path) -> None:
    """With OpenCV installed, each converted mode equals ``cv2.imread`` of the same
    picture written to disk (the 16-bit case keeps the high byte, as libpng's
    ``png_set_strip_16`` does)."""
    cv2 = pytest.importorskip("cv2")

    rgb = Image.new("RGB", (4, 3))
    pixels = rgb.load()
    assert pixels is not None
    for y in range(3):
        for x in range(4):
            pixels[x, y] = (10 + x * 40, 200 - y * 50, 30 + x * 10 + y * 5)
    depth = Image.fromarray(np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000)
    cases = {
        "1": (rgb.convert("1"), ".png"),
        "LA": (rgb.convert("LA"), ".png"),
        "P": (rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=8), ".png"),
        "CMYK": (rgb.convert("CMYK"), ".tif"),  # PNG has no CMYK color type
        "I;16": (depth, ".png"),
    }

    for mode, (image, suffix) in cases.items():
        path = tmp_path / f"{mode.replace(';', '_')}{suffix}"
        image.save(path)
        from_disk = cv2.imread(str(path), cv2.IMREAD_COLOR)
        from_memory = pillow_to_cv2(image=image)

        assert from_disk is not None, mode
        if from_memory.ndim == 2:
            np.testing.assert_array_equal(from_disk[..., 0], from_disk[..., 2], mode)
            from_disk = from_disk[..., 0]
        np.testing.assert_array_equal(from_memory, from_disk, mode)


def test_annotators_accept_grayscale_alpha_scenes() -> None:
    """Annotating an LA scene used to raise from ``cvtColor`` on two channels."""
    from supervision.annotators.core import BoxAnnotator
    from supervision.detection.core import Detections

    scene = Image.new("LA", (20, 20), color=(0, 255))
    detections = Detections(
        xyxy=np.array([[2, 2, 17, 17]], dtype=np.float32), class_id=np.array([0])
    )

    annotated = BoxAnnotator(thickness=1).annotate(scene=scene, detections=detections)

    assert annotated is scene
    assert annotated.mode == "LA"
    edge_pixel = annotated.getpixel((2, 10))
    interior_pixel = annotated.getpixel((10, 10))
    assert isinstance(edge_pixel, tuple)
    assert isinstance(interior_pixel, tuple)
    assert edge_pixel[0] != 0  # box edge drawn
    assert interior_pixel[0] == 0  # interior untouched


class TestAnnotatePillowScene:
    @pytest.mark.parametrize(
        ("make_scene", "background"),
        [
            pytest.param(
                lambda: Image.new("RGBA", (20, 20), color=(0, 0, 0, 0)),
                (0, 0, 0, 0),
                id="transparent-rgba",
            ),
            pytest.param(
                lambda: Image.new("LA", (20, 20), color=(0, 0)),
                (0, 0),
                id="transparent-la",
            ),
            pytest.param(
                lambda: Image.fromarray(np.full((20, 20), 40000, dtype=np.uint16)),
                40000,
                id="16-bit",
            ),
            pytest.param(
                lambda: Image.fromarray(np.full((20, 20), 40000, dtype=np.int32)),
                40000,
                id="32-bit-integer",
            ),
        ],
    )
    def test_keeps_pixels_outside_the_annotation(
        self,
        make_scene: Callable[[], Image.Image],
        background: tuple[int, ...] | int,
    ) -> None:
        """Pixels the annotator does not draw on keep their original value."""
        from supervision.annotators.core import BoxAnnotator
        from supervision.detection.core import Detections

        scene = make_scene()
        detections = Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32), class_id=np.array([0])
        )

        BoxAnnotator(thickness=1).annotate(scene=scene, detections=detections)

        assert scene.getpixel((18, 18)) == background
        assert scene.getpixel((7, 7)) == background

    def test_scales_drawn_pixels_back_to_16_bits(self) -> None:
        """A white box drawn on a 16-bit scene is stored at the 16-bit maximum."""
        from supervision.annotators.core import BoxAnnotator
        from supervision.detection.core import Detections

        scene = Image.fromarray(np.full((20, 20), 40000, dtype=np.uint16))
        detections = Detections(
            xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32), class_id=np.array([0])
        )

        BoxAnnotator(color=Color.WHITE, thickness=1).annotate(
            scene=scene, detections=detections
        )

        assert scene.getpixel((2, 7)) == 65535


def _indexed_scene(palette: list[int], index: int = 0) -> Image.Image:
    """Return a 20x20 `P` image filled with one index of the given RGB palette."""
    scene = Image.new("P", (20, 20), color=index)
    scene.putpalette(palette)
    return scene


def _draw_box(scene: Image.Image, color: Color) -> None:
    """Draw a one-pixel box from (2, 2) to (12, 12) on `scene` in place."""
    from supervision.annotators.core import BoxAnnotator
    from supervision.detection.core import Detections

    detections = Detections(
        xyxy=np.array([[2, 2, 12, 12]], dtype=np.float32), class_id=np.array([0])
    )
    BoxAnnotator(color=color, thickness=1).annotate(scene=scene, detections=detections)


class TestAnnotatePalettePillowScene:
    @pytest.mark.parametrize("mode", ["P", "PA"])
    def test_preserves_undefined_indices_outside_drawing(self, mode: str) -> None:
        """Adding a color preserves the appearance of undefined background indices."""
        scene = _indexed_scene([10, 20, 30, 200, 100, 50], index=2)
        scene = scene.convert(mode)
        background = scene.convert("RGBA").getpixel((18, 18))

        _draw_box(scene, Color.GREEN)

        assert scene.convert("RGBA").getpixel((18, 18)) == background
        assert scene.convert("RGBA").getpixel((2, 7)) == (0, 255, 0, 255)

    @pytest.mark.parametrize(
        "transparency",
        [
            pytest.param(2, id="reserved-gif-index"),
            pytest.param(b"\xff\xff\x00\x80", id="reserved-png-alpha"),
        ],
    )
    def test_skips_transparent_slots_beyond_palette(
        self, transparency: int | bytes
    ) -> None:
        """New colors use an opaque slot beyond transparency metadata."""
        scene = _indexed_scene([10, 20, 30, 200, 100, 50])
        scene.info["transparency"] = transparency

        _draw_box(scene, Color.GREEN)

        assert scene.convert("RGBA").getpixel((2, 7)) == (0, 255, 0, 255)
        assert scene.convert("RGBA").getpixel((18, 18)) == (10, 20, 30, 255)

    def test_matches_exact_nearest_color_with_first_index_ties(self) -> None:
        """A full palette matches brute-force distances, including duplicate colors."""
        rng = np.random.default_rng(123)
        palette = rng.integers(0, 256, (256, 3), dtype=np.uint8)
        palette[1] = palette[0]
        scene = Image.new("P", (80, 80))
        scene.putpalette(palette.tobytes())
        annotated = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
        annotated[0, 0] = palette[0, ::-1]
        changed = np.ones((80, 80), dtype=bool)
        rgb = annotated[:, :, ::-1].astype(np.int32)
        expected = (
            ((rgb[:, :, np.newaxis] - palette.astype(np.int32)) ** 2)
            .sum(axis=3)
            .argmin(axis=2)
        )

        patch = _palette_index_patch(scene, annotated, changed)

        np.testing.assert_array_equal(np.asarray(patch), expected)
        assert patch.getpixel((0, 0)) == 0

    def test_rgba_palette_uses_existing_opaque_entries(self) -> None:
        """RGBA palettes retain their entries and match the nearest opaque color."""
        scene = Image.new("P", (20, 20))
        palette = bytes([10, 20, 30, 255, 0, 255, 0, 0, 0, 250, 0, 255])
        scene.putpalette(palette, rawmode="RGBA")

        _draw_box(scene, Color.GREEN)

        assert scene.palette.mode == "RGBA"
        assert scene.getpalette("RGBA") == list(palette)
        assert scene.convert("RGBA").getpixel((2, 7)) == (0, 250, 0, 255)

    @pytest.mark.parametrize(
        "palette",
        [
            pytest.param([10, 20, 30, 0, 255, 0, 200, 100, 50], id="color-in-palette"),
            pytest.param([10, 20, 30, 200, 100, 50], id="color-added-to-palette"),
        ],
    )
    def test_draws_the_requested_color(self, palette: list[int]) -> None:
        """A green box stays green and the background keeps its index."""
        scene = _indexed_scene(palette)

        _draw_box(scene, Color.GREEN)

        assert scene.mode == "P"
        assert scene.convert("RGB").getpixel((2, 7)) == (0, 255, 0)
        assert scene.getpixel((18, 18)) == 0

    def test_full_palette_without_opaque_entries_raises(self) -> None:
        """A palette that can only hide a drawing raises instead of hiding it."""
        scene = _indexed_scene([0, 0, 0] * 256)
        scene.info["transparency"] = bytes(256)

        with pytest.raises(ValueError, match="no opaque entry"):
            _draw_box(scene, Color.GREEN)

    def test_more_colors_than_a_matching_batch_take_their_nearest_entries(self) -> None:
        """Colors past the first matching batch still get their nearest entry."""
        palette = [value for gray in range(256) for value in (gray, gray, gray)]
        scene = Image.new("P", (80, 80))
        scene.putpalette(palette)
        grid = np.arange(6400, dtype=np.uint32).reshape(80, 80)
        annotated = np.dstack([grid & 255, (grid >> 8) & 255, (grid * 7) & 255]).astype(
            np.uint8
        )
        changed = np.ones((80, 80), dtype=bool)
        rgb = annotated[:, :, ::-1].astype(np.int32)
        expected = (rgb.sum(axis=2) + 1) // 3

        patch = _palette_index_patch(scene, annotated, changed)

        indices = np.asarray(patch).astype(np.int32)
        gray_error = ((rgb - indices[:, :, np.newaxis]) ** 2).sum(axis=2)
        best_error = ((rgb - expected[:, :, np.newaxis]) ** 2).sum(axis=2)
        assert np.array_equal(gray_error, best_error)

    def test_full_palette_takes_the_nearest_entry(self) -> None:
        """Without room for a new color, the closest existing entry is drawn."""
        palette = [value for gray in range(255) for value in (gray, gray, gray)]
        palette += [0, 250, 0]
        scene = _indexed_scene(palette)

        _draw_box(scene, Color.GREEN)

        assert scene.getpixel((2, 7)) == 255
        assert len(scene.getpalette()) == 768

    @pytest.mark.parametrize(
        "transparency",
        [
            pytest.param(1, id="gif-index"),
            pytest.param(b"\xff\x00", id="png-alpha-per-index"),
        ],
    )
    def test_avoids_transparent_entries(self, transparency: int | bytes) -> None:
        """A drawn color matching a transparent entry does not disappear."""
        scene = _indexed_scene([10, 20, 30, 0, 255, 0])
        scene.info["transparency"] = transparency

        _draw_box(scene, Color.GREEN)

        assert scene.getpixel((2, 7)) not in (0, 1)
        assert scene.convert("RGB").getpixel((2, 7)) == (0, 255, 0)

    def test_palette_with_alpha_channel(self) -> None:
        """A `PA` scene shows the drawn color opaque and keeps the rest."""
        scene = _indexed_scene([10, 20, 30, 200, 100, 50]).convert("PA")
        background = scene.getpixel((18, 18))

        _draw_box(scene, Color.GREEN)

        assert scene.convert("RGBA").getpixel((2, 7)) == (0, 255, 0, 255)
        assert scene.getpixel((18, 18)) == background

    def test_annotation_survives_saving(self, tmp_path: Path) -> None:
        """The drawn color is still there after a PNG round trip."""
        path = tmp_path / "indexed.png"
        _indexed_scene([10, 20, 30, 200, 100, 50]).save(path)
        scene = Image.open(path)

        _draw_box(scene, Color.GREEN)
        scene.save(path)

        assert Image.open(path).convert("RGB").getpixel((2, 7)) == (0, 255, 0)


def test_images_to_cv2_when_empty_input_provided() -> None:
    # when
    result = images_to_cv2(images=[])

    # then
    assert result == [], "Expected empty output when empty input provided"


def test_images_to_cv2_when_only_cv2_images_provided(
    empty_cv2_image: np.ndarray,
) -> None:
    """Array-only input preserves every image object."""
    # given
    images: list[npt.NDArray[np.uint8] | Image.Image] = [empty_cv2_image] * 5

    # when
    result = images_to_cv2(images=images)

    # then
    assert len(result) == 5, "Expected the same number of output element as input ones"
    for result_element in result:
        assert result_element is empty_cv2_image, (
            "Expected CV images not to be touched by conversion"
        )


def test_images_to_cv2_when_only_pillow_images_provided(
    empty_pillow_image: Image.Image,
    empty_cv2_image: np.ndarray,
) -> None:
    """Pillow-only input produces an array for every supplied image."""
    # given
    images: list[npt.NDArray[np.uint8] | Image.Image] = [empty_pillow_image] * 5

    # when
    result = images_to_cv2(images=images)

    # then
    assert len(result) == 5, "Expected the same number of output element as input ones"
    for result_element in result:
        assert np.allclose(result_element, empty_cv2_image), (
            "Output images expected to be equal to empty OpenCV image"
        )


def test_images_to_cv2_when_mixed_input_provided(
    empty_pillow_image: Image.Image,
    empty_cv2_image: np.ndarray,
) -> None:
    """Mixed input converts Pillow images and preserves existing arrays in order."""
    # given
    images: list[npt.NDArray[np.uint8] | Image.Image] = [
        empty_pillow_image,
        empty_cv2_image,
    ]

    # when
    result = images_to_cv2(images=images)

    # then
    assert len(result) == 2, "Expected the same number of output element as input ones"
    assert np.allclose(result[0], empty_cv2_image), (
        "PIL image should be converted to OpenCV one, equal to example empty image"
    )
    assert result[1] is empty_cv2_image, (
        "Expected CV images not to be touched by conversion"
    )
