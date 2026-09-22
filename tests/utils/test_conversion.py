import numpy as np
import pytest
from PIL import Image, ImageChops

from supervision.utils.conversion import (
    cv2_to_pillow,
    ensure_cv2_image_for_standalone_function,
    images_to_cv2,
    pillow_to_cv2,
)


def test_ensure_cv2_image_for_processing_when_pillow_image_submitted(
    empty_cv2_image: np.ndarray, empty_pillow_image: Image.Image
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
        empty_pillow_image,
        param_a_value,
        param_b=param_b_value,
    )

    # then
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


def test_pillow_to_cv2_matches_opencv_reads_of_the_same_file(tmp_path) -> None:
    """With OpenCV installed, each converted mode equals ``cv2.imread`` of the same
    picture written to disk (the 16-bit case keeps the high byte, as libpng's
    ``png_set_strip_16`` does)."""
    cv2 = pytest.importorskip("cv2")

    rgb = Image.new("RGB", (4, 3))
    pixels = rgb.load()
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
    assert annotated.getpixel((2, 10))[0] != 0  # box edge drawn
    assert annotated.getpixel((10, 10))[0] == 0  # interior untouched


def test_images_to_cv2_when_empty_input_provided() -> None:
    # when
    result = images_to_cv2(images=[])

    # then
    assert result == [], "Expected empty output when empty input provided"


def test_images_to_cv2_when_only_cv2_images_provided(
    empty_cv2_image: np.ndarray,
) -> None:
    # given
    images = [empty_cv2_image] * 5

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
    # given
    images = [empty_pillow_image] * 5

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
    # given
    images = [empty_pillow_image, empty_cv2_image]

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
