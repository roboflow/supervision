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
