import numpy as np
from PIL import Image, ImageChops

from supervision.utils.conversion import (
    cv2_to_pillow,
    ensure_cv2_image,
    images_to_cv2,
    pillow_to_cv2,
)


def test_ensure_cv2_image_for_processing_when_pillow_image_submitted(
    empty_cv2_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # given
    param_a_value = 3
    param_b_value = "some"

    @ensure_cv2_image
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

    @ensure_cv2_image
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


def test_pillow_to_cv2(
    empty_cv2_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # when
    result = pillow_to_cv2(image=empty_pillow_image)

    # then
    assert np.allclose(result, empty_cv2_image), (
        "Conversion to OpenCV image expected not to change the content of image"
    )


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
