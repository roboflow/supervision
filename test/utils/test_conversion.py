import numpy as np
from PIL import Image, ImageChops

from supervision.utils.conversion import cv2_to_pillow, images_to_cv2, pillow_to_cv2


def test_cv2_to_pillow(
    empty_opencv_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # when
    result = cv2_to_pillow(image=empty_opencv_image)

    # then
    difference = ImageChops.difference(result, empty_pillow_image)
    assert (
        difference.getbbox() is None
    ), "Conversion to PIL.Image expected not to change the content of image"


def test_pillow_to_cv2(
    empty_opencv_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # when
    result = pillow_to_cv2(image=empty_pillow_image)

    # then
    assert np.allclose(
        result, empty_opencv_image
    ), "Conversion to OpenCV image expected not to change the content of image"


def test_images_to_cv2_when_empty_input_provided() -> None:
    # when
    result = images_to_cv2(images=[])

    # then
    assert result == [], "Expected empty output when empty input provided"


def test_images_to_cv2_when_only_cv2_images_provided(
    empty_opencv_image: np.ndarray,
) -> None:
    # given
    images = [empty_opencv_image] * 5

    # when
    result = images_to_cv2(images=images)

    # then
    assert len(result) == 5, "Expected the same number of output element as input ones"
    for result_element in result:
        assert (
            result_element is empty_opencv_image
        ), "Expected CV images not to be touched by conversion"


def test_images_to_cv2_when_only_pillow_images_provided(
    empty_pillow_image: Image.Image,
    empty_opencv_image: np.ndarray,
) -> None:
    # given
    images = [empty_pillow_image] * 5

    # when
    result = images_to_cv2(images=images)

    # then
    assert len(result) == 5, "Expected the same number of output element as input ones"
    for result_element in result:
        assert np.allclose(
            result_element, empty_opencv_image
        ), "Output images expected to be equal to empty OpenCV image"


def test_images_to_cv2_when_mixed_input_provided(
    empty_pillow_image: Image.Image,
    empty_opencv_image: np.ndarray,
) -> None:
    # given
    images = [empty_pillow_image, empty_opencv_image]

    # when
    result = images_to_cv2(images=images)

    # then
    assert len(result) == 2, "Expected the same number of output element as input ones"
    assert np.allclose(
        result[0], empty_opencv_image
    ), "PIL image should be converted to OpenCV one, equal to example empty image"
    assert (
        result[1] is empty_opencv_image
    ), "Expected CV images not to be touched by conversion"
