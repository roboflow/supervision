from typing import List

import numpy as np
import pytest
from PIL import Image, ImageChops

from supervision import Color, Point
from supervision.utils.image import (
    adjust_image_to_cv2_processing,
    create_tiles,
    letterbox_image,
    resize_image_keeping_aspect_ratio,
)


def test_adjust_image_to_cv2_processing_when_pillow_image_submitted(
    empty_opencv_image: np.ndarray, empty_pillow_image: Image.Image
) -> None:
    # given
    param_a_value = 3
    param_b_value = "some"

    @adjust_image_to_cv2_processing
    def my_custom_processing_function(
        image: np.ndarray,
        param_a: int,
        param_b: str,
    ) -> np.ndarray:
        assert np.allclose(
            image, empty_opencv_image
        ), "Expected conversion to OpenCV image to happen"
        assert (
            param_a == param_a_value
        ), f"Parameter a expected to be {param_a_value} in target function"
        assert (
            param_b == param_b_value
        ), f"Parameter b expected to be {param_b_value} in target function"
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


def test_adjust_image_to_cv2_processing_when_opencv_image_submitted(
    empty_opencv_image: np.ndarray,
) -> None:
    # given
    param_a_value = 3
    param_b_value = "some"

    @adjust_image_to_cv2_processing
    def my_custom_processing_function(
        image: np.ndarray,
        param_a: int,
        param_b: str,
    ) -> np.ndarray:
        assert np.allclose(
            image, empty_opencv_image
        ), "Expected conversion to OpenCV image to happen"
        assert (
            param_a == param_a_value
        ), f"Parameter a expected to be {param_a_value} in target function"
        assert (
            param_b == param_b_value
        ), f"Parameter b expected to be {param_b_value} in target function"
        return image

    # when
    result = my_custom_processing_function(
        empty_opencv_image,
        param_a_value,
        param_b=param_b_value,
    )

    # then
    assert (
        result is empty_opencv_image
    ), "Expected to return OpenCV image without changes"


def test_resize_image_keeping_aspect_ratio_for_opencv_image() -> None:
    # given
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    expected_result = np.zeros((768, 1024, 3), dtype=np.uint8)

    # when
    result = resize_image_keeping_aspect_ratio(
        image=image,
        desired_size=(1024, 1024),
    )

    # then
    assert np.allclose(
        result, expected_result
    ), "Expected output shape to be (w, h): (1024, 768)"


def test_resize_image_keeping_aspect_ratio_for_pillow_image() -> None:
    # given
    image = Image.new(mode="RGB", size=(640, 480), color=(0, 0, 0))
    expected_result = Image.new(mode="RGB", size=(1024, 768), color=(0, 0, 0))

    # when
    result = resize_image_keeping_aspect_ratio(
        image=image,
        desired_size=(1024, 1024),
    )

    # then
    assert result.size == (1024, 768), "Expected output shape to be (w, h): (1024, 768)"
    difference = ImageChops.difference(result, expected_result)
    assert (
        difference.getbbox() is None
    ), "Expected no difference in resized image content as the image is all zeros"


def test_letterbox_image_for_opencv_image() -> None:
    # given
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    expected_result = np.concatenate(
        [
            np.ones((128, 1024, 3), dtype=np.uint8) * 255,
            np.zeros((768, 1024, 3), dtype=np.uint8),
            np.ones((128, 1024, 3), dtype=np.uint8) * 255,
        ],
        axis=0,
    )

    # when
    result = letterbox_image(
        image=image, desired_size=(1024, 1024), color=(255, 255, 255)
    )

    # then
    assert np.allclose(result, expected_result), (
        "Expected output shape to be (w, h): "
        "(1024, 1024) with padding added top and bottom"
    )


def test_letterbox_image_for_pillow_image() -> None:
    # given
    image = Image.new(mode="RGB", size=(640, 480), color=(0, 0, 0))
    expected_result = Image.fromarray(
        np.concatenate(
            [
                np.ones((128, 1024, 3), dtype=np.uint8) * 255,
                np.zeros((768, 1024, 3), dtype=np.uint8),
                np.ones((128, 1024, 3), dtype=np.uint8) * 255,
            ],
            axis=0,
        )
    )

    # when
    result = letterbox_image(
        image=image, desired_size=(1024, 1024), color=(255, 255, 255)
    )

    # then
    assert result.size == (
        1024,
        1024,
    ), "Expected output shape to be (w, h): (1024, 1024)"
    difference = ImageChops.difference(result, expected_result)
    assert (
        difference.getbbox() is None
    ), "Expected padding to be added top and bottom with padding added top and bottom"


def test_create_tiles_with_one_image(
    one_image: np.ndarray, single_image_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=[one_image], single_tile_size=(240, 240))

    # # then
    assert np.allclose(result, single_image_tile, atol=5.0)


def test_create_tiles_with_one_image_and_enforced_grid(
    one_image: np.ndarray, single_image_tile_enforced_grid: np.ndarray
) -> None:
    # when
    result = create_tiles(
        images=[one_image],
        grid_size=(None, 3),
        single_tile_size=(240, 240),
    )

    # then
    assert np.allclose(result, single_image_tile_enforced_grid, atol=5.0)


def test_create_tiles_with_two_images(
    two_images: List[np.ndarray], two_images_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=two_images, single_tile_size=(240, 240))

    # then
    assert np.allclose(result, two_images_tile, atol=5.0)


def test_create_tiles_with_three_images(
    three_images: List[np.ndarray], three_images_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=three_images, single_tile_size=(240, 240))

    # then
    assert np.allclose(result, three_images_tile, atol=5.0)


def test_create_tiles_with_four_images(
    four_images: List[np.ndarray],
    four_images_tile: np.ndarray,
) -> None:
    # when
    result = create_tiles(images=four_images, single_tile_size=(240, 240))

    # then
    assert np.allclose(result, four_images_tile, atol=5.0)


def test_create_tiles_with_all_images(
    all_images: List[np.ndarray],
    all_images_tile: np.ndarray,
) -> None:
    # when
    result = create_tiles(images=all_images, single_tile_size=(240, 240))

    # then
    assert np.allclose(result, all_images_tile, atol=5.0)


def test_create_tiles_with_all_images_and_custom_grid(
    all_images: List[np.ndarray], all_images_tile_and_custom_grid: np.ndarray
) -> None:
    # when
    result = create_tiles(
        images=all_images,
        grid_size=(3, 3),
        single_tile_size=(240, 240),
    )

    # then
    assert np.allclose(result, all_images_tile_and_custom_grid, atol=5.0)


def test_create_tiles_with_all_images_and_custom_colors(
    all_images: List[np.ndarray], all_images_tile_and_custom_colors: np.ndarray
) -> None:
    # when
    result = create_tiles(
        images=all_images,
        tile_margin_color=(127, 127, 127),
        tile_padding_color=(224, 224, 224),
        single_tile_size=(240, 240),
    )

    # then
    assert np.allclose(result, all_images_tile_and_custom_colors, atol=5.0)


def test_create_tiles_with_all_images_and_titles(
    all_images: List[np.ndarray],
    all_images_tile_and_custom_colors_and_titles: np.ndarray,
) -> None:
    # when
    result = create_tiles(
        images=all_images,
        titles=["Image 1", None, "Image 3", "Image 4"],
        single_tile_size=(240, 240),
    )

    # then
    assert np.allclose(result, all_images_tile_and_custom_colors_and_titles, atol=5.0)


def test_create_tiles_with_all_images_and_titles_with_custom_configs(
    all_images: List[np.ndarray],
    all_images_tile_and_titles_with_custom_configs: np.ndarray,
) -> None:
    # when
    result = create_tiles(
        images=all_images,
        titles=["Image 1", None, "Image 3", "Image 4"],
        single_tile_size=(240, 240),
        titles_anchors=[
            Point(x=200, y=300),
            Point(x=300, y=400),
            None,
            Point(x=300, y=400),
        ],
        titles_color=Color.RED,
        titles_scale=1.5,
        titles_thickness=3,
        titles_padding=20,
        titles_background_color=Color.BLACK,
        default_title_placement="bottom",
    )

    # then
    assert np.allclose(result, all_images_tile_and_titles_with_custom_configs, atol=5.0)


def test_create_tiles_with_all_images_and_custom_grid_to_small_to_fit_images(
    all_images: List[np.ndarray],
) -> None:
    with pytest.raises(ValueError):
        _ = create_tiles(images=all_images, grid_size=(2, 2))
