import numpy as np
import pytest
from PIL import Image, ImageChops

from supervision.utils.image import letterbox_image, resize_image, crop_image, \
    get_image_resolution_wh


def test_resize_image_for_opencv_image() -> None:
    # given
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    expected_result = np.zeros((768, 1024, 3), dtype=np.uint8)

    # when
    result = resize_image(
        image=image,
        resolution_wh=(1024, 1024),
        keep_aspect_ratio=True,
    )

    # then
    assert np.allclose(result, expected_result), (
        "Expected output shape to be (w, h): (1024, 768)"
    )


def test_resize_image_for_pillow_image() -> None:
    # given
    image = Image.new(mode="RGB", size=(640, 480), color=(0, 0, 0))
    expected_result = Image.new(mode="RGB", size=(1024, 768), color=(0, 0, 0))

    # when
    result = resize_image(
        image=image,
        resolution_wh=(1024, 1024),
        keep_aspect_ratio=True,
    )

    # then
    assert result.size == (1024, 768), "Expected output shape to be (w, h): (1024, 768)"
    difference = ImageChops.difference(result, expected_result)
    assert difference.getbbox() is None, (
        "Expected no difference in resized image content as the image is all zeros"
    )


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
        image=image, resolution_wh=(1024, 1024), color=(255, 255, 255)
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
        image=image, resolution_wh=(1024, 1024), color=(255, 255, 255)
    )

    # then
    assert result.size == (
        1024,
        1024,
    ), "Expected output shape to be (w, h): (1024, 1024)"
    difference = ImageChops.difference(result, expected_result)
    assert difference.getbbox() is None, (
        "Expected padding to be added top and bottom with padding added top and bottom"
    )


@pytest.mark.parametrize(
    "image, xyxy, expected_size",
    [
        # NumPy RGB
        (
            np.zeros((4, 6, 3), dtype=np.uint8),
            (2, 1, 5, 3),
            (3, 2),   # width = 5-2, height = 3-1
        ),

        # NumPy grayscale
        (
            np.zeros((5, 5), dtype=np.uint8),
            (1, 1, 4, 4),
            (3, 3),
        ),

        # Pillow RGB
        (
            Image.new("RGB", (6, 4), color=0),
            (2, 1, 5, 3),
            (3, 2),
        ),

        # Pillow grayscale
        (
            Image.new("L", (5, 5), color=0),
            (1, 1, 4, 4),
            (3, 3),
        ),
    ],
)
def test_crop_image(image, xyxy, expected_size):
    cropped = crop_image(image=image, xyxy=xyxy)
    if isinstance(image, np.ndarray):
        assert isinstance(cropped, np.ndarray)
        assert cropped.shape[1] == expected_size[0]  # width
        assert cropped.shape[0] == expected_size[1]  # height
    else:
        assert isinstance(cropped, Image.Image)
        assert cropped.size == expected_size


@pytest.mark.parametrize(
    "image, expected",
    [
        # NumPy RGB
        (np.zeros((4, 6, 3), dtype=np.uint8), (6, 4)),

        # NumPy grayscale
        (np.zeros((10, 20), dtype=np.uint8), (20, 10)),

        # Pillow RGB
        (Image.new("RGB", (6, 4), color=0), (6, 4)),

        # Pillow grayscale
        (Image.new("L", (20, 10), color=0), (20, 10)),
    ],
)
def test_get_image_resolution_wh(image, expected):
    resolution = get_image_resolution_wh(image)
    assert resolution == expected