from __future__ import annotations

from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import requests
from PIL import Image, ImageChops

from supervision.utils.image import (
    crop_image,
    get_image_resolution_wh,
    letterbox_image,
    load_image_from_url,
    resize_image,
    scale_image,
    tint_image,
)


class TestLoadImageFromUrl:
    def test_returns_decoded_image(self, tmp_path) -> None:
        """Valid image URL returns an OpenCV image."""
        # given
        image = np.full((10, 20, 3), 127, dtype=np.uint8)
        encoded = cv2.imencode(".jpg", image)[1]
        response = Mock()
        response.content = encoded.tobytes()
        response.raise_for_status.return_value = None

        # when
        with patch(
            "supervision.utils.image.requests.get", return_value=response
        ) as get:
            result = load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )

        # then
        get.assert_called_once_with(
            "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
            timeout=30.0,
        )
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        response.close.assert_called_once()

    def test_uses_cached_image_on_repeated_calls(self, tmp_path) -> None:
        """Repeated image URL loads use the local cache."""
        # given
        image = np.full((10, 20, 3), 127, dtype=np.uint8)
        encoded = cv2.imencode(".jpg", image)[1]
        response = Mock()
        response.content = encoded.tobytes()
        response.raise_for_status.return_value = None

        # when
        with patch(
            "supervision.utils.image.requests.get", return_value=response
        ) as get:
            first_result = load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )
            second_result = load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )

        # then
        get.assert_called_once()
        assert first_result.shape == image.shape
        assert second_result.shape == image.shape

    def test_force_reload_refreshes_cached_image(self, tmp_path) -> None:
        """Force reload bypasses the cached image and refreshes it."""
        # given
        first_image = np.zeros((10, 20, 3), dtype=np.uint8)
        second_image = np.full((12, 22, 3), 127, dtype=np.uint8)
        first_response = Mock()
        first_response.content = cv2.imencode(".jpg", first_image)[1].tobytes()
        first_response.raise_for_status.return_value = None
        second_response = Mock()
        second_response.content = cv2.imencode(".jpg", second_image)[1].tobytes()
        second_response.raise_for_status.return_value = None

        # when
        with patch(
            "supervision.utils.image.requests.get",
            side_effect=[first_response, second_response],
        ) as get:
            cached_result = load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )
            refreshed_result = load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
                force_reload=True,
            )

        # then
        assert get.call_count == 2
        assert cached_result.shape == first_image.shape
        assert refreshed_result.shape == second_image.shape

    def test_redownloads_when_cached_image_is_invalid(self, tmp_path) -> None:
        """Invalid cached image bytes are discarded and downloaded again."""
        # given
        image = np.full((10, 20, 3), 127, dtype=np.uint8)
        first_response = Mock()
        first_response.content = cv2.imencode(".jpg", image)[1].tobytes()
        first_response.raise_for_status.return_value = None
        second_response = Mock()
        second_response.content = cv2.imencode(".jpg", image)[1].tobytes()
        second_response.raise_for_status.return_value = None

        # when
        with patch(
            "supervision.utils.image.requests.get",
            side_effect=[first_response, second_response],
        ) as get:
            load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )
            for cache_file in tmp_path.iterdir():
                cache_file.write_bytes(b"not an image")
            result = load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )

        # then
        assert get.call_count == 2
        assert result.shape == image.shape

    def test_raises_when_bytes_are_not_image(self, tmp_path) -> None:
        """Invalid image bytes raise ValueError."""
        # given
        response = Mock()
        response.content = b"not an image"
        response.raise_for_status.return_value = None

        # when / then
        with (
            patch("supervision.utils.image.requests.get", return_value=response),
            pytest.raises(ValueError, match="could not be decoded into image"),
        ):
            load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )
        response.close.assert_called_once()

    def test_raises_for_request_error(self, tmp_path) -> None:
        """Request failures are propagated."""
        # given
        request_error = requests.RequestException("boom")

        # when / then
        with (
            patch("supervision.utils.image.requests.get", side_effect=request_error),
            pytest.raises(requests.RequestException, match="boom"),
        ):
            load_image_from_url(
                "https://media.roboflow.com/notebooks/examples/dog-9.jpeg",
                cache_dir=tmp_path,
            )

    def test_rejects_non_http_url(self) -> None:
        """Non-HTTP URLs are rejected before making a request."""
        # given
        with patch("supervision.utils.image.requests.get") as get:
            # when / then
            with pytest.raises(ValueError, match="HTTP"):
                load_image_from_url("file:///tmp/image.jpg")

        get.assert_not_called()


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


def test_letterbox_image_for_grayscale_opencv_image() -> None:
    image = np.zeros((4, 6), dtype=np.uint8)
    expected_result = np.concatenate(
        [
            np.ones((2, 10), dtype=np.uint8) * 255,
            np.zeros((6, 10), dtype=np.uint8),
            np.ones((2, 10), dtype=np.uint8) * 255,
        ],
        axis=0,
    )

    result = letterbox_image(image=image, resolution_wh=(10, 10), color=(255, 255, 255))

    assert result.shape == (10, 10)
    assert np.array_equal(result, expected_result)


def test_letterbox_image_for_rgba_opencv_image() -> None:
    """RGBA input: padded alpha=0, interior alpha preserved, input array not mutated."""
    # given
    image = np.zeros((4, 6, 4), dtype=np.uint8)
    image[:, :, 3] = 128
    image_before = image.copy()

    # when
    result = letterbox_image(image=image, resolution_wh=(10, 10), color=(0, 0, 0))

    # then
    assert result.shape == (10, 10, 4)
    assert np.all(result[:2, :, 3] == 0), "padded top rows must have alpha=0"
    assert np.all(result[8:, :, 3] == 0), "padded bottom rows must have alpha=0"
    assert np.all(result[2:8, :, 3] == 128), "interior rows must preserve alpha"
    assert np.array_equal(image, image_before), "input must not be mutated"


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
    ("image", "xyxy", "expected_size"),
    [
        # NumPy RGB
        (
            np.zeros((4, 6, 3), dtype=np.uint8),
            (2, 1, 5, 3),
            (3, 2),  # width = 5-2, height = 3-1
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
    ("image", "expected"),
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


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        pytest.param(scale_image, {"scale_factor": 1.0}, id="scale_image"),
        pytest.param(resize_image, {"resolution_wh": (10, 10)}, id="resize_image"),
        pytest.param(
            letterbox_image, {"resolution_wh": (10, 10)}, id="letterbox_image"
        ),
        pytest.param(tint_image, {}, id="tint_image"),
    ],
)
def test_image_utils_wrong_type_raises(func, kwargs):
    """Wrong image type raises TypeError via decorator."""
    with pytest.raises(TypeError, match="Unsupported image type"):
        func(image="not_an_image", **kwargs)
