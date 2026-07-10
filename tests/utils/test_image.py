import warnings

import cv2
import numpy as np
import pytest
from PIL import Image, ImageChops

from supervision.utils.image import (
    ImageSink,
    _overlay_image,
    crop_image,
    get_image_resolution_wh,
    letterbox_image,
    overlay_image,
    resize_image,
    scale_image,
    tint_image,
)


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


def test_overlay_image_blends_rgba_with_float32_rounding() -> None:
    """RGBA overlay uses current float32 blend semantics."""
    # given
    image = np.full((1, 1, 3), 22, dtype=np.uint8)
    overlay = np.array([[[39, 39, 39, 60]]], dtype=np.uint8)
    expected = np.full((1, 1, 3), 26, dtype=np.uint8)

    # when
    result = overlay_image(image=image, overlay=overlay, anchor=(0, 0))

    # then
    np.testing.assert_array_equal(result, expected)


def test_overlay_image_public_wrapper_delegates_to_internal() -> None:
    """Public `overlay_image` still produces the internal `_overlay_image` result."""
    # given
    image = np.full((1, 1, 3), 22, dtype=np.uint8)
    overlay = np.array([[[39, 39, 39, 60]]], dtype=np.uint8)
    expected = _overlay_image(image=image.copy(), overlay=overlay, anchor=(0, 0))

    # when
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = overlay_image(image=image.copy(), overlay=overlay, anchor=(0, 0))

    # then
    np.testing.assert_array_equal(result, expected)


def test_overlay_image_emits_future_warning() -> None:
    """Public overlay_image must still emit FutureWarning after internal refactor."""
    # given
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = np.full((1, 1, 3), 255, dtype=np.uint8)
    # pyDeprecate tracks per-function warned_calls (default num_warns=1) so the
    # warning fires only once per process. Reset to make this test order-independent.
    overlay_image._state.warned_calls = 0

    # when
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overlay_image(image=image, overlay=overlay, anchor=(0, 0))

    # then
    assert any(issubclass(w.category, FutureWarning) for w in caught)


def test_overlay_image_crops_rgba_overlay_at_scene_boundary() -> None:
    """RGBA overlay is cropped when anchored outside scene bounds."""
    # given
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    overlay = np.array(
        [
            [[1, 11, 21, 255], [2, 12, 22, 255], [3, 13, 23, 255]],
            [[4, 14, 24, 255], [5, 15, 25, 255], [6, 16, 26, 255]],
            [[7, 17, 27, 255], [8, 18, 28, 255], [9, 19, 29, 255]],
        ],
        dtype=np.uint8,
    )
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[:2, :2] = overlay[1:, 1:, :3]

    # when
    result = overlay_image(image=image, overlay=overlay, anchor=(-1, -1))

    # then
    np.testing.assert_array_equal(result, expected)


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
def test_crop_image(image, xyxy, expected_size) -> None:
    cropped = crop_image(image=image, xyxy=xyxy)
    if isinstance(image, np.ndarray):
        assert isinstance(cropped, np.ndarray)
        assert cropped.shape[1] == expected_size[0]  # width
        assert cropped.shape[0] == expected_size[1]  # height
    else:
        assert isinstance(cropped, Image.Image)
        assert cropped.size == expected_size


def test_crop_image_clips_out_of_bounds_coordinates() -> None:
    """Out-of-bounds crops must clip consistently for NumPy and Pillow inputs."""
    image_np = np.arange(16, dtype=np.uint8).reshape(4, 4)
    image_pil = Image.fromarray(image_np)
    xyxy = (-2, -1, 3, 3)
    expected = image_np[0:3, 0:3]
    expected_pil = np.repeat(expected[:, :, None], 3, axis=2)

    np.testing.assert_array_equal(crop_image(image=image_np, xyxy=xyxy), expected)
    np.testing.assert_array_equal(
        np.asarray(crop_image(image=image_pil, xyxy=xyxy)), expected_pil
    )


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
def test_get_image_resolution_wh(image, expected) -> None:
    resolution = get_image_resolution_wh(image)
    assert resolution == expected


def test_image_sink_raises_when_cv2_write_fails(monkeypatch, tmp_path) -> None:
    """ImageSink.save_image raises and keeps count stable when OpenCV write fails."""
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)

    with ImageSink(str(tmp_path)) as sink:
        with pytest.raises(OSError, match="Failed to save image"):
            sink.save_image(np.zeros((2, 2, 3), dtype=np.uint8))

        assert sink.image_count == 0


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
