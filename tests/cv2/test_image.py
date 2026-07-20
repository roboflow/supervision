"""Tests for private image-operation and I/O fallbacks."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from supervision._cv2._image import (
    _add_weighted,
    _convert_scale_abs,
    _copy_make_border,
    _flip,
    _imdecode,
    _imencode,
    _imread,
    _imwrite,
    _mean,
    _resize,
)
from supervision._cv2.constants import (
    _BORDER_CONSTANT,
    _IMREAD_COLOR,
    _IMREAD_UNCHANGED,
)

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    ("flip_code", "expected"),
    [
        pytest.param(0, np.array([[3, 4], [1, 2]], dtype=np.uint8), id="vertical"),
        pytest.param(1, np.array([[2, 1], [4, 3]], dtype=np.uint8), id="horizontal"),
        pytest.param(-1, np.array([[4, 3], [2, 1]], dtype=np.uint8), id="both"),
    ],
)
def test_fallback_flip_matches_opencv(flip_code: int, expected: np.ndarray) -> None:
    """Match OpenCV flip direction and return a contiguous array."""
    source = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    np.testing.assert_array_equal(_flip(source, flip_code), expected)
    np.testing.assert_array_equal(_flip(source, flip_code), cv2.flip(source, flip_code))


@pytest.mark.parametrize(
    ("source", "value"),
    [
        pytest.param(
            np.array([[0, 100], [200, 255]], dtype=np.uint8),
            7,
            id="grayscale-scalar",
        ),
        pytest.param(
            np.array([[0, 100], [200, 255]], dtype=np.uint8),
            (5, 9, 20),
            id="grayscale-sequence-uses-first-element",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            (7, 8),
            id="multichannel-sequence-shorter-than-channels-pads-with-zero",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            (7, 8, 9, 10),
            id="multichannel-sequence-longer-than-channels-truncates",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            100,
            id="multichannel-scalar-fills-only-first-channel",
        ),
    ],
)
def test_fallback_copy_make_border_matches_opencv(
    source: np.ndarray, value: int | tuple[int, ...]
) -> None:
    """Match OpenCV constant-border padding for scalar and Sequence values."""
    np.testing.assert_array_equal(
        _copy_make_border(source, 1, 1, 2, 2, _BORDER_CONSTANT, value),
        cv2.copyMakeBorder(source, 1, 1, 2, 2, cv2.BORDER_CONSTANT, value=value),
    )


def test_fallback_add_weighted_matches_opencv() -> None:
    """Match OpenCV weighted image blending."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    np.testing.assert_array_equal(
        _add_weighted(source, 0.5, other, 0.5, 10),
        cv2.addWeighted(source, 0.5, other, 0.5, 10),
    )


def test_fallback_add_weighted_supports_destination() -> None:
    """Write weighted image blending results into the provided destination."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)
    destination = np.empty_like(source)

    actual = _add_weighted(source, 0.5, other, 0.5, 10, dst=destination)

    assert actual is destination
    np.testing.assert_array_equal(actual, cv2.addWeighted(source, 0.5, other, 0.5, 10))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(None, id="none"),
        pytest.param(-1, id="opencv-sentinel"),
    ],
)
def test_fallback_add_weighted_accepts_default_dtype(dtype: int | None) -> None:
    """Treat both None and OpenCV's -1 sentinel as the default output depth."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    np.testing.assert_array_equal(
        _add_weighted(source, 0.5, other, 0.5, 10, dtype=dtype),
        cv2.addWeighted(source, 0.5, other, 0.5, 10),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(0, id="cv-8u"),
        pytest.param(5, id="cv-32f"),
    ],
)
def test_fallback_add_weighted_rejects_non_default_dtype(dtype: int) -> None:
    """Fail loud when a caller requests an unsupported output depth."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    with pytest.raises(ValueError, match="output depth"):
        _add_weighted(source, 0.5, other, 0.5, 10, dtype=dtype)


def test_fallback_convert_scale_abs_matches_opencv() -> None:
    """Match OpenCV absolute scale-and-convert semantics."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _convert_scale_abs(source, 1.5, -20),
        cv2.convertScaleAbs(source, alpha=1.5, beta=-20),
    )


def test_fallback_mean_matches_opencv() -> None:
    """Match OpenCV masked mean semantics."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    assert _mean(source, mask) == cv2.mean(source, mask)


@pytest.mark.parametrize(
    ("interpolation", "atol"),
    [
        pytest.param(cv2.INTER_NEAREST, 0, id="nearest"),
        pytest.param(cv2.INTER_LINEAR, 1, id="linear"),
    ],
)
def test_fallback_resize_matches_opencv(interpolation: int, atol: int) -> None:
    """Match OpenCV resize shape and pixel values within the interpolation budget."""
    source = np.arange(20, dtype=np.uint8).reshape(4, 5)

    actual = _resize(source, (9, 7), interpolation=interpolation)
    expected = cv2.resize(source, (9, 7), interpolation=interpolation)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def test_fallback_linear_resize_preserves_random_uint8_contract() -> None:
    """Preserve dtype, contiguity, and the one-LSB visual interpolation budget."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)

    actual = _resize(source, (31, 29), interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, (31, 29), interpolation=cv2.INTER_LINEAR)

    assert actual.dtype == source.dtype
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)


def test_fallback_linear_resize_matches_opencv_when_downsampling_uint8() -> None:
    """Preserve OpenCV's half-pixel interpolation for uint8 downsampling."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)

    actual = _resize(source, (1, 18), interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, (1, 18), interpolation=cv2.INTER_LINEAR)

    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)


def test_fallback_linear_resize_preserves_rgba_channels() -> None:
    """Avoid Pillow alpha premultiplication when resizing uint8 RGBA arrays."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, (9, 7, 4), dtype=np.uint8)

    actual = _resize(source, (13, 4), interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, (13, 4), interpolation=cv2.INTER_LINEAR)

    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)


def test_fallback_float_resize_preserves_mask_threshold_decision() -> None:
    """Keep float-mask interpolation on the established numeric path."""
    rng = np.random.default_rng(20260717)
    source = rng.random((3836, 17, 23), dtype=np.float32)[-1]
    actual = _resize(source, (31, 29), interpolation=cv2.INTER_LINEAR)

    assert actual[8, 27] == np.float32(0.49999991059303284)
    assert not bool((actual > 0.5)[8, 27])


def test_fallback_image_io_preserves_bgr(tmp_path: Path) -> None:
    """Preserve BGR channel order when writing and reading an image."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    image_path = tmp_path / "image.png"

    assert _imwrite(str(image_path), image)
    actual = _imread(str(image_path), _IMREAD_COLOR)
    assert actual is not None
    np.testing.assert_array_equal(actual, image)


def test_fallback_image_io_returns_none_for_missing_file(tmp_path: Path) -> None:
    """Return None when reading a missing image file."""
    assert _imread(str(tmp_path / "missing.png"), _IMREAD_COLOR) is None


def test_fallback_image_io_preserves_alpha(tmp_path: Path) -> None:
    """Preserve alpha channels when reading unchanged images."""
    alpha = np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], dtype=np.uint8)
    alpha_path = tmp_path / "alpha.png"

    assert _imwrite(str(alpha_path), alpha)
    np.testing.assert_array_equal(
        _imread(str(alpha_path), _IMREAD_UNCHANGED),
        cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED),
    )


def test_fallback_image_io_preserves_sixteen_bit_unchanged(tmp_path: Path) -> None:
    """Preserve sixteen-bit pixel values when reading unchanged images."""
    sixteen_bit = np.array([[0, 12345], [54321, 65535]], dtype=np.uint16)
    sixteen_bit_path = tmp_path / "sixteen-bit.png"

    assert _imwrite(str(sixteen_bit_path), sixteen_bit)
    np.testing.assert_array_equal(
        _imread(str(sixteen_bit_path), _IMREAD_UNCHANGED),
        cv2.imread(str(sixteen_bit_path), cv2.IMREAD_UNCHANGED),
    )


def test_fallback_in_memory_codec_preserves_bgr() -> None:
    """Preserve BGR channel order across an encode and decode round trip."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    success, encoded = _imencode(".png", image)

    assert success
    assert encoded is not None
    decoded = _imdecode(encoded, _IMREAD_COLOR)
    assert decoded is not None
    np.testing.assert_array_equal(decoded, image)


def test_fallback_imdecode_matches_opencv_for_jpeg() -> None:
    """Decode OpenCV-encoded JPEG bytes identically to cv2.imdecode."""
    image = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1]

    np.testing.assert_array_equal(
        _imdecode(encoded, _IMREAD_COLOR),
        cv2.imdecode(encoded, cv2.IMREAD_COLOR),
    )


def test_fallback_imdecode_returns_none_for_invalid_bytes() -> None:
    """Return None when decoding bytes that are not an image."""
    invalid = np.frombuffer(b"not an image", dtype=np.uint8)

    assert _imdecode(invalid, _IMREAD_COLOR) is None


def test_fallback_imencode_reports_failure_for_unknown_extension() -> None:
    """Report failure when encoding to an extension Pillow cannot handle."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    success, encoded = _imencode(".unknown", image)

    assert not success
    assert encoded is None


def test_fallback_image_io_matches_opencv_color_conversion_for_sixteen_bit(
    tmp_path: Path,
) -> None:
    """Match OpenCV color conversion when reading a sixteen-bit image."""
    sixteen_bit = np.array([[0, 12345], [54321, 65535]], dtype=np.uint16)
    sixteen_bit_path = tmp_path / "sixteen-bit.png"

    assert _imwrite(str(sixteen_bit_path), sixteen_bit)
    np.testing.assert_array_equal(
        _imread(str(sixteen_bit_path), _IMREAD_COLOR),
        cv2.imread(str(sixteen_bit_path), cv2.IMREAD_COLOR),
    )
