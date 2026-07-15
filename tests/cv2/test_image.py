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


def test_fallback_array_arithmetic_matches_opencv() -> None:
    """Match OpenCV border, blend, scale, and mean semantics."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _copy_make_border(source, 1, 1, 2, 2, _BORDER_CONSTANT, 7),
        cv2.copyMakeBorder(source, 1, 1, 2, 2, cv2.BORDER_CONSTANT, value=7),
    )
    np.testing.assert_array_equal(
        _add_weighted(source, 0.5, other, 0.5, 10),
        cv2.addWeighted(source, 0.5, other, 0.5, 10),
    )
    destination = np.empty_like(source)
    actual = _add_weighted(source, 0.5, other, 0.5, 10, dst=destination)
    assert actual is destination
    np.testing.assert_array_equal(actual, cv2.addWeighted(source, 0.5, other, 0.5, 10))
    np.testing.assert_array_equal(
        _convert_scale_abs(source, 1.5, -20),
        cv2.convertScaleAbs(source, alpha=1.5, beta=-20),
    )
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


def test_fallback_image_io_preserves_bgr_and_contract(tmp_path: Path) -> None:
    """Preserve BGR channel order and OpenCV missing-file/write contracts."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    image_path = tmp_path / "image.png"

    assert _imwrite(str(image_path), image)
    actual = _imread(str(image_path), _IMREAD_COLOR)
    assert actual is not None
    np.testing.assert_array_equal(actual, image)
    assert _imread(str(tmp_path / "missing.png"), _IMREAD_COLOR) is None

    alpha = np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], dtype=np.uint8)
    alpha_path = tmp_path / "alpha.png"
    assert _imwrite(str(alpha_path), alpha)
    np.testing.assert_array_equal(
        _imread(str(alpha_path), _IMREAD_UNCHANGED),
        cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED),
    )

    sixteen_bit = np.array([[0, 12345], [54321, 65535]], dtype=np.uint16)
    sixteen_bit_path = tmp_path / "sixteen-bit.png"
    assert _imwrite(str(sixteen_bit_path), sixteen_bit)
    np.testing.assert_array_equal(
        _imread(str(sixteen_bit_path), _IMREAD_UNCHANGED),
        cv2.imread(str(sixteen_bit_path), cv2.IMREAD_UNCHANGED),
    )
    np.testing.assert_array_equal(
        _imread(str(sixteen_bit_path), _IMREAD_COLOR),
        cv2.imread(str(sixteen_bit_path), cv2.IMREAD_COLOR),
    )
