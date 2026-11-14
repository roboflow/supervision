"""Tests for private color and channel fallbacks."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from supervision._cv2._color import _cvt_color, _merge, _split
from supervision._cv2.constants import (
    _COLOR_BGR2GRAY,
    _COLOR_BGR2RGB,
    _COLOR_GRAY2BGR,
    _COLOR_HSV2BGR,
    _COLOR_RGB2BGR,
)

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    ("source", "fallback_code", "opencv_code", "atol"),
    [
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            _COLOR_BGR2RGB,
            cv2.COLOR_BGR2RGB,
            0,
            id="bgr-to-rgb",
        ),
        pytest.param(
            np.array([[[10, 20, 30, 40]]], dtype=np.uint8),
            _COLOR_RGB2BGR,
            cv2.COLOR_RGB2BGR,
            0,
            id="rgba-to-bgr",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            _COLOR_BGR2GRAY,
            cv2.COLOR_BGR2GRAY,
            0,
            id="bgr-to-gray",
        ),
        pytest.param(
            np.array([[0, 64], [128, 255]], dtype=np.uint8),
            _COLOR_GRAY2BGR,
            cv2.COLOR_GRAY2BGR,
            0,
            id="gray-to-bgr",
        ),
        pytest.param(
            np.array([[[0], [64]], [[128], [255]]], dtype=np.uint8),
            _COLOR_GRAY2BGR,
            cv2.COLOR_GRAY2BGR,
            0,
            id="single-channel-gray-to-bgr",
        ),
        pytest.param(
            np.array(
                [
                    [[10, 20, 30, 0], [40, 50, 60, 128]],
                    [[70, 80, 90, 255], [1, 2, 3, 4]],
                ],
                dtype=np.uint8,
            ),
            _COLOR_BGR2GRAY,
            cv2.COLOR_BGR2GRAY,
            0,
            id="bgra-to-gray",
        ),
        pytest.param(
            np.array(
                [[[0, 255, 255], [30, 255, 255]], [[60, 255, 255], [150, 255, 255]]],
                dtype=np.uint8,
            ),
            _COLOR_HSV2BGR,
            cv2.COLOR_HSV2BGR,
            0,
            id="hsv-to-bgr",
        ),
    ],
)
def test_fallback_color_operations_match_opencv(
    source: np.ndarray, fallback_code: int, opencv_code: int, atol: int
) -> None:
    """Match OpenCV for each supported color conversion."""
    actual = _cvt_color(source, fallback_code)
    expected = cv2.cvtColor(source, opencv_code)

    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def test_fallback_split_matches_opencv() -> None:
    """Match OpenCV channel splitting."""
    bgr = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )

    actual = _split(bgr)
    expected = cv2.split(bgr)

    assert len(actual) == len(expected)
    for actual_channel, expected_channel in zip(actual, expected):
        np.testing.assert_array_equal(actual_channel, expected_channel)


def test_fallback_bgr_to_gray_matches_opencv_fixed_point_rounding() -> None:
    """Match OpenCV at a color where floating coefficients round differently."""
    source = np.array([[[0, 1, 187]]], dtype=np.uint8)

    actual = _cvt_color(source, _COLOR_BGR2GRAY)
    expected = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_fallback_bgr_to_gray_matches_opencv_for_integer_images(
    dtype: type[np.integer],
) -> None:
    """Match OpenCV's fixed-point conversion for 8- and 16-bit images."""
    rng = np.random.default_rng(0)
    source = rng.integers(0, np.iinfo(dtype).max, (64, 64, 3), endpoint=True)
    source = source.astype(dtype)
    source[0, 0] = (0, 1, 187)
    source[0, 1] = (100, 200, 300 if dtype == np.uint16 else 255)
    source[0, 2] = np.iinfo(dtype).max

    actual = _cvt_color(source, _COLOR_BGR2GRAY)
    expected = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_fallback_hsv_to_bgr_matches_opencv_for_float32_images() -> None:
    """Read float32 hue in degrees and saturation and value in [0, 1]."""
    rng = np.random.default_rng(0)
    source = np.stack(
        (
            rng.uniform(0, 360, (32, 32)),
            rng.uniform(0, 1, (32, 32)),
            rng.uniform(0, 1, (32, 32)),
        ),
        axis=-1,
    ).astype(np.float32)
    source[0, 0] = (120, 0.5, 0.8)
    source[0, 1] = (0, 1, 1)
    source[0, 2] = (300, 0, 0.25)

    actual = _cvt_color(source, _COLOR_HSV2BGR)
    expected = cv2.cvtColor(source, cv2.COLOR_HSV2BGR)

    assert actual.dtype == expected.dtype
    _assert_matches_opencv_hsv(actual, expected, min_exact_fraction=0.9)


def _assert_matches_opencv_hsv(
    actual: np.ndarray, expected: np.ndarray, min_exact_fraction: float
) -> None:
    """Check an HSV2BGR result against OpenCV, allowing its fused multiply-adds.

    OpenCV's Linux wheels fuse ``1 - s * f`` into one multiply-add, which the
    Windows and macOS wheels and this fallback do not, so a few results there
    differ by one 8-bit level or about 6e-8 in float32. Everything else must be exact.
    """
    assert actual.shape == expected.shape
    if actual.dtype == np.uint8:
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        assert difference.max() <= 1
    else:
        assert np.abs(actual.astype(np.float64) - expected).max() <= 2.0**-22
    assert np.mean(actual == expected) >= min_exact_fraction


class TestFallbackHSVToBGRMatchesOpenCV:
    def test_uint8_hsv_grid_matches_opencv(self) -> None:
        """Match OpenCV on every hue, including wrapped ones, across an S and V grid."""
        hue, saturation, value = np.meshgrid(
            np.arange(256), np.arange(0, 256, 3), np.arange(0, 256, 3), indexing="ij"
        )
        # 256-pixel rows are whole vector blocks on every OpenCV build.
        source = np.stack((hue, saturation, value), axis=-1).reshape(-1, 256, 3)
        source = source.astype(np.uint8)

        actual = _cvt_color(source, _COLOR_HSV2BGR)

        expected = cv2.cvtColor(source, cv2.COLOR_HSV2BGR)
        _assert_matches_opencv_hsv(actual, expected, min_exact_fraction=0.999)

    @pytest.mark.parametrize("width", [1, 7, 33])
    def test_uint8_row_ends_match_opencv(self, width: int) -> None:
        """Round the pixels OpenCV's scalar code converts at the end of each row."""
        rng = np.random.default_rng(width)
        source = rng.integers(0, 256, size=(64, width, 3), dtype=np.uint8)

        actual = _cvt_color(source, _COLOR_HSV2BGR)

        expected = cv2.cvtColor(source, cv2.COLOR_HSV2BGR)
        _assert_matches_opencv_hsv(actual, expected, min_exact_fraction=0.999)

    def test_heat_map_colors_match_opencv(self) -> None:
        """Match the full-saturation hues :class:`HeatMapAnnotator` draws."""
        source = np.full((4, 192, 3), 255, dtype=np.uint8)
        source[..., 0] = np.arange(192) % 180

        actual = _cvt_color(source, _COLOR_HSV2BGR)

        np.testing.assert_array_equal(actual, cv2.cvtColor(source, cv2.COLOR_HSV2BGR))


@pytest.mark.parametrize("dtype", [np.uint16, np.float64])
def test_fallback_hsv_to_bgr_rejects_dtypes_opencv_rejects(
    dtype: type[np.generic],
) -> None:
    """Raise for the depths OpenCV's HSV2BGR does not accept."""
    source = np.zeros((2, 2, 3), dtype=dtype)

    with pytest.raises(cv2.error):
        cv2.cvtColor(source, cv2.COLOR_HSV2BGR)
    with pytest.raises(ValueError, match="8-bit or float32"):
        _cvt_color(source, _COLOR_HSV2BGR)


def test_fallback_merge_matches_opencv() -> None:
    """Match OpenCV channel merging."""
    bgr = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    channels = cv2.split(bgr)

    np.testing.assert_array_equal(_merge(channels), cv2.merge(channels))
