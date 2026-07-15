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
            np.array(
                [[[0, 255, 255], [30, 255, 255]], [[60, 255, 255], [150, 255, 255]]],
                dtype=np.uint8,
            ),
            _COLOR_HSV2BGR,
            cv2.COLOR_HSV2BGR,
            1,
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


def test_fallback_merge_matches_opencv() -> None:
    """Match OpenCV channel merging."""
    bgr = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    channels = cv2.split(bgr)

    np.testing.assert_array_equal(_merge(channels), cv2.merge(channels))
