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


def test_fallback_color_operations_match_opencv() -> None:
    """Match OpenCV for supported color conversions and channel helpers."""
    bgr = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    gray = np.array([[0, 64], [128, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _cvt_color(bgr, _COLOR_BGR2RGB),
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
    )
    np.testing.assert_array_equal(
        _cvt_color(bgr, _COLOR_BGR2GRAY),
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
    )
    np.testing.assert_array_equal(
        _cvt_color(gray, _COLOR_GRAY2BGR),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    )
    hsv = np.array(
        [[[0, 255, 255], [30, 255, 255]], [[60, 255, 255], [150, 255, 255]]],
        dtype=np.uint8,
    )
    np.testing.assert_allclose(
        _cvt_color(hsv, _COLOR_HSV2BGR),
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
        atol=1,
        rtol=0,
    )
    np.testing.assert_array_equal(_merge(_split(bgr)), bgr)
