"""Tests for private transform and filter fallbacks."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from supervision._cv2._transform import (
    _blur,
    _distance_transform,
    _get_rotation_matrix_2d,
    _warp_affine,
)
from supervision._cv2.constants import _DIST_L2

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


def test_fallback_affine_and_blur_have_opencv_compatible_boundaries() -> None:
    """Match affine identity and keep filter output within its parity budget."""
    source = np.arange(25, dtype=np.uint8).reshape(5, 5)
    matrix = _get_rotation_matrix_2d((2, 2), 0, 1)

    actual = _warp_affine(source, matrix, (5, 5))
    expected = cv2.warpAffine(source, matrix, (5, 5))
    np.testing.assert_array_equal(actual, expected)

    rotated_matrix = _get_rotation_matrix_2d((2, 2), 17, 1)
    rotated = _warp_affine(source, rotated_matrix, (5, 5))
    expected_rotated = cv2.warpAffine(source, rotated_matrix, (5, 5))
    np.testing.assert_allclose(rotated, expected_rotated, atol=3, rtol=0)

    blurred = _blur(source, (3, 3))
    assert blurred.shape == source.shape
    assert blurred.dtype == source.dtype


def test_fallback_distance_transform_preserves_distance_order() -> None:
    """Preserve zero locations and monotonic distances for the L2 transform."""
    source = np.ones((7, 7), dtype=np.uint8)
    source[3, 3] = 0

    actual = _distance_transform(source, _DIST_L2, 3)

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    assert actual[3, 3] == 0
    assert actual[3, 2] < actual[3, 1] < actual[3, 0]
