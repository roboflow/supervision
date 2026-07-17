"""Tests for private transform and filter fallbacks."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from supervision._cv2._transform import _blur

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


def test_fallback_blur_preserves_shape_and_dtype() -> None:
    """Preserve source shape and dtype during blurring."""
    source = np.arange(25, dtype=np.uint8).reshape(5, 5)
    blurred = _blur(source, (3, 3))

    assert blurred.shape == source.shape
    assert blurred.dtype == source.dtype
