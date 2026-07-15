"""Tests for the private OpenCV facade."""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

from supervision import _cv2

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


REQUIRED_SYMBOLS = {
    "VideoCapture",
    "VideoWriter",
    "VideoWriter_fourcc",
    "addWeighted",
    "approxPolyDP",
    "blur",
    "circle",
    "connectedComponents",
    "connectedComponentsWithStats",
    "contourArea",
    "convertScaleAbs",
    "copyMakeBorder",
    "cvtColor",
    "distanceTransform",
    "drawContours",
    "ellipse",
    "fillPoly",
    "findContours",
    "flip",
    "getRotationMatrix2D",
    "getTextSize",
    "imread",
    "imwrite",
    "intersectConvexConvex",
    "line",
    "mean",
    "merge",
    "polylines",
    "putText",
    "rectangle",
    "resize",
    "split",
    "warpAffine",
}


@pytest.mark.parametrize(
    "symbol",
    [
        pytest.param(symbol, id=symbol.lower().replace("_", "-"))
        for symbol in sorted(REQUIRED_SYMBOLS)
    ],
)
def test_facade_exports_required_opencv_symbol(symbol: str) -> None:
    """Expose each OpenCV symbol used by production call sites."""
    assert symbol in _cv2.__all__
    assert hasattr(_cv2, symbol)


def test_facade_reports_opencv_backend() -> None:
    """Report OpenCV when the native backend is available."""
    assert _cv2.BACKEND_NAME == "opencv"


def test_facade_routes_color_calls_to_opencv() -> None:
    """Route color conversion calls through the Supervision facade."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB),
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )


def test_facade_routes_resize_calls_to_opencv() -> None:
    """Route resize calls through the Supervision facade."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _cv2.resize(image, (4, 2), interpolation=_cv2.INTER_NEAREST),
        cv2.resize(image, (4, 2), interpolation=cv2.INTER_NEAREST),
    )


def test_facade_does_not_hide_a_breaking_opencv_import() -> None:
    """Raise when cv2 imports but no longer exposes a required symbol."""
    code = """
import sys
import types

sys.modules["cv2"] = types.ModuleType("cv2")
from supervision import _cv2
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cannot import name" in result.stderr
