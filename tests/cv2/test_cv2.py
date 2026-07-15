"""Tests for the private OpenCV compatibility surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from supervision import _cv2

REQUIRED_SYMBOLS = {
    "BORDER_CONSTANT",
    "CAP_PROP_FPS",
    "CAP_PROP_FRAME_COUNT",
    "CAP_PROP_FRAME_HEIGHT",
    "CAP_PROP_FRAME_WIDTH",
    "CAP_PROP_POS_FRAMES",
    "CC_STAT_AREA",
    "CHAIN_APPROX_SIMPLE",
    "COLOR_BGR2GRAY",
    "COLOR_BGR2RGB",
    "COLOR_GRAY2BGR",
    "COLOR_HSV2BGR",
    "COLOR_RGB2BGR",
    "DIST_L2",
    "FONT_HERSHEY_SIMPLEX",
    "IMREAD_COLOR",
    "IMREAD_UNCHANGED",
    "INTER_LINEAR",
    "INTER_NEAREST",
    "LINE_4",
    "LINE_AA",
    "RETR_CCOMP",
    "RETR_TREE",
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

OPENCV_CONSTANTS = {
    "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
    "CAP_PROP_FPS": cv2.CAP_PROP_FPS,
    "CAP_PROP_FRAME_COUNT": cv2.CAP_PROP_FRAME_COUNT,
    "CAP_PROP_FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
    "CAP_PROP_FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
    "CAP_PROP_POS_FRAMES": cv2.CAP_PROP_POS_FRAMES,
    "CC_STAT_AREA": cv2.CC_STAT_AREA,
    "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
    "COLOR_BGR2GRAY": cv2.COLOR_BGR2GRAY,
    "COLOR_BGR2RGB": cv2.COLOR_BGR2RGB,
    "COLOR_GRAY2BGR": cv2.COLOR_GRAY2BGR,
    "COLOR_HSV2BGR": cv2.COLOR_HSV2BGR,
    "COLOR_RGB2BGR": cv2.COLOR_RGB2BGR,
    "DIST_L2": cv2.DIST_L2,
    "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "IMREAD_COLOR": cv2.IMREAD_COLOR,
    "IMREAD_UNCHANGED": cv2.IMREAD_UNCHANGED,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "LINE_4": cv2.LINE_4,
    "LINE_AA": cv2.LINE_AA,
    "RETR_CCOMP": cv2.RETR_CCOMP,
    "RETR_TREE": cv2.RETR_TREE,
}


def test_facade_exports_the_required_opencv_surface() -> None:
    """Expose every OpenCV symbol used by production call sites."""
    assert REQUIRED_SYMBOLS <= set(_cv2.__all__)
    assert all(hasattr(_cv2, symbol) for symbol in REQUIRED_SYMBOLS)
    assert _cv2.BACKEND_NAME == "opencv"


def test_facade_constants_match_opencv() -> None:
    """Keep compatibility constants aligned with the OpenCV reference."""
    facade_constants = {name: getattr(_cv2, name) for name in OPENCV_CONSTANTS}

    assert facade_constants == OPENCV_CONSTANTS


def test_facade_calls_the_package_imported_surface() -> None:
    """Route production-style calls through the Supervision facade."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB),
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )
    np.testing.assert_array_equal(
        _cv2.resize(image, (4, 2), interpolation=_cv2.INTER_NEAREST),
        cv2.resize(image, (4, 2), interpolation=cv2.INTER_NEAREST),
    )


def test_facade_imports_without_opencv() -> None:
    """Keep fallback imports and constants valid when cv2 is unavailable."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    code = f"""
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())
from supervision import _cv2

assert _cv2._IS_CV2_AVAILABLE is False
assert _cv2.BACKEND_NAME == "fallback"
expected = {OPENCV_CONSTANTS!r}
actual = {{name: getattr(_cv2, name) for name in expected}}
assert actual == expected
try:
    _cv2.resize(None, (1, 1), interpolation=_cv2.INTER_NEAREST)
except _cv2.BackendUnavailableError:
    pass
else:
    raise AssertionError("missing OpenCV must fail explicitly")
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=True,
        env=env,
    )


def test_facade_does_not_hide_a_breaking_opencv_import() -> None:
    """Raise when cv2 imports but no longer exposes a required symbol."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
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
        env=env,
    )

    assert result.returncode != 0
    assert "cannot import name" in result.stderr
