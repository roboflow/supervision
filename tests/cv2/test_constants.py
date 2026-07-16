"""Tests for private OpenCV compatibility constants."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from supervision import _cv2

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


OPENCV_CONSTANTS = [
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
    "FONT_HERSHEY_COMPLEX",
    "FONT_HERSHEY_COMPLEX_SMALL",
    "FONT_HERSHEY_DUPLEX",
    "FONT_HERSHEY_PLAIN",
    "FONT_HERSHEY_SCRIPT_COMPLEX",
    "FONT_HERSHEY_SCRIPT_SIMPLEX",
    "FONT_HERSHEY_SIMPLEX",
    "FONT_HERSHEY_TRIPLEX",
    "FONT_ITALIC",
    "IMREAD_COLOR",
    "IMREAD_UNCHANGED",
    "INTER_LINEAR",
    "INTER_NEAREST",
    "LINE_4",
    "LINE_8",
    "LINE_AA",
    "RETR_TREE",
]


def _run_without_opencv(source: str) -> None:
    """Run a Python snippet with cv2 imports blocked."""
    env = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, env.get("PYTHONPATH")))
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        check=True,
        env=env,
    )


@pytest.mark.parametrize("name", OPENCV_CONSTANTS)
def test_fallback_constant_matches_opencv(name: str) -> None:
    """Keep each private fallback constant aligned with the OpenCV reference."""
    actual = getattr(_cv2, f"_{name}")
    expected = getattr(cv2, name)

    assert actual == expected


def test_facade_reports_fallback_backend_without_opencv() -> None:
    """Report the fallback backend when cv2 is unavailable."""
    _run_without_opencv(
        """
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
"""
    )


def test_facade_preserves_constants_without_opencv() -> None:
    """Preserve the OpenCV constant values when cv2 is unavailable."""
    expected_constants = {name: getattr(cv2, name) for name in OPENCV_CONSTANTS}
    _run_without_opencv(
        f"""
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())
from supervision import _cv2

assert _cv2._IS_CV2_AVAILABLE is False
expected = {expected_constants!r}
actual = {{name: getattr(_cv2, name) for name in expected}}
fallback = {{name: getattr(_cv2, f"_{{name}}") for name in expected}}
assert fallback == expected
assert actual == expected
"""
    )


def test_facade_routes_color_calls_without_opencv() -> None:
    """Route color conversion calls to the fallback without cv2."""
    _run_without_opencv(
        """
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())
from supervision import _cv2

image = __import__("numpy").array([[[10, 20, 30]]], dtype="uint8")
assert _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB).tolist() == [[[30, 20, 10]]]
"""
    )


def test_facade_routes_resize_calls_without_opencv() -> None:
    """Route resize calls to the fallback without cv2."""
    _run_without_opencv(
        """
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())
from supervision import _cv2

image = __import__("numpy").array([[[10, 20, 30]]], dtype="uint8")
assert _cv2.resize(image, (2, 1), interpolation=_cv2.INTER_NEAREST).shape == (1, 2, 3)
"""
    )


@pytest.mark.parametrize(
    ("public_name", "private_name"),
    [
        pytest.param("addWeighted", "_add_weighted", id="addWeighted"),
        pytest.param("blur", "_blur", id="blur"),
        pytest.param("convertScaleAbs", "_convert_scale_abs", id="convertScaleAbs"),
        pytest.param("copyMakeBorder", "_copy_make_border", id="copyMakeBorder"),
        pytest.param("cvtColor", "_cvt_color", id="cvtColor"),
        pytest.param(
            "distanceTransform", "_distance_transform", id="distanceTransform"
        ),
        pytest.param("flip", "_flip", id="flip"),
        pytest.param(
            "getRotationMatrix2D", "_get_rotation_matrix_2d", id="getRotationMatrix2D"
        ),
        pytest.param("imread", "_imread", id="imread"),
        pytest.param("imwrite", "_imwrite", id="imwrite"),
        pytest.param("mean", "_mean", id="mean"),
        pytest.param("merge", "_merge", id="merge"),
        pytest.param("resize", "_resize", id="resize"),
        pytest.param("split", "_split", id="split"),
        pytest.param("warpAffine", "_warp_affine", id="warpAffine"),
        pytest.param("approxPolyDP", "_approx_poly_dp", id="approxPolyDP"),
        pytest.param(
            "connectedComponents",
            "_connected_components",
            id="connectedComponents",
        ),
        pytest.param(
            "connectedComponentsWithStats",
            "_connected_components_with_stats",
            id="connectedComponentsWithStats",
        ),
        pytest.param("contourArea", "_contour_area", id="contourArea"),
        pytest.param("circle", "_circle", id="circle"),
        pytest.param("drawContours", "_draw_contours", id="drawContours"),
        pytest.param("ellipse", "_ellipse", id="ellipse"),
        pytest.param("fillPoly", "_fill_poly", id="fillPoly"),
        pytest.param("findContours", "_find_contours", id="findContours"),
        pytest.param(
            "intersectConvexConvex",
            "_intersect_convex_convex",
            id="intersectConvexConvex",
        ),
        pytest.param("line", "_line", id="line"),
        pytest.param("polylines", "_polylines", id="polylines"),
        pytest.param("rectangle", "_rectangle", id="rectangle"),
    ],
)
def test_facade_binds_fallback_operation_without_opencv(
    public_name: str, private_name: str
) -> None:
    """Bind each public fallback operation to its private implementation."""
    _run_without_opencv(
        f"""
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())
from supervision import _cv2

assert getattr(_cv2, {public_name!r}) is getattr(_cv2, {private_name!r})
"""
    )
