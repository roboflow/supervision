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
    "FONT_HERSHEY_SIMPLEX",
    "IMREAD_COLOR",
    "IMREAD_UNCHANGED",
    "INTER_LINEAR",
    "INTER_NEAREST",
    "LINE_4",
    "LINE_AA",
    "RETR_CCOMP",
    "RETR_TREE",
]


@pytest.mark.parametrize("name", OPENCV_CONSTANTS)
def test_fallback_constant_matches_opencv(name: str) -> None:
    """Keep each private fallback constant aligned with the OpenCV reference."""
    actual = getattr(_cv2, f"_{name}")
    expected = getattr(cv2, name)

    assert actual == expected


def test_facade_imports_without_opencv() -> None:
    """Keep fallback imports and constants valid when cv2 is unavailable."""
    env = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, env.get("PYTHONPATH")))
    )
    expected_constants = {name: getattr(cv2, name) for name in OPENCV_CONSTANTS}
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
expected = {expected_constants!r}
actual = {{name: getattr(_cv2, name) for name in expected}}
fallback = {{name: getattr(_cv2, f"_{{name}}") for name in expected}}
assert fallback == expected
assert actual == expected
image = __import__("numpy").array([[[10, 20, 30]]], dtype="uint8")
assert _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB).tolist() == [[[30, 20, 10]]]
assert _cv2.resize(image, (2, 1), interpolation=_cv2.INTER_NEAREST).shape == (1, 2, 3)
bindings = {{
    "addWeighted": "_add_weighted",
    "blur": "_blur",
    "convertScaleAbs": "_convert_scale_abs",
    "copyMakeBorder": "_copy_make_border",
    "cvtColor": "_cvt_color",
    "distanceTransform": "_distance_transform",
    "flip": "_flip",
    "getRotationMatrix2D": "_get_rotation_matrix_2d",
    "imread": "_imread",
    "imwrite": "_imwrite",
    "mean": "_mean",
    "merge": "_merge",
    "resize": "_resize",
    "split": "_split",
    "warpAffine": "_warp_affine",
}}
for public_name, private_name in bindings.items():
    assert getattr(_cv2, public_name) is getattr(_cv2, private_name)
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=True,
        env=env,
    )
