"""Tests for the fallback OpenCV Hershey text implementation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from supervision._cv2._text import _get_text_size, _put_text

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )

FONT_FACES = [
    pytest.param(cv2.FONT_HERSHEY_SIMPLEX, id="simplex"),
    pytest.param(cv2.FONT_HERSHEY_PLAIN, id="plain"),
    pytest.param(cv2.FONT_HERSHEY_DUPLEX, id="duplex"),
    pytest.param(cv2.FONT_HERSHEY_COMPLEX, id="complex"),
    pytest.param(cv2.FONT_HERSHEY_TRIPLEX, id="triplex"),
    pytest.param(cv2.FONT_HERSHEY_COMPLEX_SMALL, id="complex-small"),
    pytest.param(cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, id="script-simplex"),
    pytest.param(cv2.FONT_HERSHEY_SCRIPT_COMPLEX, id="script-complex"),
]


def _run_without_opencv(source: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet with imports of cv2 blocked."""
    environment = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH")))
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=environment,
    )


@pytest.mark.parametrize("font_face", FONT_FACES)
@pytest.mark.parametrize("italic", [False, True], ids=["regular", "italic"])
@pytest.mark.parametrize(
    "text",
    ["", "OpenCV 123", "Tg!?"],
    ids=["empty", "mixed", "punctuation"],
)
def test_fallback_text_metrics_match_opencv(
    font_face: int, italic: bool, text: str
) -> None:
    """Match OpenCV text dimensions and baseline for every accepted font face."""
    actual_font_face = font_face | (cv2.FONT_ITALIC if italic else 0)

    actual = _get_text_size(text, actual_font_face, 0.75, 2)
    expected = cv2.getTextSize(text, actual_font_face, 0.75, 2)

    assert actual == expected


@pytest.mark.parametrize(
    "font_face",
    [
        pytest.param(cv2.FONT_HERSHEY_SIMPLEX, id="simplex"),
        pytest.param(cv2.FONT_HERSHEY_COMPLEX, id="complex"),
        pytest.param(
            cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC,
            id="complex-italic",
        ),
    ],
)
@pytest.mark.parametrize(
    "text",
    ["é", "Ж", "\N{GREEK SMALL LETTER ALPHA}", "😀"],
    ids=["latin", "cyrillic", "greek", "emoji"],
)
def test_fallback_text_metrics_match_opencv_for_unicode(
    font_face: int, text: str
) -> None:
    """Match OpenCV's supported and replacement-glyph Unicode behavior."""
    actual = _get_text_size(text, font_face, 0.75, 2)
    expected = cv2.getTextSize(text, font_face, 0.75, 2)

    assert actual == expected


@pytest.mark.parametrize("font_face", FONT_FACES)
@pytest.mark.parametrize("italic", [False, True], ids=["regular", "italic"])
def test_fallback_text_renders_every_font_face(font_face: int, italic: bool) -> None:
    """Render visible text for every accepted font and italic combination."""
    image = np.zeros((96, 256, 3), dtype=np.uint8)
    actual_font_face = font_face | (cv2.FONT_ITALIC if italic else 0)

    result = _put_text(
        image,
        "Supervision",
        (8, 60),
        actual_font_face,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    assert result is image
    assert np.any(image)


def test_fallback_text_data_has_verified_provenance() -> None:
    """Verify the packaged glyph data against its source manifest."""
    package_data = files("supervision._cv2").joinpath("data")
    glyph_data = package_data.joinpath("hershey_fonts.json").read_bytes()
    provenance = json.loads(
        package_data.joinpath("hershey_provenance.json").read_text(encoding="utf-8")
    )

    assert hashlib.sha256(glyph_data).hexdigest() == provenance["data_sha256"]
    assert provenance["source"].endswith("modules/imgproc/src/hershey_fonts.cpp")
    assert provenance["license"] == "Intel-Willow-Garage-BSD-style"


def test_fallback_text_works_when_opencv_is_blocked() -> None:
    """Exercise production-shaped text calls in a cv2-free subprocess."""
    completed = _run_without_opencv(
        """
import sys

import numpy as np


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())

from supervision import _cv2

assert _cv2.BACKEND_NAME == "fallback"
size, baseline = _cv2.getTextSize(
    "Fallback", _cv2.FONT_HERSHEY_COMPLEX | _cv2.FONT_ITALIC, 0.75, 2
)
assert size[0] > 0
assert baseline > 0
image = np.zeros((96, 256, 3), dtype=np.uint8)
result = _cv2.putText(
    image,
    "Fallback",
    (8, 60),
    _cv2.FONT_HERSHEY_COMPLEX | _cv2.FONT_ITALIC,
    0.75,
    (255, 255, 255),
    1,
    _cv2.LINE_AA,
)
assert result is image
assert np.any(image)
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_draw_text_consumer_uses_fallback_without_opencv() -> None:
    """Exercise the public draw_text consumer in a cv2-free subprocess."""
    completed = _run_without_opencv(
        """
import sys

import numpy as np


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())

from supervision.draw.color import Color
from supervision.draw.utils import draw_text
from supervision.geometry.core import Point

image = np.zeros((96, 256, 3), dtype=np.uint8)
result = draw_text(
    image,
    "Fallback",
    Point(80, 40),
    text_font=7,
    text_scale=0.75,
    text_color=Color.WHITE,
)
assert result is image
assert np.any(image)
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_fallback_text_rejects_unknown_font_face() -> None:
    """Reject a font face outside the supported Hershey family."""
    with pytest.raises(ValueError, match="font"):
        _get_text_size("text", 8, 1.0, 1)
