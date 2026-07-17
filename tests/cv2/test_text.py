"""Tests for the Pillow-based fallback OpenCV text implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supervision._cv2._text import _get_text_size, _put_text
from supervision._cv2.constants import (
    _FONT_HERSHEY_COMPLEX,
    _FONT_HERSHEY_COMPLEX_SMALL,
    _FONT_HERSHEY_DUPLEX,
    _FONT_HERSHEY_PLAIN,
    _FONT_HERSHEY_SCRIPT_COMPLEX,
    _FONT_HERSHEY_SCRIPT_SIMPLEX,
    _FONT_HERSHEY_SIMPLEX,
    _FONT_HERSHEY_TRIPLEX,
    _FONT_ITALIC,
)

FONT_FACES = [
    pytest.param(_FONT_HERSHEY_SIMPLEX, id="simplex"),
    pytest.param(_FONT_HERSHEY_PLAIN, id="plain"),
    pytest.param(_FONT_HERSHEY_DUPLEX, id="duplex"),
    pytest.param(_FONT_HERSHEY_COMPLEX, id="complex"),
    pytest.param(_FONT_HERSHEY_TRIPLEX, id="triplex"),
    pytest.param(_FONT_HERSHEY_COMPLEX_SMALL, id="complex-small"),
    pytest.param(_FONT_HERSHEY_SCRIPT_SIMPLEX, id="script-simplex"),
    pytest.param(_FONT_HERSHEY_SCRIPT_COMPLEX, id="script-complex"),
]


def _run_without_opencv(source: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet with imports of cv2 blocked."""
    environment = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=environment,
    )


class TestGetTextSize:
    """Text-metric contract of the Pillow fallback getTextSize."""

    @pytest.mark.parametrize("font_face", FONT_FACES)
    @pytest.mark.parametrize("italic", [False, True], ids=["regular", "italic"])
    def test_returns_positive_metrics_for_every_face(
        self, font_face: int, italic: bool
    ) -> None:
        """Report positive width, height, and baseline for every accepted face."""
        actual_font_face = font_face | (_FONT_ITALIC if italic else 0)

        (width, height), baseline = _get_text_size(
            "Supervision", actual_font_face, 0.75, 2
        )

        assert width > 0
        assert height > 0
        assert baseline > 0

    def test_height_is_string_independent(self) -> None:
        """Match OpenCV's contract of a content-independent line height."""
        (_, short_height), _ = _get_text_size("i", _FONT_HERSHEY_SIMPLEX, 1.0, 1)
        (_, tall_height), _ = _get_text_size("Ag|", _FONT_HERSHEY_SIMPLEX, 1.0, 1)

        assert short_height == tall_height

    def test_width_grows_with_text_length(self) -> None:
        """Report a wider box for a longer string at the same scale."""
        (short_width, _), _ = _get_text_size("ab", _FONT_HERSHEY_SIMPLEX, 1.0, 1)
        (long_width, _), _ = _get_text_size("abcdef", _FONT_HERSHEY_SIMPLEX, 1.0, 1)

        assert long_width > short_width

    def test_metrics_grow_with_scale(self) -> None:
        """Report a larger box as the font scale increases."""
        (small_width, small_height), _ = _get_text_size(
            "text", _FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        (large_width, large_height), _ = _get_text_size(
            "text", _FONT_HERSHEY_SIMPLEX, 2.0, 1
        )

        assert large_width > small_width
        assert large_height > small_height


class TestPutText:
    """Rendering behavior of the Pillow fallback putText."""

    @pytest.mark.parametrize("font_face", FONT_FACES)
    @pytest.mark.parametrize("italic", [False, True], ids=["regular", "italic"])
    def test_renders_every_font_face_in_place(
        self, font_face: int, italic: bool
    ) -> None:
        """Draw visible text in place for every accepted font and italic combo."""
        image = np.zeros((96, 256, 3), dtype=np.uint8)
        actual_font_face = font_face | (_FONT_ITALIC if italic else 0)

        result = _put_text(
            image, "Supervision", (8, 60), actual_font_face, 0.75, (255, 255, 255), 1
        )

        assert result is image
        assert np.any(image)

    def test_empty_text_leaves_image_untouched(self) -> None:
        """Leave the scene unchanged when the string is empty."""
        image = np.zeros((32, 64, 3), dtype=np.uint8)

        result = _put_text(image, "", (4, 20), _FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

        assert result is image
        assert not np.any(image)

    def test_rendered_text_stays_within_reported_box(self) -> None:
        """Keep drawn pixels within the getTextSize rectangle above the baseline."""
        image = np.zeros((120, 320, 3), dtype=np.uint8)
        org = (20, 80)
        (width, height), baseline = _get_text_size(
            "person 0.87", _FONT_HERSHEY_SIMPLEX, 1.0, 2
        )

        _put_text(
            image, "person 0.87", org, _FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        rows, columns = np.nonzero(np.any(image, axis=2))
        assert columns.min() >= org[0]
        assert columns.max() <= org[0] + width
        assert rows.min() >= org[1] - height
        assert rows.max() <= org[1] + baseline

    def test_rejects_bottom_left_origin(self) -> None:
        """Reject the unsupported inverted-axis origin mode."""
        image = np.zeros((32, 64, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="bottomLeftOrigin"):
            _put_text(
                image,
                "x",
                (4, 20),
                _FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                bottomLeftOrigin=True,
            )


class TestFallbackWithoutOpenCV:
    """Production-shaped text calls with OpenCV imports blocked."""

    def test_facade_text_works_when_opencv_is_blocked(self) -> None:
        """Exercise the facade getTextSize and putText in a cv2-free subprocess."""
        completed = _run_without_opencv(
            """
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())

import numpy as np

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
)
assert result is image
assert np.any(image)
"""
        )

        assert completed.returncode == 0, completed.stderr

    def test_draw_text_consumer_uses_fallback_without_opencv(self) -> None:
        """Exercise the public draw_text consumer in a cv2-free subprocess."""
        completed = _run_without_opencv(
            """
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())

import numpy as np

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
