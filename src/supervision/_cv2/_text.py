"""Private Pillow-based text fallback for the OpenCV compatibility facade.

OpenCV renders text with built-in Hershey stroke fonts. The fallback instead
draws a proportional TrueType face (DejaVu Sans, shipped with Matplotlib, an
existing required dependency), so glyph shapes and text metrics differ from
OpenCV within the documented visual-divergence tier. ``getTextSize`` derives
its box from the same font ``putText`` renders with, so the reported rectangle
always encloses the drawn text.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy.typing as npt

from supervision._cv2._drawing import _drawing_mask, _paint
from supervision._cv2.constants import _FONT_ITALIC, _LINE_8

_ImageArray = npt.NDArray[Any]

# OpenCV's font_scale is unit-relative rather than a pixel size. This factor
# maps it to a Pillow point size whose cap height lands near OpenCV's Hershey
# Simplex at the same scale; it is a readability choice, not an exact metric
# match (which no proportional TrueType face can provide).
_PIXELS_PER_SCALE = 32


@cache
def _font_path(italic: bool) -> str:
    """Locate a bundled DejaVu Sans face through Matplotlib's font manager."""
    from matplotlib import font_manager

    style = "italic" if italic else "normal"
    properties = font_manager.FontProperties(family="DejaVu Sans", style=style)
    return str(font_manager.findfont(properties))


@cache
def _load_font(font_face: int, font_scale: float) -> Any:
    """Return a cached Pillow font sized for an OpenCV font scale."""
    from PIL import ImageFont

    size = max(1, round(font_scale * _PIXELS_PER_SCALE))
    return ImageFont.truetype(_font_path(bool(font_face & _FONT_ITALIC)), size)


def _get_text_size(
    text: str, fontFace: int, fontScale: float, thickness: int
) -> tuple[tuple[int, int], int]:
    """Return an OpenCV-shaped ``((width, height), baseline)`` for the face.

    Height is the font ascent and baseline the descent, both string-independent
    like OpenCV's contract, so consumers get stable row heights. Thickness pads
    the box the way OpenCV widens strokes.
    """
    font = _load_font(fontFace, fontScale)
    ascent, descent = font.getmetrics()
    width = round(font.getlength(text)) + max(0, thickness)
    height = ascent + (thickness + 1) // 2
    baseline = descent + thickness // 2
    return (width, height), baseline


def _put_text(
    img: _ImageArray,
    text: str,
    org: tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: Any,
    thickness: int = 1,
    lineType: int = _LINE_8,
    bottomLeftOrigin: bool = False,
) -> _ImageArray:
    """Render text with a Pillow face, anchored at OpenCV's baseline origin.

    Thickness maps to a Pillow stroke width to emulate OpenCV's bolder strokes.
    ``bottomLeftOrigin`` (an inverted-axis mode no Supervision caller uses) is
    rejected rather than silently ignored.
    """
    del lineType
    if bottomLeftOrigin:
        raise ValueError("bottomLeftOrigin is not supported by the fallback")
    if not text:
        return img

    font = _load_font(fontFace, fontScale)
    stroke_width = max(0, thickness - 1)
    x, y = round(org[0]), round(org[1])

    def draw_text(draw: Any) -> None:
        """Draw the string onto the one-bit mask at the baseline anchor."""
        draw.text(
            (x, y),
            text,
            fill=1,
            font=font,
            anchor="ls",
            stroke_width=stroke_width,
        )

    return _paint(img, _drawing_mask(img, draw_text), color)
