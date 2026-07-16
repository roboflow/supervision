"""Private Hershey text fallbacks for the OpenCV compatibility facade."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from functools import lru_cache
from importlib.resources import files
from itertools import pairwise
from typing import Any, cast

import numpy.typing as npt

from supervision._cv2._drawing import _line
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
    _LINE_8,
)

_ImageArray = npt.NDArray[Any]
_FontData = tuple[int, ...]
_FONT_NAMES = {
    _FONT_HERSHEY_SIMPLEX: "HersheySimplex",
    _FONT_HERSHEY_PLAIN: "HersheyPlain",
    _FONT_HERSHEY_DUPLEX: "HersheyDuplex",
    _FONT_HERSHEY_COMPLEX: "HersheyComplex",
    _FONT_HERSHEY_TRIPLEX: "HersheyTriplex",
    _FONT_HERSHEY_COMPLEX_SMALL: "HersheyComplexSmall",
    _FONT_HERSHEY_SCRIPT_SIMPLEX: "HersheyScriptSimplex",
    _FONT_HERSHEY_SCRIPT_COMPLEX: "HersheyScriptComplex",
}
_ITALIC_FONT_NAMES = {
    _FONT_HERSHEY_PLAIN: "HersheyPlainItalic",
    _FONT_HERSHEY_COMPLEX: "HersheyComplexItalic",
    _FONT_HERSHEY_TRIPLEX: "HersheyTriplexItalic",
    _FONT_HERSHEY_COMPLEX_SMALL: "HersheyComplexSmallItalic",
}
_XY_SHIFT = 16
_XY_ONE = 1 << _XY_SHIFT
_ASCII_FIRST = ord(" ")
_ASCII_LAST = ord("~")
_COORDINATE_ORIGIN = ord("R")


@lru_cache(maxsize=1)
def _load_font_data() -> tuple[tuple[str, ...], dict[str, _FontData]]:
    """Load and verify the packaged OpenCV-derived glyph tables once."""
    resource_dir = files("supervision._cv2").joinpath("data")
    glyph_resource = resource_dir.joinpath("hershey_fonts.json")
    provenance_resource = resource_dir.joinpath("hershey_provenance.json")
    glyph_bytes = glyph_resource.read_bytes()
    provenance = cast(
        dict[str, str],
        json.loads(provenance_resource.read_text(encoding="utf-8")),
    )
    digest = hashlib.sha256(glyph_bytes).hexdigest()
    if digest != provenance["data_sha256"]:
        raise RuntimeError("Packaged Hershey glyph data failed its checksum")

    payload = cast(dict[str, Any], json.loads(glyph_bytes))
    glyphs = tuple(cast(list[str], payload["glyphs"]))
    faces = {
        name: tuple(values)
        for name, values in cast(dict[str, list[int]], payload["faces"]).items()
    }
    return glyphs, faces


def _font_data(font_face: int) -> _FontData:
    """Return the OpenCV Hershey index table selected by a font face."""
    base_face = font_face & 15
    try:
        regular_name = _FONT_NAMES[base_face]
    except KeyError as error:
        raise ValueError(f"Unsupported Hershey font face: {font_face}") from error

    name = regular_name
    if font_face & _FONT_ITALIC:
        name = _ITALIC_FONT_NAMES.get(base_face, regular_name)
    _, faces = _load_font_data()
    return faces[name]


def _iter_text_bytes(text: str, font_face: int) -> Iterator[int]:
    """Yield OpenCV-compatible glyph code points from UTF-8 text."""
    encoded = text.encode("utf-8", errors="replace")
    index = 0
    while index < len(encoded):
        code = encoded[index]
        index += 1
        left_boundary = _ASCII_FIRST
        right_boundary = _ASCII_LAST + 1
        if code >= 0x80 and font_face == _FONT_HERSHEY_COMPLEX:
            # OpenCV's Complex face maps two UTF-8 Cyrillic ranges into its
            # extended glyph table; other faces render unsupported bytes as '?'.
            if code == 0xD0 and index < len(encoded) and 0x90 <= encoded[index] <= 0xBF:
                code = encoded[index] - 17
                index += 1
                right_boundary = 175
            elif (
                code == 0xD1 and index < len(encoded) and 0x80 <= encoded[index] <= 0x8F
            ):
                code = encoded[index] + 47
                index += 1
                left_boundary = 175
                right_boundary = 191
            else:
                index += _utf8_continuation_count(code, encoded, index)
                code = ord("?")
        elif code >= 0x80:
            code = ord("?")

        if code < left_boundary or code >= right_boundary:
            code = ord("?")
        yield code


def _utf8_continuation_count(code: int, encoded: bytes, index: int) -> int:
    """Return how many UTF-8 continuation bytes OpenCV skips for a lead byte."""
    if code < 0xC0:
        return 0
    expected = 1
    if code >= 0xF0:
        expected = 3
    elif code >= 0xE0:
        expected = 2
    return min(expected, len(encoded) - index)


def _glyphs_for_text(text: str, font_face: int) -> Iterator[str]:
    """Yield glyph stroke strings selected by a font face and text."""
    glyphs, _ = _load_font_data()
    face = _font_data(font_face)
    for code in _iter_text_bytes(text, font_face):
        glyphs_index = face[code - _ASCII_FIRST + 1]
        yield glyphs[glyphs_index]


def _round_fixed(value: float) -> int:
    """Round a coordinate to the fixed-point precision used by OpenCV."""
    return round(value * _XY_ONE)


def _text_metrics(
    font_face: int, text: str, font_scale: float, thickness: int
) -> tuple[int, int, int]:
    """Compute OpenCV Hershey cap height, baseline, and text width."""
    font = _font_data(font_face)
    cap_line = (font[0] >> 4) & 15
    base_line = font[0] & 15
    height = round((cap_line + base_line) * font_scale + (thickness + 1) // 2)
    width = sum(
        (ord(glyph[1]) - ord(glyph[0])) * font_scale
        for glyph in _glyphs_for_text(text, font_face)
    )
    baseline = round(base_line * font_scale + thickness * 0.5)
    return round(width + thickness), height, baseline


def _get_text_size(
    text: str,
    fontFace: int,
    fontScale: float,
    thickness: int,
) -> tuple[tuple[int, int], int]:
    """Return OpenCV-compatible Hershey text dimensions and baseline."""
    width, height, baseline = _text_metrics(fontFace, text, fontScale, thickness)
    return (width, height), baseline


def _stroke_segments(glyph: str) -> Iterator[tuple[tuple[int, int], ...]]:
    """Decode the space-separated coordinate strokes in one glyph string."""
    for stroke in glyph[2:].split():
        points = tuple(
            (
                ord(stroke[index]) - _COORDINATE_ORIGIN,
                ord(stroke[index + 1]) - _COORDINATE_ORIGIN,
            )
            for index in range(0, len(stroke), 2)
        )
        if len(points) > 1:
            yield points


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
    """Render OpenCV Hershey strokes into an image using the fallback line primitive."""
    if not text:
        return img

    scale = _round_fixed(fontScale)
    vertical_scale = -scale if bottomLeftOrigin else scale
    font = _font_data(fontFace)
    baseline = -(font[0] & 15)
    view_x = org[0] << _XY_SHIFT
    view_y = (org[1] << _XY_SHIFT) + baseline * vertical_scale

    for glyph in _glyphs_for_text(text, fontFace):
        left = ord(glyph[0]) - _COORDINATE_ORIGIN
        right = ord(glyph[1]) - _COORDINATE_ORIGIN
        advance = right * scale
        view_x -= left * scale
        for segment in _stroke_segments(glyph):
            points = [
                (
                    (x * scale + view_x) >> _XY_SHIFT,
                    (y * vertical_scale + view_y) >> _XY_SHIFT,
                )
                for x, y in segment
            ]
            for start, end in pairwise(points):
                _line(img, start, end, color, thickness, lineType)
        view_x += advance

    return img
