"""Read depth maps from the files stereo and depth datasets ship."""

from __future__ import annotations

import math
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

_UINT16_MAX = 65535

#: Pillow modes of a single-channel 16-bit grayscale PNG.
_GRAY16_PNG_MODES = frozenset({"I", "I;16", "I;16B", "I;16L"})


def _read_png16(path: str | Path) -> npt.NDArray[np.uint16]:
    """Read a single-channel 16-bit grayscale PNG file as `uint16`.

    Raises:
        OSError: If the file cannot be opened, for example it does not exist.
        ValueError: If the file is not a decodable single-channel 16-bit PNG.
    """
    # Opened outside the decode helper, so a missing or unreadable file keeps its
    # FileNotFoundError or PermissionError instead of turning into "not a PNG".
    with open(path, "rb") as file:
        return _decode_gray_png(file, label=str(path))


def _decode_gray_png(source: BinaryIO, *, label: str) -> npt.NDArray[np.uint16]:
    """Decode a single-channel 16-bit grayscale PNG stream as `uint16`.

    Args:
        source: Binary stream holding the PNG bytes.
        label: Name of the source used in error messages, such as its path.

    Raises:
        ValueError: If the stream is not a decodable single-channel 16-bit PNG.
    """
    from PIL import Image

    try:
        with Image.open(source) as image:
            if image.format != "PNG" or image.mode not in _GRAY16_PNG_MODES:
                raise ValueError(
                    f"{label} must be a single-channel 16-bit grayscale PNG, got "
                    f"format {image.format} mode {image.mode}."
                )
            # Pillow decodes lazily, so corrupt pixel data fails only here; the cast
            # narrows the int32 mode "I" Pillow before 10.3 gives a 16-bit PNG.
            return np.ascontiguousarray(image, dtype=np.uint16)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"{label} is not a decodable PNG: {error}") from error


def _read_pfm(path: str | Path) -> npt.NDArray[np.float32]:
    """Read a single-channel PFM (`Pf`) as float32, top row first.

    PFM stores rows bottom to top, and the sign of its scale line gives the byte
    order (negative is little-endian). Middlebury, ETH3D and SceneFlow disparity use
    this format, with `+inf` for unknown disparity.

    Raises:
        ValueError: If the file is not a grayscale PFM, its header is incomplete or
            malformed, its width or height is not positive, its scale is zero or
            not finite, or it is truncated.
    """
    data = Path(path).read_bytes()
    header: list[bytes] = []
    position = 0
    # The header is four whitespace-separated tokens (magic, width, height, scale)
    # that may share or span lines, the scale usually on its own line; blank and
    # '#' comment lines are skipped. Pixel data starts after the scale's line.
    while len(header) < 4:
        end = data.find(b"\n", position)
        if end < 0:
            raise ValueError(f"{path} is not a PFM file: header is incomplete.")
        line = data[position:end].strip()
        position = end + 1
        if line and not line.startswith(b"#"):
            header.extend(line.split())
    magic, width_text, height_text, scale_text = header[:4]
    if magic == b"PF":
        raise ValueError(
            f"{path} is a colour PFM (PF); a depth map must be grayscale (Pf)."
        )
    if magic != b"Pf":
        raise ValueError(f"{path} is not a PFM file.")
    try:
        width, height, scale = int(width_text), int(height_text), float(scale_text)
    except ValueError as error:
        raise ValueError(f"{path} has a malformed PFM header: {error}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} has invalid PFM dimensions {width}x{height}.")
    if scale == 0 or not math.isfinite(scale):
        raise ValueError(f"{path} has an invalid PFM scale {scale}.")
    dtype = "<f4" if scale < 0 else ">f4"
    expected = width * height * 4
    payload = data[position : position + expected]
    if len(payload) != expected:
        raise ValueError(f"{path} is truncated: {len(payload)} of {expected} bytes.")
    values = np.frombuffer(payload, dtype=dtype).reshape(height, width)
    return np.ascontiguousarray(values[::-1], dtype=np.float32)
