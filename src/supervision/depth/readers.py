"""Read depth maps from the files stereo and depth datasets ship."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

_UINT16_MAX = 65535


def _read_png16(path: str | Path) -> npt.NDArray[np.uint16]:
    """Read a single-channel 16-bit grayscale PNG as `uint16`.

    Raises:
        ValueError: If the file is not a single-channel 16-bit PNG.
    """
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in {"I", "I;16", "I;16B", "I;16L"}:
            raise ValueError(
                "Depth PNG must be a single-channel 16-bit grayscale PNG, got "
                f"format {image.format} mode {image.mode}."
            )
        # Pillow before 10.3 opens a 16-bit PNG as int32 mode "I".
        return np.ascontiguousarray(image, dtype=np.uint16)


def _read_pfm(path: str | Path) -> npt.NDArray[np.float32]:
    """Read a single-channel PFM (`Pf`) as float32, top row first.

    PFM stores rows bottom to top, and the sign of its scale line gives the byte
    order (negative is little-endian). Middlebury, ETH3D and SceneFlow disparity use
    this format, with `+inf` for unknown disparity.

    Raises:
        ValueError: If the file is not a grayscale PFM or is truncated.
    """
    data = Path(path).read_bytes()
    header: list[bytes] = []
    position = 0
    while len(header) < 3:
        end = data.find(b"\n", position)
        if end < 0:
            raise ValueError(f"{path} is not a PFM file: header is incomplete.")
        line = data[position:end].strip()
        position = end + 1
        if line and not line.startswith(b"#"):
            header.extend(line.split())
    magic, width_text, height_text = header[0], header[1], header[2]
    if magic == b"PF":
        raise ValueError(
            f"{path} is a colour PFM (PF); a depth map must be grayscale (Pf)."
        )
    if magic != b"Pf":
        raise ValueError(f"{path} is not a PFM file.")
    width, height = int(width_text), int(height_text)
    if len(header) < 4:
        end = data.find(b"\n", position)
        header.append(data[position:end].strip())
        position = end + 1
    scale = float(header[3])
    dtype = "<f4" if scale < 0 else ">f4"
    expected = width * height * 4
    payload = data[position : position + expected]
    if len(payload) != expected:
        raise ValueError(f"{path} is truncated: {len(payload)} of {expected} bytes.")
    values = np.frombuffer(payload, dtype=dtype).reshape(height, width)
    return np.ascontiguousarray(values[::-1], dtype=np.float32)
