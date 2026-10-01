"""Read and write the files of the shared depth format.

The format is the one supervision-js reads
(`packages/core/src/utils/depth-manifest.ts`): a snake_case `depth.json` manifest
next to 16-bit grayscale PNGs whose stored value divided by `storage.scale` is the
value in the map kind's unit, with 0 for no depth.
A clip manifest adds one PNG per frame and an optional 8-bit preview video.
"""

from __future__ import annotations

import math
import re
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

DEPTH_MANIFEST_SCHEMA = "supervision.depth-manifest"
DEPTH_MANIFEST_VERSION = 1
DEPTH_KIND_VALUES = ("disparity_px", "depth_m", "relative_inverse")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_FILTER_UP = 2
_PNG_COMPRESSION_LEVEL = 6
_UINT16_MAX = 65535

#: The preview code standing for the top of the range, per luma level.
PREVIEW_TOP_CODES = {"full": 255, "tv": 235}
#: The lowest reserved preview code per luma level: TV black (16) is no depth.
_PREVIEW_MIN_RESERVED_CODES = {"full": 0, "tv": 16}
_FRAME_INDEX_TOKEN = re.compile(r"\{index(?::0(\d{1,2}))?\}")


@dataclass(frozen=True)
class _DepthPreview:
    """The parsed `preview` block of a clip manifest."""

    file: str
    levels: str
    reserved_max: int
    range_px: tuple[float, float]
    codec: str | None = None


@dataclass(frozen=True)
class _DepthManifest:
    """A `depth.json` checked against every rule supervision-js enforces."""

    kind: str
    width: int
    height: int
    scale: float
    view: str | None = None
    camera: dict[str, float] | None = None
    display_range: tuple[float, float] | None = None
    image_file: str | None = None
    frame_count: int | None = None
    frame_pattern: str | None = None
    preview: _DepthPreview | None = None


def _fail(message: str) -> NoReturn:
    """Raise the manifest error format supervision-js uses, naming the wire field."""
    raise ValueError(f"depth.json: {message}")


def _is_number(value: Any) -> bool:
    """Report a JSON number; `bool` is excluded because JSON keeps it apart."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_integer(value: Any) -> bool:
    """Report a JSON number that is a whole number above zero, as `Number.isInteger`."""
    return (
        _is_number(value) and math.isfinite(value) and value > 0 and value == int(value)
    )


def _is_positive_number(value: Any) -> bool:
    """Report a finite JSON number above zero."""
    return _is_number(value) and math.isfinite(value) and value > 0


def _read_object(value: Any, path: str) -> dict[str, Any]:
    """Return `value` when it is a JSON object, else fail naming `path`."""
    if not isinstance(value, dict):
        _fail(f"{path} must be an object")
    return value


def _read_positive_integer(value: Any, path: str) -> int:
    """Return a positive integer field, else fail naming `path`."""
    if not _is_positive_integer(value):
        _fail(f"{path} must be a positive integer")
    return int(value)


def _read_positive_number(value: Any, path: str) -> float:
    """Return a finite positive number field, else fail naming `path`."""
    if not _is_positive_number(value):
        _fail(f"{path} must be a positive number")
    return float(value)


def _read_optional_finite(value: Any, path: str) -> float | None:
    """Return an optional finite number field, else fail naming `path`."""
    if value is None:
        return None
    if not (_is_number(value) and math.isfinite(value)):
        _fail(f"{path} must be a finite number")
    return float(value)


def _read_optional_string(value: Any, path: str) -> str | None:
    """Return an optional string field, else fail naming `path`."""
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(f"{path} must be a string")
    return value


def _read_file_name(value: Any, path: str) -> str:
    """Return a non-empty file name field, else fail naming `path`."""
    if not isinstance(value, str) or not value:
        _fail(f"{path} must be a non-empty file name")
    return value


def _read_optional_range(value: Any, path: str) -> tuple[float, float] | None:
    """Return an optional `[low, high]` pair of finite numbers with `low < high`."""
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_number(bound) and math.isfinite(bound) for bound in value)
    ):
        _fail(f"{path} must be two finite numbers [low, high]")
    low, high = float(value[0]), float(value[1])
    if not low < high:
        _fail(f"{path} must have low < high")
    return low, high


def _read_frame_pattern(value: Any, path: str) -> str:
    """Return a frame file pattern holding `{index}` or `{index:0N}`."""
    pattern = _read_file_name(value, path)
    if _FRAME_INDEX_TOKEN.search(pattern) is None:
        _fail(f"{path} must contain {{index}} or a padded {{index:06}}")
    return pattern


def _read_camera(value: Any) -> dict[str, float]:
    """Return the camera block with its required and optional fields checked."""
    camera = _read_object(value, "camera")
    parsed = {
        "fx_px": _read_positive_number(camera.get("fx_px"), "camera.fx_px"),
        "baseline_m": _read_positive_number(
            camera.get("baseline_m"), "camera.baseline_m"
        ),
    }
    for name in ("doffs_px", "cx_px", "cy_px"):
        optional = _read_optional_finite(camera.get(name), f"camera.{name}")
        if optional is not None:
            parsed[name] = optional
    return parsed


def _read_display_range(root: dict[str, Any], kind: str) -> tuple[float, float] | None:
    """Return the display range from `display_range` or its disparity alias."""
    in_kind_unit = _read_optional_range(root.get("display_range"), "display_range")
    in_pixels = _read_optional_range(root.get("display_range_px"), "display_range_px")
    if in_pixels is not None and kind != "disparity_px":
        _fail("display_range_px is only valid for kind disparity_px; use display_range")
    if in_kind_unit is not None and in_pixels is not None and in_kind_unit != in_pixels:
        _fail("display_range and display_range_px disagree")
    return in_kind_unit if in_kind_unit is not None else in_pixels


def _read_preview(value: Any) -> _DepthPreview:
    """Return the preview block; a preview without `levels` is full range."""
    preview = _read_object(value, "preview")
    file = _read_file_name(preview.get("file"), "preview.file")
    codec = _read_optional_string(preview.get("codec"), "preview.codec")
    levels = preview.get("levels")
    levels = "full" if levels is None else levels
    if not isinstance(levels, str) or levels not in PREVIEW_TOP_CODES:
        _fail(f"preview.levels must be one of {', '.join(PREVIEW_TOP_CODES)}")
    reserved_max = preview.get("reserved_max")
    lowest = _PREVIEW_MIN_RESERVED_CODES[levels]
    highest = PREVIEW_TOP_CODES[levels] - 2
    if (
        isinstance(reserved_max, bool)
        or not isinstance(reserved_max, (int, float))
        or not math.isfinite(reserved_max)
        or reserved_max != int(reserved_max)
        or not lowest <= reserved_max <= highest
    ):
        _fail(
            f"preview.reserved_max must be an integer from {lowest} to {highest} "
            f"at {levels} levels"
        )
    range_px = _read_optional_range(preview.get("range_px"), "preview.range_px")
    if range_px is None:
        _fail("preview.range_px is required")
    return _DepthPreview(
        file=file,
        levels=levels,
        reserved_max=int(reserved_max),
        range_px=range_px,
        codec=codec,
    )


def parse_manifest(data: Any) -> _DepthManifest:
    """Check a decoded `depth.json` with supervision-js's rules and return its fields.

    Unknown fields are ignored so newer producers stay readable. Errors are
    `ValueError` with the message format `depth.json: storage.scale must be a positive
    number`, naming the offending wire field.
    """
    root = _read_object(data, "the manifest")
    if root.get("schema") != DEPTH_MANIFEST_SCHEMA:
        _fail(f'schema must be "{DEPTH_MANIFEST_SCHEMA}"')
    version = root.get("version")
    if not (_is_number(version) and version == DEPTH_MANIFEST_VERSION):
        _fail(f"version {version} is not supported; expected 1")
    kind = root.get("kind")
    if not isinstance(kind, str) or kind not in DEPTH_KIND_VALUES:
        _fail(f"kind must be one of {', '.join(DEPTH_KIND_VALUES)}")
    width = _read_positive_integer(root.get("width"), "width")
    height = _read_positive_integer(root.get("height"), "height")
    view = _read_optional_string(root.get("view"), "view")

    storage = _read_object(root.get("storage"), "storage")
    if storage.get("format") != "png16":
        _fail('storage.format must be "png16"')
    scale = _read_positive_number(storage.get("scale"), "storage.scale")
    no_depth = storage.get("no_depth")
    if not (_is_number(no_depth) and no_depth == 0):
        _fail("storage.no_depth must be 0")

    camera = None if root.get("camera") is None else _read_camera(root["camera"])
    display_range = _read_display_range(root, kind)

    has_image = root.get("image") is not None
    has_frames = root.get("frames") is not None
    if has_image == has_frames:
        _fail("exactly one of image or frames must be present")

    image_file = None
    frame_count = None
    frame_pattern = None
    if has_image:
        image = _read_object(root["image"], "image")
        image_file = _read_file_name(image.get("file"), "image.file")
    else:
        frames = _read_object(root["frames"], "frames")
        frame_count = _read_positive_integer(frames.get("count"), "frames.count")
        frame_pattern = _read_frame_pattern(frames.get("exact"), "frames.exact")
        _check_frame_times(frames.get("times_s"), frame_count)

    preview = None
    if root.get("preview") is not None:
        if not has_frames:
            _fail("preview is only valid next to frames")
        if kind != "disparity_px":
            _fail("preview is only supported for kind disparity_px")
        preview = _read_preview(root["preview"])

    return _DepthManifest(
        kind=kind,
        width=width,
        height=height,
        scale=scale,
        view=view,
        camera=camera,
        display_range=display_range,
        image_file=image_file,
        frame_count=frame_count,
        frame_pattern=frame_pattern,
        preview=preview,
    )


def _check_frame_times(value: Any, count: int) -> None:
    """Check optional `frames.times_s`: `count` finite seconds, strictly increasing."""
    if value is None:
        return
    if not isinstance(value, list) or len(value) != count:
        _fail(f"frames.times_s must be an array of frames.count ({count}) times")
    previous = None
    for index, time in enumerate(value):
        if not (_is_number(time) and math.isfinite(time) and time >= 0):
            _fail(f"frames.times_s[{index}] must be a finite number of seconds >= 0")
        if previous is not None and time <= previous:
            _fail(f"frames.times_s must strictly increase at index {index}")
        previous = time


def resolve_frame_file(pattern: str, index: int) -> str:
    """Expand `{index}` to the plain number and `{index:06}` to six padded digits.

    Examples:
        ```pycon
        >>> from supervision.depth.manifest import resolve_frame_file
        >>> resolve_frame_file("exact/{index:06}.png", 42)
        'exact/000042.png'

        ```
    """
    if index < 0:
        raise ValueError(
            f"Depth frame index must be a non-negative integer, got {index}."
        )

    def expand(match: re.Match[str]) -> str:
        """Pad the index to the token's width, or write it plainly."""
        width = match.group(1)
        return str(index) if width is None else str(index).zfill(int(width))

    return _FRAME_INDEX_TOKEN.sub(expand, pattern)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    """Frame one PNG chunk: length, type, data and CRC-32 of type plus data."""
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def encode_png16(codes: npt.NDArray[np.uint16]) -> bytes:
    """Encode a 2D `uint16` array as a 16-bit grayscale PNG with the Up filter.

    Every row uses PNG filter type 2 (Up), which a browser's JavaScript decoder undoes
    with one addition per byte; Pillow picks mostly filter type 4 rows for depth maps,
    which decode 1.7 to 2.3 times slower (supervision-js depth benchmark). The file is a
    standard PNG: colour type 0, bit depth 16, not interlaced.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.depth.manifest import encode_png16
        >>> data = encode_png16(np.array([[0, 1], [256, 65535]], dtype=np.uint16))
        >>> data[:8] == b"\\x89PNG\\r\\n\\x1a\\n"
        True

        ```
    """
    if codes.dtype != np.uint16 or codes.ndim != 2:
        raise ValueError(
            f"PNG16 needs a 2D uint16 array, got {codes.dtype} {codes.shape}."
        )
    height, width = codes.shape
    raw = (
        np.ascontiguousarray(codes, dtype=">u2")
        .view(np.uint8)
        .reshape(height, width * 2)
    )
    # uint8 arithmetic wraps modulo 256, which is exactly the PNG Up filter.
    filtered = raw.copy()
    filtered[1:] -= raw[:-1]
    rows = np.hstack([np.full((height, 1), _PNG_FILTER_UP, dtype=np.uint8), filtered])
    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    compressed = zlib.compress(rows.tobytes(), _PNG_COMPRESSION_LEVEL)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def read_png16(source: str | Path | bytes) -> npt.NDArray[np.uint16]:
    """Read a 16-bit grayscale PNG, written with any row filter, as `uint16`.

    Raises:
        ValueError: If the file is not a single-channel 16-bit PNG.
    """
    from PIL import Image

    opened = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(opened) as image:
        if image.format != "PNG" or image.mode not in {"I", "I;16", "I;16B", "I;16L"}:
            raise ValueError(
                "Depth PNG must be a single-channel 16-bit grayscale PNG, got "
                f"format {image.format} mode {image.mode}."
            )
        values = np.asarray(image)
    if values.dtype != np.uint16:
        if values.min(initial=0) < 0 or values.max(initial=0) > _UINT16_MAX:
            raise ValueError("Depth PNG values must fit in 16 bits.")
        values = values.astype(np.uint16)
    return np.ascontiguousarray(values)


def read_pfm(path: str | Path) -> npt.NDArray[np.float32]:
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


def power_of_two_scale(max_value: float) -> float:
    """Return the largest power of two keeping `max_value * scale` within 65535.

    A power of two keeps steps exact in binary (1/1024 px, for example), the
    convention the Spring fixture and KITTI's 256 follow.

    Examples:
        ```pycon
        >>> from supervision.depth.manifest import power_of_two_scale
        >>> power_of_two_scale(63.0)
        1024.0
        >>> power_of_two_scale(255.9)
        256.0

        ```
    """
    if not (math.isfinite(max_value) and max_value > 0):
        return 1.0
    return float(2.0 ** math.floor(math.log2(_UINT16_MAX / max_value)))


def encode_codes(
    values: npt.NDArray[np.floating],
    valid: npt.NDArray[np.bool_],
    scale: float,
) -> npt.NDArray[np.uint16]:
    """Quantise float values to stored codes, 0 for no depth.

    A valid value that would round to 0 is written as 1, one step above no depth,
    as Ultralytics' `save_depth_png` does.

    Raises:
        ValueError: If a valid value exceeds `65535 / scale`.
    """
    codes = np.zeros(values.shape, dtype=np.uint16)
    if not valid.any():
        return codes
    scaled = np.rint(values[valid].astype(np.float64) * scale)
    largest = float(scaled.max())
    if largest > _UINT16_MAX:
        raise ValueError(
            f"Depth value {largest / scale:g} does not fit a 16-bit PNG at "
            f"scale={scale:g}; the largest storable value is "
            f"{_UINT16_MAX / scale:g}. Pass a smaller scale, for example "
            f"{power_of_two_scale(largest / scale):g}."
        )
    codes[valid] = np.maximum(scaled, 1).astype(np.uint16)
    return codes


def json_number(value: float) -> int | float:
    """Write whole numbers as JSON integers (scale 1024, not 1024.0)."""
    return int(value) if float(value).is_integer() else float(value)


def avc_codec_string(path: str | Path) -> str | None:
    """Read the RFC 6381 codec string `avc1.PPCCLL` from an MP4's `avcC` box."""
    data = Path(path).read_bytes()
    at = data.find(b"avcC")
    if at < 0:
        return None
    start = at + 5
    return "avc1." + data[start : start + 3].hex()
