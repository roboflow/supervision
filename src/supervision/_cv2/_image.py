"""Private image-operation and image-I/O fallbacks."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision._cv2._common import _cast_array_like_opencv
from supervision._cv2.constants import (
    _BORDER_CONSTANT,
    _IMREAD_COLOR,
    _IMREAD_UNCHANGED,
    _INTER_LINEAR,
    _INTER_NEAREST,
)


def _flip(image: npt.NDArray[Any], flip_code: int) -> npt.NDArray[Any]:
    """Flip an image vertically, horizontally, or along both axes."""
    if flip_code == 0:
        axes: tuple[int, ...] = (0,)
    elif flip_code == 1:
        axes = (1,)
    elif flip_code == -1:
        axes = (0, 1)
    else:
        raise ValueError(f"Unsupported flip code: {flip_code}")
    return np.ascontiguousarray(np.flip(image, axis=axes))


def _copy_make_border(
    image: npt.NDArray[Any],
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int,
    value: int | float | Sequence[int | float] = 0,
) -> npt.NDArray[Any]:
    """Add a constant border around an image."""
    if border_type != _BORDER_CONSTANT:
        raise ValueError("Only BORDER_CONSTANT is supported by the fallback")
    if min(top, bottom, left, right) < 0:
        raise ValueError("Border sizes must be non-negative")

    height, width = image.shape[:2]
    shape = (height + top + bottom, width + left + right, *image.shape[2:])

    # OpenCV's Scalar(v) fills only channel 0 and zero-pads the rest for
    # multichannel images — a bare scalar is treated the same as a
    # length-1 sequence, not broadcast to every channel.
    sequence_value = value if isinstance(value, Sequence) else (value,)
    values = np.asarray(sequence_value, dtype=image.dtype).reshape(-1)
    if image.ndim == 2:
        fill_value: Any = values[0] if values.size else 0
    else:
        fill = np.zeros(image.shape[2], dtype=image.dtype)
        fill[: min(values.size, image.shape[2])] = values[: image.shape[2]]
        fill_value = fill.reshape((1, 1, -1))

    result = np.full(shape, fill_value, dtype=image.dtype)
    result[top : top + height, left : left + width] = image
    return result


def _add_weighted(
    src1: npt.NDArray[Any],
    alpha: float,
    src2: npt.NDArray[Any],
    beta: float,
    gamma: float,
    dst: npt.NDArray[Any] | None = None,
    dtype: int | None = None,
) -> npt.NDArray[Any]:
    """Blend two arrays with OpenCV-compatible saturation and optional mutation."""
    if dtype is not None and dtype != -1:
        raise ValueError(
            "addWeighted fallback only supports the default output depth; "
            f"unsupported dtype: {dtype}"
        )
    if src1.shape != src2.shape:
        raise ValueError("addWeighted inputs must have equal shapes")
    result = _cast_array_like_opencv(
        src1.astype(np.float64) * alpha + src2.astype(np.float64) * beta + gamma,
        src1.dtype,
    )
    if dst is not None:
        dst[...] = result
        return dst
    return result


def _convert_scale_abs(
    image: npt.NDArray[Any], alpha: float = 1, beta: float = 0
) -> npt.NDArray[np.uint8]:
    """Scale, offset, take the absolute value, and saturate to uint8."""
    values = np.abs(image.astype(np.float64) * alpha + beta)
    return _cast_array_like_opencv(values, np.dtype(np.uint8))


def _mean(
    image: npt.NDArray[Any], mask: npt.NDArray[Any] | None = None
) -> tuple[float, float, float, float]:
    """Return per-channel means using OpenCV's four-value result contract."""
    if mask is None:
        selected = (
            image.reshape(-1, 1)
            if image.ndim == 2
            else image.reshape(-1, image.shape[2])
        )
    else:
        if mask.shape != image.shape[:2]:
            raise ValueError("Mean mask must match the image height and width")
        selected = image[mask != 0]
        if image.ndim == 2:
            selected = selected.reshape(-1, 1)
    if selected.size == 0:
        means = np.zeros(4, dtype=np.float64)
    else:
        means = np.zeros(4, dtype=np.float64)
        means[: selected.shape[1]] = np.mean(selected, axis=0)
    return cast(
        tuple[float, float, float, float],
        tuple(float(value) for value in means),
    )


def _resize(
    src: npt.NDArray[Any],
    dsize: tuple[int, int] | None,
    fx: float = 0,
    fy: float = 0,
    interpolation: int = _INTER_LINEAR,
) -> npt.NDArray[Any]:
    """Resize with exact nearest or OpenCV-compatible linear sampling."""
    source_height, source_width = src.shape[:2]
    width, height = dsize if dsize is not None else (0, 0)
    if width == 0 or height == 0:
        width = round(source_width * fx)
        height = round(source_height * fy)
    if min(width, height, source_width, source_height) <= 0:
        raise ValueError("Resize dimensions must be positive")

    if interpolation == _INTER_NEAREST:
        y_indices = np.minimum(
            (np.arange(height) * source_height // height), source_height - 1
        )
        x_indices = np.minimum(
            (np.arange(width) * source_width // width), source_width - 1
        )
        return np.ascontiguousarray(src[y_indices[:, np.newaxis], x_indices])

    if interpolation != _INTER_LINEAR:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    if src.dtype == np.uint8 and (
        src.ndim == 2 or (src.ndim == 3 and src.shape[2] == 3)
    ):
        from PIL import Image

        size = (width, height)
        image = Image.fromarray(src)
        if width >= source_width and height >= source_height:
            resized = image.resize(size, resample=Image.Resampling.BILINEAR)
        else:
            # Affine sampling keeps Pillow from widening its bilinear kernel
            # during reduction and maps pixel centers like INTER_LINEAR.
            resized = image.transform(
                size,
                Image.Transform.AFFINE,
                (source_width / width, 0, 0, 0, source_height / height, 0),
                resample=Image.Resampling.BILINEAR,
            )
        return np.ascontiguousarray(np.asarray(resized))

    y = (np.arange(height) + 0.5) * source_height / height - 0.5
    x = (np.arange(width) + 0.5) * source_width / width - 0.5
    y_floor = np.floor(y).astype(np.int64)
    x_floor = np.floor(x).astype(np.int64)
    y0 = np.clip(y_floor, 0, source_height - 1)
    y1 = np.clip(y_floor + 1, 0, source_height - 1)
    x0 = np.clip(x_floor, 0, source_width - 1)
    x1 = np.clip(x_floor + 1, 0, source_width - 1)
    wy = y - y_floor
    wx = x - x_floor

    source = src.astype(np.float64)
    top_left = source[y0[:, np.newaxis], x0]
    top_right = source[y0[:, np.newaxis], x1]
    bottom_left = source[y1[:, np.newaxis], x0]
    bottom_right = source[y1[:, np.newaxis], x1]
    if src.ndim == 3:
        wy = wy[:, np.newaxis, np.newaxis]
        wx = wx[np.newaxis, :, np.newaxis]
    else:
        wy = wy[:, np.newaxis]
        wx = wx[np.newaxis, :]
    resized = (
        top_left * (1 - wx) * (1 - wy)
        + top_right * wx * (1 - wy)
        + bottom_left * (1 - wx) * wy
        + bottom_right * wx * wy
    )
    return np.ascontiguousarray(_cast_array_like_opencv(resized, src.dtype))


_EXIF_ORIENTATION_TAG = 0x0112


def _exif_oriented(image: Any) -> Any:
    """Rotate and flip a Pillow image by its EXIF orientation tag, as OpenCV does.

    `cv2.imread` and `cv2.imdecode` apply the tag on every read except
    `IMREAD_UNCHANGED`. An image without the tag, or with the identity orientation
    `1`, is returned as it is rather than copied.
    """
    from PIL import ImageOps

    if image.getexif().get(_EXIF_ORIENTATION_TAG, 1) == 1:
        return image
    return ImageOps.exif_transpose(image)


def _opencv_unchanged_mode(image: Any) -> str | None:
    """Return the Pillow mode whose pixels match `cv2.imread(IMREAD_UNCHANGED)`.

    OpenCV keeps an image's bit depth and alpha, but never returns a two-channel or
    boolean array the way Pillow's own modes do: it expands palettes to BGR, adds an
    alpha channel for grayscale with alpha and for a palette or RGB image with a
    transparent color, and reads 1-bit images as 8-bit `0` and `255`. It also never
    returns CMYK ink values, whose black channel would pass for alpha: a CMYK JPEG
    is converted to BGR, and a CMYK TIFF to BGRA with an opaque alpha channel. `None`
    means the image's own mode already matches.
    """
    has_transparency = "transparency" in image.info
    if image.mode == "P":
        return "RGBA" if has_transparency else "RGB"
    if image.mode == "LA" or (image.mode == "RGB" and has_transparency):
        return "RGBA"
    if image.mode == "1":
        return "L"
    if image.mode == "CMYK":
        return "RGBA" if image.format == "TIFF" else "RGB"
    return None


def _read_pil_source(source: Any, flags: int) -> npt.NDArray[Any] | None:
    """Decode any source Pillow can open into BGR or BGRA arrays."""
    from PIL import Image

    try:
        with Image.open(source) as image:
            if flags == _IMREAD_UNCHANGED:
                unchanged_mode = _opencv_unchanged_mode(image)
                if unchanged_mode is not None:
                    converted = image.convert(unchanged_mode)
                    values = np.asarray(converted)
                    converted.close()
                else:
                    values = np.asarray(image)
            elif image.mode in {"I", "I;16", "I;16B", "I;16L"}:
                values = np.asarray(_exif_oriented(image)).astype(np.float64)
                values = np.clip(np.rint(values / 256), 0, 255).astype(np.uint8)
                if values.ndim == 2:
                    values = np.repeat(values[..., np.newaxis], 3, axis=2)
            else:
                values = np.asarray(_exif_oriented(image).convert("RGB"))
    except (FileNotFoundError, OSError, ValueError):
        return None

    if values.ndim == 3 and values.shape[2] == 3:
        values = values[..., ::-1]
    elif values.ndim == 3 and values.shape[2] == 4:
        values = values[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(values)


def _imread(filename: str, flags: int = _IMREAD_COLOR) -> npt.NDArray[Any] | None:
    """Read an image with Pillow while returning BGR or BGRA arrays."""
    return _read_pil_source(filename, flags)


def _imdecode(
    buf: npt.NDArray[Any], flags: int = _IMREAD_COLOR
) -> npt.NDArray[Any] | None:
    """Decode in-memory encoded image bytes, returning BGR or BGRA arrays."""
    import io

    data = np.asarray(buf, dtype=np.uint8).tobytes()
    return _read_pil_source(io.BytesIO(data), flags)


def _bgr_to_pil_values(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Reorder BGR or BGRA channels into the RGB order Pillow expects."""
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[2] == 3:
        values = values[..., ::-1]
    elif values.ndim == 3 and values.shape[2] == 4:
        values = values[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(values)


def _opencv_default_save_options(image_format: str | None) -> dict[str, Any]:
    """Return the Pillow save options that encode like OpenCV's default writers.

    Given no parameters, `cv2.imwrite` and `cv2.imencode` write JPEG at quality 95 and
    WebP losslessly, while Pillow's defaults are JPEG at quality 75 and lossy WebP at
    quality 80. Formats such as PNG, BMP and TIFF are lossless in both libraries.
    """
    if image_format == "JPEG":
        return {"quality": 95}
    if image_format == "WEBP":
        return {"lossless": True}
    return {}


def _imwrite(
    filename: str, image: npt.NDArray[Any], params: Sequence[int] | None = None
) -> bool:
    """Write a BGR or BGRA array with Pillow and return OpenCV's boolean status.

    `params` is accepted for compatibility with `cv2.imwrite`'s signature and is
    ignored: the file is encoded with the OpenCV defaults that
    `_opencv_default_save_options` returns, not with the quality or compression the
    caller asked for.
    """
    from PIL import Image

    del params
    extension = os.path.splitext(filename)[1].lower()
    image_format = Image.registered_extensions().get(extension)
    save_options = _opencv_default_save_options(image_format)
    try:
        Image.fromarray(_bgr_to_pil_values(image)).save(filename, **save_options)
    except (OSError, ValueError):
        return False
    return True


def _imencode(
    ext: str, image: npt.NDArray[Any], params: Sequence[int] | None = None
) -> tuple[bool, npt.NDArray[np.uint8] | None]:
    """Encode a BGR or BGRA array in memory, mirroring `cv2.imencode`'s return.

    `params` is accepted for compatibility with `cv2.imencode`'s signature and is
    ignored: the image is encoded with the OpenCV defaults that
    `_opencv_default_save_options` returns, not with the quality or compression the
    caller asked for.
    """
    import io

    from PIL import Image

    del params
    # A suffix is not its codec's name, so resolve it through the same registry
    # `_imwrite` and `Image.save` consult for a file path. An unregistered suffix
    # resolves to None, which `Image.save` rejects for a buffer with no file name.
    extension = f".{ext.lstrip('.').lower()}"
    image_format = Image.registered_extensions().get(extension)
    save_options = _opencv_default_save_options(image_format)
    buffer = io.BytesIO()
    try:
        Image.fromarray(_bgr_to_pil_values(image)).save(
            buffer, format=image_format, **save_options
        )
    except (KeyError, OSError, ValueError):
        return False, None
    return True, np.frombuffer(buffer.getvalue(), dtype=np.uint8)
