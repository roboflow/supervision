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


def _flip(src: npt.NDArray[Any], flipCode: int) -> npt.NDArray[Any]:
    """Flip an image vertically, horizontally, or along both axes.

    Flip direction follows the sign of `flipCode`, as in `cv2.flip`: `0` flips
    vertically, any positive code flips horizontally, and any negative code flips along
    both axes. The camelCase `flipCode` mirrors `cv2.flip(src, flipCode)` so keyword
    calls bind on both backends; keep it when editing.
    """
    if flipCode == 0:
        axes: tuple[int, ...] = (0,)
    elif flipCode > 0:
        axes = (1,)
    else:
        axes = (0, 1)
    return np.ascontiguousarray(np.flip(src, axis=axes))


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
    """Resize with OpenCV-compatible nearest or linear sampling."""
    source_height, source_width = src.shape[:2]
    width, height = dsize if dsize is not None else (0, 0)
    # Like OpenCV, when the size comes from `fx`/`fy`, pixels are mapped with
    # those factors rather than with the ratio of the rounded sizes.
    scale_from_factors = width == 0 or height == 0
    if scale_from_factors:
        width = round(source_width * fx)
        height = round(source_height * fy)
        # OpenCV copies the image unchanged when the size stays the same.
        scale_from_factors = (width, height) != (source_width, source_height)
    if min(width, height, source_width, source_height) <= 0:
        raise ValueError("Resize dimensions must be positive")
    if scale_from_factors:
        scale_x, scale_y = 1 / fx, 1 / fy
    else:
        # OpenCV steps through the source with the reciprocal of the size ratio
        # (`ifx = 1. / (dsize.width / ssize.width)`), whose rounding can put a
        # whole-number position such as 17 * 6 / 34 just below 3.0, so the
        # exact integer ratio would pick a different pixel there.
        scale_x = 1 / (width / source_width)
        scale_y = 1 / (height / source_height)

    if interpolation == _INTER_NEAREST:
        y_indices = np.floor(np.arange(height) * scale_y).astype(np.int64)
        x_indices = np.floor(np.arange(width) * scale_x).astype(np.int64)
        y_indices = np.minimum(y_indices, source_height - 1)
        x_indices = np.minimum(x_indices, source_width - 1)
        return np.ascontiguousarray(src[y_indices[:, np.newaxis], x_indices])

    if interpolation != _INTER_LINEAR:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    if src.dtype == np.uint8:
        return _resize_linear_uint8(src, width, height, scale_x, scale_y)

    y = (np.arange(height) + 0.5) * scale_y - 0.5
    x = (np.arange(width) + 0.5) * scale_x - 0.5
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


# OpenCV's uint8 `INTER_LINEAR` resize uses 11-bit fixed-point weights.
_INTER_RESIZE_COEF_BITS = 11
_INTER_RESIZE_COEF_SCALE = 1 << _INTER_RESIZE_COEF_BITS


def _linear_resize_taps(
    size: int, source_size: int, scale: float, clamp_weights: bool
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Return OpenCV's two source taps and fixed-point weights for one axis.

    Positions are computed in `float32` like `resizeGeneric_`. Along columns OpenCV
    snaps samples left of the first or right of the last pixel to that pixel with a
    zero fraction; along rows it only clamps the row index and keeps the fraction, so
    the weights of an edge row come from the unclamped position.
    """
    position = ((np.arange(size) + 0.5) * scale - 0.5).astype(np.float32)
    first = np.floor(position).astype(np.intp)
    fraction = (position - first).astype(np.float32)
    if clamp_weights:
        before = first < 0
        fraction[before] = 0
        first[before] = 0
        after = first >= source_size - 1
        fraction[after] = 0
        first[after] = source_size - 1
    second = np.clip(first + 1, 0, source_size - 1)
    first = np.clip(first, 0, source_size - 1)
    weight_second = np.rint(fraction * _INTER_RESIZE_COEF_SCALE).astype(np.int32)
    weight_first = np.rint(
        (np.float32(1) - fraction) * _INTER_RESIZE_COEF_SCALE
    ).astype(np.int32)
    return first, second, weight_first, weight_second


def _resize_linear_uint8(
    src: npt.NDArray[np.uint8],
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> npt.NDArray[np.uint8]:
    """Resize uint8 images bit-exactly like OpenCV's fixed-point `INTER_LINEAR`.

    Each row is first interpolated horizontally into `int32` sums scaled by
    `2**11`. The vertical pass then mirrors OpenCV's SIMD `VResizeLinearVec_32s8u`:
    each sum is shifted right by 4, multiplied by its 11-bit row weight keeping the
    high 16 bits, and the two products are added and rounded with `(sum + 2) >> 2`.
    """
    source_height, source_width = src.shape[:2]
    x0, x1, wx0, wx1 = _linear_resize_taps(
        width, source_width, scale_x, clamp_weights=True
    )
    y0, y1, wy0, wy1 = _linear_resize_taps(
        height, source_height, scale_y, clamp_weights=False
    )
    # Work on `(rows, width * channels)` so every weight broadcasts along the
    # contiguous axis; each channel of a column shares that column's taps.
    channels = 1 if src.ndim == 2 else src.shape[2]
    channel_offsets = np.arange(channels)
    columns0 = (x0[:, np.newaxis] * channels + channel_offsets).ravel()
    columns1 = (x1[:, np.newaxis] * channels + channel_offsets).ravel()
    weights0 = np.repeat(wx0, channels)
    weights1 = np.repeat(wx1, channels)

    # Interpolate only the source rows the vertical pass reads.
    used_rows, row_index = np.unique(np.concatenate((y0, y1)), return_inverse=True)
    y0, y1 = row_index[:height], row_index[height:]
    rows = src.reshape(source_height, source_width * channels)[used_rows]
    horizontal = np.take(rows, columns0, axis=1).astype(np.int32)
    horizontal *= weights0
    right = np.take(rows, columns1, axis=1).astype(np.int32)
    right *= weights1
    horizontal += right
    horizontal >>= 4

    resized = np.take(horizontal, y0, axis=0)
    resized *= wy0[:, np.newaxis]
    resized >>= 16
    bottom = np.take(horizontal, y1, axis=0)
    bottom *= wy1[:, np.newaxis]
    bottom >>= 16
    resized += bottom
    resized += 2
    resized >>= 2
    np.clip(resized, 0, 255, out=resized)
    return resized.astype(np.uint8).reshape((height, width, *src.shape[2:]))


_EXIF_ORIENTATION_TAG = 0x0112


def _exif_oriented(image: Any) -> Any:
    """Rotate and flip a Pillow image by its EXIF orientation tag, as OpenCV does.

    `cv2.imread` and `cv2.imdecode` apply the tag on every read except
    `IMREAD_UNCHANGED`. An image without the tag, or with the identity orientation `1`,
    is returned as it is rather than copied.
    """
    from PIL import ImageOps

    if image.getexif().get(_EXIF_ORIENTATION_TAG, 1) == 1:
        return image
    return ImageOps.exif_transpose(image)


def _opencv_unchanged_mode(image: Any) -> str | None:
    """Return the Pillow mode whose pixels match `cv2.imread(IMREAD_UNCHANGED)`.

    OpenCV keeps an image's bit depth and alpha, but never returns a two-channel or
    boolean array the way Pillow's own modes do: it expands palettes to BGR, whether
    or not they carry an alpha channel of their own, adds an alpha channel for
    grayscale with alpha and for a palette or RGB image with a transparent color, and
    reads 1-bit images as 8-bit `0` and `255`. It also never returns CMYK ink values,
    whose black channel would pass for alpha: a CMYK JPEG is converted to BGR, and a
    CMYK TIFF to BGRA with an opaque alpha channel. `None` means the image's own mode
    already matches.
    """
    has_transparency = "transparency" in image.info
    if image.mode in {"P", "PA"}:
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


def _bgr_to_pil_values(
    image: npt.NDArray[Any], image_format: str | None = None
) -> npt.NDArray[Any]:
    """Normalize channel layouts and order for a Pillow image encoder.

    Pillow expects grayscale as a 2D array. JPEG has no alpha channel, so discard alpha
    without compositing, as OpenCV does; other formats retain it.
    """
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[..., 0]
    if values.ndim == 3 and values.shape[2] == 3:
        values = values[..., ::-1]
    elif values.ndim == 3 and values.shape[2] == 4:
        channels = [2, 1, 0] if image_format == "JPEG" else [2, 1, 0, 3]
        values = values[..., channels]
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
    :func:`~supervision._cv2._image._opencv_default_save_options` returns, not
    with the quality or compression the caller asked for.
    """
    from PIL import Image

    del params
    extension = os.path.splitext(filename)[1].lower()
    image_format = Image.registered_extensions().get(extension)
    save_options = _opencv_default_save_options(image_format)
    try:
        Image.fromarray(_bgr_to_pil_values(image, image_format)).save(
            filename, **save_options
        )
    except (OSError, ValueError):
        return False
    return True


def _imencode(
    ext: str, image: npt.NDArray[Any], params: Sequence[int] | None = None
) -> tuple[bool, npt.NDArray[np.uint8] | None]:
    """Encode a BGR or BGRA array in memory, mirroring `cv2.imencode`'s return.

    `params` is accepted for compatibility with `cv2.imencode`'s signature and is
    ignored: the image is encoded with the OpenCV defaults that
    :func:`~supervision._cv2._image._opencv_default_save_options` returns, not
    with the quality or compression the caller asked for.
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
        Image.fromarray(_bgr_to_pil_values(image, image_format)).save(
            buffer, format=image_format, **save_options
        )
    except (KeyError, OSError, ValueError):
        return False, None
    return True, np.frombuffer(buffer.getvalue(), dtype=np.uint8)
