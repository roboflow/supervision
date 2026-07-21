"""Private image-operation and image-I/O fallbacks."""

from __future__ import annotations

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
    source1: npt.NDArray[Any],
    alpha: float,
    source2: npt.NDArray[Any],
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
    if source1.shape != source2.shape:
        raise ValueError("addWeighted inputs must have equal shapes")
    result = _cast_array_like_opencv(
        source1.astype(np.float64) * alpha + source2.astype(np.float64) * beta + gamma,
        source1.dtype,
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


def _imread(filename: str, flags: int = _IMREAD_COLOR) -> npt.NDArray[Any] | None:
    """Read an image with Pillow while returning BGR or BGRA arrays."""
    from PIL import Image

    try:
        with Image.open(filename) as image:
            if flags == _IMREAD_UNCHANGED:
                if image.mode == "P":
                    converted = image.convert(
                        "RGBA" if "transparency" in image.info else "RGB"
                    )
                    values = np.asarray(converted)
                    converted.close()
                else:
                    values = np.asarray(image)
            elif image.mode in {"I", "I;16", "I;16B", "I;16L"}:
                values = np.asarray(image).astype(np.float64)
                values = np.clip(np.rint(values / 256), 0, 255).astype(np.uint8)
                if values.ndim == 2:
                    values = np.repeat(values[..., np.newaxis], 3, axis=2)
            else:
                values = np.asarray(image.convert("RGB"))
    except (FileNotFoundError, OSError):
        return None

    if values.ndim == 3 and values.shape[2] == 3:
        values = values[..., ::-1]
    elif values.ndim == 3 and values.shape[2] == 4:
        values = values[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(values)


def _imwrite(
    filename: str, image: npt.NDArray[Any], params: Sequence[int] | None = None
) -> bool:
    """Write a BGR or BGRA array with Pillow and return OpenCV's boolean status."""
    from PIL import Image

    del params
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[2] == 3:
        values = values[..., ::-1]
    elif values.ndim == 3 and values.shape[2] == 4:
        values = values[..., [2, 1, 0, 3]]
    try:
        Image.fromarray(np.ascontiguousarray(values)).save(filename)
    except (OSError, ValueError):
        return False
    return True
