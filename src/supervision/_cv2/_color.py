"""Private color and channel-operation fallbacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision._cv2._common import _cast_array_like_opencv
from supervision._cv2.constants import (
    _COLOR_BGR2GRAY,
    _COLOR_BGR2RGB,
    _COLOR_GRAY2BGR,
    _COLOR_HSV2BGR,
    _COLOR_RGB2BGR,
)


def _cvt_color(image: npt.NDArray[Any], code: int) -> npt.NDArray[Any]:
    """Convert the BGR, RGB, grayscale, and 8-bit HSV formats used by Supervision."""
    if code in (_COLOR_BGR2RGB, _COLOR_RGB2BGR):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(
                "BGR/RGB conversion requires a three- or four-channel image"
            )
        return np.ascontiguousarray(image[..., 2::-1])

    if code == _COLOR_GRAY2BGR:
        if image.ndim != 2:
            raise ValueError("GRAY2BGR conversion requires a two-dimensional image")
        return np.repeat(image[..., np.newaxis], 3, axis=2)

    if code == _COLOR_BGR2GRAY:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BGR2GRAY conversion requires a three-channel image")
        if image.dtype == np.uint8:
            values = image.astype(np.uint32)
            weighted = (
                values[..., 0] * 3735
                + values[..., 1] * 19235
                + values[..., 2] * 9798
                + (1 << 14)
            ) >> 15
            return weighted.astype(np.uint8)
        float_values = (
            image[..., 0].astype(np.float64) * 0.114
            + image[..., 1].astype(np.float64) * 0.587
            + image[..., 2].astype(np.float64) * 0.299
        )
        return _cast_array_like_opencv(float_values, image.dtype)

    if code == _COLOR_HSV2BGR:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("HSV2BGR conversion requires a three-channel image")
        return _hsv_to_bgr(image)

    raise ValueError(f"Unsupported color conversion code: {code}")


def _hsv_to_bgr(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert OpenCV's 8-bit HSV representation to BGR."""
    values = image.astype(np.float64)
    hue = values[..., 0] / 30.0
    saturation = values[..., 1] / 255.0
    value = values[..., 2] / 255.0

    chroma = value * saturation
    sector_index = np.floor(hue).astype(np.int64) % 6
    sector = hue - np.floor(hue)
    x = chroma * (1 - np.abs(((sector_index + sector) % 2) - 1))
    match = value - chroma
    zeros = np.zeros_like(chroma)

    red = np.choose(sector_index, (chroma, x, zeros, zeros, x, chroma))
    green = np.choose(sector_index, (x, chroma, chroma, x, zeros, zeros))
    blue = np.choose(sector_index, (zeros, zeros, x, chroma, chroma, x))
    bgr = np.stack((blue + match, green + match, red + match), axis=-1) * 255
    return _cast_array_like_opencv(bgr, image.dtype)


def _split(image: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], ...]:
    """Split an image into contiguous single-channel arrays."""
    if image.ndim == 2:
        return (np.ascontiguousarray(image),)
    return tuple(
        np.ascontiguousarray(image[..., index]) for index in range(image.shape[2])
    )


def _merge(channels: Sequence[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Merge single-channel arrays along their final axis."""
    if not channels:
        raise ValueError("At least one channel is required")
    return np.ascontiguousarray(np.stack(channels, axis=-1))
