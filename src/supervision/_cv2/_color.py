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
    """Convert the BGR, RGB, grayscale, and HSV formats used by Supervision."""
    if code in (_COLOR_BGR2RGB, _COLOR_RGB2BGR):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(
                "BGR/RGB conversion requires a three- or four-channel image"
            )
        return np.ascontiguousarray(image[..., 2::-1])

    if code == _COLOR_GRAY2BGR:
        # OpenCV reads a (H, W, 1) array as a single-channel image too.
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError("GRAY2BGR conversion requires a single-channel image")
        return np.repeat(image[..., np.newaxis], 3, axis=2)

    if code == _COLOR_BGR2GRAY:
        # OpenCV also takes BGRA here and ignores the alpha channel.
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(
                "BGR2GRAY conversion requires a three- or four-channel image"
            )
        if image.dtype in (np.uint8, np.uint16):
            # OpenCV converts 8- and 16-bit images with the same 15-bit
            # fixed-point coefficients. The weights sum to 1 << 15, so even a
            # 16-bit pixel (65535 * 32768 + 16384) fits in uint32.
            values = image.astype(np.uint32)
            weighted = (
                values[..., 0] * 3735
                + values[..., 1] * 19235
                + values[..., 2] * 9798
                + (1 << 14)
            ) >> 15
            return weighted.astype(image.dtype)
        float_values = (
            image[..., 0].astype(np.float64) * 0.114
            + image[..., 1].astype(np.float64) * 0.587
            + image[..., 2].astype(np.float64) * 0.299
        )
        return _cast_array_like_opencv(float_values, image.dtype)

    if code == _COLOR_HSV2BGR:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("HSV2BGR conversion requires a three-channel image")
        if image.dtype not in (np.uint8, np.float32):
            raise ValueError("HSV2BGR conversion requires an 8-bit or float32 image")
        return _hsv_to_bgr(image)

    raise ValueError(f"Unsupported color conversion code: {code}")


# OpenCV's x86-64 wheels convert 8-bit HSV in AVX2 blocks of 32 pixels per row.
_HSV_VECTOR_BLOCK = 32

# For each hue sector, which of OpenCV's four candidate values (v, p, q, t) becomes
# blue, green and red.
_HSV_SECTOR_CHANNELS = np.array(
    [[1, 3, 0], [1, 0, 2], [3, 0, 1], [0, 2, 1], [0, 1, 3], [2, 1, 0]]
)


def _hsv_to_bgr(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert OpenCV's HSV representation to BGR with OpenCV's float32 arithmetic.

    Like OpenCV, an 8-bit image holds hue in [0, 180) and saturation and value in [0,
    255], and gives BGR in [0, 255]. A float32 image holds hue in degrees, [0, 360), and
    saturation and value in [0, 1], and gives BGR in [0, 1].

    OpenCV converts both depths in float32: 8-bit hue is scaled by ``6 / 180`` and
    saturation and value by ``1 / 255``, and each channel is picked from ``v``,
    ``v * (1 - s)``, ``v * (1 - s * f)`` and ``v * (1 - s * (1 - f))``. Float32 results
    then match OpenCV's Windows and macOS wheels bit for bit. The Linux wheels fuse
    ``1 - s * f`` into one multiply-add, so a few of their results differ by about
    6e-8 in float32, or one 8-bit level. Hues of 360 degrees (180) and more wrap
    around, as in OpenCV; OpenCV's own scalar and vector code disagree on negative
    hues, so those are not guaranteed to match.

    8-bit results are ``255 * channel``, which OpenCV's vector code truncates and its
    scalar code, for the pixels left over at the end of each row, rounds. This follows
    the AVX2 code of OpenCV's x86-64 wheels: whole blocks of 32 pixels are truncated
    and the rest of the row is rounded. Builds converting other block sizes, such as
    ARM ones, can differ by one level only in the last pixels of a row.

    Example:
        ```pycon
        >>> hsv = np.full((1, 33, 3), (1, 255, 255), dtype=np.uint8)
        >>> bgr = _hsv_to_bgr(hsv)
        >>> bgr[0, 0].tolist(), bgr[0, 32].tolist()
        ([0, 8, 255], [0, 9, 255])

        ```
    """
    float32 = np.float32
    one = float32(1)
    if image.dtype == np.float32:
        hue = image[..., 0] * float32(1 / 60)
        saturation = image[..., 1]
        value = image[..., 2]
    else:
        hue = image[..., 0].astype(float32) * float32(6 / 180)
        saturation = image[..., 1].astype(float32) * float32(1 / 255)
        value = image[..., 2].astype(float32) * float32(1 / 255)

    with np.errstate(invalid="ignore"):
        whole = np.floor(hue)
        fraction = hue - whole
        sector = np.nan_to_num(whole).astype(np.int64) % 6
    candidates = np.stack(
        (
            value,
            value * (one - saturation),
            value * (one - saturation * fraction),
            value * (one - saturation * (one - fraction)),
        ),
        axis=-1,
    )
    bgr = np.take_along_axis(candidates, _HSV_SECTOR_CHANNELS[sector], axis=-1)
    # Gray pixels take the value directly, whatever the hue.
    bgr = np.where((saturation == 0)[..., np.newaxis], value[..., np.newaxis], bgr)
    if image.dtype == np.float32:
        return np.ascontiguousarray(bgr, dtype=np.float32)
    scaled = bgr * float32(255)
    width = image.shape[1]
    vector_columns = np.arange(width) < width - width % _HSV_VECTOR_BLOCK
    levels = np.where(vector_columns[:, np.newaxis], np.trunc(scaled), np.rint(scaled))
    return _cast_array_like_opencv(levels, image.dtype)


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
