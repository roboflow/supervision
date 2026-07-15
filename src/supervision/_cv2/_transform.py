"""Private transform and filter fallbacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision._cv2._common import _cast_array_like_opencv
from supervision._cv2.constants import (
    _BORDER_CONSTANT,
    _DIST_L2,
    _INTER_LINEAR,
    _INTER_NEAREST,
)


def _get_rotation_matrix_2d(
    center: tuple[float, float], angle: float, scale: float
) -> npt.NDArray[np.float64]:
    """Build OpenCV's two-dimensional rotation matrix."""
    radians = np.deg2rad(angle)
    alpha = scale * np.cos(radians)
    beta = scale * np.sin(radians)
    center_x, center_y = center
    return np.array(
        [
            [alpha, beta, (1 - alpha) * center_x - beta * center_y],
            [-beta, alpha, beta * center_x + (1 - alpha) * center_y],
        ],
        dtype=np.float64,
    )


def _warp_affine(
    image: npt.NDArray[Any],
    matrix: npt.NDArray[Any],
    dsize: tuple[int, int],
    flags: int = _INTER_LINEAR,
    border_mode: int = _BORDER_CONSTANT,
    border_value: float | Sequence[float] = 0,
) -> npt.NDArray[Any]:
    """Warp an image through an affine matrix using SciPy's inverse sampler."""
    if flags not in (_INTER_NEAREST, _INTER_LINEAR):
        raise ValueError(f"Unsupported interpolation mode: {flags}")
    if border_mode != _BORDER_CONSTANT:
        raise ValueError("Only BORDER_CONSTANT is supported by the fallback")

    from scipy import ndimage

    width, height = dsize
    linear = np.asarray(matrix, dtype=np.float64)[:, :2]
    translation = np.asarray(matrix, dtype=np.float64)[:, 2]
    inverse = np.linalg.inv(linear)
    offset_xy = -inverse @ translation
    transform = inverse[[1, 0]][:, [1, 0]]
    offset = offset_xy[[1, 0]]
    order = 0 if flags == _INTER_NEAREST else 1
    values = np.asarray(image)

    def transform_channel(channel: npt.NDArray[Any], cval: float) -> npt.NDArray[Any]:
        """Apply the shared affine mapping to one channel with padded borders."""
        padded = np.pad(channel, 1, mode="constant", constant_values=cval)
        return cast(
            npt.NDArray[Any],
            ndimage.affine_transform(
                padded,
                transform,
                offset=offset + 1,
                output_shape=(height, width),
                order=order,
                mode="constant",
                cval=cval,
                prefilter=False,
            ),
        )

    if values.ndim == 2:
        cval = (
            float(border_value[0])
            if isinstance(border_value, Sequence)
            else float(border_value)
        )
        return transform_channel(values, cval)

    channels = []
    for channel in range(values.shape[2]):
        cval = (
            float(border_value[channel])
            if isinstance(border_value, Sequence)
            else float(border_value)
        )
        channels.append(transform_channel(values[..., channel], cval))
    return np.stack(channels, axis=-1).astype(image.dtype, copy=False)


def _blur(
    image: npt.NDArray[Any], ksize: tuple[int, int], border_type: int = 4
) -> npt.NDArray[Any]:
    """Apply a box filter with OpenCV's default reflect-101 boundary behavior."""
    if min(ksize) <= 0:
        raise ValueError("Blur kernel dimensions must be positive")
    if border_type != 4:
        raise ValueError("Only OpenCV's default blur border is supported")

    from scipy import ndimage

    size = (*ksize[::-1], 1) if image.ndim == 3 else ksize[::-1]
    values = ndimage.uniform_filter(image.astype(np.float64), size=size, mode="mirror")
    return np.ascontiguousarray(_cast_array_like_opencv(values, image.dtype))


def _distance_transform(
    image: npt.NDArray[Any], distance_type: int, mask_size: int, dst_type: int = 5
) -> npt.NDArray[np.float32]:
    """Compute the L2 distance to the nearest zero pixel."""
    if distance_type != _DIST_L2:
        raise ValueError("Only DIST_L2 is supported by the fallback")
    del mask_size, dst_type
    from scipy import ndimage

    return cast(
        npt.NDArray[np.float32],
        ndimage.distance_transform_edt(image != 0).astype(np.float32),
    )
