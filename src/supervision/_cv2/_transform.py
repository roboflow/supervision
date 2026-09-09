"""Private transform and filter fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from supervision._cv2._common import _cast_array_like_opencv


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
