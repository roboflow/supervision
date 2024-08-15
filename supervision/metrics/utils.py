from typing import Tuple

import numpy as np
import numpy.typing as npt


def pad_mask(mask: npt.NDArray, new_shape: Tuple[int, int]) -> npt.NDArray:
    """Pad a mask to a new shape, inserting zeros on the right and bottom."""
    if len(mask.shape) != 3:
        raise ValueError(f"Invalid mask shape: {mask.shape}. Expected: (N, H, W)")

    new_mask = np.pad(
        mask,
        (
            (0, 0),
            (0, new_shape[0] - mask.shape[1]),
            (0, new_shape[1] - mask.shape[2]),
        ),
        mode="constant",
        constant_values=0,
    )

    return new_mask


def len0_like(data: npt.NDArray) -> npt.NDArray:
    """Create an empty array with the same shape as input, but with 0 rows."""
    return np.empty((0, *data.shape[1:]), dtype=data.dtype)
