from typing import List, Tuple

import numpy as np
import numpy.typing as npt


def ensure_pandas_installed():
    try:
        import pandas  # noqa
    except ImportError:
        raise ImportError(
            "`metrics` extra is required to run the function."
            " Run `pip install 'supervision[metrics]'` or"
            " `poetry add supervision -E metrics`"
        )


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


def unify_pad_masks_shape(
    *masks: npt.NDArray[np.bool_],
) -> List[npt.NDArray[np.bool_]]:
    """
    Given any number of (N, H, W) mask objects, return copies of the
    same (H, W), padded to the largest dimensions.

    Args:
        *masks (np.ndarray): The mask arrays to unify. Each shaped (_, H, W),

    Returns:
        List[np.ndarray]: The masks, padded to the largest dimensions.
            Each list element shaped (_, H_max, W_max)
    """
    new_h = 0
    new_w = 0
    for mask in masks:
        if len(mask.shape) != 3:
            raise ValueError(f"Invalid mask shape: {mask.shape}. Expected: (N, H, W)")

        _, h, w = mask.shape
        new_h = max(new_h, h)
        new_w = max(new_w, w)

    results = []
    for mask in masks:
        results.append(pad_mask(mask, (new_h, new_w)))

    return results


def len0_like(data: npt.NDArray) -> npt.NDArray:
    """Create an empty array with the same shape as input, but with 0 rows."""
    return np.empty((0, *data.shape[1:]), dtype=data.dtype)
