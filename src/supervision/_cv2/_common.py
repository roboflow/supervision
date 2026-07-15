"""Private helpers shared by OpenCV fallback implementations."""

from __future__ import annotations

from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt


class BackendUnavailableError(RuntimeError):
    """Raised when an OpenCV operation is used without an available backend."""


def _unavailable(*args: object, **kwargs: object) -> NoReturn:
    """Fail clearly until a later domain PR provides the fallback operation."""
    del args, kwargs
    raise BackendUnavailableError(
        "OpenCV is not installed and this operation has no fallback yet."
    )


def _cast_array_like_opencv(
    values: npt.NDArray[Any], dtype: np.dtype[Any]
) -> npt.NDArray[Any]:
    """Round integer results using OpenCV's saturating conversion convention."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.rint(values)
        values = np.clip(values, info.min, info.max)
    return values.astype(dtype, copy=False)
