"""Colour tables that `sv.DepthAnnotator` paints depth maps with.

Each table is 256 RGB entries, darkest or coldest first, in the order the colour
coordinate `t` runs from 0 to 1: matplotlib's colormap of the same name, or a plain
0 to 255 ramp for grayscale.
"""

from __future__ import annotations

from enum import Enum
from functools import cache

import numpy as np
import numpy.typing as npt

#: Entries in every depth colour table; the colour coordinate `t` picks one.
_TABLE_ENTRIES = 256
#: Interpolated steps between two neighbouring table entries in the expanded lookup.
_STEPS_PER_ENTRY = 16
_EXPANDED_ENTRIES = (_TABLE_ENTRIES - 1) * _STEPS_PER_ENTRY + 1


class DepthColormap(Enum):
    """Colour tables a depth map can be painted with.

    The near end of the depth range is always the warm or bright end of the table.
    Turbo separates the most depth steps and is the default; Viridis and Cividis keep
    their order in grayscale and for colour-blind viewers; Inferno and Magma start
    near black, so pixels without depth should stay unpainted with them.

    Attributes:
        TURBO: Google's Turbo rainbow, built for depth and disparity.
        VIRIDIS: Perceptually uniform, blue to yellow.
        CIVIDIS: Perceptually uniform and optimised for colour-vision deficiency.
        INFERNO: Perceptually uniform, black to pale yellow.
        MAGMA: Perceptually uniform, black to pale pink.
        GRAYSCALE: Black far, white near.
    """

    TURBO = "turbo"
    VIRIDIS = "viridis"
    CIVIDIS = "cividis"
    INFERNO = "inferno"
    MAGMA = "magma"
    GRAYSCALE = "grayscale"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string value of every colormap."""
        return [member.value for member in cls]

    @classmethod
    def from_value(cls, value: DepthColormap | str) -> DepthColormap:
        """Resolve a colormap from an enum member or its case-insensitive name.

        Args:
            value: A `DepthColormap` member or one of its string values.

        Returns:
            The matching `DepthColormap`.

        Raises:
            ValueError: If `value` names no colormap.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> sv.DepthColormap.from_value("Viridis")
            <DepthColormap.VIRIDIS: 'viridis'>

            ```
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"Invalid depth colormap: {value!r}. Must be one of {cls.list()}."
        )


@cache
def _rgb_lut(colormap: DepthColormap) -> npt.NDArray[np.uint8]:
    """Build a colormap's read-only `(256, 3)` RGB table once, rounding half up."""
    lut: npt.NDArray[np.uint8]
    if colormap is DepthColormap.GRAYSCALE:
        ramp = np.arange(_TABLE_ENTRIES, dtype=np.uint8)
        lut = np.repeat(ramp[:, np.newaxis], 3, axis=1)
    else:
        import matplotlib

        rgba = matplotlib.colormaps[colormap.value](np.arange(_TABLE_ENTRIES))
        lut = (rgba[:, :3] * 255 + 0.5).astype(np.uint8)
    lut.flags.writeable = False
    return lut


@cache
def _expanded_bgr_lut(colormap: DepthColormap) -> npt.NDArray[np.uint8]:
    """Return the table interpolated to 16 steps per entry, in BGR order.

    A colour between two table entries is their linear blend, so smooth surfaces get
    smooth gradients instead of the visible bands that 256 flat steps draw in Turbo.
    Expanding the table once to `255 * 16 + 1` entries gives that blend to a
    sixteenth of an entry with one `np.take` per frame; entry `16 * i` is exactly
    table entry `i`.
    """
    rgb = _rgb_lut(colormap).astype(np.float64)
    positions = np.arange(_EXPANDED_ENTRIES) / _STEPS_PER_ENTRY
    expanded = np.column_stack(
        [
            np.interp(positions, np.arange(_TABLE_ENTRIES), rgb[:, channel])
            for channel in range(3)
        ]
    )
    bgr: npt.NDArray[np.uint8] = np.rint(expanded[:, ::-1]).astype(np.uint8)
    bgr.flags.writeable = False
    return bgr


def _colorize(
    t: npt.NDArray[np.floating], colormap: DepthColormap
) -> npt.NDArray[np.uint8]:
    """Map colour coordinates in `[0, 1]` to BGR colours, shape `t.shape + (3,)`."""
    index = np.rint(t * (_EXPANDED_ENTRIES - 1)).astype(np.intp)
    colors: npt.NDArray[np.uint8] = np.take(_expanded_bgr_lut(colormap), index, axis=0)
    return colors
