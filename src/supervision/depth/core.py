from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class DepthKind(Enum):
    """What the values of a `sv.DepthMap` measure.

    Attributes:
        DISPARITY_PX: Stereo disparity in pixels of the map. Larger is nearer.
            Metric depth is `fx_px * baseline_m / (disparity + doffs_px)`.
        DEPTH_M: Metric depth along the optical axis, in metres. Smaller is nearer.
        RELATIVE_INVERSE: Unitless relative depth from a monocular model, with no
            metric scale, where larger is nearer and 0 is the farthest valid value.
            Depth Anything V1, V2 and DPT output inverse depth up to an unknown scale
            and shift, which fits as it is; Roboflow Inference's maps run from 0 for
            the farthest pixel to 1 for the nearest. Invert a relative map that
            grows with distance, such as Depth Anything V3's, which is linear in
            depth: pass `1 / values`, not `-values`, whose pixels would all be
            negative and so without depth. Set pixels without depth to `NaN` first,
            because 0 is a valid value of this kind.
    """

    DISPARITY_PX = "disparity_px"
    DEPTH_M = "depth_m"
    RELATIVE_INVERSE = "relative_inverse"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string value of every kind."""
        return [member.value for member in cls]

    @classmethod
    def from_value(cls, value: DepthKind | str) -> DepthKind:
        """Resolve a kind from an enum member or its case-insensitive value.

        Args:
            value: A `DepthKind` member or one of its string values.

        Returns:
            The matching `DepthKind`.

        Raises:
            ValueError: If `value` names no kind.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> sv.DepthKind.from_value("depth_m")
            <DepthKind.DEPTH_M: 'depth_m'>

            ```
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(f"Invalid depth kind: {value!r}. Must be one of {cls.list()}.")


@dataclass(frozen=True)
class _Conversion:
    """How a value in the map kind's unit becomes the ranged or coloured quantity.

    `reciprocal ? 1 / x : x`.
    """

    reciprocal: bool = False

    def apply(self, values: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Convert an array in its own float precision; x / 0 yields infinities."""
        if not self.reciprocal:
            return values
        dtype = values.dtype.type
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            converted: npt.NDArray[np.floating] = dtype(1.0) / values
        return converted

    def apply_range(
        self, value_range: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Convert a range, swapping its ends under a reciprocal; None if not finite.

        Equal ends stay equal; colouring handles a flat range.
        """
        converted = self.apply(np.array(value_range, dtype=np.float64))
        low, high = float(converted.min()), float(converted.max())
        if math.isfinite(low) and math.isfinite(high):
            return low, high
        return None


def _resolve_conversion(kind: DepthKind) -> _Conversion:
    """Return the conversion from `kind` values to the coloured quantity.

    Disparity and relative maps are coloured as they are. A metric map is coloured as
    inverse depth, which spends colour on near detail the way disparity does.
    """
    return _Conversion(reciprocal=kind is DepthKind.DEPTH_M)


def _index_map(source: int, target: int) -> npt.NDArray[np.intp]:
    """Return the source index under each target pixel centre (nearest sampling).

    Sampling at pixel centres keeps the scaled map aligned with the scene and lets
    the valid-depth mask be resampled as plain booleans, which `sv.resize_image`
    does not offer.
    """
    positions = (np.arange(target, dtype=np.float64) + 0.5) * source / target
    return np.minimum(positions.astype(np.intp), source - 1)


class DepthMap:
    """A per-pixel depth, disparity or relative depth map for one frame.

    `sv.DepthMap` is to depth what `sv.KeyPoints` is to pose: its own container with
    its own annotator ([`sv.DepthAnnotator`](/latest/depth/annotators/)). It belongs
    to the whole frame, so it is not a `sv.Detections` field.

    `values` are float32 in the kind's unit. For `disparity_px` and `depth_m`,
    non-finite values and values `<= 0` are no depth; for `relative_inverse`,
    non-finite and negative values are no depth, because a normalised map puts its
    farthest real pixel at exactly 0.

    Attributes:
        values: `(H, W)` float32 values in the kind's unit.
        kind: What the values measure.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> disparity = np.array([[0.0, 10.0], [20.0, 40.0]], dtype=np.float32)
        >>> depth_map = sv.DepthMap(disparity, kind="disparity_px")
        >>> depth_map.valid_mask
        array([[False,  True],
               [ True,  True]])

        ```
    """

    values: npt.NDArray[np.float32]
    kind: DepthKind

    def __init__(
        self,
        values: npt.NDArray[np.floating],
        kind: DepthKind | str,
    ) -> None:
        """Create a depth map from float values in the kind's unit.

        Args:
            values: `(H, W)` float array in the kind's unit; any float dtype is
                stored as float32. A float32 array is stored without a copy, so
                later changes to it show in the map. Masked-array masks are
                ignored: mark missing depth with `NaN`, an infinity or a value the
                kind treats as no depth.
            kind: What the values measure, as a `sv.DepthKind` or its string value.

        Raises:
            ValueError: If the values are not a 2D float array.
        """
        array = np.asarray(values)
        if array.ndim != 2 or array.size == 0:
            raise ValueError(
                f"DepthMap values must be a non-empty (H, W) array, got {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(
                f"DepthMap values must be float, got {array.dtype}. Divide integer "
                "codes, such as a 16-bit PNG's, by your dataset's scale first."
            )
        self.values = array.astype(np.float32, copy=False)
        self.kind = DepthKind.from_value(kind)

    def __repr__(self) -> str:
        """Summarise the map without printing its values."""
        width, height = self.resolution_wh
        return f"DepthMap(kind={self.kind.value!r}, resolution_wh=({width}, {height}))"

    def __eq__(self, other: object) -> bool:
        """Compare every field; values compare NaN equal to NaN."""
        if not isinstance(other, DepthMap):
            return NotImplemented
        return self.kind is other.kind and np.array_equal(
            self.values, other.values, equal_nan=True
        )

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """The map's `(width, height)` in pixels."""
        height, width = self.values.shape
        return int(width), int(height)

    @property
    def valid_mask(self) -> npt.NDArray[np.bool_]:
        """A boolean `(H, W)` array, `True` where the map holds depth."""
        if self.kind is DepthKind.RELATIVE_INVERSE:
            return np.asarray(np.isfinite(self.values) & (self.values >= 0))
        return np.asarray(np.isfinite(self.values) & (self.values > 0))

    def to_float(self, no_depth_value: float = np.nan) -> npt.NDArray[np.float32]:
        """Return the values in the kind's unit as a new float32 array.

        Args:
            no_depth_value: The value written where the map has no depth.

        Returns:
            A `(H, W)` float32 array.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> disparity = np.array([[np.nan, 1.0, 10.0]], dtype=np.float32)
            >>> sv.DepthMap(disparity, kind="disparity_px").to_float(0.0)
            array([[ 0.,  1., 10.]], dtype=float32)

            ```
        """
        result: npt.NDArray[np.float32] = self.values.copy()
        result[~self.valid_mask] = no_depth_value
        return result
