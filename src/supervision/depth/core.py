from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

#: A frame adds about this many samples, or fewer, to a clip-wide percentile estimate.
_CLIP_SAMPLES_PER_FRAME = 65536
#: A clip-wide percentile estimate keeps at most about this many float64 samples.
_CLIP_SAMPLE_BUDGET = 1 << 22


class DepthKind(Enum):
    """What the values of a `sv.DepthMap` measure.

    Attributes:
        DISPARITY_PX: Stereo disparity in pixels of the map. Larger is nearer.
            Metric depth is `fx_px * baseline_m / (disparity + doffs_px)`.
        DEPTH_M: Metric depth along the optical axis, in metres. Smaller is nearer.
        RELATIVE_INVERSE: Unitless relative depth from a monocular model,
            normalised so larger is nearer, with no metric scale. The kind promises
            only that larger is nearer: Depth Anything V1, V2 and DPT output inverse
            depth up to an unknown scale and shift, while Depth Anything V3 output is
            linear in depth. Roboflow Inference's maps run from 0 for the farthest
            pixel to 1 for the nearest.
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
        with np.errstate(divide="ignore", invalid="ignore"):
            converted: npt.NDArray[np.floating] = dtype(1.0) / values
        return converted

    def apply_range(
        self, value_range: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Convert a range, swapping its ends under a reciprocal; None if degenerate."""
        converted = self.apply(np.array(value_range, dtype=np.float64))
        low, high = float(converted.min()), float(converted.max())
        if math.isfinite(low) and math.isfinite(high) and low < high:
            return low, high
        return None


def _resolve_conversion(kind: DepthKind) -> _Conversion:
    """Return the conversion from `kind` values to the coloured quantity.

    Disparity and relative maps are coloured as they are. A metric map is coloured as
    inverse depth, which spends colour on near detail the way disparity does.
    """
    return _Conversion(reciprocal=kind is DepthKind.DEPTH_M)


def _plain_float(value: Any, field: str) -> float:
    """Return a Python, NumPy or one-value tensor number as a plain float."""
    if not isinstance(value, (str, bytes, bool, np.bool_)):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    raise TypeError(f"{field} must be a real number, got {value!r}.")


def _nearest_rank(count: int, fraction: float) -> int:
    """Return the nearest rank `round(fraction * (count - 1))`, rounding half up."""
    return math.floor(fraction * (count - 1) + 0.5)


def _check_percentiles(low: float, high: float) -> None:
    """Reject percentiles outside `0 <= low < high <= 100`."""
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        raise ValueError(
            f"Depth percentiles need 0 <= low < high <= 100, got {low} and {high}."
        )


def _values_at_ranks(
    values: npt.NDArray[Any], low: float, high: float
) -> tuple[float, float]:
    """Return the values at the low and high nearest ranks of an unsorted array.

    Ends that coincide are widened by one float32 step, so the range stays usable as an
    explicit range.
    """
    low_rank = _nearest_rank(values.size, low / 100)
    high_rank = _nearest_rank(values.size, high / 100)
    ordered = np.partition(values, [low_rank, high_rank])
    low_value, high_value = float(ordered[low_rank]), float(ordered[high_rank])
    if low_value == high_value:
        high_value = float(np.nextafter(np.float32(high_value), np.float32(np.inf)))
    return low_value, high_value


def _index_map(source: int, target: int) -> npt.NDArray[np.intp]:
    """Return the source index under each target pixel centre (nearest sampling)."""
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
                stored as float32.
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

    def _percentile_range(
        self,
        low: float = 2.0,
        high: float = 98.0,
    ) -> tuple[float, float] | None:
        """Return this map's own percentile range in the coloured quantity's unit.

        It is the range `sv.DepthAnnotator` uses with `display_range="auto"`: the
        nearest-rank percentiles of every valid value, converted to inverse depth for
        a metric map.

        Args:
            low: Lower percentile, from 0 to 100.
            high: Upper percentile, from 0 to 100.

        Returns:
            `(low, high)` in the coloured quantity's unit, or `None` when no pixel
            holds depth.

        Raises:
            ValueError: If the percentiles are out of order.
        """
        _check_percentiles(low, high)
        conversion = _resolve_conversion(self.kind)
        values = self.values[self.valid_mask]
        if values.size == 0:
            return None
        return conversion.apply_range(_values_at_ranks(values, low, high)) or (0.0, 1.0)


@dataclass(frozen=True)
class DepthClipRange:
    """One colour range for a whole clip of depth maps.

    Colouring each frame with its own range makes a still wall change colour as
    things enter and leave the frame. A clip range, computed in a first pass over the
    clip, keeps the colour scale fixed on every frame. Colours then stay put only
    where the depth values are steady, as in ground truth or calibrated stereo; a
    model's own frame-to-frame wobble becomes more visible.

    Attributes:
        display_range: `(low, high)` colour range in the maps' kind unit.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> frames = [
        ...     sv.DepthMap(np.full((4, 4), d, dtype=np.float32), kind="disparity_px")
        ...     for d in (10.0, 20.0, 30.0)
        ... ]
        >>> sv.DepthClipRange.from_depth_maps(frames)
        DepthClipRange(display_range=(10.0, 30.0))

        ```
    """

    display_range: tuple[float, float]

    def __post_init__(self) -> None:
        """Store plain floats and reject an empty range."""
        low, high = (
            _plain_float(bound, "DepthClipRange display_range")
            for bound in self.display_range
        )
        object.__setattr__(self, "display_range", (low, high))
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(
                "DepthClipRange display_range must be two finite numbers with "
                f"low < high, got {self.display_range}."
            )

    @classmethod
    def from_depth_maps(
        cls,
        depth_maps: Iterable[DepthMap],
        low: float = 2.0,
        high: float = 98.0,
    ) -> DepthClipRange:
        """Compute a clip's percentile range in one pass.

        Each map contributes its valid values, thinned at random to about 65,536
        values when it has more, so a large frame weighs no more than a small one.
        Whenever the clip's samples pass 4,194,304, each one is kept with probability
        1/2 and later maps are kept at half the previous rate, so memory stays bounded
        however long the clip is and early and late frames are sampled alike. The
        random draws are seeded, so the result is reproducible. Pass a generator to
        read the clip once without holding it.

        Args:
            depth_maps: The clip's maps, all of one kind.
            low: Lower percentile, from 0 to 100.
            high: Upper percentile, from 0 to 100.

        Returns:
            The clip's `sv.DepthClipRange`.

        Raises:
            ValueError: If the maps mix kinds or hold no depth at all.
        """
        _check_percentiles(low, high)
        kind: DepthKind | None = None
        samples: list[npt.NDArray[np.float64]] = []
        retained = 0
        keep_share = 1.0
        rng = np.random.default_rng(0)
        for depth_map in depth_maps:
            if kind is None:
                kind = depth_map.kind
            elif depth_map.kind is not kind:
                raise ValueError(
                    f"A clip range needs maps of one kind; got {kind.value} and "
                    f"{depth_map.kind.value}."
                )
            values = depth_map.values[depth_map.valid_mask]
            if values.size == 0:
                continue
            share = keep_share * min(1.0, _CLIP_SAMPLES_PER_FRAME / values.size)
            if share < 1.0:
                # Random rather than every n-th sample: a fixed step aliases with
                # structured frames and skews the percentiles.
                values = values[rng.random(values.size) < share]
            sample = values.astype(np.float64)
            samples.append(sample)
            retained += sample.size
            if retained > _CLIP_SAMPLE_BUDGET:
                # Keeping each sample with probability 1/2 leaves every sample of the
                # clip kept with probability `keep_share / 2`.
                kept = np.concatenate(samples)
                kept = kept[rng.random(kept.size) < 0.5]
                samples, retained, keep_share = [kept], kept.size, keep_share / 2
        if not samples or sum(sample.size for sample in samples) == 0:
            raise ValueError("The clip holds no depth to compute a range from.")
        display_range = _values_at_ranks(np.concatenate(samples), low, high)
        return cls(display_range=display_range)
