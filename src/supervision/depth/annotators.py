from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision import _cv2 as cv2
from supervision.depth.colormaps import DepthColormap, _colorize
from supervision.depth.core import (
    DepthKind,
    DepthMap,
    _Conversion,
    _index_map,
    _resolve_conversion,
)
from supervision.draw.base import ImageType
from supervision.utils.conversion import ensure_cv2_image_for_class_method

#: Colour range, in the coloured quantity's unit, for a map that holds no depth.
_FALLBACK_RANGE = (0.0, 1.0)
#: A frame adds about this many samples, or fewer, to a clip-wide percentile estimate.
_CLIP_SAMPLES_PER_FRAME = 65536
#: A clip-wide percentile estimate keeps at most about this many float64 samples.
_CLIP_SAMPLE_BUDGET = 1 << 22


class DepthAnnotator:
    """Colours a `sv.DepthMap` over an image, near objects warm and far ones cool.

    Pixels without depth are left unpainted, so the scene shows through.

    A metric map is coloured as inverse depth, which spends colour on near detail the
    way disparity does.

    === "Image"

        ```python
        import numpy as np
        import supervision as sv
        from PIL import Image

        image = Image.open("<SOURCE_IMAGE_PATH>")
        depth_map = sv.DepthMap(np.load("<DEPTH_NPY_PATH>"), kind="depth_m")

        depth_annotator = sv.DepthAnnotator(display_range="auto", opacity=0.6)
        annotated_image = depth_annotator.annotate(image.copy(), depth_map)
        ```
    """

    def __init__(
        self,
        colormap: DepthColormap | str = DepthColormap.TURBO,
        display_range: str | tuple[float, float] | DepthClipRange = "auto",
        opacity: float = 1.0,
    ) -> None:
        """
        Args:
            colormap: Colour table: `"turbo"` (default), `"viridis"`, `"cividis"`,
                `"inferno"`, `"magma"` or `"grayscale"`. The near end is always the
                warm or bright end.
            display_range: The values the colour table spans; values outside clamp
                to its ends.
                `"auto"` (default) uses this map's 2nd to 98th percentile. When
                they coincide, as on a flat map, every pixel takes the far-end
                colour.
                A `(low, high)` tuple fixes the range in the map's own unit: metres
                for `depth_m`, pixels for `disparity_px` and the raw values for
                `relative_inverse`. Whatever the unit, the near end of the range
                takes the warm colour.
                A `sv.DepthClipRange` uses one range, in the maps' own unit like a
                tuple, for every frame of a clip.
            opacity: Opacity of the colours over the scene, from 0 to 1. Values
                outside are clamped: `<= 0` draws nothing and `>= 1` fully replaces
                the pixels that have depth.

        Raises:
            ValueError: If `colormap` or `display_range` is invalid.
        """
        self.colormap = DepthColormap.from_value(colormap)
        self.display_range = _check_display_range_option(display_range)
        self.opacity = opacity

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, depth_map: DepthMap) -> ImageType:
        """Colour the depth map over the scene.

        A map whose size differs from the scene's is stretched over the whole scene,
        each scene pixel showing the map pixel under its centre, so a smaller map
        with the scene's aspect ratio lines up. A map with another aspect ratio is
        stretched along each axis on its own; if the model saw a letterboxed or
        cropped frame, undo that on the map before annotating.

        Args:
            scene: The image to draw on, a 3-channel `uint8` `numpy.ndarray` (BGR) or
                a `PIL.Image.Image`. It is drawn on in place.
            depth_map: The depth map to colour.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray` or
                `PIL.Image.Image`).

        Raises:
            TypeError: If `scene` is not a `numpy.ndarray` or `PIL.Image.Image`.
            ValueError: If `scene` is not a 3-channel `uint8` image, or if
                `display_range` gives no usable colour range for this map's kind,
                such as a `depth_m` range with `low <= 0`.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> disparity = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
            >>> depth_map = sv.DepthMap(disparity, kind="disparity_px")
            >>> annotator = sv.DepthAnnotator(display_range=(1.0, 3.0))
            >>> image = np.zeros((2, 2, 3), dtype=np.uint8)
            >>> annotator.annotate(image, depth_map)[..., ::-1].tolist()
            [[[0, 0, 0], [48, 18, 59]], [[162, 252, 60], [122, 4, 3]]]

            ```
        """
        # The conversion decorator rejects other types and turns a Pillow image into
        # an array, so this always holds; it narrows `ImageType` for the type checker.
        assert isinstance(scene, np.ndarray)
        if scene.ndim != 3 or scene.shape[2] != 3:
            raise ValueError(
                f"DepthAnnotator draws on 3-channel images, got shape {scene.shape}."
            )
        # The colours are uint8; a wider scene would blend differently on OpenCV and
        # on the NumPy fallback, and `opacity >= 1` would write 0-255 into a float one.
        if scene.dtype != np.uint8:
            raise ValueError(
                f"DepthAnnotator draws on uint8 images, got dtype {scene.dtype}."
            )
        conversion = _resolve_conversion(depth_map.kind)
        valid = depth_map.valid_mask
        value_range = self._resolve_range(depth_map, valid)
        coordinates = _color_coordinates(depth_map, valid, conversion, value_range)
        colors = _colorize(coordinates, self.colormap)

        scene_height, scene_width = scene.shape[:2]
        map_width, map_height = depth_map.resolution_wh
        if (map_height, map_width) != (scene_height, scene_width):
            rows = _index_map(map_height, scene_height)
            columns = _index_map(map_width, scene_width)
            # One 1-D take per axis is several times faster than a broadcast
            # `[rows[:, None], columns]` fancy index and samples the same pixels.
            colors = colors.take(rows, axis=0).take(columns, axis=1)
            valid = valid.take(rows, axis=0).take(columns, axis=1)

        _blend(scene, valid, colors, self.opacity)
        return scene

    def _resolve_range(
        self, depth_map: DepthMap, valid: npt.NDArray[np.bool_]
    ) -> tuple[float, float]:
        """Return the colour range in the quantity's unit for this map.

        An explicit range or clip range is converted from the map's unit. `"auto"`
        spans the pixels set in `valid`; a map without any depth falls back to
        `(0, 1)`.
        """
        option = self.display_range
        if isinstance(option, DepthClipRange):
            option = option.display_range
        if isinstance(option, tuple):
            return _convert_display_range(option, depth_map.kind)
        conversion = _resolve_conversion(depth_map.kind)
        percentile = _percentile_range(depth_map.values[valid], conversion)
        return percentile if percentile is not None else _FALLBACK_RANGE


@dataclass(frozen=True)
class DepthClipRange:
    """One colour range for a whole clip of depth maps.

    Colouring each frame with its own range makes a still wall change colour as
    things enter and leave the frame. A clip range, computed in a first pass over the
    clip, keeps the colour scale fixed on every frame. Colours then stay put only
    where the depth values are steady, as in ground truth or calibrated stereo; a
    model's own frame-to-frame wobble becomes more visible.

    Attributes:
        display_range: `(low, high)` colour range in the maps' own unit, read like a
            `(low, high)` tuple passed to `sv.DepthAnnotator`. Equal ends, as from a
            clip of one flat value, paint every pixel the far-end colour, as `"auto"`
            does on a flat map.

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
        """Store plain floats and reject a reversed or unusable range."""
        low, high = _display_range_pair(self.display_range)
        object.__setattr__(self, "display_range", (low, high))
        finite = math.isfinite(low) and math.isfinite(high)
        if not (finite and (low == high or _has_usable_span(low, high))):
            raise ValueError(
                "DepthClipRange display_range must be two finite numbers with "
                "low <= high and a span that fits in float32, got "
                f"{self.display_range}."
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
            The clip's `sv.DepthClipRange`, in the maps' own unit.

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


def _check_display_range_option(
    display_range: str | tuple[float, float] | DepthClipRange,
) -> str | tuple[float, float] | DepthClipRange:
    """Validate the annotator's `display_range` option and normalise it."""
    if isinstance(display_range, DepthClipRange):
        return display_range
    if isinstance(display_range, str) and display_range.lower() == "auto":
        return "auto"
    # The pair reader's own message names only pairs, as `DepthClipRange` takes
    # nothing else; the annotator also takes "auto" and a clip range.
    try:
        low, high = _display_range_pair(display_range)
    except ValueError as error:
        raise ValueError(
            "display_range must be 'auto', a (low, high) pair of numbers or an "
            f"sv.DepthClipRange, got {display_range!r}."
        ) from error
    if not _has_usable_span(low, high):
        raise ValueError(
            "display_range must have finite bounds with low < high and a span that "
            f"fits in float32, got {display_range}."
        )
    return low, high


def _display_range_pair(display_range: Any) -> tuple[float, float]:
    """Read a `(low, high)` pair of numbers as plain floats, or raise `ValueError`."""
    # A string would unpack into characters and `"12"` pass as (1.0, 2.0).
    try:
        if isinstance(display_range, str):
            raise TypeError("a string is not a (low, high) pair")
        low, high = (float(bound) for bound in display_range)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "display_range must be a (low, high) pair of numbers, got "
            f"{display_range!r}."
        ) from error
    return low, high


def _convert_display_range(
    display_range: tuple[float, float], kind: DepthKind
) -> tuple[float, float]:
    """Convert an explicit display range from the map's unit to the coloured unit.

    Equal ends stay a flat range, which paints every pixel the far-end colour. So do
    ends a reciprocal brings to one float32 value.

    Raises:
        ValueError: If the kind is coloured as a reciprocal and `low <= 0`, or the
            converted range is not finite or its span does not fit in float32.
    """
    conversion = _resolve_conversion(kind)
    low, high = display_range
    # A reciprocal sends low <= 0 to infinity or flips its sign, so it bounds nothing.
    converted = None
    if not (conversion.reciprocal and low <= 0):
        converted = conversion.apply_range(display_range)
    if converted is not None:
        span = _float32_span(*converted)
        if low == high or (np.isfinite(span) and span > 0):
            return converted
        # Depths a float32 step apart can share one float32 reciprocal. Every pixel
        # between them converts to that value too, so the range is flat in the
        # coloured unit, as on a flat map.
        if span == 0:
            return converted[0], converted[0]
    raise ValueError(
        f"display_range {display_range} gives no usable colour range for a "
        f"{kind.value!r} map. The range is in the map's own unit; a 'depth_m' "
        "range is in metres and needs 0 < low < high."
    )


def _has_usable_span(low: float, high: float) -> bool:
    """Tell whether `high - low` is finite and positive in float32, as maps colour."""
    span = _float32_span(low, high)
    return bool(np.isfinite(span) and span > 0)


def _float32_span(low: float, high: float) -> np.float32:
    """Return `high - low` in float32; not finite if an end or the span overflows."""
    with np.errstate(over="ignore", invalid="ignore"):
        span: np.float32 = np.float32(high) - np.float32(low)
    return span


def _percentile_range(
    values: npt.NDArray[np.float32],
    conversion: _Conversion,
    low: float = 2.0,
    high: float = 98.0,
) -> tuple[float, float] | None:
    """Return the nearest-rank percentile range of a map's valid values.

    It is the range `display_range="auto"` uses, converted to the coloured quantity,
    so inverse depth for a metric map.

    Args:
        values: The map's valid values in the kind's unit, as a 1D array.
        conversion: The conversion from the kind's unit to the coloured quantity.
        low: Lower percentile, from 0 to 100.
        high: Upper percentile, from 0 to 100.

    Returns:
        `(low, high)` in the coloured quantity's unit, equal ends for a flat map, or
        `None` when there are no values or a converted end is not finite.

    Raises:
        ValueError: If the percentiles are out of order.
    """
    _check_percentiles(low, high)
    if values.size == 0:
        return None
    return conversion.apply_range(_values_at_ranks(values, low, high))


def _check_percentiles(low: float, high: float) -> None:
    """Reject percentiles outside `0 <= low < high <= 100`."""
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        raise ValueError(
            f"Depth percentiles need 0 <= low < high <= 100, got {low} and {high}."
        )


def _values_at_ranks(
    values: npt.NDArray[np.float32], low: float, high: float
) -> tuple[float, float]:
    """Return the values at the low and high nearest ranks of an unsorted array."""
    low_rank = _nearest_rank(values.size, low / 100)
    high_rank = _nearest_rank(values.size, high / 100)
    ordered = np.partition(values, [low_rank, high_rank])
    return float(ordered[low_rank]), float(ordered[high_rank])


def _nearest_rank(count: int, fraction: float) -> int:
    """Return the nearest rank `round(fraction * (count - 1))`, rounding half up."""
    return math.floor(fraction * (count - 1) + 0.5)


def _color_coordinates(
    depth_map: DepthMap,
    valid: npt.NDArray[np.bool_],
    conversion: _Conversion,
    value_range: tuple[float, float],
) -> npt.NDArray[np.floating]:
    """Return each pixel's colour coordinate in `[0, 1]`, 1 at the near end.

    `t = clamp((v - low) / (high - low))`. A flat range, `low == high`, maps every
    pixel to 0, as matplotlib's `Normalize` does. Pixels not set in `valid` get 0;
    they are not painted.
    """
    low, high = value_range
    if low == high:
        return np.zeros(depth_map.values.shape, dtype=np.float32)
    converted = conversion.apply(depth_map.values)
    span = np.float32(high - low)
    # The subtraction allocates the result, so the map's values are never written;
    # pixels without depth may hold any float until they are zeroed below.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coordinates: npt.NDArray[np.floating] = (converted - np.float32(low)) / span
    np.clip(coordinates, 0.0, 1.0, out=coordinates)
    coordinates[~valid] = 0.0
    return coordinates


def _blend(
    scene: npt.NDArray[np.uint8],
    where: npt.NDArray[np.bool_],
    colors: npt.NDArray[np.uint8],
    opacity: float,
) -> None:
    """Blend `colors` into `scene` at `opacity`, only where `where` is set.

    `colors` is `(H, W, 3)`. The blend is `addWeighted`, as in the other annotators.
    Blending the whole image and copying it through the mask is several times faster
    than gathering and scattering the masked pixels.
    """
    if opacity <= 0 or not where.any():
        return
    painted: npt.NDArray[np.uint8]
    if opacity >= 1:
        painted = colors
    else:
        painted = cv2.addWeighted(colors, opacity, scene, 1 - opacity, 0)
    if where.all():
        scene[...] = painted
        return
    # `np.copyto` runs faster with a mask spelled out per channel than with a
    # broadcast `(H, W, 1)` one, so the mask is repeated over the channels.
    channel_mask = np.repeat(where[..., np.newaxis], scene.shape[2], axis=2)
    np.copyto(scene, painted, where=channel_mask)
