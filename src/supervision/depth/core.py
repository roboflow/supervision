from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from supervision.annotators.core import _iter_mask_crops
from supervision.config import (
    DEPTH_M_DATA_FIELD,
    DISPARITY_PX_DATA_FIELD,
    RELATIVE_INVERSE_DATA_FIELD,
)
from supervision.detection.utils.boxes import clip_boxes

if TYPE_CHECKING:
    from supervision.detection.core import Detections


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


#: The `detections.data` key `DepthMap.measure_detections` writes for each kind.
_DATA_FIELD_BY_KIND: dict[DepthKind, str] = {
    DepthKind.DEPTH_M: DEPTH_M_DATA_FIELD,
    DepthKind.DISPARITY_PX: DISPARITY_PX_DATA_FIELD,
    DepthKind.RELATIVE_INVERSE: RELATIVE_INVERSE_DATA_FIELD,
}


class DepthResizeMethod(Enum):
    """How `sv.DepthMap.resize` picks the value of each target pixel.

    Both methods only pick existing values and never blend them.

    Attributes:
        NEAREST: The pixel under each target pixel's centre.
        FOREGROUND: In each group of source pixels a target pixel covers, the valid
            value nearest the camera, so thin near objects survive shrinking.
    """

    NEAREST = "nearest"
    FOREGROUND = "foreground"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string value of every resize method."""
        return [member.value for member in cls]

    @classmethod
    def from_value(cls, value: DepthResizeMethod | str) -> DepthResizeMethod:
        """Resolve a resize method from an enum member or its case-insensitive value.

        Args:
            value: A `DepthResizeMethod` member or one of its string values.

        Returns:
            The matching `DepthResizeMethod`.

        Raises:
            ValueError: If `value` names no resize method.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"Unknown resize method: {value!r}. Must be one of {cls.list()}."
        )


class DepthScale(Enum):
    """The scale a depth map is coloured on: inverse depth or metric depth.

    An explicit `display_range` stays in the map's own unit whatever the scale.

    Attributes:
        INVERSE: Inverse depth, where larger is nearer: a disparity or relative map
            as it is, and a metric map as 1 / depth (disparity with a camera). It
            spends colour on near detail, the way stereo measures it.
        METRIC: Metric depth in metres, linear in distance. Needs a `depth_m` map,
            or a `disparity_px` map with a camera.
    """

    INVERSE = "inverse"
    METRIC = "metric"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string value of every scale."""
        return [member.value for member in cls]

    @classmethod
    def from_value(cls, value: DepthScale | str) -> DepthScale:
        """Resolve a scale from an enum member or its case-insensitive value.

        Args:
            value: A `DepthScale` member or one of its string values.

        Returns:
            The matching `DepthScale`.

        Raises:
            ValueError: If `value` names no scale.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"Invalid depth scale: {value!r}. Must be one of {cls.list()}."
        )


@dataclass(frozen=True)
class DepthCamera:
    """Pinhole stereo camera parameters that relate disparity to metric depth.

    All pixel values are in pixels of the depth map they belong to, so resizing a
    `sv.DepthMap` updates them.

    Attributes:
        fx_px: Focal length in pixels.
        baseline_m: Stereo baseline in metres.
        doffs_px: Difference between the two views' principal points in pixels
            (Middlebury's `doffs`), so that depth is
            `fx_px * baseline_m / (disparity + doffs_px)`.

    Examples:
        ```pycon
        >>> import supervision as sv
        >>> camera = sv.DepthCamera(fx_px=721.5, baseline_m=0.54)
        >>> round(camera.fx_px * camera.baseline_m / 38.96, 2)
        10.0

        ```
    """

    fx_px: float
    baseline_m: float
    doffs_px: float = 0.0

    def __post_init__(self) -> None:
        """Store plain floats and reject parameters that cannot convert disparity."""
        for name in ("fx_px", "baseline_m", "doffs_px"):
            value = getattr(self, name)
            object.__setattr__(self, name, _plain_float(value, f"DepthCamera {name}"))
        for name in ("fx_px", "baseline_m"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"DepthCamera {name} must be a positive number.")
        if not math.isfinite(self.doffs_px):
            raise ValueError("DepthCamera doffs_px must be a finite number.")

    def _scaled(self, scale_x: float) -> DepthCamera:
        """Return the camera for the map resized to `scale_x` times its width."""
        return replace(
            self, fx_px=self.fx_px * scale_x, doffs_px=self.doffs_px * scale_x
        )


@dataclass(frozen=True)
class _Conversion:
    """How a value in the map kind's unit becomes the ranged or coloured scale.

    `reciprocal ? numerator / (x + inner_offset) + outer_offset : x`. Depth grows
    away from the camera, so for depth the low end of a range is the near, warm end
    (`near_is_low`).
    """

    reciprocal: bool = False
    numerator: float = 1.0
    inner_offset: float = 0.0
    outer_offset: float = 0.0
    near_is_low: bool = False

    def apply(self, values: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Convert an array in its own float precision; x / 0 yields infinities."""
        if not self.reciprocal:
            return values
        dtype = values.dtype.type
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            converted: npt.NDArray[np.floating] = dtype(self.numerator) / (
                values + dtype(self.inner_offset)
            ) + dtype(self.outer_offset)
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


def _resolve_conversion(
    kind: DepthKind, camera: DepthCamera | None, scale: DepthScale
) -> _Conversion:
    """Return the conversion from `kind` values to `scale`, or raise if impossible."""
    focal_baseline = 1.0 if camera is None else camera.fx_px * camera.baseline_m
    doffs = 0.0 if camera is None else camera.doffs_px
    if scale is DepthScale.INVERSE:
        if kind is DepthKind.DEPTH_M:
            # d = fx * B / Z - doffs, or plain inverse depth without a camera.
            return _Conversion(
                reciprocal=True, numerator=focal_baseline, outer_offset=-doffs
            )
        return _Conversion()
    if kind is DepthKind.DEPTH_M:
        return _Conversion(near_is_low=True)
    if kind is DepthKind.DISPARITY_PX and camera is not None:
        # Z = fx * B / (d + doffs)
        return _Conversion(
            reciprocal=True,
            numerator=focal_baseline,
            inner_offset=doffs,
            near_is_low=True,
        )
    reason = (
        "a disparity map needs a DepthCamera for metric depth"
        if kind is DepthKind.DISPARITY_PX
        else "relative inverse depth has no metric scale"
    )
    raise ValueError(f"Cannot use scale 'metric' for a {kind.value} map: {reason}.")


def _plain_float(value: Any, field: str) -> float:
    """Return a Python, NumPy or one-value tensor number as a plain float."""
    if not isinstance(value, (str, bytes, bool, np.bool_)):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    raise ValueError(f"{field} must be a real number, got {value!r}.")


def _holds_depth(
    values: npt.NDArray[np.floating] | np.floating[Any], kind: DepthKind
) -> npt.NDArray[np.bool_]:
    """Return where values of `kind` are depth: finite and `> 0`, or `>= 0` if relative.

    The one statement of the no-depth rule, shared by the full-frame mask, single
    pixels and the regions `DepthMap.measure_detections` gathers.
    """
    lowest_ok = values >= 0 if kind is DepthKind.RELATIVE_INVERSE else values > 0
    return np.asarray(np.isfinite(values) & lowest_ok)


def _index_map(source: int, target: int) -> npt.NDArray[np.intp]:
    """Return the source index under each target pixel centre (nearest sampling).

    Sampling at pixel centres keeps the scaled map aligned with the scene and lets
    the valid-depth mask be resampled as plain booleans, which `sv.resize_image`
    does not offer.
    """
    positions = (np.arange(target, dtype=np.float64) + 0.5) * source / target
    return np.minimum(positions.astype(np.intp), source - 1)


def _pool_axis(keys: npt.NDArray[Any], target: int, axis: int) -> npt.NDArray[Any]:
    """Keep the largest key in each group of source pixels covering a target pixel.

    Source pixel `i` belongs to target pixel `floor(i * target / source)`; the groups
    are contiguous and non-empty when shrinking, so one `reduceat` pools them. A
    growing axis has nothing to pool and samples the nearest pixel instead.
    """
    source = keys.shape[axis]
    if target >= source:
        return np.take(keys, _index_map(source, target), axis=axis)
    groups = (np.arange(source, dtype=np.int64) * target) // source
    starts = np.searchsorted(groups, np.arange(target))
    pooled: npt.NDArray[Any] = np.maximum.reduceat(keys, starts, axis=axis)
    return pooled


def _stretch(
    values: npt.NDArray[np.float32], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Return the values stretched over an image of `resolution_wh`, unscaled.

    Each image pixel takes the map pixel under its centre, as `sv.DepthAnnotator`
    draws the map, so the values keep the map's unit, unlike `DepthMap.resize`,
    which rescales disparity with the width.
    """
    width, height = _check_resolution(resolution_wh)
    source_height, source_width = values.shape
    if (width, height) == (source_width, source_height):
        return values
    stretched: npt.NDArray[np.float32] = values.take(
        _index_map(source_height, height), axis=0
    ).take(_index_map(source_width, width), axis=1)
    return stretched


def _pixel_boxes(
    xyxy: npt.ArrayLike, resolution_wh: tuple[int, int]
) -> npt.NDArray[np.int64]:
    """Return `(N, 4)` boxes as whole pixels inside the map, as `sv.crop_image` does.

    Boxes are clipped in float before the cast, so a huge coordinate cannot overflow
    int64. Coordinates must be finite, as `sv.Detections` already ensures.
    """
    clipped = np.round(clip_boxes(np.asarray(xyxy, dtype=np.float64), resolution_wh))
    pixel_boxes: npt.NDArray[np.int64] = clipped.astype(np.int64)
    return pixel_boxes


def _values_in_box(
    values: npt.NDArray[np.float32], box: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    """Return the values inside a whole-pixel box already clipped to the map."""
    x_min, y_min, x_max, y_max = box
    region: npt.NDArray[np.float32] = values[y_min:y_max, x_min:x_max].ravel()
    return region


def _values_under_mask(
    values: npt.NDArray[np.float32],
    mask: npt.NDArray[Any],
    offset: npt.NDArray[np.int32] | None,
) -> npt.NDArray[np.float32]:
    """Return the values under a full-frame mask, or under a crop placed at `offset`.

    `offset` is the crop's `(x1, y1)` origin in the map, as `CompactMask` gives it.
    """
    if offset is not None:
        x_min, y_min = int(offset[0]), int(offset[1])
        crop_height, crop_width = mask.shape
        values = values[y_min : y_min + crop_height, x_min : x_min + crop_width]
    region: npt.NDArray[np.float32] = values[np.asarray(mask, dtype=bool)]
    return region


def _check_resolution(resolution_wh: tuple[int, int]) -> tuple[int, int]:
    """Return a `(width, height)` pair of positive integers, else raise."""
    message = f"resolution_wh must be two positive integers, got {resolution_wh!r}."
    try:
        width, height = resolution_wh
    except (TypeError, ValueError):
        # None, a scalar or a sequence of another length is not a pair, and would
        # otherwise escape as a TypeError or a generic unpacking error.
        raise ValueError(message) from None
    for side in (width, height):
        # The type is checked before any comparison, so None or infinity raise
        # ValueError here rather than TypeError or OverflowError.
        is_integer = isinstance(side, (int, np.integer)) and not isinstance(side, bool)
        if not (is_integer and side > 0):
            raise ValueError(message)
    return int(width), int(height)


class DepthMap:
    """A per-pixel depth, disparity or relative depth map for one frame.

    `sv.DepthMap` is to depth what `sv.KeyPoints` is to pose: its own container with
    its own annotator ([`sv.DepthAnnotator`](/latest/depth/annotators/)). It belongs
    to the whole frame, so it is not a `sv.Detections` field; `measure_detections`
    brings the depth under each object into `detections.data`.

    `values` are float32 in the kind's unit. For `disparity_px` and `depth_m`,
    non-finite values and values `<= 0` are no depth; for `relative_inverse`,
    non-finite and negative values are no depth, because a normalised map puts its
    farthest real pixel at exactly 0. A disparity of 0 or less stays no depth even
    when `camera.doffs_px > 0` would make a value in `(-doffs_px, 0]` convert to a
    finite distance, so such pixels become `NaN` in metres instead.

    Attributes:
        values: `(H, W)` float32 values in the kind's unit.
        kind: What the values measure.
        camera: Optional stereo camera, needed to convert disparity to metric depth.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> disparity = np.array([[0.0, 10.0], [20.0, 40.0]], dtype=np.float32)
        >>> depth_map = sv.DepthMap(
        ...     disparity,
        ...     kind="disparity_px",
        ...     camera=sv.DepthCamera(fx_px=1000.0, baseline_m=0.1),
        ... )
        >>> depth_map.valid_mask
        array([[False,  True],
               [ True,  True]])
        >>> depth_map.to_depth().to_float()
        array([[ nan, 10. ],
               [ 5. ,  2.5]], dtype=float32)

        ```
    """

    values: npt.NDArray[np.float32]
    kind: DepthKind
    camera: DepthCamera | None

    def __init__(
        self,
        values: npt.NDArray[np.floating],
        kind: DepthKind | str,
        camera: DepthCamera | None = None,
    ) -> None:
        """Create a depth map from float values in the kind's unit.

        Args:
            values: `(H, W)` float array in the kind's unit; any float dtype is
                stored as float32. A float32 array is stored without a copy, so
                later changes to it show in the map. Masked-array masks are
                ignored: mark missing depth with `NaN`, an infinity or a value the
                kind treats as no depth.
            kind: What the values measure, as a `sv.DepthKind` or its string value.
            camera: Optional stereo camera parameters. A `relative_inverse` map has
                no metric scale for a camera to give, so it takes none.

        Raises:
            ValueError: If the values are not a 2D float array, or the camera is
                invalid or given for a `relative_inverse` map.
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
        if camera is not None and not isinstance(camera, DepthCamera):
            raise ValueError("DepthMap camera must be a sv.DepthCamera or None.")
        depth_kind = DepthKind.from_value(kind)
        if camera is not None and depth_kind is DepthKind.RELATIVE_INVERSE:
            raise ValueError(
                "A relative_inverse map has no metric scale, so no DepthCamera can "
                "convert it; pass camera=None."
            )
        self.values = array.astype(np.float32, copy=False)
        self.kind = depth_kind
        self.camera = camera

    def __repr__(self) -> str:
        """Summarise the map without printing its values."""
        width, height = self.resolution_wh
        return (
            f"DepthMap(kind={self.kind.value!r}, resolution_wh=({width}, {height}), "
            f"camera={self.camera})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare every field; values compare NaN equal to NaN."""
        if not isinstance(other, DepthMap):
            return NotImplemented
        return (
            self.kind is other.kind
            and self.camera == other.camera
            and np.array_equal(self.values, other.values, equal_nan=True)
        )

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """The map's `(width, height)` in pixels."""
        height, width = self.values.shape
        return int(width), int(height)

    @property
    def valid_mask(self) -> npt.NDArray[np.bool_]:
        """A boolean `(H, W)` array, `True` where the map holds depth."""
        return _holds_depth(self.values, self.kind)

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

    def to_depth(self) -> DepthMap:
        """Return the map as metric depth in metres.

        A disparity map converts with `fx_px * baseline_m / (disparity + doffs_px)`;
        pixels where that is not a positive distance become no depth. A `depth_m`
        map is returned as it is.

        Returns:
            A `depth_m` map with float32 values.

        Raises:
            ValueError: If the map is `relative_inverse`, or a disparity map without
                a camera.
        """
        if self.kind is DepthKind.DEPTH_M:
            return self
        conversion = _resolve_conversion(self.kind, self.camera, DepthScale.METRIC)
        return self._converted(DepthKind.DEPTH_M, conversion)

    def _converted(self, kind: DepthKind, conversion: _Conversion) -> DepthMap:
        """Apply a reciprocal conversion, keeping only positive finite results."""
        converted = conversion.apply(self.to_float())
        with np.errstate(invalid="ignore"):
            converted[~_holds_depth(converted, kind)] = np.nan
        return DepthMap(converted, kind=kind, camera=self.camera)

    def _region_depths(
        self, region: npt.NDArray[np.float32], conversion: _Conversion | None
    ) -> npt.NDArray[np.float32]:
        """Return a region's depth values, converted to metres by `conversion` if any.

        Keeps the pixels `to_depth().to_float()` would keep: valid in the map, then,
        after the conversion, a positive finite distance.
        """
        depths = region[_holds_depth(region, self.kind)]
        if conversion is not None:
            depths = conversion.apply(depths)
            with np.errstate(invalid="ignore"):
                depths = depths[_holds_depth(depths, DepthKind.DEPTH_M)]
        return depths

    def resize(
        self,
        resolution_wh: tuple[int, int],
        method: DepthResizeMethod | str = DepthResizeMethod.NEAREST,
    ) -> DepthMap:
        """Return the map resized to `resolution_wh` without blending any pixels.

        Blending would mix no-depth pixels into their neighbours and average a
        foreground edge into the background, inventing depths nobody measured, so
        both methods only pick existing values:

        - `"nearest"` takes the pixel under each target pixel's centre.
        - `"foreground"` keeps, in each group of source pixels a target pixel covers,
          the valid value nearest the camera (the largest disparity, the smallest
          depth), so thin near objects survive shrinking and holes never win over
          depth. An axis that grows samples the nearest pixel.

        Disparity is measured in pixels of the map, so a disparity map's values scale
        with the width ratio, and so do the camera's pixel parameters.

        Args:
            resolution_wh: Target `(width, height)`.
            method: A `sv.DepthResizeMethod` or its case-insensitive string value,
                `"nearest"` (default) or `"foreground"`.

        Returns:
            A new `sv.DepthMap`.

        Raises:
            ValueError: If the resolution is not positive or the method is unknown.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> disparity = np.array([[8.0, 0.0, 2.0, 2.0]], dtype=np.float32)
            >>> depth_map = sv.DepthMap(disparity, kind="disparity_px")
            >>> depth_map.resize((2, 1), method="foreground").to_float()
            array([[4., 1.]], dtype=float32)

            ```
        """
        width, height = _check_resolution(resolution_wh)
        resize_method = DepthResizeMethod.from_value(method)
        source_width, source_height = self.resolution_wh
        if resize_method is DepthResizeMethod.NEAREST:
            rows = _index_map(source_height, height)
            columns = _index_map(source_width, width)
            values = self.values.take(rows, axis=0).take(columns, axis=1)
        else:
            values = self._pool_foreground(width, height)
        ratio_x = width / source_width
        if self.kind is DepthKind.DISPARITY_PX:
            values = values * np.float32(ratio_x)
        camera = None if self.camera is None else self.camera._scaled(ratio_x)
        return DepthMap(values, kind=self.kind, camera=camera)

    def _pool_foreground(self, width: int, height: int) -> npt.NDArray[np.float32]:
        """Pool each target pixel's source block to its nearest valid value."""
        source_width, source_height = self.resolution_wh
        if width >= source_width and height >= source_height:
            # Neither axis shrinks, so there is nothing to pool: a nearest take on
            # the values with holes as NaN gives the same map without the full-size
            # key temporaries.
            filled = self.to_float()
            return filled.take(_index_map(source_height, height), axis=0).take(
                _index_map(source_width, width), axis=1
            )
        valid = self.valid_mask
        near_is_low = self.kind is DepthKind.DEPTH_M
        signed = -self.values if near_is_low else self.values
        keys = np.where(valid, signed, -np.inf).astype(np.float32, copy=False)
        pooled = _pool_axis(_pool_axis(keys, height, axis=0), width, axis=1)
        restored = -pooled if near_is_low else pooled
        foreground: npt.NDArray[np.float32] = np.where(
            np.isfinite(pooled), restored, np.nan
        ).astype(np.float32, copy=False)
        return foreground

    def crop(self, xyxy: npt.ArrayLike) -> DepthMap:
        """Return the part of the map inside a box.

        Coordinates are rounded and clipped to the map, as in `sv.crop_image`. Values
        are unchanged.

        Args:
            xyxy: Box as `(x_min, y_min, x_max, y_max)` in map pixels.

        Returns:
            A new `sv.DepthMap`.

        Raises:
            ValueError: If the box is not four finite numbers or does not overlap
                the map.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> depth_map = sv.DepthMap(np.ones((100, 200), np.float32), kind="depth_m")
            >>> depth_map.crop((50, 10, 150, 60)).resolution_wh
            (100, 50)

            ```
        """
        width, height = self.resolution_wh
        box = np.asarray(xyxy, dtype=np.float64).flatten()
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(
                "Crop box must be four finite numbers (x_min, y_min, x_max, y_max), "
                f"got {np.asarray(xyxy).tolist()}."
            )
        pixel_box = _pixel_boxes(box[np.newaxis], (width, height))[0]
        x_min, y_min, x_max, y_max = pixel_box.tolist()
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Crop box {np.asarray(xyxy).tolist()} does not overlap the "
                f"{width}x{height} map."
            )
        return DepthMap(
            self.values[y_min:y_max, x_min:x_max].copy(),
            kind=self.kind,
            camera=self.camera,
        )

    def value_at(
        self,
        x: float,
        y: float,
        resolution_wh: tuple[int, int] | None = None,
    ) -> float | None:
        """Return the value in the kind's unit under a point, or `None`.

        The unit is the map's own: metres for `depth_m`, pixels for `disparity_px`.
        Call `to_depth` first for metres from a disparity map with a camera.
        Missing values are `None` here, but `NaN` in `to_float` and in the columns
        `measure_detections` writes.

        Args:
            x: Column of the point.
            y: Row of the point.
            resolution_wh: Size of the space the point is measured in, normally the
                image's, when the map is stretched over an image of another size.
                Without it the point is in map pixels. The point is a continuous
                coordinate, where pixel `i` spans `[i, i + 1)`: for a pixel index,
                pass its centre, `x + 0.5` and `y + 0.5`, to read the value that
                `resize` and `sv.DepthAnnotator` show at that pixel.

        Returns:
            The value, or `None` where the map has no depth or the point is outside
            the map.

        Raises:
            ValueError: If `resolution_wh` is not two positive integers.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> disparity = np.array([[np.nan, 10.0]], dtype=np.float32)
            >>> depth_map = sv.DepthMap(disparity, kind="disparity_px")
            >>> depth_map.value_at(1, 0), depth_map.value_at(0, 0)
            (10.0, None)
            >>> depth_map.value_at(300, 100, resolution_wh=(400, 200))
            10.0
            >>> metric = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            >>> depth_map = sv.DepthMap(metric, kind="depth_m")
            >>> depth_map.resize((3, 1)).value_at(1, 0)
            3.0
            >>> depth_map.value_at(1 + 0.5, 0 + 0.5, resolution_wh=(3, 1))
            3.0

            ```
        """
        width, height = self.resolution_wh
        scale_x = scale_y = 1.0
        if resolution_wh is not None:
            target_width, target_height = _check_resolution(resolution_wh)
            scale_x, scale_y = width / target_width, height / target_height
        column, row = x * scale_x, y * scale_y
        if not (math.isfinite(column) and math.isfinite(row)):
            return None
        column_index, row_index = math.floor(column), math.floor(row)
        if not (0 <= column_index < width and 0 <= row_index < height):
            return None
        # Only this pixel is checked; `valid_mask` would test the whole frame.
        value = self.values[row_index, column_index]
        if not _holds_depth(value, self.kind):
            return None
        return float(value)

    def measure_detections(
        self,
        detections: Detections,
        resolution_wh: tuple[int, int] | None = None,
    ) -> Detections:
        """Return the detections with the median depth under each object in `data`.

        For each detection, the median of the valid depth pixels inside its mask
        (dense or `sv.CompactMask`) or, without masks, inside its box (rounded and
        clipped as in `sv.crop_image`) is stored in a new `data` column. The depth is
        in metres, under `"depth_m"`, when the map can give them (a `depth_m` map, or
        a disparity map with a camera), and in the map's own kind otherwise, under
        `"disparity_px"` or `"relative_inverse"`
        (`supervision.config.DEPTH_M_DATA_FIELD` and its siblings), so a disparity
        map without a camera never writes `"depth_m"`. A column of the same name
        already in `data` is replaced. Objects whose
        region holds no depth get `NaN`.

        Pass `resolution_wh` when the detections come from an image of another size
        than the map, such as a monocular model's output: the map is then stretched
        over the image as `sv.DepthAnnotator` draws it, each image pixel taking the
        map value under its centre, so even a small box gets depth. The values keep
        the map's own unit; unlike `resize`, a disparity map's values are not
        rescaled.

        Args:
            detections: Detections in the map's pixel coordinates, or in the
                coordinates of an image of `resolution_wh`.
            resolution_wh: Size of the image the detections and their masks belong
                to, when it differs from the map's. Without it they are in map
                pixels.

        Returns:
            A new `sv.Detections` with the added `data` column, float32 of shape
            `(N,)`. It shares the input's arrays (`xyxy`, `mask` and the other `data`
            columns) rather than copying them.

        Raises:
            ValueError: If `resolution_wh` is not two positive integers, or the masks
                do not match the map's size or `resolution_wh`.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> depth = np.full((100, 100), 20.0, dtype=np.float32)
            >>> depth[20:40, 20:40] = 4.0
            >>> depth_map = sv.DepthMap(depth, kind="depth_m")
            >>> detections = sv.Detections(
            ...     xyxy=np.array([[20, 20, 40, 40], [60, 60, 90, 90]], dtype=float)
            ... )
            >>> detections = depth_map.measure_detections(detections)
            >>> detections.data["depth_m"]
            array([ 4., 20.], dtype=float32)
            >>> labels = [f"{d:.1f} m" for d in detections.data["depth_m"]]

            ```
        """
        # Only the gathered pixels are converted to metres, not the whole frame;
        # the conversion is elementwise, so the medians match `to_depth()` exactly.
        conversion = None
        kind = self.kind
        if kind is DepthKind.DISPARITY_PX and self.camera is not None:
            conversion = _resolve_conversion(kind, self.camera, DepthScale.METRIC)
            kind = DepthKind.DEPTH_M
        values = self.values
        if resolution_wh is not None:
            values = _stretch(values, resolution_wh)
        height, width = values.shape
        medians = np.full(len(detections), np.nan, dtype=np.float32)
        mask = detections.mask
        if mask is not None:
            mask_shape = mask.shape[1:]
            if tuple(mask_shape) != (height, width):
                raise ValueError(
                    f"Detection masks are {mask_shape[1]}x{mask_shape[0]} but the "
                    f"depth map is measured at {width}x{height}; pass the masks' "
                    "image size as resolution_wh, or resize the map with "
                    "depth_map.resize(...) first."
                )
        # A CompactMask yields each crop with its origin, so no full-frame mask is
        # decoded per detection.
        regions = (
            (
                _values_in_box(values, box)
                for box in _pixel_boxes(detections.xyxy, (width, height))
            )
            if mask is None
            else (
                _values_under_mask(values, crop, offset)
                for _, crop, offset in _iter_mask_crops(detections)
            )
        )
        for index, region in enumerate(regions):
            depths = self._region_depths(region, conversion)
            if depths.size:
                medians[index] = np.median(depths)
        field = _DATA_FIELD_BY_KIND[kind]
        return replace(detections, data={**detections.data, field: medians})
