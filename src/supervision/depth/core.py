from __future__ import annotations

import base64
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision.config import (
    DEPTH_M_DATA_FIELD,
    DISPARITY_PX_DATA_FIELD,
    RELATIVE_INVERSE_DATA_FIELD,
)
from supervision.depth.readers import read_pfm, read_png16
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections

#: A frame adds about this many samples, or fewer, to a clip-wide percentile estimate.
_CLIP_SAMPLES_PER_FRAME = 65536
#: A clip-wide percentile estimate keeps at most about this many float64 samples.
_CLIP_SAMPLE_BUDGET = 1 << 22
_UINT16_MAX = 65535


class DepthKind(Enum):
    """What the values of a `sv.DepthMap` measure.

    Attributes:
        DISPARITY_PX: Stereo disparity in pixels of the map. Larger is nearer.
            Metric depth is `fx_px * baseline_m / (disparity + doffs_px)`.
        DEPTH_M: Metric depth along the optical axis, in metres. Smaller is nearer.
        RELATIVE_INVERSE: Unitless relative depth from a monocular model,
            normalised so larger is nearer, with no metric scale. Roboflow
            Inference's maps run from 0 for the farthest pixel to 1 for the
            nearest. Depth Anything V1 and V2 and DPT output is inverse depth up to
            an unknown scale and shift; Depth Anything V3 output is linear in depth.
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


class DepthQuantity(Enum):
    """The quantity a depth map is coloured or ranged by.

    Attributes:
        DISPARITY: Disparity, or inverse depth for a metric map without a camera. It
            spends colour on near detail, the way stereo measures it.
        DEPTH: Metric depth in metres. Needs a `depth_m` map, or a `disparity_px`
            map with a camera.
    """

    DISPARITY = "disparity"
    DEPTH = "depth"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string value of every quantity."""
        return [member.value for member in cls]

    @classmethod
    def from_value(cls, value: DepthQuantity | str) -> DepthQuantity:
        """Resolve a quantity from an enum member or its case-insensitive value.

        Args:
            value: A `DepthQuantity` member or one of its string values.

        Returns:
            The matching `DepthQuantity`.

        Raises:
            ValueError: If `value` names no quantity.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"Invalid depth quantity: {value!r}. Must be one of {cls.list()}."
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
        return DepthCamera(
            fx_px=self.fx_px * scale_x,
            baseline_m=self.baseline_m,
            doffs_px=self.doffs_px * scale_x,
        )


@dataclass(frozen=True)
class _Conversion:
    """How a value in the map kind's unit becomes the ranged or coloured quantity.

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
        with np.errstate(divide="ignore", invalid="ignore"):
            converted: npt.NDArray[np.floating] = dtype(self.numerator) / (
                values + dtype(self.inner_offset)
            ) + dtype(self.outer_offset)
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


def _resolve_conversion(
    kind: DepthKind, camera: DepthCamera | None, quantity: DepthQuantity
) -> _Conversion:
    """Return the conversion from `kind` values to `quantity`, or raise if
    impossible."""
    focal_baseline = 1.0 if camera is None else camera.fx_px * camera.baseline_m
    doffs = 0.0 if camera is None else camera.doffs_px
    if quantity is DepthQuantity.DISPARITY:
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
    raise ValueError(f"Cannot use quantity 'depth' for a {kind.value} map: {reason}.")


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


def _check_resolution(resolution_wh: tuple[int, int]) -> tuple[int, int]:
    """Return a `(width, height)` pair of positive integers, else raise."""
    width, height = resolution_wh
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError(
            f"resolution_wh must be two positive integers, got {resolution_wh}."
        )
    return int(width), int(height)


def _to_numpy(value: Any) -> npt.NDArray[np.float32]:
    """Turn a framework tensor or array-like into a float32 NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and not isinstance(value, np.ndarray):
        # bfloat16 and half tensors have no NumPy view; widen them first.
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _squeeze_to_2d(
    values: npt.NDArray[np.float32], source: str
) -> npt.NDArray[np.float32]:
    """Drop leading unit axes, as in `(1, H, W)` model outputs, and require 2D."""
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(
            f"{source} depth must be a single (H, W) map, got {values.shape}."
        )
    return values


class DepthMap:
    """A per-pixel depth, disparity or relative depth map for one frame.

    `sv.DepthMap` is to depth what `sv.KeyPoints` is to pose: its own container with
    its own annotator ([`sv.DepthAnnotator`](/latest/depth/annotators/)), its own
    model connectors. It belongs to the whole frame, so it is not a `sv.Detections`
    field; `measure_detections` brings the depth under each object into
    `detections.data`.

    `values` are float32 in the kind's unit. For `disparity_px` and `depth_m`,
    non-finite values and values `<= 0` are no depth; for `relative_inverse`,
    non-finite and negative values are no depth, because a normalised map puts its
    farthest real pixel at exactly 0.

    === "Inference"

        Roboflow depth models return a normalised map where 1 is nearest; it loads as
        `relative_inverse`.

        ```python
        import supervision as sv
        from inference import get_model

        model = get_model(model_id="depth-anything-v3/small")
        result = model.infer("<SOURCE_IMAGE_PATH>")[0]
        depth_map = sv.DepthMap.from_inference(result)
        ```

    === "Ultralytics"

        YOLO26 depth models predict metric depth in metres.

        ```python
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO("yolo26n-depth.pt")
        result = model("<SOURCE_IMAGE_PATH>")[0]
        depth_map = sv.DepthMap.from_ultralytics(result)
        ```

    === "Transformers"

        The `depth-estimation` pipeline returns whatever the model predicts, so name
        the kind: `relative_inverse` for Depth Anything, `depth_m` for metric models.

        ```python
        import supervision as sv
        from transformers import pipeline

        estimator = pipeline(
            "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
        )
        result = estimator("<SOURCE_IMAGE_PATH>")
        depth_map = sv.DepthMap.from_transformers(result, kind="relative_inverse")
        ```

    === "Datasets"

        ```python
        import supervision as sv

        kitti = sv.DepthMap.from_png16(
            "disp_occ_0/000000_10.png", scale=256, kind="disparity_px"
        )
        middlebury = sv.DepthMap.from_pfm("disp0.pfm")
        ```

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
                stored as float32.
            kind: What the values measure, as a `sv.DepthKind` or its string value.
            camera: Optional stereo camera parameters.

        Raises:
            ValueError: If the values are not a 2D float array or the camera is
                invalid.
        """
        array = np.asarray(values)
        if array.ndim != 2 or array.size == 0:
            raise ValueError(
                f"DepthMap values must be a non-empty (H, W) array, got {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(
                f"DepthMap values must be float, got {array.dtype}. Divide integer "
                "codes, such as a 16-bit PNG's, by your dataset's scale first, or "
                "load the file with sv.DepthMap.from_png16."
            )
        if camera is not None and not isinstance(camera, DepthCamera):
            raise ValueError("DepthMap camera must be a sv.DepthCamera or None.")
        self.values = array.astype(np.float32, copy=False)
        self.kind = DepthKind.from_value(kind)
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
        conversion = _resolve_conversion(self.kind, self.camera, DepthQuantity.DEPTH)
        return self._converted(DepthKind.DEPTH_M, conversion)

    def _converted(self, kind: DepthKind, conversion: _Conversion) -> DepthMap:
        """Apply a reciprocal conversion, keeping only positive finite results."""
        converted = conversion.apply(self.to_float())
        with np.errstate(invalid="ignore"):
            converted[~(np.isfinite(converted) & (converted > 0))] = np.nan
        return DepthMap(converted, kind=kind, camera=self.camera)

    def resize(
        self, resolution_wh: tuple[int, int], method: str = "nearest"
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
            method: `"nearest"` or `"foreground"`.

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
        if method not in ("nearest", "foreground"):
            raise ValueError(
                f"Unknown resize method {method!r}; use 'nearest' or 'foreground'."
            )
        source_width, source_height = self.resolution_wh
        if method == "nearest":
            rows = _index_map(source_height, height)
            columns = _index_map(source_width, width)
            values = self.values[np.ix_(rows, columns)]
        else:
            values = self._pool_foreground(width, height)
        ratio_x = width / source_width
        if self.kind is DepthKind.DISPARITY_PX:
            values = values * np.float32(ratio_x)
        camera = None if self.camera is None else self.camera._scaled(ratio_x)
        return DepthMap(values, kind=self.kind, camera=camera)

    def _pool_foreground(self, width: int, height: int) -> npt.NDArray[np.float32]:
        """Pool each target pixel's source block to its nearest valid value."""
        valid = self.valid_mask
        near_is_low = self.kind is DepthKind.DEPTH_M
        signed = -self.values if near_is_low else self.values
        keys = np.where(valid, signed, -np.inf).astype(np.float32)
        pooled = _pool_axis(_pool_axis(keys, height, axis=0), width, axis=1)
        restored = -pooled if near_is_low else pooled
        return np.where(np.isfinite(pooled), restored, np.nan).astype(np.float32)

    def crop(self, xyxy: npt.ArrayLike) -> DepthMap:
        """Return the part of the map inside a box.

        Coordinates are rounded and clipped to the map, as in `sv.crop_image`. Values
        are unchanged.

        Args:
            xyxy: Box as `(x_min, y_min, x_max, y_max)` in map pixels.

        Returns:
            A new `sv.DepthMap`.

        Raises:
            ValueError: If the box does not overlap the map.

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
        x_min, y_min, x_max, y_max = (
            np.asarray(xyxy, dtype=np.float64).round().astype(np.int64).flatten()
        )
        x_min, x_max = int(np.clip(x_min, 0, width)), int(np.clip(x_max, 0, width))
        y_min, y_max = int(np.clip(y_min, 0, height)), int(np.clip(y_max, 0, height))
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

        Args:
            x: Column of the point.
            y: Row of the point.
            resolution_wh: Size of the space the point is measured in, normally the
                image's, when the map is stretched over an image of another size.
                Without it the point is in map pixels.

        Returns:
            The value, or `None` where the map has no depth or the point is outside
            the map.

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

            ```
        """
        width, height = self.resolution_wh
        scale_x = 1.0 if resolution_wh is None else width / resolution_wh[0]
        scale_y = 1.0 if resolution_wh is None else height / resolution_wh[1]
        column, row = x * scale_x, y * scale_y
        if not (math.isfinite(column) and math.isfinite(row)):
            return None
        column_index, row_index = math.floor(column), math.floor(row)
        if not (0 <= column_index < width and 0 <= row_index < height):
            return None
        if not self.valid_mask[row_index, column_index]:
            return None
        return float(self.values[row_index, column_index])

    def _percentile_range(
        self,
        low: float = 2.0,
        high: float = 98.0,
        quantity: DepthQuantity | str = DepthQuantity.DISPARITY,
    ) -> tuple[float, float] | None:
        """Return this map's own percentile range in the quantity's unit.

        It is the range `sv.DepthAnnotator` uses with `display_range="auto"`: the
        nearest-rank percentiles of every valid value, converted to the quantity.

        Args:
            low: Lower percentile, from 0 to 100.
            high: Upper percentile, from 0 to 100.
            quantity: `"disparity"` or `"depth"`.

        Returns:
            `(low, high)` in the quantity's unit, or `None` when no pixel holds depth.

        Raises:
            ValueError: If the percentiles are out of order or the quantity is
                impossible for this map.
        """
        _check_percentiles(low, high)
        conversion = _resolve_conversion(
            self.kind, self.camera, DepthQuantity.from_value(quantity)
        )
        values = self.values[self.valid_mask]
        if values.size == 0:
            return None
        return conversion.apply_range(_values_at_ranks(values, low, high)) or (0.0, 1.0)

    def measure_detections(self, detections: Detections) -> Detections:
        """Return the detections with the median depth under each object in `data`.

        For each detection, the median of the valid depth pixels inside its mask
        (dense or `sv.CompactMask`) or, without masks, inside its box (rounded and
        clipped as in `sv.crop_image`) is stored in a new `data` column. The depth is
        in metres, under `"depth_m"`, when the map can give them (a `depth_m` map, or
        a disparity map with a camera), and in the map's own kind otherwise, under
        `"disparity_px"` or `"relative_inverse"`
        (`supervision.config.DEPTH_M_DATA_FIELD` and its siblings). Objects whose
        region holds no depth get `NaN`. The map must have the detections' image
        size; resize it first otherwise.

        Args:
            detections: Detections in the map's pixel coordinates.

        Returns:
            A copy of `detections` with the new `data` column, float32 of shape
            `(N,)`.

        Raises:
            ValueError: If the masks do not match the map's size.

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
        measured = (
            self.to_depth()
            if self.kind is DepthKind.DISPARITY_PX and self.camera is not None
            else self
        )
        values = measured.to_float()
        height, width = values.shape
        medians = np.full(len(detections), np.nan, dtype=np.float32)
        mask = detections.mask
        if mask is not None:
            mask_shape = (
                mask.image_shape if isinstance(mask, CompactMask) else mask.shape[1:]
            )
            if tuple(mask_shape) != (height, width):
                raise ValueError(
                    f"Detection masks are {mask_shape[1]}x{mask_shape[0]} but the "
                    f"depth map is {width}x{height}; resize the map with "
                    "depth_map.resize(...) first."
                )
        for index in range(len(detections)):
            if isinstance(mask, CompactMask):
                crop = mask.crop(index)
                x_offset, y_offset = (int(v) for v in mask.offsets[index])
                region = values[
                    y_offset : y_offset + crop.shape[0],
                    x_offset : x_offset + crop.shape[1],
                ][crop]
            elif mask is not None:
                region = values[np.asarray(mask[index], dtype=bool)]
            else:
                x_min, y_min, x_max, y_max = (
                    detections.xyxy[index].round().astype(np.int64)
                )
                region = values[
                    max(y_min, 0) : max(y_max, 0), max(x_min, 0) : max(x_max, 0)
                ].ravel()
            region = region[np.isfinite(region)]
            if region.size:
                medians[index] = np.median(region)
        field = {
            DepthKind.DEPTH_M: DEPTH_M_DATA_FIELD,
            DepthKind.DISPARITY_PX: DISPARITY_PX_DATA_FIELD,
            DepthKind.RELATIVE_INVERSE: RELATIVE_INVERSE_DATA_FIELD,
        }[measured.kind]
        return replace(detections, data={**detections.data, field: medians})

    @classmethod
    def from_inference(cls, inference_result: Any) -> DepthMap:
        """Create a `sv.DepthMap` from a Roboflow depth estimation result.

        Accepts the `normalized_depth` of every `depth_map_format` (`json` nested
        lists, `png16` or `png8` base64 PNGs, or the NumPy array the Inference SDK
        and Workflows decode them to). The map is normalised per image with 1 for
        the nearest pixel and 0 for the farthest, so larger is nearer (Depth Anything
        V3 maps are linear in depth, not inverse depth). It loads as
        `relative_inverse` float32; values from different images are not comparable.

        Args:
            inference_result: One result from Inference, the Inference SDK or a
                Workflow's depth estimation step.

        Returns:
            A `relative_inverse` `sv.DepthMap`.

        Raises:
            ValueError: If the result is a list, carries no `normalized_depth`, or
                holds a PNG that is not 8- or 16-bit grayscale.

        Examples:
            ```python
            import supervision as sv
            from inference_sdk import InferenceHTTPClient

            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com", api_key="<ROBOFLOW_API_KEY>"
            )
            result = client.depth_estimation(
                "<SOURCE_IMAGE_PATH>",
                model_id="depth-anything-v3/small",
                depth_map_format="png16",
            )
            depth_map = sv.DepthMap.from_inference(result)
            ```
        """
        if isinstance(inference_result, list):
            raise ValueError(
                "from_inference() operates on a single result at a time. You can "
                "retrieve it like so: inference_result = model.infer(image)[0]"
            )
        # In-process models return an LMMInferenceResponse holding the depth in a
        # `response` dict; Inference's DepthEstimationResponse holds it as a field.
        response = getattr(inference_result, "response", None)
        if isinstance(response, dict):
            inference_result = response
        if isinstance(inference_result, dict):
            normalized = inference_result.get("normalized_depth")
        else:
            normalized = getattr(inference_result, "normalized_depth", None)
        if normalized is None:
            raise ValueError(
                "The inference result has no 'normalized_depth'; pass the result of "
                "a depth estimation model."
            )
        if isinstance(normalized, str):
            values = _decode_normalized_png(normalized)
        else:
            values = np.asarray(normalized, dtype=np.float32)
        return cls(_squeeze_to_2d(values, "Inference"), kind=DepthKind.RELATIVE_INVERSE)

    @classmethod
    def from_ultralytics(cls, ultralytics_results: Any) -> DepthMap:
        """Create a `sv.DepthMap` from an Ultralytics YOLO26 depth result.

        `result.depth.data` holds metric depth in metres at the original image size,
        with 0 where the model has no depth.

        Args:
            ultralytics_results: One `Results` object from a depth model.

        Returns:
            A `depth_m` `sv.DepthMap`.

        Raises:
            ValueError: If the result carries no depth map.

        Examples:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO("yolo26n-depth.pt")
            result = model("<SOURCE_IMAGE_PATH>")[0]
            depth_map = sv.DepthMap.from_ultralytics(result)
            ```
        """
        depth = getattr(ultralytics_results, "depth", None)
        if depth is None:
            raise ValueError(
                "The Ultralytics result has no depth map; run a depth model such as "
                "yolo26n-depth.pt."
            )
        values = _to_numpy(getattr(depth, "data", depth))
        return cls(_squeeze_to_2d(values, "Ultralytics"), kind=DepthKind.DEPTH_M)

    @classmethod
    def from_transformers(
        cls, transformers_results: Any, *, kind: DepthKind | str
    ) -> DepthMap:
        """Create a `sv.DepthMap` from a Hugging Face depth estimation result.

        Reads `predicted_depth` from a `depth-estimation` pipeline result, from a
        `post_process_depth_estimation` entry, or from a model output. The pipeline's
        `depth` image is an 8-bit per-image stretch for display and is ignored.
        Transformers does not say what the model predicts, so `kind` is required:
        `relative_inverse` for Depth Anything and DPT relative models, `depth_m` for
        metric models such as Depth Pro, ZoeDepth or Depth Anything metric.

        Args:
            transformers_results: One result holding `predicted_depth` of shape
                `(H, W)` or `(1, H, W)`.
            kind: What the model predicts.

        Returns:
            A float32 `sv.DepthMap`.

        Raises:
            ValueError: If the result is a list or has no `predicted_depth`.

        Examples:
            ```python
            import supervision as sv
            from transformers import pipeline

            estimator = pipeline(
                "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
            )
            result = estimator("<SOURCE_IMAGE_PATH>")
            depth_map = sv.DepthMap.from_transformers(result, kind="relative_inverse")
            ```
        """
        if isinstance(transformers_results, list):
            raise ValueError(
                "from_transformers() operates on a single result at a time; pass "
                "results[0]."
            )
        if isinstance(transformers_results, dict):
            predicted = transformers_results.get("predicted_depth")
        else:
            predicted = getattr(transformers_results, "predicted_depth", None)
        if predicted is None:
            raise ValueError("The transformers result has no 'predicted_depth'.")
        values = _squeeze_to_2d(_to_numpy(predicted), "Transformers")
        return cls(values, kind=kind)

    @classmethod
    def from_png16(
        cls, path: str | Path, scale: float, kind: DepthKind | str
    ) -> DepthMap:
        """Load a 16-bit PNG whose value divided by `scale` is the map, 0 for none.

        This is the layout of KITTI stereo and depth (`scale=256`), DrivingStereo
        (256, or 128 at full resolution), InStereo2K (100) and Ultralytics depth
        datasets (1000 by default). Values load as float32 `code / scale`, with `NaN`
        where the code is 0.

        Args:
            path: Path to the PNG.
            scale: Divisor from stored value to the kind's unit.
            kind: What the values measure.

        Returns:
            A float32 `sv.DepthMap`.

        Raises:
            ValueError: If the file is not a single-channel 16-bit PNG or `scale` is
                not a positive number.

        Examples:
            ```python
            import supervision as sv

            depth_map = sv.DepthMap.from_png16(
                "disp_occ_0/000000_10.png", scale=256, kind="disparity_px"
            )
            ```
        """
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(
                f"from_png16 scale must be a positive number, got {scale}."
            )
        codes = read_png16(path)
        values = codes.astype(np.float32) / np.float32(scale)
        values[codes == 0] = np.nan
        return cls(values, kind=kind)

    @classmethod
    def from_pfm(
        cls, path: str | Path, kind: DepthKind | str = DepthKind.DISPARITY_PX
    ) -> DepthMap:
        """Load a single-channel PFM file, the Middlebury and SceneFlow format.

        Rows are flipped to top first. Infinite values (Middlebury's unknown
        disparity) are no depth.

        Args:
            path: Path to the `.pfm` file.
            kind: What the values measure; disparity by default.

        Returns:
            A float32 `sv.DepthMap`.

        Raises:
            ValueError: If the file is not a grayscale PFM.

        Examples:
            ```python
            import supervision as sv

            depth_map = sv.DepthMap.from_pfm("Adirondack/disp0.pfm")
            ```
        """
        return cls(read_pfm(path), kind=kind)


def _decode_normalized_png(payload: str) -> npt.NDArray[np.float32]:
    """Decode a base64 8- or 16-bit grayscale PNG into floats from 0 to 1."""
    from PIL import Image

    data = base64.b64decode(payload, validate=True)
    try:
        with Image.open(BytesIO(data)) as image:
            mode = image.mode
            values = np.asarray(image)
    except OSError as error:
        raise ValueError("normalized_depth is not a decodable PNG.") from error
    if mode == "L":
        top = np.float32(255)
    elif mode in {"I", "I;16", "I;16B", "I;16L"}:
        top = np.float32(_UINT16_MAX)
    else:
        raise ValueError(
            f"normalized_depth must be an 8- or 16-bit grayscale PNG, got {mode}."
        )
    normalized: npt.NDArray[np.float32] = values.astype(np.float32) / top
    return normalized


@dataclass(frozen=True)
class DepthClipRange:
    """One colour range for a whole clip of depth maps.

    Colouring each frame with its own range makes a still wall change colour as
    things enter and leave the frame. A clip range, computed in a first pass over the
    clip, keeps colours meaning the same distance on every frame.

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
