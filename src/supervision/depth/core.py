from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision.depth.readers import _UINT16_MAX, _read_pfm, _read_png16


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
    field.

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
                "codes, such as a 16-bit PNG's, by your dataset's scale first, or "
                "load the file with sv.DepthMap.from_png16."
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
        codes = _read_png16(path)
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
        return cls(_read_pfm(path), kind=kind)


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
