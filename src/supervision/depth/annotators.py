from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from supervision.depth.colormaps import DepthColormap, _colorize
from supervision.depth.core import (
    DepthClipRange,
    DepthMap,
    DepthQuantity,
    _Conversion,
    _index_map,
    _resolve_conversion,
)
from supervision.draw.base import ImageType
from supervision.draw.color import Color
from supervision.utils.conversion import ensure_cv2_image_for_class_method

_FALLBACK_RANGE = (0.0, 1.0)
_DISPLAY_RANGE_MODES = ("clip", "auto")


class DepthAnnotator:
    """Colours a `sv.DepthMap` over an image, near objects warm and far ones cool.

    The options and the colours match supervision-js's depth renderer, so a map looks
    the same in Python and in the browser: the same 256-entry colour tables, the same
    quantity and range rules, and pixels without depth left unpainted.

    === "Image"

        ```python
        import supervision as sv
        from inference import get_model
        from supervision import _cv2 as cv2

        image = cv2.imread("<SOURCE_IMAGE_PATH>")
        model = get_model(model_id="depth-anything-v3/small")
        depth_map = sv.DepthMap.from_inference(model.infer(image)[0])

        depth_annotator = sv.DepthAnnotator(display_range="auto", opacity=0.6)
        annotated_image = depth_annotator.annotate(image.copy(), depth_map)
        ```

    === "Video with one range"

        ```python
        import supervision as sv

        depth_maps = [sv.DepthMap.load("clip/depth.json", i) for i in range(192)]
        clip_range = sv.DepthClipRange.from_depth_maps(depth_maps)
        depth_annotator = sv.DepthAnnotator(display_range=clip_range)
        ```
    """

    def __init__(
        self,
        colormap: DepthColormap | str = DepthColormap.TURBO,
        quantity: DepthQuantity | str = DepthQuantity.DISPARITY,
        display_range: str | tuple[float, float] | DepthClipRange = "clip",
        opacity: float = 1.0,
        no_depth_color: Color | None = None,
    ) -> None:
        """
        Args:
            colormap: Colour table: `"turbo"` (default), `"viridis"`, `"cividis"`,
                `"inferno"`, `"magma"` or `"grayscale"`. The near end is always the
                warm or bright end.
            quantity: `"disparity"` (default) colours disparity, or inverse depth for a
                metric map, which spends colour on near detail the way stereo
                measures it. `"depth"` colours metres; it needs a `depth_m` map or a
                disparity map with a camera.
            display_range: The values the colour table spans; values outside clamp
                to its ends.
                `"clip"` (default) uses the map's own `display_range`, converted to
                the quantity, and behaves as `"auto"` for a map without one.
                `"auto"` uses this map's 2nd to 98th percentile.
                A `(low, high)` tuple fixes the range in the quantity's unit (pixels
                for disparity, metres for depth).
                A `sv.DepthClipRange` uses one range, in the maps' kind unit, for
                every frame of a clip.
            opacity: Opacity of the colours over the scene, from 0 to 1.
            no_depth_color: Colour for pixels without depth, painted at `opacity`.
                `None` (default) leaves them unpainted, so the scene shows through.

        Raises:
            ValueError: If an option is out of range.
        """
        self.colormap = DepthColormap.from_value(colormap)
        self.quantity = DepthQuantity.from_value(quantity)
        self.display_range = _check_display_range_option(display_range)
        if not (math.isfinite(opacity) and 0 <= opacity <= 1):
            raise ValueError(f"opacity must be between 0 and 1, got {opacity}.")
        self.opacity = float(opacity)
        self.no_depth_color = no_depth_color

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, depth_map: DepthMap) -> ImageType:
        """Colour the depth map over the scene.

        A map whose size differs from the scene's is stretched over the whole scene,
        each scene pixel showing the map pixel under its centre, so a smaller map
        with the scene's aspect ratio lines up.

        Args:
            scene: The image to draw on, a 3-channel `numpy.ndarray` (BGR) or a
                `PIL.Image.Image`. It is drawn on in place.
            depth_map: The depth map to colour.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray` or
                `PIL.Image.Image`).

        Raises:
            ValueError: If the quantity is impossible for this map, for example
                `"depth"` for a relative map.
            TypeError: If `scene` is not a `numpy.ndarray` or `PIL.Image.Image`.

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
        if not isinstance(scene, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(scene)}")
        if scene.ndim != 3 or scene.shape[2] != 3:
            raise ValueError(
                f"DepthAnnotator draws on 3-channel images, got shape {scene.shape}."
            )
        conversion = _resolve_conversion(
            depth_map.kind, depth_map.camera, self.quantity
        )
        low, high = self._resolve_range(depth_map, conversion)
        coordinates = _color_coordinates(depth_map, conversion, low, high)
        colors = _colorize(coordinates, self.colormap)
        valid = depth_map.valid_mask

        scene_height, scene_width = scene.shape[:2]
        map_width, map_height = depth_map.resolution_wh
        if (map_height, map_width) != (scene_height, scene_width):
            rows = _index_map(map_height, scene_height)[:, np.newaxis]
            columns = _index_map(map_width, scene_width)[np.newaxis, :]
            colors = colors[rows, columns]
            valid = valid[rows, columns]

        _blend(scene, valid, colors, self.opacity)
        if self.no_depth_color is not None:
            no_depth = np.array(self.no_depth_color.as_bgr(), dtype=np.uint8)
            _blend(scene, ~valid, no_depth, self.opacity)
        return scene

    def _resolve_range(
        self, depth_map: DepthMap, conversion: _Conversion
    ) -> tuple[float, float]:
        """Return the colour range in the quantity's unit for this map.

        Falls back step by step like supervision-js: a clip range that cannot be
        converted to the percentile range, and that, when too few pixels hold depth,
        to the full range of the valid values, so a sparse frame still colours what
        it has.
        """
        option = self.display_range
        if isinstance(option, tuple):
            return option
        clip_range = None
        if isinstance(option, DepthClipRange):
            clip_range = option.display_range
        elif option == "clip":
            clip_range = depth_map.display_range
        if clip_range is not None:
            converted = conversion.apply_range(clip_range)
            if converted is not None:
                return converted
        percentile = depth_map.percentile_range(quantity=self.quantity)
        if percentile is not None:
            return percentile
        full = depth_map._full_range(conversion)
        return full if full is not None else _FALLBACK_RANGE


def _check_display_range_option(
    display_range: str | tuple[float, float] | DepthClipRange,
) -> str | tuple[float, float] | DepthClipRange:
    """Validate the annotator's `display_range` option and normalise tuples."""
    if isinstance(display_range, DepthClipRange):
        return display_range
    if isinstance(display_range, str):
        if display_range not in _DISPLAY_RANGE_MODES:
            raise ValueError(
                f"display_range must be 'clip', 'auto', a (low, high) tuple or a "
                f"sv.DepthClipRange, got {display_range!r}."
            )
        return display_range
    low, high = (float(bound) for bound in display_range)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(
            "display_range must have finite bounds with low < high, got "
            f"{display_range}."
        )
    return low, high


def _color_coordinates(
    depth_map: DepthMap, conversion: _Conversion, low: float, high: float
) -> npt.NDArray[np.floating]:
    """Return each pixel's colour coordinate in `[0, 1]`, 1 at the near end.

    Mirrors the supervision-js shader: `t = clamp((v - low) / (high - low))`, flipped
    when the quantity's low end is near. Pixels without depth get 0; they are not
    painted.
    """
    converted = conversion.apply(depth_map.to_float())
    span = np.float32(max(high - low, 1e-20))
    with np.errstate(invalid="ignore"):
        coordinates: npt.NDArray[np.floating] = (converted - np.float32(low)) / span
    np.clip(coordinates, 0.0, 1.0, out=coordinates)
    if conversion.near_is_low:
        np.subtract(1.0, coordinates, out=coordinates)
    coordinates[np.isnan(coordinates)] = 0.0
    return coordinates


def _blend(
    scene: npt.NDArray[np.uint8],
    where: npt.NDArray[np.bool_],
    colors: npt.NDArray[np.uint8],
    opacity: float,
) -> None:
    """Blend `colors` into `scene` at `opacity`, only where `where` is set.

    `colors` is `(H, W, 3)` or a single `(3,)` colour. Whole-image arithmetic and a
    masked copy are several times faster than gathering and scattering the masked
    pixels.
    """
    if opacity <= 0 or not where.any():
        return
    mask = where[..., np.newaxis]
    if opacity >= 1:
        np.copyto(scene, colors, where=mask)
        return
    blended = scene.astype(np.float32)
    blended *= np.float32(1 - opacity)
    blended += colors.astype(np.float32) * np.float32(opacity)
    np.copyto(scene, np.rint(blended).astype(np.uint8), where=mask)
