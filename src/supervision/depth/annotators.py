from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from supervision.depth.colormaps import DepthColormap, _colorize
from supervision.depth.core import (
    DepthMap,
    DepthQuantity,
    _Conversion,
    _index_map,
    _resolve_conversion,
)
from supervision.draw.base import ImageType
from supervision.utils.conversion import ensure_cv2_image_for_class_method

_FALLBACK_RANGE = (0.0, 1.0)


class DepthAnnotator:
    """Colours a `sv.DepthMap` over an image, near objects warm and far ones cool.

    Pixels without depth are left unpainted, so the scene shows through.

    === "Image"

        ```python
        import numpy as np
        import supervision as sv
        from supervision import _cv2 as cv2

        image = cv2.imread("<SOURCE_IMAGE_PATH>")
        depth_map = sv.DepthMap(np.load("<DEPTH_NPY_PATH>"), kind="depth_m")

        depth_annotator = sv.DepthAnnotator(display_range="auto", opacity=0.6)
        annotated_image = depth_annotator.annotate(image.copy(), depth_map)
        ```
    """

    def __init__(
        self,
        colormap: DepthColormap | str = DepthColormap.TURBO,
        quantity: DepthQuantity | str = DepthQuantity.DISPARITY,
        display_range: str | tuple[float, float] = "auto",
        opacity: float = 1.0,
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
                `"auto"` (default) uses this map's 2nd to 98th percentile.
                A `(low, high)` tuple fixes the range in the quantity's unit (pixels
                for disparity, metres for depth).
            opacity: Opacity of the colours over the scene, from 0 to 1.

        Raises:
            ValueError: If an option is out of range.
        """
        self.colormap = DepthColormap.from_value(colormap)
        self.quantity = DepthQuantity.from_value(quantity)
        self.display_range = _check_display_range_option(display_range)
        if not (math.isfinite(opacity) and 0 <= opacity <= 1):
            raise ValueError(f"opacity must be between 0 and 1, got {opacity}.")
        self.opacity = float(opacity)

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
        low, high = self._resolve_range(depth_map)
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
        return scene

    def _resolve_range(self, depth_map: DepthMap) -> tuple[float, float]:
        """Return the colour range in the quantity's unit for this map.

        A map without any depth falls back to `(0, 1)`.
        """
        option = self.display_range
        if isinstance(option, tuple):
            return option
        percentile = depth_map._percentile_range(quantity=self.quantity)
        return percentile if percentile is not None else _FALLBACK_RANGE


def _check_display_range_option(
    display_range: str | tuple[float, float],
) -> str | tuple[float, float]:
    """Validate the annotator's `display_range` option and normalise tuples."""
    if isinstance(display_range, str):
        if display_range != "auto":
            raise ValueError(
                "display_range must be 'auto' or a (low, high) tuple, got "
                f"{display_range!r}."
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

    `t = clamp((v - low) / (high - low))`, flipped when the quantity's low end is
    near. Pixels without depth get 0; they are not painted.
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

    `colors` is `(H, W, 3)`. Whole-image arithmetic and a masked copy are several
    times faster than gathering and scattering the masked pixels.
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
