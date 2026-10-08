from __future__ import annotations

import numpy as np
import numpy.typing as npt

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

_FALLBACK_RANGE = (0.0, 1.0)


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
        display_range: str | tuple[float, float] = "auto",
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
        with the scene's aspect ratio lines up.

        Args:
            scene: The image to draw on, a 3-channel `numpy.ndarray` (BGR) or a
                `PIL.Image.Image`. It is drawn on in place.
            depth_map: The depth map to colour.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray` or
                `PIL.Image.Image`).

        Raises:
            TypeError: If `scene` is not a `numpy.ndarray` or `PIL.Image.Image`.
            ValueError: If `scene` is not a 3-channel image, or if `display_range`
                gives no usable colour range for this map's kind, such as a
                `depth_m` range with `low <= 0`.

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
        conversion = _resolve_conversion(depth_map.kind)
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

        An explicit range is converted from the map's unit. A map without any depth
        falls back to `(0, 1)`.
        """
        option = self.display_range
        if isinstance(option, tuple):
            return _convert_display_range(option, depth_map.kind)
        percentile = depth_map._percentile_range()
        return percentile if percentile is not None else _FALLBACK_RANGE


def _check_display_range_option(
    display_range: str | tuple[float, float],
) -> str | tuple[float, float]:
    """Validate the annotator's `display_range` option and normalise it."""
    if isinstance(display_range, str) and display_range.lower() == "auto":
        return "auto"
    if isinstance(display_range, str) or len(display_range) != 2:
        raise ValueError(
            "display_range must be 'auto' or a (low, high) tuple, got "
            f"{display_range!r}."
        )
    low, high = (float(bound) for bound in display_range)
    if not _has_usable_span(low, high):
        raise ValueError(
            "display_range must have finite bounds with low < high and a span that "
            f"fits in float32, got {display_range}."
        )
    return low, high


def _convert_display_range(
    display_range: tuple[float, float], kind: DepthKind
) -> tuple[float, float]:
    """Convert an explicit display range from the map's unit to the coloured unit.

    Raises:
        ValueError: If the converted range is not finite, has no float32 span, or the
            kind is coloured as a reciprocal and `low <= 0`.
    """
    conversion = _resolve_conversion(kind)
    converted = conversion.apply_range(display_range)
    # A reciprocal sends low <= 0 to infinity or flips its sign, so it bounds nothing.
    if conversion.reciprocal and display_range[0] <= 0:
        converted = None
    if converted is None or not _has_usable_span(*converted):
        raise ValueError(
            f"display_range {display_range} gives no usable colour range for a "
            f"{kind.value!r} map. The range is in the map's own unit; a 'depth_m' "
            "range is in metres and needs 0 < low < high."
        )
    return converted


def _has_usable_span(low: float, high: float) -> bool:
    """Tell whether `high - low` is finite and positive in float32, as maps colour."""
    with np.errstate(over="ignore"):
        span = np.float32(high) - np.float32(low)
    return bool(np.isfinite(span) and span > 0)


def _color_coordinates(
    depth_map: DepthMap, conversion: _Conversion, low: float, high: float
) -> npt.NDArray[np.floating]:
    """Return each pixel's colour coordinate in `[0, 1]`, 1 at the near end.

    `t = clamp((v - low) / (high - low))`. A flat range, `low == high`, maps every
    pixel to 0, as matplotlib's `Normalize` does. Pixels without depth get 0; they
    are not painted.
    """
    if low == high:
        return np.zeros(depth_map.values.shape, dtype=np.float32)
    converted = conversion.apply(depth_map.to_float())
    span = np.float32(high - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        coordinates: npt.NDArray[np.floating] = (converted - np.float32(low)) / span
    np.clip(coordinates, 0.0, 1.0, out=coordinates)
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
