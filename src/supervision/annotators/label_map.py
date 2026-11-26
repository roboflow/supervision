from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from supervision import _cv2 as cv2
from supervision.draw.base import ImageType
from supervision.draw.color import ColorPalette
from supervision.utils.conversion import ensure_cv2_image_for_class_method
from supervision.utils.internal import _to_numpy

#: Widest range of class ids, from the smallest to the largest in one map, that the
#: colour lookup table covers. A map spanning more ids is not a class-id map.
_MAX_ID_SPAN = 1 << 20


class LabelMapAnnotator:
    """Colours a dense class-id map over an image, one palette colour per class.

    A semantic segmentation model labels every pixel with a class id. This annotator
    paints that `(H, W)` map directly through a colour lookup table, so it holds one
    array the size of the image instead of one mask per class, which is what
    :class:`~supervision.annotators.core.MaskAnnotator` draws from
    :class:`~supervision.detection.core.Detections`.

    Class `i` takes palette colour `i`, wrapping around a palette shorter than the
    class list, as :class:`~supervision.annotators.core.MaskAnnotator` does, so the
    two annotators colour a class alike.

    === "Image"

        ```python
        import supervision as sv
        from PIL import Image

        image = Image.open("<SOURCE_IMAGE_PATH>")
        label_map = ...  # (H, W) integer class ids from a semantic segmentation model

        label_map_annotator = sv.LabelMapAnnotator(opacity=0.6)
        annotated_image = label_map_annotator.annotate(image.copy(), label_map)
        ```
    """

    def __init__(
        self,
        color: ColorPalette = ColorPalette.DEFAULT,
        opacity: float = 0.6,
        background_id: int | None = 0,
    ) -> None:
        """
        Args:
            color: Colour palette; class `i` takes the colour at index `i`.
            opacity: Opacity of the colours over the scene, from 0 to 1. Values
                outside are clamped: `<= 0` draws nothing and `>= 1` fully replaces
                the painted pixels.
            background_id: Class id left unpainted, so the scene shows through. `0`
                by default, as in
                :meth:`~supervision.detection.core.Detections.from_semantic_segmentation`.
                Pass `None` to paint every class. Negative ids are ignore labels and
                are never painted.

        Raises:
            ValueError: If the palette holds no colour.
        """
        if not color.colors:
            raise ValueError("color must be a palette with at least one colour.")
        self.color = color
        self.opacity = opacity
        self.background_id = background_id

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, label_map: Any) -> ImageType:
        """Colour the class-id map over the scene.

        Args:
            scene: The image to draw on, a 3-channel `uint8` `numpy.ndarray` (BGR) or
                a `PIL.Image.Image`. It is drawn on in place.
            label_map: Per-pixel class ids, an `(H, W)` integer array or tensor with
                the scene's height and width.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray` or
                `PIL.Image.Image`).

        Raises:
            TypeError: If `scene` is not a `numpy.ndarray` or `PIL.Image.Image`.
            ValueError: If `scene` is not a 3-channel `uint8` image, if `label_map` is
                not a 2-D integer array of the scene's size, or if its ids span more
                than 1,048,576 values from smallest to largest.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> palette = sv.ColorPalette.from_hex(["#ff0000", "#00ff00", "#0000ff"])
            >>> annotator = sv.LabelMapAnnotator(color=palette, opacity=1.0)
            >>> scene = np.zeros((1, 2, 3), dtype=np.uint8)
            >>> annotator.annotate(scene, np.array([[1, 0]]))[..., ::-1].tolist()
            [[[0, 255, 0], [0, 0, 0]]]

            ```
        """
        # The conversion decorator rejects other types and turns a Pillow image into
        # an array, so this always holds; it narrows `ImageType` for the type checker.
        assert isinstance(scene, np.ndarray)
        if scene.ndim != 3 or scene.shape[2] != 3:
            raise ValueError(
                f"LabelMapAnnotator draws on 3-channel images, got shape {scene.shape}."
            )
        # The colours are uint8; a wider scene would blend differently on OpenCV and
        # on the NumPy fallback, and `opacity >= 1` would write 0-255 into a float one.
        if scene.dtype != np.uint8:
            raise ValueError(
                f"LabelMapAnnotator draws on uint8 images, got dtype {scene.dtype}."
            )
        class_ids = self._check_label_map(label_map, scene.shape[:2])
        if class_ids.size == 0:
            return scene

        painted = class_ids >= 0
        if self.background_id is not None:
            painted &= class_ids != self.background_id
        _blend(scene, painted, self._colors_of(class_ids), self.opacity)
        return scene

    def _colors_of(self, class_ids: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
        """Return an `(H, W, 3)` BGR colour per pixel from one lookup table.

        The table runs from the smallest to the largest id in the map, so its size
        follows the ids present, not the palette or the dtype. Negative ids index like
        any other and are masked out by the caller.
        """
        palette = np.array(
            [color.as_bgr() for color in self.color.colors], dtype=np.uint8
        )
        # Python ints, because the difference of two `int8` ids would wrap.
        lowest, highest = int(class_ids.min()), int(class_ids.max())
        span = highest - lowest
        if span >= _MAX_ID_SPAN:
            raise ValueError(
                f"label_map ids span more than {_MAX_ID_SPAN} values, from {lowest} "
                f"to {highest}; it does not look like a class-id map."
            )
        # Bounded offsets keep the LUT indices integral even at the dtype limits.
        palette_size = len(palette)
        palette_indices = (np.arange(span + 1) + lowest % palette_size) % palette_size
        table = palette[palette_indices]
        # Unsigned subtraction is safe because every id is at least the minimum.
        # Widen signed inputs so narrow dtypes can hold the full range of offsets.
        wide = (
            class_ids
            if class_ids.dtype.kind == "u"
            else class_ids.astype(np.int64, copy=False)
        )
        offsets = wide - np.asarray(lowest, dtype=wide.dtype)
        colors: npt.NDArray[np.uint8] = table[offsets]
        return colors

    @staticmethod
    def _check_label_map(
        label_map: Any, scene_size: tuple[int, int]
    ) -> npt.NDArray[np.integer]:
        """Return `label_map` as an integer array of the scene's size, or raise."""
        if not isinstance(label_map, np.ndarray) and not hasattr(label_map, "cpu"):
            raise ValueError(
                "label_map must be an (H, W) integer array or tensor, got "
                f"{type(label_map).__name__}."
            )
        class_ids = _to_numpy(label_map)
        if class_ids.ndim != 2:
            raise ValueError(
                f"label_map must be a 2-D array, got shape {class_ids.shape}."
            )
        if class_ids.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"label_map must hold integer class ids, got dtype {class_ids.dtype}."
            )
        if class_ids.shape != scene_size:
            raise ValueError(
                f"label_map shape {class_ids.shape} does not match the scene's "
                f"height and width {scene_size}."
            )
        return class_ids


def _blend(
    scene: npt.NDArray[np.uint8],
    where: npt.NDArray[np.bool_],
    colors: npt.NDArray[np.uint8],
    opacity: float,
) -> None:
    """Blend `colors` into `scene` at `opacity`, only where `where` is set.

    The blend is `addWeighted`, as in the other annotators. Blending the whole image and
    copying it through the mask is faster than gathering and scattering the masked
    pixels.
    """
    if opacity <= 0 or not where.any():
        return
    blended: npt.NDArray[np.uint8]
    if opacity >= 1:
        blended = colors
    else:
        blended = cv2.addWeighted(colors, opacity, scene, 1 - opacity, 0)
    channel_mask = np.repeat(where[..., np.newaxis], scene.shape[2], axis=2)
    np.copyto(scene, blended, where=channel_mask)
