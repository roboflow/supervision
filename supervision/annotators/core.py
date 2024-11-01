from functools import lru_cache
from math import sqrt
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from supervision.annotators.base import BaseAnnotator, ImageType
from supervision.annotators.utils import (
    ColorLookup,
    Trace,
    resolve_color,
    resolve_text_background_xyxy,
)
from supervision.config import CLASS_NAME_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils import clip_boxes, mask_to_polygons
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import draw_polygon
from supervision.geometry.core import Position
from supervision.utils.conversion import (
    ensure_cv2_image_for_annotation,
    ensure_pil_image_for_annotation,
)
from supervision.utils.image import (
    crop_image,
    letterbox_image,
    overlay_image,
    scale_image,
)
from supervision.utils.internal import deprecated


class BoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with bounding boxes based on the provided detections.

        Args:
            scene (ImageType): The image where bounding boxes will be drawn. `ImageType`
                is a flexible type, accepting either `numpy.ndarray` or
                `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![bounding-box-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/bounding-box-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
        return scene


@deprecated(
    "`BoundingBoxAnnotator` is deprecated and has been renamed to `BoxAnnotator`."
    " `BoundingBoxAnnotator` will be removed in supervision-0.26.0."
)
class BoundingBoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with bounding boxes based on the provided detections.

        Args:
            scene (ImageType): The image where bounding boxes will be drawn. `ImageType`
            is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            bounding_box_annotator = sv.BoundingBoxAnnotator()
            annotated_frame = bounding_box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![bounding-box-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/bounding-box-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
        return scene


class OrientedBoxAnnotator(BaseAnnotator):
    """
    A class for drawing oriented bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with oriented bounding boxes based on the provided detections.

        Args:
            scene (ImageType): The image where bounding boxes will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO("yolov8n-obb.pt")

            result = model(image)[0]
            detections = sv.Detections.from_ultralytics(result)

            oriented_box_annotator = sv.OrientedBoxAnnotator()
            annotated_frame = oriented_box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```
        """  # noqa E501 // docs
        assert isinstance(scene, np.ndarray)
        if detections.data is None or ORIENTED_BOX_COORDINATES not in detections.data:
            return scene
        obb_boxes = np.array(detections.data[ORIENTED_BOX_COORDINATES]).astype(int)

        for detection_idx in range(len(detections)):
            obb = obb_boxes[detection_idx]
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )

            cv2.drawContours(scene, [obb], 0, color.as_bgr(), self.thickness)

        return scene


class MaskAnnotator(BaseAnnotator):
    """
    A class for drawing masks on an image using provided detections.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        opacity: float = 0.5,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with masks based on the provided detections.

        Args:
            scene (ImageType): The image where masks will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/mask-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if detections.mask is None:
            return scene

        colored_mask = np.array(scene, copy=True, dtype=np.uint8)

        for detection_idx in np.flip(np.argsort(detections.area)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            mask = detections.mask[detection_idx]
            colored_mask[mask] = color.as_bgr()

        cv2.addWeighted(
            colored_mask, self.opacity, scene, 1 - self.opacity, 0, dst=scene
        )
        return scene


class PolygonAnnotator(BaseAnnotator):
    """
    A class for drawing polygons on an image using provided detections.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the polygon lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with polygons based on the provided detections.

        Args:
            scene (ImageType): The image where polygons will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            polygon_annotator = sv.PolygonAnnotator()
            annotated_frame = polygon_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![polygon-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/polygon-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if detections.mask is None:
            return scene

        for detection_idx in range(len(detections)):
            mask = detections.mask[detection_idx]
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            for polygon in mask_to_polygons(mask=mask):
                scene = draw_polygon(
                    scene=scene,
                    polygon=polygon,
                    color=color,
                    thickness=self.thickness,
                )

        return scene


class ColorAnnotator(BaseAnnotator):
    """
    A class for drawing box masks on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        opacity: float = 0.5,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.color_lookup: ColorLookup = color_lookup
        self.opacity = opacity

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with box masks based on the provided detections.

        Args:
            scene (ImageType): The image where bounding boxes will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            color_annotator = sv.ColorAnnotator()
            annotated_frame = color_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![box-mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-mask-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        scene_with_boxes = scene.copy()
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene_with_boxes,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=-1,
            )

        cv2.addWeighted(
            scene_with_boxes, self.opacity, scene, 1 - self.opacity, gamma=0, dst=scene
        )
        return scene


class HaloAnnotator(BaseAnnotator):
    """
    A class for drawing Halos on an image using provided detections.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        opacity: float = 0.8,
        kernel_size: int = 40,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            kernel_size (int): The size of the average pooling kernel used for creating
                the halo.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup
        self.kernel_size: int = kernel_size

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with halos based on the provided detections.

        Args:
            scene (ImageType): The image where masks will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            halo_annotator = sv.HaloAnnotator()
            annotated_frame = halo_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![halo-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/halo-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if detections.mask is None:
            return scene
        colored_mask = np.zeros_like(scene, dtype=np.uint8)
        fmask = np.array([False] * scene.shape[0] * scene.shape[1]).reshape(
            scene.shape[0], scene.shape[1]
        )

        for detection_idx in np.flip(np.argsort(detections.area)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            mask = detections.mask[detection_idx]
            fmask = np.logical_or(fmask, mask)
            color_bgr = color.as_bgr()
            colored_mask[mask] = color_bgr

        colored_mask = cv2.blur(colored_mask, (self.kernel_size, self.kernel_size))
        colored_mask[fmask] = [0, 0, 0]
        gray = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
        alpha = self.opacity * gray / gray.max()
        alpha_mask = alpha[:, :, np.newaxis]
        blended_scene = np.uint8(scene * (1 - alpha_mask) + colored_mask * self.opacity)
        np.copyto(scene, blended_scene)
        return scene


class EllipseAnnotator(BaseAnnotator):
    """
    A class for drawing ellipses on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        start_angle: int = -45,
        end_angle: int = 235,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the ellipse lines.
            start_angle (int): Starting angle of the ellipse.
            end_angle (int): Ending angle of the ellipse.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.start_angle: int = start_angle
        self.end_angle: int = end_angle
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with ellipses based on the provided detections.

        Args:
            scene (ImageType): The image where ellipses will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            ellipse_annotator = sv.EllipseAnnotator()
            annotated_frame = ellipse_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![ellipse-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/ellipse-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            center = (int((x1 + x2) / 2), y2)
            width = x2 - x1
            cv2.ellipse(
                scene,
                center=center,
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=self.start_angle,
                endAngle=self.end_angle,
                color=color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_4,
            )
        return scene


class BoxCornerAnnotator(BaseAnnotator):
    """
    A class for drawing box corners on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 4,
        corner_length: int = 15,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the corner lines.
            corner_length (int): Length of each corner line.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.corner_length: int = corner_length
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with box corners based on the provided detections.

        Args:
            scene (ImageType): The image where box corners will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            corner_annotator = sv.BoxCornerAnnotator()
            annotated_frame = corner_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![box-corner-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-corner-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

            for x, y in corners:
                x_end = x + self.corner_length if x == x1 else x - self.corner_length
                cv2.line(
                    scene, (x, y), (x_end, y), color.as_bgr(), thickness=self.thickness
                )

                y_end = y + self.corner_length if y == y1 else y - self.corner_length
                cv2.line(
                    scene, (x, y), (x, y_end), color.as_bgr(), thickness=self.thickness
                )
        return scene


class CircleAnnotator(BaseAnnotator):
    """
    A class for drawing circle on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the circle line.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with circles based on the provided detections.

        Args:
            scene (ImageType): The image where box corners will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            circle_annotator = sv.CircleAnnotator()
            annotated_frame = circle_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```


        ![circle-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/circle-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.circle(
                img=scene,
                center=center,
                radius=int(distance),
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        return scene


class DotAnnotator(BaseAnnotator):
    """
    A class for drawing dots on an image at specific coordinates based on provided
    detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        radius: int = 4,
        position: Position = Position.CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        outline_thickness: int = 0,
        outline_color: Union[Color, ColorPalette] = Color.BLACK,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            radius (int): Radius of the drawn dots.
            position (Position): The anchor position for placing the dot.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            outline_thickness (int): Thickness of the outline of the dot.
            outline_color (Union[Color, ColorPalette]): The color or color palette to
                use for outline. It is activated by setting outline_thickness to a value
                greater than 0.
        """
        self.color: Union[Color, ColorPalette] = color
        self.radius: int = radius
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup
        self.outline_thickness = outline_thickness
        self.outline_color: Union[Color, ColorPalette] = outline_color

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with dots based on the provided detections.

        Args:
            scene (ImageType): The image where dots will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            dot_annotator = sv.DotAnnotator()
            annotated_frame = dot_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![dot-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/dot-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        xy = detections.get_anchors_coordinates(anchor=self.position)
        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            center = (int(xy[detection_idx, 0]), int(xy[detection_idx, 1]))

            cv2.circle(scene, center, self.radius, color.as_bgr(), -1)
            if self.outline_thickness:
                outline_color = resolve_color(
                    color=self.outline_color,
                    detections=detections,
                    detection_idx=detection_idx,
                    color_lookup=self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup,
                )
                cv2.circle(
                    scene,
                    center,
                    self.radius,
                    outline_color.as_bgr(),
                    self.outline_thickness,
                )
        return scene


class LabelAnnotator(BaseAnnotator):
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        text_color: Union[Color, ColorPalette] = Color.WHITE,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        border_radius: int = 0,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (Union[Color, ColorPalette]): The color or color palette to use
                for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            border_radius (int): The radius to apply round edges. If the selected
                value is higher than the lower dimension, width or height, is clipped.
        """
        self.border_radius: int = border_radius
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Union[Color, ColorPalette] = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        labels: Optional[List[str]] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with labels based on the provided detections.

        Args:
            scene (ImageType): The image where labels will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            labels (Optional[List[str]]): Custom labels for each detection.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(detections['class_name'], detections.confidence)
            ]

            label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            annotated_frame = label_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            ```

        ![label-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/label-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)
        if labels is not None and len(labels) != len(detections):
            raise ValueError(
                f"The number of labels ({len(labels)}) does not match the "
                f"number of detections ({len(detections)}). Each detection "
                f"should have exactly 1 label."
            )

        for detection_idx, center_coordinates in enumerate(anchors_coordinates):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            text_color = resolve_color(
                color=self.text_color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            if labels is not None:
                text = labels[detection_idx]
            elif CLASS_NAME_DATA_FIELD in detections.data:
                text = detections.data[CLASS_NAME_DATA_FIELD][detection_idx]
            elif detections.class_id is not None:
                text = str(detections.class_id[detection_idx])
            else:
                text = str(detection_idx)

            text_w, text_h = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]
            text_w_padded = text_w + 2 * self.text_padding
            text_h_padded = text_h + 2 * self.text_padding
            text_background_xyxy = resolve_text_background_xyxy(
                center_coordinates=tuple(center_coordinates),
                text_wh=(text_w_padded, text_h_padded),
                position=self.text_anchor,
            )

            text_x = text_background_xyxy[0] + self.text_padding
            text_y = text_background_xyxy[1] + self.text_padding + text_h

            self.draw_rounded_rectangle(
                scene=scene,
                xyxy=text_background_xyxy,
                color=color.as_bgr(),
                border_radius=self.border_radius,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene

    @staticmethod
    def draw_rounded_rectangle(
        scene: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        border_radius: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1

        border_radius = min(border_radius, min(width, height) // 2)

        rectangle_coordinates = [
            ((x1 + border_radius, y1), (x2 - border_radius, y2)),
            ((x1, y1 + border_radius), (x2, y2 - border_radius)),
        ]
        circle_centers = [
            (x1 + border_radius, y1 + border_radius),
            (x2 - border_radius, y1 + border_radius),
            (x1 + border_radius, y2 - border_radius),
            (x2 - border_radius, y2 - border_radius),
        ]

        for coordinates in rectangle_coordinates:
            cv2.rectangle(
                img=scene,
                pt1=coordinates[0],
                pt2=coordinates[1],
                color=color,
                thickness=-1,
            )
        for center in circle_centers:
            cv2.circle(
                img=scene,
                center=center,
                radius=border_radius,
                color=color,
                thickness=-1,
            )
        return scene


class RichLabelAnnotator(BaseAnnotator):
    """
    A class for annotating labels on an image using provided detections,
    with support for Unicode characters by using a custom font.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        text_color: Union[Color, ColorPalette] = Color.WHITE,
        font_path: Optional[str] = None,
        font_size: int = 10,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        border_radius: int = 0,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (Union[Color, ColorPalette]): The color to use for the text.
            font_path (Optional[str]): Path to the font file (e.g., ".ttf" or ".otf")
                to use for rendering text. If `None`, the default PIL font will be used.
            font_size (int): Font size for the text.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            border_radius (int): The radius to apply round edges. If the selected
                value is higher than the lower dimension, width or height, is clipped.
        """
        self.color = color
        self.text_color = text_color
        self.text_padding = text_padding
        self.text_anchor = text_position
        self.color_lookup = color_lookup
        self.border_radius = border_radius
        if font_path is not None:
            try:
                self.font = ImageFont.truetype(font_path, font_size)
            except OSError:
                print(f"Font path '{font_path}' not found. Using PIL's default font.")
                self.font = self._load_default_font(font_size)
        else:
            self.font = self._load_default_font(font_size)

    @ensure_pil_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        labels: Optional[List[str]] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with labels based on the provided
        detections, with support for Unicode characters.

        Args:
            scene (ImageType): The image where labels will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            labels (Optional[List[str]]): Custom labels for each detection.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(detections['class_name'], detections.confidence)
            ]

            rich_label_annotator = sv.RichLabelAnnotator(font_path="path/to/font.ttf")
            annotated_frame = label_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            ```

        """
        assert isinstance(scene, Image.Image)
        draw = ImageDraw.Draw(scene)
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)
        if labels is not None and len(labels) != len(detections):
            raise ValueError(
                f"The number of labels provided ({len(labels)}) does not match the "
                f"number of detections ({len(detections)}). Each detection should have "
                f"a corresponding label."
            )
        for detection_idx, center_coordinates in enumerate(anchors_coordinates):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            text_color = resolve_color(
                color=self.text_color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=(
                    self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup
                ),
            )

            if labels is not None:
                text = labels[detection_idx]
            elif CLASS_NAME_DATA_FIELD in detections.data:
                text = detections.data[CLASS_NAME_DATA_FIELD][detection_idx]
            elif detections.class_id is not None:
                text = str(detections.class_id[detection_idx])
            else:
                text = str(detection_idx)

            left, top, right, bottom = draw.textbbox((0, 0), text, font=self.font)
            text_width = right - left
            text_height = bottom - top
            text_w_padded = text_width + 2 * self.text_padding
            text_h_padded = text_height + 2 * self.text_padding
            text_background_xyxy = resolve_text_background_xyxy(
                center_coordinates=tuple(center_coordinates),
                text_wh=(text_w_padded, text_h_padded),
                position=self.text_anchor,
            )

            text_x = text_background_xyxy[0] + self.text_padding - left
            text_y = text_background_xyxy[1] + self.text_padding - top

            draw.rounded_rectangle(
                text_background_xyxy,
                radius=self.border_radius,
                fill=color.as_rgb(),
                outline=None,
            )
            draw.text(
                xy=(text_x, text_y),
                text=text,
                font=self.font,
                fill=text_color.as_rgb(),
            )
        return scene

    @staticmethod
    def _load_default_font(size):
        """
        PIL either loads a font that accepts a size (e.g. on my machine)
        or raises an error saying `load_default` does not accept arguments
        (e.g. in Colab).
        """
        try:
            font = ImageFont.load_default(size)
        except TypeError:
            font = ImageFont.load_default()
        return font


class IconAnnotator(BaseAnnotator):
    """
    A class for drawing an icon on an image, using provided detections.
    """

    def __init__(
        self,
        icon_resolution_wh: Tuple[int, int] = (64, 64),
        icon_position: Position = Position.TOP_CENTER,
        offset_xy: Tuple[int, int] = (0, 0),
    ):
        """
        Args:
            icon_resolution_wh (Tuple[int, int]): The size of drawn icons.
                All icons will be resized to this resolution, keeping the aspect ratio.
            icon_position (Position): The position of the icon.
            offset_xy (Tuple[int, int]): The offset to apply to the icon position,
                in pixels. Can be both positive and negative.
        """
        self.icon_resolution_wh = icon_resolution_wh
        self.position = icon_position
        self.offset_xy = offset_xy

    @ensure_cv2_image_for_annotation
    def annotate(
        self, scene: ImageType, detections: Detections, icon_path: Union[str, List[str]]
    ) -> ImageType:
        """
        Annotates the given scene with given icons.

        Args:
            scene (ImageType): The image where labels will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            icon_path (Union[str, List[str]]): The path to the PNG image to use as an
                icon. Must be a single path or a list of paths, one for each detection.
                Pass an empty string `""` to draw nothing.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            available_icons = ["roboflow.png", "lenny.png"]
            icon_paths = [np.random.choice(available_icons) for _ in detections]

            icon_annotator = sv.IconAnnotator()
            annotated_frame = icon_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                icon_path=icon_paths
            )
            ```

        ![icon-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/icon-annotator-example.png)
        """
        assert isinstance(scene, np.ndarray)
        if isinstance(icon_path, list) and len(icon_path) != len(detections):
            raise ValueError(
                f"The number of icon paths provided ({len(icon_path)}) does not match "
                f"the number of detections ({len(detections)}). Either provide a single"
                f" icon path or one for each detection."
            )

        xy = detections.get_anchors_coordinates(anchor=self.position).astype(int)

        for detection_idx in range(len(detections)):
            current_path = (
                icon_path if isinstance(icon_path, str) else icon_path[detection_idx]
            )
            if current_path == "":
                continue
            icon = self._load_icon(current_path)
            icon_h, icon_w = icon.shape[:2]

            x = int(xy[detection_idx, 0] - icon_w / 2 + self.offset_xy[0])
            y = int(xy[detection_idx, 1] - icon_h / 2 + self.offset_xy[1])

            scene[:] = overlay_image(scene, icon, (x, y))
        return scene

    @lru_cache
    def _load_icon(self, icon_path: str) -> np.ndarray:
        icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            raise FileNotFoundError(
                f"Error: Couldn't load the icon image from {icon_path}"
            )
        icon = letterbox_image(image=icon, resolution_wh=self.icon_resolution_wh)
        return icon


class BlurAnnotator(BaseAnnotator):
    """
    A class for blurring regions in an image using provided detections.
    """

    def __init__(self, kernel_size: int = 15):
        """
        Args:
            kernel_size (int): The size of the average pooling kernel used for blurring.
        """
        self.kernel_size: int = kernel_size

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
    ) -> ImageType:
        """
        Annotates the given scene by blurring regions based on the provided detections.

        Args:
            scene (ImageType): The image where blurring will be applied.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            blur_annotator = sv.BlurAnnotator()
            annotated_frame = circle_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![blur-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/blur-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)

        for x1, y1, x2, y2 in clipped_xyxy:
            roi = scene[y1:y2, x1:x2]
            roi = cv2.blur(roi, (self.kernel_size, self.kernel_size))
            scene[y1:y2, x1:x2] = roi

        return scene


class TraceAnnotator(BaseAnnotator):
    """
    A class for drawing trace paths on an image based on detection coordinates.

    !!! warning

        This annotator uses the `sv.Detections.tracker_id`. Read
        [here](/latest/trackers/) to learn how to plug
        tracking into your inference pipeline.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        position: Position = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color to draw the trace, can be
                a single color or a color palette.
            position (Position): The position of the trace.
                Defaults to `CENTER`.
            trace_length (int): The maximum length of the trace in terms of historical
                points. Defaults to `30`.
            thickness (int): The thickness of the trace lines. Defaults to `2`.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.trace = Trace(max_size=trace_length, anchor=position)
        self.thickness = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Draws trace paths on the frame based on the detection coordinates provided.

        Args:
            scene (ImageType): The image on which the traces will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): The detections which include coordinates for
                which the traces will be drawn.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO('yolov8x.pt')
            trace_annotator = sv.TraceAnnotator()

            video_info = sv.VideoInfo.from_video_path(video_path='...')
            frames_generator = sv.get_video_frames_generator(source_path='...')
            tracker = sv.ByteTrack()

            with sv.VideoSink(target_path='...', video_info=video_info) as sink:
               for frame in frames_generator:
                   result = model(frame)[0]
                   detections = sv.Detections.from_ultralytics(result)
                   detections = tracker.update_with_detections(detections)
                   annotated_frame = trace_annotator.annotate(
                       scene=frame.copy(),
                       detections=detections)
                   sink.write_frame(frame=annotated_frame)
            ```

        ![trace-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/trace-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if detections.tracker_id is None:
            raise ValueError(
                "The `tracker_id` field is missing in the provided detections."
                " See more: https://supervision.roboflow.com/latest/how_to/track_objects"
            )

        self.trace.put(detections)
        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            xy = self.trace.get(tracker_id=tracker_id)
            if len(xy) > 1:
                scene = cv2.polylines(
                    scene,
                    [xy.astype(np.int32)],
                    False,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return scene


class HeatMapAnnotator(BaseAnnotator):
    """
    A class for drawing heatmaps on an image based on provided detections.
    Heat accumulates over time and is drawn as a semi-transparent overlay
    of blurred circles.
    """

    def __init__(
        self,
        position: Position = Position.BOTTOM_CENTER,
        opacity: float = 0.2,
        radius: int = 40,
        kernel_size: int = 25,
        top_hue: int = 0,
        low_hue: int = 125,
    ):
        """
        Args:
            position (Position): The position of the heatmap. Defaults to
                `BOTTOM_CENTER`.
            opacity (float): Opacity of the overlay mask, between 0 and 1.
            radius (int): Radius of the heat circle.
            kernel_size (int): Kernel size for blurring the heatmap.
            top_hue (int): Hue at the top of the heatmap. Defaults to 0 (red).
            low_hue (int): Hue at the bottom of the heatmap. Defaults to 125 (blue).
        """
        self.position = position
        self.opacity = opacity
        self.radius = radius
        self.kernel_size = kernel_size
        self.top_hue = top_hue
        self.low_hue = low_hue
        self.heat_mask: Optional[npt.NDArray[np.float32]] = None

    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        """
        Annotates the scene with a heatmap based on the provided detections.

        Args:
            scene (ImageType): The image where the heatmap will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO('yolov8x.pt')

            heat_map_annotator = sv.HeatMapAnnotator()

            video_info = sv.VideoInfo.from_video_path(video_path='...')
            frames_generator = sv.get_video_frames_generator(source_path='...')

            with sv.VideoSink(target_path='...', video_info=video_info) as sink:
               for frame in frames_generator:
                   result = model(frame)[0]
                   detections = sv.Detections.from_ultralytics(result)
                   annotated_frame = heat_map_annotator.annotate(
                       scene=frame.copy(),
                       detections=detections)
                   sink.write_frame(frame=annotated_frame)
            ```

        ![heatmap-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/heat-map-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if self.heat_mask is None:
            self.heat_mask = np.zeros(scene.shape[:2], dtype=np.float32)

        mask = np.zeros(scene.shape[:2])
        for xy in detections.get_anchors_coordinates(self.position):
            x, y = int(xy[0]), int(xy[1])
            cv2.circle(
                img=mask,
                center=(x, y),
                radius=self.radius,
                color=(1,),
                thickness=-1,  # fill
            )
        self.heat_mask = mask + self.heat_mask
        temp = self.heat_mask.copy()
        temp = self.low_hue - temp / temp.max() * (self.low_hue - self.top_hue)
        temp = temp.astype(np.uint8)
        if self.kernel_size is not None:
            temp = cv2.blur(temp, (self.kernel_size, self.kernel_size))
        hsv = np.zeros(scene.shape)
        hsv[..., 0] = temp
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(self.heat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) > 0
        scene[mask] = cv2.addWeighted(temp, self.opacity, scene, 1 - self.opacity, 0)[
            mask
        ]
        return scene


class PixelateAnnotator(BaseAnnotator):
    """
    A class for pixelating regions in an image using provided detections.
    """

    def __init__(self, pixel_size: int = 20):
        """
        Args:
            pixel_size (int): The size of the pixelation.
        """
        self.pixel_size: int = pixel_size

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
    ) -> ImageType:
        """
        Annotates the given scene by pixelating regions based on the provided
            detections.

        Args:
            scene (ImageType): The image where pixelating will be applied.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            pixelate_annotator = sv.PixelateAnnotator()
            annotated_frame = pixelate_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![pixelate-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/pixelate-annotator-example-10.png)
        """
        assert isinstance(scene, np.ndarray)
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)

        for x1, y1, x2, y2 in clipped_xyxy:
            roi = scene[y1:y2, x1:x2]
            scaled_up_roi = cv2.resize(
                src=roi, dsize=None, fx=1 / self.pixel_size, fy=1 / self.pixel_size
            )
            scaled_down_roi = cv2.resize(
                src=scaled_up_roi,
                dsize=(roi.shape[1], roi.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            scene[y1:y2, x1:x2] = scaled_down_roi

        return scene


class TriangleAnnotator(BaseAnnotator):
    """
    A class for drawing triangle markers on an image at specific coordinates based on
    provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        base: int = 10,
        height: int = 10,
        position: Position = Position.TOP_CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        outline_thickness: int = 0,
        outline_color: Union[Color, ColorPalette] = Color.BLACK,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            base (int): The base width of the triangle.
            height (int): The height of the triangle.
            position (Position): The anchor position for placing the triangle.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            outline_thickness (int): Thickness of the outline of the triangle.
            outline_color (Union[Color, ColorPalette]): The color or color palette to
                use for outline. It is activated by setting outline_thickness to a value
                greater than 0.
        """
        self.color: Union[Color, ColorPalette] = color
        self.base: int = base
        self.height: int = height
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup
        self.outline_thickness: int = outline_thickness
        self.outline_color: Union[Color, ColorPalette] = outline_color

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with triangles based on the provided detections.

        Args:
            scene (ImageType): The image where triangles will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            triangle_annotator = sv.TriangleAnnotator()
            annotated_frame = triangle_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![triangle-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/triangle-annotator-example.png)
        """
        assert isinstance(scene, np.ndarray)
        xy = detections.get_anchors_coordinates(anchor=self.position)
        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            tip_x, tip_y = int(xy[detection_idx, 0]), int(xy[detection_idx, 1])
            vertices = np.array(
                [
                    [tip_x - self.base // 2, tip_y - self.height],
                    [tip_x + self.base // 2, tip_y - self.height],
                    [tip_x, tip_y],
                ],
                np.int32,
            )

            cv2.fillPoly(scene, [vertices], color.as_bgr())
            if self.outline_thickness:
                outline_color = resolve_color(
                    color=self.outline_color,
                    detections=detections,
                    detection_idx=detection_idx,
                    color_lookup=self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup,
                )
                cv2.polylines(
                    scene,
                    [vertices],
                    True,
                    outline_color.as_bgr(),
                    thickness=self.outline_thickness,
                )
        return scene


class RoundBoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes with round edges on an image
    using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        roundness: float = 0.6,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            roundness (float): Percent of roundness for edges of bounding box.
                Value must be float 0 < roundness <= 1.0
                By default roundness percent is calculated based on smaller side
                length (width or height).
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup
        if not 0 < roundness <= 1.0:
            raise ValueError("roundness attribute must be float between (0, 1.0]")
        self.roundness: float = roundness

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with bounding boxes with rounded edges
        based on the provided detections.

        Args:
            scene (ImageType): The image where rounded bounding boxes will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            round_box_annotator = sv.RoundBoxAnnotator()
            annotated_frame = round_box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![round-box-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/round-box-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )

            radius = (
                int((x2 - x1) // 2 * self.roundness)
                if abs(x1 - x2) < abs(y1 - y2)
                else int((y2 - y1) // 2 * self.roundness)
            )

            circle_coordinates = [
                ((x1 + radius), (y1 + radius)),
                ((x2 - radius), (y1 + radius)),
                ((x2 - radius), (y2 - radius)),
                ((x1 + radius), (y2 - radius)),
            ]

            line_coordinates = [
                ((x1 + radius, y1), (x2 - radius, y1)),
                ((x2, y1 + radius), (x2, y2 - radius)),
                ((x1 + radius, y2), (x2 - radius, y2)),
                ((x1, y1 + radius), (x1, y2 - radius)),
            ]

            start_angles = (180, 270, 0, 90)
            end_angles = (270, 360, 90, 180)

            for center_coordinates, line, start_angle, end_angle in zip(
                circle_coordinates, line_coordinates, start_angles, end_angles
            ):
                cv2.ellipse(
                    img=scene,
                    center=center_coordinates,
                    axes=(radius, radius),
                    angle=0,
                    startAngle=start_angle,
                    endAngle=end_angle,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )

                cv2.line(
                    img=scene,
                    pt1=line[0],
                    pt2=line[1],
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )

        return scene


class PercentageBarAnnotator(BaseAnnotator):
    """
    A class for drawing percentage bars on an image using provided detections.
    """

    def __init__(
        self,
        height: int = 16,
        width: int = 80,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        border_color: Color = Color.BLACK,
        position: Position = Position.TOP_CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        border_thickness: Optional[int] = None,
    ):
        """
        Args:
            height (int): The height in pixels of the percentage bar.
            width (int): The width in pixels of the percentage bar.
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            border_color (Color): The color of the border lines.
            position (Position): The anchor position of drawing the percentage bar.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            border_thickness (Optional[int]): The thickness of the border lines.
        """
        self.height: int = height
        self.width: int = width
        self.color: Union[Color, ColorPalette] = color
        self.border_color: Color = border_color
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup

        if border_thickness is None:
            self.border_thickness = int(0.15 * self.height)

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
        custom_values: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with percentage bars based on the provided
        detections. The percentage bars visually represent the confidence or custom
        values associated with each detection.

        Args:
            scene (ImageType): The image where percentage bars will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.
            custom_values (Optional[np.ndarray]): Custom values array to use instead
                of the default detection confidences. This array should have the
                same length as the number of detections and contain a value between
                0 and 1 (inclusive) for each detection, representing the percentage
                to be displayed.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            percentage_bar_annotator = sv.PercentageBarAnnotator()
            annotated_frame = percentage_bar_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![percentage-bar-example](https://media.roboflow.com/
        supervision-annotator-examples/percentage-bar-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        self.validate_custom_values(custom_values=custom_values, detections=detections)

        anchors = detections.get_anchors_coordinates(anchor=self.position)
        for detection_idx in range(len(detections)):
            anchor = anchors[detection_idx]
            border_coordinates = self.calculate_border_coordinates(
                anchor_xy=(int(anchor[0]), int(anchor[1])),
                border_wh=(self.width, self.height),
                position=self.position,
            )
            border_width = border_coordinates[1][0] - border_coordinates[0][0]

            if custom_values is not None:
                value = custom_values[detection_idx]
            else:
                assert detections.confidence is not None  # MyPy type hint
                value = detections.confidence[detection_idx]

            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene,
                pt1=border_coordinates[0],
                pt2=(
                    border_coordinates[0][0] + int(border_width * value),
                    border_coordinates[1][1],
                ),
                color=color.as_bgr(),
                thickness=-1,
            )
            cv2.rectangle(
                img=scene,
                pt1=border_coordinates[0],
                pt2=border_coordinates[1],
                color=self.border_color.as_bgr(),
                thickness=self.border_thickness,
            )
        return scene

    @staticmethod
    def calculate_border_coordinates(
        anchor_xy: Tuple[int, int], border_wh: Tuple[int, int], position: Position
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        cx, cy = anchor_xy
        width, height = border_wh

        if position == Position.TOP_LEFT:
            return (cx - width, cy - height), (cx, cy)
        elif position == Position.TOP_CENTER:
            return (cx - width // 2, cy), (cx + width // 2, cy - height)
        elif position == Position.TOP_RIGHT:
            return (cx, cy), (cx + width, cy - height)
        elif position == Position.CENTER_LEFT:
            return (cx - width, cy - height // 2), (cx, cy + height // 2)
        elif position == Position.CENTER or position == Position.CENTER_OF_MASS:
            return (
                (cx - width // 2, cy - height // 2),
                (cx + width // 2, cy + height // 2),
            )
        elif position == Position.CENTER_RIGHT:
            return (cx, cy - height // 2), (cx + width, cy + height // 2)
        elif position == Position.BOTTOM_LEFT:
            return (cx - width, cy), (cx, cy + height)
        elif position == Position.BOTTOM_CENTER:
            return (cx - width // 2, cy), (cx + width // 2, cy + height)
        elif position == Position.BOTTOM_RIGHT:
            return (cx, cy), (cx + width, cy + height)

    @staticmethod
    def validate_custom_values(
        custom_values: Optional[Union[np.ndarray, List[float]]], detections: Detections
    ) -> None:
        if custom_values is None:
            if detections.confidence is None:
                raise ValueError(
                    "The provided detections do not contain confidence values. "
                    "Please provide `custom_values` or ensure that the detections "
                    "contain confidence values (e.g. by using a different model)."
                )

        else:
            if not isinstance(custom_values, (np.ndarray, list)):
                raise TypeError(
                    "custom_values must be either a numpy array or a list of floats."
                )

            if len(custom_values) != len(detections):
                raise ValueError(
                    "The length of custom_values must match the number of detections."
                )

            if not all(0 <= value <= 1 for value in custom_values):
                raise ValueError("All values in custom_values must be between 0 and 1.")


class CropAnnotator(BaseAnnotator):
    """
    A class for drawing scaled up crops of detections on the scene.
    """

    def __init__(
        self,
        position: Position = Position.TOP_CENTER,
        scale_factor: float = 2.0,
        border_color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        border_thickness: int = 2,
        border_color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            position (Position): The anchor position for placing the cropped and scaled
                part of the detection in the scene.
            scale_factor (float): The factor by which to scale the cropped image part. A
                factor of 2, for example, would double the size of the cropped area,
                allowing for a closer view of the detection.
            border_color (Union[Color, ColorPalette]): The color or color palette to
                use for annotating border around the cropped area.
            border_thickness (int): The thickness of the border around the cropped area.
            border_color_lookup (ColorLookup): Strategy for mapping colors to
                annotations. Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.position: Position = position
        self.scale_factor: float = scale_factor
        self.border_color: Union[Color, ColorPalette] = border_color
        self.border_thickness: int = border_thickness
        self.border_color_lookup: ColorLookup = border_color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the provided scene with scaled and cropped parts of the image based
        on the provided detections. Each detection is cropped from the original scene
        and scaled according to the annotator's scale factor before being placed back
        onto the scene at the specified position.


        Args:
            scene (ImageType): The image where cropped detection will be placed.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image.

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            crop_annotator = sv.CropAnnotator()
            annotated_frame = crop_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```
        """
        assert isinstance(scene, np.ndarray)
        crops = [
            crop_image(image=scene, xyxy=xyxy) for xyxy in detections.xyxy.astype(int)
        ]
        resized_crops = [
            scale_image(image=crop, scale_factor=self.scale_factor) for crop in crops
        ]
        anchors = detections.get_anchors_coordinates(anchor=self.position).astype(int)

        for idx, (resized_crop, anchor) in enumerate(zip(resized_crops, anchors)):
            crop_wh = resized_crop.shape[1], resized_crop.shape[0]
            (x1, y1), (x2, y2) = self.calculate_crop_coordinates(
                anchor=anchor, crop_wh=crop_wh, position=self.position
            )
            scene = overlay_image(image=scene, overlay=resized_crop, anchor=(x1, y1))
            color = resolve_color(
                color=self.border_color,
                detections=detections,
                detection_idx=idx,
                color_lookup=self.border_color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.border_thickness,
            )

        return scene

    @staticmethod
    def calculate_crop_coordinates(
        anchor: Tuple[int, int], crop_wh: Tuple[int, int], position: Position
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        anchor_x, anchor_y = anchor
        width, height = crop_wh

        if position == Position.TOP_LEFT:
            return (anchor_x - width, anchor_y - height), (anchor_x, anchor_y)
        elif position == Position.TOP_CENTER:
            return (
                (anchor_x - width // 2, anchor_y - height),
                (anchor_x + width // 2, anchor_y),
            )
        elif position == Position.TOP_RIGHT:
            return (anchor_x, anchor_y - height), (anchor_x + width, anchor_y)
        elif position == Position.CENTER_LEFT:
            return (
                (anchor_x - width, anchor_y - height // 2),
                (anchor_x, anchor_y + height // 2),
            )
        elif position == Position.CENTER or position == Position.CENTER_OF_MASS:
            return (
                (anchor_x - width // 2, anchor_y - height // 2),
                (anchor_x + width // 2, anchor_y + height // 2),
            )
        elif position == Position.CENTER_RIGHT:
            return (
                (anchor_x, anchor_y - height // 2),
                (anchor_x + width, anchor_y + height // 2),
            )
        elif position == Position.BOTTOM_LEFT:
            return (anchor_x - width, anchor_y), (anchor_x, anchor_y + height)
        elif position == Position.BOTTOM_CENTER:
            return (
                (anchor_x - width // 2, anchor_y),
                (anchor_x + width // 2, anchor_y + height),
            )
        elif position == Position.BOTTOM_RIGHT:
            return (anchor_x, anchor_y), (anchor_x + width, anchor_y + height)


class BackgroundOverlayAnnotator(BaseAnnotator):
    """
    A class for drawing a colored overlay on the background of an image outside
    the region of detections.

    If masks are provided, the background is colored outside the masks.
    If masks are not provided, the background is colored outside the bounding boxes.

    You can use the `force_box` parameter to force the annotator to use bounding boxes.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Color = Color.BLACK,
        opacity: float = 0.5,
        force_box: bool = False,
    ):
        """
        Args:
            color (Color): The color to use for annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            force_box (bool): If `True`, forces the annotator to use bounding boxes when
                masks are provided in the supplied sv.Detections.
        """
        self.color: Color = color
        self.opacity = opacity
        self.force_box = force_box

    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        """
        Applies a colored overlay to the scene outside of the detected regions.

        Args:
            scene (ImageType): The image where masks will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            background_overlay_annotator = sv.BackgroundOverlayAnnotator()
            annotated_frame = background_overlay_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![background-overlay-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/background-color-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        colored_mask = np.full_like(scene, self.color.as_bgr(), dtype=np.uint8)

        cv2.addWeighted(
            scene, 1 - self.opacity, colored_mask, self.opacity, 0, dst=colored_mask
        )

        if detections.mask is None or self.force_box:
            for x1, y1, x2, y2 in detections.xyxy.astype(int):
                colored_mask[y1:y2, x1:x2] = scene[y1:y2, x1:x2]
        else:
            for mask in detections.mask:
                colored_mask[mask] = scene[mask]

        np.copyto(scene, colored_mask)
        return scene
