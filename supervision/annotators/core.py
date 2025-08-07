from __future__ import annotations

from functools import lru_cache
from math import sqrt

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splev, splprep

from supervision.annotators.base import BaseAnnotator, ImageType
from supervision.annotators.utils import (
    PENDING_TRACK_ID,
    ColorLookup,
    Trace,
    get_labels_text,
    resolve_color,
    resolve_text_background_xyxy,
    snap_boxes,
    validate_labels,
    wrap_text,
)
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import clip_boxes, spread_out_boxes
from supervision.detection.utils.converters import (
    mask_to_polygons,
    polygon_to_mask,
    xyxy_to_polygons,
)
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import draw_polygon, draw_rounded_rectangle, draw_text
from supervision.geometry.core import Point, Position, Rect
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

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


class _BaseLabelAnnotator(BaseAnnotator):
    """
    Base class for annotators that add labels to detections.

    Attributes:
        color (Union[Color, ColorPalette]): The color to use for the label background.
        color_lookup (ColorLookup): The method used to determine the color of the label.
        text_color (Union[Color, ColorPalette]): The color to use for the label text.
        text_padding (int): The padding around the label text, in pixels.
        text_anchor (Position): The position of the text relative to the detection
            bounding box.
        text_offset (Tuple[int, int]): A tuple of 2D coordinates `(x, y)` to
            offset the text position from the anchor point, in pixels.
        border_radius (int): The radius of the label background corners, in pixels.
        smart_position (bool): Whether to intelligently adjust the label position to
            avoid overlapping with other elements.
        max_line_length (Optional[int]): Maximum number of characters per line before
            wrapping the text. None means no wrapping.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        text_color: Color | ColorPalette = Color.WHITE,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
    ):
        """
        Initializes the _BaseLabelAnnotator.

        Args:
            color (Union[Color, ColorPalette], optional): The color to use for the label
                background.
            color_lookup (ColorLookup, optional): The method used to determine the color
                of the label
            text_color (Union[Color, ColorPalette], optional): The color to use for the
                label text.
            text_padding (int, optional): The padding around the label text, in pixels.
            text_position (Position, optional): The position of the text relative to the
                detection bounding box.
            text_offset (Tuple[int, int], optional): A tuple of 2D coordinates
                `(x, y)` to offset the text position from the anchor point, in pixels.
            border_radius (int, optional): The radius of the label background corners,
                in pixels.
            smart_position (bool, optional): Whether to intelligently adjust the label
                position to avoid overlapping with other elements.
            max_line_length (Optional[int], optional): Maximum number of characters per
                line before wrapping the text. None means no wrapping.
        """
        self.color: Color | ColorPalette = color
        self.color_lookup: ColorLookup = color_lookup
        self.text_color: Color | ColorPalette = text_color
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position
        self.text_offset: tuple[int, int] = text_offset
        self.border_radius: int = border_radius
        self.smart_position = smart_position
        self.max_line_length: int | None = max_line_length

    def _adjust_labels_in_frame(
        self,
        resolution_wh: tuple[int, int],
        labels: list[str],
        label_properties: np.ndarray,
    ) -> np.ndarray:
        """
        Adjusts the position of labels to ensure they stay within the frame boundaries.

        Args:
            frame_width (int): The width of the frame.
            resolution_wh (int, int): The width and height of the frame.
            labels (List[str]): The list of text labels.
            label_properties (np.ndarray): An array of label properties, where each row
                            contains [x1, y1, x2, y2, text_height, ...].

        Returns:
            np.ndarray: The adjusted label properties.
        """
        adjusted_properties = label_properties.copy()

        # First, make sure the boxes don't go outside the frame
        adjusted_properties[:, :4] = snap_boxes(
            adjusted_properties[:, :4],
            resolution_wh,
        )

        # Apply the spread out algorithm to avoid box overlaps
        if len(labels) > 1:
            # Extract the box coordinates
            boxes = adjusted_properties[:, :4]
            # Use the spread_out_boxes function to adjust overlapping boxes
            spread_boxes = spread_out_boxes(boxes)
            # Update the properties with the spread out boxes
            adjusted_properties[:, :4] = spread_boxes

            # Additional check to ensure boxes are still within frame after spreading
            adjusted_properties[:, :4] = snap_boxes(
                adjusted_properties[:, :4], resolution_wh
            )

        return adjusted_properties


class BoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
            # Use color alpha for opacity if not 255
            effective_opacity = (color.a / 255.0) if color.a != 255 else self.opacity
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            # If opacity < 1, blend with background
            if effective_opacity < 1.0:
                overlay = scene.copy()
                cv2.rectangle(
                    img=overlay,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
                cv2.addWeighted(overlay, effective_opacity, scene, 1 - effective_opacity, 0, dst=scene)
        return scene


class OrientedBoxAnnotator(BaseAnnotator):
    """
    A class for drawing oriented bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.color_lookup: ColorLookup = color_lookup
        self.opacity = opacity

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
            # Use color alpha for opacity if not 255
            effective_opacity = (color.a / 255.0) if color.a != 255 else self.opacity
            cv2.rectangle(
                img=scene_with_boxes,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=-1,
            )
            # If opacity < 1, blend with background
            if effective_opacity < 1.0:
                overlay = scene_with_boxes.copy()
                cv2.rectangle(
                    img=overlay,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=color.as_bgr(),
                    thickness=-1,
                )
                cv2.addWeighted(overlay, effective_opacity, scene_with_boxes, 1 - effective_opacity, 0, dst=scene_with_boxes)

        return scene_with_boxes


class HaloAnnotator(BaseAnnotator):
    """
    A class for drawing Halos on an image using provided detections.

    !!! warning

        This annotator uses `sv.Detections.mask`.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup
        self.kernel_size: int = kernel_size

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.start_angle: int = start_angle
        self.end_angle: int = end_angle
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.corner_length: int = corner_length
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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

        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        radius: int = 4,
        position: Position = Position.CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        outline_thickness: int = 0,
        outline_color: Color | ColorPalette = Color.BLACK,
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
        self.color: Color | ColorPalette = color
        self.radius: int = radius
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup
        self.outline_thickness = outline_thickness
        self.outline_color: Color | ColorPalette = outline_color

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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


class LabelAnnotator(_BaseLabelAnnotator):
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        text_color: Color | ColorPalette = Color.WHITE,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            text_color (Union[Color, ColorPalette]): The color or color palette to use
                for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            text_offset (Tuple[int, int]): A tuple of 2D coordinates `(x, y)` to
                offset the text position from the anchor point, in pixels.
            border_radius (int): The radius to apply round edges. If the selected
                value is higher than the lower dimension, width or height, is clipped.
            smart_position (bool): Spread out the labels to avoid overlapping.
            max_line_length (Optional[int]): Maximum number of characters per line
                before wrapping the text. None means no wrapping.
        """
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        super().__init__(
            color=color,
            color_lookup=color_lookup,
            text_color=text_color,
            text_padding=text_padding,
            text_position=text_position,
            text_offset=text_offset,
            border_radius=border_radius,
            smart_position=smart_position,
            max_line_length=max_line_length,
        )

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        labels: list[str] | None = None,
        custom_color_lookup: np.ndarray | None = None,
    ) -> np.ndarray:
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
        validate_labels(labels, detections)

        labels = get_labels_text(detections, labels)
        label_properties = self._get_label_properties(detections, labels)

        if self.smart_position:
            xyxy = label_properties[:, :4]
            xyxy = spread_out_boxes(xyxy)
            label_properties[:, :4] = xyxy

            label_properties = self._adjust_labels_in_frame(
                (scene.shape[1], scene.shape[0]),
                labels,
                label_properties,
            )

        self._draw_labels(
            scene=scene,
            labels=labels,
            label_properties=label_properties,
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )

        return scene

    def _get_label_properties(
        self,
        detections: Detections,
        labels: list[str],
    ) -> np.ndarray:
        label_properties = []
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)

        for label, center_coordinates in zip(labels, anchors_coordinates):
            center_coordinates = (
                center_coordinates[0] + self.text_offset[0],
                center_coordinates[1] + self.text_offset[1],
            )

            wrapped_lines = wrap_text(label, self.max_line_length)
            line_heights = []
            line_widths = []

            for line in wrapped_lines:
                (text_w, text_h) = cv2.getTextSize(
                    text=line,
                    fontFace=CV2_FONT,
                    fontScale=self.text_scale,
                    thickness=self.text_thickness,
                )[0]
                line_heights.append(text_h)
                line_widths.append(text_w)

            # Get the maximum width and total height
            max_width = max(line_widths) if line_widths else 0
            total_height = (
                sum(line_heights) + (len(line_heights) - 1) * self.text_padding
            )

            # Add padding around all sides
            width_padded = max_width + 2 * self.text_padding
            height_padded = total_height + 2 * self.text_padding

            text_background_xyxy = resolve_text_background_xyxy(
                center_coordinates=center_coordinates,
                text_wh=(width_padded, height_padded),
                position=self.text_anchor,
            )

            label_properties.append(
                [
                    *text_background_xyxy,
                    total_height,
                ]
            )
        return np.array(label_properties).reshape(-1, 5)

    def _draw_labels(
        self,
        scene: np.ndarray,
        labels: list[str],
        label_properties: np.ndarray,
        detections: Detections,
        custom_color_lookup: np.ndarray | None,
    ) -> None:
        assert len(labels) == len(label_properties) == len(detections), (
            f"Number of label properties ({len(label_properties)}), "
            f"labels ({len(labels)}) and detections ({len(detections)}) "
            "do not match."
        )

        color_lookup = (
            custom_color_lookup
            if custom_color_lookup is not None
            else self.color_lookup
        )

        for idx, label_property in enumerate(label_properties):
            background_color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=idx,
                color_lookup=color_lookup,
            )
            text_color = resolve_color(
                color=self.text_color,
                detections=detections,
                detection_idx=idx,
                color_lookup=color_lookup,
            )

            box_xyxy = label_property[:4].astype(int)

            self.draw_rounded_rectangle(
                scene=scene,
                xyxy=box_xyxy,
                color=background_color.as_bgr(),
                border_radius=self.border_radius,
            )

            # Handle multiline text
            wrapped_lines = wrap_text(labels[idx], self.max_line_length)
            current_y = box_xyxy[1] + self.text_padding  # Start y position

            for line in wrapped_lines:
                if not line:
                    # Use a character with ascenders and descenders as height reference
                    (_, text_h) = cv2.getTextSize(
                        text="Tg",
                        fontFace=CV2_FONT,
                        fontScale=self.text_scale,
                        thickness=self.text_thickness,
                    )[0]
                    current_y += text_h + self.text_padding
                    continue

                (_, text_h) = cv2.getTextSize(
                    text=line,
                    fontFace=CV2_FONT,
                    fontScale=self.text_scale,
                    thickness=self.text_thickness,
                )[0]

                text_x = box_xyxy[0] + self.text_padding
                text_y = current_y + text_h  # Add height to get to text baseline

                cv2.putText(
                    img=scene,
                    text=line,
                    org=(text_x, text_y),
                    fontFace=CV2_FONT,
                    fontScale=self.text_scale,
                    color=text_color.as_bgr(),
                    thickness=self.text_thickness,
                    lineType=cv2.LINE_AA,
                )

                current_y += text_h + self.text_padding  # Move to next line position

    @staticmethod
    def draw_rounded_rectangle(
        scene: np.ndarray,
        xyxy: tuple[int, int, int, int],
        color: tuple[int, int, int],
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


class RichLabelAnnotator(_BaseLabelAnnotator):
    """
    A class for annotating labels on an image using provided detections,
    with support for Unicode characters by using a custom font.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        text_color: Color | ColorPalette = Color.WHITE,
        font_path: str | None = None,
        font_size: int = 10,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            text_color (Union[Color, ColorPalette]): The color to use for the text.
            font_path (Optional[str]): Path to the font file (e.g., ".ttf" or ".otf")
                to use for rendering text. If `None`, the default PIL font will be used.
            font_size (int): Font size for the text.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            text_offset (Tuple[int, int]): A tuple of 2D coordinates `(x, y)` to
                offset the text position from the anchor point, in pixels.
            border_radius (int): The radius to apply round edges. If the selected
                value is higher than the lower dimension, width or height, is clipped.
            smart_position (bool): Spread out the labels to avoid overlapping.
            max_line_length (Optional[int]): Maximum number of characters per line
                before wrapping the text. None means no wrapping.
        """
        self.font_path = font_path
        self.font_size = font_size
        self.font = self._load_font(font_size, font_path)
        super().__init__(
            color=color,
            color_lookup=color_lookup,
            text_color=text_color,
            text_padding=text_padding,
            text_position=text_position,
            text_offset=text_offset,
            border_radius=border_radius,
            smart_position=smart_position,
            max_line_length=max_line_length,
        )

    @ensure_pil_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        labels: list[str] | None = None,
        custom_color_lookup: np.ndarray | None = None,
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
        validate_labels(labels, detections)

        draw = ImageDraw.Draw(scene)
        labels = get_labels_text(detections, labels)
        label_properties = self._get_label_properties(draw, detections, labels)

        if self.smart_position:
            xyxy = label_properties[:, :4]
            xyxy = spread_out_boxes(xyxy)
            label_properties[:, :4] = xyxy

            label_properties = self._adjust_labels_in_frame(
                (scene.width, scene.height),
                labels,
                label_properties,
            )

        self._draw_labels(
            draw=draw,
            labels=labels,
            label_properties=label_properties,
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )

        return scene

    def _get_label_properties(
        self, draw: ImageDraw.ImageDraw, detections: Detections, labels: list[str]
    ) -> np.ndarray:
        label_properties = []

        anchor_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)

        for label, center_coordinates in zip(labels, anchor_coordinates):
            center_coordinates = (
                center_coordinates[0] + self.text_offset[0],
                center_coordinates[1] + self.text_offset[1],
            )

            wrapped_lines = wrap_text(label, self.max_line_length)

            # Calculate the total text height and maximum width
            max_width = 0
            total_height = 0

            for line in wrapped_lines:
                left, top, right, bottom = draw.textbbox((0, 0), line, font=self.font)
                line_width = right - left
                line_height = bottom - top

                max_width = max(max_width, line_width)
                total_height += line_height

            # Add inter-line spacing
            if len(wrapped_lines) > 1:
                total_height += (len(wrapped_lines) - 1) * self.text_padding

            width_padded = int(max_width + 2 * self.text_padding)
            height_padded = int(total_height + 2 * self.text_padding)

            text_background_xyxy = resolve_text_background_xyxy(
                center_coordinates=center_coordinates,
                text_wh=(width_padded, height_padded),
                position=self.text_anchor,
            )

            # Get the text origin offsets
            text_left, text_top, _, _ = draw.textbbox((0, 0), "Tg", font=self.font)

            label_properties.append([*text_background_xyxy, text_left, text_top])

        return np.array(label_properties).reshape(-1, 6)

    def _draw_labels(
        self,
        draw: ImageDraw.ImageDraw,
        labels: list[str],
        label_properties: np.ndarray,
        detections: Detections,
        custom_color_lookup: np.ndarray | None,
    ) -> None:
        assert len(labels) == len(label_properties) == len(detections), (
            f"Number of label properties ({len(label_properties)}), "
            f"labels ({len(labels)}) and detections ({len(detections)}) "
            "do not match."
        )
        color_lookup = (
            custom_color_lookup
            if custom_color_lookup is not None
            else self.color_lookup
        )

        for idx, label_property in enumerate(label_properties):
            background_color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=idx,
                color_lookup=color_lookup,
            )
            text_color = resolve_color(
                color=self.text_color,
                detections=detections,
                detection_idx=idx,
                color_lookup=color_lookup,
            )

            box_xyxy = label_property[:4].astype(int)
            text_left = label_property[4]
            text_top = label_property[5]

            # Draw the rounded rectangle background
            draw.rounded_rectangle(
                tuple(box_xyxy),
                radius=self.border_radius,
                fill=background_color.as_rgb(),
                outline=None,
            )

            # Draw each line of text
            wrapped_lines = wrap_text(labels[idx], self.max_line_length)
            x_position = box_xyxy[0] + self.text_padding - text_left
            y_position = box_xyxy[1] + self.text_padding - text_top

            for line in wrapped_lines:
                draw.text(
                    xy=(x_position, y_position),
                    text=line,
                    font=self.font,
                    fill=text_color.as_rgb(),
                )

                # Move to the next line position
                left, top, right, bottom = draw.textbbox((0, 0), line, font=self.font)
                line_height = bottom - top
                y_position += line_height + self.text_padding

    @staticmethod
    def _load_font(font_size: int, font_path: str | None):
        def load_default_font(size):
            try:
                return ImageFont.load_default(size)
            except TypeError:
                return ImageFont.load_default()

        if font_path is None:
            return load_default_font(font_size)

        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"Font path '{font_path}' not found. Using PIL's default font.")
            return load_default_font(font_size)


class IconAnnotator(BaseAnnotator):
    """
    A class for drawing an icon on an image, using provided detections.
    """

    def __init__(
        self,
        icon_resolution_wh: tuple[int, int] = (64, 64),
        icon_position: Position = Position.TOP_CENTER,
        offset_xy: tuple[int, int] = (0, 0),
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
        self, scene: ImageType, detections: Detections, icon_path: str | list[str]
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        position: Position = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        smooth: bool = False,
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
            smooth (bool): Smooth the trace lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Color | ColorPalette = color
        self.trace = Trace(max_size=trace_length, anchor=position)
        self.thickness = thickness
        self.smooth = smooth
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        detections = detections[detections.tracker_id != PENDING_TRACK_ID]

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
            spline_points = xy.astype(np.int32)

            if len(xy) > 3 and self.smooth:
                x, y = xy[:, 0], xy[:, 1]
                tck, u = splprep([x, y], s=20)
                x_new, y_new = splev(np.linspace(0, 1, 100), tck)
                spline_points = np.stack([x_new, y_new], axis=1).astype(np.int32)

            if len(xy) > 1:
                scene = cv2.polylines(
                    scene,
                    [spline_points],
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
        self.heat_mask: npt.NDArray[np.float32] | None = None

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
        hsv = np.full(scene.shape, 255, dtype=np.uint8)
        hsv[..., 0] = temp
        temp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        base: int = 10,
        height: int = 10,
        position: Position = Position.TOP_CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        outline_thickness: int = 0,
        outline_color: Color | ColorPalette = Color.BLACK,
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
        self.color: Color | ColorPalette = color
        self.base: int = base
        self.height: int = height
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup
        self.outline_thickness: int = outline_thickness
        self.outline_color: Color | ColorPalette = outline_color

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.color: Color | ColorPalette = color
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
        custom_color_lookup: np.ndarray | None = None,
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
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        border_color: Color = Color.BLACK,
        position: Position = Position.TOP_CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        border_thickness: int | None = None,
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
        self.color: Color | ColorPalette = color
        self.border_color: Color = border_color
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup

        self.border_thickness = (
            border_thickness
            if border_thickness is not None
            else int(0.15 * self.height)
        )

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
        custom_values: np.ndarray | None = None,
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
        anchor_xy: tuple[int, int], border_wh: tuple[int, int], position: Position
    ) -> tuple[tuple[int, int], tuple[int, int]]:
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
        custom_values: np.ndarray | list[float] | None, detections: Detections
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
        border_color: Color | ColorPalette = ColorPalette.DEFAULT,
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
        self.border_color: Color | ColorPalette = border_color
        self.border_thickness: int = border_thickness
        self.border_color_lookup: ColorLookup = border_color_lookup

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
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

        ![crop-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/crop-annotator-example.png)
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
        anchor: tuple[int, int], crop_wh: tuple[int, int], position: Position
    ) -> tuple[tuple[int, int], tuple[int, int]]:
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


class ComparisonAnnotator:
    """
    Highlights the differences between two sets of detections.
    Useful for comparing results from two different models, or the difference
    between a ground truth and a prediction.

    If present, uses the oriented bounding box data.
    Otherwise, if present, uses a mask.
    Otherwise, uses the bounding box data.
    """

    def __init__(
        self,
        color_1: Color = Color.RED,
        color_2: Color = Color.GREEN,
        color_overlap: Color = Color.BLUE,
        *,
        opacity: float = 0.75,
        label_1: str = "",
        label_2: str = "",
        label_overlap: str = "",
        label_scale: float = 1.0,
    ):
        """
        Args:
            color_1 (Color): Color of areas only present in the first set of
                detections.
            color_2 (Color): Color of areas only present in the second set of
                detections.
            color_overlap (Color): Color of areas present in both sets of detections.
            opacity (float): Annotator opacity, from `0` to `1`.
            label_1 (str): Label for the first set of detections.
            label_2 (str): Label for the second set of detections.
            label_overlap (str): Label for areas present in both sets of detections.
            label_scale (float): Controls how large the labels are.
        """

        self.color_1 = color_1
        self.color_2 = color_2
        self.color_overlap = color_overlap

        self.opacity = opacity
        self.label_1 = label_1
        self.label_2 = label_2
        self.label_overlap = label_overlap
        self.label_scale = label_scale
        self.text_thickness = int(self.label_scale + 1.2)

    @ensure_cv2_image_for_annotation
    def annotate(
        self, scene: ImageType, detections_1: Detections, detections_2: Detections
    ) -> ImageType:
        """
        Highlights the differences between two sets of detections.

        Args:
            scene (ImageType): The image where detections will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections_1 (Detections): The first set of detections or predictions.
            detections_2 (Detections): The second set of detections to compare or
                ground truth.

        Returns:
            The annotated image.

        Example:
            ```python
            import supervision as sv

            image = ...
            detections_1 = sv.Detections(...)
            detections_2 = sv.Detections(...)

            comparison_annotator = sv.ComparisonAnnotator()
            annotated_frame = comparison_annotator.annotate(
                scene=image.copy(),
                detections_1=detections_1,
                detections_2=detections_2
            )
            ```

        ![comparison-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/comparison-annotator-example.png)
        """
        assert isinstance(scene, np.ndarray)
        if detections_1.is_empty() and detections_2.is_empty():
            return scene

        use_obb = self._use_obb(detections_1, detections_2)
        use_mask = self._use_mask(detections_1, detections_2)

        if use_obb:
            mask_1 = self._mask_from_obb(scene, detections_1)
            mask_2 = self._mask_from_obb(scene, detections_2)

        elif use_mask:
            mask_1 = self._mask_from_mask(scene, detections_1)
            mask_2 = self._mask_from_mask(scene, detections_2)

        else:
            mask_1 = self._mask_from_xyxy(scene, detections_1)
            mask_2 = self._mask_from_xyxy(scene, detections_2)

        mask_overlap = mask_1 & mask_2
        mask_1 = mask_1 & ~mask_overlap
        mask_2 = mask_2 & ~mask_overlap

        color_layer = np.zeros_like(scene, dtype=np.uint8)
        color_layer[mask_overlap] = self.color_overlap.as_bgr()
        color_layer[mask_1] = self.color_1.as_bgr()
        color_layer[mask_2] = self.color_2.as_bgr()

        scene[mask_overlap] = (1 - self.opacity) * scene[
            mask_overlap
        ] + self.opacity * color_layer[mask_overlap]
        scene[mask_1] = (1 - self.opacity) * scene[mask_1] + self.opacity * color_layer[
            mask_1
        ]
        scene[mask_2] = (1 - self.opacity) * scene[mask_2] + self.opacity * color_layer[
            mask_2
        ]

        self._draw_labels(scene)

        return scene

    @staticmethod
    def _use_obb(detections_1: Detections, detections_2: Detections) -> bool:
        assert not detections_1.is_empty() or not detections_2.is_empty()
        is_obb_1 = ORIENTED_BOX_COORDINATES in detections_1.data
        is_obb_2 = ORIENTED_BOX_COORDINATES in detections_2.data
        return (
            (is_obb_1 and is_obb_2)
            or (is_obb_1 and detections_2.is_empty())
            or (detections_1.is_empty() and is_obb_2)
        )

    @staticmethod
    def _use_mask(detections_1: Detections, detections_2: Detections) -> bool:
        assert not detections_1.is_empty() or not detections_2.is_empty()
        is_mask_1 = detections_1.mask is not None
        is_mask_2 = detections_2.mask is not None
        return (
            (is_mask_1 and is_mask_2)
            or (is_mask_1 and detections_2.is_empty())
            or (detections_1.is_empty() and is_mask_2)
        )

    @staticmethod
    def _mask_from_xyxy(scene: np.ndarray, detections: Detections) -> np.ndarray:
        mask = np.zeros(scene.shape[:2], dtype=np.bool_)
        if detections.is_empty():
            return mask

        resolution_wh = scene.shape[1], scene.shape[0]
        polygons = xyxy_to_polygons(detections.xyxy)

        for polygon in polygons:
            polygon_mask = polygon_to_mask(polygon, resolution_wh=resolution_wh)
            mask |= polygon_mask.astype(np.bool_)
        return mask

    @staticmethod
    def _mask_from_obb(scene: np.ndarray, detections: Detections) -> np.ndarray:
        mask = np.zeros(scene.shape[:2], dtype=np.bool_)
        if detections.is_empty():
            return mask

        resolution_wh = scene.shape[1], scene.shape[0]

        for polygon in detections.data[ORIENTED_BOX_COORDINATES]:
            polygon_mask = polygon_to_mask(polygon, resolution_wh=resolution_wh)
            mask |= polygon_mask.astype(np.bool_)
        return mask

    @staticmethod
    def _mask_from_mask(scene: np.ndarray, detections: Detections) -> np.ndarray:
        mask = np.zeros(scene.shape[:2], dtype=np.bool_)
        if detections.is_empty():
            return mask
        assert detections.mask is not None

        for detections_mask in detections.mask:
            mask |= detections_mask.astype(np.bool_)
        return mask

    def _draw_labels(self, scene: np.ndarray) -> None:
        """
        Draw the labels, explaining what each color represents, with automatically
        computed positions.

        Args:
            scene (np.ndarray): The image where the labels will be drawn.
        """
        margin = int(50 * self.label_scale)
        gap = int(40 * self.label_scale)
        y0 = int(50 * self.label_scale)
        height = int(50 * self.label_scale)

        marker_size = int(20 * self.label_scale)
        padding = int(10 * self.label_scale)
        text_box_corner_radius = int(10 * self.label_scale)
        marker_corner_radius = int(4 * self.label_scale)
        text_scale = self.label_scale

        label_color_pairs = [
            (self.label_1, self.color_1),
            (self.label_2, self.color_2),
            (self.label_overlap, self.color_overlap),
        ]

        x0 = margin
        for text, color in label_color_pairs:
            if not text:
                continue

            (text_w, _) = cv2.getTextSize(
                text=text,
                fontFace=CV2_FONT,
                fontScale=self.label_scale,
                thickness=self.text_thickness,
            )[0]

            width = text_w + marker_size + padding * 4
            center_x = x0 + width // 2
            center_y = y0 + height // 2

            draw_rounded_rectangle(
                scene=scene,
                rect=Rect(x=x0, y=y0, width=width, height=height),
                color=Color.WHITE,
                border_radius=text_box_corner_radius,
            )

            draw_rounded_rectangle(
                scene=scene,
                rect=Rect(
                    x=x0 + padding,
                    y=center_y - marker_size / 2,
                    width=marker_size,
                    height=marker_size,
                ),
                color=color,
                border_radius=marker_corner_radius,
            )

            draw_text(
                scene,
                text,
                text_anchor=Point(x=center_x + marker_size, y=center_y),
                text_scale=text_scale,
                text_thickness=self.text_thickness,
            )

            x0 += width + gap
