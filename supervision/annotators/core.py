import os.path
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


class ColorMap(Enum):
    """
    Enum for annotator color mapping.
    """

    INDEX = "index"
    CLASS = "class"
    TRACK = "track"


def resolve_color_idx(
    detections: Detections, detection_idx: int, color_map: ColorMap = ColorMap.CLASS
) -> int:
    if detection_idx >= len(detections):
        raise ValueError(
            f"Detection index {detection_idx}"
            f"is out of bounds for detections of length {len(detections)}"
        )

    if color_map == ColorMap.INDEX:
        return detection_idx
    elif color_map == ColorMap.CLASS:
        if detections.class_id is None:
            raise ValueError(
                "Could not resolve color by class because"
                "Detections do not have class_id"
            )
        return detections.class_id[detection_idx]
    elif color_map == ColorMap.TRACK:
        if detections.tracker_id is None:
            raise ValueError(
                "Could not resolve color by track because"
                "Detections do not have tracker_id"
            )
        return detections.tracker_id[detection_idx]


def resolve_color(color: Union[Color, ColorPalette], idx: int) -> Color:
    if isinstance(color, ColorPalette):
        return color.by_idx(idx)
    else:
        return color


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class BoxLineAnnotator(BaseAnnotator):
    """
    Basic line bounding box annotator.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        color_map: str = "class",
    ):
        """
        Parameters:
            color (Union[Color, ColorPalette]): The color to use for
                annotating detections.
            thickness (int): The thickness of the bounding box lines.
            color_map (ColorMap): The color mapping to use for annotating detections.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which
                the bounding boxes will be drawn
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> box_line_annotator = sv.BoxLineAnnotator()
            >>> annotated_frame = box_line_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
        return scene


class MaskAnnotator(BaseAnnotator):
    """
    A class for overlaying masks on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to fill the mask,
            can be a single color or a color palette
        opacity (float): The opacity of the masks, between 0 and 1.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
        color_by_track: bool = False,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_by_track = color_by_track

    def annotate(
        self, scene: np.ndarray, detections: Detections, **kwargs
    ) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections,
            with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid

        Returns:
            np.ndarray: The image with the masks overlaid
        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> mask_annotator = sv.MaskAnnotator()
            >>> annotated_frame = mask_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        if detections.mask is None:
            return scene

        for i in np.flip(np.argsort(detections.area)):
            if self.color_by_track:
                tracker_id = (
                    detections.tracker_id[i]
                    if detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i

            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            mask = detections.mask[i]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()

            scene = np.where(
                np.expand_dims(mask, axis=-1),
                np.uint8(self.opacity * colored_mask + (1 - self.opacity) * scene),
                scene,
            )

        return scene


def default_label_formatter(
    detections: Detections,
) -> List[str]:
    return [str(class_id) for class_id in detections.class_id]


def build_label_formatter(classes: List[str]) -> Callable[[Detections], List[str]]:
    def default_label_formatter(detections: Detections) -> List[str]:
        return [classes[class_id] for class_id in detections.class_id]

    return default_label_formatter


class LabelAnnotator(BaseAnnotator):
    """
    A class for putting text on an image using provided detections.

    Attributes:
        color (Union[Color, ColorPalette]): The color to text on the image,
            can be a single color or a color palette
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        color_by_track: bool = False,
        classes: Optional[List[str]] = None,
        label_formatter: Optional[
            Callable[[Detections], List[str]]
        ] = default_label_formatter,
    ):
        """
        Args:
            color_by_track: pick color by tracker id
            classes: Optional list of class name
            label_formatter: Optional callback function for label generator,
                avoided if classes is provided
        """
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.color_by_track = color_by_track
        self.classes = classes
        self.label_formatter = label_formatter

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        """
        Draws text on the frame using the detections provided and label.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which
                the bounding boxes will be drawn
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)
            >>> label_annotator = sv.LabelAnnotator(classes=classes)
            >>> annotated_frame = label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections,
            ... )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        labels = None
        if self.classes:
            label_formatter = build_label_formatter(self.classes)
            labels = label_formatter(detections)
        else:
            labels = self.label_formatter(detections=detections)

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if self.color_by_track:
                tracker_id = (
                    detections.tracker_id[i]
                    if detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i

            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            text = (
                f"{idx}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


class BoxMaskAnnotator(BaseAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
        color_by_track: bool = False,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_by_track = color_by_track

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        """
        Overlays the rectangle masks on the given image based on the
            provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid
        Returns:
            np.ndarray: The image with the masks overlaid
        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> box_mask_annotator = sv.MaskAnnotator()
            >>> annotated_frame = box_mask_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        overlay_img = np.zeros_like(scene, np.uint8)
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if self.color_by_track:
                tracker_id = (
                    detections.tracker_id[i]
                    if detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=overlay_img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=-1,
            )

        mask = overlay_img.astype(bool)
        scene[mask] = cv2.addWeighted(
            scene, self.opacity, overlay_img, 1 - self.opacity, 0
        )[mask]
        return scene


def default_label_formatter(detections: Detections) -> List[str]:
    return [str(class_id) for class_id in detections.class_id]


class LabelAdvancedAnnotator(BaseAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_color: Color = Color.black(),
        text_padding: int = 20,
        color_by_track: bool = False,
        font: Optional[str] = None,
        font_size: Optional[int] = 15,
    ):
        if font and os.path.exists(font):
            self.font = ImageFont.truetype(font, font_size)
        else:
            self.font = ImageFont.load_default()
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Color = text_color
        self.text_padding: int = text_padding
        self.color_by_track = color_by_track

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Draws text on the frame using the detections provided and label.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which
                the bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels corresponding
                to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> pil_label_annotator = sv.LabelAdvancedAnnotator()
            >>> labels = [
            ...     f"{classes[class_id]} {confidence:0.2f}"
            ...     for _, _, confidence, class_id, _
            ...     in detections
            ... ]
            >>> annotated_frame = pil_label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections,
            ...     labels=labels,
            ... )
            ```
        """
        pil_image = Image.fromarray(scene)
        draw = ImageDraw.Draw(pil_image)
        text_color = "#fff"

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if self.color_by_track:
                tracker_id = (
                    detections.tracker_id[i]
                    if detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i

            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            text = (
                f"{idx}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_bbox = draw.textbbox((x1, y1), text, font=self.font)

            text_height = text_bbox[3] - text_bbox[1]
            text_width = text_bbox[2] - text_bbox[0]

            text_x = x1 + self.text_padding / 2
            text_y = y1 - self.text_padding / 2 - text_height

            text_background_x1 = x1
            text_background_y1 = y1 - self.text_padding / 2 - text_height

            text_background_x2 = x1 + 2 * self.text_padding / 2 + text_width
            text_background_y2 = y1  # correct

            draw.rectangle(
                (
                    text_background_x1,
                    text_background_y1,
                    text_background_x2,
                    text_background_y2,
                ),
                fill=color.as_bgr(),
            )
            draw.text((text_x, text_y), text, font=self.font, fill=text_color)

        scene = np.asarray(pil_image)
        return scene


class BoxCornerAnnotator(BaseAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        color_by_track: bool = False,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_by_track = color_by_track

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ):
        """
        Draws cornered bounding boxes on the frame using the detections provided.
        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which
                the bounding boxes will be drawn
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it
        Example:
            ```python
            >>> import supervision as sv
            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)
            >>> corner_box_annotator = sv.CorneredBoxAnotator()
            >>> annotated_frame = corner_box_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections,
            ...     labels=labels
            ... )
            ```
        """

        line_thickness = self.thickness + 2
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if self.color_by_track:
                tracker_id = (
                    detections.tracker_id[i]
                    if detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            box_width = x2 - x1
            box_height = y2 - y1

            cv2.line(
                scene,
                (x1, y1),
                (x1 + int(0.2 * box_width), y1),
                color.as_bgr(),
                thickness=line_thickness,
            )
            cv2.line(
                scene,
                (x2 - int(0.2 * box_width), y1),
                (x2, y1),
                color.as_bgr(),
                thickness=line_thickness,
            )

            cv2.line(
                scene,
                (x1, y2),
                (x1 + int(0.2 * box_width), y2),
                color.as_bgr(),
                thickness=line_thickness,
            )
            cv2.line(
                scene,
                (x2 - int(0.2 * box_width), y2),
                (x2, y2),
                color.as_bgr(),
                thickness=line_thickness,
            )

            cv2.line(
                scene,
                (x1, y1),
                (x1, y1 + int(0.2 * box_height)),
                color.as_bgr(),
                thickness=line_thickness,
            )
            cv2.line(
                scene,
                (x2, y1 + int(0.2 * box_height)),
                (x2, y1),
                color.as_bgr(),
                thickness=line_thickness,
            )

            cv2.line(
                scene,
                (x1, y2 - int(0.2 * box_height)),
                (x1, y2),
                color.as_bgr(),
                thickness=line_thickness,
            )
            cv2.line(
                scene,
                (x2, y2 - int(0.2 * box_height)),
                (x2, y2),
                color.as_bgr(),
                thickness=line_thickness,
            )

        return scene


class EllipseAnnotator(BaseAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        color_by_track: bool = False,
        start_angle: int = -45,
        end_angle: int = 360,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_by_track = color_by_track
        self.start_angle = start_angle
        self.end_angle = end_angle

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ):
        """
        Draws ellipse at bottom of the objects on the frame using the detections
            provided.
        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which
                the bounding boxes will be drawn
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it
        Example:
            ```python
            >>> import supervision as sv
            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)
            >>> ellipse_annotator = sv.EllipseAnotator()
            >>> annotated_frame = ellipse_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections,
            ...     labels=labels
            ... )
            ```
        """

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if self.color_by_track:
                tracker_id = (
                    detections.tracker_id[i]
                    if detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
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


class TraceAnnotator(BaseAnnotator):
    """
    A class for drawing trajectory of a tracker on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the trajectory,
            can be a single color or a color palette
        color_by_track (bool): Whther to use tracker id to pick the color
        position (Optional[Position]): Choose position of trajectory such as
            center position, top left corner, etc
        trace_length (int): Length of the previous points
        thickness (int): thickness of the line
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        color_by_track: bool = False,
        position: Optional[Position] = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.color_by_track = color_by_track
        self.position = position
        self.tracker_storage = defaultdict(lambda: [])
        self.trace_length = trace_length
        self.thickness = thickness

    def annotate(
        self, scene: np.ndarray, detections: Detections, **kwargs
    ) -> np.ndarray:
        """
        Draw the object trajectory based on history of tracked objects

        Args:
            scene (np.ndarray): The image on which the trace will be drawn
            detections (Detections): The detections for trajectory and points

        Returns:
            np.ndarray: The image with the masks overlaid
        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> trace_annotator = sv.TraceAnnotator()
            >>> annotated_frame = trace_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        if detections.tracker_id is None:
            return scene

        anchor_points = detections.get_anchor_coordinates(anchor=self.position)

        for i, tracker_id in enumerate(detections.tracker_id):
            track = self.tracker_storage[tracker_id]
            track.append((anchor_points[i][0], anchor_points[i][1]))
            if len(track) > self.trace_length:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

            if self.color_by_track:
                idx = tracker_id if tracker_id is not None else i
            else:
                class_id = (
                    detections.class_id[i] if detections.class_id is not None else None
                )
                idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.polylines(
                scene,
                [points],
                isClosed=False,
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        return scene
