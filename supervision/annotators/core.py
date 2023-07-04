from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List

import numpy as np
import cv2

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class BoxAnnotator(BaseAnnotator):
    def __init__(
            self,
            color: Union[Color, ColorPalette] = ColorPalette.default(),
            thickness: int = 2,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness

    def annotate(
            self,
            scene: np.ndarray,
            detections: Detections,
            color_by_track: bool = False,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
            color_by_track (bool): It allows to pick color by tracker id if present
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> box_annotator = sv.BoxAnnotator()
            >>> annotated_frame = box_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)

            if color_by_track:
                tracker_id = (
                    detections.tracker_id[i] if detections.tracker_id is not None else None
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
        color (Union[Color, ColorPalette]): The color to fill the mask, can be a single color or a color palette
        opacity (float): The opacity of the masks, between 0 and 1.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity

    def annotate(self, scene: np.ndarray, detections: Detections, color_by_track: bool = False,) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid

        Returns:
            np.ndarray: The image with the masks overlaid
        """
        if detections.mask is None:
            return scene

        for i in np.flip(np.argsort(detections.area)):
            if color_by_track:
                tracker_id = (
                    detections.tracker_id[i] if detections.tracker_id is not None else None
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


class LabelAnnotator(BaseAnnotator):

    def __init__(self, color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(self, scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        color_by_track: bool = False,) -> np.ndarray:
        """
         Draws bounding boxes on the frame using the detections provided.

         Args:
             scene (np.ndarray): The image on which the bounding boxes will be drawn
             detections (Detections): The detections for which the bounding boxes will be drawn
             labels (Optional[List[str]]): An optional list of labels corresponding to each detection. If `labels` are not provided, corresponding `class_id` will be used as label.
             skip_label (bool): Is set to `True`, skips bounding box label annotation.
         Returns:
             np.ndarray: The image with the bounding boxes drawn on it

         Example:
             ```python
             >>> import supervision as sv

             >>> classes = ['person', ...]
             >>> image = ...
             >>> detections = sv.Detections(...)

             >>> label_annotator = sv.LabelAnnotator()
             >>> labels = [
             ...     f"{classes[class_id]} {confidence:0.2f}"
             ...     for _, _, confidence, class_id, _
             ...     in detections
             ... ]
             >>> annotated_frame = label_annotator.annotate(
             ...     scene=image.copy(),
             ...     detections=detections,
             ...     labels=labels
             ... )
             ```
         """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            if skip_label:
                continue

            text = (
                f"{class_id}"
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


class PillowLabelAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class TrackAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class BoxMaskAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass
