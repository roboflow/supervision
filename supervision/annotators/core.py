import os.path
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import cv2
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw, ImageFont

from supervision.detection.core import Detections
from supervision.detection.track import TrackStorage
from supervision.draw.color import Color, ColorPalette


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class BoxAnnotator(BaseAnnotator):
    """
    Basic bounding box annotation class
    """

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

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        color_by_track: bool = False,
    ) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections, with a specified opacity.

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
            if color_by_track:
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


class LabelAnnotator(BaseAnnotator):
    """
    A class for putting text on an image using provided detections.

    Attributes:
        color (Union[Color, ColorPalette]): The color to text on the image, can be a single color or a color palette
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        color_by_track: bool = False,
    ) -> np.ndarray:
        """
        Draws text on the frame using the detections provided and label.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels corresponding to each detection. If `labels` are not provided, corresponding `class_id` will be used as label.
            color_by_track (bool): If set then color will be chosen by tracker id if provided
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
            if color_by_track:
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
    ):
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        color_by_track: bool = False,
    ) -> np.ndarray:
        """
        Overlays the rectangle masks on the given image based on the provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid
            color_by_track (bool): If set then color will be chosen by tracker id if provided
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
            if color_by_track:
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


class TrackAnnotator(BaseAnnotator):
    """
    Initialize TrackAnnotator
    """

    def __init__(
        self,
        tracks: TrackStorage,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
    ):
        self.track_storage = tracks
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.boundry_tolerance = 20

    def annotate(self, scene: np.ndarray, color_by_track: bool = False) -> np.ndarray:
        """
        Draws the object trajectory on the frame using the trace provided.
        Attributes:
            scene (np.ndarray): The image on which the object trajectories will be drawn.
            tracks (TrackStorage): The track storage that will be used to draw the previous and current position.

        Returns:
            np.ndarray: The image with the object trajectories on it.
            ```python
            >>> import supervision as sv
            >>> track_storage = sv.TrackStorage()
            >>> track_annotator = sv.TrackAnnotator(track_storage)
            >>> for frame in sv.get_video_frames_generator(source_path='source_video.mp4'):
            >>>     detections = sv.Detections(...)
            >>>     tracked_objects = tracker(...)
            >>>     tracked_detections = sv.Detections(tracked_objects)
            >>>     track_storage.update(tracked_detections)
            >>>     track_annotator.annotate(scene)
        """
        img_h, img_w, _ = scene.shape
        unique_ids = np.unique(self.track_storage.storage[:, -1])
        for unique_id in unique_ids:
            valid = np.where(self.track_storage.storage[:, -1] == unique_id)[0]

            frames = self.track_storage.storage[valid, 0]
            latest_frame = np.argmax(frames)
            points_to_draw = self.track_storage.storage[valid, 1:3]

            n_pts = points_to_draw.shape[0]
            headx, heady = int(points_to_draw[latest_frame][0]), int(
                points_to_draw[latest_frame][1]
            )

            if headx > self.boundry_tolerance and heady > self.boundry_tolerance:
                if color_by_track:
                    idx = int(unique_id)
                else:
                    idx = int(self.track_storage.storage[0, -2])
                color = (
                    self.color.by_idx(idx)
                    if isinstance(self.color, ColorPalette)
                    else self.color
                )

                for i in range(n_pts - 1):
                    px, py = int(points_to_draw[i][0]), int(points_to_draw[i][1])
                    qx, qy = int(points_to_draw[i + 1][0]), int(
                        points_to_draw[i + 1][1]
                    )
                    cv2.line(scene, (px, py), (qx, qy), color.as_bgr(), self.thickness)
                    scene = cv2.circle(
                        scene, (headx, heady), int(10), color.as_bgr(), thickness=-1
                    )
        return scene


class PillowLabelAnnotator(BaseAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_color: Color = Color.black(),
        text_padding: int = 20,
    ):
        self.font = ImageFont.load_default()
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Color = text_color
        self.text_padding: int = text_padding

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        color_by_track: bool = False,
        font: Optional[str] = None,
        font_size: Optional[int] = 15,
    ) -> np.ndarray:
        """
        Draws text on the frame using the detections provided and label.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels corresponding to each detection. If `labels` are not provided, corresponding `class_id` will be used as label.
            color_by_track (bool): If set then color will be chosen by tracker id if provided
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> pil_label_annotator = sv.PillowLabelAnnotator()
            >>> labels = [
            ...     f"{classes[class_id]} {confidence:0.2f}"
            ...     for _, _, confidence, class_id, _
            ...     in detections
            ... ]
            >>> annotated_frame = pil_label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections,
            ...     labels=labels,
            ...     font=FONT_FILE_PATH,
            ... )
            ```
        """
        if font and os.path.exists(font):
            self.font = ImageFont.truetype(font, font_size)

        pil_image = Image.fromarray(scene)
        draw = ImageDraw.Draw(pil_image)
        text_color = "#fff"

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if color_by_track:
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


class CorneredBoxAnotator(BaseAnnotator):
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
    ):
        """
        Draws cornered bounding boxes on the frame using the detections provided.
        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
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
            if color_by_track:
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
            #
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
