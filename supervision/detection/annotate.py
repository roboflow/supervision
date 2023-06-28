from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette


class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box, can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box, default is 1
        text_padding (int): The padding around the text on the bounding box, default is 5

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
        skip_label: bool = False,
    ) -> np.ndarray:
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

            >>> box_annotator = sv.BoxAnnotator()
            >>> labels = [
            ...     f"{classes[class_id]} {confidence:0.2f}"
            ...     for _, _, confidence, class_id, _
            ...     in detections
            ... ]
            >>> annotated_frame = box_annotator.annotate(
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
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
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

    # Why a font file is needed?
    # Non-ascii characters are not rendered properly with cv2.putText, so we need to use PIL
    # PIL uses truetype fonts, so we need to provide a true type font file.

    # Why not embed the font file in the package?
    # The font file is not a part of the package because of licensing issues.

    # Why not use a default font file?
    # The default font file is not provided because of licensing issues. And the default font file may not be available on all systems or not suitable for all languages.

    # Why a font size is needed?
    # As for unicode support, different characters have different best visual sizes. So, we need to provide a font size to render the text properly.
    def annotation_pil(
        self,
        scene: Image.Image,
        detections: Detections,
        font_file: str,
        labels: Optional[List[str]] = None,
        font_size: int = 15,
        skip_label: bool = False,
    ) -> Image.Image:
        """
        Draws bounding boxes on the frame using the detections provided with **PIL**. If non-ascii labels are provided, you should use this method.
        For rendering non-ascii labels, you should provide a true type font file path to `font_file` argument.

        Args:
            scene (Image.Image): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
            font_file (str): The true type font file path
            font_size (int): The font size of the label.
            labels (Optional[List[str]]): An optional list of labels corresponding to each detection. If `labels` are not provided, corresponding `class_id` will be used as label.
        Returns:
            Image.Image: The image with the bounding boxes drawn on it

        Example:
            ```python
            >>> import supervision as sv
            >>> from PIL import Image

            >>> classes = ['person', ...]
            >>> image = Image.open(...)
            >>> font_file = Path('path/to/font/file').as_posix()
            >>> detections = sv.Detections(...)

            >>> box_annotator = sv.BoxAnnotator()
            >>> labels = [
            ...     '世界', '你好', 'hello', 'world'
            ... ]
            >>> annotated_frame = box_annotator.annotate_pil(
            ...     scene=image,
            ...     detections=detections,
            ...     font_file=font_file,
            ...     labels=labels
            ... )
            ```
        """
        color = ColorPalette.default()
        padding_size = 3
        text_color = "#fff"
        draw = ImageDraw.Draw(scene)
        font = ImageFont.truetype(font_file, font_size)
        for i in range(len(detections)):
            outline_color = color.by_idx(detections.class_id[i]).as_rgb()
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            draw.rectangle((x1, y1, x2, y2), fill=None, outline=outline_color)
            if skip_label:
                continue
            if labels:
                text = str(labels[i])
            else:
                text = str(detections.class_id[i])
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            text_x, text_y = x1, y1 - text_height - padding_size
            text_bg_y = text_y + padding_size
            draw.rectangle(
                (
                    text_x,
                    text_bg_y,
                    text_x + text_bbox[2] - text_bbox[0],
                    text_bg_y + text_bbox[3] - text_bbox[1],
                ),
                fill=outline_color,
            )
            draw.text((text_x, text_y), text, font=font, fill=text_color)

        return scene


class MaskAnnotator:
    """
    A class for overlaying masks on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to fill the mask, can be a single color or a color palette
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
    ):
        self.color: Union[Color, ColorPalette] = color

    def annotate(
        self, scene: np.ndarray, detections: Detections, opacity: float = 0.5
    ) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid
            opacity (float): The opacity of the masks, between 0 and 1, default is 0.5

        Returns:
            np.ndarray: The image with the masks overlaid
        """
        if detections.mask is None:
            return scene

        for i in np.flip(np.argsort(detections.area)):
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
                np.uint8(opacity * colored_mask + (1 - opacity) * scene),
                scene,
            )

        return scene
