from math import sqrt
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from supervision.annotators.base import BaseAnnotator
from supervision.annotators.utils import (
    ColorMap,
    Trace,
    resolve_color,
    resolve_color_idx,
)
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


class BoundingBoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with bounding boxes based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> bounding_box_annotator = sv.BoundingBoxAnnotator()
            >>> annotated_frame = bounding_box_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![bounding-box-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/bounding-box-annotator-example-purple.png)
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
    A class for drawing masks on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with masks based on the provided detections.

        Args:
            scene (np.ndarray): The image where masks will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> mask_annotator = sv.MaskAnnotator()
            >>> annotated_frame = mask_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/mask-annotator-example-purple.png)
        """
        if detections.mask is None:
            return scene

        for detection_idx in np.flip(np.argsort(detections.area)):
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)
            mask = detections.mask[detection_idx]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()
            scene[mask] = cv2.addWeighted(
                colored_mask, self.opacity, scene, 1 - self.opacity, 0
            )[mask]

        return scene


class BoxMaskAnnotator(BaseAnnotator):
    """
    A class for drawing box masks on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.color_map: ColorMap = ColorMap(color_map)
        self.opacity = opacity

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with box masks based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> box_mask_annotator = sv.BoxMaskAnnotator()
            >>> annotated_frame = box_mask_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![box-mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-mask-annotator-example-purple.png)
        """
        mask_image = scene.copy()
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
                thickness=-1,
            )
        scene = cv2.addWeighted(
            scene, self.opacity, mask_image, 1 - self.opacity, gamma=0
        )
        return scene


class HaloAnnotator(BaseAnnotator):
    """
    A class for drawing Halos on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.8,
        kernel_size: int = 40,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
            kernel_size (int): The size of the average pooling kernel used for creating
                the halo.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_map: ColorMap = ColorMap(color_map)
        self.kernel_size: int = kernel_size

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with halos based on the provided detections.

        Args:
            scene (np.ndarray): The image where masks will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> halo_annotator = sv.HaloAnnotator()
            >>> annotated_frame = halo_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![halo-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/halo-annotator-example-purple.png)
        """
        if detections.mask is None:
            return scene
        colored_mask = np.zeros_like(scene, dtype=np.uint8)
        fmask = np.array([False] * scene.shape[0] * scene.shape[1]).reshape(
            scene.shape[0], scene.shape[1]
        )

        for detection_idx in np.flip(np.argsort(detections.area)):
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)
            mask = detections.mask[detection_idx]
            fmask = np.logical_or(fmask, mask)
            color_bgr = color.as_bgr()
            colored_mask[mask] = color_bgr

        colored_mask = cv2.blur(colored_mask, (self.kernel_size, self.kernel_size))
        colored_mask[fmask] = [0, 0, 0]
        gray = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
        alpha = self.opacity * gray / gray.max()
        alpha_mask = alpha[:, :, np.newaxis]
        scene = np.uint8(scene * (1 - alpha_mask) + colored_mask * self.opacity)
        return scene


class EllipseAnnotator(BaseAnnotator):
    """
    A class for drawing ellipses on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        start_angle: int = -45,
        end_angle: int = 235,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the ellipse lines.
            start_angle (int): Starting angle of the ellipse.
            end_angle (int): Ending angle of the ellipse.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.start_angle: int = start_angle
        self.end_angle: int = end_angle
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with ellipses based on the provided detections.

        Args:
            scene (np.ndarray): The image where ellipses will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> ellipse_annotator = sv.EllipseAnnotator()
            >>> annotated_frame = ellipse_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![ellipse-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/ellipse-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)

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
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 4,
        corner_length: int = 15,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the corner lines.
            corner_length (int): Length of each corner line.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.corner_length: int = corner_length
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the given scene with box corners based on the provided detections.

        Args:
            scene (np.ndarray): The image where box corners will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> corner_annotator = sv.BoxCornerAnnotator()
            >>> annotated_frame = corner_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![box-corner-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-corner-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)
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
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 4,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the circle line.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        """
        Annotates the given scene with circles based on the provided detections.

        Args:
            scene (np.ndarray): The image where box corners will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> circle_annotator = sv.CircleAnnotator()
            >>> annotated_frame = circle_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```


        ![circle-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/circle-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2)

            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )

            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            cv2.circle(
                img=scene,
                center=center,
                radius=int(distance),
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        return scene


class LabelAnnotator:
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_position: Position = text_position
        self.color_map: ColorMap = ColorMap(color_map)

    @staticmethod
    def resolve_text_background_xyxy(
        detection_xyxy: Tuple[int, int, int, int],
        text_wh: Tuple[int, int],
        text_padding: int,
        position: Position,
    ) -> Tuple[int, int, int, int]:
        padded_text_wh = (text_wh[0] + 2 * text_padding, text_wh[1] + 2 * text_padding)
        x1, y1, x2, y2 = detection_xyxy
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if position == Position.TOP_LEFT:
            return x1, y1 - padded_text_wh[1], x1 + padded_text_wh[0], y1
        elif position == Position.TOP_RIGHT:
            return x2 - padded_text_wh[0], y1 - padded_text_wh[1], x2, y1
        elif position == Position.TOP_CENTER:
            return (
                center_x - padded_text_wh[0] // 2,
                y1 - padded_text_wh[1],
                center_x + padded_text_wh[0] // 2,
                y1,
            )
        elif position == Position.CENTER:
            return (
                center_x - padded_text_wh[0] // 2,
                center_y - padded_text_wh[1] // 2,
                center_x + padded_text_wh[0] // 2,
                center_y + padded_text_wh[1] // 2,
            )
        elif position == Position.BOTTOM_LEFT:
            return x1, y2, x1 + padded_text_wh[0], y2 + padded_text_wh[1]
        elif position == Position.BOTTOM_RIGHT:
            return x2 - padded_text_wh[0], y2, x2, y2 + padded_text_wh[1]
        elif position == Position.BOTTOM_CENTER:
            return (
                center_x - padded_text_wh[0] // 2,
                y2,
                center_x + padded_text_wh[0] // 2,
                y2 + padded_text_wh[1],
            )

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: List[str] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided detections.

        Args:
            scene (np.ndarray): The image where labels will be drawn.
            detections (Detections): Object detections to annotate.
            labels (List[str]): Optional. Custom labels for each detection.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            >>> annotated_frame = label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![label-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/label-annotator-example-purple.png)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection_idx in range(len(detections)):
            detection_xyxy = detections.xyxy[detection_idx].astype(int)
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)
            text = (
                f"{detections.class_id[detection_idx]}"
                if (labels is None or len(detections) != len(labels))
                else labels[detection_idx]
            )
            text_wh = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_background_xyxy = self.resolve_text_background_xyxy(
                detection_xyxy=detection_xyxy,
                text_wh=text_wh,
                text_padding=self.text_padding,
                position=self.text_position,
            )

            text_x = text_background_xyxy[0] + self.text_padding
            text_y = text_background_xyxy[1] + self.text_padding + text_wh[1]

            cv2.rectangle(
                img=scene,
                pt1=(text_background_xyxy[0], text_background_xyxy[1]),
                pt2=(text_background_xyxy[2], text_background_xyxy[3]),
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

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        """
        Annotates the given scene by blurring regions based on the provided detections.

        Args:
            scene (np.ndarray): The image where blurring will be applied.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> blur_annotator = sv.BlurAnnotator()
            >>> annotated_frame = blur_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![blur-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/blur-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            roi = scene[y1:y2, x1:x2]

            roi = cv2.blur(roi, (self.kernel_size, self.kernel_size))
            scene[y1:y2, x1:x2] = roi

        return scene


class TraceAnnotator:
    """
    A class for drawing trace paths on an image based on detection coordinates.

    !!! warning

        This annotator utilizes the `tracker_id`. Read
        [here](https://supervision.roboflow.com/trackers/) to learn how to plug
        tracking into your inference pipeline.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        position: Optional[Position] = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        color_map: str = "class",
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color to draw the trace, can be
                a single color or a color palette.
            position (Optional[Position]): The position of the trace.
                Defaults to `CENTER`.
            trace_length (int): The maximum length of the trace in terms of historical
                points. Defaults to `30`.
            thickness (int): The thickness of the trace lines. Defaults to `2`.
            color_map (str): Strategy for mapping colors to annotations.
                Options are `index`, `class`, or `track`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.position = position
        self.trace = Trace(max_size=trace_length)
        self.thickness = thickness
        self.color_map: ColorMap = ColorMap(color_map)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Draws trace paths on the frame based on the detection coordinates provided.

        Args:
            scene (np.ndarray): The image on which the traces will be drawn.
            detections (Detections): The detections which include coordinates for
                which the traces will be drawn.

        Returns:
            np.ndarray: The image with the trace paths drawn on it.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> trace_annotator = sv.TraceAnnotator()
            >>> annotated_frame = trace_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![trace-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/trace-annotator-example-purple.png)
        """
        self.trace.put(detections)

        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            idx = resolve_color_idx(
                detections=detections,
                detection_idx=detection_idx,
                color_map=self.color_map,
            )
            color = resolve_color(color=self.color, idx=idx)
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
