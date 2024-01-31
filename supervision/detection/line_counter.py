import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect, Vector


class LineZone:
    """
    Count the number of objects that cross a line.

        This class is responsible for counting the number of objects that cross a
    predefined line.

    !!! warning

        LineZone utilizes the `tracker_id`. Read
        [here](https://supervision.roboflow.com/trackers/) to learn how to plug
        tracking into your inference pipeline.

    Attributes:
        in_count (int): The number of objects that have crossed the line from outside
            to inside.
        out_count (int): The number of objects that have crossed the line from inside
            to outside.
    """

    def __init__(
        self,
        start: Point,
        end: Point,
        trigger_in: bool = True,
        trigger_out: bool = True,
    ):
        """
        Args:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            trigger_in (bool): Count object crossing in the line.
            trigger_out (bool): Count object crossing out the line.
        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        self.trigger_in = trigger_in
        self.trigger_out = trigger_out

    def trigger(self, detections: Detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the `in_count` and `out_count` based on the objects that cross the line.

        Args:
            detections (Detections): A list of detections for which to update the
                counts.

        Returns:
            A tuple of two boolean NumPy arrays. The first array indicates which
                detections have crossed the line from outside to inside. The second
                array indicates which detections have crossed the line from inside to
                outside.
        """
        crossed_in = np.full(len(detections), False)
        crossed_out = np.full(len(detections), False)

        for i, (xyxy, _, confidence, class_id, tracker_id) in enumerate(detections):
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]

            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
                crossed_in[i] = True
            else:
                self.out_count += 1
                crossed_out[i] = True

        return crossed_in, crossed_out


class LineZoneAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: Optional[str] = None,
        custom_out_text: Optional[str] = None,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        Attributes:
            thickness (float): The thickness of the line that will be drawn.
            color (Color): The color of the line that will be drawn.
            text_thickness (float): The thickness of the text that will be drawn.
            text_color (Color): The color of the text that will be drawn.
            text_scale (float): The scale of the text that will be drawn.
            text_offset (float): The offset of the text that will be drawn.
            text_padding (int): The padding of the text that will be drawn.

        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.custom_in_text: str = custom_in_text
        self.custom_out_text: str = custom_out_text

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineZone): The line counter object used to annotate.

        Returns:
            np.ndarray: The image with the line drawn on it.

        """

        # Draw line.
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=self.thickness,
            color=self.color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # Create in/out text.
        in_text = (
            f"{self.custom_in_text}: {line_counter.in_count}"
            if self.custom_in_text is not None
            else f"in: {line_counter.in_count}"
        )
        out_text = (
            f"{self.custom_out_text}: {line_counter.out_count}"
            if self.custom_out_text is not None
            else f"out: {line_counter.out_count}"
        )

        if line_counter.trigger_in:
            frame = self._annotate_count(frame, line_counter, in_text, text_over=True)
        if line_counter.trigger_out:
            frame = self._annotate_count(frame, line_counter, out_text, text_over=False)

        return frame

    def _annotate_count(
        self,
        frame: np.ndarray,
        line_counter: LineZone,
        text: str,
        text_over: bool = True,
    ):
        """
        Draws the counter for in/out counts aligned to the line.

        Attributes:
            text (str): Line of text to annotate alongside the count.
            text_over (bool): Position of the text over/under the line.

        Returns:
            np.ndarray: Annotated frame.
        """
        
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        background_dim = max(text_width, text_height) + 30

        text_background_img = self._create_background_img(background_dim)
        box_background_img = text_background_img.copy()

        text_position = self._get_text_position(background_dim, text_width, text_height)

        box_img = self._draw_box(
            box_background_img, text_width, text_height, text_position
        )
        text_img = self._draw_text(text_background_img, text, text_position)

        box_img_rotated = self._rotate_img(box_img, line_counter)
        text_img_rotated = self._rotate_img(text_img, line_counter)

        img_position = self._get_img_position(
            line_counter, text_width, text_height, text_over
        )

        img_bbox = self._get_img_bbox(box_img_rotated, frame, img_position)

        box_img_rotated = self._trim_img(box_img_rotated, frame, img_bbox)
        text_img_rotated = self._trim_img(text_img_rotated, frame, img_bbox)

        frame = self._annotate_box(frame, box_img_rotated, img_bbox)
        frame = self._annotate_text(frame, text_img_rotated, img_bbox)

        return frame

    def _create_background_img(self, background_dim: int):
        """
        Create squared background image to place text or text-box.

        Attributes:
            background_dim (int): Dimension of the squared background image.

        Returns:
            np.ndarray: Squared array representing an empty background image.
        """
        return np.zeros((background_dim, background_dim), dtype=np.uint8)

    def _get_text_position(self, background_dim: int, text_width: int, text_height: int):
        """
        Get insertion point to center text in background image.

        Attributes:
            background_dim (int): Dimension of the squared background image.
            text_width (int): Text width.
            text_height (int): Text height.

        Returns:
            (int, int): xy point to center text insertion.
        """
        text_position = (
            (background_dim // 2) - (text_width // 2),
            (background_dim // 2) + (text_height // 2),
        )

        return text_position

    def _draw_box(
        self,
        box_background_img: np.ndarray,
        text_width: int,
        text_height: int,
        text_position: tuple,
    ):
        """
        Draw text-box centered in the background image.

        Attributes:
            box_background_img (np.ndarray): Empty background image.
            text_width (int): Text width.
            text_height (int): Text height.
            text_position (int, int): xy point to center text insertion.

        Returns:
            np.ndarray: Background image with text-box drawed in it.
        """
        box = Rect(
            x=text_position[0],
            y=text_position[1] - text_height,
            width=text_width,
            height=text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            box_background_img,
            box.top_left.as_xy_int_tuple(),
            box.bottom_right.as_xy_int_tuple(),
            (255, 255, 255),
            -1,
        )

        return box_background_img

    def _draw_text(
        self, text_background_img: np.ndarray, text: str, text_position: tuple
    ):
        """
        Draw text-box centered in the background image.

        Attributes:
            text_background_img (np.ndarray): Empty background image.
            text (str): Text to draw in the background image.
            text_position (int, int): xy insertion point to center text in background.

        Returns:
            np.ndarray: Background image with text drawed in it.
        """
        cv2.putText(
            text_background_img,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            (255, 255, 255),
            self.text_thickness,
            cv2.LINE_AA,
        )

        return text_background_img

    def _get_line_angle(self, line_counter: LineZone):
        """
        Calculate the line counter angle using trigonometry.

        Attributes:
            line_counter (LineZone): The line counter object used to annotate.

        Returns:
            float: Line counter angle.
        """
        start_point = line_counter.vector.start.as_xy_int_tuple()
        end_point = line_counter.vector.end.as_xy_int_tuple()

        try:
            line_angle = math.degrees(
                math.atan(
                    (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
                )
            )
            if (end_point[0] - start_point[0]) < 0:
                line_angle = 180 + line_angle
        except ZeroDivisionError:
            # Add support for vertical lines.
            line_angle = 90
            if (end_point[1] - start_point[1]) < 0:
                line_angle = 180 + line_angle

        return line_angle

    def _rotate_img(self, img: np.ndarray, line_counter: LineZone):
        """
        Rotate img using line counter angle.

        Attributes:
            img (np.ndarray): Original image to rotate.
            line_counter (LineZone): The line counter object used to annotate.

        Returns:
            np.ndarray: Image with the same shape as input but with rotated content.
        """
        img_dim = img.shape[0]

        line_angle = self._get_line_angle(line_counter)

        rotation_center = ((img_dim // 2), (img_dim // 2))
        rotation_angle = -(line_angle)
        rotation_scale = 1

        rotation_matrix = cv2.getRotationMatrix2D(
            rotation_center, rotation_angle, rotation_scale
        )

        img_rotated = cv2.warpAffine(img, rotation_matrix, (img_dim, img_dim))

        return img_rotated

    def _get_img_position(
        self, line_counter: LineZone, text_width: int, text_height: int, text_over: bool
    ):
        """
        Set the position of the rotated image using line counter end point as reference.

        Attributes:
            line_counter (LineZone): The line counter object used to annotate.
            text_width (int): Text width.
            text_height (int): Text height.
            text_over (bool): Whether the text should be placed over or below the line.

        Returns:
            [int, int]: xy insertion point to place text/text-box images in frame.
        """
        end_point = line_counter.vector.end.as_xy_int_tuple()
        line_angle = self._get_line_angle(line_counter)
        # Set position of the text along and perpendicular to the line.
        img_position = list(end_point)

        move_along_x = int(
            math.cos(math.radians(line_angle)) * (text_width / 2 + self.text_padding)
        )
        move_along_y = int(
            math.sin(math.radians(line_angle)) * (text_width / 2 + self.text_padding)
        )

        move_perp_x = int(
            math.sin(math.radians(line_angle))
            * (text_height / 2 + self.text_padding * 2)
        )
        move_perp_y = int(
            math.cos(math.radians(line_angle))
            * (text_height / 2 + self.text_padding * 2)
        )

        img_position[0] -= move_along_x
        img_position[1] -= move_along_y
        if text_over:
            img_position[0] += move_perp_x
            img_position[1] -= move_perp_y
        else:
            img_position[0] -= move_perp_x
            img_position[1] += move_perp_y

        return img_position

    def _get_img_bbox(self, img: np.ndarray, frame: np.ndarray, img_position: list):
        """
        Calculate xyxy insertion bbox in the frame for the text/text-box images.

        Attributes:
            img (np.ndarray): text/text-box image.
            frame (np.ndarray): Frame in which to insert the text/text-box images.
            img_position (list): xy insertion point to place text/text-box images.

        Returns:
            (int, int, int, int): xyxy insertion bbox to place text/text-box images.
        """
        img_dim = img.shape[0]

        y1 = max(img_position[1] - img_dim // 2, 0)
        y2 = min(
            img_position[1] + img_dim // 2 + img_dim % 2,
            frame.shape[0],
        )
        x1 = max(img_position[0] - img_dim // 2, 0)
        x2 = min(
            img_position[0] + img_dim // 2 + img_dim % 2,
            frame.shape[1],
        )

        return (x1, y1, x2, y2)

    def _trim_img(self, img, frame, img_bbox):
        """
        Trim text/text-box images to the limits of the frame if needed.

        Attributes:
            img (np.ndarray): text/text-box image.
            frame (np.ndarray): Frame in which to insert the text/text-box images.
            img_bbox (list): xyxy insertion bbox to place text/text-box images.

        Returns:
            np.ndarray: Trimmed text/text-box images.
        """
        img_dim = img.shape[0]
        (x1, y1, x2, y2) = img_bbox

        if y2 - y1 != img_dim:
            if y1 == 0:
                img = img[(img_dim - y2) :, :]
            elif y2 == frame.shape[0]:
                img = img[: (y2 - y1), :]

        if x2 - x1 != img_dim:
            if x1 == 0:
                img = img[:, (img_dim - x2) :]

            elif x2 == frame.shape[1]:
                img = img[:, : (x2 - x1)]

        return img

    def _annotate_box(self, frame, img, img_bbox):
        """
        Annotate text-box image in the original frame.

        Attributes:
            frame (np.ndarray): The base image on which to insert the text-box image.
            img (np.ndarray): text-box image.
            img_bbox (list): xyxy insertion bbox to place text-box image in frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        (x1, y1, x2, y2) = img_bbox

        frame[y1:y2, x1:x2, 0][img > 95] = self.color.as_bgr()[0]
        frame[y1:y2, x1:x2, 1][img > 95] = self.color.as_bgr()[1]
        frame[y1:y2, x1:x2, 2][img > 95] = self.color.as_bgr()[2]

        return frame

    def _annotate_text(self, frame, img, img_bbox):
        """
        Annotate text image in the original frame.

        Attributes:
            frame (np.ndarray): The base image on which to insert the text image.
            img (np.ndarray): text image.
            img_bbox (list): xyxy insertion bbox to place text image in frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        (x1, y1, x2, y2) = img_bbox

        frame[y1:y2, x1:x2, 0][img != 0] = self.text_color.as_bgr()[0] * (
            img[img != 0] / 255
        ) + self.color.as_bgr()[0] * (1 - (img[img != 0] / 255))
        frame[y1:y2, x1:x2, 1][img != 0] = self.text_color.as_bgr()[1] * (
            img[img != 0] / 255
        ) + self.color.as_bgr()[1] * (1 - (img[img != 0] / 255))
        frame[y1:y2, x1:x2, 2][img != 0] = self.text_color.as_bgr()[2] * (
            img[img != 0] / 255
        ) + self.color.as_bgr()[2] * (1 - (img[img != 0] / 255))

        return frame
