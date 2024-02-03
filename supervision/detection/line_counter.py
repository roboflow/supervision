import math
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.draw.utils import draw_text
from supervision.geometry.core import Point, Position, Vector


class LineZone:
    """
    This class is responsible for counting the number of objects that cross a
    predefined line.

    !!! warning

        LineZone uses the `tracker_id`. Read
        [here](/latest/trackers/) to learn how to plug
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
        triggering_anchors: Iterable[Position] = (
            Position.TOP_LEFT,
            Position.TOP_RIGHT,
            Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT,
        ),
    ):
        """
        Args:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            triggering_anchors (List[sv.Position]): A list of positions
                specifying which anchors of the detections bounding box
                to consider when deciding on whether the detection
                has passed the line counter or not. By default, this
                contains the four corners of the detection's bounding box.
        """
        self.vector = Vector(start=start, end=end)
        self.limits = self.calculate_region_of_interest_limits(vector=self.vector)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        self.triggering_anchors = triggering_anchors

    @staticmethod
    def calculate_region_of_interest_limits(vector: Vector) -> Tuple[Vector, Vector]:
        magnitude = vector.magnitude

        if magnitude == 0:
            raise ValueError("The magnitude of the vector cannot be zero.")

        delta_x = vector.end.x - vector.start.x
        delta_y = vector.end.y - vector.start.y

        unit_vector_x = delta_x / magnitude
        unit_vector_y = delta_y / magnitude

        perpendicular_vector_x = -unit_vector_y
        perpendicular_vector_y = unit_vector_x

        start_region_limit = Vector(
            start=vector.start,
            end=Point(
                x=vector.start.x + perpendicular_vector_x,
                y=vector.start.y + perpendicular_vector_y,
            ),
        )
        end_region_limit = Vector(
            start=vector.end,
            end=Point(
                x=vector.end.x - perpendicular_vector_x,
                y=vector.end.y - perpendicular_vector_y,
            ),
        )
        return start_region_limit, end_region_limit

    @staticmethod
    def is_point_in_limits(point: Point, limits: Tuple[Vector, Vector]) -> bool:
        cross_product_1 = limits[0].cross_product(point)
        cross_product_2 = limits[1].cross_product(point)
        return (cross_product_1 > 0) == (cross_product_2 > 0)

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

        if len(detections) == 0:
            return crossed_in, crossed_out

        all_anchors = np.array(
            [
                detections.get_anchors_coordinates(anchor)
                for anchor in self.triggering_anchors
            ]
        )

        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            box_anchors = [Point(x=x, y=y) for x, y in all_anchors[:, i, :]]

            in_limits = all(
                [
                    self.is_point_in_limits(point=anchor, limits=self.limits)
                    for anchor in box_anchors
                ]
            )

            if not in_limits:
                continue

            triggers = [
                self.vector.cross_product(point=anchor) < 0 for anchor in box_anchors
            ]

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
        color: Color = Color.WHITE,
        text_thickness: float = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        draw_text_box: bool = True,
        draw_centered: bool = True,
        custom_in_text: Optional[str] = None,
        custom_out_text: Optional[str] = None,
        display_in_count: bool = True,
        display_out_count: bool = True,
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
            draw_text_box (bool): Whether to draw a text box under the text or not.
            draw_centered (bool): Wheter to draw the count centered in the line or not.
            display_in_count (bool): Whether to display the in count or not.
            display_out_count (bool): Whether to display the out count or not.

        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.draw_text_box: bool = draw_text_box
        self.draw_centered: bool = draw_centered
        self.custom_in_text: str = custom_in_text
        self.custom_out_text: str = custom_out_text
        self.display_in_count: bool = display_in_count
        self.display_out_count: bool = display_out_count

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line and the count on the frame using the line_counter provided.

        Args:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineCounter): The line counter
                that will be used to draw the line.

        Returns:
            np.ndarray: The image with count drawn on it.
        """
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
        if self.draw_centered:
            cv2.circle(
                frame,
                line_counter.vector.end.as_xy_int_tuple(),
                radius=self.thickness,
                color=self.color.as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if self.display_in_count:
            in_text = (
                f"{self.custom_in_text}: {line_counter.in_count}"
                if self.custom_in_text is not None
                else f"in: {line_counter.in_count}"
            )
            frame = self._annotate_count(
                frame=frame, line_counter=line_counter, text=in_text, is_in_count=True
            )

        if self.display_out_count:
            out_text = (
                f"{self.custom_out_text}: {line_counter.out_count}"
                if self.custom_out_text is not None
                else f"out: {line_counter.out_count}"
            )
            frame = self._annotate_count(
                frame=frame, line_counter=line_counter, text=out_text, is_in_count=False
            )

        return frame

    def _annotate_count(
        self,
        frame: np.ndarray,
        line_counter: LineZone,
        text: str,
        is_in_count: bool,
    ) -> np.ndarray:
        """
        Draws the in-count/out-count aligned to the line counter object.

        Args:
            frame (np.ndarray): The image on which the count will be drawn.
            line_counter (LineCounter): The line counter
                that will be used to draw the line.
            text (str): The text that will be drawn.
            is_in_count (bool): Whether to display the in-count or out-count.

        Returns:
            np.ndarray: The image with the count drawn on it.
        """
        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )[0]

        # Create an auxiliar squared image for the count and its alpha channel
        image_dim = int((max(text_width, text_height) + self.text_padding * 2) * 1.5)
        image = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)  # bgr
        image_alpha = np.zeros((image_dim, image_dim, 1), dtype=np.uint8)  # gray

        text_args = {
            "text": text,
            "text_anchor": Point(image_dim // 2, image_dim // 2),
            "text_scale": self.text_scale,
            "text_thickness": self.text_thickness,
            "text_padding": self.text_padding,
        }
        draw_text(
            scene=image,
            text_color=self.text_color,
            background_color=self.color if self.draw_text_box else None,
            **text_args,
        )
        draw_text(
            scene=image_alpha,
            text_color=Color.WHITE,
            background_color=Color.WHITE if self.draw_text_box else None,
            **text_args,
        )
        image = np.dstack((image, image_alpha))  # Stack bgr and alpha channels

        anchor_in_frame = self._calculate_anchor_in_frame(
            line_counter=line_counter,
            text_width=text_width,
            text_height=text_height,
            is_in_count=is_in_count,
        )

        xyxy_in_frame = self._calculate_xyxy_in_frame(
            frame_dims=frame.shape[:2],
            img_dim=image_dim,
            anchor_in_frame=anchor_in_frame,
        )

        image_rotated = self._rotate_img(img=image, line_counter=line_counter)

        image_cropped = self._crop_img(img=image_rotated, xyxy_in_frame=xyxy_in_frame)

        frame = self._annotate_in_frame(
            frame=frame, img=image_cropped, xyxy_in_frame=xyxy_in_frame
        )

        return frame

    def _get_line_angle(self, line_counter: LineZone) -> float:
        """
        Calculate the line counter angle using trigonometry.

        Args:
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

    def _calculate_anchor_in_frame(
        self,
        line_counter: LineZone,
        text_width: int,
        text_height: int,
        is_in_count: bool,
    ) -> Point:
        """
        Calculate insertion anchor in frame to position the center of the count image.

        Args:
            line_counter (LineZone): The line counter object used for counting.
            text_width (int): Text width.
            text_height (int): Text height.
            is_in_count (bool): Whether the count should be placed over or below line.

        Returns:
            Point: xy insertion anchor to position count image in frame.
        """
        line_angle = self._get_line_angle(line_counter)

        if self.draw_centered:
            mid_point = Vector(
                start=line_counter.vector.start, end=line_counter.vector.end
            ).center.as_xy_int_tuple()
            anchor = list(mid_point)
        else:
            end_point = line_counter.vector.end.as_xy_int_tuple()
            anchor = list(end_point)

            move_along_x = int(
                math.cos(math.radians(line_angle))
                * (text_width / 2 + self.text_padding)
            )
            move_along_y = int(
                math.sin(math.radians(line_angle))
                * (text_width / 2 + self.text_padding)
            )

            anchor[0] -= move_along_x
            anchor[1] -= move_along_y

        move_perp_x = int(
            math.sin(math.radians(line_angle)) * (self.text_offset * text_height)
        )
        move_perp_y = int(
            math.cos(math.radians(line_angle)) * (self.text_offset * text_height)
        )

        if is_in_count:
            anchor[0] += move_perp_x
            anchor[1] -= move_perp_y
        else:
            anchor[0] -= move_perp_x
            anchor[1] += move_perp_y

        return Point(x=anchor[0], y=anchor[1])

    def _calculate_xyxy_in_frame(
        self, frame_dims: tuple, img_dim: int, anchor_in_frame: Point
    ) -> tuple:
        """
        Calculate insertion bbox in frame to position count image.

        Args:
            frame_dims (int, int): Width and height of the frame.
            img_dim (int): Width/height of squared count image.
            anchor_in_frame (Point): xy insertion anchor to position image.

        Returns:
            (int, int, int, int): xyxy insertion bbox to position count image.
        """
        y1 = max(anchor_in_frame.y - img_dim // 2, 0)
        y2 = min(
            anchor_in_frame.y + img_dim // 2 + img_dim % 2,
            frame_dims[0],
        )
        x1 = max(anchor_in_frame.x - img_dim // 2, 0)
        x2 = min(
            anchor_in_frame.x + img_dim // 2 + img_dim % 2,
            frame_dims[1],
        )

        return (x1, y1, x2, y2)

    def _rotate_img(self, img: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Rotate count image to align text with the line counter.

        Attributes:
            img (np.ndarray): Image to rotate.
            line_counter (LineZone): The line counter object.

        Returns:
            np.ndarray: Image with the same shape as the input with aligned text.
        """
        line_angle = self._get_line_angle(line_counter)

        rotation_center = (img.shape[0] // 2, img.shape[0] // 2)
        rotation_angle = -(line_angle)
        rotation_scale = 1

        rotation_matrix = cv2.getRotationMatrix2D(
            rotation_center, rotation_angle, rotation_scale
        )

        img_rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        return img_rotated

    def _crop_img(self, img, xyxy_in_frame) -> np.ndarray:
        """
        Crop image to fit insertion bbox boundaries.

        Args:
            img (np.ndarray): Image to crop.
            xyxy_in_frame (list): xyxy insertion bbox used to crop image.

        Returns:
            np.ndarray: Cropped image.
        """
        img_dim = img.shape[0]
        (x1, y1, x2, y2) = xyxy_in_frame

        if y2 - y1 != img_dim:
            img = img[(img_dim - y2) :, ...] if y1 == 0 else img[: (y2 - y1), ...]

        if x2 - x1 != img_dim:
            img = img[:, (img_dim - x2) :, ...] if x1 == 0 else img[:, : (x2 - x1), ...]

        return img

    def _annotate_in_frame(
        self, frame: np.ndarray, img: np.ndarray, xyxy_in_frame: tuple
    ) -> np.ndarray:
        """
        Annotate count image in the original frame.

        Attributes:
            frame (np.ndarray): The base image on which to insert the text-box image.
            img (np.ndarray): Count image with bgr channels + alpha channel.
            xyxy_in_frame (int, int, int, int): xyxy insertion bbox.

        Returns:
            np.ndarray: Annotated frame.
        """
        (x1, y1, x2, y2) = xyxy_in_frame

        # Paste count image and alpha in empty backgrounds with frame width and height.
        img_in_frame = np.zeros_like(frame, dtype=np.uint8)
        img_in_frame[y1:y2, x1:x2, ...] = img[:, :, :3]
        alpha_in_frame = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        alpha_in_frame[y1:y2, x1:x2] = img[:, :, 3]

        if self.draw_text_box:
            mask = alpha_in_frame > 95
            for i in range(3):
                frame[:, :, i][mask] = img_in_frame[:, :, i][mask]
        else:
            mask = alpha_in_frame != 0
            opacity = alpha_in_frame[mask] / 255
            for i in range(3):
                frame[:, :, i][mask] = (
                    frame[:, :, i][mask] * (1 - opacity)
                ) + self.text_color.as_bgr()[i] * opacity

        return frame
