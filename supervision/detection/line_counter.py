from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Point, Position, Rect, Vector


class LineZone:
    """
    This class is responsible for counting the number of objects that cross a
    predefined line.

    !!! warning

        LineZone uses the `tracker_id`. Read
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
                contains the four corners of the detection's bounding box
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
                self.vector.cross_product(point=anchor) > 0 for anchor in box_anchors
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
            line_counter (LineCounter): The line counter
                that will be used to draw the line.

        Returns:
            np.ndarray: The image with the line drawn on it.

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
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

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

        (in_text_width, in_text_height), _ = cv2.getTextSize(
            in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        in_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - in_text_width)
            / 2
        )
        in_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + in_text_height)
            / 2
            - self.text_offset * in_text_height
        )

        out_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width)
            / 2
        )
        out_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height)
            / 2
            + self.text_offset * out_text_height
        )

        in_text_background_rect = Rect(
            x=in_text_x,
            y=in_text_y - in_text_height,
            width=in_text_width,
            height=in_text_height,
        ).pad(padding=self.text_padding)
        out_text_background_rect = Rect(
            x=out_text_x,
            y=out_text_y - out_text_height,
            width=out_text_width,
            height=out_text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            frame,
            in_text_background_rect.top_left.as_xy_int_tuple(),
            in_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )
        cv2.rectangle(
            frame,
            out_text_background_rect.top_left.as_xy_int_tuple(),
            out_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            in_text,
            (in_text_x, in_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        return frame
