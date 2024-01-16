from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect, Vector


class LineZone:
    """
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

    def __init__(self, start: Point, end: Point, count_condition="whole_crossed"):
        """
        Args:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            count_condition (str): The condition which determines
                how detections are counted as having crossed the line
                counter. Can either be "whole_crossed" or "center_point_crossed".
                
                If condition is set to "whole_crossed", trigger() determines
                whether if the whole bounding box of the detection has crossed
                the line or not. This is the default behaviour.

                If condition is set to "center_point_crossed", trigger() determines
                whether if the center point of the detection's bounding box has
                crossed the line or not.
        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        self.count_condition = count_condition
        if count_condition not in ["whole_crossed", "center_point_crossed"]:
            raise ValueError("Argument count_condition must be 'whole_crossed' or 'center_point_crossed'")

    def is_point_in_line_range(self, point: Point) -> bool:
        """
        Check if the given point is within the line's x and y range.
        This should be used with trigger() to determine points that are
        precisely within the range of the line counter's start and end points.

        Args:
            point (Point): The point to check
        """
        line_min_x, line_max_x = min(self.vector.start.x, self.vector.end.x), max(self.vector.start.x, self.vector.end.x)
        line_min_y, line_max_y = min(self.vector.start.y, self.vector.end.y), max(self.vector.start.y, self.vector.end.y)

        within_line_range_x = line_min_x != line_max_x and line_min_x <= point.x <= line_max_x
        within_line_range_y = line_min_y != line_max_y and line_min_y <= point.y <= line_max_y

        return (within_line_range_x or line_min_x == line_max_x) and \
                            (within_line_range_y or line_min_y == line_max_y)

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

        if self.count_condition == "whole_crossed":
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

                triggers = [(self.vector.cross_product(point=anchor) < 0) for anchor in anchors]

                if len(set(triggers)) == 2:
                    continue

                tracker_state = triggers[0]

                if tracker_id not in self.tracker_state:
                    self.tracker_state[tracker_id] = tracker_state
                    continue

                if self.tracker_state.get(tracker_id) == tracker_state:
                    continue

                self.tracker_state[tracker_id] = tracker_state
                
                all_anchors_in_range = True
                for anchor in anchors:
                    if not self.is_point_in_line_range(anchor):
                        all_anchors_in_range = False
                        break

                if not all_anchors_in_range:
                    continue

                if tracker_state:
                    self.in_count += 1
                    crossed_in[i] = True
                else:
                    self.out_count += 1
                    crossed_out[i] = True

            return self.in_count, self.out_count
        
        elif self.count_condition == "center_point_crossed":
            for i, (xyxy, _, confidence, class_id, tracker_id) in enumerate(detections):
                if tracker_id is None:
                    continue

                x1, y1, x2, y2 = xyxy

                # Calculate the center point of the box
                center_point = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)

                current_state = self.vector.cross_product(center_point)

                if tracker_id not in self.tracker_state:
                    self.tracker_state[tracker_id] = current_state
                    continue

                previous_state = self.tracker_state[tracker_id]

                # Update the tracker state and check for crossing
                if previous_state * current_state < 0 and self.is_point_in_line_range(center_point):
                    self.tracker_state[tracker_id] = current_state
                    if current_state > 0:
                        self.in_count += 1
                        crossed_in[i] = True
                    elif current_state < 0:
                        self.out_count += 1
                        crossed_out[i] = True

            return self.in_count, self.out_count

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
