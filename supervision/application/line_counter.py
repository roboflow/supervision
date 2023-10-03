from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Point, Position, Rect, Vector


class LineZone:
    """
    Count the number of objects that cross a line.
    """

    def __init__(self, start: Point, end: Point, anchor: Position = Position.CENTER):
        """
        Initialize a LineZone object.

        Attributes:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            anchor (Position): The position of the anchor point of detections
                for counting
        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.counts: Dict[str, int] = {}
        self.total_counts = {"in": 0, "out": 0}
        self.anchor = anchor
        self.previous_detections = None
        self.counted_tracker_ids: Dict[str, List] = {"in": [], "out": []}
        self.new_trigger = False

    def trigger(self, detections: Detections) -> None:
        """
        Update the in_count and out_count for the detections that cross the line.

        Attributes:
            detections (Detections): The detections for which to update the counts.
        """
        if detections.tracker_id is None:
            return

        if not self.previous_detections:
            self.previous_detections = detections
            return

        tracks_previous = self.previous_detections.tracker_id
        tracks_current = detections.tracker_id
        anchors_current = detections.get_anchor_coordinates(anchor=self.anchor)

        common_tracks, previous_common_tracks, current_common_tracks = np.intersect1d(
            tracks_previous, tracks_current, return_indices=True
        )

        common_current_anchors = anchors_current[current_common_tracks]
        common_current_class_ids = detections.class_id[current_common_tracks]

        for i in range(len(common_tracks)):
            track_id = common_tracks[i]

            if (
                track_id in self.counted_tracker_ids["in"]
                or track_id in self.counted_tracker_ids["out"]
            ):
                continue

            class_id = str(common_current_class_ids[i])
            current_position = Point(
                x=common_current_anchors[i][0], y=common_current_anchors[i][1]
            )

            if class_id not in self.counts:
                self.counts[class_id] = {"in": 0, "out": 0}

            tracker_state = self.vector.is_in(point=current_position)

            if track_id in self.tracker_state:
                if tracker_state != self.tracker_state[track_id]:
                    if tracker_state:
                        self.total_counts["in"] += 1
                        self.counts[class_id]["in"] += 1
                        self.new_trigger = True
                        if track_id not in self.counted_tracker_ids["in"]:
                            self.counted_tracker_ids["in"].append(track_id)
                        if track_id in self.counted_tracker_ids["out"]:
                            self.counted_tracker_ids["out"].remove(track_id)
                    else:
                        self.total_counts["out"] += 1
                        self.counts[class_id]["out"] += 1
                        self.new_trigger = True
                        if track_id not in self.counted_tracker_ids["out"]:
                            self.counted_tracker_ids["out"].append(track_id)
                        if track_id in self.counted_tracker_ids["in"]:
                            self.counted_tracker_ids["in"].remove(track_id)
            self.tracker_state[track_id] = tracker_state
        self.previous_detections = detections

    def get_in_out_detections(self) -> Tuple[Detections, Detections]:
        """
        Returns: (sv.Detections, sv.Detections) : return detections going
                in and out as sv.Detection objects
        """
        in_track_ids = self.counted_tracker_ids["in"]
        out_track_ids = self.counted_tracker_ids["out"]

        detections = self.previous_detections
        in_detections = detections[np.isin(detections.tracker_id, in_track_ids)]
        out_detections = detections[np.isin(detections.tracker_id, out_track_ids)]
        return in_detections, out_detections

    def get_counts(self, classes: Optional[Tuple[int]] = None) -> Dict:
        """
        Args:
            classes (Optional(Tuple[int])): tuple specific/priority classes

        Returns:
            Dictionary of requested class count
        """
        if classes:
            counts = {}
            for class_id in classes:
                counts[class_id] = self.counts.get(str(class_id))
            return counts
        return self.counts

    def get_total_counts(self) -> Dict:
        """
        Returns:
            (Dict) : Total counts dictionary containing "in" and "out" values
        """
        return self.total_counts


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
        on_trigger_action: Optional[bool] = True,
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
            custom_in_text (str): Replace "in" with provided custom text.
            custom_out_text (str): Replace "out" with provided custom text.
            on_trigger_action (bool): Change line color when any object crossed the line
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
        self.on_trigger_action = on_trigger_action

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
        line_color = self.color
        text_color = self.text_color
        if self.on_trigger_action and line_counter.new_trigger:
            text_color = self.color
            line_color = self.text_color
            line_counter.new_trigger = False

        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            line_color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )

        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        in_text = (
            f"{self.custom_in_text}: {line_counter.total_counts['in']}"
            if self.custom_in_text is not None
            else f"in: {line_counter.total_counts['in']}"
        )
        out_text = (
            f"{self.custom_out_text}: {line_counter.total_counts['out']}"
            if self.custom_out_text is not None
            else f"out: {line_counter.total_counts['out']}"
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
            line_color.as_bgr(),
            -1,
        )
        cv2.rectangle(
            frame,
            out_text_background_rect.top_left.as_xy_int_tuple(),
            out_text_background_rect.bottom_right.as_xy_int_tuple(),
            line_color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            in_text,
            (in_text_x, in_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )

        return frame
