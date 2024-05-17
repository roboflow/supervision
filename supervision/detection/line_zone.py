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

    <video controls>
        <source
            src="https://media.roboflow.com/supervision/cookbooks/count-objects-crossing-the-line-result-1280x720.mp4"
            type="video/mp4">
    </video>

    !!! warning

        LineZone uses the `tracker_id`. Read
        [here](/latest/trackers/) to learn how to plug
        tracking into your inference pipeline.

    Attributes:
        in_count (int): The number of objects that have crossed the line from outside
            to inside.
        out_count (int): The number of objects that have crossed the line from inside
            to outside.

    Example:
        ```python
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO(<SOURCE_MODEL_PATH>)
        tracker = sv.ByteTrack()
        frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)
        start, end = sv.Point(x=0, y=1080), sv.Point(x=3840, y=1080)
        line_zone = sv.LineZone(start=start, end=end)

        for frame in frames_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)
            crossed_in, crossed_out = line_zone.trigger(detections)

        line_zone.in_count, line_zone.out_count
        # 7, 2
        ```
    """  # noqa: E501 // docs
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
        self._vector = Vector(start=start, end=end)
        if self._vector.magnitude == 0:
            raise ValueError("The magnitude of the line cannot be zero.")
        vector, rotation, translation = self._transform_to_vertical_at_zero(self._vector)
        self.limits = self.calculate_region_of_interest_limits(vector=self._vector)
        self._transformed_vector = vector
        self._rotation = rotation
        self._translation = translation
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        self.triggering_anchors = triggering_anchors

    @property
    def vector(self):
        return self._vector

    @staticmethod
    def calculate_region_of_interest_limits(vector: Vector) -> Tuple[Vector, Vector]:
        """
        !!! Warning "This method is not a part of the API and as such it's
        not stable and may change in the future. Use at your own risk."
        """
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
        """
        !!! Warning "This method is not a part of the API and as such it's
        not stable and may change in the future. Use at your own risk."
        """
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

        if len(detections) == 0 or detections.tracker_id is None:
            return crossed_in, crossed_out
        vector = self._transformed_vector
        all_anchors = self._get_detection_anchors(detections)
        max_vector_y = max(vector.start.y, vector.end.y)
        min_vector_y = min(vector.start.y, vector.end.y)
        max_anchor_y = np.max(all_anchors[:, :, 1], axis=0)
        min_anchor_y = np.min(all_anchors[:, :, 1], axis=0)
        in_limits = np.logical_and(max_anchor_y <= max_vector_y, min_anchor_y >= min_vector_y)
        trigger_out = np.min(all_anchors[:, :, 0], axis=0) < 0
        trigger_in = np.max(all_anchors[:, :, 0], axis=0) >= 0
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None or not in_limits[i]:
                continue

            if trigger_out[i] and trigger_in[i]:
                continue

            if self._transformed_vector.end.y > 0:
                tracker_state = trigger_in[i]
            else:
                tracker_state = trigger_out[i]
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                crossed_in[i] = True
            else:
                crossed_out[i] = True

        self.in_count += np.sum(crossed_in)
        self.out_count += np.sum(crossed_out)
        return crossed_in, crossed_out

    def _get_detection_anchors(self, detections: Detections) -> np.ndarray:
        """
        Create array of detection anchors in coordinates where the line is vertical.
        Args:
            detections: Detections from which the anchors will be extracted
        Returns: Numpy array of shape `(anchors, detections, 2)`
        """
        all_anchors = np.array(
            [
                detections.get_anchors_coordinates(anchor)
                for anchor in self.triggering_anchors
            ]
        )
        # Transform anchors to coordinates where the line is vertical
        all_anchors = all_anchors - self._translation
        all_anchors = all_anchors @ self._rotation.T
        return all_anchors

    @staticmethod
    def _transform_to_vertical_at_zero(
            vector: Vector,
    ) -> Tuple[Vector, np.ndarray, np.ndarray]:
        """
        Translate and rotate vector so that it is vertical, centered at zero.

        Args:
            vector (Vector): A vector which will be transformed.

        Returns:
            A tuple consisting of transformed vector and two float NumPy arrays. The
            first array is a 2x2 rotation matrix which was used to rotate the vector.
            The second array is a translation. Original vector can be retrieved
            by A^T * v + w where w is the translation and A is the rotation matrix.
        """
        translation = np.array((vector.start.x, vector.start.y))
        vector = Vector(
            start=Point(x=0.0, y=0.0),
            end=Point(x=vector.end.x - vector.start.x, y=vector.end.y - vector.start.y),
        )
        # Calculate the angle of rotation
        theta = np.arccos(vector.end.y / vector.magnitude)
        # Define the rotation matrix
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        vector = rotation @ np.array([vector.end.x, vector.end.y])
        vector = Vector(
            start=Point(x=0.0, y=0.0),
            end=Point(x=vector[0], y=vector[1]),
        )
        return vector, rotation, translation

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
        self.custom_in_text: str = custom_in_text
        self.custom_out_text: str = custom_out_text
        self.display_in_count: bool = display_in_count
        self.display_out_count: bool = display_out_count

    def _annotate_count(
        self,
        frame: np.ndarray,
        center_text_anchor: Point,
        text: str,
        is_in_count: bool,
    ) -> None:
        """This method is drawing the text on the frame.

        Args:
            frame (np.ndarray): The image on which the text will be drawn.
            center_text_anchor: The center point that the text will be drawn.
            text (str): The text that will be drawn.
            is_in_count (bool): Whether to display the in count or out count.
        """
        _, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )[0]

        if is_in_count:
            center_text_anchor.y -= int(self.text_offset * text_height)
        else:
            center_text_anchor.y += int(self.text_offset * text_height)

        draw_text(
            scene=frame,
            text=text,
            text_anchor=center_text_anchor,
            text_color=self.text_color,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
            text_padding=self.text_padding,
            background_color=self.color,
        )

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

        text_anchor = Vector(
            start=line_counter.vector.start, end=line_counter.vector.end
        )

        if self.display_in_count:
            in_text = (
                f"{self.custom_in_text}: {line_counter.in_count}"
                if self.custom_in_text is not None
                else f"in: {line_counter.in_count}"
            )
            self._annotate_count(
                frame=frame,
                center_text_anchor=text_anchor.center,
                text=in_text,
                is_in_count=True,
            )

        if self.display_out_count:
            out_text = (
                f"{self.custom_out_text}: {line_counter.out_count}"
                if self.custom_out_text is not None
                else f"out: {line_counter.out_count}"
            )
            self._annotate_count(
                frame=frame,
                center_text_anchor=text_anchor.center,
                text=out_text,
                is_in_count=False,
            )
        return frame
