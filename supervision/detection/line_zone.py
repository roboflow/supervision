from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from functools import lru_cache
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.utils.internal import cross_product
from supervision.draw.color import Color
from supervision.draw.utils import draw_rectangle, draw_text
from supervision.geometry.core import Point, Position, Rect, Vector
from supervision.utils.image import overlay_image
from supervision.utils.internal import SupervisionWarnings

TEXT_MARGIN = 10


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
        in_count_per_class (Dict[int, int]): Number of objects of each class that have
            crossed the line from outside to inside.
        out_count_per_class (Dict[int, int]): Number of objects of each class that have
            crossed the line from inside to outside.

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
        minimum_crossing_threshold: int = 1,
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
            minimum_crossing_threshold (int): Detection needs to be seen
                on the other side of the line for this many frames to be
                considered as having crossed the line. This is useful when
                dealing with unstable bounding boxes or when detections
                may linger on the line.
        """
        self.vector = Vector(start=start, end=end)
        self.limits = self._calculate_region_of_interest_limits(vector=self.vector)
        self.crossing_history_length = max(2, minimum_crossing_threshold + 1)
        self.crossing_state_history: dict[tuple[int, int | None], deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.crossing_history_length)
        )
        self._in_count_per_class: Counter = Counter()
        self._out_count_per_class: Counter = Counter()
        self.triggering_anchors = triggering_anchors
        if not list(self.triggering_anchors):
            raise ValueError("Triggering anchors cannot be empty.")
        self.class_id_to_name: dict[int, str] = {}

    @property
    def in_count(self) -> int:
        return sum(self._in_count_per_class.values())

    @property
    def out_count(self) -> int:
        return sum(self._out_count_per_class.values())

    @property
    def in_count_per_class(self) -> dict[int, int]:
        return dict(self._in_count_per_class)

    @property
    def out_count_per_class(self) -> dict[int, int]:
        return dict(self._out_count_per_class)

    def trigger(self, detections: Detections) -> tuple[np.ndarray, np.ndarray]:
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

        if detections.tracker_id is None:
            warnings.warn(
                "Line zone counting skipped. LineZone requires tracker_id. Refer to "
                "https://supervision.roboflow.com/latest/trackers for more "
                "information.",
                category=SupervisionWarnings,
            )
            return crossed_in, crossed_out

        self._update_class_id_to_name(detections)

        in_limits, has_any_left_trigger, has_any_right_trigger = (
            self._compute_anchor_sides(detections)
        )

        class_ids: list[int | None] = (
            list(detections.class_id)
            if detections.class_id is not None
            else [None] * len(detections)
        )

        for i, (class_id, tracker_id) in enumerate(
            zip(class_ids, detections.tracker_id)
        ):
            if not in_limits[i]:
                continue

            if has_any_left_trigger[i] and has_any_right_trigger[i]:
                continue

            tracker_state: bool = has_any_left_trigger[i]
            # Use (tracker_id, class_id) as key to handle class changes during tracking
            crossing_key = (tracker_id, class_id)
            crossing_history = self.crossing_state_history[crossing_key]
            crossing_history.append(tracker_state)

            if len(crossing_history) < self.crossing_history_length:
                continue

            oldest_state = crossing_history[0]
            if crossing_history.count(oldest_state) > 1:
                continue

            if tracker_state:
                self._in_count_per_class[class_id] += 1
                crossed_in[i] = True
            else:
                self._out_count_per_class[class_id] += 1
                crossed_out[i] = True

        return crossed_in, crossed_out

    @staticmethod
    def _calculate_region_of_interest_limits(vector: Vector) -> tuple[Vector, Vector]:
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

    def _compute_anchor_sides(
        self, detections: Detections
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """
        Find if detections' anchors are within the limit of the line
        zone and which anchors are on its left and right side.

        Assumes:
            * At least 1 detection is provided
            * Detections have `tracker_id`

        The limit is defined as the region between the two lines,
        perpendicular to the line zone, and passing through its start
        and end points, as shown below:

        Limits:
        ```
                |    IN    ↑
                |          |
          OUT   o---LINE---o   OUT
                |          |
                ↓    IN    |
        ```

        Args:
            detections (Detections): The detections to check.

        Returns:
            result (Tuple[np.ndarray, np.ndarray, np.ndarray]):
                All 3 arrays are boolean arrays of shape (N, ) where N is the
                number of detections. The first array, `in_limits`, indicates
                if the detection's anchor is within the line zone limits.
                The second array, `has_any_left_trigger`, indicates if the
                detection's anchor is on the left side of the line zone.
                The third array, `has_any_right_trigger`, indicates if the
                detection's anchor is on the right side of the line zone.
        """
        assert len(detections) > 0
        assert detections.tracker_id is not None

        all_anchors = np.array(
            [
                detections.get_anchors_coordinates(anchor)
                for anchor in self.triggering_anchors
            ]
        )

        cross_products_1 = cross_product(all_anchors, self.limits[0])
        cross_products_2 = cross_product(all_anchors, self.limits[1])

        # Works because limit vectors are pointing in opposite directions
        in_limits = (cross_products_1 > 0) == (cross_products_2 > 0)
        in_limits = np.all(in_limits, axis=0)

        triggers = cross_product(all_anchors, self.vector) < 0
        has_any_left_trigger = np.any(triggers, axis=0)
        has_any_right_trigger = np.any(~triggers, axis=0)

        return in_limits, has_any_left_trigger, has_any_right_trigger

    def _update_class_id_to_name(self, detections: Detections) -> None:
        """
        Update the attribute keeping track of which class
        IDs correspond to which class names.

        Assumes that class_names are only provided when class_ids are.
        """
        class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        assert class_names is None or detections.class_id is not None

        if detections.class_id is None:
            return

        if class_names is None:
            new_names = {class_id: str(class_id) for class_id in detections.class_id}
        else:
            new_names = {
                class_id: class_name
                for class_id, class_name in zip(detections.class_id, class_names)
            }
        self.class_id_to_name.update(new_names)


class LineZoneAnnotator:
    def __init__(
        self,
        thickness: int = 2,
        color: Color = Color.WHITE,
        text_thickness: int = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: str | None = None,
        custom_out_text: str | None = None,
        display_in_count: bool = True,
        display_out_count: bool = True,
        display_text_box: bool = True,
        text_orient_to_line: bool = False,
        text_centered: bool = True,
    ):
        """
        A class for drawing the `LineZone` and its detected object count
        on an image.

        Attributes:
            thickness (int): Line thickness.
            color (Color): Line color.
            text_thickness (int): Text thickness.
            text_color (Color): Text color.
            text_scale (float): Text scale.
            text_offset (float): How far the text will be from the line.
            text_padding (int): The empty space in the text box, surrounding the text.
            custom_in_text (Optional[str]): Write something else instead of "in".
            custom_out_text (Optional[str]): Write something else instead of "out".
            display_in_count (bool): Pass `False` to hide the "in" count.
            display_out_count (bool): Pass `False` to hide the "out" count.
            display_text_box (bool): Pass `False` to hide the text background box.
            text_orient_to_line (bool): ⭐ Match text orientation to the line.
                Recommended to set to `True`.
            text_centered (bool): Pass `False` to disable text centering. Useful
                when the label overlaps something important.

        """
        self.thickness: int = thickness
        self.color: Color = color
        self.text_thickness: int = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.in_text: str = custom_in_text if custom_in_text else "in"
        self.out_text: str = custom_out_text if custom_out_text else "out"
        self.display_in_count: bool = display_in_count
        self.display_out_count: bool = display_out_count
        self.display_text_box: bool = display_text_box
        self.text_orient_to_line: bool = text_orient_to_line
        self.text_centered: bool = text_centered

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line zone provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineZone): The line zone
                that will be used to draw the line.

        Returns:
            (np.ndarray): The image with the line drawn on it.

        """
        line_start = line_counter.vector.start.as_xy_int_tuple()
        line_end = line_counter.vector.end.as_xy_int_tuple()
        cv2.line(
            frame,
            line_start,
            line_end,
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_start,
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_end,
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        in_text = f"{self.in_text}: {line_counter.in_count}"
        out_text = f"{self.out_text}: {line_counter.out_count}"
        line_angle_degrees = self._get_line_angle(line_counter)

        for text, is_shown, is_in_count in [
            (in_text, self.display_in_count, True),
            (out_text, self.display_out_count, False),
        ]:
            if not is_shown:
                continue

            if line_angle_degrees == 0 or not self.text_orient_to_line:
                self._draw_basic_label(
                    frame=frame,
                    line_center=line_counter.vector.center,
                    text=text,
                    is_in_count=is_in_count,
                )
            else:
                self._draw_oriented_label(
                    frame=frame,
                    line_zone=line_counter,
                    text=text,
                    is_in_count=is_in_count,
                )

        return frame

    def _get_line_angle(self, line_zone: LineZone) -> float:
        """
        Calculate the line counter angle (in degrees).

        Args:
            line_zone (LineZone): The line zone object.

        Returns:
            (float): Line counter angle, in degrees.
        """
        start_point = line_zone.vector.start.as_xy_int_tuple()
        end_point = line_zone.vector.end.as_xy_int_tuple()

        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]

        if delta_x == 0:
            line_angle = 90.0
            line_angle += 180 if delta_y < 0 else 0
        else:
            line_angle = math.degrees(math.atan(delta_y / delta_x))
            line_angle += 180 if delta_x < 0 else 0

        return line_angle

    def _calculate_anchor_in_frame(
        self,
        line_zone: LineZone,
        text_width: int,
        text_height: int,
        is_in_count: bool,
        label_dimension: int,
    ) -> tuple[int, int]:
        """
        Calculate insertion anchor in frame to position the center of the count image.

        Args:
            line_zone (LineZone): The line counter object used for counting.
            text_width (int): Text width.
            text_height (int): Text height.
            is_in_count (bool): Whether the count should be placed over or below line.
            label_dimension (int): Size of the label image. Assumes the
                label is rectangular.

        Returns:
            (Tuple[int, int]): xy, point in an image where the label will be placed.
        """
        line_angle = self._get_line_angle(line_zone)

        if self.text_centered:
            mid_point = Vector(
                start=line_zone.vector.start, end=line_zone.vector.end
            ).center.as_xy_int_tuple()
            anchor = list(mid_point)
        else:
            end_point = line_zone.vector.end.as_xy_int_tuple()
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

        move_perpendicular_x = int(
            math.sin(math.radians(line_angle)) * (self.text_offset * text_height)
        )
        move_perpendicular_y = int(
            math.cos(math.radians(line_angle)) * (self.text_offset * text_height)
        )

        if is_in_count:
            anchor[0] += move_perpendicular_x
            anchor[1] -= move_perpendicular_y
        else:
            anchor[0] -= move_perpendicular_x
            anchor[1] += move_perpendicular_y

        x1 = max(anchor[0] - label_dimension // 2, 0)
        y1 = max(anchor[1] - label_dimension // 2, 0)

        return x1, y1

    def _draw_basic_label(
        self,
        frame: np.ndarray,
        line_center: Point,
        text: str,
        is_in_count: bool,
    ) -> np.ndarray:
        """
        Draw the count label on the frame. For example: "out: 7".
        The label contains horizontal text and is not rotated.

        Args:
            frame (np.ndarray): The entire scene, on which the label will be placed.
            line_center (Point): The center of the line zone.
            text (str): The text that will be drawn.
            is_in_count (bool): Whether to display the in count (above line)
                or out count (below line).

        Returns:
            (np.ndarray): The scene with the label drawn on it.
        """
        _, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )[0]

        if is_in_count:
            line_center.y -= int(self.text_offset * text_height)
        else:
            line_center.y += int(self.text_offset * text_height)

        draw_text(
            scene=frame,
            text=text,
            text_anchor=line_center,
            text_color=self.text_color,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
            text_padding=self.text_padding,
            background_color=self.color if self.display_text_box else None,
        )

        return frame

    def _draw_oriented_label(
        self,
        frame: np.ndarray,
        line_zone: LineZone,
        text: str,
        is_in_count: bool,
    ) -> np.ndarray:
        """
        Draw the count label on the frame. For example: "out: 7".
        The label is oriented to match the line angle.

        Args:
            frame (np.ndarray): The entire scene, on which the label will be placed.
            line_zone (LineZone): The line zone responsible for counting
                objects crossing it.
            text (str): The text that will be drawn.
            is_in_count (bool): Whether to display the in count (above line)
                or out count (below line).

        Returns:
            (np.ndarray): The scene with the label drawn on it.
        """

        line_angle_degrees = self._get_line_angle(line_zone)
        label_image = self._make_label_image(
            text,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
            text_padding=self.text_padding,
            text_color=self.text_color,
            text_box_show=self.display_text_box,
            text_box_color=self.color,
            line_angle_degrees=line_angle_degrees,
        )
        assert label_image.shape[0] == label_image.shape[1]

        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )[0]

        label_anchor = self._calculate_anchor_in_frame(
            line_zone=line_zone,
            text_width=text_width,
            text_height=text_height,
            is_in_count=is_in_count,
            label_dimension=label_image.shape[0],
        )

        frame = overlay_image(frame, label_image, label_anchor)

        return frame

    @staticmethod
    @lru_cache(maxsize=32)
    def _make_label_image(
        text: str,
        *,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        text_color: Color,
        text_box_show: bool,
        text_box_color: Color,
        line_angle_degrees: float,
    ) -> np.ndarray:
        """
        Create the small text box displaying line zone count. E.g. "out: 7".

        Args:
            text (str): The text to display.
            text_scale (float): The scale of the text.
            text_thickness (int): The thickness of the text.
            text_padding (int): The padding around the text.
            text_color (Color): The color of the text.
            text_box_show (bool): Whether to display the text box.
            text_box_color (Color): The color of the text box.
            line_angle_degrees (float): The angle of the line in degrees.

        Returns:
            (np.ndarray): The label of shape (H, W, 4), in BGRA format.
        """
        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )[0]

        annotation_dim = int((max(text_width, text_height) + text_padding * 2) * 1.5)
        annotation_shape = (annotation_dim, annotation_dim)
        annotation_center = Point(annotation_dim // 2, annotation_dim // 2)

        annotation = np.zeros((*annotation_shape, 3), dtype=np.uint8)
        annotation_alpha = np.zeros((*annotation_shape, 1), dtype=np.uint8)

        text_args: dict[str, Any] = dict(
            text=text,
            text_anchor=annotation_center,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
        )
        draw_text(
            scene=annotation,
            text_color=text_color,
            background_color=text_box_color if text_box_show else None,
            **text_args,
        )
        draw_text(
            scene=annotation_alpha,
            text_color=Color.WHITE,
            background_color=Color.WHITE if text_box_show else None,
            **text_args,
        )
        annotation = np.dstack((annotation, annotation_alpha))

        # Make sure text is displayed upright
        if 90 < line_angle_degrees % 360 < 270:
            annotation = cv2.flip(annotation, flipCode=-1).astype(np.uint8)

        rotation_angle = -line_angle_degrees
        rotation_matrix = cv2.getRotationMatrix2D(
            annotation_center.as_xy_float_tuple(), rotation_angle, scale=1
        )
        annotation = cv2.warpAffine(annotation, rotation_matrix, annotation_shape)

        return annotation


class LineZoneAnnotatorMulticlass:
    def __init__(
        self,
        *,
        table_position: Literal[
            Position.TOP_LEFT,
            Position.TOP_RIGHT,
            Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT,
        ] = Position.TOP_RIGHT,
        table_color: Color = Color.WHITE,
        table_margin: int = 10,
        table_padding: int = 10,
        table_max_width: int = 400,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.75,
        text_thickness: int = 1,
        force_draw_class_ids: bool = False,
    ):
        """
        Draw a table showing how many items of each class crossed each line.

        Args:
            table_position (Position): The position of the table.
            table_color (Color): The color of the table.
            table_margin (int): The margin of the table from the image border.
            table_padding (int): The padding of the table.
            table_max_width (int): The maximum width of the table.
            text_color (Color): The color of the text.
            text_scale (float): The scale of the text.
            text_thickness (int): The thickness of the text.
            force_draw_class_ids (bool): Instead of writing the class names,
                on the table, write the class IDs. E.g. instead of `person: 6`,
                write `0: 6`.
        """
        if table_position not in {
            Position.TOP_LEFT,
            Position.TOP_RIGHT,
            Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT,
        }:
            raise ValueError(
                "Invalid table position. Supported values are:"
                " TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT."
            )

        self.table_position = table_position
        self.table_color = table_color
        self.table_margin = table_margin
        self.table_padding = table_padding
        self.table_max_width = table_max_width
        self.text_color = text_color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.force_draw_class_ids = force_draw_class_ids

    def annotate(
        self,
        frame: np.ndarray,
        line_zones: list[LineZone],
        line_zone_labels: list[str] | None = None,
    ) -> np.ndarray:
        """
        Draws a table with the number of objects of each class that crossed each line.

        Attributes:
            frame (np.ndarray): The image on which the table will be drawn.
            line_zones (List[LineZone]): The line zones to be annotated.
            line_zone_labels (Optional[List[str]]): The labels, one for each
                line zone. If not provided, the default labels will be used.

        Returns:
            (np.ndarray): The image with the table drawn on it.

        """
        if line_zone_labels is None:
            line_zone_labels = [f"Line {i + 1}:" for i in range(len(line_zones))]
        if len(line_zones) != len(line_zone_labels):
            raise ValueError("The number of line zones and their labels must match.")

        text_lines = ["Line Crossings:"]
        for line_zone, line_zone_label in zip(line_zones, line_zone_labels):
            text_lines.append(line_zone_label)
            class_id_to_name = line_zone.class_id_to_name

            for direction, count_per_class in [
                ("In", line_zone.in_count_per_class),
                ("Out", line_zone.out_count_per_class),
            ]:
                if not count_per_class:
                    continue

                text_lines.append(f" {direction}:")
                for class_id, count in count_per_class.items():
                    class_name = (
                        class_id_to_name.get(class_id, str(class_id))
                        if not self.force_draw_class_ids
                        else str(class_id)
                    )
                    text_lines.append(f"  {class_name}: {count}")

        table_width, table_height = 0, 0
        for line in text_lines:
            text_width, text_height = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )[0]
            text_height += TEXT_MARGIN
            table_width = max(table_width, text_width)
            table_height += text_height

        table_width += 2 * self.table_padding
        table_height += 2 * self.table_padding
        table_max_height = frame.shape[0] - 2 * self.table_margin
        table_height = min(table_height, table_max_height)
        table_width = min(table_width, self.table_max_width)

        position_map = {
            Position.TOP_LEFT: (self.table_margin, self.table_margin),
            Position.TOP_RIGHT: (
                frame.shape[1] - table_width - self.table_margin,
                self.table_margin,
            ),
            Position.BOTTOM_LEFT: (
                self.table_margin,
                frame.shape[0] - table_height - self.table_margin,
            ),
            Position.BOTTOM_RIGHT: (
                frame.shape[1] - table_width - self.table_margin,
                frame.shape[0] - table_height - self.table_margin,
            ),
        }
        table_x1, table_y1 = position_map[self.table_position]

        table_rect = Rect(
            x=table_x1, y=table_y1, width=table_width, height=table_height
        )
        frame = draw_rectangle(
            scene=frame, rect=table_rect, color=self.table_color, thickness=-1
        )

        for i, line in enumerate(text_lines):
            _, text_height = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )[0]
            text_height += TEXT_MARGIN
            anchor_x = table_x1 + self.table_padding
            anchor_y = table_y1 + self.table_padding + (i + 1) * text_height

            cv2.putText(
                img=frame,
                text=line,
                org=(anchor_x, anchor_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.text_scale,
                color=self.text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

        return frame
