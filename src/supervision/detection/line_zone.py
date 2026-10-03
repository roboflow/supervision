import math
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from functools import lru_cache
from itertools import islice
from typing import Literal

import numpy as np
import numpy.typing as npt

from supervision import _cv2 as cv2
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.utils.internal import cross_product
from supervision.draw.color import Color
from supervision.draw.utils import draw_rectangle, draw_text
from supervision.geometry.core import (
    CoordinatesTransform,
    Point,
    Position,
    Rect,
    Vector,
    _transform_points,
)
from supervision.geometry.utils import _clip_segment_to_box
from supervision.utils.image import _overlay_image
from supervision.utils.internal import SupervisionWarnings

TEXT_MARGIN = 10


class LineZone:
    """This class is responsible for counting the number of objects that cross a
    predefined line.

    <video controls>
        <source
            src="https://media.roboflow.com/supervision/cookbooks/count-objects-crossing-the-line-result-1280x720.mp4"
            type="video/mp4">
    </video>

    !!! warning

        LineZone uses the `tracker_id`. Read
        [here](https://trackers.roboflow.com/latest/) to learn how to plug
        tracking into your inference pipeline. Detections with a negative
        `tracker_id`, which trackers report for tracks they have not confirmed
        yet, are ignored and never counted.

    Attributes:
        in_count: The number of objects that have crossed the line from outside
            to inside.
        out_count: The number of objects that have crossed the line from inside
            to outside.
        in_count_per_class: Number of objects of each class that have
            crossed the line from outside to inside, keyed by `class_id`
            (`int` for classified detections, `None` for unclassified ones).
        out_count_per_class: Number of objects of each class that have
            crossed the line from inside to outside, keyed by `class_id`
            (`int` for classified detections, `None` for unclassified ones).

    Example:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> def frames_generator():
        ...     yield np.zeros((1080, 1920, 3), dtype=np.uint8)
        ...
        >>> start = sv.Point(x=0, y=100)
        >>> end = sv.Point(x=200, y=100)
        >>> line_zone = sv.LineZone(start=start, end=end)
        >>> for frame in frames_generator():
        ...     detections = sv.Detections(
        ...         xyxy=np.array([[10, 110, 20, 150]]),
        ...         tracker_id=np.array([1])
        ...     )
        ...     crossed_in, crossed_out = line_zone.trigger(
        ...         detections
        ...     )
        ...
        >>> line_zone.in_count, line_zone.out_count
        (0, 0)
        >>> for frame in frames_generator():
        ...     detections = sv.Detections(
        ...         xyxy=np.array([[10, 50, 20, 90]]),
        ...         tracker_id=np.array([1])
        ...     )
        ...     crossed_in, crossed_out = line_zone.trigger(
        ...         detections
        ...     )
        ...
        >>> line_zone.in_count, line_zone.out_count
        (1, 0)

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
    ) -> None:
        """
        Args:
            start: The starting point of the line.
            end: The ending point of the line.
            triggering_anchors: Any iterable of positions specifying which anchors
                of the detections bounding box to consider when deciding on whether
                the detection has passed the line counter or not. By default,
                this contains the four corners of the detection's bounding box.
            minimum_crossing_threshold: Detection needs to be seen on the other
                side of the line for this many consecutive observations to be
                considered as having crossed the line. Only frames that place the
                detection on one side of the line count: a frame in which it is
                temporarily missing, falls outside the line's limits, or straddles
                the line is skipped, and neither extends the run nor resets it.
                This is useful when dealing with unstable bounding boxes or when
                detections may linger on the line. Excursions shorter than this
                are treated as noise: they neither count as a crossing nor as a
                return crossing once the detection settles back on the side it
                started from. This holds only for excursions from a side the
                detection is already established on — the first side it is seen
                on, at the start of its life and again after it has been absent
                long enough for its state to be dropped, becomes its reference
                immediately, with no confirmation.
        """
        self.vector = Vector(start=start, end=end)
        self.limits = self._calculate_region_of_interest_limits(vector=self.vector)
        self.crossing_history_length = max(2, minimum_crossing_threshold + 1)
        self.crossing_state_history: dict[int, deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.crossing_history_length)
        )
        # Last side of the line a tracker was *confirmed* on. Crossings are counted
        # against this rather than against the oldest history entry, so a
        # sub-threshold excursion never becomes the reference side once a reference
        # exists, and cannot be mistaken for a crossing back to the side the tracker
        # never left. The reference itself is seeded by the first observation with no
        # confirmation, so a noisy first frame — at the start of a tracker's life or
        # on its first frame back after eviction — still decides where it starts.
        self._confirmed_crossing_side: dict[int, bool] = {}
        # Tracks consecutive frames a tracker key has been absent; eviction
        # requires crossing_history_length absent frames so that ByteTrack
        # coasting gaps (single-frame detection drops) don't reset mid-crossing
        # state prematurely.
        self._tracker_frames_absent: dict[int, int] = {}
        self._in_count_per_class: Counter[int | None] = Counter()
        self._out_count_per_class: Counter[int | None] = Counter()
        # Materialize once so we can safely accept generators without exhausting them.
        self.triggering_anchors = list(triggering_anchors)
        if not self.triggering_anchors:
            raise ValueError("Triggering anchors cannot be empty.")
        self.class_id_to_name: dict[int, str] = {}

    @property
    def in_count(self) -> int:
        return sum(self._in_count_per_class.values())

    @property
    def out_count(self) -> int:
        return sum(self._out_count_per_class.values())

    @property
    def in_count_per_class(self) -> dict[int | None, int]:
        return dict(self._in_count_per_class)

    @property
    def out_count_per_class(self) -> dict[int | None, int]:
        return dict(self._out_count_per_class)

    def trigger(
        self,
        detections: Detections,
        coord_transform: CoordinatesTransform | None = None,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Update the `in_count` and `out_count` based on the objects that cross the
        line.

        Detections whose `tracker_id` is negative are treated as unconfirmed
        tracks and ignored: they are never counted, leave no crossing state
        behind, and their entries in both returned arrays are always `False`.

        Args:
            detections: A Detections object for which to update the counts.
            coord_transform: Optional per-frame camera motion, such as
                `sv.MatrixTransform`; anchors are mapped into the line's reference
                frame with `rel_to_abs`, and non-finite ones are outside its limits.
                Pass it on every call; see the "Follow a Moving Camera" how-to.

        Returns:
            A tuple of two boolean NumPy arrays. The first array indicates which
                detections have crossed the line from outside to inside. The second
                array indicates which detections have crossed the line from inside to
                outside.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> line = sv.LineZone(start=sv.Point(0, 100), end=sv.Point(200, 100))
            >>> track_id = np.array([1])
            >>> frame_1 = sv.Detections(
            ...     xyxy=np.array([[10.0, 50.0, 20.0, 90.0]]), tracker_id=track_id
            ... )
            >>> # The camera tilts: the same parked object now appears 60 px lower.
            >>> camera_moved = sv.MatrixTransform(np.array([[1, 0, 0], [0, 1, 60]]))
            >>> frame_2 = sv.Detections(
            ...     xyxy=np.array([[10.0, 110.0, 20.0, 150.0]]), tracker_id=track_id
            ... )
            >>> _ = line.trigger(frame_1)
            >>> _ = line.trigger(frame_2, coord_transform=camera_moved)
            >>> line.in_count + line.out_count
            0

            ```
        """
        crossed_in = np.full(len(detections), False)
        crossed_out = np.full(len(detections), False)

        if len(detections) == 0:
            self._evict_stale_crossing_history(set())
            return crossed_in, crossed_out

        if detections.tracker_id is None:
            warnings.warn(
                "Line zone counting skipped. LineZone requires tracker_id. Refer to "
                "https://trackers.roboflow.com/latest/ for more "
                "information.",
                category=SupervisionWarnings,
            )
            return crossed_in, crossed_out

        class_ids: list[int | None] = (
            list(detections.class_id)
            if detections.class_id is not None
            else [None] * len(detections)
        )
        # Trackers report a negative id for tracks they have not confirmed yet, so
        # several distinct objects can share it. Such detections get no crossing
        # state at all: they are left out of the eviction keys here and skipped in
        # the loop below, otherwise they would collapse into one phantom track.
        unconfirmed = detections.tracker_id < 0
        current_keys = {
            int(tracker_id) for tracker_id in detections.tracker_id[~unconfirmed]
        }
        self._evict_stale_crossing_history(current_keys)
        self._update_class_id_to_name(detections)

        in_limits, has_any_left_trigger, has_any_right_trigger = (
            self._compute_anchor_sides(detections, coord_transform)
        )

        for i, (class_id, tracker_id) in enumerate(
            zip(class_ids, detections.tracker_id)
        ):
            if unconfirmed[i]:
                continue

            if not in_limits[i]:
                continue

            if has_any_left_trigger[i] and has_any_right_trigger[i]:
                continue

            tracker_state = bool(has_any_left_trigger[i])
            key = int(tracker_id)
            crossing_history = self.crossing_state_history[key]
            crossing_history.append(tracker_state)
            # The first observed side seeds the reference; there is nothing to have
            # crossed from before it.
            confirmed_side = self._confirmed_crossing_side.setdefault(
                key, tracker_state
            )

            # Subsumed by the confirmed-side check that follows, but not removable:
            # this is what guarantees the deque is full — exactly
            # crossing_history_length entries — by the time the sustained-run check
            # below reads it. Without it a partial window would satisfy that check
            # after a single frame on the new side.
            if len(crossing_history) < self.crossing_history_length:
                continue

            if tracker_state == confirmed_side:
                continue

            # Only promote the new side once it has been held for the full threshold,
            # so a flicker that reverts before the threshold elapses is discarded.
            # The gate above leaves exactly crossing_history_length entries, so
            # skipping the oldest one leaves precisely the minimum_crossing_threshold
            # frames that must all be on the new side.
            sustained_run = islice(crossing_history, 1, None)
            if any(state != tracker_state for state in sustained_run):
                continue

            self._confirmed_crossing_side[key] = tracker_state

            if tracker_state:
                self._in_count_per_class[class_id] += 1
                crossed_in[i] = True
            else:
                self._out_count_per_class[class_id] += 1
                crossed_out[i] = True

        return crossed_in, crossed_out

    def _evict_stale_crossing_history(self, current_keys: set[int]) -> None:
        for key in list(self.crossing_state_history):
            if key in current_keys:
                self._tracker_frames_absent.pop(key, None)
            else:
                absent = self._tracker_frames_absent.get(key, 0) + 1
                if absent >= self.crossing_history_length:
                    del self.crossing_state_history[key]
                    self._confirmed_crossing_side.pop(key, None)
                    self._tracker_frames_absent.pop(key, None)
                else:
                    self._tracker_frames_absent[key] = absent

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
        self,
        detections: Detections,
        coord_transform: CoordinatesTransform | None = None,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Find if detections' anchors are within the limit of the line zone and which
        anchors are on its left and right side.

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
            detections: The detections to check.
            coord_transform: Optional transform whose `rel_to_abs` maps the
                anchors into the line's reference frame before the tests. A
                detection with any non-finite mapped anchor is not in limits.

        Returns:
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
        is_mapped_finite = None
        if coord_transform is not None:
            all_anchors = _transform_points(all_anchors, coord_transform.rel_to_abs)
            # NaN compares False on both limit tests, which would read as "in
            # limits"; such anchors are zeroed here and excluded explicitly below.
            is_mapped_finite = np.all(np.isfinite(all_anchors), axis=(0, 2))
            all_anchors = np.nan_to_num(all_anchors, nan=0.0, posinf=0.0, neginf=0.0)

        cross_products_1 = cross_product(all_anchors, self.limits[0])
        cross_products_2 = cross_product(all_anchors, self.limits[1])

        # Works because limit vectors are pointing in opposite directions
        in_limits = (cross_products_1 > 0) == (cross_products_2 > 0)
        in_limits = np.all(in_limits, axis=0)
        if is_mapped_finite is not None:
            in_limits &= is_mapped_finite

        triggers = cross_product(all_anchors, self.vector) < 0
        has_any_left_trigger = np.any(triggers, axis=0)
        has_any_right_trigger = np.any(~triggers, axis=0)

        return in_limits, has_any_left_trigger, has_any_right_trigger

    def _update_class_id_to_name(self, detections: Detections) -> None:
        """Update the attribute keeping track of which class IDs correspond to which
        class names.

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
    """Draw a `LineZone` and its in/out counts on a video frame.

    Use this annotator after calling `LineZone.trigger` so the rendered counts reflect
    the latest tracked detections.
    """

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
    ) -> None:
        """A class for drawing the `LineZone` and its detected object count on an image.

        Args:
            thickness: Line thickness.
            color: Line color.
            text_thickness: Text thickness.
            text_color: Text color.
            text_scale: Text scale.
            text_offset: How far the text will be from the line.
            text_padding: The empty space in the text box, surrounding the text.
            custom_in_text: Write something else instead of "in".
            custom_out_text: Write something else instead of "out".
            display_in_count: Pass `False` to hide the "in" count.
            display_out_count: Pass `False` to hide the "out" count.
            display_text_box: Pass `False` to hide the text background box.
            text_orient_to_line: Match text orientation to the line.
                Recommended to set to `True`.
            text_centered: Pass `False` to disable text centering. Useful
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

    def annotate(
        self,
        frame: npt.NDArray[np.uint8],
        line_counter: LineZone,
        coord_transform: CoordinatesTransform | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Draws the line on the frame using the line zone provided.

        Args:
            frame: The image on which the line will be drawn.
            line_counter: The line zone that will be used to draw the line.
            coord_transform: The transform passed to `LineZone.trigger`, such as
                `sv.MatrixTransform`; the line and labels are drawn at `abs_to_rel`
                of its end points, clipped to 4x the frame size, and nothing is
                drawn if an end point is non-finite. See the how-to guide.

        Returns:
            The image with the line drawn on it.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> line_zone = sv.LineZone(start=sv.Point(10, 20), end=sv.Point(90, 20))
            >>> line_annotator = sv.LineZoneAnnotator(
            ...     display_in_count=False, display_out_count=False
            ... )
            >>> camera_moved = sv.MatrixTransform(np.array([[1, 0, 0], [0, 1, 50]]))
            >>> frame = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> frame = line_annotator.annotate(
            ...     frame, line_zone, coord_transform=camera_moved
            ... )
            >>> bool(frame[70, 50].any()), bool(frame[20, 50].any())
            (True, False)

            ```
        """
        vector = line_counter.vector
        if coord_transform is not None:
            mapped_vector = self._map_vector(
                vector, coord_transform, limit=4 * max(frame.shape[:2])
            )
            if mapped_vector is None:
                return frame
            vector = mapped_vector

        line_start = vector.start.as_xy_int_tuple()
        line_end = vector.end.as_xy_int_tuple()
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
        line_angle_degrees = self._get_line_angle(vector)

        for text, is_shown, is_in_count in [
            (in_text, self.display_in_count, True),
            (out_text, self.display_out_count, False),
        ]:
            if not is_shown:
                continue

            if line_angle_degrees == 0 or not self.text_orient_to_line:
                self._draw_basic_label(
                    frame=frame,
                    line_center=vector.center,
                    text=text,
                    is_in_count=is_in_count,
                )
            else:
                self._draw_oriented_label(
                    frame=frame,
                    vector=vector,
                    text=text,
                    is_in_count=is_in_count,
                )

        return frame

    @staticmethod
    def _map_vector(
        vector: Vector, coord_transform: CoordinatesTransform, limit: float
    ) -> Vector | None:
        """Map a line with `abs_to_rel`, clip it to `[-limit, limit]²` and round it.

        Clipping keeps near-horizon or far-translated end points drawable. Returns
        `None` if an end point is non-finite or no part of the line is in bounds.
        """
        end_points = np.array(
            [vector.start.as_xy_float_tuple(), vector.end.as_xy_float_tuple()]
        )
        mapped = _transform_points(end_points, coord_transform.abs_to_rel)
        clipped = _clip_segment_to_box(mapped, limit=limit)
        if clipped is None:
            return None
        mapped = np.rint(clipped)
        return Vector(
            start=Point(x=float(mapped[0, 0]), y=float(mapped[0, 1])),
            end=Point(x=float(mapped[1, 0]), y=float(mapped[1, 1])),
        )

    def _get_line_angle(self, vector: Vector) -> float:
        """Calculate the line counter angle (in degrees).

        Args:
            vector: The line, in the coordinates it is drawn in.

        Returns:
            Line counter angle, in degrees.
        """
        start_point = vector.start.as_xy_int_tuple()
        end_point = vector.end.as_xy_int_tuple()

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
        vector: Vector,
        text_width: int,
        text_height: int,
        is_in_count: bool,
        label_dimension: int,
    ) -> tuple[int, int]:
        """Calculate insertion anchor in frame to position the center of the count
        image.

        Args:
            vector: The line, in the coordinates it is drawn in.
            text_width: Text width.
            text_height: Text height.
            is_in_count: Whether the count should be placed over or below line.
            label_dimension: Size of the label image. Assumes the
                label is rectangular.

        Returns:
            xy, point in an image where the label will be placed.
        """
        line_angle = self._get_line_angle(vector)

        if self.text_centered:
            mid_point = Vector(
                start=vector.start, end=vector.end
            ).center.as_xy_int_tuple()
            anchor = list(mid_point)
        else:
            end_point = vector.end.as_xy_int_tuple()
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
        frame: npt.NDArray[np.uint8],
        line_center: Point,
        text: str,
        is_in_count: bool,
    ) -> npt.NDArray[np.uint8]:
        """Draw the count label on the frame.

        For example: "out: 7". The label contains horizontal text and is not rotated.

        Args:
            frame: The entire scene, on which the label will be placed.
            line_center: The center of the line zone.
            text: The text that will be drawn.
            is_in_count: Whether to display the in count (above line)
                or out count (below line).

        Returns:
            The scene with the label drawn on it.
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
        frame: npt.NDArray[np.uint8],
        vector: Vector,
        text: str,
        is_in_count: bool,
    ) -> npt.NDArray[np.uint8]:
        """Draw the count label on the frame.

        For example: "out: 7". The label is oriented to match the line angle.

        Args:
            frame: The entire scene, on which the label will be placed.
            vector: The line, in the coordinates it is drawn in.
            text: The text that will be drawn.
            is_in_count: Whether to display the in count (above line)
                or out count (below line).

        Returns:
            The scene with the label drawn on it.
        """
        line_angle_degrees = self._get_line_angle(vector)
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
            vector=vector,
            text_width=text_width,
            text_height=text_height,
            is_in_count=is_in_count,
            label_dimension=label_image.shape[0],
        )

        frame = _overlay_image(frame, label_image, label_anchor)

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
    ) -> npt.NDArray[np.uint8]:
        """Create the small text box displaying line zone count, E.g.

        "out: 7".
                Args:
                    text: The text to display.
                    text_scale: The scale of the text.
                    text_thickness: The thickness of the text.
                    text_padding: The padding around the text.
                    text_color: The color of the text.
                    text_box_show: Whether to display the text box.
                    text_box_color: The color of the text box.
                    line_angle_degrees: The angle of the line in degrees.

                Returns:
                    The label of shape (H, W, 4), in BGRA format.
        """
        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )[0]

        annotation_dim = int((max(text_width, text_height) + text_padding * 2) * 1.5)
        annotation_shape = (annotation_dim, annotation_dim)
        annotation_center = Point(annotation_dim // 2, annotation_dim // 2)

        annotation: npt.NDArray[np.uint8] = np.zeros(
            (*annotation_shape, 3), dtype=np.uint8
        )
        annotation_alpha: npt.NDArray[np.uint8] = np.zeros(
            (*annotation_shape, 1), dtype=np.uint8
        )
        draw_text(
            scene=annotation,
            text=text,
            text_anchor=annotation_center,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            text_color=text_color,
            background_color=text_box_color if text_box_show else None,
        )
        draw_text(
            scene=annotation_alpha,
            text=text,
            text_anchor=annotation_center,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            text_color=Color.WHITE,
            background_color=Color.WHITE if text_box_show else None,
        )
        annotation = np.dstack((annotation, annotation_alpha)).astype(np.uint8)

        # Make sure text is displayed upright
        if 90 < line_angle_degrees % 360 < 270:
            annotation = cv2.flip(annotation, flipCode=-1).astype(np.uint8)

        from PIL import Image

        annotation = np.asarray(
            Image.fromarray(annotation).rotate(
                -line_angle_degrees,
                resample=Image.Resampling.BILINEAR,
                center=annotation_center.as_xy_float_tuple(),
            )
        ).astype(np.uint8)

        return annotation


class LineZoneAnnotatorMulticlass:
    """Draw per-class crossing counts for one or more `LineZone` instances.

    The annotator renders a table with one row per line zone and one column per class
    observed by the zones.
    """

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
    ) -> None:
        """Draw a table showing how many items of each class crossed each line.

        Args:
            table_position: The position of the table.
            table_color: The color of the table.
            table_margin: The margin of the table from the image border.
            table_padding: The padding of the table.
            table_max_width: The maximum width of the table.
            text_color: The color of the text.
            text_scale: The scale of the text.
            text_thickness: The thickness of the text.
            force_draw_class_ids: Instead of writing the class names,
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
        frame: npt.NDArray[np.uint8],
        line_zones: list[LineZone],
        line_zone_labels: list[str] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Draws a table with the number of objects of each class that crossed each
        line.

        Args:
            frame: The image on which the table will be drawn.
            line_zones: The line zones to be annotated.
            line_zone_labels: The labels, one for each line zone. If not
                provided, the default labels will be used.

        Returns:
            The image with the table drawn on it.
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
                    if self.force_draw_class_ids:
                        class_name = str(class_id)
                    elif class_id is None:
                        class_name = "None"
                    else:
                        class_name = class_id_to_name.get(class_id, str(class_id))
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
