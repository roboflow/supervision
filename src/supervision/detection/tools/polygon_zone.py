from collections import Counter
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision import Detections
from supervision import _cv2 as cv2
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask, _compact_mask_union
from supervision.detection.utils.converters import (
    MIN_POLYGON_POINT_COUNT,
    polygon_to_mask,
)
from supervision.draw.color import Color
from supervision.draw.utils import draw_filled_polygon, draw_polygon, draw_text
from supervision.geometry.core import (
    CoordinatesTransformation,
    Position,
    _transform_points,
)
from supervision.geometry.utils import get_polygon_center

# Largest canvas, in pixels, on which one oriented box is rasterized whole for
# PolygonZone.occupancy; larger polygons are rasterized on the zone's window.
_MAX_POLYGON_CANVAS_PIXELS = 1 << 24


class PolygonZone:
    """A class for defining a polygon-shaped zone within a frame for detecting objects.

    !!! warning

        Entry and exit counts
        (:attr:`~supervision.detection.tools.polygon_zone.PolygonZone.in_count`,
        :attr:`~supervision.detection.tools.polygon_zone.PolygonZone.out_count`
        and their per-class variants) and
        :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.time_in_zone`
        use the `tracker_id`. Read
        [here](https://trackers.roboflow.com/latest/) to learn how to plug
        tracking into your inference pipeline. Without a `tracker_id` the counts
        stay at zero and dwell time is `NaN`, while `current_count` and
        :attr:`~supervision.detection.tools.polygon_zone.PolygonZone.current_count_per_class`
        still work. Detections with a negative `tracker_id`, which trackers
        report for tracks they have not confirmed yet, or a non-finite one never
        enter or exit.

    Attributes:
        polygon: A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        triggering_anchors: Any iterable of positions specifying
            which anchors of the detections bounding box to consider when deciding on
            whether the detection fits within the PolygonZone
            (default: :class:`~supervision.geometry.core.Position` member
            ``BOTTOM_CENTER``).
        require_all_anchors: If `True` (default), a detection is considered inside
            the zone only when *every* anchor in `triggering_anchors` is inside.
            If `False`, the detection triggers as soon as *any* anchor is inside.
            Has no effect when `triggering_anchors` has a single entry.
            This is anchor-based, not a true geometric box/polygon intersection
            test: it fires only when a listed anchor point lands inside the mask.
            Use mask/IoU-based approaches instead when full-overlap semantics are
            required.
        minimum_crossing_threshold: Number of consecutive observations a tracked
            object must spend on the other side of the zone boundary before an
            entry or exit is counted.
        max_absent_frames: Number of consecutive frames a tracked object may be
            missing before its entry/exit state is dropped.
        current_count: The current count of detected objects within the zone
        current_count_per_class: The current count of detected objects within
            the zone, keyed by `class_id` (`int` for classified detections,
            `None` for unclassified ones). Classes with no object in the zone
            are left out.
        in_count: The number of tracked objects that have entered the zone.
        out_count: The number of tracked objects that have exited the zone.
        in_count_per_class: The number of tracked objects of each class that
            have entered the zone, keyed by `class_id`.
        out_count_per_class: The number of tracked objects of each class that
            have exited the zone, keyed by `class_id`.
        mask: The 2D bool mask for the polygon zone

    Example:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> polygon = np.array([[100, 200], [200, 100], [300, 200], [200, 300]])
        >>> polygon_zone = sv.PolygonZone(polygon=polygon)
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[180, 100, 220, 200], [400, 400, 450, 500]])
        ... )
        >>> is_detections_in_zone = polygon_zone.trigger(detections)
        >>> is_detections_in_zone
        array([ True, False])
        >>> polygon_zone.current_count
        1

        ```

        ```pycon
        >>> polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> polygon_zone = sv.PolygonZone(
        ...     polygon=polygon,
        ...     triggering_anchors=[sv.Position.TOP_LEFT, sv.Position.BOTTOM_RIGHT],
        ...     require_all_anchors=False,
        ... )
        >>> detections = sv.Detections(xyxy=np.array([[80, 80, 120, 120]]))
        >>> polygon_zone.trigger(detections)
        array([ True])

        ```

        Count tracked objects entering and leaving the zone:

        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> zone = sv.PolygonZone(
        ...     polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        ... )
        >>> outside = sv.Detections(
        ...     xyxy=np.array([[140.0, 40.0, 160.0, 60.0]]),
        ...     class_id=np.array([2]),
        ...     tracker_id=np.array([7]),
        ... )
        >>> inside = sv.Detections(
        ...     xyxy=np.array([[40.0, 40.0, 60.0, 60.0]]),
        ...     class_id=np.array([2]),
        ...     tracker_id=np.array([7]),
        ... )
        >>> for detections in [outside, inside, inside, outside]:
        ...     _ = zone.trigger(detections)
        >>> zone.in_count, zone.out_count
        (1, 1)
        >>> zone.in_count_per_class
        {2: 1}

        ```
    """

    def __init__(
        self,
        polygon: npt.NDArray[np.int64],
        triggering_anchors: Iterable[Position] = (Position.BOTTOM_CENTER,),
        require_all_anchors: bool = True,
        minimum_crossing_threshold: int = 1,
        max_absent_frames: int = 30,
    ) -> None:
        """Build a zone from a polygon.

        Args:
            polygon: Zone boundary of shape `(N, 2)` holding the `x`, `y`
                coordinates of its vertices.
            triggering_anchors: Which anchors of a detection's bounding box
                decide whether it falls inside the zone.
            require_all_anchors: Whether every triggering anchor must be inside
                for the detection to count, rather than any one of them.
            minimum_crossing_threshold: A tracked object must be seen on the
                other side of the zone boundary for this many consecutive
                observations before its entry or exit is counted. Frames in
                which the object is missing are skipped: they neither extend
                nor reset the run. Raise it to ignore anchors that flicker
                across the boundary. The first side an object is seen on is
                taken as its starting side with no count, so an object that
                first appears inside the zone has not entered it.
            max_absent_frames: The entry/exit state of a tracked object is kept
                while it is missing for up to this many consecutive frames, so
                a short occlusion does not reset it, and dropped after that to
                bound memory on long streams. Match it to the tracker's
                lost-track buffer. An object that is dropped while inside the
                zone, for example because it left the frame through the zone,
                is not counted as exiting.

        Raises:
            ValueError: If `polygon` is not of shape `(N, 2)`, has fewer than
                `MIN_POLYGON_POINT_COUNT` vertices, `triggering_anchors` is
                empty, or `minimum_crossing_threshold` or `max_absent_frames`
                is less than 1.
        """
        polygon = np.asarray(polygon)
        if polygon.ndim != 2 or polygon.shape[-1] != 2:
            raise ValueError(f"Polygon must have shape (N, 2); got {polygon.shape}.")
        # Fewer than three vertices enclose no area, so the mask below comes out
        # empty or a bare line and the zone silently never triggers. Reject it
        # here rather than let a zone that can never count look like a zone that
        # simply saw nothing. Zero vertices additionally breaks `np.max`.
        if len(polygon) < MIN_POLYGON_POINT_COUNT:
            raise ValueError(
                f"Polygon must have at least {MIN_POLYGON_POINT_COUNT} vertices "
                f"to enclose an area; got {len(polygon)}."
            )

        self.polygon = polygon.astype(int)
        # Materialize once so we can safely accept generators without exhausting them.
        self.triggering_anchors = list(triggering_anchors)
        if not self.triggering_anchors:
            raise ValueError("Triggering anchors cannot be empty.")
        self.require_all_anchors = require_all_anchors
        if minimum_crossing_threshold < 1:
            raise ValueError(
                "minimum_crossing_threshold must be at least 1; "
                f"got {minimum_crossing_threshold}."
            )
        if max_absent_frames < 1:
            raise ValueError(
                f"max_absent_frames must be at least 1; got {max_absent_frames}."
            )
        self.minimum_crossing_threshold = minimum_crossing_threshold
        self.max_absent_frames = max_absent_frames

        self.current_count = 0
        self._current_count_per_class: Counter[int | None] = Counter()
        self._in_count_per_class: Counter[int | None] = Counter()
        self._out_count_per_class: Counter[int | None] = Counter()
        # Side of the boundary (True = inside) each tracker was last confirmed on.
        # Entries and exits are counted against it, never against the raw previous
        # frame, so a sub-threshold flicker cannot become the reference side.
        self._confirmed_inside: dict[int, bool] = {}
        # Consecutive observations a tracker has spent away from its confirmed side.
        self._observations_away: dict[int, int] = {}
        self._frames_absent: dict[int, int] = {}
        # Timestamp at which each track's current stay inside the zone began.
        self._entered_at: dict[int, float] = {}
        # Timestamp of the first observation in each track's current run away from
        # its confirmed side, so a confirmed entry is dated when it really happened.
        self._away_since: dict[int, float] = {}
        self._frame_index = -1
        self._last_timestamp: float | None = None
        # Whether trigger calls pass explicit timestamps; fixed by the first call so
        # seconds and frame counts never mix in one dwell measurement.
        self._uses_explicit_timestamps: bool | None = None

        x_max, y_max = np.max(polygon, axis=0)
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
        )

    @property
    def current_count_per_class(self) -> dict[int | None, int]:
        """Objects inside the zone on the last frame, keyed by `class_id`."""
        return dict(self._current_count_per_class)

    @property
    def in_count(self) -> int:
        """Tracked objects that have entered the zone since creation or reset."""
        return sum(self._in_count_per_class.values())

    @property
    def out_count(self) -> int:
        """Tracked objects that have exited the zone since creation or reset."""
        return sum(self._out_count_per_class.values())

    @property
    def in_count_per_class(self) -> dict[int | None, int]:
        """Tracked objects that have entered the zone, keyed by `class_id`."""
        return dict(self._in_count_per_class)

    @property
    def out_count_per_class(self) -> dict[int | None, int]:
        """Tracked objects that have exited the zone, keyed by `class_id`."""
        return dict(self._out_count_per_class)

    def reset(self) -> None:
        """Clear all counts and per-track state so the zone can be reused across
        independent streams. The polygon and configuration are preserved.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            ... )
            >>> detections = sv.Detections(xyxy=np.array([[40.0, 40.0, 60.0, 60.0]]))
            >>> _ = zone.trigger(detections)
            >>> zone.current_count
            1
            >>> zone.reset()
            >>> zone.current_count, zone.in_count, zone.current_count_per_class
            (0, 0, {})

            ```
        """
        self.current_count = 0
        self._current_count_per_class.clear()
        self._in_count_per_class.clear()
        self._out_count_per_class.clear()
        self._confirmed_inside.clear()
        self._observations_away.clear()
        self._frames_absent.clear()
        self._entered_at.clear()
        self._away_since.clear()
        self._frame_index = -1
        self._last_timestamp = None
        self._uses_explicit_timestamps = None

    def time_in_zone(self, detections: Detections) -> npt.NDArray[np.float64]:
        """Report how long each tracked detection has been inside the zone.

        Dwell is measured from the start of each object's current stay to the
        latest
        :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`
        call, in the units of the `timestamp` values passed to it, or in frames
        when none are passed. A stay starts when an entry is confirmed, dated to
        the first observation of the run that confirmed it, or when an object is
        first seen already inside. It ends when an exit is confirmed, so a
        re-entry starts a new stay, while flicker shorter than
        `minimum_crossing_threshold` and absences up to `max_absent_frames` do
        not interrupt it. Detections are matched to the zone's state by
        `tracker_id`, so pass the same frame's detections given to
        :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`.

        Args:
            detections: Tracked detections to look up.

        Returns:
            An array aligned with `detections` holding each object's time in the
            zone, or `NaN` for an object that is not confirmed inside (with
            `minimum_crossing_threshold` above 1, an entry or exit takes effect
            only once confirmed), has a missing, negative or non-finite
            `tracker_id`, or when
            :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`
            was never called.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            ... )
            >>> def frame(x):
            ...     return sv.Detections(
            ...         xyxy=np.array([[x, 40.0, x + 20.0, 60.0]]),
            ...         tracker_id=np.array([7]),
            ...     )
            >>> fps = 2
            >>> for index, x in enumerate([140.0, 40.0, 40.0, 40.0]):
            ...     _ = zone.trigger(frame(x), timestamp=index / fps)
            >>> zone.time_in_zone(frame(40.0))
            array([1.])

            ```
        """
        entered_at = np.full(len(detections), np.nan)
        if detections.tracker_id is None or self._last_timestamp is None:
            return entered_at
        tracker_ids = np.asarray(detections.tracker_id)
        # Only finite, non-negative ids can have state; anything else stays NaN.
        trackable = np.flatnonzero(np.isfinite(tracker_ids) & (tracker_ids >= 0))
        entered_at[trackable] = [
            self._entered_at.get(int(tracker_ids[i]), np.nan) for i in trackable
        ]
        return self._last_timestamp - entered_at

    def occupancy(self, detections: Detections) -> float:
        """Measure the fraction of the zone covered by the detections.

        Unlike :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`,
        which tests anchor points, this compares the detections' full geometry
        with the zone's pixels. It uses the richest geometry present: masks if
        set, otherwise oriented boxes stored in `data` under
        `ORIENTED_BOX_COORDINATES`, otherwise the `xyxy` boxes. Overlapping
        detections are counted once, and pixels on a box or polygon edge count
        as covered, matching the zone mask. Every
        detection passed in is measured, so filter first to measure a subset,
        for example one class. Detections are measured in the zone's own frame
        coordinates; no `coord_transform` is applied. Calling it does not
        change the zone's counts.

        Args:
            detections: The detections whose geometry covers the zone.

        Returns:
            Covered zone pixels divided by all zone pixels, from `0.0` (empty)
            to `1.0` (fully covered).

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[0, 0], [99, 0], [99, 99], [0, 99]])
            ... )
            >>> detections = sv.Detections(
            ...     xyxy=np.array([[0.0, 0.0, 49.0, 99.0], [25.0, 0.0, 74.0, 99.0]]),
            ...     class_id=np.array([0, 1]),
            ... )
            >>> zone.occupancy(detections)
            0.75
            >>> zone.occupancy(detections[detections.class_id == 0])
            0.5

            ```
        """
        x0, y0 = (int(value) for value in np.maximum(self.polygon.min(axis=0), 0))
        zone = self.mask[y0:, x0:].astype(bool)
        zone_pixels = np.count_nonzero(zone)
        if zone_pixels == 0 or len(detections) == 0:
            return 0.0
        covered = _rasterize_detections(detections, origin=(x0, y0), shape=zone.shape)
        return float(np.count_nonzero(covered & zone) / zone_pixels)

    def trigger(
        self,
        detections: Detections,
        coord_transform: CoordinatesTransformation | None = None,
        timestamp: float | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Determines if the detections are within the polygon zone.

        Anchor points are calculated from original (unclipped) detection boxes to
        avoid per-zone clipping shifting anchor positions. This prevents a single
        detection from being counted in multiple non-overlapping zones due to
        clipping artifacts, although overlapping zones may still legitimately
        contain the same detection.

        Args:
            detections: The detections to be checked against the polygon zone
            coord_transform: Optional per-frame camera motion. Anchors are mapped
                into the zone's reference frame with
                :meth:`~supervision.geometry.core.CoordinatesTransformation.rel_to_abs`
                before the inside test; anchors mapped outside the zone's bounds
                or to non-finite values are outside.
            timestamp: Time of this frame, used by
                :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.time_in_zone`.
                Pass it on every call, for example `frame_index / fps` for
                seconds in a video or `time.monotonic()` for a live stream, or
                never, to measure time in frames. Equal consecutive timestamps
                are allowed.

        Returns:
            A boolean numpy array indicating
                if each detection is within the polygon zone

        Raises:
            ValueError: If `timestamp` is not finite, is earlier than the
                previous frame's timestamp, or is given on some calls but not
                others since construction or the last
                :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.reset`.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> class CameraShift:
            ...     def abs_to_rel(self, points): return points + [50, 0]
            ...     def rel_to_abs(self, points): return points - [50, 0]
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            ... )
            >>> detections = sv.Detections(xyxy=np.array([[120.0, 40.0, 140.0, 60.0]]))
            >>> zone.trigger(detections)
            array([False])
            >>> zone.trigger(detections, coord_transform=CameraShift())
            array([ True])

            ```
        """
        uses_explicit_timestamps = timestamp is not None
        if self._uses_explicit_timestamps not in (None, uses_explicit_timestamps):
            raise ValueError(
                "timestamp must be passed on every trigger call or on none, so "
                "dwell is not measured in mixed units. Call reset() to switch."
            )
        frame_timestamp = (
            float(timestamp) if timestamp is not None else self._frame_index + 1.0
        )
        if not np.isfinite(frame_timestamp):
            raise ValueError(f"timestamp must be finite; got {frame_timestamp}.")
        if self._last_timestamp is not None and frame_timestamp < self._last_timestamp:
            raise ValueError(
                f"timestamp must not decrease; got {frame_timestamp} after "
                f"{self._last_timestamp}. Call reset() before reusing the zone."
            )

        if len(detections) == 0:
            is_in_zone = np.array([], dtype=bool)
        else:
            is_in_zone = self._is_in_zone(detections, coord_transform)
        # The frame is committed only once the inside test has succeeded, so a
        # failing coord_transform leaves the clock where it was.
        self._uses_explicit_timestamps = uses_explicit_timestamps
        self._frame_index += 1
        self._last_timestamp = frame_timestamp
        self._update_counts(detections, is_in_zone, frame_timestamp)
        return cast(npt.NDArray[np.bool_], is_in_zone)

    def _is_in_zone(
        self,
        detections: Detections,
        coord_transform: CoordinatesTransformation | None,
    ) -> npt.NDArray[np.bool_]:
        """Test each detection's triggering anchors against the zone mask."""
        anchor_coordinates = np.array(
            [
                detections.get_anchors_coordinates(anchors)
                for anchors in self.triggering_anchors
            ]
        )
        mapped_in_bounds = None
        if coord_transform is not None:
            mapped = _transform_points(anchor_coordinates, coord_transform.rel_to_abs)
            # -1 is just outside the mask, so non-finite anchors fail the bounds
            # check below; clipping far finite ones to the same range keeps them
            # outside without overflowing the integer cast.
            mapped = np.nan_to_num(mapped, nan=-1.0, posinf=-1.0, neginf=-1.0)
            mask_h, mask_w = self.mask.shape
            mapped_in_bounds = (
                (mapped[:, :, 0] >= 0)
                & (mapped[:, :, 1] >= 0)
                & (mapped[:, :, 0] < mask_w)
                & (mapped[:, :, 1] < mask_h)
            )
            anchor_coordinates = np.clip(mapped, -1.0, float(max(self.mask.shape)))
        all_anchors = np.rint(anchor_coordinates).astype(int)

        mask_h, mask_w = self.mask.shape
        x, y = all_anchors[:, :, 0], all_anchors[:, :, 1]
        in_bounds = (x >= 0) & (y >= 0) & (x < mask_w) & (y < mask_h)
        if mapped_in_bounds is not None:
            # Rounding must not pull a truly out-of-bounds point onto the mask edge.
            in_bounds &= mapped_in_bounds
        x_safe = np.clip(x, 0, mask_w - 1)
        y_safe = np.clip(y, 0, mask_h - 1)
        anchor_hits = in_bounds & self.mask[y_safe, x_safe]
        reduce = np.all if self.require_all_anchors else np.any
        is_in_zone = reduce(anchor_hits, axis=0)
        return cast(npt.NDArray[np.bool_], is_in_zone.astype(bool))

    def _update_counts(
        self,
        detections: Detections,
        is_in_zone: npt.NDArray[np.bool_],
        timestamp: float,
    ) -> None:
        """Refresh this frame's occupancy counts and advance each track's entry/exit
        state, counting confirmed boundary crossings."""
        self.current_count = int(np.sum(is_in_zone))
        self._current_count_per_class.clear()
        if detections.class_id is None:
            if self.current_count:
                self._current_count_per_class[None] = self.current_count
        else:
            class_ids, counts = np.unique(
                detections.class_id[is_in_zone], return_counts=True
            )
            self._current_count_per_class.update(
                {int(c): int(n) for c, n in zip(class_ids, counts)}
            )

        if detections.tracker_id is None:
            self._age_absent_tracks(present=set())
            return

        # Trackers report a negative id for tracks they have not confirmed yet, and
        # several distinct objects can share it, so such detections get no state.
        # Non-finite ids in float arrays cannot be keys either, so they are skipped.
        tracker_ids = detections.tracker_id
        confirmed = np.flatnonzero(np.isfinite(tracker_ids) & (tracker_ids >= 0))
        # An id repeated within one frame, for example after merging overlapping
        # sources, is one observation; counting each copy would let a single frame
        # satisfy minimum_crossing_threshold. The first copy is kept.
        _, first_copies = np.unique(tracker_ids[confirmed], return_index=True)
        confirmed = confirmed[np.sort(first_copies)]
        self._age_absent_tracks(
            present={int(detections.tracker_id[i]) for i in confirmed}
        )
        for i in confirmed:
            key = int(detections.tracker_id[i])
            inside = bool(is_in_zone[i])
            # The first observed side seeds the reference; nothing was crossed yet,
            # but an object first seen inside is already dwelling there.
            if key not in self._confirmed_inside:
                self._confirmed_inside[key] = inside
                if inside:
                    self._entered_at[key] = timestamp
                continue
            if inside == self._confirmed_inside[key]:
                self._observations_away.pop(key, None)
                self._away_since.pop(key, None)
                continue

            observations_away = self._observations_away.get(key, 0) + 1
            self._away_since.setdefault(key, timestamp)
            if observations_away < self.minimum_crossing_threshold:
                self._observations_away[key] = observations_away
                continue

            self._observations_away.pop(key, None)
            crossed_at = self._away_since.pop(key)
            self._confirmed_inside[key] = inside
            class_id = (
                None if detections.class_id is None else int(detections.class_id[i])
            )
            if inside:
                self._in_count_per_class[class_id] += 1
                self._entered_at[key] = crossed_at
            else:
                self._out_count_per_class[class_id] += 1
                self._entered_at.pop(key, None)

    def _age_absent_tracks(self, present: set[int]) -> None:
        """Advance the absence counter of every known track missing from this frame,
        dropping the state of tracks absent beyond `max_absent_frames`."""
        for key in list(self._confirmed_inside):
            if key in present:
                self._frames_absent.pop(key, None)
                continue
            frames_absent = self._frames_absent.get(key, 0) + 1
            if frames_absent > self.max_absent_frames:
                del self._confirmed_inside[key]
                self._observations_away.pop(key, None)
                self._away_since.pop(key, None)
                self._entered_at.pop(key, None)
                self._frames_absent.pop(key, None)
            else:
                self._frames_absent[key] = frames_absent


def _rasterize_detections(
    detections: Detections, origin: tuple[int, int], shape: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Rasterize the union of the detections' richest geometry into a window of `shape`
    whose top-left pixel is at absolute `origin`."""
    if detections.mask is not None:
        return _rasterize_masks(detections.mask, origin, shape)
    if ORIENTED_BOX_COORDINATES in detections.data:
        polygons = np.asarray(detections.data[ORIENTED_BOX_COORDINATES])
        if polygons.ndim != 3 or polygons.shape[2] != 2:
            raise ValueError(
                f"detections.data[{ORIENTED_BOX_COORDINATES!r}] must have shape "
                f"(N, K, 2); got {polygons.shape}."
            )
        return _rasterize_polygons(polygons, origin, shape)
    return _rasterize_boxes(detections.xyxy, origin, shape)


def _or_into_window(
    window: npt.NDArray[np.bool_],
    origin: tuple[int, int],
    region: npt.NDArray[np.bool_],
    region_origin: tuple[int, int],
) -> None:
    """OR a boolean `region` placed at absolute `region_origin` into `window`, whose
    top-left pixel is at absolute `origin`, ignoring parts outside it."""
    x0, y0 = origin
    region_x, region_y = region_origin
    left, top = max(region_x, x0), max(region_y, y0)
    right = min(region_x + region.shape[1], x0 + window.shape[1])
    bottom = min(region_y + region.shape[0], y0 + window.shape[0])
    if right <= left or bottom <= top:
        return
    window[top - y0 : bottom - y0, left - x0 : right - x0] |= region[
        top - region_y : bottom - region_y, left - region_x : right - region_x
    ]


def _rasterize_masks(
    masks: npt.NDArray[np.bool_] | CompactMask,
    origin: tuple[int, int],
    shape: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Union instance masks inside the window without materializing a full `(N, H, W)`
    stack for compact masks."""
    x0, y0 = origin
    height, width = shape
    window = np.zeros(shape, dtype=bool)
    if isinstance(masks, CompactMask):
        # Only masks whose crop touches the window can cover it; unioning the rest
        # would make the cost grow with every object in the frame.
        bounds = masks.bbox_xyxy
        near = (
            (bounds[:, 0] < x0 + width)
            & (bounds[:, 2] >= x0)
            & (bounds[:, 1] < y0 + height)
            & (bounds[:, 3] >= y0)
        )
        if not near.any():
            return window
        union = _compact_mask_union([masks[near]])
        union_x, union_y = (int(value) for value in union.offsets[0])
        _or_into_window(window, origin, union.crop(0), (union_x, union_y))
    else:
        region = np.any(masks[:, y0 : y0 + height, x0 : x0 + width], axis=0)
        _or_into_window(window, origin, region, origin)
    return window


def _rasterize_polygons(
    polygons: npt.NDArray[np.number],
    origin: tuple[int, int],
    shape: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Union filled polygons of shape `(N, K, 2)` inside the window, matching what
    `polygon_to_mask` paints on an unclipped canvas."""
    x0, y0 = origin
    height, width = shape
    window = np.zeros(shape, dtype=bool)
    # Truncate absolute vertices, as polygon_to_mask does; the clip only keeps
    # far-off vertices inside the int32 range fillPoly accepts.
    vertices = np.clip(polygons, -(2**30), 2**30).astype(np.int32)
    # One fillPoly per polygon: a single call over several polygons uses the
    # even-odd rule and would clear the pixels where two detections overlap.
    for polygon in vertices:
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        if (
            high[0] < x0
            or high[1] < y0
            or low[0] >= x0 + width
            or low[1] >= y0 + height
        ):
            continue
        span_x, span_y = (int(value) for value in high - low + 1)
        # fillPoly shifts boundary pixels of edges it clips at the canvas border,
        # so each polygon is painted on a canvas that contains it whole; only
        # implausibly large polygons fall back to the window, at ±1 px accuracy.
        if span_x * span_y <= _MAX_POLYGON_CANVAS_PIXELS:
            canvas_origin = (int(low[0]), int(low[1]))
            canvas = np.zeros((span_y, span_x), dtype=np.uint8)
        else:
            canvas_origin = origin
            canvas = np.zeros(shape, dtype=np.uint8)
        shifted = polygon - np.array(canvas_origin, dtype=np.int32)
        cv2.fillPoly(canvas, [shifted], color=(1,))
        _or_into_window(window, origin, canvas.astype(bool), canvas_origin)
    return window


def _rasterize_boxes(
    xyxy: npt.NDArray[np.number],
    origin: tuple[int, int],
    shape: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Union axis-aligned boxes inside the window, edges inclusive, with a 2-D
    difference array so the cost does not grow with the box count."""
    height, width = shape
    offset = np.array(origin)
    size = np.array([width, height])
    corners = np.asarray(xyxy, dtype=np.float64).reshape(-1, 2, 2)
    # Clipping to one pixel beyond the window before truncating keeps every
    # in-window coordinate exact and avoids integer overflow for far-off boxes.
    corners = np.clip(corners, offset - 1, offset + size)
    corners = np.trunc(corners).astype(np.int64) - offset
    top_left = corners.min(axis=1)
    bottom_right = corners.max(axis=1)
    visible = np.all(bottom_right >= 0, axis=1) & np.all(top_left < size, axis=1)
    top_left = np.maximum(top_left[visible], 0)
    bottom_right = np.minimum(bottom_right[visible], size - 1) + 1

    edges = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(edges, (top_left[:, 1], top_left[:, 0]), 1)
    np.add.at(edges, (top_left[:, 1], bottom_right[:, 0]), -1)
    np.add.at(edges, (bottom_right[:, 1], top_left[:, 0]), -1)
    np.add.at(edges, (bottom_right[:, 1], bottom_right[:, 0]), 1)
    coverage = edges.cumsum(axis=0).cumsum(axis=1)
    return coverage[:height, :width] > 0


class PolygonZoneAnnotator:
    """A class for annotating a polygon-shaped zone within a frame with a count of
    detected objects.

    Attributes:
        zone: The polygon zone to be annotated
        color: The color to draw the polygon lines, default is white
        thickness: The thickness of the polygon lines, default is 2
        text_color: The color of the text on the polygon, default is black
        text_scale: The scale of the text on the polygon, default is 0.5
        text_thickness: The thickness of the text on the polygon, default is 1
        text_padding: The padding around the text on the polygon, default is 10
        font: The font type for the text on the polygon,
            default is cv2.FONT_HERSHEY_SIMPLEX
        center: The center of the polygon for text placement
        display_in_zone_count: Show the label of the zone or not. Default is True
        opacity: The opacity of zone filling when drawn on the scene. Default is 0

    Example:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> polygon = np.array([[100, 200], [200, 100], [300, 200], [200, 300]])
        >>> polygon_zone = sv.PolygonZone(polygon=polygon)
        >>> zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, thickness=2)
        >>> scene = np.zeros((400, 400, 3), dtype=np.uint8)
        >>> annotated_scene = zone_annotator.annotate(scene=scene)
        >>> annotated_scene.shape
        (400, 400, 3)

        ```
    """

    def __init__(
        self,
        zone: PolygonZone,
        color: Color = Color.WHITE,
        thickness: int = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        display_in_zone_count: bool = True,
        opacity: float = 0,
    ) -> None:
        self.zone = zone
        self.color = color
        self.thickness = thickness
        self.text_color = text_color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.center = get_polygon_center(polygon=zone.polygon)
        self.display_in_zone_count = display_in_zone_count
        self.opacity = opacity

    def annotate(
        self, scene: npt.NDArray[Any], label: str | None = None
    ) -> npt.NDArray[Any]:
        """Annotates the polygon zone within a frame with a count of detected objects.

        Args:
            scene: The image on which the polygon zone will be annotated
            label: A label for the count of detected objects
                within the polygon zone (default: None)

        Returns:
            The image with the polygon zone and count of detected objects
        """
        if self.opacity == 0:
            annotated_frame = draw_polygon(
                scene=scene,
                polygon=self.zone.polygon,
                color=self.color,
                thickness=self.thickness,
            )
        else:
            annotated_frame = draw_filled_polygon(
                scene=scene.copy(),
                polygon=self.zone.polygon,
                color=self.color,
                opacity=self.opacity,
            )
            annotated_frame = draw_polygon(
                scene=annotated_frame,
                polygon=self.zone.polygon,
                color=self.color,
                thickness=self.thickness,
            )

        if self.display_in_zone_count:
            annotated_frame = draw_text(
                scene=annotated_frame,
                text=str(self.zone.current_count) if label is None else label,
                text_anchor=self.center,
                background_color=self.color,
                text_color=self.text_color,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                text_padding=self.text_padding,
                text_font=self.font,
            )

        return cast(npt.NDArray[Any], annotated_frame)
