from collections import Counter
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision import Detections
from supervision import _cv2 as cv2
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import (
    CompactMask,
    _rle_split_cols,
)
from supervision.detection.utils.converters import (
    MIN_POLYGON_POINT_COUNT,
    polygon_to_mask,
)
from supervision.draw.color import Color
from supervision.draw.utils import (
    _map_to_drawing_coordinates,
    draw_filled_polygon,
    draw_polygon,
    draw_text,
)
from supervision.geometry.core import (
    CoordinatesTransformation,
    Point,
    Position,
    _transform_points,
)
from supervision.geometry.utils import get_polygon_center


def _polygons_union_mask(
    polygons: npt.NDArray[np.number], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Rasterize the union of filled polygons onto one boolean canvas.

    Args:
        polygons: Polygon vertices of shape ``(N, K, 2)`` in ``x``, ``y`` pixels,
            truncated to integers as
            :func:`~supervision.detection.utils.converters.polygon_to_mask` does.
            Polygons with non-finite vertices cover nothing.
        resolution_wh: Canvas width and height in pixels.

    Returns:
        Boolean array of shape ``(height, width)`` marking covered pixels.
    """
    width, height = resolution_wh
    canvas = np.zeros((height, width), dtype=np.uint8)
    finite = np.isfinite(polygons).all(axis=(1, 2))
    # One `fillPoly` call shares a single even-odd fill across all its contours,
    # so overlapping polygons would cancel each other out. Painting each polygon
    # separately onto the same canvas keeps their union.
    valid = np.trunc(polygons[finite].astype(np.float64))
    limits = np.iinfo(np.int32)
    if np.any((valid < limits.min) | (valid > limits.max)):
        raise ValueError("Oriented box coordinates must fit in int32.")
    for polygon in valid.astype(np.int32):
        cv2.fillPoly(canvas, [polygon], color=(1,))
    return canvas.astype(bool)


def _boxes_union_mask(
    xyxy: npt.NDArray[np.number], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Rasterize the union of half-open axis-aligned boxes onto one boolean canvas.

    Covers the same pixels as
    :func:`~supervision.detection.utils.converters.xyxy_to_mask` with
    ``coordinate_convention="exclusive"`` reduced over the boxes, but allocates
    only the canvas, never one mask per box: box corners are accumulated in a
    2D difference array whose prefix sums count how many boxes cover each pixel.

    Args:
        xyxy: Boxes of shape ``(N, 4)`` as ``x_min``, ``y_min``, ``x_max``,
            ``y_max`` pixels, truncated toward zero. Boxes with a non-finite
            coordinate cover nothing.
        resolution_wh: Canvas width and height in pixels.

    Returns:
        Boolean array of shape ``(height, width)`` marking covered pixels.
    """
    width, height = resolution_wh
    boxes = np.asarray(xyxy, dtype=np.float64)
    boxes = boxes[np.isfinite(boxes).all(axis=1)]
    # Truncate like `int()` in `xyxy_to_mask`, then clip every edge into the
    # canvas so each half-open span [min, max) indexes inside the counter below.
    limits = np.array([width, height, width, height], dtype=np.float64)
    edges = np.clip(np.trunc(boxes), 0, limits).astype(np.intp)
    edges = edges[(edges[:, 2] > edges[:, 0]) & (edges[:, 3] > edges[:, 1])]
    x_min, y_min, x_max, y_max = edges.T
    # +1 at the top-left and bottom-right corners, -1 at the other two; the
    # prefix sums along both axes turn this into a per-pixel coverage count.
    # `np.add.at` accumulates corners shared by duplicate or adjacent boxes.
    coverage = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(coverage, (y_min, x_min), 1)
    np.add.at(coverage, (y_min, x_max), -1)
    np.add.at(coverage, (y_max, x_min), -1)
    np.add.at(coverage, (y_max, x_max), 1)
    np.cumsum(coverage, axis=0, out=coverage)
    np.cumsum(coverage, axis=1, out=coverage)
    return coverage[:height, :width] > 0


def _integral_keys(
    values: npt.ArrayLike | None, count: int, allow_negative: bool
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Convert per-detection ids into integer dictionary keys without ever raising.

    Integer ids are used as they are and integral floats such as `3.0` become
    `3`. Ids that cannot name one object reliably are marked unusable instead:
    non-numeric or boolean arrays, non-finite or fractional floats, values that
    do not fit `int64`, and negative values unless `allow_negative` is set.

    Args:
        values: Per-detection ids, or `None` when the detections carry none.
        count: Number of detections, used when `values` is `None`.
        allow_negative: Whether negative ids are usable keys.

    Returns:
        A pair of arrays of length `count`: the `int64` keys, `0` where unusable,
        and a mask marking which keys are usable.
    """
    if values is None:
        return np.zeros(count, dtype=np.int64), np.zeros(count, dtype=bool)
    ids = np.asarray(values)
    kind = ids.dtype.kind
    if kind not in "iuf":
        return np.zeros(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=bool)
    if kind == "i":
        usable = np.ones(len(ids), dtype=bool)
    elif kind == "u":
        usable = ids <= np.iinfo(np.int64).max
    else:
        # Fractional ids would collapse onto one key when truncated, and values
        # beyond 2**63 cannot be cast to int64, so both are unusable.
        usable = np.isfinite(ids) & (ids == np.trunc(ids)) & (np.abs(ids) < 2.0**63)
    if not allow_negative:
        usable &= ids >= 0
    keys = np.where(usable, ids, 0).astype(np.int64)
    return keys, usable


def _validate_shape(
    actual: tuple[int, ...], expected: tuple[int | str, ...], name: str
) -> None:
    """Reject an array whose shape differs from ``expected`` before it is reduced.

    Args:
        actual: Shape of the array to check.
        expected: Required shape; a string entry names a free dimension of any size.
        name: How the array is called in the error message.

    Raises:
        ValueError: If the dimension count differs or a fixed dimension mismatches.
    """
    fixed_match = all(
        isinstance(want, str) or want == got
        for want, got in zip(expected, actual, strict=False)
    )
    if len(actual) != len(expected) or not fixed_match:
        expected_text = "(" + ", ".join(str(dim) for dim in expected) + ")"
        raise ValueError(
            f"{name} must have shape {expected_text}; got shape {tuple(actual)}."
        )


def _detections_to_union_mask(
    detections: Detections, resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Rasterize the union of all detection footprints onto one boolean canvas.

    The footprint source follows the order
    :attr:`~supervision.detection.core.Detections.area` uses: segmentation masks
    (dense or :class:`~supervision.detection.compact_mask.CompactMask`, which is
    never densified), then oriented boxes stored in
    ``data[ORIENTED_BOX_COORDINATES]``, then axis-aligned boxes. Axis-aligned
    boxes use the exclusive convention of
    :attr:`~supervision.detection.core.Detections.box_area`, so a box
    ``[0, 0, 10, 10]`` covers 100 pixels.

    The canvas shares its origin with the frame the detections come from, but
    need not share its size: masks are cropped to the canvas when the frame is
    larger and zero-padded when it is smaller, and boxes are clipped to it.

    Args:
        detections: Detections whose footprints are unioned.
        resolution_wh: Canvas width and height in pixels.

    Returns:
        Boolean array of shape ``(height, width)`` marking covered pixels; all
            `False` for empty `detections`.

    Raises:
        ValueError: If a dense mask is not of shape ``(N, H, W)`` or the oriented
            box corners are not of shape ``(N, K, 2)``, with ``N`` detections.
    """
    width, height = resolution_wh
    if len(detections) == 0:
        return np.zeros((height, width), dtype=bool)
    mask = detections.mask
    if mask is not None:
        if not isinstance(mask, CompactMask):
            _validate_shape(np.shape(mask), (len(detections), "H", "W"), "mask")
        canvas = np.zeros((height, width), dtype=bool)
        if isinstance(mask, CompactMask):
            # Read only columns intersecting the zone; even a single foreground
            # run can span a crop too large to decode or expand into intervals.
            for rle, shape, offset in zip(mask._rles, mask._crop_shapes, mask.offsets):
                crop_h, crop_w = (int(value) for value in shape)
                offset_x, offset_y = (int(value) for value in offset)
                x_start = max(0, -offset_x)
                x_stop = min(crop_w, width - offset_x)
                if x_stop <= x_start or offset_y >= height or offset_y + crop_h <= 0:
                    continue
                columns = _rle_split_cols(rle, crop_h, crop_w, x_start, x_stop - 1)
                for x, runs in enumerate(columns, start=x_start + offset_x):
                    y = offset_y
                    for index, length in enumerate(runs):
                        end = y + length
                        if index % 2 and end > 0 and y < height:
                            canvas[max(y, 0) : min(end, height), x] = True
                        y = end
        else:
            # Reduce only the overlapping window: pixels beyond the canvas are
            # outside the zone, and canvas pixels beyond the frame stay empty.
            overlap_h = min(int(mask.shape[1]), height)
            overlap_w = min(int(mask.shape[2]), width)
            canvas[:overlap_h, :overlap_w] = np.any(
                mask[:, :overlap_h, :overlap_w], axis=0
            )
        return canvas
    if ORIENTED_BOX_COORDINATES in detections.data:
        polygons = np.asarray(detections.data[ORIENTED_BOX_COORDINATES])
        _validate_shape(
            polygons.shape,
            (len(detections), "K", 2),
            f"data[{ORIENTED_BOX_COORDINATES!r}]",
        )
        return _polygons_union_mask(polygons, resolution_wh)
    return _boxes_union_mask(detections.xyxy, resolution_wh)


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
        report for tracks they have not confirmed yet, or a non-finite,
        fractional or non-numeric one never enter or exit, but still count in
        `current_count`. A `class_id` that is not a finite integer counts under
        the `None` class key.

    Entry and exit counts share their names, their `class_id` keys and their
    `minimum_crossing_threshold` debounce and `max_absent_frames` eviction with
    :class:`~supervision.detection.line_zone.LineZone`, so the same side
    sequence of a tracked object gives the same counts. They differ in that a
    `tracker_id` repeated within one frame is a single observation, ids that are
    not finite non-negative integers are skipped rather than used as keys, and
    frames without a `tracker_id` emit no warning.

    Call :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`
    exactly once per frame for each zone; frame-mode dwell,
    `minimum_crossing_threshold` and `max_absent_frames` count `trigger` calls,
    and a tracked object missing from a call is counted absent. Use one
    `PolygonZone` per stream (or
    :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.reset` between
    streams); tracker ids from different streams collide.

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
        minimum_crossing_threshold: Number of consecutive observations, that is
            `trigger` calls in which it is present, a tracked object must spend
            on the other side of the zone boundary before an entry or exit is
            counted.
        max_absent_frames: Number of consecutive `trigger` calls missing a
            tracked object at which its entry/exit state is dropped.
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
        max_absent_frames: int | None = None,
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
                observations, that is `trigger` calls in which it is present,
                before its entry or exit is counted. Calls in which the object
                is missing are skipped: they neither extend nor reset the run.
                Raise it to ignore anchors that flicker across the boundary.
                The first side an object is seen on is taken as its starting
                side with no count, so an object that first appears inside the
                zone has not entered it.
            max_absent_frames: Number of consecutive `trigger` calls missing a
                tracked object at which its entry/exit state is dropped, as in
                :class:`~supervision.detection.line_zone.LineZone`. An object
                that returns sooner keeps its state, so a short occlusion does
                not reset it, while dropping bounds memory on long streams. If
                `None`, the limit is the larger of 30 and
                `minimum_crossing_threshold + 1`. Match it to the tracker's
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
        if max_absent_frames is None:
            max_absent_frames = max(30, minimum_crossing_threshold + 1)
        if max_absent_frames < 1:
            raise ValueError("max_absent_frames must be at least 1.")
        self.minimum_crossing_threshold = minimum_crossing_threshold
        self.max_absent_frames: int = max_absent_frames

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
        if np.issubdtype(polygon.dtype, np.integer):
            x_max, y_max = int(x_max), int(y_max)
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
        counted as `trigger` calls when none are passed. A stay starts when an
        entry is confirmed, dated to the first observation of the run that
        confirmed it, or when an object is first seen already inside. It ends
        when an exit is confirmed, so a re-entry starts a new stay, while
        flicker shorter than `minimum_crossing_threshold` and absences shorter
        than `max_absent_frames` do not interrupt it. Detections are matched to
        the zone's state by
        `tracker_id`, so pass the same frame's detections given to
        :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`.

        Args:
            detections: Tracked detections to look up.

        Returns:
            An array aligned with `detections` holding each object's time in the
            zone, or `NaN` for an object that is not confirmed inside (with
            `minimum_crossing_threshold` above 1, an entry or exit takes effect
            only once confirmed), has a missing, negative, non-finite,
            fractional or non-numeric `tracker_id`, or when
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
        if self._last_timestamp is None:
            return entered_at
        # Only ids `trigger` could have followed can have state; the rest stay NaN.
        tracker_keys, tracked = _integral_keys(
            detections.tracker_id, len(detections), allow_negative=False
        )
        entered_at[tracked] = [
            self._entered_at.get(key, np.nan) for key in tracker_keys[tracked].tolist()
        ]
        return self._last_timestamp - entered_at

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
        contain the same detection. Negative and non-finite anchors are outside.

        Call this exactly once per frame for each zone; frame-mode dwell,
        `minimum_crossing_threshold` and `max_absent_frames` count `trigger`
        calls, and a tracked object missing from a call is counted absent. Use
        one :class:`~supervision.detection.tools.polygon_zone.PolygonZone` per
        stream (or
        :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.reset`
        between streams); tracker ids from different streams collide. A shared
        zone is not reliably detected: with explicit timestamps it raises only
        when a call's timestamp is earlier than the previous one, so streams
        sharing a monotonic clock mix their ids silently.

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
        # Ids are normalized before any state changes, and never raise, so an
        # unusual tracker or class id cannot leave a half-updated frame behind.
        tracker_keys, tracked = _integral_keys(
            detections.tracker_id, len(detections), allow_negative=False
        )
        class_keys, classified = _integral_keys(
            detections.class_id, len(detections), allow_negative=True
        )
        # The frame is committed only once the inside test has succeeded, so a
        # failing coord_transform leaves the clock where it was.
        self._uses_explicit_timestamps = uses_explicit_timestamps
        self._frame_index += 1
        self._last_timestamp = frame_timestamp
        self._update_counts(
            is_in_zone, tracker_keys, tracked, class_keys, classified, frame_timestamp
        )
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
        if coord_transform is not None:
            mapped = _transform_points(anchor_coordinates, coord_transform.rel_to_abs)
            anchor_coordinates = np.nan_to_num(
                mapped, nan=-1.0, posinf=-1.0, neginf=-1.0
            )

        mask_h, mask_w = self.mask.shape
        rounded = np.rint(anchor_coordinates)
        mask_size = np.array([mask_w, mask_h], dtype=np.intp)
        # Test the lower bounds before rounding so a negative subpixel cannot
        # become zero. The rounded coordinates must fit the mask before casting.
        valid_coordinates = (anchor_coordinates >= 0) & (rounded < mask_size)
        in_bounds = valid_coordinates[..., 0] & valid_coordinates[..., 1]
        safe_anchors = np.where(in_bounds[..., None], rounded, 0).astype(np.intp)
        anchor_hits = in_bounds & self.mask[safe_anchors[..., 1], safe_anchors[..., 0]]
        reduce = np.all if self.require_all_anchors else np.any
        is_in_zone = reduce(anchor_hits, axis=0)
        return cast(npt.NDArray[np.bool_], is_in_zone.astype(bool))

    def _update_counts(
        self,
        is_in_zone: npt.NDArray[np.bool_],
        tracker_keys: npt.NDArray[np.int64],
        tracked: npt.NDArray[np.bool_],
        class_keys: npt.NDArray[np.int64],
        classified: npt.NDArray[np.bool_],
        timestamp: float,
    ) -> None:
        """Refresh this frame's occupancy counts and advance each track's entry/exit
        state, counting confirmed boundary crossings.

        Takes keys built by
        :func:`~supervision.detection.tools.polygon_zone._integral_keys`, so
        nothing here can raise once the frame is committed. Detections with an
        unusable `class_id` are counted under the `None` key, so the per-class
        counts always add up to `current_count`.
        """
        self.current_count = int(np.sum(is_in_zone))
        self._current_count_per_class.clear()
        class_ids, counts = np.unique(
            class_keys[is_in_zone & classified], return_counts=True
        )
        self._current_count_per_class.update(
            {int(c): int(n) for c, n in zip(class_ids, counts)}
        )
        unclassified_count = int(np.count_nonzero(is_in_zone & ~classified))
        if unclassified_count:
            self._current_count_per_class[None] = unclassified_count

        # Trackers report a negative id for tracks they have not confirmed yet, and
        # several distinct objects can share it, so such detections get no state,
        # nor do ids that are not finite integers.
        confirmed = np.flatnonzero(tracked)
        # An id repeated within one frame, for example after merging overlapping
        # sources, is one observation; counting each copy would let a single frame
        # satisfy minimum_crossing_threshold. The first copy is kept.
        _, first_copies = np.unique(tracker_keys[confirmed], return_index=True)
        confirmed = confirmed[np.sort(first_copies)]
        self._age_absent_tracks(present=set(tracker_keys[confirmed].tolist()))
        for i in confirmed:
            key = int(tracker_keys[i])
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
            class_id = int(class_keys[i]) if classified[i] else None
            if inside:
                self._in_count_per_class[class_id] += 1
                self._entered_at[key] = crossed_at
            else:
                self._out_count_per_class[class_id] += 1
                self._entered_at.pop(key, None)

    def _age_absent_tracks(self, present: set[int]) -> None:
        """Advance the absence counter of every known track missing from this frame,
        dropping the state of tracks absent for `max_absent_frames` calls."""
        for key in list(self._confirmed_inside):
            if key in present:
                self._frames_absent.pop(key, None)
                continue
            frames_absent = self._frames_absent.get(key, 0) + 1
            if frames_absent >= self.max_absent_frames:
                del self._confirmed_inside[key]
                self._observations_away.pop(key, None)
                self._away_since.pop(key, None)
                self._entered_at.pop(key, None)
                self._frames_absent.pop(key, None)
            else:
                self._frames_absent[key] = frames_absent

    def occupancy_ratio(self, detections: Detections) -> float:
        """Measure which fraction of the zone's pixels the detections cover.

        Every detection footprint is rasterized onto one boolean canvas and the
        union is intersected with the zone, so overlapping detections are counted
        once. The footprint comes from the first geometry the detections carry,
        in the order
        :attr:`~supervision.detection.core.Detections.area` uses: segmentation
        masks, then oriented boxes stored in
        ``data[ORIENTED_BOX_COORDINATES]``, then axis-aligned boxes. Axis-aligned
        boxes are half-open, matching
        :attr:`~supervision.detection.core.Detections.box_area`, while the zone
        includes the pixels on its vertices, so a box ``[0, 0, 100, 100]`` covers
        exactly a zone with vertices from ``0`` to ``99``.

        The detections must be in the pixel coordinates of the frame the zone's
        polygon was drawn on. No frame size is needed: the union is computed over
        the zone's own
        :attr:`~supervision.detection.tools.polygon_zone.PolygonZone.mask`
        extent, so frame-sized masks are cropped to it, masks of a smaller frame
        are zero-padded, and anything outside the zone is ignored. Camera motion
        compensation is not supported yet: unlike
        :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`,
        this method takes no ``coord_transform``, so map the detections into the
        zone's reference frame before calling it.

        The ratio divides pixel counts, never sums of mask values: a pixel is
        covered when any detection mask is nonzero there, so non-boolean masks, which
        :class:`~supervision.detection.core.Detections` deprecates, treat every
        nonzero value, soft scores included, as foreground. The method leaves
        :attr:`~supervision.detection.tools.polygon_zone.PolygonZone.current_count`
        unchanged.

        Args:
            detections: The detections whose combined footprint is measured.

        Returns:
            The covered share of the zone's pixels, between `0.0` and `1.0`;
                `0.0` for empty `detections`.

        Raises:
            ValueError: If the zone mask is not 2D or has no nonzero pixels, for
                example because the polygon lies entirely at negative coordinates,
                even when `detections` is empty; or if a dense detection mask is
                not of shape `(N, H, W)` or the oriented box corners are not of
                shape `(N, K, 2)`, with `N` detections, or their finite coordinates
                cannot be represented as `int32` after truncation.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[0, 0], [99, 0], [99, 99], [0, 99]])
            ... )
            >>> detections = sv.Detections(
            ...     xyxy=np.array([[0, 0, 50, 100], [25, 0, 75, 100]], dtype=float)
            ... )
            >>> zone.occupancy_ratio(detections)
            0.75
            >>> zone.occupancy_ratio(sv.Detections.empty())
            0.0

            ```
        """
        _validate_shape(np.shape(self.mask), ("H", "W"), "zone mask")
        zone = self.mask.astype(bool)
        zone_pixels = int(np.count_nonzero(zone))
        # Checked before the empty-detections shortcut so that frames without
        # detections cannot hide a zone that has no pixels to cover.
        if zone_pixels == 0:
            raise ValueError(
                "Zone mask covers no pixels, so its occupancy ratio is undefined. "
                "Use a polygon that encloses at least one pixel at non-negative "
                f"coordinates; got a zone mask of shape {zone.shape} with no "
                "nonzero pixels."
            )
        if len(detections) == 0:
            return 0.0
        mask_h, mask_w = zone.shape
        occupied = _detections_to_union_mask(detections, resolution_wh=(mask_w, mask_h))
        return int(np.count_nonzero(occupied & zone)) / zone_pixels


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
        self,
        scene: npt.NDArray[Any],
        label: str | None = None,
        coord_transform: CoordinatesTransformation | None = None,
    ) -> npt.NDArray[Any]:
        """Annotates the polygon zone within a frame with a count of detected objects.

        Args:
            scene: The image on which the polygon zone will be annotated
            label: A label for the count of detected objects
                within the polygon zone (default: None)
            coord_transform: Optional per-frame camera motion, the same one
                passed to
                :meth:`~supervision.detection.tools.polygon_zone.PolygonZone.trigger`.
                The zone's vertices, and its count label when shown, are mapped
                into the current frame with
                :meth:`~supervision.geometry.core.CoordinatesTransformation.abs_to_rel`
                and rounded to the nearest pixel, where drawing without a transform
                truncates. If any of them maps to a non-finite coordinate or one
                beyond `2**20` pixels, the zone is not drawn on this frame.

        Returns:
            The image with the polygon zone and count of detected objects

        Raises:
            ValueError: If `coord_transform` does not return one point per input
                point.

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
            >>> zone_annotator = sv.PolygonZoneAnnotator(
            ...     zone=zone, display_in_zone_count=False
            ... )
            >>> scene = np.zeros((120, 200, 3), dtype=np.uint8)
            >>> annotated_scene = zone_annotator.annotate(
            ...     scene=scene, coord_transform=CameraShift()
            ... )
            >>> annotated_scene[50, 150].tolist()
            [255, 255, 255]
            >>> annotated_scene[50, 100].tolist()
            [0, 0, 0]

            ```
        """
        polygon = self.zone.polygon
        center = self.center
        if coord_transform is not None:
            reference_points = polygon
            if self.display_in_zone_count:
                reference_points = np.vstack([polygon, [[center.x, center.y]]])
            mapped = _map_to_drawing_coordinates(reference_points, coord_transform)
            if mapped is None:
                # Alias the result like the drawn path: a copy when filling.
                return scene if self.opacity == 0 else scene.copy()
            mapped_pixels = np.rint(mapped).astype(np.int64)
            polygon = mapped_pixels[: len(self.zone.polygon)]
            if self.display_in_zone_count:
                center_x, center_y = mapped_pixels[-1].tolist()
                center = Point(x=center_x, y=center_y)

        if self.opacity == 0:
            annotated_frame = draw_polygon(
                scene=scene,
                polygon=polygon,
                color=self.color,
                thickness=self.thickness,
            )
        else:
            annotated_frame = draw_filled_polygon(
                scene=scene.copy(),
                polygon=polygon,
                color=self.color,
                opacity=self.opacity,
            )
            annotated_frame = draw_polygon(
                scene=annotated_frame,
                polygon=polygon,
                color=self.color,
                thickness=self.thickness,
            )

        if self.display_in_zone_count:
            annotated_frame = draw_text(
                scene=annotated_frame,
                text=str(self.zone.current_count) if label is None else label,
                text_anchor=center,
                background_color=self.color,
                text_color=self.text_color,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                text_padding=self.text_padding,
                text_font=self.font,
            )

        return cast(npt.NDArray[Any], annotated_frame)
