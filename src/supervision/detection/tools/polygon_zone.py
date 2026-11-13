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


def _paste_mask(
    canvas: npt.NDArray[np.bool_],
    patch: npt.NDArray[np.bool_],
    offset_xy: tuple[int, int],
) -> None:
    """OR a local boolean patch into ``canvas`` at ``offset_xy``, clipped to bounds.

    Args:
        canvas: Target boolean array of shape ``(H, W)``, updated in place.
        patch: Boolean array of shape ``(h, w)`` to merge into the canvas.
        offset_xy: Canvas ``(x, y)`` position of the patch's top-left pixel.
    """
    canvas_h, canvas_w = canvas.shape
    patch_h, patch_w = patch.shape
    x0, y0 = offset_xy
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + patch_w, canvas_w), min(y0 + patch_h, canvas_h)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] |= patch[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]


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
    for polygon in polygons[finite].astype(np.int32):
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
            # Union the RLE crops without decoding the full frame, then paste the
            # single bbox-local union crop at its offset, clipped to the canvas.
            union = _compact_mask_union([mask])
            offset_x, offset_y = (int(value) for value in union.offsets[0])
            _paste_mask(canvas, union.crop(0), (offset_x, offset_y))
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

        :class:`~supervision.detection.tools.polygon_zone.PolygonZone` uses the
        `tracker_id`. Read
        [here](https://trackers.roboflow.com/latest/) to learn how to plug
        tracking into your inference pipeline.

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
        current_count: The current count of detected objects within the zone
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
    """

    def __init__(
        self,
        polygon: npt.NDArray[np.int64],
        triggering_anchors: Iterable[Position] = (Position.BOTTOM_CENTER,),
        require_all_anchors: bool = True,
    ) -> None:
        """Build a zone from a polygon.

        Args:
            polygon: Zone boundary of shape `(N, 2)` holding the `x`, `y`
                coordinates of its vertices.
            triggering_anchors: Which anchors of a detection's bounding box
                decide whether it falls inside the zone.
            require_all_anchors: Whether every triggering anchor must be inside
                for the detection to count, rather than any one of them.

        Raises:
            ValueError: If `polygon` is not of shape `(N, 2)`, has fewer than
                `MIN_POLYGON_POINT_COUNT` vertices, or `triggering_anchors` is
                empty.
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

        self.current_count = 0

        x_max, y_max = np.max(polygon, axis=0)
        if np.issubdtype(polygon.dtype, np.integer):
            x_max, y_max = int(x_max), int(y_max)
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
        )

    def trigger(
        self,
        detections: Detections,
        coord_transform: CoordinatesTransformation | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Determines if the detections are within the polygon zone.

        Anchor points are calculated from original (unclipped) detection boxes to
        avoid per-zone clipping shifting anchor positions. This prevents a single
        detection from being counted in multiple non-overlapping zones due to
        clipping artifacts, although overlapping zones may still legitimately
        contain the same detection. Negative and non-finite anchors are outside.

        Args:
            detections: The detections to be checked against the polygon zone
            coord_transform: Optional per-frame camera motion. Anchors are mapped
                into the zone's reference frame with
                :meth:`~supervision.geometry.core.CoordinatesTransformation.rel_to_abs`
                before the inside test; anchors mapped outside the zone's bounds
                or to non-finite values are outside.

        Returns:
            A boolean numpy array indicating
                if each detection is within the polygon zone

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
        if len(detections) == 0:
            self.current_count = 0
            return cast(npt.NDArray[np.bool_], np.array([], dtype=bool))

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
        self.current_count = int(np.sum(is_in_zone))
        return cast(npt.NDArray[np.bool_], is_in_zone.astype(bool))

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
                shape `(N, K, 2)`, with `N` detections.

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
