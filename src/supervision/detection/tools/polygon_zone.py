from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision import Detections
from supervision import _cv2 as cv2
from supervision.detection.utils.converters import (
    MIN_POLYGON_POINT_COUNT,
    polygon_to_mask,
)
from supervision.draw.color import Color
from supervision.draw.utils import draw_filled_polygon, draw_polygon, draw_text
from supervision.geometry.core import (
    CoordinatesTransform,
    Position,
    _transform_points,
)
from supervision.geometry.utils import _clip_polygon_to_box, get_polygon_center


class PolygonZone:
    """A class for defining a polygon-shaped zone within a frame for detecting objects.

    !!! warning

        PolygonZone uses the `tracker_id`. Read
        [here](https://trackers.roboflow.com/latest/) to learn how to plug
        tracking into your inference pipeline.

    Attributes:
        polygon: A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        triggering_anchors: Any iterable of positions specifying
            which anchors of the detections bounding box to consider when deciding on
            whether the detection fits within the PolygonZone
            (default: (sv.Position.BOTTOM_CENTER,)).
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
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
        )

    def trigger(
        self,
        detections: Detections,
        coord_transform: CoordinatesTransform | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Determines if the detections are within the polygon zone.

        Anchor points are calculated from original (unclipped) detection boxes to
        avoid per-zone clipping shifting anchor positions. This prevents a single
        detection from being counted in multiple non-overlapping zones due to
        clipping artifacts, although overlapping zones may still legitimately
        contain the same detection.

        Args:
            detections: The detections to be checked against the polygon zone
            coord_transform: Optional per-frame camera motion, such as
                `sv.MatrixTransform`; anchors are mapped into the zone's reference
                frame with `rel_to_abs`, and ones mapped out of its bounds or to
                non-finite values are outside. See the "Follow a Moving Camera" how-to.

        Returns:
            A boolean numpy array indicating
                if each detection is within the polygon zone

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            ... )
            >>> camera_moved = sv.MatrixTransform(np.array([[1, 0, 50], [0, 1, 0]]))
            >>> detections = sv.Detections(xyxy=np.array([[120.0, 40.0, 140.0, 60.0]]))
            >>> zone.trigger(detections)
            array([False])
            >>> zone.trigger(detections, coord_transform=camera_moved)
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
            # Non-finite or far anchors land just outside the mask, so they fail
            # the bounds check below instead of overflowing the integer cast.
            mapped = np.nan_to_num(mapped, nan=-1.0, posinf=-1.0, neginf=-1.0)
            anchor_coordinates = np.clip(mapped, -1.0, float(max(self.mask.shape)))
        all_anchors = np.rint(anchor_coordinates).astype(int)

        mask_h, mask_w = self.mask.shape
        x, y = all_anchors[:, :, 0], all_anchors[:, :, 1]
        in_bounds = (x >= 0) & (y >= 0) & (x < mask_w) & (y < mask_h)
        x_safe = np.clip(x, 0, mask_w - 1)
        y_safe = np.clip(y, 0, mask_h - 1)
        anchor_hits = in_bounds & self.mask[y_safe, x_safe]
        reduce = np.all if self.require_all_anchors else np.any
        is_in_zone = reduce(anchor_hits, axis=0)
        self.current_count = int(np.sum(is_in_zone))
        return cast(npt.NDArray[np.bool_], is_in_zone.astype(bool))


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
        coord_transform: CoordinatesTransform | None = None,
    ) -> npt.NDArray[Any]:
        """Annotates the polygon zone within a frame with a count of detected objects.

        Args:
            scene: The image on which the polygon zone will be annotated
            label: A label for the count of detected objects
                within the polygon zone (default: None)
            coord_transform: The transform passed to `PolygonZone.trigger`, such as
                `sv.MatrixTransform`; the zone and label are drawn at `abs_to_rel` of
                its vertices, clipped to 4x the scene size, and nothing is drawn if
                a vertex is non-finite. See the "Follow a Moving Camera" how-to.

        Returns:
            The image with the polygon zone and count of detected objects

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> zone = sv.PolygonZone(
            ...     polygon=np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
            ... )
            >>> zone_annotator = sv.PolygonZoneAnnotator(
            ...     zone=zone, display_in_zone_count=False
            ... )
            >>> camera_moved = sv.MatrixTransform(np.array([[1, 0, 40], [0, 1, 0]]))
            >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> scene = zone_annotator.annotate(scene, coord_transform=camera_moved)
            >>> bool(scene[30, 90].any()), bool(scene[30, 10].any())
            (True, False)

            ```
        """
        polygon = self.zone.polygon
        center = self.center
        if coord_transform is not None:
            mapped_polygon = _transform_points(polygon, coord_transform.abs_to_rel)
            # Clipping the polygon, not each vertex, keeps far edges on their lines.
            clipped_polygon = _clip_polygon_to_box(
                mapped_polygon, limit=4 * max(scene.shape[:2])
            )
            if len(clipped_polygon) == 0:
                return scene
            polygon = np.rint(clipped_polygon).astype(int)
            center = get_polygon_center(polygon=polygon)

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
