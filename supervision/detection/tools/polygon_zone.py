from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import cv2
import numpy as np
import numpy.typing as npt

from supervision import Detections
from supervision.detection.utils.boxes import clip_boxes
from supervision.detection.utils.converters import polygon_to_mask
from supervision.draw.color import Color
from supervision.draw.utils import draw_filled_polygon, draw_polygon, draw_text
from supervision.geometry.core import Position
from supervision.geometry.utils import get_polygon_center


class PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    !!! warning

        PolygonZone uses the `tracker_id`. Read
        [here](/latest/trackers/) to learn how to plug
        tracking into your inference pipeline.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        triggering_anchors (Iterable[sv.Position]): A list of positions specifying
            which anchors of the detections bounding box to consider when deciding on
            whether the detection fits within the PolygonZone
            (default: (sv.Position.BOTTOM_CENTER,)).
        current_count (int): The current count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone

    Example:
        ```python
        import supervision as sv
        from ultralytics import YOLO
        import numpy as np
        import cv2

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = YOLO("yolo11s")
        tracker = sv.ByteTrack()

        polygon = np.array([[100, 200], [200, 100], [300, 200], [200, 300]])
        polygon_zone = sv.PolygonZone(polygon=polygon)

        result = model.infer(image)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        is_detections_in_zone = polygon_zone.trigger(detections)
        print(polygon_zone.current_count)
        ```
    """

    def __init__(
        self,
        polygon: npt.NDArray[np.int64],
        triggering_anchors: Iterable[Position] = (Position.BOTTOM_CENTER,),
    ):
        self.polygon = polygon.astype(int)
        self.triggering_anchors = triggering_anchors
        if not list(self.triggering_anchors):
            raise ValueError("Triggering anchors cannot be empty.")

        self.current_count = 0

        x_max, y_max = np.max(polygon, axis=0)
        self.frame_resolution_wh = (x_max + 1, y_max + 1)
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
        )

    def trigger(self, detections: Detections) -> npt.NDArray[np.bool_]:
        """
    Determines if the detections are within the polygon zone.

    Anchor points are calculated from original (unclipped) detection boxes.
    to ensure a single object can only appear in one zone. This prevents 
    detections spanning multiple non-overlapping ROIs from being counted
    in multiple zones simulataneously.

    Parameters:
        detections (Detections): The detections
            to be checked against the polygon zone

    Returns:
        np.ndarray: A boolean numpy array indicating
            if each detection is within the polygon zone
    """
        if len(detections) == 0:
            return np.array([], dtype=bool)
        

        all_anchors = np.array([
            np.ceil(detections.get_anchors_coordinates(anchors)).astype(int)
            for anchors in self.triggering_anchors
        ])

        is_in_zone: npt.NDArray[np.bool_] = np.zeros(len(detections), dtype=bool)

        for detection_idx in range(len(detections)):
            anchors = all_anchors[:, detection_idx, :]
            all_in_zone = True

            for anchor in anchors:
                x, y = anchor

                if x < 0 or y < 0 or x >= self.mask.shape[1] or y >= self.mask.shape[0]:
                    all_in_zone = False
                    break

                if not self.mask[y, x]:
                    all_in_zone = False
                    break

            is_in_zone[detection_idx] = all_in_zone
            
        self.current_count = int(np.sum(is_in_zone))
        return is_in_zone.astype(bool)



class PolygonZoneAnnotator:
    """
    A class for annotating a polygon-shaped zone within a
        frame with a count of detected objects.

    Attributes:
        zone (PolygonZone): The polygon zone to be annotated
        color (Color): The color to draw the polygon lines, default is white
        thickness (int): The thickness of the polygon lines, default is 2
        text_color (Color): The color of the text on the polygon, default is black
        text_scale (float): The scale of the text on the polygon, default is 0.5
        text_thickness (int): The thickness of the text on the polygon, default is 1
        text_padding (int): The padding around the text on the polygon, default is 10
        font (int): The font type for the text on the polygon,
            default is cv2.FONT_HERSHEY_SIMPLEX
        center (Tuple[int, int]): The center of the polygon for text placement
        display_in_zone_count (bool): Show the label of the zone or not. Default is True
        opacity: The opacity of zone filling when drawn on the scene. Default is 0
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
    ):
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

    def annotate(self, scene: np.ndarray, label: str | None = None) -> np.ndarray:
        """
        Annotates the polygon zone within a frame with a count of detected objects.

        Parameters:
            scene (np.ndarray): The image on which the polygon zone will be annotated
            label (Optional[str]): A label for the count of detected objects
                within the polygon zone (default: None)

        Returns:
            np.ndarray: The image with the polygon zone and count of detected objects
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

        return annotated_frame
