import warnings
from dataclasses import replace
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from supervision import Detections
from supervision.detection.utils import clip_boxes, polygon_to_mask
from supervision.draw.color import Color
from supervision.draw.utils import draw_polygon, draw_text
from supervision.geometry.core import Position
from supervision.geometry.utils import get_polygon_center
from supervision.utils.internal import SupervisionWarnings


class PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        triggering_anchors (Iterable[sv.Position]): A list of positions specifying
            which anchors of the detections bounding box to consider when deciding on
            whether the detection fits within the PolygonZone
            (default: (sv.Position.BOTTOM_CENTER,)).
        current_count (int): The current count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone
    """

    def __init__(
        self,
        polygon: npt.NDArray[np.int64],
        frame_resolution_wh: Optional[Tuple[int, int]] = None,
        triggering_anchors: Iterable[Position] = (Position.BOTTOM_CENTER,),
    ):
        if frame_resolution_wh is not None:
            warnings.warn(
                "The `frame_resolution_wh` parameter is no longer required and will be "
                "dropped in version supervision-0.24.0. The mask resolution is now "
                "calculated automatically based on the polygon coordinates.",
                category=SupervisionWarnings,
            )

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

        Parameters:
            detections (Detections): The detections
                to be checked against the polygon zone

        Returns:
            np.ndarray: A boolean numpy array indicating
                if each detection is within the polygon zone
        """

        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=self.frame_resolution_wh
        )
        clipped_detections = replace(detections, xyxy=clipped_xyxy)
        all_clipped_anchors = np.array(
            [
                np.ceil(clipped_detections.get_anchors_coordinates(anchor)).astype(int)
                for anchor in self.triggering_anchors
            ]
        )

        is_in_zone: npt.NDArray[np.bool_] = (
            self.mask[all_clipped_anchors[:, :, 1], all_clipped_anchors[:, :, 0]]
            .transpose()
            .astype(bool)
        )

        is_in_zone: npt.NDArray[np.bool_] = np.all(is_in_zone, axis=1)
        self.current_count = int(np.sum(is_in_zone))
        return is_in_zone.astype(bool)


class PolygonZoneAnnotator:
    """
    A class for annotating a polygon-shaped zone within a
        frame with a count of detected objects.

    Attributes:
        zone (PolygonZone): The polygon zone to be annotated
        color (Color): The color to draw the polygon lines
        thickness (int): The thickness of the polygon lines, default is 2
        text_color (Color): The color of the text on the polygon, default is black
        text_scale (float): The scale of the text on the polygon, default is 0.5
        text_thickness (int): The thickness of the text on the polygon, default is 1
        text_padding (int): The padding around the text on the polygon, default is 10
        font (int): The font type for the text on the polygon,
            default is cv2.FONT_HERSHEY_SIMPLEX
        center (Tuple[int, int]): The center of the polygon for text placement
        display_in_zone_count (bool): Show the label of the zone or not. Default is True
    """

    def __init__(
        self,
        zone: PolygonZone,
        color: Color,
        thickness: int = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        display_in_zone_count: bool = True,
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

    def annotate(self, scene: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        """
        Annotates the polygon zone within a frame with a count of detected objects.

        Parameters:
            scene (np.ndarray): The image on which the polygon zone will be annotated
            label (Optional[str]): A label for the count of detected objects
                within the polygon zone (default: None)

        Returns:
            np.ndarray: The image with the polygon zone and count of detected objects
        """
        annotated_frame = draw_polygon(
            scene=scene,
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
