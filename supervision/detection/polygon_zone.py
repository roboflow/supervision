from dataclasses import replace
from typing import Optional, Tuple

import cv2
import numpy as np

from supervision import Detections
from supervision.detection.utils import clip_boxes, generate_2d_mask
from supervision.draw.color import Color
from supervision.draw.utils import draw_polygon, draw_text
from supervision.geometry.core import Position
from supervision.geometry.utils import get_polygon_center


class PolygonZone:
    def __init__(
        self,
        polygon: np.ndarray,
        frame_resolution_wh: Tuple[int, int],
        triggering_position: Position = Position.BOTTOM_CENTER,
    ):
        self.polygon = polygon
        self.frame_resolution_wh = frame_resolution_wh
        self.triggering_position = triggering_position
        self.current_count = 0

        width, height = frame_resolution_wh
        self.mask = generate_2d_mask(
            polygon=polygon, resolution_wh=(width + 1, height + 1)
        )

    def trigger(self, detections: Detections) -> np.ndarray:
        clipped_xyxy = clip_boxes(
            boxes_xyxy=detections.xyxy, frame_resolution_wh=self.frame_resolution_wh
        )
        clipped_detections = replace(detections, xyxy=clipped_xyxy)
        clipped_anchors = np.ceil(
            clipped_detections.get_anchor_coordinates(anchor=self.triggering_position)
        ).astype(int)
        is_in_zone = self.mask[clipped_anchors[:, 1], clipped_anchors[:, 0]]
        self.current_count = np.sum(is_in_zone)
        return is_in_zone.astype(bool)


class PolygonZoneAnnotator:
    def __init__(
        self,
        zone: PolygonZone,
        color: Color,
        thickness: int = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
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

    def annotate(self, scene: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        annotated_frame = draw_polygon(
            scene=scene,
            polygon=self.zone.polygon,
            color=self.color,
            thickness=self.thickness,
        )

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
