from typing import Union

import cv2
import numpy as np

from supervision import Color, ColorPalette, Detections, Position
from supervision.annotators.base import BaseAnnotator


class DotMarkerAnnotator(BaseAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        radius: int = 4,
        position: Position = Position.CENTER,
        color_map: str = "class",
    ):
        self.color = color
        self.radius = radius
        self.position = position
        self.color_map = color_map

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        annotated_scene = scene.copy()

        for detection in detections:
            dot_color = self._get_color(detection)
            box_position = detection.bounding_box.get_position(self.position)
            dot_position = (int(box_position[0]), int(box_position[1]))
            cv2.circle(annotated_scene, dot_position, self.radius, dot_color, -1)

        return annotated_scene

    def _get_color(self, detection):
        if self.color_map == "class":
            return self.color[detection.class_id]
        elif self.color_map == "index":
            return self.color[detection.index % len(self.color)]
        elif self.color_map == "track":
            return self.color[detection.track_id % len(self.color)]
        else:
            raise ValueError("Invalid color mapping strategy")
