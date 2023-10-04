from enum import Enum
from typing import Optional, Union

import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


class ColorMap(Enum):
    """
    Enum for annotator color mapping.
    """

    INDEX = "index"
    CLASS = "class"
    TRACK = "track"


def resolve_color_idx(
    detections: Detections, detection_idx: int, color_map: ColorMap = ColorMap.CLASS
) -> int:
    if detection_idx >= len(detections):
        raise ValueError(
            f"Detection index {detection_idx}"
            f"is out of bounds for detections of length {len(detections)}"
        )

    if color_map == ColorMap.INDEX:
        return detection_idx
    elif color_map == ColorMap.CLASS:
        if detections.class_id is None:
            raise ValueError(
                "Could not resolve color by class because"
                "Detections do not have class_id"
            )
        return detections.class_id[detection_idx]
    elif color_map == ColorMap.TRACK:
        if detections.tracker_id is None:
            raise ValueError(
                "Could not resolve color by track because"
                "Detections do not have tracker_id"
            )
        return detections.tracker_id[detection_idx]


def resolve_color(color: Union[Color, ColorPalette], idx: int) -> Color:
    if isinstance(color, ColorPalette):
        return color.by_idx(idx)
    return color


class Trace:
    def __init__(
        self,
        max_size: Optional[int] = None,
        start_frame_id: int = 0,
        anchor: Position = Position.CENTER,
    ) -> None:
        self.current_frame_id = start_frame_id
        self.max_size = max_size
        self.anchor = anchor

        self.frame_id = np.array([], dtype=int)
        self.xy = np.empty((0, 2), dtype=np.float32)
        self.tracker_id = np.array([], dtype=int)

    def put(self, detections: Detections) -> None:
        frame_id = np.full(len(detections), self.current_frame_id, dtype=int)
        self.frame_id = np.concatenate([self.frame_id, frame_id])
        self.xy = np.concatenate(
            [self.xy, detections.get_anchor_coordinates(self.anchor)]
        )
        self.tracker_id = np.concatenate([self.tracker_id, detections.tracker_id])

        unique_frame_id = np.unique(self.frame_id)

        if 0 < self.max_size < len(unique_frame_id):
            max_allowed_frame_id = self.current_frame_id - self.max_size + 1
            filtering_mask = self.frame_id >= max_allowed_frame_id
            self.frame_id = self.frame_id[filtering_mask]
            self.xy = self.xy[filtering_mask]
            self.tracker_id = self.tracker_id[filtering_mask]

        self.current_frame_id += 1

    def get(self, tracker_id: int) -> np.ndarray:
        return self.xy[self.tracker_id == tracker_id]
