from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette


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
    else:
        return color


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass
