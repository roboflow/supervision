from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class BoxAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class MaskAnnotator(BaseAnnotator):
    """
    A class for overlaying masks on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to fill the mask, can be a single color or a color palette
        opacity (float): The opacity of the masks, between 0 and 1.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Overlays the masks on the given image based on the provided detections, with a specified opacity.

        Args:
            scene (np.ndarray): The image on which the masks will be overlaid
            detections (Detections): The detections for which the masks will be overlaid

        Returns:
            np.ndarray: The image with the masks overlaid
        """
        if detections.mask is None:
            return scene

        for i in np.flip(np.argsort(detections.area)):
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )

            mask = detections.mask[i]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()

            scene = np.where(
                np.expand_dims(mask, axis=-1),
                np.uint8(self.opacity * colored_mask + (1 - self.opacity) * scene),
                scene,
            )

        return scene


class LabelAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class PillowLabelAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class TrackAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass


class BoxMaskAnnotator(BaseAnnotator):
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass
