from abc import ABC, abstractmethod

from supervision.detection.core import Detections
from supervision.draw.base import ImageType


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        pass
