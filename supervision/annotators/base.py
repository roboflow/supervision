from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from PIL import Image

from supervision.detection.core import Detections

ImageType = TypeVar("ImageType", np.ndarray, Image.Image)


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        pass
