from abc import ABC, abstractmethod

import numpy as np

from supervision.detection.core import Detections


class BaseAnnotator(ABC):
    """
    This is the abstract base class for annotators.
    """
    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        pass
