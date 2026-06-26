from abc import ABC, abstractmethod
from typing import Any

from supervision.detection.core import Detections


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(
        self, scene: Any, detections: Detections, *args: Any, **kwargs: Any
    ) -> Any:
        pass
