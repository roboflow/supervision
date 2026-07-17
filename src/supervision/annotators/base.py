from abc import ABC, abstractmethod
from typing import Any

from supervision.detection.core import Detections


class BaseAnnotator(ABC):
    """Base class for annotators that consume :class:`Detections`.

    Attributes:
        requires_mask: Whether integrations must provide ``Detections.mask`` for
            this annotator. Check this before materializing expensive mask payloads.
    """

    requires_mask: bool = False

    @abstractmethod
    def annotate(
        self, scene: Any, detections: Detections, *args: Any, **kwargs: Any
    ) -> Any:
        pass
