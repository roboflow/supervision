from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

CLASS_ID_NONE = -1
"""Used by metrics module as class ID, when none is present"""


class Metric(ABC):
    """
    The base class for all supervision metrics.
    """

    @abstractmethod
    def update(self, *args, **kwargs) -> "Metric":
        """
        Add data to the metric, without computing the result.
        Return the metric itself to allow method chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal metric state.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """
        Compute the metric from the internal state and return the result.
        """
        raise NotImplementedError


class MetricTarget(Enum):
    """
    Specifies what type of detection is used to compute the metric.

    * BOXES: xyxy bounding boxes
    * MASKS: Binary masks
    * ORIENTED_BOUNDING_BOXES: Oriented bounding boxes (OBB)
    """

    BOXES = "boxes"
    MASKS = "masks"
    ORIENTED_BOUNDING_BOXES = "obb"
