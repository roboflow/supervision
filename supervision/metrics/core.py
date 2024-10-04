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


class AveragingMethod(Enum):
    """
    Defines different ways of averaging the metric results.

    Suppose, before returning the final result, a metric is computed for each class.
    How do you combine those to get the final number?

    * MACRO: Calculate the metric for each class and average the results. The simplest
        averaging method, but it does not take class imbalance into account.
    * MICRO: Calculate the metric globally by counting the total true positives, false
        positives, and false negatives. Micro averaging is useful when you want to give
        more importance to classes with more samples. It's also more appropriate if you
        have an imbalance in the number of instances per class.
    * WEIGHTED: Calculate the metric for each class and average the results, weighted by
        the number of true instances of each class. Use weighted averaging if you want
        to take class imbalance into account.
    """

    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"
