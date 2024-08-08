from abc import ABC, abstractmethod
from __future__ import annotations
from enum import Enum
from typing import Any


class Metric(ABC):
    """
    The base class for all supervision metrics.
    """

    @abstractmethod
    def update(self, *args, **kwargs) -> Metric:
        """
        Add data to the metric, without computing the result.
        Return the metric itself to allow method chaining, for example:

        Example:
            ```python
            result = metric.update(data).compute()
            ```
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
    
    @abstractmethod
    def to_pandas(self, *args, **kwargs) -> Any:
        """
        Return a pandas DataFrame representation of the metric.
        """
        self._ensure_pandas_installed()
        raise NotImplementedError

    def _ensure_pandas_installed(self):
        try:
            import pandas
        except ImportError:
            raise ImportError(
                "Function `to_pandas` requires the `metrics` extra to be installed."
                " Run `pip install 'supervision[metrics]'` or `poetry add supervision -E metrics`.")

class MetricTarget(Enum):
    """
    Specifies what type of detection is used to compute the metric.
    """

    BOXES = "boxes"
    MASKS = "masks"
    ORIENTED_BOUNDING_BOXES = "obb"


class UnsupportedMetricTargetError(Exception):
    """
    Raised when a metric does not support the specified target. (and never will!)
    If the support might be added in the future, raise `NotImplementedError` instead.
    """
    def __init__(self, metric: Metric, target: MetricTarget):
        super().__init__(f"Metric {metric} does not support target {target}")