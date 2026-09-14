from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from supervision.draw.color import LEGACY_COLOR_PALETTE

if TYPE_CHECKING:
    import pandas as pd

R = TypeVar("R")
T = TypeVar("T")


def _append_object_size_plot_details(
    labels: list[str],
    values: list[float],
    colors: list[str],
    *,
    include_object_sizes: bool,
    metric_labels: list[str],
    small_objects: T | None,
    medium_objects: T | None,
    large_objects: T | None,
    value_getter: Callable[[T], list[float]],
) -> None:
    """Append available object-size bars using the shared category order and colors."""
    if not include_object_sizes:
        return

    for name, palette_index, object_sizes in (
        ("Small", 3, small_objects),
        ("Medium", 2, medium_objects),
        ("Large", 4, large_objects),
    ):
        if object_sizes is None:
            continue
        labels.extend(f"{name}: {metric_label}" for metric_label in metric_labels)
        values.extend(value_getter(object_sizes))
        colors.extend([LEGACY_COLOR_PALETTE[palette_index]] * len(metric_labels))


@dataclass
class PlotDetails:
    """Container for bar-chart data returned by ``MetricResult._get_plot_details``.

    Attributes:
        labels: Bar labels (x-axis tick labels).
        values: Bar heights (metric values).
        colors: One hex color string per bar (e.g. ``"#A351FB"``).
        title: Chart title.
    """

    labels: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    title: str = ""


class MetricResult(ABC):
    """Abstract base class shared by all metric result dataclasses."""

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """Convert the result to a :class:`~pandas.DataFrame`."""
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> None:
        """Render a bar-chart of the result."""
        raise NotImplementedError

    @abstractmethod
    def _get_plot_details(self, include_object_sizes: bool = True) -> PlotDetails:
        """Return labels, values, colors, and title for a bar chart.

        Args:
            include_object_sizes: When ``True`` (default), include bars for
                small / medium / large object-size categories.
        """
        raise NotImplementedError


class Metric(ABC, Generic[R]):
    """The base class for all supervision metrics."""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Metric[R]:
        """Add data to the metric, without computing the result.

        Return the metric itself to allow method chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal metric state."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> R:
        """Compute the metric from the internal state and return the result."""
        raise NotImplementedError


class MetricTarget(Enum):
    """Specifies what type of detection is used to compute the metric.

    Attributes:
        BOXES: xyxy bounding boxes
        MASKS: Binary masks
        ORIENTED_BOUNDING_BOXES: Oriented bounding boxes (OBB)
    """

    BOXES = "boxes"
    MASKS = "masks"
    ORIENTED_BOUNDING_BOXES = "obb"


class AveragingMethod(Enum):
    """Defines different ways of averaging the metric results.

    Suppose, before returning the final result, a metric is computed for each class.
    How do you combine those to get the final number?

    Attributes:
        MACRO: Calculate the metric for each class and average the results. The simplest
            averaging method, but it does not take class imbalance into account.
        MICRO: Calculate the metric globally by counting the total true positives, false
            positives, and false negatives. Micro averaging is useful when you want to
            give more importance to classes with more samples. It's also more
            appropriate if you have an imbalance in the number of instances per class.
        WEIGHTED: Calculate the metric for each class and average the results, weighted
            by the number of true instances of each class. Use weighted averaging if
            you want to take class imbalance into account.
    """

    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"
