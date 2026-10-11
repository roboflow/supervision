from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.utils.utils import ensure_pandas_installed

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


def _mean_valid_score(scores: npt.NDArray[np.float64]) -> float:
    """Average the scores that are not the `-1` sentinel, or return `-1`."""
    valid_scores = scores[scores > -1]
    if len(valid_scores) > 0:
        return float(valid_scores.mean())
    return -1


def _scores_to_pandas(
    scores: dict[str, float],
    object_sizes: list[tuple[str, MetricResult | None]],
) -> pd.DataFrame:
    """Build a one-row DataFrame of scores and prefixed per-size scores.

    Args:
        scores: Column name to score for the overall result.
        object_sizes: `(prefix, result)` pairs; each present result's own
            `to_pandas` columns are added as `{prefix}_{column}`.

    Returns:
        A DataFrame with a single row.
    """
    ensure_pandas_installed()
    import pandas as pd

    pandas_data: dict[str, object] = dict(scores)
    for prefix, result in object_sizes:
        if result is None:
            continue
        for key, value in result.to_pandas().items():
            pandas_data[f"{prefix}_{key}"] = value
    return pd.DataFrame(pandas_data, index=[0])


def _show_bar_plot(details: PlotDetails) -> None:
    """Draw score bars with their values on a `[0, 1]` axis and show them."""
    from matplotlib import pyplot as plt

    plt.rcParams["font.family"] = "monospace"

    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(details.title, fontweight="bold")

    x_positions = range(len(details.labels))
    bars = ax.bar(x_positions, details.values, color=details.colors, align="center")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(details.labels, rotation=45, ha="right")

    for bar in bars:
        y_value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_value + 0.02,
            f"{y_value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.rcParams["font.family"] = "sans-serif"

    plt.tight_layout()
    plt.show()


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
