from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import MetricResult
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd


def aggregate_metric_results(
    metric_results: list[MetricResult],
    *,
    model_names: list[str] | None = None,
    include_object_sizes: bool = False,
) -> pd.DataFrame:
    """Combine several :class:`MetricResult` objects into a single DataFrame.

    Each row corresponds to one result (one model). All results must be of the
    same concrete type (e.g. all :class:`F1ScoreResult`).

    Args:
        metric_results: A list of metric results to aggregate.
        model_names: Optional display names for each result. When provided,
            the DataFrame index is set to these names. Must have the same
            length as *metric_results*.
        include_object_sizes: When ``True``, include columns for
            small / medium / large object-size categories.

    Returns:
        A :class:`~pandas.DataFrame` with one row per result and columns for
        each metric value.

    Raises:
        ValueError: If the list is empty, contains mixed result types, or
            *model_names* length does not match *metric_results*.
    """
    if not metric_results:
        raise ValueError("metric_results must not be empty.")

    ensure_pandas_installed()
    import pandas as pd

    result_type = type(metric_results[0])
    for result in metric_results[1:]:
        if type(result) is not result_type:
            raise TypeError(
                f"All metric results must be the same type. "
                f"Expected {result_type.__name__}, "
                f"got {type(result).__name__}."
            )

    if model_names is not None and len(model_names) != len(metric_results):
        raise ValueError(
            f"model_names length ({len(model_names)}) must match "
            f"metric_results length ({len(metric_results)})."
        )

    frames = [result.to_pandas() for result in metric_results]
    df = pd.concat(frames, ignore_index=True)

    if not include_object_sizes:
        size_prefixes = ("small_objects_", "medium_objects_", "large_objects_")
        cols_to_drop = [col for col in df.columns if col.startswith(size_prefixes)]
        df = df.drop(columns=cols_to_drop)

    if model_names is not None:
        df.index = pd.Index(model_names)

    return df


def plot_aggregate_metric_results(
    metric_results: list[MetricResult],
    *,
    model_names: list[str] | None = None,
    include_object_sizes: bool = False,
) -> None:
    """Plot multiple :class:`MetricResult` objects on a single grouped bar chart.

    Each group of bars corresponds to a metric label (e.g. ``"F1@50"``), and
    each bar within the group corresponds to one model.

    Args:
        metric_results: A list of metric results to plot.
        model_names: Optional display names for each result (used in the
            legend). When ``None``, results are labelled ``"Model 1"``,
            ``"Model 2"``, etc.
        include_object_sizes: When ``True``, include bars for
            small / medium / large object-size categories.

    Raises:
        ValueError: If the list is empty, contains mixed result types, or
            *model_names* length does not match *metric_results*.
    """
    from matplotlib import pyplot as plt

    if not metric_results:
        raise ValueError("metric_results must not be empty.")

    result_type = type(metric_results[0])
    for result in metric_results[1:]:
        if type(result) is not result_type:
            raise TypeError(
                f"All metric results must be the same type. "
                f"Expected {result_type.__name__}, "
                f"got {type(result).__name__}."
            )

    if model_names is not None and len(model_names) != len(metric_results):
        raise ValueError(
            f"model_names length ({len(model_names)}) must match "
            f"metric_results length ({len(metric_results)})."
        )

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(metric_results))]

    all_details = [
        result._get_plot_details(include_object_sizes=include_object_sizes)
        for result in metric_results
    ]

    labels = all_details[0].labels
    title = all_details[0].title
    num_models = len(metric_results)
    num_labels = len(labels)

    x = np.arange(num_labels)
    bar_width = 0.8 / num_models

    plt.rcParams["font.family"] = "monospace"
    _, ax = plt.subplots(figsize=(max(10, num_labels * 1.5), 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    for model_idx, (name, details) in enumerate(zip(model_names, all_details)):
        offset = (model_idx - num_models / 2 + 0.5) * bar_width
        color = LEGACY_COLOR_PALETTE[model_idx % len(LEGACY_COLOR_PALETTE)]
        bars = ax.bar(
            x + offset,
            details.values,
            bar_width,
            label=name,
            color=color,
        )
        for bar in bars:
            y_value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_value + 0.02,
                f"{y_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=max(6, 8 - num_models),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    plt.rcParams["font.family"] = "sans-serif"

    plt.tight_layout()
    plt.show()
