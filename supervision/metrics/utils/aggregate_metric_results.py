from typing import TYPE_CHECKING, List

import numpy as np
from matplotlib import pyplot as plt

from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import MetricResult
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd


def aggregate_metric_results(
    metrics_results: List[MetricResult],
    model_names: List[str],
    include_object_sizes=False,
) -> "pd.DataFrame":
    """
    Convert a list of results to a pandas DataFrame.

    Args:
        metrics_results (List[MetricResult]): List of results to be aggregated.
        model_names (List[str]): List of model names corresponding to the results.
        include_object_sizes (bool, optional): Whether to include object sizes in the
            DataFrame. Defaults to False.

    Raises:
        ValueError: List `metrics_results` can not be empty
        ValueError: All elements of `metrics_results` must be of the same type
        ValueError: Base class of elements in `metrics_results` must be of type
            `MetricResult`

    Returns:
        pd.DataFrame: The results as a DataFrame.
    """
    ensure_pandas_installed()
    import pandas as pd

    assert len(metrics_results) == len(
        model_names
    ), "Length of metrics_results and model_names must be equal"

    if len(metrics_results) == 0:
        raise ValueError("List metrics_results must not be empty")

    first_elem_type = type(metrics_results[0])
    all_same_type = all(isinstance(x, first_elem_type) for x in metrics_results)
    if not all_same_type:
        raise ValueError("All metrics_results elements must be of the same type")

    if not isinstance(metrics_results[0], MetricResult):
        raise ValueError("Base class of metrics_results must be of type MetricResult")

    pd_results = []
    for metric_result, model_name in zip(metrics_results, model_names):
        pd_result = metric_result.to_pandas()
        pd_result.insert(loc=0, column="Model Name", value=model_name)
        pd_results.append(pd_result)

    df_merged = pd.concat(pd_results)

    if not include_object_sizes:
        regex_pattern = "small|medium|large"
        df_merged = df_merged.drop(columns=list(df_merged.filter(regex=regex_pattern)))

    return df_merged


def plot_aggregate_metric_results(
    metrics_results: List[MetricResult],
    model_names: List[str],
    include_object_sizes=False,
):
    """
    Plot a bar chart with the results of multiple metrics.

    Args:
        metrics_results (List[MetricResult]): List of results to be plotted.
        model_names (List[str]): List of model names corresponding to the results.
        include_object_sizes (bool, optional): Whether to include object sizes in the
            plot. Defaults to False.

    Raises:
        ValueError: List `metrics_results` can not be empty
        ValueError: All elements of `metrics_results` must be of the same type
        ValueError: Base class of elements in `metrics_results` must be of type
            `MetricResult`
    """
    assert len(metrics_results) == len(
        model_names
    ), "Length of metrics_results and model_names must be equal"

    if len(metrics_results) == 0:
        raise ValueError("List metrics_results must not be empty")

    first_elem_type = type(metrics_results[0])
    all_same_type = all(isinstance(x, first_elem_type) for x in metrics_results)
    if not all_same_type:
        raise ValueError("All metrics_results elements must be of the same type")

    if not isinstance(metrics_results[0], MetricResult):
        raise ValueError("Base class of metrics_results must be of type MetricResult")

    model_values = []
    labels, values, title, _ = metrics_results[0]._get_plot_details()
    model_values.append(values)

    for metric in metrics_results[1:]:
        _, values, _, _ = metric._get_plot_details()
        model_values.append(values)

    if not include_object_sizes:
        labels_length = 3 if len(labels) % 3 == 0 else 2
        labels = labels[:labels_length]
        aux_values = []
        for values in model_values:
            aux_values.append(values[:labels_length])
        model_values = aux_values

    n = len(model_names)
    x_positions = np.arange(len(labels))
    width = 0.8 / n
    value_text_rotation = 90 if include_object_sizes else 0

    plt.rcParams["font.family"] = "monospace"

    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    colors = LEGACY_COLOR_PALETTE[:n]

    for i, model_value in enumerate(model_values):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(
            x_positions + offset,
            model_value,
            width=width,
            label=model_names[i],
            color=colors[i % len(colors)],
        )

        for bar in bars:
            y_value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_value + 0.02,
                f"{y_value:.2f}",
                ha="center",
                va="bottom",
                rotation=value_text_rotation,
            )

    plt.rcParams["font.family"] = "sans-serif"

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
