from typing import TYPE_CHECKING, List

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
    Raises when different types of metrics results are passed in
    """
    ensure_pandas_installed()
    import pandas as pd

    assert len(metrics_results) == len(
        model_names
    ), "Number of metrics results and model names must be equal"

    if len(metrics_results) == 0:
        raise ValueError("metrics_results must not be empty")

    first_elem_type = type(metrics_results[0])
    all_same_type = all(isinstance(x, first_elem_type) for x in metrics_results)
    if not all_same_type:
        raise ValueError("All metrics_results must be of the same type")

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

