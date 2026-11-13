---
comments: true
---

# Aggregate Comparison

Compare metric results across multiple models side-by-side — as a table or a grouped bar chart.

Install the metrics extra before using these APIs:

```bash
pip install "supervision[metrics]"
```

## Compare Metric Results

Compute the same metric for each model, then aggregate the resulting scores in a table or a grouped bar chart.

```python
import numpy as np
import supervision as sv
from supervision.metrics import (
    F1Score,
    aggregate_metric_results,
    plot_aggregate_metric_results,
)

targets = sv.Detections(
    xyxy=np.array([[0, 0, 10, 10]]),
    class_id=np.array([0]),
)
model_a_predictions = sv.Detections(
    xyxy=np.array([[0, 0, 10, 10]]),
    class_id=np.array([0]),
    confidence=np.array([0.9]),
)
model_b_predictions = sv.Detections(
    xyxy=np.array([[3, 3, 10, 10]]),
    class_id=np.array([0]),
    confidence=np.array([0.9]),
)

metric_results = [
    F1Score().update(model_a_predictions, targets).compute(),
    F1Score().update(model_b_predictions, targets).compute(),
]
model_names = ["Model A", "Model B"]

comparison = aggregate_metric_results(metric_results, model_names=model_names)
print(comparison[["F1@50", "F1@75"]])

plot_aggregate_metric_results(
    metric_results,
    model_names=model_names,
    show=True,
)
```

## Functions

<div class="md-typeset">
    <h3><a href="#supervision.metrics.utils.aggregate.aggregate_metric_results">aggregate_metric_results</a></h3>
</div>

:::supervision.metrics.utils.aggregate.aggregate_metric_results

<div class="md-typeset">
    <h3><a href="#supervision.metrics.utils.aggregate.plot_aggregate_metric_results">plot_aggregate_metric_results</a></h3>
</div>

:::supervision.metrics.utils.aggregate.plot_aggregate_metric_results

## Supporting Types

<div class="md-typeset">
    <h3><a href="#supervision.metrics.core.MetricResult">MetricResult</a></h3>
</div>

:::supervision.metrics.core.MetricResult

<div class="md-typeset">
    <h3><a href="#supervision.metrics.core.PlotDetails">PlotDetails</a></h3>
</div>

:::supervision.metrics.core.PlotDetails
