import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget
from supervision.metrics.f1_score import F1Score
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from supervision.metrics.mean_average_recall import MeanAverageRecall
from supervision.metrics.precision import Precision
from supervision.metrics.recall import Recall


def _non_square_obb_detections(confidence: bool = False) -> Detections:
    obb = np.array(
        [[[10, 0], [0, 1], [30, 4], [40, 3]]],
        dtype=np.float32,
    )
    return Detections(
        xyxy=np.array([[0, 0, 40, 4]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]) if confidence else None,
        data={ORIENTED_BOX_COORDINATES: obb},
    )


@pytest.mark.parametrize(
    ("metric_cls", "score_name"),
    [
        (Precision, "precision_at_50"),
        (Recall, "recall_at_50"),
        (F1Score, "f1_50"),
        (MeanAveragePrecision, "map50_95"),
        (MeanAverageRecall, "mAR_at_100"),
    ],
)
def test_perfect_non_square_oriented_boxes_score_as_perfect(
    metric_cls: type,
    score_name: str,
) -> None:
    predictions = _non_square_obb_detections(confidence=True)
    targets = _non_square_obb_detections()

    metric = metric_cls(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
    result = metric.update([predictions], [targets]).compute()

    assert getattr(result, score_name) == pytest.approx(1.0)
