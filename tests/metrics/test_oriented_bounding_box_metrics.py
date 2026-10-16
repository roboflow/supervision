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
        (MeanAverageRecall, "mAR_at_100"),
    ],
)
def test_perfect_non_square_oriented_boxes_score_as_perfect(
    metric_cls: type,
    score_name: str,
) -> None:
    """Perfect non-square OBB predictions score 1.0 for metrics that use OBB IoU."""
    predictions = _non_square_obb_detections(confidence=True)
    targets = _non_square_obb_detections()

    metric = metric_cls(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
    result = metric.update([predictions], [targets]).compute()

    assert getattr(result, score_name) == pytest.approx(1.0)


def test_mean_average_precision_accepts_obb_metric_target() -> None:
    """MeanAveragePrecision routes metric_target=ORIENTED_BOUNDING_BOXES through
    oriented_box_iou_batch; perfect OBB predictions score 1.0."""
    predictions = _non_square_obb_detections(confidence=True)
    targets = _non_square_obb_detections()

    metric = MeanAveragePrecision(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
    result = metric.update([predictions], [targets]).compute()

    assert result.map50_95 == pytest.approx(1.0)


def _axis_aligned_detections(
    xyxy: list[float], confidence: float | None = None
) -> Detections:
    """Build one detection whose oriented box is its axis-aligned box.

    Examples:
        ```pycon
        >>> detections = _axis_aligned_detections([0, 0, 10, 10])
        >>> detections.data[ORIENTED_BOX_COORDINATES].tolist()
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]

        ```
    """
    x_min, y_min, x_max, y_max = xyxy
    corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return Detections(
        xyxy=np.array([xyxy], dtype=np.float64),
        class_id=np.array([0]),
        confidence=None if confidence is None else np.array([confidence]),
        data={ORIENTED_BOX_COORDINATES: np.array([corners], dtype=np.float64)},
    )


@pytest.mark.parametrize(
    ("metric_cls", "scores_name"),
    [
        (Precision, "precision_scores"),
        (Recall, "recall_scores"),
        (F1Score, "f1_scores"),
        (MeanAverageRecall, "recall_per_class"),
    ],
)
def test_oriented_box_iou_on_a_threshold_matches_like_axis_aligned_box(
    metric_cls: type,
    scores_name: str,
) -> None:
    """An IoU of exactly 0.6 passes the 0.6 threshold for OBB as it does for boxes."""
    # Overlap 9 x 10 = 90 over union 100 + 140 - 90 = 150 gives IoU 0.6.
    predictions = _axis_aligned_detections([0, 1, 14, 11], confidence=0.9)
    targets = _axis_aligned_detections([0, 0, 10, 10])

    boxes_result = (
        metric_cls(metric_target=MetricTarget.BOXES)
        .update([predictions], [targets])
        .compute()
    )
    obb_result = (
        metric_cls(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
        .update([predictions], [targets])
        .compute()
    )

    np.testing.assert_array_equal(
        getattr(obb_result, scores_name), getattr(boxes_result, scores_name)
    )
