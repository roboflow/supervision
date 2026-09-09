"""Compare detection-style metric outcomes with live scikit-learn results.

Unmatched targets and predictions are represented by ``BACKGROUND_CLASS_ID`` in
the sklearn event stream, while evaluated labels contain only foreground classes.
This produces the same per-class TP, FP, and FN counts as Supervision's metrics.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod
from supervision.metrics.f1_score import F1Score
from supervision.metrics.precision import Precision
from supervision.metrics.recall import Recall

BACKGROUND_CLASS_ID = -1


@dataclass(frozen=True)
class ParityCase:
    """Define equivalent Supervision detection batches and sklearn label events."""

    name: str
    prediction_batches: tuple[tuple[int, ...], ...]
    target_batches: tuple[tuple[int, ...], ...]
    sklearn_y_true: tuple[int, ...]
    sklearn_y_pred: tuple[int, ...]
    sklearn_labels: tuple[int, ...]
    matched_classes: tuple[int, ...]


@dataclass(frozen=True)
class MetricSpec:
    """Connect a Supervision metric result to its scikit-learn function."""

    name: Literal["precision", "recall", "f1"]
    metric_class: type[Precision] | type[Recall] | type[F1Score]
    sklearn_function: Callable[..., float | np.ndarray]
    scores_attribute: str
    per_class_attribute: str


PARITY_CASES = (
    ParityCase(
        name="balanced",
        prediction_batches=((0, 2, 2, 5, 0),),
        target_batches=((0, 0, 2, 5, 5),),
        sklearn_y_true=(0, 0, 2, 5, 5),
        sklearn_y_pred=(0, 2, 2, 5, 0),
        sklearn_labels=(0, 2, 5),
        matched_classes=(0, 2, 5),
    ),
    ParityCase(
        name="prediction-only-label",
        prediction_batches=((0, 0, 7),),
        target_batches=((0, 0, 0),),
        sklearn_y_true=(0, 0, 0),
        sklearn_y_pred=(0, 0, 7),
        sklearn_labels=(0, 7),
        matched_classes=(0, 7),
    ),
    ParityCase(
        name="target-only-label",
        prediction_batches=((0, 0, 0),),
        target_batches=((0, 0, 9),),
        sklearn_y_true=(0, 0, 9),
        sklearn_y_pred=(0, 0, 0),
        sklearn_labels=(0, 9),
        matched_classes=(0, 9),
    ),
    ParityCase(
        name="non-contiguous-ids",
        prediction_batches=((2, 11, 11, 42, 2, 42),),
        target_batches=((2, 2, 11, 42, 42, 42),),
        sklearn_y_true=(2, 2, 11, 42, 42, 42),
        sklearn_y_pred=(2, 11, 11, 42, 2, 42),
        sklearn_labels=(2, 11, 42),
        matched_classes=(2, 11, 42),
    ),
    ParityCase(
        name="background-image",
        prediction_batches=((0, 2), (0, 11, 11)),
        target_batches=((0, 2), ()),
        sklearn_y_true=(
            0,
            2,
            BACKGROUND_CLASS_ID,
            BACKGROUND_CLASS_ID,
            BACKGROUND_CLASS_ID,
        ),
        sklearn_y_pred=(0, 2, 0, 11, 11),
        sklearn_labels=(0, 2, 11),
        matched_classes=(0, 2, 11),
    ),
    ParityCase(
        name="targets-only",
        prediction_batches=((),),
        target_batches=((3, 10),),
        sklearn_y_true=(3, 10),
        sklearn_y_pred=(BACKGROUND_CLASS_ID, BACKGROUND_CLASS_ID),
        sklearn_labels=(3, 10),
        matched_classes=(3, 10),
    ),
    ParityCase(
        name="predictions-only",
        prediction_batches=((3, 10),),
        target_batches=((),),
        sklearn_y_true=(BACKGROUND_CLASS_ID, BACKGROUND_CLASS_ID),
        sklearn_y_pred=(3, 10),
        sklearn_labels=(3, 10),
        matched_classes=(3, 10),
    ),
)

METRIC_SPECS = (
    MetricSpec(
        name="precision",
        metric_class=Precision,
        sklearn_function=precision_score,
        scores_attribute="precision_scores",
        per_class_attribute="precision_per_class",
    ),
    MetricSpec(
        name="recall",
        metric_class=Recall,
        sklearn_function=recall_score,
        scores_attribute="recall_scores",
        per_class_attribute="recall_per_class",
    ),
    MetricSpec(
        name="f1",
        metric_class=F1Score,
        sklearn_function=f1_score,
        scores_attribute="f1_scores",
        per_class_attribute="f1_per_class",
    ),
)

AVERAGING_CASES = (
    pytest.param(AveragingMethod.MICRO, "micro", id="micro"),
    pytest.param(AveragingMethod.MACRO, "macro", id="macro"),
    pytest.param(AveragingMethod.WEIGHTED, "weighted", id="weighted"),
)


def _detections_from_classes(
    class_ids: tuple[int, ...], *, predictions: bool
) -> Detections:
    """Build isolated boxes whose equal indices are the only possible matches."""
    if not class_ids:
        return Detections.empty()

    left = np.arange(len(class_ids), dtype=np.float32) * 30
    zeros = np.zeros(len(class_ids), dtype=np.float32)
    xyxy = np.column_stack((left, zeros, left + 10, zeros + 10))
    confidence = (
        np.full(len(class_ids), 0.99, dtype=np.float32) if predictions else None
    )
    return Detections(
        xyxy=xyxy,
        class_id=np.asarray(class_ids, dtype=np.int32),
        confidence=confidence,
    )


def _build_detection_batches(
    case: ParityCase,
) -> tuple[list[Detections], list[Detections]]:
    """Convert a parity case's class batches into Supervision inputs."""
    predictions = [
        _detections_from_classes(class_ids, predictions=True)
        for class_ids in case.prediction_batches
    ]
    targets = [
        _detections_from_classes(class_ids, predictions=False)
        for class_ids in case.target_batches
    ]
    return predictions, targets


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case.name) for case in PARITY_CASES]
)
@pytest.mark.parametrize(
    "metric_spec",
    [pytest.param(spec, id=spec.name) for spec in METRIC_SPECS],
)
class TestSklearnParity:
    """Verify metric outputs against independently computed sklearn results."""

    @pytest.mark.parametrize(("averaging_method", "average_name"), AVERAGING_CASES)
    def test_aggregate_scores_match_sklearn(
        self,
        case: ParityCase,
        metric_spec: MetricSpec,
        averaging_method: AveragingMethod,
        average_name: Literal["micro", "macro", "weighted"],
    ) -> None:
        """Aggregate scores match live sklearn results at every IoU threshold."""
        predictions, targets = _build_detection_batches(case)
        metric = metric_spec.metric_class(averaging_method=averaging_method)

        result = metric.update(predictions, targets).compute()

        expected_value = metric_spec.sklearn_function(
            case.sklearn_y_true,
            case.sklearn_y_pred,
            labels=case.sklearn_labels,
            average=average_name,
            zero_division=0,
        )
        expected = np.full(result.iou_thresholds.shape, expected_value)
        actual = getattr(result, metric_spec.scores_attribute)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_per_class_scores_match_sklearn(
        self, case: ParityCase, metric_spec: MetricSpec
    ) -> None:
        """Per-class scores and class ordering match live sklearn results."""
        predictions, targets = _build_detection_batches(case)
        metric = metric_spec.metric_class(averaging_method=AveragingMethod.MACRO)

        result = metric.update(predictions, targets).compute()

        expected_values = metric_spec.sklearn_function(
            case.sklearn_y_true,
            case.sklearn_y_pred,
            labels=case.matched_classes,
            average=None,
            zero_division=0,
        )
        expected = np.repeat(
            np.asarray(expected_values, dtype=np.float64)[:, None],
            result.iou_thresholds.size,
            axis=1,
        )
        actual = getattr(result, metric_spec.per_class_attribute)
        np.testing.assert_array_equal(result.matched_classes, case.matched_classes)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "metric_spec",
    [pytest.param(spec, id=spec.name) for spec in METRIC_SPECS],
)
class TestEmptyInput:
    """Verify Supervision's defined behavior when no detections exist."""

    @pytest.mark.parametrize(
        "averaging_method",
        [
            pytest.param(AveragingMethod.MICRO, id="micro"),
            pytest.param(AveragingMethod.MACRO, id="macro"),
            pytest.param(AveragingMethod.WEIGHTED, id="weighted"),
        ],
    )
    def test_returns_zero_aggregate_scores(
        self, metric_spec: MetricSpec, averaging_method: AveragingMethod
    ) -> None:
        """Empty batches produce zero aggregate scores at every IoU threshold."""
        predictions = [Detections.empty()]
        targets = [Detections.empty()]
        metric = metric_spec.metric_class(averaging_method=averaging_method)

        result = metric.update(predictions, targets).compute()

        actual = getattr(result, metric_spec.scores_attribute)
        np.testing.assert_array_equal(actual, np.zeros_like(result.iou_thresholds))

    def test_returns_no_per_class_scores(self, metric_spec: MetricSpec) -> None:
        """Empty batches produce no matched classes or per-class score rows."""
        predictions = [Detections.empty()]
        targets = [Detections.empty()]
        metric = metric_spec.metric_class(averaging_method=AveragingMethod.MACRO)

        result = metric.update(predictions, targets).compute()

        actual = getattr(result, metric_spec.per_class_attribute)
        np.testing.assert_array_equal(result.matched_classes, np.array([]))
        assert actual.shape == (0, result.iou_thresholds.size)
