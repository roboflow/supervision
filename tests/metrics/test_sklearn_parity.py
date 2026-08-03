"""Golden parity tests for classification-style detection outcomes.

The constants in this module were generated with scikit-learn 1.7.1. Unmatched
targets and predictions are represented by ``BACKGROUND_CLASS_ID`` in the sklearn
event stream, while ``labels`` contains only foreground classes. This produces the
same per-class TP, FP, and FN counts as Supervision's detection metrics.

To regenerate the constants locally without adding sklearn to the test environment::

    from sklearn.metrics import f1_score, precision_score, recall_score

    functions = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    for case in PARITY_CASES:
        for metric_name, function in functions.items():
            print(case.name, metric_name)
            print(
                function(
                    case.sklearn_y_true,
                    case.sklearn_y_pred,
                    labels=case.sklearn_labels,
                    average=None,
                    zero_division=0,
                )
            )
            for average in ("micro", "macro", "weighted"):
                print(
                    average,
                    function(
                        case.sklearn_y_true,
                        case.sklearn_y_pred,
                        labels=case.sklearn_labels,
                        average=average,
                        zero_division=0,
                    ),
                )

For the fully empty case, sklearn needs an explicit generation-only label to apply
its zero-division convention to macro and weighted averages. Supervision correctly
tracks no class for that case, so its per-class result remains empty.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod
from supervision.metrics.f1_score import F1Score
from supervision.metrics.precision import Precision
from supervision.metrics.recall import Recall

BACKGROUND_CLASS_ID = -1


@dataclass(frozen=True)
class MetricGoldens:
    """Store sklearn outputs for one metric and one detection scenario."""

    per_class: tuple[float, ...]
    micro: float
    macro: float
    weighted: float

    def average(self, name: Literal["micro", "macro", "weighted"]) -> float:
        """Return the golden aggregate for an averaging method."""
        if name == "micro":
            return self.micro
        if name == "macro":
            return self.macro
        return self.weighted


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
    goldens: dict[str, MetricGoldens]


@dataclass(frozen=True)
class MetricSpec:
    """Connect a Supervision metric result to its sklearn golden values."""

    name: Literal["precision", "recall", "f1"]
    metric_class: type[Precision] | type[Recall] | type[F1Score]
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
        goldens={
            "precision": MetricGoldens(
                per_class=(0.5, 0.5, 1.0),
                micro=0.6,
                macro=0.6666666666666666,
                weighted=0.7,
            ),
            "recall": MetricGoldens(
                per_class=(0.5, 1.0, 0.5),
                micro=0.6,
                macro=0.6666666666666666,
                weighted=0.6,
            ),
            "f1": MetricGoldens(
                per_class=(0.5, 0.6666666666666666, 0.6666666666666666),
                micro=0.6,
                macro=0.611111111111111,
                weighted=0.6,
            ),
        },
    ),
    ParityCase(
        name="prediction-only-label",
        prediction_batches=((0, 0, 7),),
        target_batches=((0, 0, 0),),
        sklearn_y_true=(0, 0, 0),
        sklearn_y_pred=(0, 0, 7),
        sklearn_labels=(0, 7),
        matched_classes=(0, 7),
        goldens={
            "precision": MetricGoldens(
                per_class=(1.0, 0.0),
                micro=0.6666666666666666,
                macro=0.5,
                weighted=1.0,
            ),
            "recall": MetricGoldens(
                per_class=(0.6666666666666666, 0.0),
                micro=0.6666666666666666,
                macro=0.3333333333333333,
                weighted=0.6666666666666666,
            ),
            "f1": MetricGoldens(
                per_class=(0.8, 0.0),
                micro=0.6666666666666666,
                macro=0.4,
                weighted=0.8000000000000002,
            ),
        },
    ),
    ParityCase(
        name="target-only-label",
        prediction_batches=((0, 0, 0),),
        target_batches=((0, 0, 9),),
        sklearn_y_true=(0, 0, 9),
        sklearn_y_pred=(0, 0, 0),
        sklearn_labels=(0, 9),
        matched_classes=(0, 9),
        goldens={
            "precision": MetricGoldens(
                per_class=(0.6666666666666666, 0.0),
                micro=0.6666666666666666,
                macro=0.3333333333333333,
                weighted=0.4444444444444444,
            ),
            "recall": MetricGoldens(
                per_class=(1.0, 0.0),
                micro=0.6666666666666666,
                macro=0.5,
                weighted=0.6666666666666666,
            ),
            "f1": MetricGoldens(
                per_class=(0.8, 0.0),
                micro=0.6666666666666666,
                macro=0.4,
                weighted=0.5333333333333333,
            ),
        },
    ),
    ParityCase(
        name="non-contiguous-ids",
        prediction_batches=((2, 11, 11, 42, 2, 42),),
        target_batches=((2, 2, 11, 42, 42, 42),),
        sklearn_y_true=(2, 2, 11, 42, 42, 42),
        sklearn_y_pred=(2, 11, 11, 42, 2, 42),
        sklearn_labels=(2, 11, 42),
        matched_classes=(2, 11, 42),
        goldens={
            "precision": MetricGoldens(
                per_class=(0.5, 0.5, 1.0),
                micro=0.6666666666666666,
                macro=0.6666666666666666,
                weighted=0.75,
            ),
            "recall": MetricGoldens(
                per_class=(0.5, 1.0, 0.6666666666666666),
                micro=0.6666666666666666,
                macro=0.7222222222222222,
                weighted=0.6666666666666666,
            ),
            "f1": MetricGoldens(
                per_class=(0.5, 0.6666666666666666, 0.8),
                micro=0.6666666666666666,
                macro=0.6555555555555556,
                weighted=0.6777777777777777,
            ),
        },
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
        goldens={
            "precision": MetricGoldens(
                per_class=(0.5, 1.0, 0.0),
                micro=0.4,
                macro=0.5,
                weighted=0.75,
            ),
            "recall": MetricGoldens(
                per_class=(1.0, 1.0, 0.0),
                micro=1.0,
                macro=0.6666666666666666,
                weighted=1.0,
            ),
            "f1": MetricGoldens(
                per_class=(0.6666666666666666, 1.0, 0.0),
                micro=0.5714285714285714,
                macro=0.5555555555555555,
                weighted=0.8333333333333333,
            ),
        },
    ),
    ParityCase(
        name="targets-only",
        prediction_batches=((),),
        target_batches=((3, 10),),
        sklearn_y_true=(3, 10),
        sklearn_y_pred=(BACKGROUND_CLASS_ID, BACKGROUND_CLASS_ID),
        sklearn_labels=(3, 10),
        matched_classes=(3, 10),
        goldens={
            metric_name: MetricGoldens(
                per_class=(0.0, 0.0), micro=0.0, macro=0.0, weighted=0.0
            )
            for metric_name in ("precision", "recall", "f1")
        },
    ),
    ParityCase(
        name="predictions-only",
        prediction_batches=((3, 10),),
        target_batches=((),),
        sklearn_y_true=(BACKGROUND_CLASS_ID, BACKGROUND_CLASS_ID),
        sklearn_y_pred=(3, 10),
        sklearn_labels=(3, 10),
        matched_classes=(3, 10),
        goldens={
            metric_name: MetricGoldens(
                per_class=(0.0, 0.0), micro=0.0, macro=0.0, weighted=0.0
            )
            for metric_name in ("precision", "recall", "f1")
        },
    ),
    ParityCase(
        name="empty",
        prediction_batches=((),),
        target_batches=((),),
        sklearn_y_true=(),
        sklearn_y_pred=(),
        sklearn_labels=(0,),
        matched_classes=(),
        goldens={
            metric_name: MetricGoldens(per_class=(), micro=0.0, macro=0.0, weighted=0.0)
            for metric_name in ("precision", "recall", "f1")
        },
    ),
)

METRIC_SPECS = (
    MetricSpec(
        name="precision",
        metric_class=Precision,
        scores_attribute="precision_scores",
        per_class_attribute="precision_per_class",
    ),
    MetricSpec(
        name="recall",
        metric_class=Recall,
        scores_attribute="recall_scores",
        per_class_attribute="recall_per_class",
    ),
    MetricSpec(
        name="f1",
        metric_class=F1Score,
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
    """Verify metric outputs against independently generated sklearn goldens."""

    @pytest.mark.parametrize(("averaging_method", "average_name"), AVERAGING_CASES)
    def test_aggregate_scores_match_goldens(
        self,
        case: ParityCase,
        metric_spec: MetricSpec,
        averaging_method: AveragingMethod,
        average_name: Literal["micro", "macro", "weighted"],
    ) -> None:
        """Aggregate scores match sklearn at every exact-match IoU threshold."""
        predictions, targets = _build_detection_batches(case)
        metric = metric_spec.metric_class(averaging_method=averaging_method)

        result = metric.update(predictions, targets).compute()

        expected_value = case.goldens[metric_spec.name].average(average_name)
        expected = np.full(result.iou_thresholds.shape, expected_value)
        actual = getattr(result, metric_spec.scores_attribute)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_per_class_scores_match_goldens(
        self, case: ParityCase, metric_spec: MetricSpec
    ) -> None:
        """Per-class scores and class ordering match sklearn golden outputs."""
        predictions, targets = _build_detection_batches(case)
        metric = metric_spec.metric_class(averaging_method=AveragingMethod.MACRO)

        result = metric.update(predictions, targets).compute()

        expected_values = case.goldens[metric_spec.name].per_class
        expected = np.repeat(
            np.asarray(expected_values, dtype=np.float64)[:, None],
            result.iou_thresholds.size,
            axis=1,
        )
        actual = getattr(result, metric_spec.per_class_attribute)
        np.testing.assert_array_equal(result.matched_classes, case.matched_classes)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
