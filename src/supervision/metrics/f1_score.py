from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.metrics._confusion_matrix_metric import (
    _ConfusionMatrixMetric,
    _format_result,
    _optional_result_view,
    _plot_result,
    _result_to_pandas,
    _ResultLabels,
    _ResultView,
    _validate_confusion_matrix,
)
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.utils.object_size import ObjectSizeCategory

if TYPE_CHECKING:
    import pandas as pd

_F1_LABELS = _ResultLabels(
    short="F1",
    long="F1",
    plot_title="F1 Score",
    metric_target_padding=" ",
)


class F1Score(_ConfusionMatrixMetric["F1ScoreResult"]):
    """F1 Score is a metric used to evaluate object detection models. It is the harmonic
    mean of precision and recall, calculated at different IoU thresholds.

    In simple terms, F1 Score is a measure of a model's balance between precision and
    recall (accuracy and completeness), calculated as:

    `F1 = 2 * (precision * recall) / (precision + recall)`

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import F1Score
        >>> predictions = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0]),
        ...     confidence=np.array([0.9])
        ... )
        >>> targets = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0])
        ... )
        >>> f1_metric = F1Score()
        >>> f1_result = f1_metric.update(predictions, targets).compute()
        >>> round(float(f1_result.f1_50), 2)
        1.0

        ```

    ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/f1_plot_example.png
    ){ align=center width="800" }
    """

    _metric_name = "F1Score"

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        averaging_method: AveragingMethod = AveragingMethod.WEIGHTED,
    ):
        """Initialize the F1Score metric.

        Args:
            metric_target: The type of detection data to use.
            averaging_method: The averaging method used to compute the
                F1 scores. Determines how the F1 scores are aggregated across classes.
        """
        self._metric_target = metric_target
        self.averaging_method = averaging_method

        self._predictions_list: list[Detections] = []
        self._targets_list: list[Detections] = []

    def reset(self) -> None:
        """Reset the metric to its initial state, clearing all stored data."""
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: Detections | list[Detections],
        targets: Detections | list[Detections],
    ) -> F1Score:
        """Add new predictions and targets to the metric, but do not compute the result.

        Args:
            predictions: The predicted detections.
            targets: The target detections.

        Returns:
            The updated metric instance.
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]

        if len(predictions) != len(targets):
            raise ValueError(
                f"The number of predictions ({len(predictions)}) and"
                f" targets ({len(targets)}) during the update must be the same."
            )

        self._predictions_list.extend(predictions)
        self._targets_list.extend(targets)

        return self

    def compute(self) -> F1ScoreResult:
        """Calculate the F1 score metric based on the stored predictions and ground-
        truth data, at different IoU thresholds.

        Returns:
            The F1 score metric result.
        """
        result = self._compute(self._predictions_list, self._targets_list)
        result.small_objects = self._compute(
            self._predictions_list, self._targets_list, ObjectSizeCategory.SMALL
        )
        result.medium_objects = self._compute(
            self._predictions_list, self._targets_list, ObjectSizeCategory.MEDIUM
        )
        result.large_objects = self._compute(
            self._predictions_list, self._targets_list, ObjectSizeCategory.LARGE
        )

        return result

    def _score_from_confusion_matrix(
        self, confusion_matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return the F1 score implied by each entry of the confusion matrix."""
        return self._compute_f1(confusion_matrix)

    def _build_result(
        self,
        scores: npt.NDArray[np.float64],
        per_class_scores: npt.NDArray[np.float64],
        iou_thresholds: npt.NDArray[np.float32],
        matched_classes: npt.NDArray[np.int32],
    ) -> F1ScoreResult:
        """Wrap the computed F1 scores in an `F1ScoreResult`."""
        return F1ScoreResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            f1_scores=scores,
            f1_per_class=per_class_scores,
            iou_thresholds=iou_thresholds,
            matched_classes=matched_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

    @staticmethod
    def _compute_f1(
        confusion_matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Broadcastable function, computing the F1 score from the confusion matrix.

        Args:
            confusion_matrix: shape (N, ..., 3), where the last dimension
                contains the true positives, false positives, and false negatives.

        Returns:
            shape (N, ...), containing the F1 score for each element.
        """
        _validate_confusion_matrix(confusion_matrix)
        true_positives = confusion_matrix[..., 0]
        false_positives = confusion_matrix[..., 1]
        false_negatives = confusion_matrix[..., 2]

        # Alternate formula, avoids multiple zero division checks
        denominator = 2 * true_positives + false_positives + false_negatives
        f1_score = np.divide(
            2 * true_positives,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator != 0,
        )

        result_f1_score: npt.NDArray[np.float64] = f1_score
        return result_f1_score


@dataclass
class F1ScoreResult:
    """The results of the F1 score metric calculation.

    Defaults to `0` if no detections or targets were provided.

    Attributes:
        metric_target: the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        averaging_method: the averaging method used to compute the
            F1 scores. Determines how the F1 scores are aggregated across classes.
        f1_50: the F1 score at IoU threshold of `0.5`.
        f1_75: the F1 score at IoU threshold of `0.75`.
        f1_scores: the F1 scores at each IoU threshold.
            Shape: `(num_iou_thresholds,)`
        f1_per_class: the F1 scores per class and IoU threshold.
            Shape: `(num_classes, num_iou_thresholds)`
        iou_thresholds: the IoU thresholds used in the calculations.
        matched_classes: the class IDs present in either predictions or ground
            truth. Corresponds to the rows of `f1_per_class`. Classes that
            appear only in predictions (no ground-truth instances) are
            included; their per-threshold F1 values will be `0.0`.
        small_objects: the F1 metric results
            for small objects (area < 32²).
        medium_objects: the F1 metric results
            for medium objects (32² ≤ area < 96²).
        large_objects: the F1 metric results
            for large objects (area ≥ 96²).
    """

    metric_target: MetricTarget
    averaging_method: AveragingMethod

    @property
    def f1_50(self) -> float:
        return float(self.f1_scores[0])

    @property
    def f1_75(self) -> float:
        return float(self.f1_scores[5])

    f1_scores: npt.NDArray[np.float64]
    f1_per_class: npt.NDArray[np.float64]
    iou_thresholds: npt.NDArray[np.float32]
    matched_classes: npt.NDArray[np.int32]

    small_objects: F1ScoreResult | None
    medium_objects: F1ScoreResult | None
    large_objects: F1ScoreResult | None

    def _result_view(self) -> _ResultView:
        """Describe this result in the form the shared renderers consume."""
        return _ResultView(
            class_name=self.__class__.__name__,
            labels=_F1_LABELS,
            metric_target=self.metric_target,
            averaging_method=self.averaging_method,
            score_at_50=self.f1_50,
            score_at_75=self.f1_75,
            scores=self.f1_scores,
            per_class_scores=self.f1_per_class,
            iou_thresholds=self.iou_thresholds,
            matched_classes=self.matched_classes,
            small_objects=_optional_result_view(self.small_objects),
            medium_objects=_optional_result_view(self.medium_objects),
            large_objects=_optional_result_view(self.large_objects),
        )

    def __str__(self) -> str:
        """Format as a pretty string.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> from supervision.metrics import F1Score
            >>> predictions = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0]),
            ...     confidence=np.array([0.9])
            ... )
            >>> targets = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0])
            ... )
            >>> f1_metric = F1Score()
            >>> f1_result = f1_metric.update(predictions, targets).compute()
            >>> print(f1_result)  # doctest: +ELLIPSIS
            F1ScoreResult:
            Metric target: MetricTarget.BOXES
            Averaging method: AveragingMethod.WEIGHTED
            F1 @ 50:     1.0000
            F1 @ 75:     1.0000
            F1 @ thresh: [1. ... 1.]
            IoU thresh:  [0.5  0.55 ... 0.95]
            F1 per class:
              0: [1. ... 1.]
            ...
            Medium objects:
              F1ScoreResult:
              Metric target: MetricTarget.BOXES
              Averaging method: AveragingMethod.WEIGHTED
              F1 @ 50:     0.0000
              ...

            ```
        """
        return _format_result(self._result_view())

    def to_pandas(self) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame.

        Returns:
            The result as a DataFrame.
        """
        return _result_to_pandas(self._result_view())

    def plot(self) -> None:
        """Plot the F1 results.

        ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/f1_plot_example.png
        ){ align=center width="800" }
        """
        _plot_result(self._result_view())
