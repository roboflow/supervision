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

_PRECISION_LABELS = _ResultLabels(
    short="P",
    long="Precision",
    plot_title="Precision",
    metric_target_padding="    ",
)


class Precision(_ConfusionMatrixMetric["PrecisionResult"]):
    """Precision is a metric used to evaluate object detection models. It is the ratio
    of true positive detections to the total number of predicted detections. We
    calculate it at different IoU thresholds.

    In simple terms, Precision is a measure of a model's accuracy, calculated as:

    `Precision = TP / (TP + FP)`

    Here, `TP` is the number of true positives (correct detections), and `FP` is the
    number of false positive detections (detected, but incorrectly).

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import Precision
        >>> predictions = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0]),
        ...     confidence=np.array([0.9])
        ... )
        >>> targets = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0])
        ... )
        >>> precision_metric = Precision()
        >>> precision_result = precision_metric.update(predictions, targets).compute()
        >>> round(float(precision_result.precision_at_50), 2)
        1.0

        ```

    ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/precision_plot_example.png
    ){ align=center width="800" }
    """

    _metric_name = "Precision"

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        averaging_method: AveragingMethod = AveragingMethod.WEIGHTED,
    ):
        """Initialize the Precision metric.

        Args:
            metric_target: The type of detection data to use.
            averaging_method: The averaging method used to compute the
                precision. Determines how the precision is aggregated across classes.
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
    ) -> Precision:
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

    def compute(self) -> PrecisionResult:
        """Calculate the precision metric based on the stored predictions and ground-
        truth data, at different IoU thresholds.

        Returns:
            The precision metric result.
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
        """Return the precision implied by each entry of the confusion matrix."""
        return self._compute_precision(confusion_matrix)

    def _build_result(
        self,
        scores: npt.NDArray[np.float64],
        per_class_scores: npt.NDArray[np.float64],
        iou_thresholds: npt.NDArray[np.float32],
        matched_classes: npt.NDArray[np.int32],
    ) -> PrecisionResult:
        """Wrap the computed precision scores in a `PrecisionResult`."""
        return PrecisionResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            precision_scores=scores,
            precision_per_class=per_class_scores,
            iou_thresholds=iou_thresholds,
            matched_classes=matched_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

    @staticmethod
    def _compute_precision(
        confusion_matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Broadcastable function, computing the precision from the confusion matrix.

        Args:
            confusion_matrix: shape (N, ..., 3), where the last dimension
                contains the true positives, false positives, and false negatives.

        Returns:
            shape (N, ...), containing the precision for each element.
        """
        _validate_confusion_matrix(confusion_matrix)
        true_positives = confusion_matrix[..., 0]
        false_positives = confusion_matrix[..., 1]

        denominator = true_positives + false_positives
        precision = np.divide(
            true_positives,
            denominator,
            out=np.zeros_like(true_positives),
            where=denominator != 0,
        )

        result_precision: npt.NDArray[np.float64] = precision
        return result_precision


@dataclass
class PrecisionResult:
    """The results of the precision metric calculation.

    Defaults to `0` if no detections or targets were provided.

    Attributes:
        metric_target: the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        averaging_method: the averaging method used to compute the
            precision. Determines how the precision is aggregated across classes.
        precision_at_50: the precision at IoU threshold of `0.5`.
        precision_at_75: the precision at IoU threshold of `0.75`.
        precision_scores: the precision scores at each IoU threshold.
            Shape: `(num_iou_thresholds,)`
        precision_per_class: the precision scores per class and
            IoU threshold. Shape: `(num_classes, num_iou_thresholds)`
        iou_thresholds: the IoU thresholds used in the calculations.
        matched_classes: the class IDs present in either predictions or ground
            truth. Corresponds to the rows of `precision_per_class`. Classes
            that appear only in predictions (no ground-truth instances) are
            included; their per-threshold precision values will be `0.0`.
        small_objects: the Precision metric results
            for small objects (area < 32²).
        medium_objects: the Precision metric results
            for medium objects (32² ≤ area < 96²).
        large_objects: the Precision metric results
            for large objects (area ≥ 96²).
    """

    metric_target: MetricTarget
    averaging_method: AveragingMethod

    @property
    def precision_at_50(self) -> float:
        return float(self.precision_scores[0])

    @property
    def precision_at_75(self) -> float:
        return float(self.precision_scores[5])

    precision_scores: npt.NDArray[np.float64]
    precision_per_class: npt.NDArray[np.float64]
    iou_thresholds: npt.NDArray[np.float32]
    matched_classes: npt.NDArray[np.int32]

    small_objects: PrecisionResult | None
    medium_objects: PrecisionResult | None
    large_objects: PrecisionResult | None

    def _result_view(self) -> _ResultView:
        """Describe this result in the form the shared renderers consume."""
        return _ResultView(
            class_name=self.__class__.__name__,
            labels=_PRECISION_LABELS,
            metric_target=self.metric_target,
            averaging_method=self.averaging_method,
            score_at_50=self.precision_at_50,
            score_at_75=self.precision_at_75,
            scores=self.precision_scores,
            per_class_scores=self.precision_per_class,
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
            >>> from supervision.metrics import Precision
            >>> predictions = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0]),
            ...     confidence=np.array([0.9])
            ... )
            >>> targets = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0])
            ... )
            >>> precision_metric = Precision()
            >>> precision_result = precision_metric.update(
            ...     predictions, targets
            ... ).compute()
            >>> print(precision_result)  # doctest: +ELLIPSIS
            PrecisionResult:
            Metric target:    MetricTarget.BOXES
            Averaging method: AveragingMethod.WEIGHTED
            P @ 50:     1.0000
            P @ 75:     1.0000
            P @ thresh: [1. ... 1.]
            IoU thresh: [0.5  0.55 ... 0.95]
            Precision per class:
              0: [1. ... 1.]
            ...
            Medium objects:
              PrecisionResult:
              Metric target:    MetricTarget.BOXES
              Averaging method: AveragingMethod.WEIGHTED
              P @ 50:     0.0000
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
        """Plot the precision results.

        ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/precision_plot_example.png
        ){ align=center width="800" }
        """
        _plot_result(self._result_view())
