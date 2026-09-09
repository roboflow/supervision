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

_RECALL_LABELS = _ResultLabels(
    short="R",
    long="Recall",
    plot_title="Recall",
    metric_target_padding="    ",
)


class Recall(_ConfusionMatrixMetric["RecallResult"]):
    """Recall is a metric used to evaluate object detection models. It is the ratio of
    true positive detections to the total number of ground truth instances. We calculate
    it at different IoU thresholds.

    In simple terms, Recall is a measure of a model's completeness, calculated as:

    `Recall = TP / (TP + FN)`

    Here, `TP` is the number of true positives (correct detections), and `FN` is the
    number of false negatives (missed detections).

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import Recall
        >>> predictions = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0]),
        ...     confidence=np.array([0.9])
        ... )
        >>> targets = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0])
        ... )
        >>> recall_metric = Recall()
        >>> recall_result = recall_metric.update(predictions, targets).compute()
        >>> round(float(recall_result.recall_at_50), 2)
        1.0

        ```

        A class that only ever appears in the predictions (for example a detection
        on a background image with no ground truth) has no instances to recall, so
        it is tracked with a recall of `0.0` rather than dropped. This keeps the
        tracked class set aligned with Precision and F1Score for the same input:

        ```pycon
        >>> predictions = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10], [100, 0, 110, 10]]),
        ...     class_id=np.array([0, 1]),  # class 1 has no ground-truth instance
        ...     confidence=np.array([0.9, 0.8])
        ... )
        >>> targets = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0])
        ... )
        >>> recall_result = Recall().update(predictions, targets).compute()
        >>> recall_result.matched_classes.tolist()
        [0, 1]
        >>> round(float(recall_result.recall_per_class[0][0]), 2)  # matched class 0
        1.0
        >>> round(float(recall_result.recall_per_class[1][0]), 2)  # prediction-only
        0.0

        ```

    ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/recall_plot_example.png
    ){ align=center width="800" }
    """

    _metric_name = "Recall"

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        averaging_method: AveragingMethod = AveragingMethod.WEIGHTED,
    ):
        """Initialize the Recall metric.

        Args:
            metric_target: The type of detection data to use.
            averaging_method: The averaging method used to compute the
                recall. Determines how the recall is aggregated across classes.
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
    ) -> Recall:
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

    def compute(self) -> RecallResult:
        """Calculate the recall metric based on the stored predictions and ground-truth
        data, at different IoU thresholds.

        Returns:
            The recall metric result.
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
        """Return the recall implied by each entry of the confusion matrix."""
        return self._compute_recall(confusion_matrix)

    def _build_result(
        self,
        scores: npt.NDArray[np.float64],
        per_class_scores: npt.NDArray[np.float64],
        iou_thresholds: npt.NDArray[np.float32],
        matched_classes: npt.NDArray[np.int32],
    ) -> RecallResult:
        """Wrap the computed recall scores in a `RecallResult`."""
        return RecallResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            recall_scores=scores,
            recall_per_class=per_class_scores,
            iou_thresholds=iou_thresholds,
            matched_classes=matched_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

    @staticmethod
    def _compute_recall(
        confusion_matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Broadcastable function, computing the recall from the confusion matrix.

        Args:
            confusion_matrix: shape (N, ..., 3), where the last dimension
                contains the true positives, false positives, and false negatives.

        Returns:
            shape (N, ...), containing the recall for each element.
        """
        _validate_confusion_matrix(confusion_matrix)
        true_positives = confusion_matrix[..., 0]
        false_negatives = confusion_matrix[..., 2]

        denominator = true_positives + false_negatives
        recall = np.divide(
            true_positives,
            denominator,
            out=np.zeros_like(true_positives),
            where=denominator != 0,
        )

        result_recall: npt.NDArray[np.float64] = recall
        return result_recall


@dataclass
class RecallResult:
    """The results of the recall metric calculation.

    Defaults to `0` if no detections or targets were provided.

    Attributes:
        metric_target: the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        averaging_method: the averaging method used to compute the
            recall. Determines how the recall is aggregated across classes.
        recall_at_50: the recall at IoU threshold of `0.5`.
        recall_at_75: the recall at IoU threshold of `0.75`.
        recall_scores: the recall scores at each IoU threshold.
            Shape: `(num_iou_thresholds,)`
        recall_per_class: the recall scores per class and IoU threshold.
            Shape: `(num_classes, num_iou_thresholds)`
        iou_thresholds: the IoU thresholds used in the calculations.
        matched_classes: the class IDs present in either predictions or ground
            truth. Corresponds to the rows of `recall_per_class`. Classes that
            appear only in predictions (no ground-truth instances) are included;
            their per-threshold recall values will be `0.0`.
        small_objects: the Recall metric results
            for small objects (area < 32²).
        medium_objects: the Recall metric results
            for medium objects (32² ≤ area < 96²).
        large_objects: the Recall metric results
            for large objects (area ≥ 96²).
    """

    metric_target: MetricTarget
    averaging_method: AveragingMethod

    @property
    def recall_at_50(self) -> float:
        return float(self.recall_scores[0])

    @property
    def recall_at_75(self) -> float:
        return float(self.recall_scores[5])

    recall_scores: npt.NDArray[np.float64]
    recall_per_class: npt.NDArray[np.float64]
    iou_thresholds: npt.NDArray[np.float32]
    matched_classes: npt.NDArray[np.int32]

    small_objects: RecallResult | None
    medium_objects: RecallResult | None
    large_objects: RecallResult | None

    def _result_view(self) -> _ResultView:
        """Describe this result in the form the shared renderers consume."""
        return _ResultView(
            class_name=self.__class__.__name__,
            labels=_RECALL_LABELS,
            metric_target=self.metric_target,
            averaging_method=self.averaging_method,
            score_at_50=self.recall_at_50,
            score_at_75=self.recall_at_75,
            scores=self.recall_scores,
            per_class_scores=self.recall_per_class,
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
            >>> from supervision.metrics import Recall
            >>> predictions = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0]),
            ...     confidence=np.array([0.9])
            ... )
            >>> targets = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0])
            ... )
            >>> recall_metric = Recall()
            >>> recall_result = recall_metric.update(predictions, targets).compute()
            >>> print(recall_result)  # doctest: +ELLIPSIS
            RecallResult:
            Metric target:    MetricTarget.BOXES
            Averaging method: AveragingMethod.WEIGHTED
            R @ 50:     1.0000
            R @ 75:     1.0000
            R @ thresh: [1. ... 1.]
            IoU thresh: [0.5  0.55 ... 0.95]
            Recall per class:
              0: [1. ... 1.]
            ...
            Medium objects:
              RecallResult:
              Metric target:    MetricTarget.BOXES
              Averaging method: AveragingMethod.WEIGHTED
              R @ 50:     0.0000
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
        """Plot the recall results.

        ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/recall_plot_example.png
        ){ align=center width="800" }
        """
        _plot_result(self._result_view())
