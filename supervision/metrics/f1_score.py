from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch, mask_iou_batch
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import AveragingMethod, Metric, MetricTarget
from supervision.metrics.utils.object_size import (
    ObjectSizeCategory,
    get_detection_size_category,
)
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd


class F1Score(Metric):
    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        averaging_method: AveragingMethod = AveragingMethod.WEIGHTED,
    ):
        self._metric_target = metric_target
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            raise NotImplementedError(
                "F1 score is not implemented for oriented bounding boxes."
            )

        self._metric_target = metric_target
        self.averaging_method = averaging_method
        self._predictions_list: List[Detections] = []
        self._targets_list: List[Detections] = []

    def reset(self) -> None:
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: Union[Detections, List[Detections]],
        targets: Union[Detections, List[Detections]],
    ) -> F1Score:
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
        result = self._compute(self._predictions_list, self._targets_list)

        small_predictions, small_targets = self._filter_predictions_and_targets_by_size(
            self._predictions_list, self._targets_list, ObjectSizeCategory.SMALL
        )
        result.small_objects = self._compute(small_predictions, small_targets)

        medium_predictions, medium_targets = (
            self._filter_predictions_and_targets_by_size(
                self._predictions_list, self._targets_list, ObjectSizeCategory.MEDIUM
            )
        )
        result.medium_objects = self._compute(medium_predictions, medium_targets)

        large_predictions, large_targets = self._filter_predictions_and_targets_by_size(
            self._predictions_list, self._targets_list, ObjectSizeCategory.LARGE
        )
        result.large_objects = self._compute(large_predictions, large_targets)

        return result

    def _compute(
        self, predictions_list: List[Detections], targets_list: List[Detections]
    ) -> F1ScoreResult:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        stats = []

        for predictions, targets in zip(predictions_list, targets_list):
            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)

            if len(targets) > 0:
                if len(predictions) == 0:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=int),
                            targets.class_id,
                        )
                    )

                else:
                    if self._metric_target == MetricTarget.BOXES:
                        iou = box_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.MASKS:
                        iou = mask_iou_batch(target_contents, prediction_contents)
                    else:
                        raise NotImplementedError(
                            "Unsupported metric target for IoU calculation"
                        )

                    matches = self._match_detection_batch(
                        predictions.class_id, targets.class_id, iou, iou_thresholds
                    )
                    stats.append(
                        (
                            matches,
                            predictions.confidence,
                            predictions.class_id,
                            targets.class_id,
                        )
                    )

        if not stats:
            return F1ScoreResult(
                metric_target=self._metric_target,
                averaging_method=self.averaging_method,
                f1_scores=np.zeros(iou_thresholds.shape[0]),
                f1_per_class=np.zeros((0, iou_thresholds.shape[0])),
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
                small_objects=None,
                medium_objects=None,
                large_objects=None,
            )

        concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
        f1_scores, f1_per_class, unique_classes = self._compute_f1_for_classes(
            *concatenated_stats
        )

        return F1ScoreResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            f1_scores=f1_scores,
            f1_per_class=f1_per_class,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

    def _compute_f1_for_classes(
        self,
        matches: np.ndarray,
        prediction_confidence: np.ndarray,
        prediction_class_ids: np.ndarray,
        true_class_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]
        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)

        # Shape: PxTh,P,C,C -> CxThx3
        confusion_matrix = self._compute_confusion_matrix(
            matches, prediction_class_ids, unique_classes, class_counts
        )

        # Shape: CxThx3 -> CxTh
        f1_per_class = self._compute_f1(confusion_matrix)

        # Shape: CxTh -> Th
        if self.averaging_method == AveragingMethod.MACRO:
            f1_scores = np.mean(f1_per_class, axis=0)
        elif self.averaging_method == AveragingMethod.MICRO:
            confusion_matrix_merged = confusion_matrix.sum(0)
            f1_scores = self._compute_f1(confusion_matrix_merged)
        elif self.averaging_method == AveragingMethod.WEIGHTED:
            class_counts = class_counts.astype(np.float32)
            f1_scores = np.average(f1_per_class, axis=0, weights=class_counts)

        return f1_scores, f1_per_class, unique_classes

    @staticmethod
    def _match_detection_batch(
        predictions_classes: np.ndarray,
        target_classes: np.ndarray,
        iou: np.ndarray,
        iou_thresholds: np.ndarray,
    ) -> np.ndarray:
        num_predictions, num_iou_levels = (
            predictions_classes.shape[0],
            iou_thresholds.shape[0],
        )
        correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
        correct_class = target_classes[:, None] == predictions_classes

        for i, iou_level in enumerate(iou_thresholds):
            matched_indices = np.where((iou >= iou_level) & correct_class)

            if matched_indices[0].shape[0]:
                combined_indices = np.stack(matched_indices, axis=1)
                iou_values = iou[matched_indices][:, None]
                matches = np.hstack([combined_indices, iou_values])

                if matched_indices[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                correct[matches[:, 1].astype(int), i] = True

        return correct

    @staticmethod
    def _compute_confusion_matrix(
        sorted_matches: np.ndarray,
        sorted_prediction_class_ids: np.ndarray,
        unique_classes: np.ndarray,
        class_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the confusion matrix for each class and IoU threshold.

        Assumes the matches and prediction_class_ids are sorted by confidence
        in descending order.

        Arguments:
            sorted_matches: np.ndarray, bool, shape (P, Th), that is True
                if the prediction is a true positive at the given IoU threshold.
            sorted_prediction_class_ids: np.ndarray, int, shape (P,), containing
                the class id for each prediction.
            unique_classes: np.ndarray, int, shape (C,), containing the unique
                class ids.
            class_counts: np.ndarray, int, shape (C,), containing the number
                of true instances for each class.

        Returns:
            np.ndarray, shape (C, Th, 3), containing the true positives, false
                positives, and false negatives for each class and IoU threshold.
        """

        num_thresholds = sorted_matches.shape[1]
        num_classes = unique_classes.shape[0]

        confusion_matrix = np.zeros((num_classes, num_thresholds, 3))
        for class_idx, class_id in enumerate(unique_classes):
            is_class = sorted_prediction_class_ids == class_id
            num_true = class_counts[class_idx]
            num_predictions = is_class.sum()

            if num_predictions == 0:
                true_positives = np.zeros(num_thresholds)
                false_positives = np.zeros(num_thresholds)
                false_negatives = np.full(num_thresholds, num_true)
            elif num_true == 0:
                true_positives = np.zeros(num_thresholds)
                false_positives = np.full(num_thresholds, num_predictions)
                false_negatives = np.zeros(num_thresholds)
            else:
                true_positives = sorted_matches[is_class].sum(0)
                false_positives = (1 - sorted_matches[is_class]).sum(0)
                false_negatives = num_true - true_positives
            confusion_matrix[class_idx] = np.stack(
                [true_positives, false_positives, false_negatives], axis=1
            )

        return confusion_matrix

    @staticmethod
    def _compute_f1(confusion_matrix: np.ndarray) -> np.ndarray:
        """
        Broadcastable function, computing the F1 score from the confusion matrix.

        Arguments:
            confusion_matrix: np.ndarray, shape (N, ..., 3), where the last dimension
                contains the true positives, false positives, and false negatives.

        Returns:
            np.ndarray, shape (N, ...), containing the F1 score for each element.
        """
        if not confusion_matrix.shape[-1] == 3:
            raise ValueError(
                f"Confusion matrix must have shape (..., 3), got "
                f"{confusion_matrix.shape}"
            )
        true_positives = confusion_matrix[..., 0]
        false_positives = confusion_matrix[..., 1]
        false_negatives = confusion_matrix[..., 2]

        # Alternate formula, avoids multiple zero division checks
        denominator = 2 * true_positives + false_positives + false_negatives
        f1_score = np.where(denominator == 0, 0, 2 * true_positives / denominator)

        return f1_score

    def _detections_content(self, detections: Detections) -> np.ndarray:
        """Return boxes, masks or oriented bounding boxes from detections."""
        if self._metric_target == MetricTarget.BOXES:
            return detections.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return (
                detections.mask
                if detections.mask is not None
                else np.empty((0, 0, 0), dtype=bool)
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            if obb := detections.data.get(ORIENTED_BOX_COORDINATES):
                return np.ndarray(obb, dtype=np.float32)
            return np.empty((0, 8), dtype=np.float32)
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _filter_detections_by_size(
        self, detections: Detections, size_category: ObjectSizeCategory
    ) -> Detections:
        """Return a copy of detections with contents filtered by object size."""
        new_detections = deepcopy(detections)
        if detections.is_empty() or size_category == ObjectSizeCategory.ANY:
            return new_detections

        sizes = get_detection_size_category(new_detections, self._metric_target)
        size_mask = sizes == size_category.value

        new_detections.xyxy = new_detections.xyxy[size_mask]
        if new_detections.mask is not None:
            new_detections.mask = new_detections.mask[size_mask]
        if new_detections.class_id is not None:
            new_detections.class_id = new_detections.class_id[size_mask]
        if new_detections.confidence is not None:
            new_detections.confidence = new_detections.confidence[size_mask]
        if new_detections.tracker_id is not None:
            new_detections.tracker_id = new_detections.tracker_id[size_mask]
        if new_detections.data is not None:
            for key, value in new_detections.data.items():
                new_detections.data[key] = np.array(value)[size_mask]

        return new_detections

    def _filter_predictions_and_targets_by_size(
        self,
        predictions_list: List[Detections],
        targets_list: List[Detections],
        size_category: ObjectSizeCategory,
    ) -> Tuple[List[Detections], List[Detections]]:
        """
        Filter predictions and targets by object size category.
        """
        new_predictions_list = []
        new_targets_list = []
        for predictions, targets in zip(predictions_list, targets_list):
            new_predictions_list.append(
                self._filter_detections_by_size(predictions, size_category)
            )
            new_targets_list.append(
                self._filter_detections_by_size(targets, size_category)
            )
        return new_predictions_list, new_targets_list


@dataclass
class F1ScoreResult:
    """
    The results of the F1 score metric calculation.

    Defaults to `0` if no detections or targets were provided.
    Provides a custom `__str__` method for pretty printing.

    Attributes:
        metric_target (MetricTarget): the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        averaging_method (AveragingMethod): the averaging method used to compute the
            F1 scores. Determines how the F1 scores are aggregated across classes.
        f1_50 (float): the F1 score at IoU threshold of `0.5`.
        f1_75 (float): the F1 score at IoU threshold of `0.75`.
        f1_scores (np.ndarray): the F1 scores at each IoU threshold.
            Shape: `(num_iou_thresholds,)`
        f1_per_class (np.ndarray): the F1 scores per class and IoU threshold.
            Shape: `(num_target_classes, num_iou_thresholds)`
        iou_thresholds (np.ndarray): the IoU thresholds used in the calculations.
        matched_classes (np.ndarray): the class IDs of all matched classes.
            Corresponds to the rows of `f1_per_class`.
        small_objects (Optional[F1ScoreResult]): the F1 metric results
            for small objects.
        medium_objects (Optional[F1ScoreResult]): the F1 metric results
            for medium objects.
        large_objects (Optional[F1ScoreResult]): the F1 metric results
            for large objects.
    """

    metric_target: MetricTarget
    averaging_method: AveragingMethod

    @property
    def f1_50(self) -> float:
        return self.f1_scores[0]

    @property
    def f1_75(self) -> float:
        return self.f1_scores[5]

    f1_scores: np.ndarray
    f1_per_class: np.ndarray
    iou_thresholds: np.ndarray
    matched_classes: np.ndarray

    small_objects: Optional[F1ScoreResult]
    medium_objects: Optional[F1ScoreResult]
    large_objects: Optional[F1ScoreResult]

    def __str__(self) -> str:
        """
        Format as a pretty string.

        Example:
            ```python
            print(f1_result)
            ```
        """
        out_str = (
            f"{self.__class__.__name__}:\n"
            f"Metric target: {self.metric_target}\n"
            f"Averaging method: {self.averaging_method}\n"
            f"F1 @ 50:     {self.f1_50:.4f}\n"
            f"F1 @ 75:     {self.f1_75:.4f}\n"
            f"F1 @ thresh: {self.f1_scores}\n"
            f"IoU thresh:  {self.iou_thresholds}\n"
            f"F1 per class:\n"
        )
        if self.f1_per_class.size == 0:
            out_str += "  No results\n"
        for class_id, f1_of_class in zip(self.matched_classes, self.f1_per_class):
            out_str += f"  {class_id}: {f1_of_class}\n"

        indent = "  "
        if self.small_objects is not None:
            indented = indent + str(self.small_objects).replace("\n", f"\n{indent}")
            out_str += f"\nSmall objects:\n{indented}"
        if self.medium_objects is not None:
            indented = indent + str(self.medium_objects).replace("\n", f"\n{indent}")
            out_str += f"\nMedium objects:\n{indented}"
        if self.large_objects is not None:
            indented = indent + str(self.large_objects).replace("\n", f"\n{indent}")
            out_str += f"\nLarge objects:\n{indented}"

        return out_str

    def to_pandas(self) -> "pd.DataFrame":
        """
        Convert the result to a pandas DataFrame.

        Returns:
            (pd.DataFrame): The result as a DataFrame.
        """
        ensure_pandas_installed()
        import pandas as pd

        pandas_data = {
            "F1@50": self.f1_50,
            "F1@75": self.f1_75,
        }

        if self.small_objects is not None:
            small_objects_df = self.small_objects.to_pandas()
            for key, value in small_objects_df.items():
                pandas_data[f"small_objects_{key}"] = value
        if self.medium_objects is not None:
            medium_objects_df = self.medium_objects.to_pandas()
            for key, value in medium_objects_df.items():
                pandas_data[f"medium_objects_{key}"] = value
        if self.large_objects is not None:
            large_objects_df = self.large_objects.to_pandas()
            for key, value in large_objects_df.items():
                pandas_data[f"large_objects_{key}"] = value

        return pd.DataFrame(pandas_data, index=[0])

    def plot(self):
        """
        Plot the F1 results.
        """

        labels = ["F1@50", "F1@75"]
        values = [self.f1_50, self.f1_75]
        colors = [LEGACY_COLOR_PALETTE[0]] * 2

        if self.small_objects is not None:
            small_objects = self.small_objects
            labels += ["Small: F1@50", "Small: F1@75"]
            values += [small_objects.f1_50, small_objects.f1_75]
            colors += [LEGACY_COLOR_PALETTE[3]] * 2

        if self.medium_objects is not None:
            medium_objects = self.medium_objects
            labels += ["Medium: F1@50", "Medium: F1@75"]
            values += [medium_objects.f1_50, medium_objects.f1_75]
            colors += [LEGACY_COLOR_PALETTE[2]] * 2

        if self.large_objects is not None:
            large_objects = self.large_objects
            labels += ["Large: F1@50", "Large: F1@75"]
            values += [large_objects.f1_50, large_objects.f1_75]
            colors += [LEGACY_COLOR_PALETTE[4]] * 2

        plt.rcParams["font.family"] = "monospace"

        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value", fontweight="bold")
        title = (
            f"F1 Score, by Object Size"
            f"\n(target: {self.metric_target.value},"
            f" averaging: {self.averaging_method.value})"
        )
        ax.set_title(title, fontweight="bold")

        x_positions = range(len(labels))
        bars = ax.bar(x_positions, values, color=colors, align="center")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        for bar in bars:
            y_value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_value + 0.02,
                f"{y_value:.2f}",
                ha="center",
                va="bottom",
            )

        plt.rcParams["font.family"] = "sans-serif"

        plt.tight_layout()
        plt.show()
