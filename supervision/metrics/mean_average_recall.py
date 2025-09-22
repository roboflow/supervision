from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import Metric, MetricTarget
from supervision.metrics.utils.object_size import (
    ObjectSizeCategory,
    get_detection_size_category,
)
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd


class MeanAverageRecall(Metric):
    """
    Mean Average Recall (mAR) measures how well the model detects
    and retrieves relevant objects by averaging recall over multiple
    IoU thresholds, classes and detection limits.

    Intuitively, while Recall measures the ability to find all relevant
    objects, mAR narrows down how many detections are considered for each
    class. For example, mAR @ 100 considers the top 100 highest confidence
    detections for each class. mAR @ 1 considers only the highest
    confidence detection for each class.

    Example:
        ```python
        import supervision as sv
        from supervision.metrics import MeanAverageRecall

        predictions = sv.Detections(...)
        targets = sv.Detections(...)

        map_metric = MeanAverageRecall()
        map_result = map_metric.update(predictions, targets).compute()

        print(mar_results.mar_at_100)
        # 0.5241

        print(mar_results)
        # MeanAverageRecallResult:
        # Metric target:    MetricTarget.BOXES
        # mAR @ 1:    0.1362
        # mAR @ 10:   0.4239
        # mAR @ 100:  0.5241
        # max detections: [1  10 100]
        # IoU thresh:     [0.5  0.55  0.6  ...]
        # mAR per class:
        # 0: [0.78571  0.78571  0.78571  ...]
        # ...
        # Small objects: ...
        # Medium objects: ...
        # Large objects: ...

        mar_results.plot()
        ```

    ![example_plot](\
        https://media.roboflow.com/supervision-docs/metrics/mAR_plot_example.png\
        ){ align=center width="800" }
    """

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
    ):
        """
        Initialize the Mean Average Recall metric.

        Args:
            metric_target (MetricTarget): The type of detection data to use.
        """
        self._metric_target = metric_target

        self._predictions_list: list[Detections] = []
        self._targets_list: list[Detections] = []

        self.max_detections = np.array([1, 10, 100])

    def reset(self) -> None:
        """
        Reset the metric to its initial state, clearing all stored data.
        """
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: Detections | list[Detections],
        targets: Detections | list[Detections],
    ) -> MeanAverageRecall:
        """
        Add new predictions and targets to the metric, but do not compute the result.

        Args:
            predictions (Union[Detections, List[Detections]]): The predicted detections.
            targets (Union[Detections, List[Detections]]): The target detections.

        Returns:
            (Recall): The updated metric instance.
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

    def compute(self) -> MeanAverageRecallResult:
        """
        Calculate the Mean Average Recall metric based on the stored predictions
        and ground-truth, at different IoU thresholds and maximum detection counts.

        Returns:
            (MeanAverageRecallResult): The Mean Average Recall metric result.
        """
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
        self, predictions_list: list[Detections], targets_list: list[Detections]
    ) -> MeanAverageRecallResult:
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
                    elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                        iou = oriented_box_iou_batch(
                            target_contents, prediction_contents
                        )
                    else:
                        raise ValueError(
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
            return MeanAverageRecallResult(
                metric_target=self._metric_target,
                recall_scores=np.zeros(iou_thresholds.shape[0]),
                recall_per_class=np.zeros((0, iou_thresholds.shape[0])),
                max_detections=self.max_detections,
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
                small_objects=None,
                medium_objects=None,
                large_objects=None,
            )

        recall_scores_per_k, recall_per_class, unique_classes = (
            self._compute_average_recall_for_classes(stats)
        )

        return MeanAverageRecallResult(
            metric_target=self._metric_target,
            recall_scores=recall_scores_per_k,
            recall_per_class=recall_per_class,
            max_detections=self.max_detections,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

    def _compute_average_recall_for_classes(
        self,
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        recalls_at_k = []

        for max_detections in self.max_detections:
            filtered_stats = []
            for matches, confidence, class_id, true_class_id in stats:
                sorted_indices = np.argsort(-confidence)[:max_detections]
                filtered_stats.append(
                    (
                        matches[sorted_indices],
                        class_id[sorted_indices],
                        true_class_id,
                    )
                )
            concatenated_stats = [
                np.concatenate(items, 0) for items in zip(*filtered_stats)
            ]

            filtered_matches, prediction_class_ids, true_class_ids = concatenated_stats
            unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)

            # Shape: PxTh,P,C,C -> CxThx3
            confusion_matrix = self._compute_confusion_matrix(
                filtered_matches,
                prediction_class_ids,
                unique_classes,
                class_counts,
            )

            # Shape: CxThx3 -> CxTh
            recall_per_class = self._compute_recall(confusion_matrix)
            recalls_at_k.append(recall_per_class)

        # Shape: KxCxTh -> KxC
        recalls_at_k = np.array(recalls_at_k)
        average_recall_per_class = np.mean(recalls_at_k, axis=2)

        # Shape: KxC -> K
        recall_scores = np.mean(average_recall_per_class, axis=1)

        return recall_scores, recall_per_class, unique_classes

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
        max_detections: int | None = None,
    ) -> np.ndarray:
        """
        Compute the confusion matrix for each class and IoU threshold.

        Assumes the matches and prediction_class_ids are sorted by confidence
        in descending order.

        Args:
            sorted_matches: np.ndarray, bool, shape (P, Th), that is True
                if the prediction is a true positive at the given IoU threshold.
            sorted_prediction_class_ids: np.ndarray, int, shape (P,), containing
                the class id for each prediction.
            unique_classes: np.ndarray, int, shape (C,), containing the unique
                class ids.
            class_counts: np.ndarray, int, shape (C,), containing the number
                of true instances for each class.
            max_detections: Optional[int], the maximum number of detections to
                consider for each class. Extra detections are considered false
                positives. By default, all detections are considered.

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
                limited_matches = sorted_matches[is_class][slice(max_detections)]
                true_positives = limited_matches.sum(0)

                false_positives = (1 - limited_matches).sum(0)
                false_negatives = num_true - true_positives
                false_negatives = num_true - true_positives
            confusion_matrix[class_idx] = np.stack(
                [true_positives, false_positives, false_negatives], axis=1
            )

        return confusion_matrix

    @staticmethod
    def _compute_recall(confusion_matrix: np.ndarray) -> np.ndarray:
        """
        Broadcastable function, computing the recall from the confusion matrix.

        Arguments:
            confusion_matrix: np.ndarray, shape (N, ..., 3), where the last dimension
                contains the true positives, false positives, and false negatives.

        Returns:
            np.ndarray, shape (N, ...), containing the recall for each element.
        """
        if not confusion_matrix.shape[-1] == 3:
            raise ValueError(
                f"Confusion matrix must have shape (..., 3), got "
                f"{confusion_matrix.shape}"
            )
        true_positives = confusion_matrix[..., 0]
        false_negatives = confusion_matrix[..., 2]

        denominator = true_positives + false_negatives
        recall = np.where(denominator == 0, 0, true_positives / denominator)

        return recall

    def _detections_content(self, detections: Detections) -> np.ndarray:
        """Return boxes, masks or oriented bounding boxes from detections."""
        if self._metric_target == MetricTarget.BOXES:
            return detections.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return (
                detections.mask
                if detections.mask is not None
                else self._make_empty_content()
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = detections.data.get(ORIENTED_BOX_COORDINATES)
            if obb is not None and len(obb) > 0:
                return np.array(obb, dtype=np.float32)
            return self._make_empty_content()
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _make_empty_content(self) -> np.ndarray:
        if self._metric_target == MetricTarget.BOXES:
            return np.empty((0, 4), dtype=np.float32)
        if self._metric_target == MetricTarget.MASKS:
            return np.empty((0, 0, 0), dtype=bool)
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            return np.empty((0, 4, 2), dtype=np.float32)
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
        predictions_list: list[Detections],
        targets_list: list[Detections],
        size_category: ObjectSizeCategory,
    ) -> tuple[list[Detections], list[Detections]]:
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
class MeanAverageRecallResult:
    # """
    # The results of the recall metric calculation.

    # Defaults to `0` if no detections or targets were provided.

    # Attributes:
    #     metric_target (MetricTarget): the type of data used for the metric -
    #         boxes, masks or oriented bounding boxes.
    #     averaging_method (AveragingMethod): the averaging method used to compute the
    #         recall. Determines how the recall is aggregated across classes.
    #     recall_at_50 (float): the recall at IoU threshold of `0.5`.
    #     recall_at_75 (float): the recall at IoU threshold of `0.75`.
    #     recall_scores (np.ndarray): the recall scores at each IoU threshold.
    #         Shape: `(num_iou_thresholds,)`
    #     recall_per_class (np.ndarray): the recall scores per class and IoU threshold.
    #         Shape: `(num_target_classes, num_iou_thresholds)`
    #     iou_thresholds (np.ndarray): the IoU thresholds used in the calculations.
    #     matched_classes (np.ndarray): the class IDs of all matched classes.
    #         Corresponds to the rows of `recall_per_class`.
    #     small_objects (Optional[RecallResult]): the Recall metric results
    #         for small objects.
    #     medium_objects (Optional[RecallResult]): the Recall metric results
    #         for medium objects.
    #     large_objects (Optional[RecallResult]): the Recall metric results
    #         for large objects.
    # """
    """
    The results of the Mean Average Recall metric calculation.

    Defaults to `0` if no detections or targets were provided.

    Attributes:
        metric_target (MetricTarget): the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        mAR_at_1 (float): the Mean Average Recall, when considering only the top
            highest confidence detection for each class.
        mAR_at_10 (float): the Mean Average Recall, when considering top 10
            highest confidence detections for each class.
        mAR_at_100 (float): the Mean Average Recall, when considering top 100
            highest confidence detections for each class.
        recall_per_class (np.ndarray): the recall scores per class and IoU threshold.
            Shape: `(num_target_classes, num_iou_thresholds)`
        max_detections (np.ndarray): the array with maximum number of detections
            considered.
        iou_thresholds (np.ndarray): the IoU thresholds used in the calculations.
        matched_classes (np.ndarray): the class IDs of all matched classes.
            Corresponds to the rows of `recall_per_class`.
        small_objects (Optional[MeanAverageRecallResult]): the Mean Average Recall
            metric results for small objects (area < 32²).
        medium_objects (Optional[MeanAverageRecallResult]): the Mean Average Recall
            metric results for medium objects (32² ≤ area < 96²).
        large_objects (Optional[MeanAverageRecallResult]): the Mean Average Recall
            metric results for large objects (area ≥ 96²).
    """

    metric_target: MetricTarget

    @property
    def mAR_at_1(self) -> float:
        return self.recall_scores[0]

    @property
    def mAR_at_10(self) -> float:
        return self.recall_scores[1]

    @property
    def mAR_at_100(self) -> float:
        return self.recall_scores[2]

    recall_scores: np.ndarray
    recall_per_class: np.ndarray
    max_detections: np.ndarray
    iou_thresholds: np.ndarray
    matched_classes: np.ndarray

    small_objects: MeanAverageRecallResult | None
    medium_objects: MeanAverageRecallResult | None
    large_objects: MeanAverageRecallResult | None

    def __str__(self) -> str:
        """
        Format as a pretty string.

        Example:
            ```python
            # MeanAverageRecallResult:
            # Metric target:    MetricTarget.BOXES
            # mAR @ 1:    0.1362
            # mAR @ 10:   0.4239
            # mAR @ 100:  0.5241
            # max detections: [1  10 100]
            # IoU thresh:     [0.5  0.55  0.6  ...]
            # mAR per class:
            # 0: [0.78571  0.78571  0.78571  ...]
            # ...
            # Small objects: ...
            # Medium objects: ...
            # Large objects: ...
            ```
        """
        out_str = (
            f"{self.__class__.__name__}:\n"
            f"Metric target:  {self.metric_target}\n"
            f"mAR @ 1:    {self.mAR_at_1:.4f}\n"
            f"mAR @ 10:   {self.mAR_at_10:.4f}\n"
            f"mAR @ 100:  {self.mAR_at_100:.4f}\n"
            f"max detections: {self.max_detections}\n"
            f"IoU thresh:     {self.iou_thresholds}\n"
            f"mAR per class:\n"
        )
        if self.recall_per_class.size == 0:
            out_str += "  No results\n"
        for class_id, recall_of_class in zip(
            self.matched_classes, self.recall_per_class
        ):
            out_str += f"  {class_id}: {recall_of_class}\n"

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

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the result to a pandas DataFrame.

        Returns:
            (pd.DataFrame): The result as a DataFrame.
        """
        ensure_pandas_installed()
        import pandas as pd

        pandas_data = {
            "mAR @ 1": self.mAR_at_1,
            "mAR @ 10": self.mAR_at_10,
            "mAR @ 100": self.mAR_at_100,
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
        Plot the Mean Average Recall results.

        ![example_plot](\
            https://media.roboflow.com/supervision-docs/metrics/mAR_plot_example.png\
            ){ align=center width="800" }
        """
        labels = ["mAR @ 1", "mAR @ 10", "mAR @ 100"]
        values = [self.mAR_at_1, self.mAR_at_10, self.mAR_at_100]
        colors = [LEGACY_COLOR_PALETTE[0]] * 3

        if self.small_objects is not None:
            small_objects = self.small_objects
            labels += ["Small: mAR @ 1", "Small: mAR @ 10", "Small: mAR @ 100"]
            values += [
                small_objects.mAR_at_1,
                small_objects.mAR_at_10,
                small_objects.mAR_at_100,
            ]
            colors += [LEGACY_COLOR_PALETTE[3]] * 3

        if self.medium_objects is not None:
            medium_objects = self.medium_objects
            labels += ["Medium: mAR @ 1", "Medium: mAR @ 10", "Medium: mAR @ 100"]
            values += [
                medium_objects.mAR_at_1,
                medium_objects.mAR_at_10,
                medium_objects.mAR_at_100,
            ]
            colors += [LEGACY_COLOR_PALETTE[2]] * 3

        if self.large_objects is not None:
            large_objects = self.large_objects
            labels += ["Large: mAR @ 1", "Large: mAR @ 10", "Large: mAR @ 100"]
            values += [
                large_objects.mAR_at_1,
                large_objects.mAR_at_10,
                large_objects.mAR_at_100,
            ]
            colors += [LEGACY_COLOR_PALETTE[4]] * 3

        plt.rcParams["font.family"] = "monospace"

        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value", fontweight="bold")
        title = (
            f"Mean Average Recall, by Object Size\n(target: {self.metric_target.value})"
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
