from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch, mask_iou_batch
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import Metric, MetricTarget
from supervision.metrics.utils.object_size import (
    ObjectSizeCategory,
    get_detection_size_category,
)
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
    ):
        """
        Initialize the Mean Average Precision metric.

        Args:
            metric_target (MetricTarget): The type of detection data to use.
            class_agnostic (bool): Whether to treat all data as a single class.
        """
        self._metric_target = metric_target
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            raise NotImplementedError(
                "Mean Average Precision is not implemented for oriented bounding boxes."
            )

        self._class_agnostic = class_agnostic

        self._predictions_list: List[Detections] = []
        self._targets_list: List[Detections] = []

    def reset(self) -> None:
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: Union[Detections, List[Detections]],
        targets: Union[Detections, List[Detections]],
    ) -> MeanAveragePrecision:
        """
        Add new predictions and targets to the metric, but do not compute the result.

        Args:
            predictions (Union[Detections, List[Detections]]): The predicted detections.
            targets (Union[Detections, List[Detections]]): The ground-truth detections.

        Returns:
            (MeanAveragePrecision): The updated metric instance.
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

    def compute(
        self,
    ) -> MeanAveragePrecisionResult:
        """
        Calculate Mean Average Precision based on predicted and ground-truth
            detections at different threshold.

        Args:
            predictions (List[np.ndarray]): Each element of the list describes
                a single image and has `shape = (M, 6)` where `M` is
                the number of detected objects. Each row is expected to be
                in `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets (List[np.ndarray]): Each element of the list describes a single
                image and has `shape = (N, 5)` where `N` is the
                number of ground-truth objects. Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
        Returns:
            (MeanAveragePrecision): New instance of MeanAveragePrecision.

        Example:
            ```python
            import supervision as sv
            from supervision.metrics import MeanAveragePrecision

            predictions = sv.Detections(...)
            targets = sv.Detections(...)

            map_metric = MeanAveragePrecision()
            map_result = map_metric.update(predictions, targets).compute()

            print(map_result)
            print(map_result.map50_95)
            map_result.plot()
            ```
        """
        result = self._compute(self._predictions_list, self._targets_list)

        small_predictions = []
        small_targets = []
        for predictions, targets in zip(self._predictions_list, self._targets_list):
            small_predictions.append(
                self._filter_detections_by_size(predictions, ObjectSizeCategory.SMALL)
            )
            small_targets.append(
                self._filter_detections_by_size(targets, ObjectSizeCategory.SMALL)
            )
        result.small_objects = self._compute(small_predictions, small_targets)

        medium_predictions = []
        medium_targets = []
        for predictions, targets in zip(self._predictions_list, self._targets_list):
            medium_predictions.append(
                self._filter_detections_by_size(predictions, ObjectSizeCategory.MEDIUM)
            )
            medium_targets.append(
                self._filter_detections_by_size(targets, ObjectSizeCategory.MEDIUM)
            )
        result.medium_objects = self._compute(medium_predictions, medium_targets)

        large_predictions = []
        large_targets = []
        for predictions, targets in zip(self._predictions_list, self._targets_list):
            large_predictions.append(
                self._filter_detections_by_size(predictions, ObjectSizeCategory.LARGE)
            )
            large_targets.append(
                self._filter_detections_by_size(targets, ObjectSizeCategory.LARGE)
            )
        result.large_objects = self._compute(large_predictions, large_targets)

        return result

    def _compute(
        self,
        predictions_list: List[Detections],
        targets_list: List[Detections],
    ) -> MeanAveragePrecisionResult:
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

        # Compute average precisions if any matches exist
        if stats:
            concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
            average_precisions = self._average_precisions_per_class(*concatenated_stats)
            map50 = average_precisions[:, 0].mean()
            map75 = average_precisions[:, 5].mean()
            map50_95 = average_precisions.mean()
        else:
            map50, map75, map50_95 = 0, 0, 0
            average_precisions = np.empty((0, len(iou_thresholds)), dtype=np.float32)

        return MeanAveragePrecisionResult(
            iou_thresholds=iou_thresholds,
            map50_95=map50_95,
            map50=map50,
            map75=map75,
            per_class_ap50_95=average_precisions,
            metric_target=self._metric_target,
        )

    @staticmethod
    def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute the average precision using 101-point interpolation (COCO), given
            the recall and precision curves.

        Args:
            recall (np.ndarray): The recall curve.
            precision (np.ndarray): The precision curve.

        Returns:
            (float): Average precision.
        """
        extended_recall = np.concatenate(([0.0], recall, [1.0]))
        extended_precision = np.concatenate(([1.0], precision, [0.0]))
        max_accumulated_precision = np.flip(
            np.maximum.accumulate(np.flip(extended_precision))
        )
        interpolated_recall_levels = np.linspace(0, 1, 101)
        interpolated_precision = np.interp(
            interpolated_recall_levels, extended_recall, max_accumulated_precision
        )
        average_precision = np.trapz(interpolated_precision, interpolated_recall_levels)
        return average_precision

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
    def _average_precisions_per_class(
        matches: np.ndarray,
        prediction_confidence: np.ndarray,
        prediction_class_ids: np.ndarray,
        true_class_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

        Args:
            matches (np.ndarray): True positives.
            prediction_confidence (np.ndarray): Objectness value from 0-1.
            prediction_class_ids (np.ndarray): Predicted object classes.
            true_class_ids (np.ndarray): True object classes.
            eps (float, optional): Small value to prevent division by zero.

        Returns:
            (np.ndarray): Average precision for different IoU levels.
        """
        eps = 1e-16

        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]

        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
        num_classes = unique_classes.shape[0]

        average_precisions = np.zeros((num_classes, matches.shape[1]))

        for class_idx, class_id in enumerate(unique_classes):
            is_class = prediction_class_ids == class_id
            total_true = class_counts[class_idx]
            total_prediction = is_class.sum()

            if total_prediction == 0 or total_true == 0:
                continue

            false_positives = (1 - matches[is_class]).cumsum(0)
            true_positives = matches[is_class].cumsum(0)
            true_negatives = total_true - true_positives

            recall = true_positives / (true_positives + true_negatives + eps)
            precision = true_positives / (true_positives + false_positives)

            for iou_level_idx in range(matches.shape[1]):
                average_precisions[class_idx, iou_level_idx] = (
                    MeanAveragePrecision.compute_average_precision(
                        recall[:, iou_level_idx], precision[:, iou_level_idx]
                    )
                )

        return average_precisions

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
            if obb := detections.data.get(ORIENTED_BOX_COORDINATES):
                return np.ndarray(obb, dtype=np.float32)
            return self._make_empty_content()
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _make_empty_content(self) -> np.ndarray:
        if self._metric_target == MetricTarget.BOXES:
            return np.empty((0, 4), dtype=np.float32)
        if self._metric_target == MetricTarget.MASKS:
            return np.empty((0, 0, 0), dtype=bool)
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
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


@dataclass
class MeanAveragePrecisionResult:
    iou_thresholds: np.ndarray
    """Array of IoU thresholds used in the calculations"""
    map50_95: float
    """Mean Average Precision over IoU thresholds from 0.5 to 0.95"""

    map50: float
    """Mean Average Precision at IoU threshold of 0.5"""

    map75: float
    """Mean Average Precision at IoU threshold of 0.75"""

    per_class_ap50_95: np.ndarray
    """Average precision for each class at different IoU thresholds"""

    metric_target: MetricTarget
    """
    Defines the type of data used for the metric - boxes, masks or
    oriented bounding boxes.
    """

    small_objects: Optional[MeanAveragePrecisionResult] = None
    """Mean Average Precision results for small objects"""

    medium_objects: Optional[MeanAveragePrecisionResult] = None
    """Mean Average Precision results for medium objects"""

    large_objects: Optional[MeanAveragePrecisionResult] = None
    """Mean Average Precision results for large objects"""

    def __str__(self) -> str:
        """
        Format the mAP results as a pretty string.

        Example:
            ```python
            print(map_result)
            ```
        """

        out_str = (
            f"{self.__class__.__name__}:\n"
            f"iou_thresholds: {self.iou_thresholds}\n"
            f"map50_95:  {self.map50_95:.4f}\n"
            f"map50:     {self.map50:.4f}\n"
            f"map75:     {self.map75:.4f}\n"
            f"per_class_ap50_95:"
        )

        for class_id, ap in enumerate(self.per_class_ap50_95):
            out_str += f"\n  {class_id}:  {ap}"

        indent = "  "
        if self.small_objects is not None:
            indented_str = indent + str(self.small_objects).replace("\n", f"\n{indent}")
            out_str += f"\nSmall objects:\n{indented_str}"
        if self.medium_objects is not None:
            indented_str = indent + str(self.medium_objects).replace(
                "\n", f"\n{indent}"
            )
            out_str += f"\nMedium objects:\n{indented_str}"
        if self.large_objects is not None:
            indented_str = indent + str(self.large_objects).replace("\n", f"\n{indent}")
            out_str += f"\nLarge objects:\n{indented_str}"

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
            "mAP_50_95": self.map50_95,
            "mAP_50": self.map50,
            "mAP_75": self.map75,
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

        # Average precisions are currently not included in the DataFrame.

        return pd.DataFrame(
            pandas_data,
            index=[0],
        )

    def plot(self):
        """
        Plot the mAP results.
        """

        labels = ["mAP_50_95", "mAP_50", "mAP_75"]
        values = [self.map50_95, self.map50, self.map75]
        colors = [LEGACY_COLOR_PALETTE[0]] * 3

        if self.small_objects is not None:
            labels += [
                "small_objects_mAP_50_95",
                "small_objects_mAP_50",
                "small_objects_mAP_75",
            ]
            values += [
                self.small_objects.map50_95,
                self.small_objects.map50,
                self.small_objects.map75,
            ]
            colors += [LEGACY_COLOR_PALETTE[3]] * 3

        if self.medium_objects is not None:
            labels += [
                "medium_objects_mAP_50_95",
                "medium_objects_mAP_50",
                "medium_objects_mAP_75",
            ]
            values += [
                self.medium_objects.map50_95,
                self.medium_objects.map50,
                self.medium_objects.map75,
            ]
            colors += [LEGACY_COLOR_PALETTE[2]] * 3

        if self.large_objects is not None:
            labels += [
                "large_objects_mAP_50_95",
                "large_objects_mAP_50",
                "large_objects_mAP_75",
            ]
            values += [
                self.large_objects.map50_95,
                self.large_objects.map50,
                self.large_objects.map75,
            ]
            colors += [LEGACY_COLOR_PALETTE[4]] * 3

        plt.rcParams["font.family"] = "monospace"

        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value", fontweight="bold")
        ax.set_title("Mean Average Precision", fontweight="bold")

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
