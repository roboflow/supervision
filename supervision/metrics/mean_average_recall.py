from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils import (
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
    Mean Average Recall (mAR) metric for object detection evaluation.
    Calculates the average recall across different IoU thresholds and detection limits.

    The metric evaluates:
        - IoU thresholds from 0.5 to 0.95 with 0.05 step
        - Different maximum detection limits [1, 10, 100]
        - Size-specific evaluation (small, medium, large objects)

    When no detections or targets are present, returns 0.0.

    Example:
        ```python
        import supervision as sv
        from supervision.metrics import MeanAverageRecall

        predictions = sv.Detections(...)
        targets = sv.Detections(...)

        mar_metric = MeanAverageRecall()
        mar_result = mar_metric.update(predictions, targets).compute()

        print(mar_result)
        print(mar_result.mean_average_recall)
        mar_result.plot()
        ```
    """

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
        max_detections: List[int] = [1, 10, 100],  # Add max_detections parameter
    ):
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic
        self._max_detections = max_detections
        self._predictions_list: List[Detections] = []
        self._targets_list: List[Detections] = []

    def reset(self) -> None:
        self._predictions_list = []
        self._targets_list = []

    def update(
        self,
        predictions: Union[Detections, List[Detections]],
        targets: Union[Detections, List[Detections]],
    ) -> MeanAverageRecall:
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]

        if len(predictions) != len(targets):
            raise ValueError(
                f"The number of predictions ({len(predictions)}) and"
                f" targets ({len(targets)}) must be the same."
            )

        if self._class_agnostic:
            predictions = deepcopy(predictions)
            targets = deepcopy(targets)
            for prediction in predictions:
                prediction.class_id[:] = -1
            for target in targets:
                target.class_id[:] = -1

        self._predictions_list.extend(predictions)
        self._targets_list.extend(targets)
        return self

    def compute(self) -> MeanAverageRecallResult:
        result = self._compute(self._predictions_list, self._targets_list)

        # Compute size-specific results
        for size_category, container in [
            (ObjectSizeCategory.SMALL, "small_objects"),
            (ObjectSizeCategory.MEDIUM, "medium_objects"),
            (ObjectSizeCategory.LARGE, "large_objects"),
        ]:
            size_predictions = []
            size_targets = []
            for predictions, targets in zip(self._predictions_list, self._targets_list):
                size_predictions.append(
                    self._filter_detections_by_size(predictions, size_category)
                )
                size_targets.append(
                    self._filter_detections_by_size(targets, size_category)
                )
            setattr(result, container, self._compute(size_predictions, size_targets))

        return result

    def _compute(
        self,
        predictions_list: List[Detections],
        targets_list: List[Detections],
    ) -> MeanAverageRecallResult:
        if not predictions_list or not targets_list:
            return MeanAverageRecallResult(
                metric_target=self._metric_target,
                is_class_agnostic=self._class_agnostic,
                mean_average_recall=0.0,
                ar_per_class=np.array([]),
                matched_classes=np.array([]),
            )

        all_recalls = []
        all_class_ids = []
        iou_thresholds = np.linspace(0.5, 0.95, 10)

        for predictions, targets in zip(predictions_list, targets_list):
            if targets.is_empty():
                continue

            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)

            if predictions.is_empty():
                unique_classes = np.unique(targets.class_id)
                all_class_ids.extend(unique_classes)
                all_recalls.extend([0.0] * len(unique_classes))
                continue

            if self._metric_target == MetricTarget.BOXES:
                iou = box_iou_batch(target_contents, prediction_contents)
            elif self._metric_target == MetricTarget.MASKS:
                iou = mask_iou_batch(target_contents, prediction_contents)
            elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                iou = oriented_box_iou_batch(target_contents, prediction_contents)
            else:
                raise ValueError("Unsupported metric target for IoU calculation")

            # For each class
            unique_classes = np.unique(targets.class_id)
            for class_id in unique_classes:
                target_mask = targets.class_id == class_id
                pred_mask = (
                    predictions.class_id == class_id
                    if not self._class_agnostic
                    else slice(None)
                )

                if not any(target_mask):
                    continue

                class_iou = iou[target_mask]
                if any(pred_mask):
                    class_iou = class_iou[:, pred_mask]

                # Calculate recall for each max detection limit
                recalls_at_k = []
                for k in self._max_detections:
                    recalls_at_iou = []
                    for threshold in iou_thresholds:
                        matches = (class_iou > threshold).any(axis=1)
                        recall = matches[:k].mean() if len(matches) > 0 else 0.0
                        recalls_at_iou.append(recall)
                    recalls_at_k.append(np.mean(recalls_at_iou))

                # Store the maximum recall across different k values
                all_recalls.append(max(recalls_at_k))
                all_class_ids.append(class_id)

        if not all_recalls:
            return MeanAverageRecallResult(
                metric_target=self._metric_target,
                is_class_agnostic=self._class_agnostic,
                mean_average_recall=0.0,
                ar_per_class=np.array([]),
                matched_classes=np.array([]),
            )

        # Convert lists to numpy arrays
        all_recalls = np.array(all_recalls)
        all_class_ids = np.array(all_class_ids)

        # Aggregate per-class recalls
        unique_classes = np.unique(all_class_ids)
        ar_per_class = np.zeros(len(unique_classes))
        for i, class_id in enumerate(unique_classes):
            class_mask = all_class_ids == class_id
            ar_per_class[i] = np.mean(all_recalls[class_mask])

        return MeanAverageRecallResult(
            metric_target=self._metric_target,
            is_class_agnostic=self._class_agnostic,
            mean_average_recall=np.mean(ar_per_class),
            ar_per_class=ar_per_class,
            matched_classes=unique_classes,
        )

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
        """Return a copy of detections with contents filtered by object size.
        Small: area < 32^2
        Medium: 32^2 <= area < 96^2
        Large: area >= 96^2
        """
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
class MeanAverageRecallResult:
    """
    The result of the Mean Average Recall calculation.

    Defaults to `0.0` when no detections or targets are present.

    Attributes:
        metric_target (MetricTarget): The type of data used for the metric
            (boxes, masks, or oriented bounding boxes)
        is_class_agnostic (bool): When computing class-agnostic results,
            class ID is set to `-1`
        mean_average_recall (float): The global mAR score averaged across classes,
            IoU thresholds, and detection limits
        ar_per_class (np.ndarray): The average recall scores per class
        matched_classes (np.ndarray): The class IDs of all matched classes
        small_objects (Optional[MeanAverageRecallResult]): The mAR results for
            small objects (area < 32²)
        medium_objects (Optional[MeanAverageRecallResult]): The mAR results for
            medium objects (32² ≤ area < 96²)
        large_objects (Optional[MeanAverageRecallResult]): The mAR results for
            large objects (area ≥ 96²)
    """

    metric_target: MetricTarget
    is_class_agnostic: bool
    mean_average_recall: float
    ar_per_class: np.ndarray
    matched_classes: np.ndarray
    small_objects: Optional[MeanAverageRecallResult] = None
    medium_objects: Optional[MeanAverageRecallResult] = None
    large_objects: Optional[MeanAverageRecallResult] = None

    def __str__(self) -> str:
        out_str = (
            f"{self.__class__.__name__}:\n"
            f"Metric target: {self.metric_target}\n"
            f"Class agnostic: {self.is_class_agnostic}\n"
            f"mAR: {self.mean_average_recall:.4f}\n"
            f"AR per class:\n"
        )
        if self.ar_per_class.size == 0:
            out_str += "  No results\n"
        for class_id, ar_of_class in zip(self.matched_classes, self.ar_per_class):
            out_str += f"  {class_id}: {ar_of_class:.4f}\n"

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
        ensure_pandas_installed()
        import pandas as pd

        pandas_data = {
            "mAR": self.mean_average_recall,
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
        labels = ["mAR"]
        values = [self.mean_average_recall]
        colors = [LEGACY_COLOR_PALETTE[0]]

        if self.small_objects is not None:
            labels.append("Small: mAR")
            values.append(self.small_objects.mean_average_recall)
            colors.append(LEGACY_COLOR_PALETTE[3])

        if self.medium_objects is not None:
            labels.append("Medium: mAR")
            values.append(self.medium_objects.mean_average_recall)
            colors.append(LEGACY_COLOR_PALETTE[2])

        if self.large_objects is not None:
            labels.append("Large: mAR")
            values.append(self.large_objects.mean_average_recall)
            colors.append(LEGACY_COLOR_PALETTE[4])

        plt.rcParams["font.family"] = "monospace"

        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value", fontweight="bold")
        ax.set_title("Mean Average Recall", fontweight="bold")

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
