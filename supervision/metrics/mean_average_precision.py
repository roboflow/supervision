from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.metrics.core import (
    InternalMetricDataStore,
    Metric,
    MetricData,
    MetricTarget,
)
from supervision.metrics.intersection_over_union import IntersectionOverUnion


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
        iou_threshold: float = 0.25,
    ):
        self._metric_target = metric_target
        if self._metric_target != MetricTarget.BOXES:
            raise NotImplementedError(
                f"mAP is not implemented for {self._metric_target}."
            )

        self._class_agnostic = class_agnostic
        self._iou_threshold = iou_threshold

        self._store = InternalMetricDataStore(metric_target, class_agnostic)
        self._iou_metric = IntersectionOverUnion(metric_target, class_agnostic)

        self.reset()

    def reset(self) -> None:
        self._iou_metric.reset()
        self._store.reset()

    def update(
        self,
        predictions: Union[Detections, List[Detections]],
        targets: Union[Detections, List[Detections]],
    ) -> MeanAveragePrecision:
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]

        for d1, d2 in zip_longest(predictions, targets, fillvalue=Detections.empty()):
            self._update(d1, d2)

        return self

    def _update(self, predictions: Detections, targets: Detections) -> None:
        self._store.update(predictions, targets)
        self._iou_metric.update(predictions, targets)

    def compute(self) -> MeanAveragePrecisionResult:
        ious = self._iou_metric.compute()
        iou_thresholds = np.linspace(0.5, 0.95, 10)

        average_precisions: Dict[int, npt.NDArray] = {}
        for class_id, prediction_data, target_data in self._store:
            if len(target_data) == 0:
                continue

            if len(prediction_data) == 0:
                stats = (
                    np.zeros((0, iou_thresholds.size), dtype=bool),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32),
                    target_data.class_id,
                )
            else:
                ious_of_class = ious[class_id]
                matches = self._match_predictions_to_targets(
                    prediction_data, target_data, ious_of_class, iou_thresholds
                )
                stats = (
                    matches,
                    prediction_data.confidence,
                    prediction_data.class_id,
                    target_data.class_id,
                )

            to_concat = [np.expand_dims(item, 0) for item in stats]
            for x in to_concat:
                print(x.shape)
            # print(to_concat)

            # TODO: class_id size mismatch
            # (1, 9, 10)
            # (1, 9)
            # (1, 9)
            # (1, 12)

            concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
            average_precisions[class_id] = self._average_precisions_per_class(
                *concatenated_stats
            )

        return MeanAveragePrecisionResult(average_precisions)

    def _match_predictions_to_targets(
        self,
        prediction_data: MetricData,
        target_data: MetricData,
        predictions_iou: npt.NDArray[np.float32],
        iou_thresholds: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        """
        Match predictions to targets based on IoU.

        Given N predictions, M targets and T IoU thresholds,
        returns a boolean array (N, T), where each element is True
        if the prediction is a true positive at the given IoU threshold.

        Assumes that predictions were already filtered by class.
        """
        if set(prediction_data.class_id) != set(target_data.class_id):
            raise ValueError(
                f"Class IDs of predictions and targets"
                f" do not match: {prediction_data.class_id}, {target_data.class_id}"
            )

        correct = np.zeros((len(prediction_data), len(iou_thresholds)), dtype=bool)
        for i, iou_level in enumerate(iou_thresholds):
            matched_indices = np.where((predictions_iou >= iou_level))

            if matched_indices[0].shape[0]:
                combined_indices = np.stack(matched_indices, axis=1)
                iou_values = predictions_iou[matched_indices][:, None]
                matches = np.hstack([combined_indices, iou_values])

                if matched_indices[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]

                    _, unique_pred_idx = np.unique(matches[:, 1], return_index=True)
                    matches = matches[unique_pred_idx]
                    _, unique_target_idx = np.unique(matches[:, 0], return_index=True)
                    matches = matches[unique_target_idx]

                correct[matches[:, 1].astype(int), i] = True

        return correct

    def _average_precisions_per_class(
        self,
        matches: np.ndarray,
        prediction_confidence: np.ndarray,
        prediction_class_ids: np.ndarray,
        true_class_ids: np.ndarray,
        eps: float = 1e-16,
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
            np.ndarray: Average precision for different IoU levels.
        """
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
            recall = true_positives / (total_true + eps)
            precision = true_positives / (true_positives + false_positives)

            for iou_level_idx in range(matches.shape[1]):
                average_precisions[class_idx, iou_level_idx] = (
                    self._compute_average_precision(
                        recall[:, iou_level_idx], precision[:, iou_level_idx]
                    )
                )

        return average_precisions

    @staticmethod
    def _compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute the average precision using 101-point interpolation (COCO), given
            the recall and precision curves.

        Args:
            recall (np.ndarray): The recall curve.
            precision (np.ndarray): The precision curve.

        Returns:
            float: Average precision.
        """
        assert len(recall) == len(precision)

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


@dataclass
class MeanAveragePrecisionResult:
    # TODO: continue here
    average_precisions: Dict[int, npt.NDArray]
