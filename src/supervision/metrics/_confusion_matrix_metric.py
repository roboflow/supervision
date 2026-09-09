"""Shared implementation behind the confusion-matrix detection metrics.

`Precision`, `Recall` and `F1Score` reduce detections to the very same per-class,
per-IoU-threshold `(TP, FP, FN)` matrix. They differ only in the formula applied to
that matrix, in the result dataclass they fill, and in the labels used when a result
is rendered. This module owns everything they share.

The public classes stay in their own modules on purpose: the API reference does not
enable `inherited_members`, so a public method moved onto a shared base would silently
disappear from the rendered documentation.
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import AveragingMethod, Metric, MetricTarget
from supervision.metrics.utils.matching import (
    _match_detection_batch_with_target_indices,
)
from supervision.metrics.utils.object_size import (
    ObjectSizeCategory,
    get_detection_size_category,
)
from supervision.metrics.utils.utils import ensure_pandas_installed

if TYPE_CHECKING:
    import pandas as pd

_ResultT = TypeVar("_ResultT")


def _validate_confusion_matrix(confusion_matrix: npt.NDArray[np.float64]) -> None:
    """Raise when the last axis does not hold `(TP, FP, FN)` triplets.

    Guards every metric formula, which indexes that axis positionally and would
    otherwise silently score whatever the caller passed in.
    """
    if not confusion_matrix.shape[-1] == 3:
        raise ValueError(
            f"Confusion matrix must have shape (..., 3), got {confusion_matrix.shape}"
        )


class _ConfusionMatrixMetric(Metric[_ResultT]):
    """Shared machinery for metrics derived from a `(TP, FP, FN)` confusion matrix.

    Subclasses keep their own public `__init__`, `reset`, `update` and `compute` so
    that the API reference keeps rendering them, and supply the metric-specific parts
    through three hooks: the `_metric_name` used in error messages, the
    `_score_from_confusion_matrix` formula, and the `_build_result` factory.
    """

    #: Name of the metric as it appears in error messages, e.g. `"Precision"`.
    _metric_name: ClassVar[str]

    _metric_target: MetricTarget
    averaging_method: AveragingMethod
    _predictions_list: list[Detections]
    _targets_list: list[Detections]

    @abstractmethod
    def _score_from_confusion_matrix(
        self, confusion_matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return the metric value implied by each entry of the confusion matrix."""
        raise NotImplementedError

    @abstractmethod
    def _build_result(
        self,
        scores: npt.NDArray[np.float64],
        per_class_scores: npt.NDArray[np.float64],
        iou_thresholds: npt.NDArray[np.float32],
        matched_classes: npt.NDArray[np.int32],
    ) -> _ResultT:
        """Wrap computed scores in the metric-specific result dataclass."""
        raise NotImplementedError

    def _compute(
        self,
        predictions_list: list[Detections],
        targets_list: list[Detections],
        size_category: ObjectSizeCategory = ObjectSizeCategory.ANY,
    ) -> _ResultT:
        """Build per-image stats tuples and delegate to class-level computation.

        Each stats tuple is
        ``(matches, ignored_matches, confidence, class_ids, true_class_ids)``:
        - Both empty: skip (no information).
        - Targets empty, predictions present: all predictions are FPs; true_class_ids
          is ``zeros((0,))``.
        - Targets present: IoU matching produces ``matches`` array.
        """
        iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)
        stats: list[Any] = []

        for predictions, targets in zip(predictions_list, targets_list):
            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)
            prediction_size_mask = np.ones(len(predictions), dtype=bool)
            target_size_mask = np.ones(len(targets), dtype=bool)
            if size_category != ObjectSizeCategory.ANY:
                if len(predictions) > 0:
                    prediction_size_mask = (
                        get_detection_size_category(predictions, self._metric_target)
                        == size_category.value
                    )
                if len(targets) > 0:
                    target_size_mask = (
                        get_detection_size_category(targets, self._metric_target)
                        == size_category.value
                    )

            if len(targets) == 0 and len(predictions) > 0:
                # Only predictions are present (e.g. a background image); every
                # prediction is a false positive. Their classes are still tracked so
                # that `matched_classes` agrees across the three metrics.
                if predictions.class_id is None or predictions.confidence is None:
                    raise ValueError(
                        f"{self._metric_name} metric requires `class_id` and "
                        "`confidence` on predictions."
                    )
                prediction_class_ids = np.asarray(predictions.class_id, dtype=np.int32)[
                    prediction_size_mask
                ]
                prediction_confidence = np.asarray(
                    predictions.confidence, dtype=np.float32
                )[prediction_size_mask]
                if len(prediction_class_ids) == 0:
                    continue
                stats.append(
                    (
                        np.zeros(
                            (len(prediction_class_ids), iou_thresholds.size),
                            dtype=np.bool_,
                        ),
                        np.zeros(
                            (len(prediction_class_ids), iou_thresholds.size),
                            dtype=np.bool_,
                        ),
                        prediction_confidence,
                        prediction_class_ids,
                        np.zeros((0,), dtype=np.int32),
                    )
                )
            elif len(targets) > 0:
                if predictions.class_id is None or targets.class_id is None:
                    raise ValueError(
                        f"{self._metric_name} metric requires `class_id` on both "
                        "predictions and targets."
                    )
                if len(predictions) == 0:
                    target_class_ids = np.asarray(targets.class_id, dtype=np.int32)[
                        target_size_mask
                    ]
                    if len(target_class_ids) == 0:
                        continue
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=int),
                            target_class_ids,
                        )
                    )

                else:
                    if predictions.confidence is None:
                        raise ValueError(
                            f"{self._metric_name} metric requires `confidence` on "
                            "predictions."
                        )
                    prediction_class_ids = np.asarray(
                        predictions.class_id, dtype=np.int32
                    )
                    target_class_ids = np.asarray(targets.class_id, dtype=np.int32)
                    prediction_confidence = np.asarray(
                        predictions.confidence, dtype=np.float32
                    )
                    if self._metric_target == MetricTarget.BOXES:
                        # BOXES target never yields CompactMask; narrow for mypy.
                        iou = box_iou_batch(
                            cast(npt.NDArray[np.number], target_contents),
                            cast(npt.NDArray[np.number], prediction_contents),
                        )
                    elif self._metric_target == MetricTarget.MASKS:
                        iou = mask_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                        # OBB target never yields CompactMask; narrow for mypy.
                        iou = oriented_box_iou_batch(
                            cast(npt.NDArray[np.number], target_contents),
                            cast(npt.NDArray[np.number], prediction_contents),
                        )
                    else:
                        raise ValueError(
                            "Unsupported metric target for IoU calculation"
                        )

                    # None keeps the matcher on its single-round fast path
                    # when no size bucket is scored.
                    target_scored_mask = (
                        target_size_mask
                        if size_category != ObjectSizeCategory.ANY
                        else None
                    )
                    matches, matched_target_indices = (
                        _match_detection_batch_with_target_indices(
                            prediction_class_ids,
                            target_class_ids,
                            iou,
                            iou_thresholds,
                            target_scored_mask=target_scored_mask,
                        )
                    )
                    ignored_matches = np.zeros_like(matches, dtype=bool)
                    if size_category != ObjectSizeCategory.ANY:
                        valid_target_match = matched_target_indices >= 0
                        matched_scored_target = np.zeros_like(matches, dtype=bool)
                        if np.any(valid_target_match):
                            matched_scored_target[valid_target_match] = (
                                target_size_mask[
                                    matched_target_indices[valid_target_match]
                                ]
                            )
                        prediction_scored = (
                            prediction_size_mask[:, None] | matched_scored_target
                        )
                        ignored_matches = ~prediction_scored | (
                            valid_target_match & ~matched_scored_target
                        )
                        prediction_keep = np.any(~ignored_matches, axis=1)
                        matches = (
                            matches[prediction_keep]
                            & matched_scored_target[prediction_keep]
                        )
                        ignored_matches = ignored_matches[prediction_keep]
                        prediction_confidence = prediction_confidence[prediction_keep]
                        prediction_class_ids = prediction_class_ids[prediction_keep]
                        target_class_ids = target_class_ids[target_size_mask]
                        if (
                            len(prediction_class_ids) == 0
                            and len(target_class_ids) == 0
                        ):
                            continue
                    stats.append(
                        (
                            matches,
                            ignored_matches,
                            prediction_confidence,
                            prediction_class_ids,
                            target_class_ids,
                        )
                    )

        if not stats:
            return self._build_result(
                scores=np.zeros(iou_thresholds.shape[0]),
                per_class_scores=np.zeros((0, iou_thresholds.shape[0])),
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
            )

        concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
        scores, per_class_scores, unique_classes = self._compute_scores_for_classes(
            *concatenated_stats
        )

        return self._build_result(
            scores=scores,
            per_class_scores=per_class_scores,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
        )

    def _compute_scores_for_classes(
        self,
        matches: npt.NDArray[np.bool_],
        ignored_matches: npt.NDArray[np.bool_],
        prediction_confidence: npt.NDArray[np.float32],
        prediction_class_ids: npt.NDArray[np.int32],
        true_class_ids: npt.NDArray[np.int32],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
    ]:
        """Compute per-class and aggregated scores from stats of all images.

        ``unique_classes`` is the union of ground-truth and predicted classes, so a
        class that only ever appears in the predictions is tracked with a score of
        `0.0` rather than dropped. This keeps the tracked class set identical across
        the three metrics and matches sklearn, which infers labels from the union of
        ``y_true`` and ``y_pred``.
        """
        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        ignored_matches = ignored_matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]
        true_classes, true_counts = np.unique(true_class_ids, return_counts=True)
        pred_classes = np.unique(prediction_class_ids)
        # Dedupe each side first, then union1d the already-unique arrays: this skips
        # union1d's internal re-sort/re-unique of the full concatenation and measured
        # ~1.4x faster than np.unique(np.concatenate(...)). Deduping after the union
        # (or union1d on the raw arrays) yields no speedup.
        unique_classes = np.union1d(true_classes, pred_classes)
        class_counts = np.zeros(unique_classes.shape[0], dtype=int)
        class_counts[np.searchsorted(unique_classes, true_classes)] = true_counts

        # Shape: PxTh,P,C,C -> CxThx3
        confusion_matrix = self._compute_confusion_matrix(
            matches, ignored_matches, prediction_class_ids, unique_classes, class_counts
        )

        # Shape: CxThx3 -> CxTh
        per_class_scores = self._score_from_confusion_matrix(confusion_matrix)

        # Shape: CxTh -> Th
        if self.averaging_method == AveragingMethod.MACRO:
            scores = np.mean(per_class_scores, axis=0)
        elif self.averaging_method == AveragingMethod.MICRO:
            confusion_matrix_merged = confusion_matrix.sum(0)
            scores = self._score_from_confusion_matrix(confusion_matrix_merged)
        elif self.averaging_method == AveragingMethod.WEIGHTED:
            class_counts = class_counts.astype(np.float32)
            if class_counts.sum() == 0:
                # No ground-truth support (e.g. only false-positive classes, or a
                # size bucket with predictions but no targets): weighting is
                # undefined, so report 0 as the empty case did before.
                scores = np.zeros(per_class_scores.shape[1])
            else:
                scores = np.average(per_class_scores, axis=0, weights=class_counts)

        return scores, per_class_scores, unique_classes

    @staticmethod
    def _compute_confusion_matrix(
        sorted_matches: npt.NDArray[np.bool_],
        sorted_ignored_matches: npt.NDArray[np.bool_],
        sorted_prediction_class_ids: npt.NDArray[np.int32],
        unique_classes: npt.NDArray[np.integer],
        class_counts: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.float64]:
        """Compute the confusion matrix for each class and IoU threshold.

        Assumes the matches and prediction_class_ids are sorted by confidence
        in descending order.

        Args:
            sorted_matches: shape (P, Th), that is True
                if the prediction is a true positive at the given IoU threshold.
            sorted_ignored_matches: shape (P, Th), that is True
                if the prediction should not affect the given IoU threshold.
            sorted_prediction_class_ids: shape (P,), containing
                the class id for each prediction.
            unique_classes: shape (C,), containing the unique
                class ids.
            class_counts: shape (C,), containing the number
                of true instances for each class.

        Returns:
            shape (C, Th, 3), containing the true positives, false
                positives, and false negatives for each class and IoU threshold.
        """
        num_thresholds = sorted_matches.shape[1]
        num_classes = unique_classes.shape[0]

        confusion_matrix: npt.NDArray[np.float64] = np.zeros(
            (num_classes, num_thresholds, 3), dtype=np.float64
        )
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
                false_positives = (~sorted_ignored_matches[is_class]).sum(0)
                false_negatives = np.zeros(num_thresholds)
            else:
                true_positives = sorted_matches[is_class].sum(0)
                false_positives = (
                    ~sorted_matches[is_class] & ~sorted_ignored_matches[is_class]
                ).sum(0)
                false_negatives = num_true - true_positives
            confusion_matrix[class_idx] = np.stack(
                [true_positives, false_positives, false_negatives], axis=1
            )
        return confusion_matrix

    def _detections_content(
        self, detections: Detections
    ) -> npt.NDArray[Any] | CompactMask:
        """Return boxes, masks or oriented bounding boxes from detections.

        For the mask target this may return a
        :class:`~supervision.detection.compact_mask.CompactMask` rather than a
        dense boolean array when the detections carry compact masks.
        """
        if self._metric_target == MetricTarget.BOXES:
            return cast(npt.NDArray[Any], detections.xyxy)
        if self._metric_target == MetricTarget.MASKS:
            if detections.mask is not None:
                # detections.mask is NDArray[bool] | CompactMask; return as-is.
                return detections.mask
            if len(detections) > 0:
                raise ValueError(
                    f"{self._metric_name} with `MetricTarget.MASKS` requires "
                    "detections to include masks."
                )
            return self._make_empty_content()
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = detections.data.get(ORIENTED_BOX_COORDINATES)
            if obb is not None and len(obb) > 0:
                result_obb: npt.NDArray[np.float32] = np.array(obb, dtype=np.float32)
                return result_obb
            return self._make_empty_content()
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _make_empty_content(self) -> npt.NDArray[Any]:
        """Return the empty content array matching the configured metric target."""
        if self._metric_target == MetricTarget.BOXES:
            empty_boxes: npt.NDArray[np.float32] = np.empty((0, 4), dtype=np.float32)
            return empty_boxes
        if self._metric_target == MetricTarget.MASKS:
            empty_masks: npt.NDArray[np.bool_] = np.empty((0, 0, 0), dtype=bool)
            return empty_masks
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            empty_obb: npt.NDArray[np.float32] = np.empty((0, 4, 2), dtype=np.float32)
            return empty_obb
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
        """Filter predictions and targets by object size category."""
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


@dataclass(frozen=True)
class _ResultLabels:
    """Metric-specific wording used when a result is printed, tabulated or plotted.

    Attributes:
        short: compact label, e.g. `"P"`; used in score lines, DataFrame columns
            and object-size plot labels.
        long: full metric name, e.g. `"Precision"`; heads the per-class listing and
            the two headline plot labels.
        plot_title: metric name as it opens the plot title, e.g. `"F1 Score"`.
        metric_target_padding: spacing between `"Metric target:"` and its value.
    """

    short: str
    long: str
    plot_title: str
    metric_target_padding: str


@dataclass(frozen=True)
class _ResultView:
    """Field-name-agnostic snapshot of one metric result.

    The three result dataclasses expose the same values under different public field
    names (`precision_scores`, `recall_scores`, `f1_scores`, ...), so the shared
    renderers work on this view instead of on the results themselves.
    """

    class_name: str
    labels: _ResultLabels
    metric_target: MetricTarget
    averaging_method: AveragingMethod
    score_at_50: float
    score_at_75: float
    scores: npt.NDArray[np.float64]
    per_class_scores: npt.NDArray[np.float64]
    iou_thresholds: npt.NDArray[np.float32]
    matched_classes: npt.NDArray[np.int32]
    small_objects: _ResultView | None
    medium_objects: _ResultView | None
    large_objects: _ResultView | None


class _SupportsResultView(Protocol):
    """A metric result able to describe itself as a `_ResultView`."""

    def _result_view(self) -> _ResultView:
        """Return the field-name-agnostic view of this result."""
        ...


def _optional_result_view(result: _SupportsResultView | None) -> _ResultView | None:
    """Return the view of an optional nested object-size result."""
    if result is None:
        return None
    return result._result_view()


def _format_result(view: _ResultView) -> str:
    """Render a metric result and its object-size breakdown as a pretty string.

    Every score line is padded to a column derived from the longest of them, so metric
    labels of differing width still line up under each other.
    """
    labels = view.labels
    value_column = len(f"{labels.short} @ thresh:") + 1
    score_at_50_label = f"{labels.short} @ 50:".ljust(value_column)
    score_at_75_label = f"{labels.short} @ 75:".ljust(value_column)
    scores_label = f"{labels.short} @ thresh:".ljust(value_column)
    iou_thresholds_label = "IoU thresh:".ljust(value_column)

    out_str = (
        f"{view.class_name}:\n"
        f"Metric target:{labels.metric_target_padding}{view.metric_target}\n"
        f"Averaging method: {view.averaging_method}\n"
        f"{score_at_50_label}{view.score_at_50:.4f}\n"
        f"{score_at_75_label}{view.score_at_75:.4f}\n"
        f"{scores_label}{view.scores}\n"
        f"{iou_thresholds_label}{view.iou_thresholds}\n"
        f"{labels.long} per class:\n"
    )
    if view.per_class_scores.size == 0:
        out_str += "  No results\n"
    for class_id, score_of_class in zip(view.matched_classes, view.per_class_scores):
        out_str += f"  {class_id}: {score_of_class}\n"

    indent = "  "
    for name, bucket in (
        ("Small", view.small_objects),
        ("Medium", view.medium_objects),
        ("Large", view.large_objects),
    ):
        if bucket is None:
            continue
        indented = indent + _format_result(bucket).replace("\n", f"\n{indent}")
        out_str += f"\n{name} objects:\n{indented}"

    return out_str


def _result_to_pandas(view: _ResultView) -> pd.DataFrame:
    """Convert a metric result and its object-size breakdown to a one-row DataFrame."""
    ensure_pandas_installed()
    import pandas as pd

    short = view.labels.short
    pandas_data: dict[str, Any] = {
        f"{short}@50": view.score_at_50,
        f"{short}@75": view.score_at_75,
    }

    for prefix, bucket in (
        ("small_objects", view.small_objects),
        ("medium_objects", view.medium_objects),
        ("large_objects", view.large_objects),
    ):
        if bucket is None:
            continue
        bucket_frame = _result_to_pandas(bucket)
        for key, value in bucket_frame.items():
            pandas_data[f"{prefix}_{key}"] = value

    return pd.DataFrame(pandas_data, index=[0])


def _plot_result(view: _ResultView) -> None:
    """Plot the metric value at IoU 0.5 and 0.75, split by object size."""
    from matplotlib import pyplot as plt

    result_labels = view.labels
    labels = [f"{result_labels.long}@50", f"{result_labels.long}@75"]
    values = [view.score_at_50, view.score_at_75]
    colors = [LEGACY_COLOR_PALETTE[0]] * 2

    for name, palette_index, bucket in (
        ("Small", 3, view.small_objects),
        ("Medium", 2, view.medium_objects),
        ("Large", 4, view.large_objects),
    ):
        if bucket is None:
            continue
        labels += [
            f"{name}: {result_labels.short}@50",
            f"{name}: {result_labels.short}@75",
        ]
        values += [bucket.score_at_50, bucket.score_at_75]
        colors += [LEGACY_COLOR_PALETTE[palette_index]] * 2

    plt.rcParams["font.family"] = "monospace"

    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value", fontweight="bold")
    title = (
        f"{result_labels.plot_title}, by Object Size"
        f"\n(target: {view.metric_target.value},"
        f" averaging: {view.averaging_method.value})"
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
