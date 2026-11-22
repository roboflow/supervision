from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void

from supervision.config import ISCROWD_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.internal import get_data_item
from supervision.detection.utils.iou_and_nms import (
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.draw.color import LEGACY_COLOR_PALETTE
from supervision.metrics.core import (
    Metric,
    MetricResult,
    MetricTarget,
    PlotDetails,
    _append_object_size_plot_details,
)
from supervision.metrics.utils.object_size import (
    SIZE_THRESHOLDS,
    ObjectSizeCategory,
    get_detection_areas,
)
from supervision.metrics.utils.object_size import (
    get_detection_size_category as _get_detection_size_category,
)
from supervision.metrics.utils.utils import ensure_pandas_installed
from supervision.utils.internal import _DEPRECATION_WARNING_STREAM

if TYPE_CHECKING:
    import pandas as pd

#: Warning text for the copy of the size-category helper that this module exposed as
#: a side effect of using it. The helper itself stays public in its own module.
_SIZE_CATEGORY_MODULE_DEPRECATION = (
    "`%(source_name)s` imported from `supervision.metrics.mean_average_recall` is "
    "deprecated in `%(deprecated_in)s` and will be removed in `%(remove_in)s`. "
    "Import it from `supervision.metrics.utils.object_size` instead."
)


@deprecated(  # type: ignore[untyped-decorator]
    target=_get_detection_size_category,
    deprecated_in="0.32.0",
    remove_in="0.35.0",
    stream=_DEPRECATION_WARNING_STREAM,
    template_mgs=_SIZE_CATEGORY_MODULE_DEPRECATION,
)
def get_detection_size_category(
    detections: Detections, metric_target: MetricTarget = MetricTarget.BOXES
) -> npt.NDArray[np.int_]:
    """Deprecated: import `get_detection_size_category` from its own module.

    Releases before `0.32.0` exposed this helper from this module, so the name stays
    importable here until `0.35.0`.
    """
    return void(detections, metric_target)  # type: ignore[no-any-return]


@dataclass
class MeanAverageRecallResult(MetricResult):
    """The results of the Mean Average Recall metric calculation.

    Defaults to `0` if no detections or targets were provided.

    Attributes:
        metric_target: the type of data used for the metric -
            boxes, masks or oriented bounding boxes.
        mAR_at_1: the Mean Average Recall, when considering only the top
            highest confidence detection for each class in each image.
        mAR_at_10: the Mean Average Recall, when considering top 10
            highest confidence detections for each class in each image.
        mAR_at_100: the Mean Average Recall, when considering top 100
            highest confidence detections for each class in each image.
        recall_per_class: the recall scores per max detection count, class and
            IoU threshold. Shape:
            `(num_max_detections, num_target_classes, num_iou_thresholds)`.
        max_detections: the array with maximum number of detections
            considered.
        iou_thresholds: the IoU thresholds used in the calculations.
        matched_classes: the class IDs of all matched classes.
            Corresponds to the class axis of `recall_per_class`.
        small_objects: the Mean Average Recall
            metric results for small objects (area < 32²).
        medium_objects: the Mean Average Recall
            metric results for medium objects (32² ≤ area < 96²).
        large_objects: the Mean Average Recall
            metric results for large objects (area ≥ 96²).
    """

    metric_target: MetricTarget

    @property
    def mAR_at_1(self) -> float:
        return float(self.recall_scores[0])

    @property
    def mAR_at_10(self) -> float:
        return float(self.recall_scores[1])

    @property
    def mAR_at_100(self) -> float:
        return float(self.recall_scores[2])

    recall_scores: npt.NDArray[np.float64]
    recall_per_class: npt.NDArray[np.float64]
    max_detections: npt.NDArray[np.int32]
    iou_thresholds: npt.NDArray[np.float32]
    matched_classes: npt.NDArray[np.int32]

    small_objects: MeanAverageRecallResult | None
    medium_objects: MeanAverageRecallResult | None
    large_objects: MeanAverageRecallResult | None

    def __str__(self) -> str:
        """Format as a pretty string.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> from supervision.metrics import MeanAverageRecall
            >>> predictions = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0]),
            ...     confidence=np.array([0.9])
            ... )
            >>> targets = sv.Detections(
            ...     xyxy=np.array([[0, 0, 10, 10]]),
            ...     class_id=np.array([0])
            ... )
            >>> mar_metric = MeanAverageRecall()
            >>> mar_result = mar_metric.update(predictions, targets).compute()
            >>> print(mar_result)  # doctest: +ELLIPSIS
            MeanAverageRecallResult:
            Metric target:  MetricTarget.BOXES
            mAR @ 1:    1.0000
            mAR @ 10:   1.0000
            mAR @ 100:  1.0000
            max detections: [  1  10 100]
            IoU thresh:     [0.5  0.55 ... 0.95]
            mAR per class:
              @1:
                0: [1. ... 1.]
              @10:
                0: [1. ... 1.]
              @100:
                0: [1. ... 1.]
            ...
            Medium objects:
              MeanAverageRecallResult:
              Metric target:  MetricTarget.BOXES
              mAR @ 1:    0.0000
              ...

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
        for max_detections, recalls_at_k in zip(
            self.max_detections, self.recall_per_class
        ):
            out_str += f"  @{max_detections}:\n"
            for class_id, recall_of_class in zip(self.matched_classes, recalls_at_k):
                out_str += f"    {class_id}: {recall_of_class}\n"

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
        """Convert the result to a pandas DataFrame.

        Returns:
            The result as a DataFrame.
        """
        ensure_pandas_installed()
        import pandas as pd

        pandas_data: dict[str, Any] = {
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

    def _get_plot_details(self, include_object_sizes: bool = True) -> PlotDetails:
        """Return bar-chart data for mAR scores.

        Args:
            include_object_sizes: When ``True``, include bars for
                small / medium / large object-size categories.
        """
        labels = ["mAR @ 1", "mAR @ 10", "mAR @ 100"]
        values = [self.mAR_at_1, self.mAR_at_10, self.mAR_at_100]
        colors = [LEGACY_COLOR_PALETTE[0]] * 3
        _append_object_size_plot_details(
            labels,
            values,
            colors,
            include_object_sizes=include_object_sizes,
            metric_labels=["mAR @ 1", "mAR @ 10", "mAR @ 100"],
            small_objects=self.small_objects,
            medium_objects=self.medium_objects,
            large_objects=self.large_objects,
            value_getter=lambda result: [
                result.mAR_at_1,
                result.mAR_at_10,
                result.mAR_at_100,
            ],
        )

        size_suffix = ", by Object Size" if include_object_sizes else ""
        title = (
            f"Mean Average Recall{size_suffix}\n(target: {self.metric_target.value})"
        )
        return PlotDetails(labels=labels, values=values, colors=colors, title=title)

    def plot(self) -> None:
        """Plot the Mean Average Recall results.

        ![example_plot](\
        https://media.roboflow.com/supervision-docs/metrics/mAR_plot_example.png\
        ){ align=center width="800" }
        """
        from matplotlib import pyplot as plt

        details = self._get_plot_details()

        plt.rcParams["font.family"] = "monospace"

        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value", fontweight="bold")
        ax.set_title(details.title, fontweight="bold")

        x_positions = range(len(details.labels))
        bars = ax.bar(x_positions, details.values, color=details.colors, align="center")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(details.labels, rotation=45, ha="right")

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


class MeanAverageRecall(Metric["MeanAverageRecallResult"]):
    """Mean Average Recall (mAR) measures how well the model detects and retrieves
    relevant objects by averaging recall over multiple IoU thresholds, classes and
    detection limits.

    Intuitively, while Recall measures the ability to find all relevant
    objects, mAR narrows down how many detections are considered for each
    image and class. For example, mAR @ 100 considers the top 100 highest
    confidence detections of each class in each image. mAR @ 1 considers only
    the highest confidence detection of each class in each image. This matches
    the `maxDets` limit of COCO evaluation, so with a single class the limit
    applies to the whole image.

    Targets marked as crowd regions in `targets.data["iscrowd"]`, loaded by
    :meth:`~supervision.dataset.core.DetectionDataset.from_coco`, or marked in
    `targets.data["ignore"]` are COCO ignore regions: they are not counted as
    missed objects.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics import MeanAverageRecall
        >>> predictions = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0]),
        ...     confidence=np.array([0.9])
        ... )
        >>> targets = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     class_id=np.array([0])
        ... )
        >>> mar_metric = MeanAverageRecall()
        >>> mar_result = mar_metric.update(predictions, targets).compute()
        >>> round(float(mar_result.mAR_at_100), 2)
        1.0

        ```

    ![example_plot](
        https://media.roboflow.com/supervision-docs/metrics/mAR_plot_example.png
    ){ align=center width="800" }
    """

    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
    ):
        """Initialize the Mean Average Recall metric.

        Args:
            metric_target: The type of detection data to use.
        """
        self._metric_target = metric_target

        self._predictions_list: list[Detections] = []
        self._targets_list: list[Detections] = []

        self.max_detections = np.array([1, 10, 100])

    def reset(self) -> None:
        """Reset the metric to its initial state, clearing all stored data."""
        self._predictions_list = []
        self._targets_list = []

    @staticmethod
    def _compact_if_dense_mask(detections: Detections) -> Detections:
        """Convert dense bool masks to tight CompactMask on a shallow copy."""
        if (
            detections.mask is None
            or not isinstance(detections.mask, np.ndarray)
            or detections.mask.dtype != np.bool_
            or len(detections) == 0
        ):
            return detections
        from supervision.detection.utils.converters import mask_to_xyxy

        dense = detections.mask
        image_shape = (int(dense.shape[1]), int(dense.shape[2]))
        tight_xyxy = mask_to_xyxy(dense)
        compact = CompactMask.from_dense(dense, tight_xyxy, image_shape=image_shape)
        shallow = copy(detections)
        shallow.mask = compact
        return shallow

    def update(
        self,
        predictions: Detections | list[Detections],
        targets: Detections | list[Detections],
    ) -> MeanAverageRecall:
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

        if self._metric_target == MetricTarget.MASKS:
            predictions = [self._compact_if_dense_mask(p) for p in predictions]
            targets = [self._compact_if_dense_mask(t) for t in targets]

        self._predictions_list.extend(predictions)
        self._targets_list.extend(targets)

        return self

    def _compute_iou_matrix(
        self,
        target_contents: npt.NDArray[Any] | CompactMask,
        prediction_contents: npt.NDArray[Any] | CompactMask,
    ) -> npt.NDArray[np.float32]:
        """Compute the pairwise target-by-prediction IoU matrix for one image."""
        if self._metric_target == MetricTarget.BOXES:
            # BOXES target never yields CompactMask; narrow for mypy.
            return cast(
                npt.NDArray[np.float32],
                box_iou_batch(
                    cast(npt.NDArray[np.number], target_contents),
                    cast(npt.NDArray[np.number], prediction_contents),
                ),
            )
        if self._metric_target == MetricTarget.MASKS:
            return cast(
                npt.NDArray[np.float32],
                mask_iou_batch(target_contents, prediction_contents),
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            # OBB target never yields CompactMask; narrow for mypy.
            # The float32 cast matches `box_iou_batch`: the thresholds
            # are float32, which rounds 0.55, 0.6, 0.8 and 0.85 up, so
            # a float64 IoU exactly on one of them would fall below it.
            return oriented_box_iou_batch(
                cast(npt.NDArray[np.number], target_contents),
                cast(npt.NDArray[np.number], prediction_contents),
            ).astype(np.float32)
        raise ValueError("Unsupported metric target for IoU calculation")

    def _collect_image_stats(
        self,
        predictions: Detections,
        base_targets: Detections,
        size_categories: tuple[ObjectSizeCategory, ...],
        iou_thresholds: npt.NDArray[np.float32],
        stats_by_size: dict[ObjectSizeCategory, list[Any]],
    ) -> None:
        """Validate one image, compute its IoU once, and collect stats per size bucket.

        Processes all requested size buckets per image so each image's IoU matrix is
        released before advancing, keeping memory O(1 image) rather than O(dataset).
        """
        # Always validate target/prediction contents before empty-side branches so
        # maskless detections under `MetricTarget.MASKS` still raise on background,
        # ignored-only, or prediction-free images.
        prediction_contents = self._detections_content(predictions)
        base_target_contents = self._detections_content(base_targets)

        if len(base_targets) == 0:
            return

        if predictions.class_id is None or base_targets.class_id is None:
            raise ValueError(
                "MeanAverageRecall metric requires `class_id` on both "
                "predictions and targets."
            )
        base_target_class_ids = np.asarray(base_targets.class_id, dtype=np.int32)

        needs_size = any(cat != ObjectSizeCategory.ANY for cat in size_categories)
        target_areas = (
            get_detection_areas(base_targets, self._metric_target)
            if needs_size
            else None
        )
        small_area, medium_area = SIZE_THRESHOLDS
        max_all_area = 1e5**2

        def _mask_for_size(
            size_category: ObjectSizeCategory,
        ) -> npt.NDArray[np.bool_] | None:
            """Return inclusive COCO target size mask for ``size_category``."""
            if size_category == ObjectSizeCategory.ANY:
                return None
            assert target_areas is not None
            if size_category == ObjectSizeCategory.SMALL:
                return (target_areas >= 0) & (target_areas <= small_area)
            if size_category == ObjectSizeCategory.MEDIUM:
                return (target_areas >= small_area) & (target_areas <= medium_area)
            if size_category == ObjectSizeCategory.LARGE:
                return (target_areas >= medium_area) & (target_areas <= max_all_area)
            raise ValueError(f"Invalid size category: {size_category}")

        if len(predictions) == 0:
            for size_category in size_categories:
                size_mask = _mask_for_size(size_category)
                target_class_ids = (
                    base_target_class_ids[size_mask]
                    if size_mask is not None
                    else base_target_class_ids
                )
                if len(target_class_ids) == 0:
                    continue
                stats_by_size[size_category].append(
                    (
                        np.zeros(
                            (0, iou_thresholds.size, self.max_detections.size),
                            dtype=bool,
                        ),
                        np.zeros((0, iou_thresholds.size), dtype=bool),
                        np.zeros((0,), dtype=int),
                        np.zeros((0,), dtype=int),
                        target_class_ids,
                    )
                )
            return

        if predictions.confidence is None:
            raise ValueError(
                "MeanAverageRecall metric requires `confidence` on predictions."
            )
        prediction_class_ids = np.asarray(predictions.class_id, dtype=np.int32)
        prediction_confidence = np.asarray(predictions.confidence, dtype=np.float32)

        # Sort predictions and compute pairwise IoU once for all size buckets.
        full_iou = self._compute_iou_matrix(base_target_contents, prediction_contents)
        # A stable sort keeps tied scores in input order, as COCO's
        # mergesort does, so equal-score boxes cross the K cutoff alike.
        sorted_indices = np.argsort(-prediction_confidence, kind="stable")
        sorted_class_ids = prediction_class_ids[sorted_indices]
        # COCO applies the detection limit to each (image, class) pair,
        # so every prediction is ranked among its own class only.
        class_ranks = self._rank_within_class(sorted_class_ids)
        sorted_iou = full_iou[:, sorted_indices]
        ignored_matches = np.zeros(
            (len(sorted_indices), iou_thresholds.size), dtype=bool
        )

        for size_category in size_categories:
            size_mask = _mask_for_size(size_category)
            if size_mask is not None:
                if not np.any(size_mask):
                    continue
                target_class_ids = base_target_class_ids[size_mask]
                bucket_iou = sorted_iou[size_mask]
            else:
                target_class_ids = base_target_class_ids
                bucket_iou = sorted_iou

            matches = self._match_top_predictions(
                sorted_class_ids,
                class_ranks,
                target_class_ids,
                bucket_iou,
                iou_thresholds,
            )
            stats_by_size[size_category].append(
                (
                    matches,
                    ignored_matches,
                    class_ranks,
                    sorted_class_ids,
                    target_class_ids,
                )
            )

    def _finalize_result(
        self,
        stats: list[Any],
        iou_thresholds: npt.NDArray[np.float32],
    ) -> MeanAverageRecallResult:
        """Aggregate collected per-image stats into a MeanAverageRecallResult."""
        if not stats:
            max_detection_count = self.max_detections.shape[0]
            return MeanAverageRecallResult(
                metric_target=self._metric_target,
                recall_scores=np.zeros(max_detection_count),
                recall_per_class=np.zeros(
                    (max_detection_count, 0, iou_thresholds.shape[0])
                ),
                max_detections=self.max_detections,
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
                small_objects=None,
                medium_objects=None,
                large_objects=None,
            )

        concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
        recall_scores_per_k, recall_per_class_at_k, unique_classes = (
            self._compute_average_recall_for_classes(*concatenated_stats)
        )

        return MeanAverageRecallResult(
            metric_target=self._metric_target,
            recall_scores=recall_scores_per_k,
            recall_per_class=recall_per_class_at_k,
            max_detections=self.max_detections,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

    def compute(self) -> MeanAverageRecallResult:
        """Calculate the Mean Average Recall metric based on the stored predictions and
        ground-truth, at different IoU thresholds and maximum detection counts.

        Returns:
            The Mean Average Recall metric result.
        """
        categories = (
            ObjectSizeCategory.ANY,
            ObjectSizeCategory.SMALL,
            ObjectSizeCategory.MEDIUM,
            ObjectSizeCategory.LARGE,
        )
        iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)
        stats_by_size: dict[ObjectSizeCategory, list[Any]] = {
            cat: [] for cat in categories
        }
        # Compute all 4 size buckets per image and release each IoU matrix
        # immediately before advancing to keep memory O(1 image).
        for predictions, targets in zip(self._predictions_list, self._targets_list):
            base_targets = self._drop_ignored_targets(targets)
            self._collect_image_stats(
                predictions, base_targets, categories, iou_thresholds, stats_by_size
            )

        result = self._finalize_result(
            stats_by_size[ObjectSizeCategory.ANY], iou_thresholds
        )
        result.small_objects = self._finalize_result(
            stats_by_size[ObjectSizeCategory.SMALL], iou_thresholds
        )
        result.medium_objects = self._finalize_result(
            stats_by_size[ObjectSizeCategory.MEDIUM], iou_thresholds
        )
        result.large_objects = self._finalize_result(
            stats_by_size[ObjectSizeCategory.LARGE], iou_thresholds
        )
        return result

    def _compute(
        self,
        predictions_list: list[Detections],
        targets_list: list[Detections],
        size_category: ObjectSizeCategory = ObjectSizeCategory.ANY,
    ) -> MeanAverageRecallResult:
        """Compute MeanAverageRecall for a single size category."""
        iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)
        stats_by_size: dict[ObjectSizeCategory, list[Any]] = {size_category: []}
        for predictions, targets in zip(predictions_list, targets_list):
            base_targets = self._drop_ignored_targets(targets)
            self._collect_image_stats(
                predictions,
                base_targets,
                (size_category,),
                iou_thresholds,
                stats_by_size,
            )
        return self._finalize_result(stats_by_size[size_category], iou_thresholds)

    @staticmethod
    def _rank_within_class(
        sorted_class_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Rank each prediction among the predictions of its own class.

        COCO evaluates every (image, category) pair on its own and keeps the top
        `maxDets` detections of that category, so the detection limit has to count
        predictions per class rather than per image.

        Args:
            sorted_class_ids: shape (P,), prediction class ids sorted by descending
                confidence.

        Returns:
            shape (P,), the zero-based rank of each prediction within its class,
                where rank 0 is that class's highest confidence prediction.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from supervision.metrics import MeanAverageRecall
            >>> class_ids = np.array([0, 1, 0, 0, 1])
            >>> MeanAverageRecall._rank_within_class(class_ids).tolist()
            [0, 0, 1, 2, 1]

            ```
        """
        positions = np.arange(sorted_class_ids.shape[0])
        # A stable sort groups each class while keeping its confidence order intact.
        grouped_order = np.argsort(sorted_class_ids, kind="stable")
        grouped_class_ids = sorted_class_ids[grouped_order]
        is_group_start = np.ones(positions.shape, dtype=bool)
        is_group_start[1:] = grouped_class_ids[1:] != grouped_class_ids[:-1]
        group_start = np.maximum.accumulate(np.where(is_group_start, positions, 0))

        class_ranks = np.empty(positions.shape, dtype=np.int64)
        class_ranks[grouped_order] = positions - group_start
        return class_ranks

    def _match_top_predictions(
        self,
        sorted_prediction_class_ids: npt.NDArray[np.int32],
        class_ranks: npt.NDArray[np.int64],
        target_class_ids: npt.NDArray[np.int32],
        sorted_iou: npt.NDArray[np.float32],
        iou_thresholds: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        """Match one image's predictions separately for each detection limit.

        For each IoU threshold, predictions are visited in descending confidence
        order and each claims its highest-IoU eligible unmatched target. Each limit
        matches only the top K predictions of each class, as COCO does for every
        (image, class) pair. Matching never pairs predictions with targets of another
        class, so dropping one class's excess predictions cannot change how another
        class is matched. Equal-IoU ties select the last eligible target in input
        order, matching COCO.

        Args:
            sorted_prediction_class_ids: shape (P,), prediction class ids sorted by
                descending confidence.
            class_ranks: shape (P,), rank of each prediction within its own class, in
                the same order.
            target_class_ids: shape (T,), target class ids.
            sorted_iou: shape (T, P), IoU with prediction columns in the same order.
            iou_thresholds: shape (Th,), IoU thresholds.

        Returns:
            shape (P, Th, K), whether each prediction is a true positive when only
                the top `max_detections[k]` predictions of each class are matched;
                predictions ranked below that limit within their class are `False`.
        """
        prediction_count = sorted_prediction_class_ids.shape[0]
        num_thresholds = iou_thresholds.size
        num_limits = self.max_detections.size
        if prediction_count == 0 or num_thresholds == 0 or num_limits == 0:
            return np.zeros(
                (prediction_count, num_thresholds, num_limits),
                dtype=bool,
            )

        max_limit = int(self.max_detections.max())
        min_threshold = float(iou_thresholds.min())
        thresholds_list = iou_thresholds.tolist()

        # Matching is independent per class and strictly sequential in score order,
        # so the match state for any limit K <= max_limit is the exact prefix of
        # matching up to max_limit once.
        full_matches = np.zeros((prediction_count, num_thresholds), dtype=bool)
        if max_limit > 0 and target_class_ids.size > 0:
            targets_by_class: dict[int, npt.NDArray[np.intp]] = {}
            for cls_id in np.unique(target_class_ids).tolist():
                # Reverse target order so argmax picks the last target on exact ties,
                # matching COCO's tie-breaking rule.
                targets_by_class[cls_id] = np.flatnonzero(target_class_ids == cls_id)[
                    ::-1
                ]

            within_max = np.flatnonzero(class_ranks < max_limit)
            for cls_id, cls_targets in targets_by_class.items():
                cls_preds = within_max[
                    sorted_prediction_class_ids[within_max] == cls_id
                ]
                if len(cls_preds) == 0:
                    continue
                cls_iou = sorted_iou[np.ix_(cls_targets, cls_preds)]
                pred_max_iou = cls_iou.max(axis=0)
                candidate_cols = np.flatnonzero(pred_max_iou >= min_threshold)
                if len(candidate_cols) == 0:
                    continue
                matched = np.zeros((num_thresholds, len(cls_targets)), dtype=bool)
                for col_pos in candidate_cols.tolist():
                    col = cls_iou[:, col_pos]
                    col_max = float(pred_max_iou[col_pos])
                    pred_idx = int(cls_preds[col_pos])
                    for t_idx, thresh in enumerate(thresholds_list):
                        if col_max < thresh:
                            continue
                        eligible = (~matched[t_idx]) & (col >= thresh)
                        if not eligible.any():
                            continue
                        best_local = int(np.argmax(np.where(eligible, col, -1.0)))
                        matched[t_idx, best_local] = True
                        full_matches[pred_idx, t_idx] = True

        matches: npt.NDArray[np.bool_] = full_matches[:, :, None] & (
            class_ranks[:, None, None] < self.max_detections[None, None, :]
        )
        return matches

    def _compute_average_recall_for_classes(
        self,
        matches: npt.NDArray[np.bool_],
        ignored_matches: npt.NDArray[np.bool_],
        class_ranks: npt.NDArray[np.int64],
        prediction_class_ids: npt.NDArray[np.int32],
        true_class_ids: npt.NDArray[np.int32],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
    ]:
        """Compute recall per detection limit and class from all images' matches.

        `matches` has shape (P, Th, K) and holds, for each detection limit, the matches
        of that limit's own top predictions of each class; the other arrays have one row
        or entry per prediction, or per target for `true_class_ids`. `class_ranks` is
        each prediction's rank within its class in its own image.
        """
        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)

        if unique_classes.size == 0:
            max_detection_count = self.max_detections.shape[0]
            num_thresholds = matches.shape[1]
            return (
                np.zeros(max_detection_count, dtype=np.float64),
                np.zeros((max_detection_count, 0, num_thresholds), dtype=np.float64),
                unique_classes,
            )

        recalls_at_k: list[npt.NDArray[np.float64]] = []
        for limit_index, max_detections in enumerate(self.max_detections):
            # Shape: PxThxK,P,C,C -> CxThx3
            is_within_limit = class_ranks < max_detections
            confusion_matrix = self._compute_confusion_matrix(
                matches[is_within_limit, :, limit_index],
                ignored_matches[is_within_limit],
                prediction_class_ids[is_within_limit],
                unique_classes,
                class_counts,
            )

            # Shape: CxThx3 -> CxTh
            recall_per_class = self._compute_recall(confusion_matrix)
            recalls_at_k.append(recall_per_class)

        # Shape: KxCxTh -> KxC
        recall_per_class_at_k = np.array(recalls_at_k)
        average_recall_per_class = np.mean(recall_per_class_at_k, axis=2)

        # Shape: KxC -> K
        recall_scores = np.mean(average_recall_per_class, axis=1)

        return recall_scores, recall_per_class_at_k, unique_classes

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
            max_detections: The maximum number of detections to
                consider for each class. Extra detections are considered false
                positives. By default, all detections are considered.

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
                limited_matches = sorted_matches[is_class]
                limited_ignored_matches = sorted_ignored_matches[is_class]
                true_positives = limited_matches.sum(0)

                false_positives = (~limited_matches & ~limited_ignored_matches).sum(0)
                false_negatives = num_true - true_positives

            confusion_matrix[class_idx] = np.stack(
                [true_positives, false_positives, false_negatives], axis=1
            )

        result_confusion_matrix: npt.NDArray[np.float64] = confusion_matrix
        return result_confusion_matrix

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
        if not confusion_matrix.shape[-1] == 3:
            raise ValueError(
                f"Confusion matrix must have shape (..., 3), got "
                f"{confusion_matrix.shape}"
            )
        true_positives = confusion_matrix[..., 0]
        false_negatives = confusion_matrix[..., 2]

        denominator = true_positives + false_negatives
        recall = np.divide(
            true_positives,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator != 0,
        )

        result_recall: npt.NDArray[np.float64] = recall
        return result_recall

    def _detections_content(
        self, detections: Detections
    ) -> npt.NDArray[Any] | CompactMask:
        """Return boxes, masks or oriented bounding boxes from detections.

        For the mask target this may return a
        :class:`~supervision.detection.compact_mask.CompactMask` rather than a
        dense boolean array when the detections carry compact masks. Oriented
        corners retain their precision until IoU translates them into a local frame;
        float16 corners are promoted before subtracting potentially large extents.
        """
        if self._metric_target == MetricTarget.BOXES:
            return cast(npt.NDArray[Any], detections.xyxy)
        if self._metric_target == MetricTarget.MASKS:
            if detections.mask is not None:
                # detections.mask is NDArray[bool] | CompactMask; return as-is.
                return detections.mask
            if len(detections) > 0:
                raise ValueError(
                    "MeanAverageRecall with `MetricTarget.MASKS` requires "
                    "detections to include masks."
                )
            return self._make_empty_content()
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = detections.data.get(ORIENTED_BOX_COORDINATES)
            if obb is not None and len(obb) > 0:
                result_obb: npt.NDArray[Any] = np.array(obb)
                if np.issubdtype(result_obb.dtype, np.float16):
                    result_obb = result_obb.astype(np.float32)
                return result_obb
            return self._make_empty_content()
        raise ValueError(f"Invalid metric target: {self._metric_target}")

    def _make_empty_content(self) -> npt.NDArray[Any]:
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

    @staticmethod
    def _drop_ignored_targets(targets: Detections) -> Detections:
        """Return targets without the COCO ignore regions among them.

        A target is an ignore region when ``data["iscrowd"]`` or ``data["ignore"]``
        is set for it, the same rule
        :class:`~supervision.metrics.mean_average_precision.MeanAveragePrecision`
        applies.
        """
        data = targets.data or {}
        ignored = np.zeros(len(targets), dtype=bool)
        for field in (ISCROWD_DATA_FIELD, "ignore"):
            if field in data:
                ignored |= np.asarray(data[field]).astype(bool)
        if not ignored.any():
            return targets
        # A boolean mask always selects a `Detections`; narrow for mypy.
        return cast(Detections, targets[~ignored])

    def _size_bucket_mask(
        self, detections: Detections, size_category: ObjectSizeCategory
    ) -> npt.NDArray[np.bool_]:
        """Return a boolean mask for rows in the inclusive COCO size range."""
        if detections.is_empty() or size_category == ObjectSizeCategory.ANY:
            return np.ones(len(detections), dtype=bool)
        areas = get_detection_areas(detections, self._metric_target)
        small_area, medium_area = SIZE_THRESHOLDS
        max_all_area = 1e5**2
        if size_category == ObjectSizeCategory.SMALL:
            return (areas >= 0) & (areas <= small_area)
        if size_category == ObjectSizeCategory.MEDIUM:
            return (areas >= small_area) & (areas <= medium_area)
        if size_category == ObjectSizeCategory.LARGE:
            return (areas >= medium_area) & (areas <= max_all_area)
        raise ValueError(f"Invalid size category: {size_category}")

    def _filter_detections_by_size(
        self, detections: Detections, size_category: ObjectSizeCategory
    ) -> Detections:
        """Select a copy of the rows in the inclusive COCO size range.

        Slice the selected rows on a shallow copy to avoid copying unused
        masks and metadata. The targets were validated at construction, so
        do not revalidate accepted non-bool masks during each size filter.

        Matches
        :class:`~supervision.metrics.mean_average_precision.MeanAveragePrecision`
        and pycocotools: small is ``[0, 32²]``, medium ``[32², 96²]``, large
        ``[96², 1e5²]``. Boundary areas therefore belong to both neighbouring
        buckets. The half-open bins in
        :class:`~supervision.metrics.utils.object_size.ObjectSizeCategory` stay
        for general filtering; mAR needs the inclusive COCO semantics.
        """
        size_mask = self._size_bucket_mask(detections, size_category)

        selected = copy(detections)
        selected.xyxy = detections.xyxy[size_mask]
        if detections.mask is not None:
            selected.mask = detections.mask[size_mask]
        if detections.class_id is not None:
            selected.class_id = detections.class_id[size_mask]
        if detections.confidence is not None:
            selected.confidence = detections.confidence[size_mask]
        if detections.tracker_id is not None:
            selected.tracker_id = detections.tracker_id[size_mask]
        if detections.data is not None:
            selected.data = get_data_item(detections.data, size_mask)
        return selected
