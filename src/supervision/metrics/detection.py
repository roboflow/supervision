from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from supervision.dataset.core import DetectionDataset
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch
from supervision.utils.internal import deprecated


def detections_to_tensor(
    detections: Detections, with_confidence: bool = False
) -> npt.NDArray[np.float32]:
    """
    Convert Supervision Detections to numpy tensors for further computation

    Args:
        detections: Detections/Targets in the format of sv.Detections
        with_confidence: Whether to include confidence in the tensor

    Returns:
        Detections as numpy tensors as in (xyxy, class_id, confidence) order
    """
    if detections.class_id is None:
        raise ValueError(
            "ConfusionMatrix can only be calculated for Detections with class_id"
        )

    arrays_to_concat = [detections.xyxy, np.expand_dims(detections.class_id, 1)]

    if with_confidence:
        if detections.confidence is None:
            raise ValueError(
                "ConfusionMatrix can only be calculated for Detections with confidence"
            )
        arrays_to_concat.append(np.expand_dims(detections.confidence, 1))

    result: npt.NDArray[np.float32] = np.concatenate(arrays_to_concat, axis=1)
    return result


def validate_input_tensors(
    predictions: list[npt.NDArray[np.float32]],
    targets: list[npt.NDArray[np.float32]],
) -> None:
    """
    Checks for shape consistency of input tensors.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) and"
            f"targets ({len(targets)}) must be equal."
        )
    if len(predictions) > 0:
        if not isinstance(predictions[0], np.ndarray) or not isinstance(
            targets[0], np.ndarray
        ):
            raise ValueError(
                f"Predictions and targets must be lists of numpy arrays."
                f"Got {type(predictions[0])} and {type(targets[0])} instead."
            )
        if predictions[0].shape[1] != 6:
            raise ValueError(
                f"Predictions must have shape (N, 6)."
                f"Got {predictions[0].shape} instead."
            )
        if targets[0].shape[1] != 5:
            raise ValueError(
                f"Targets must have shape (N, 5). Got {targets[0].shape} instead."
            )


@dataclass
class ConfusionMatrix:
    """
    Confusion matrix for object detection tasks.

    Attributes:
        matrix: An 2D `np.ndarray` of shape `(len(classes) + 1, len(classes) + 1)`
            containing the number of `TP`, `FP`, `FN` and `TN` for each class.
        classes: Model class names.
        conf_threshold: Detection confidence threshold between `0` and `1`.
            Detections with lower confidence will be excluded from the matrix.
        iou_threshold: Detection IoU threshold between `0` and `1`.
            Detections with lower IoU will be classified as `FP`.
    """

    matrix: npt.NDArray[np.int32]
    classes: list[str]
    conf_threshold: float
    iou_threshold: float

    @classmethod
    def from_detections(
        cls,
        predictions: list[Detections],
        targets: list[Detections],
        classes: list[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix based on predicted and ground-truth detections.

        Args:
            targets: Detections objects from ground-truth.
            predictions: Detections objects predicted by the model.
            classes: Model class names.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection IoU threshold between `0` and `1`.
                Detections with lower IoU will be classified as `FP`.

        Returns:
            New instance of ConfusionMatrix.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> targets = [
            ...     sv.Detections(
            ...         xyxy=np.array([[0, 0, 10, 10]]),
            ...         class_id=np.array([0])
            ...     )
            ... ]
            >>> predictions = [
            ...     sv.Detections(
            ...         xyxy=np.array([[0, 0, 10, 10]]),
            ...         class_id=np.array([0]),
            ...         confidence=np.array([0.9])
            ...     )
            ... ]
            >>> confusion_matrix = sv.ConfusionMatrix.from_detections(
            ...     predictions=predictions,
            ...     targets=targets,
            ...     classes=['person']
            ... )
            >>> confusion_matrix.matrix
            array([[1., 0.],
                   [0., 0.]])

            ```
        """
        prediction_tensors = []
        target_tensors = []
        for prediction, target in zip(predictions, targets):
            prediction_tensors.append(
                detections_to_tensor(prediction, with_confidence=True)
            )
            target_tensors.append(detections_to_tensor(target, with_confidence=False))
        return cls.from_tensors(
            predictions=prediction_tensors,
            targets=target_tensors,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    @classmethod
    def from_tensors(
        cls,
        predictions: list[npt.NDArray[np.float32]],
        targets: list[npt.NDArray[np.float32]],
        classes: list[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix based on predicted and ground-truth detections.

        Args:
            predictions: Each element of the list describes a single
                image and has `shape = (M, 6)` where `M` is the number of detected
                objects. Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets: Each element of the list describes a single
                image and has `shape = (N, 5)` where `N` is the number of
                ground-truth objects. Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
            classes: Model class names.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection iou  threshold between `0` and `1`.
                Detections with lower iou will be classified as `FP`.

        Returns:
            New instance of ConfusionMatrix.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> import numpy as np
            >>> targets = [
            ...     np.array([
            ...         [0.0, 0.0, 3.0, 3.0, 0],
            ...         [2.0, 2.0, 5.0, 5.0, 0],
            ...         [6.0, 1.0, 8.0, 3.0, 1],
            ...     ])
            ... ]
            >>> predictions = [
            ...     np.array([
            ...         [0.0, 0.0, 3.0, 3.0, 0, 0.9],
            ...         [0.1, 0.1, 3.0, 3.0, 0, 0.9],
            ...         [6.0, 1.0, 8.0, 3.0, 1, 0.8],
            ...     ])
            ... ]
            >>> confusion_matrix = sv.ConfusionMatrix.from_tensors(
            ...     predictions=predictions,
            ...     targets=targets,
            ...     classes=['person', 'dog']
            ... )
            >>> confusion_matrix.matrix
            array([[1., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]])

            ```
        """
        validate_input_tensors(predictions, targets)

        num_classes = len(classes)
        matrix = np.zeros((num_classes + 1, num_classes + 1))
        for true_batch, detection_batch in zip(targets, predictions):
            matrix += cls.evaluate_detection_batch(
                predictions=detection_batch,
                targets=true_batch,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        return cls(
            matrix=matrix,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    @staticmethod
    def evaluate_detection_batch(
        predictions: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        num_classes: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> npt.NDArray[np.int32]:
        """
        Calculate confusion matrix for a batch of detections for a single image.

        Args:
            predictions: Batch prediction. Describes a single image and
                has `shape = (M, 6)` where `M` is the number of detected objects.
                Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets: Batch target labels. Describes a single image and
                has `shape = (N, 5)` where `N` is the number of ground-truth objects.
                Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
            num_classes: Number of classes.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection iou  threshold between `0` and `1`.
                Detections with lower iou will be classified as `FP`.

        Returns:
            Confusion matrix based on a single image.
        """
        result_matrix = np.zeros((num_classes + 1, num_classes + 1))

        # Filter predictions by confidence threshold
        conf_idx = 5
        confidence = predictions[:, conf_idx]
        detection_batch_filtered = predictions[confidence >= conf_threshold]

        if len(detection_batch_filtered) == 0:
            # No detections pass confidence threshold - all GT are FN
            class_id_idx = 4
            true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
            for gt_class in true_classes:
                result_matrix[gt_class, num_classes] += 1
            return result_matrix

        if len(targets) == 0:
            # No ground truth - all detections are FP
            class_id_idx = 4
            detection_classes = np.array(
                detection_batch_filtered[:, class_id_idx], dtype=np.int16
            )
            for det_class in detection_classes:
                result_matrix[num_classes, det_class] += 1
            return result_matrix

        class_id_idx = 4
        true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
        detection_classes = np.array(
            detection_batch_filtered[:, class_id_idx], dtype=np.int16
        )
        true_boxes = targets[:, :class_id_idx]
        detection_boxes = detection_batch_filtered[:, :class_id_idx]

        # Calculate IoU matrix
        iou_batch = box_iou_batch(
            boxes_true=true_boxes, boxes_detection=detection_boxes
        )

        # Find all valid matches (IoU > threshold, regardless of class)
        # Use vectorized operations to avoid nested Python loops
        iou_mask = iou_batch > iou_threshold
        gt_indices, det_indices = np.nonzero(iou_mask)

        # If no pairs exceed the IoU threshold, skip matching
        if gt_indices.size == 0:
            valid_matches = []
        else:
            ious = iou_batch[gt_indices, det_indices]
            gt_match_classes = true_classes[gt_indices]
            det_match_classes = detection_classes[det_indices]
            class_matches = gt_match_classes == det_match_classes

            # Sort matches by class match first (True before False),
            # then by IoU descending.
            # np.lexsort sorts by the last key first, in ascending order.
            # We use ~class_matches so that True becomes 0
            # and False becomes 1 (True first),
            # and -ious so that larger IoUs come first.
            sort_indices = np.lexsort((-ious, ~class_matches))

            # Build list of matches in the same format as before:
            # (gt_idx, det_idx, iou, class_match)
            valid_matches = [
                (
                    int(gt_indices[idx]),
                    int(det_indices[idx]),
                    float(ious[idx]),
                    bool(class_matches[idx]),
                )
                for idx in sort_indices
            ]
        # Greedily assign matches, ensuring each GT
        # and detection is matched at most once
        matched_gt_idx = set()
        matched_det_idx = set()

        for gt_idx, det_idx, iou, class_match in valid_matches:
            if gt_idx not in matched_gt_idx and det_idx not in matched_det_idx:
                # Valid spatial match - record the class prediction
                gt_class = true_classes[gt_idx]
                det_class = detection_classes[det_idx]

                # This handles both correct classification (TP) and misclassification
                result_matrix[gt_class, det_class] += 1
                matched_gt_idx.add(gt_idx)
                matched_det_idx.add(det_idx)

        # Count unmatched ground truth as FN
        for gt_idx, gt_class in enumerate(true_classes):
            if gt_idx not in matched_gt_idx:
                result_matrix[gt_class, num_classes] += 1

        # Count unmatched detections as FP
        for det_idx, det_class in enumerate(detection_classes):
            if det_idx not in matched_det_idx:
                result_matrix[num_classes, det_class] += 1

        return result_matrix

    @staticmethod
    def _drop_extra_matches(
        matches: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        Deduplicate matches. If there are multiple matches for the same true or
        predicted box, only the one with the highest IoU is kept.
        """
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        result: npt.NDArray[np.float32] = matches
        return result

    @classmethod
    def benchmark(
        cls,
        dataset: DetectionDataset,
        callback: Callable[[npt.NDArray[np.uint8]], Detections],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix from dataset and callback function.

        Args:
            dataset: Object detection dataset used for evaluation.
            callback: Function that takes an image as input and returns a
                Detections object.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection IoU threshold between `0` and `1`.
                Detections with lower IoU will be classified as `FP`.

        Returns:
            New instance of ConfusionMatrix.

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            dataset = sv.DetectionDataset.from_yolo(...)

            model = YOLO(...)
            def callback(image: np.ndarray) -> sv.Detections:
                result = model(image)[0]
                return sv.Detections.from_ultralytics(result)

            confusion_matrix = sv.ConfusionMatrix.benchmark(
                dataset = dataset,
                callback = callback
            )

            print(confusion_matrix.matrix)
            # np.array([
            #     [0., 0., 0., 0.],
            #     [0., 1., 0., 1.],
            #     [0., 1., 1., 0.],
            #     [1., 1., 0., 0.]
            # ])
            ```
        """
        predictions, targets = [], []
        for _, image, annotation in dataset:
            predictions_batch = callback(image)
            predictions.append(predictions_batch)
            targets.append(annotation)
        return cls.from_detections(
            predictions=predictions,
            targets=targets,
            classes=dataset.classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    def plot(
        self,
        save_path: str | None = None,
        title: str | None = None,
        classes: list[str] | None = None,
        normalize: bool = False,
        fig_size: tuple[int, int] = (12, 10),
    ) -> matplotlib.figure.Figure:
        """
        Create confusion matrix plot and save it at selected location.

        Args:
            save_path: Path to save the plot. If not provided,
                plot will be displayed.
            title: Title of the plot.
            classes: List of classes to be displayed on the plot.
                If not provided, all classes will be displayed.
            normalize: If True, normalize the confusion matrix.
            fig_size: Size of the plot.

        Returns:
            Confusion matrix plot.
        """

        array = self.matrix.copy()

        if normalize:
            eps = 1e-8
            array = array / (array.sum(0).reshape(1, -1) + eps)

        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(figsize=fig_size, tight_layout=True, facecolor="white")

        class_names = classes if classes is not None else self.classes
        use_labels_for_ticks = class_names is not None and (0 < len(class_names) < 99)
        if use_labels_for_ticks:
            x_tick_labels = [*class_names, "FN"]
            y_tick_labels = [*class_names, "FP"]
            num_ticks = len(x_tick_labels)
        else:
            x_tick_labels = None
            y_tick_labels = None
            num_ticks = len(array)
        im = ax.imshow(array, cmap="Blues")

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.mappable.set_clim(vmin=0, vmax=np.nanmax(array))

        if x_tick_labels is None:
            tick_interval = 2
        else:
            tick_interval = 1
        ax.set_xticks(np.arange(0, num_ticks, tick_interval), labels=x_tick_labels)
        ax.set_yticks(np.arange(0, num_ticks, tick_interval), labels=y_tick_labels)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="default")

        labelsize = 10 if num_ticks < 50 else 8
        ax.tick_params(axis="both", which="both", labelsize=labelsize)

        if num_ticks < 30:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    n_preds = array[i, j]
                    if not np.isnan(n_preds):
                        ax.text(
                            j,
                            i,
                            f"{n_preds:.2f}" if normalize else f"{n_preds:.0f}",
                            ha="center",
                            va="center",
                            color="black"
                            if n_preds < 0.5 * np.nanmax(array)
                            else "white",
                        )

        if title:
            ax.set_title(title, fontsize=20)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_facecolor("white")
        if save_path:
            fig.savefig(
                save_path, dpi=250, facecolor=fig.get_facecolor(), transparent=True
            )
        return fig


@deprecated(
    "`MeanAveragePrecision` is deprecated and will be removed in "
    "`supervision-0.31.0`. Use "
    "`supervision.metrics.mean_average_precision.MeanAveragePrecision` instead. The "
    "deprecated implementation provides results that are inconsistent with pycocotools."
)
@dataclass(frozen=True)
class MeanAveragePrecision:
    """
    !!! deprecated "Deprecated"
        `MeanAveragePrecision` is **deprecated** and will be removed in
        `supervision-0.31.0`.

        The deprecated implementation provides results that are inconsistent with
        `pycocotools`. Please use
        `supervision.metrics.mean_average_precision.MeanAveragePrecision` instead,
        which matches the results of `pycocotools` and is now the recommended approach.

    Mean Average Precision for object detection tasks.

    Attributes:
        map50_95: Mean Average Precision (mAP) calculated over IoU thresholds
            ranging from `0.50` to `0.95` with a step size of `0.05`.
        map50: Mean Average Precision (mAP) calculated specifically at
            an IoU threshold of `0.50`.
        map75: Mean Average Precision (mAP) calculated specifically at
            an IoU threshold of `0.75`.
        per_class_ap50_95: Average Precision (AP) values calculated over
            IoU thresholds ranging from `0.50` to `0.95` with a step size of `0.05`,
            provided for each individual class.
    """

    map50_95: float
    map50: float
    map75: float
    per_class_ap50_95: npt.NDArray[np.float64]

    @classmethod
    def from_detections(
        cls,
        predictions: list[Detections],
        targets: list[Detections],
    ) -> MeanAveragePrecision:
        """
        Calculate mean average precision based on predicted and ground-truth detections.

        Args:
            targets: Detections objects from ground-truth.
            predictions: Detections objects predicted by the model.
        Returns:
            New instance of ConfusionMatrix.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> targets = [
            ...     sv.Detections(
            ...         xyxy=np.array([[0, 0, 10, 10]]),
            ...         class_id=np.array([0])
            ...     )
            ... ]
            >>> predictions = [
            ...     sv.Detections(
            ...         xyxy=np.array([[0, 0, 10, 10]]),
            ...         class_id=np.array([0]),
            ...         confidence=np.array([0.9])
            ...     )
            ... ]
            >>> mAP = sv.MeanAveragePrecision.from_detections(
            ...     predictions=predictions,
            ...     targets=targets,
            ... )
            >>> round(float(mAP.map50), 2)
            0.99

            ```
        """
        prediction_tensors = []
        target_tensors = []
        for prediction, target in zip(predictions, targets):
            prediction_tensors.append(
                detections_to_tensor(prediction, with_confidence=True)
            )
            target_tensors.append(detections_to_tensor(target, with_confidence=False))
        return cls.from_tensors(
            predictions=prediction_tensors,
            targets=target_tensors,
        )

    @classmethod
    def benchmark(
        cls,
        dataset: DetectionDataset,
        callback: Callable[[npt.NDArray[np.uint8]], Detections],
    ) -> MeanAveragePrecision:
        """
        Calculate mean average precision from dataset and callback function.

        Args:
            dataset: Object detection dataset used for evaluation.
            callback: Function that takes
                an image as input and returns Detections object.
        Returns:
            New instance of MeanAveragePrecision.

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            dataset = sv.DetectionDataset.from_yolo(...)

            model = YOLO(...)
            def callback(image: np.ndarray) -> sv.Detections:
                result = model(image)[0]
                return sv.Detections.from_ultralytics(result)

            mean_average_precision = sv.MeanAveragePrecision.benchmark(
                dataset = dataset,
                callback = callback
            )

            print(mean_average_precision.map50_95)
            # 0.433
            ```
        """
        predictions, targets = [], []
        for _, image, annotation in dataset:
            predictions_batch = callback(image)
            predictions.append(predictions_batch)
            targets.append(annotation)
        return cls.from_detections(
            predictions=predictions,
            targets=targets,
        )

    @classmethod
    def from_tensors(
        cls,
        predictions: list[npt.NDArray[np.float32]],
        targets: list[npt.NDArray[np.float32]],
    ) -> MeanAveragePrecision:
        """
        Calculate Mean Average Precision based on predicted and ground-truth
            detections at different threshold.

        Args:
            predictions: Each element of the list describes
                a single image and has `shape = (M, 6)` where `M` is
                the number of detected objects. Each row is expected to be
                in `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets: Each element of the list describes a single
                image and has `shape = (N, 5)` where `N` is the
                number of ground-truth objects. Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
        Returns:
            New instance of MeanAveragePrecision.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> import numpy as np
            >>> targets = [
            ...     np.array([
            ...         [0.0, 0.0, 3.0, 3.0, 0],
            ...         [2.0, 2.0, 5.0, 5.0, 0],
            ...         [6.0, 1.0, 8.0, 3.0, 1],
            ...     ])
            ... ]
            >>> predictions = [
            ...     np.array([
            ...         [0.0, 0.0, 3.0, 3.0, 0, 0.9],
            ...         [0.1, 0.1, 3.0, 3.0, 0, 0.9],
            ...         [6.0, 1.0, 8.0, 3.0, 1, 0.8],
            ...     ])
            ... ]
            >>> mAP = sv.MeanAveragePrecision.from_tensors(
            ...     predictions=predictions,
            ...     targets=targets,
            ... )
            >>> round(float(mAP.map50), 2)
            0.81

            ```
        """
        validate_input_tensors(predictions, targets)
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        stats = []

        # Gather matching stats for predictions and targets
        for true_objs, predicted_objs in zip(targets, predictions):
            if predicted_objs.shape[0] == 0:
                if true_objs.shape[0]:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            *np.zeros((2, 0)),
                            true_objs[:, 4],
                        )
                    )
                continue

            if true_objs.shape[0]:
                matches = cls._match_detection_batch(
                    predicted_objs, true_objs, iou_thresholds
                )
                stats.append(
                    (
                        matches,
                        predicted_objs[:, 5],
                        predicted_objs[:, 4],
                        true_objs[:, 4],
                    )
                )

        # Compute average precisions if any matches exist
        if stats:
            concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
            average_precisions = cls._average_precisions_per_class(*concatenated_stats)
            map50 = average_precisions[:, 0].mean()
            map75 = average_precisions[:, 5].mean()
            map50_95 = average_precisions.mean()
        else:
            map50, map75, map50_95 = 0, 0, 0
            average_precisions = np.array([])

        return cls(
            map50_95=map50_95,
            map50=map50,
            map75=map75,
            per_class_ap50_95=average_precisions,
        )

    @staticmethod
    def compute_average_precision(
        recall: npt.NDArray[np.float64],
        precision: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute the average precision using 101-point interpolation (COCO), given
            the recall and precision curves.

        Args:
            recall: The recall curve.
            precision: The precision curve.

        Returns:
            Average precision.
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

        # Check if we are running on NumPy 2.0+ or older
        if hasattr(np, "trapezoid"):
            average_precision = np.trapezoid(
                interpolated_precision, interpolated_recall_levels
            )
        else:
            average_precision = getattr(np, "trapz")(
                interpolated_precision, interpolated_recall_levels
            )

        return float(average_precision)

    @staticmethod
    def _match_detection_batch(
        predictions: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        iou_thresholds: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        """
        Match predictions with target labels based on IoU levels.

        Args:
            predictions: Batch prediction. Describes a single image and
                has `shape = (M, 6)` where `M` is the number of detected objects.
                Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets: Batch target labels. Describes a single image and
                has `shape = (N, 5)` where `N` is the number of ground-truth objects.
                Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
            iou_thresholds: Array contains different IoU thresholds.

        Returns:
            Matched prediction with target labels result.
        """
        num_predictions, num_iou_levels = predictions.shape[0], iou_thresholds.shape[0]
        correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
        iou = box_iou_batch(targets[:, :4], predictions[:, :4])
        correct_class = targets[:, 4:5] == predictions[:, 4]

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
        result: npt.NDArray[np.bool_] = correct
        return result

    @staticmethod
    def _average_precisions_per_class(
        matches: npt.NDArray[np.bool_],
        prediction_confidence: npt.NDArray[np.float32],
        prediction_class_ids: npt.NDArray[np.int32],
        true_class_ids: npt.NDArray[np.int32],
        eps: float = 1e-16,
    ) -> npt.NDArray[np.float64]:
        """
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

        Args:
            matches: True positives.
            prediction_confidence: Objectness value from 0-1.
            prediction_class_ids: Predicted object classes.
            true_class_ids: True object classes.
            eps: Small value to prevent division by zero.

        Returns:
            Average precision for different IoU levels.
        """
        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]

        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
        num_classes = unique_classes.shape[0]

        average_precisions: npt.NDArray[np.float64] = np.zeros(
            (num_classes, matches.shape[1]), dtype=np.float64
        )

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
                    MeanAveragePrecision.compute_average_precision(
                        recall[:, iou_level_idx], precision[:, iou_level_idx]
                    )
                )

        result: npt.NDArray[np.float64] = average_precisions
        return result
