from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from supervision.dataset.core import DetectionDataset
from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch


@dataclass
class ConfusionMatrix:
    """
    Confusion matrix for object detection tasks.

    Attributes:
        matrix (np.ndarray): An 2D `np.ndarray` of shape `(len(classes) + 1, len(classes) + 1)` containing the number of `TP`, `FP`, `FN` and `TN` for each class.
        classes (List[str]): Model class names.
        conf_threshold (float): Detection confidence threshold between `0` and `1`. Detections with lower confidence will be excluded from the matrix.
        iou_threshold (float): Detection IoU threshold between `0` and `1`. Detections with lower IoU will be classified as `FP`.
    """

    matrix: np.ndarray
    classes: List[str]
    conf_threshold: float
    iou_threshold: float

    @classmethod
    def from_detections(
        cls,
        predictions: List[Detections],
        targets: List[Detections],
        classes: List[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix based on predicted and ground-truth detections.

        Args:
            targets (List[Detections]): Detections objects from ground-truth.
            predictions (List[Detections]): Detections objects predicted by the model.
            classes (List[str]): Model class names.
            conf_threshold (float): Detection confidence threshold between `0` and `1`. Detections with lower confidence will be excluded.
            iou_threshold (float): Detection IoU threshold between `0` and `1`. Detections with lower IoU will be classified as `FP`.

        Returns:
            ConfusionMatrix: New instance of ConfusionMatrix.

        Example:
            ```python
            >>> import supervision as sv

            >>> targets = [
            ...     sv.Detections(...),
            ...     sv.Detections(...)
            ... ]

            >>> predictions = [
            ...     sv.Detections(...),
            ...     sv.Detections(...)
            ... ]

            >>> confusion_matrix = sv.ConfusionMatrix.from_detections(
            ...     predictions=predictions,
            ...     targets=target,
            ...     classes=['person', ...]
            ... )

            >>> confusion_matrix.matrix
            array([
                [0., 0., 0., 0.],
                [0., 1., 0., 1.],
                [0., 1., 1., 0.],
                [1., 1., 0., 0.]
            ])
            ```
        """

        prediction_tensors = []
        target_tensors = []
        for prediction, target in zip(predictions, targets):
            prediction_tensors.append(
                cls.detections_to_tensor(prediction, with_confidence=True)
            )
            target_tensors.append(
                cls.detections_to_tensor(target, with_confidence=False)
            )
        return cls.from_tensors(
            predictions=prediction_tensors,
            targets=target_tensors,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    @classmethod
    def detections_to_tensor(
        cls, detections: Detections, with_confidence: bool = False
    ) -> np.ndarray:
        if detections == Detections.empty():
            if with_confidence:
                return np.zeros((0, 6))
            else:
                return np.zeros((0, 5))

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

        return np.concatenate(arrays_to_concat, axis=1)

    @classmethod
    def from_tensors(
        cls,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        classes: List[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix based on predicted and ground-truth detections.

        Args:
            predictions (List[np.ndarray]): Each element of the list describes a single image and has `shape = (M, 6)` where `M` is the number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets (List[np.ndarray]): Each element of the list describes a single image and has `shape = (N, 5)` where `N` is the number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)` format.
            classes (List[str]): Model class names.
            conf_threshold (float): Detection confidence threshold between `0` and `1`. Detections with lower confidence will be excluded.
            iou_threshold (float): Detection iou  threshold between `0` and `1`. Detections with lower iou will be classified as `FP`.

        Returns:
            ConfusionMatrix: New instance of ConfusionMatrix.

        Example:
            ```python
            >>> import supervision as sv

            >>> targets = (
            ...     [
            ...         array(
            ...             [
            ...                 [0.0, 0.0, 3.0, 3.0, 1],
            ...                 [2.0, 2.0, 5.0, 5.0, 1],
            ...                 [6.0, 1.0, 8.0, 3.0, 2],
            ...             ]
            ...         ),
            ...         array([1.0, 1.0, 2.0, 2.0, 2]),
            ...     ]
            ... )

            >>> predictions = [
            ...     array(
            ...         [
            ...             [0.0, 0.0, 3.0, 3.0, 1, 0.9],
            ...             [0.1, 0.1, 3.0, 3.0, 0, 0.9],
            ...             [6.0, 1.0, 8.0, 3.0, 1, 0.8],
            ...             [1.0, 6.0, 2.0, 7.0, 1, 0.8],
            ...         ]
            ...     ),
            ...     array([[1.0, 1.0, 2.0, 2.0, 2, 0.8]])
            ... ]

            >>> confusion_matrix = sv.ConfusionMatrix.from_tensors(
            ...     predictions=predictions,
            ...     targets=targets,
            ...     classes=['person', ...]
            ... )

            >>> confusion_matrix.matrix
            array([
                [0., 0., 0., 0.],
                [0., 1., 0., 1.],
                [0., 1., 1., 0.],
                [1., 1., 0., 0.]
            ])
            ```
        """
        cls._validate_input_tensors(predictions, targets)

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

    @classmethod
    def _validate_input_tensors(
        cls, predictions: List[np.ndarray], targets: List[np.ndarray]
    ):
        """
        Checks for shape consistency of input tensors.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) and targets ({len(targets)}) must be equal."
            )
        if len(predictions) > 0:
            if not isinstance(predictions[0], np.ndarray) or not isinstance(
                targets[0], np.ndarray
            ):
                raise ValueError(
                    f"Predictions and targets must be lists of numpy arrays. Got {type(predictions[0])} and {type(targets[0])} instead."
                )
            if predictions[0].shape[1] != 6:
                raise ValueError(
                    f"Predictions must have shape (N, 6). Got {predictions[0].shape} instead."
                )
            if targets[0].shape[1] != 5:
                raise ValueError(
                    f"Targets must have shape (N, 5). Got {targets[0].shape} instead."
                )

    @staticmethod
    def evaluate_detection_batch(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_classes: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> np.ndarray:
        """
        Calculate confusion matrix for a batch of detections for a single image.

        Args:
            predictions (List[np.ndarray]): Each element of the list describes a single image and has `shape = (M, 6)` where `M` is the number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets (List[np.ndarray]): Each element of the list describes a single image and has `shape = (N, 5)` where `N` is the number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)` format.
            num_classes (int): Number of classes.
            conf_threshold (float): Detection confidence threshold between `0` and `1`. Detections with lower confidence will be excluded.
            iou_threshold (float): Detection iou  threshold between `0` and `1`. Detections with lower iou will be classified as `FP`.

        Returns:
            np.ndarray: Confusion matrix based on a single image.
        """
        result_matrix = np.zeros((num_classes + 1, num_classes + 1))

        conf_idx = 5
        confidence = predictions[:, conf_idx]
        detection_batch_filtered = predictions[confidence > conf_threshold]

        class_id_idx = 4
        true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
        detection_classes = np.array(
            detection_batch_filtered[:, class_id_idx], dtype=np.int16
        )
        true_boxes = targets[:, :class_id_idx]
        detection_boxes = detection_batch_filtered[:, :class_id_idx]

        iou_batch = box_iou_batch(
            boxes_true=true_boxes, boxes_detection=detection_boxes
        )
        matched_idx = np.asarray(iou_batch > iou_threshold).nonzero()

        if matched_idx[0].shape[0]:
            matches = np.stack(
                (matched_idx[0], matched_idx[1], iou_batch[matched_idx]), axis=1
            )
            matches = ConfusionMatrix._drop_extra_matches(matches=matches)
        else:
            matches = np.zeros((0, 3))

        matched_true_idx, matched_detection_idx, _ = matches.transpose().astype(
            np.int16
        )

        for i, true_class_value in enumerate(true_classes):
            j = matched_true_idx == i
            if matches.shape[0] > 0 and sum(j) == 1:
                result_matrix[
                    true_class_value, detection_classes[matched_detection_idx[j]]
                ] += 1  # TP
            else:
                result_matrix[true_class_value, num_classes] += 1  # FN

        for i, detection_class_value in enumerate(detection_classes):
            if not any(matched_detection_idx == i):
                result_matrix[num_classes, detection_class_value] += 1  # FP

        return result_matrix

    @staticmethod
    def _drop_extra_matches(matches: np.ndarray) -> np.ndarray:
        """
        Deduplicate matches. If there are multiple matches for the same true or predicted box,
        only the one with the highest IoU is kept.
        """
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches

    @classmethod
    def benchmark(
        cls,
        dataset: DetectionDataset,
        callback: Callable[[np.ndarray], Detections],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrix:
        """
        Create confusion matrix from dataset and callback function.

        Args:
            dataset (DetectionDataset): Object detection dataset used for evaluation.
            callback (Callable[[np.ndarray], Detections]): Function that takes an image as input and returns Detections object.
            conf_threshold (float): Detection confidence threshold between `0` and `1`. Detections with lower confidence will be excluded.
            iou_threshold (float): Detection IoU threshold between `0` and `1`. Detections with lower IoU will be classified as `FP`.

        Returns:
            ConfusionMatrix: New instance of ConfusionMatrix.

        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> dataset = sv.DetectionDataset.from_yolo(...)

            >>> model = YOLO(...)
            >>> def callback(image: np.ndarray) -> sv.Detections:
            ...     result = model(image)[0]
            ...     return sv.Detections.from_yolov8(result)

            >>> confusion_matrix = sv.ConfusionMatrix.benchmark(
            ...     dataset = dataset,
            ...     callback = callback
            ... )

            >>> confusion_matrix.matrix
            array([
                [0., 0., 0., 0.],
                [0., 1., 0., 1.],
                [0., 1., 1., 0.],
                [1., 1., 0., 0.]
            ])
            ```
        """
        predictions, targets = [], []
        for img_name, img in dataset.images.items():
            predictions_batch = callback(img)
            predictions.append(predictions_batch)
            targets_batch = dataset.annotations[img_name]
            targets.append(targets_batch)
        return cls.from_detections(
            predictions=predictions,
            targets=targets,
            classes=dataset.classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    def plot(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        classes: Optional[List[str]] = None,
        normalize: bool = False,
        fig_size: Tuple[int, int] = (12, 10),
    ) -> matplotlib.figure.Figure:
        """
        Create confusion matrix plot and save it at selected location.

        Args:
            save_path (Optional[str]): Path to save the plot. If not provided, plot will be displayed.
            title (Optional[str]): Title of the plot.
            classes (Optional[List[str]]): List of classes to be displayed on the plot. If not provided, all classes will be displayed.
            normalize (bool): If True, normalize the confusion matrix.
            fig_size (Tuple[int, int]): Size of the plot.

        Returns:
            matplotlib.figure.Figure: Confusion matrix plot.
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
            x_tick_labels = class_names + ["FN"]
            y_tick_labels = class_names + ["FP"]
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


@dataclass(frozen=True)
class MeanAveragePrecision:
    map: float
    map50: float
    map75: float
    """
    Mean Average Precision for object detection tasks.

    Attributes:
        map (float): map value.
        map50 (float): map value at iou threshold=0.5.
        map75 (float): map value at iou threshold=0.75
    """

    @classmethod
    def from_detections(
        cls,
        predictions: List[Detections],
        targets: List[Detections],
    ) -> MeanAveragePrecision:
        """
        Calculate MeanAveragePrecision based on predicted and ground-truth detections.

        Args:
            targets (List[Detections]): Detections objects from ground-truth.
            predictions (List[Detections]): Detections objects predicted by the model.
        Returns:
            MeanAveragePrecision: New instance of ConfusionMatrix.

        Example:
            ```python
            >>> import supervision as sv

            >>> targets = [
            ...     sv.Detections(...),
            ...     sv.Detections(...)
            ... ]

            >>> predictions = [
            ...     sv.Detections(...),
            ...     sv.Detections(...)
            ... ]

            >>> mean_average_precison = sv.MeanAveragePrecision.from_detections(
            ...     predictions=predictions,
            ...     targets=target,
            ... )

            >>> mean_average_precison.matrix
            0.433
           ```
        """
        prediction_tensors = []
        target_tensors = []
        for prediction, target in zip(predictions, targets):
            prediction_tensors.append(
                MeanAveragePrecision.detections_to_tensor(
                    prediction, with_confidence=True
                )
            )
            target_tensors.append(
                MeanAveragePrecision.detections_to_tensor(target, with_confidence=False)
            )
        return cls.from_tensors(
            predictions=prediction_tensors,
            targets=target_tensors,
        )

    @classmethod
    def benchmark(
            cls,
            dataset: DetectionDataset,
            callback: Callable[[np.ndarray], Detections],
    ) -> MeanAveragePrecision:
        """
        Get map from dataset and callback function.

        Args:
            dataset (DetectionDataset): Object detection dataset used for evaluation.
            callback (Callable[[np.ndarray], Detections]): Function that takes an image as input and returns Detections object.
        Returns:
            MeanAveragePrecision: New instance of MeanAveragePrecision.

        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> dataset = sv.DetectionDataset.from_yolo(...)

            >>> model = YOLO(...)
            >>> def callback(image: np.ndarray) -> sv.Detections:
            ...     result = model(image)[0]
            ...     return sv.Detections.from_yolov8(result)

            >>> mean_average_precision = sv.MeanAveragePrecision.benchmark(
            ...     dataset = dataset,
            ...     callback = callback
            ... )

            >>> mean_average_precision.map
            0.433
            ```
        """
        predictions, targets = [], []
        for img_name, img in dataset.images.items():
            predictions_batch = callback(img)
            predictions.append(predictions_batch)
            targets_batch = dataset.annotations[img_name]
            targets.append(targets_batch)
        return cls.from_detections(
            predictions=predictions,
            targets=targets,
        )

    @classmethod
    def from_tensors(
        cls,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> MeanAveragePrecision:
        """
        Calculate Mean Average Precision based on predicted and ground-truth detections at different threshold.

        Args:
            predictions (List[np.ndarray]): Each element of the list describes a single image and has `shape = (M, 6)` where `M` is the number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets (List[np.ndarray]): Each element of the list describes a single image and has `shape = (N, 5)` where `N` is the number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)` format.
        Returns:
            MeanAveragePrecision: New instance of MeanAveragePrecision.

        Example:
            ```python
            >>> import supervision as sv

            >>> targets = (
            ...     [
            ...         array(
            ...             [
            ...                 [0.0, 0.0, 3.0, 3.0, 1],
            ...                 [2.0, 2.0, 5.0, 5.0, 1],
            ...                 [6.0, 1.0, 8.0, 3.0, 2],
            ...             ]
            ...         ),
            ...         array([1.0, 1.0, 2.0, 2.0, 2]),
            ...     ]
            ... )

            >>> predictions = [
            ...     array(
            ...         [
            ...             [0.0, 0.0, 3.0, 3.0, 1, 0.9],
            ...             [0.1, 0.1, 3.0, 3.0, 0, 0.9],
            ...             [6.0, 1.0, 8.0, 3.0, 1, 0.8],
            ...             [1.0, 6.0, 2.0, 7.0, 1, 0.8],
            ...         ]
            ...     ),
            ...     array([[1.0, 1.0, 2.0, 2.0, 2, 0.8]])
            ... ]

            >>> mean_average_precison = sv.MeanAveragePrecision.from_tensors(
            ...     predictions=predictions,
            ...     targets=targets,
            ... )

            >>> mean_average_precison.map
            0.433
            ```
        """
        cls._validate_input_tensors(predictions, targets)
        map, map50, map75 = 0, 0, 0

        stats, ap = [], []
        iou_levels = np.linspace(0.5, 0.95, 10)
        num_ious = iou_levels.size

        for true_batch, detection_batch in zip(targets, predictions):
            nl, npr = (
                true_batch.shape[0],
                detection_batch.shape[0],
            )  # number of labels, predictions
            correct = np.zeros((npr, num_ious), dtype=bool)  # init

            if npr == 0:
                if nl:
                    stats.append((correct, *np.zeros((2, 0)), true_batch[:, 4]))
                continue
            if nl:
                correct, iouv = cls._match_detection_batch(
                    predictions=detection_batch,
                    targets=true_batch,
                    iou_levels=iou_levels,
                )
                stats.append(
                    (
                        correct,
                        detection_batch[:, 5],
                        detection_batch[:, 4],
                        true_batch[:, 4],
                    )
                )

        stats = [np.concatenate(x, 0) for x in zip(*stats)]

        if len(stats) and stats[0].any():
            ap = MeanAveragePrecision.ap_per_class(*stats)

            ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            map50, map75, map = ap50.mean(), ap75.mean(), ap.mean()

        return cls(map=map, map50=map50, map75=map75)

    @staticmethod
    def detections_to_tensor(
            detections: Detections, with_confidence: bool = False
    ) -> np.ndarray:

        if detections.class_id is None:
            raise ValueError(
                "MeanAveragePrecision can only be calculated for Detections with class_id"
            )

        arrays_to_concat = [detections.xyxy, np.expand_dims(detections.class_id, 1)]

        if with_confidence:
            if detections.confidence is not None:
                arrays_to_concat.append(np.expand_dims(detections.confidence, 1))

        return np.concatenate(arrays_to_concat, axis=1)

    @staticmethod
    def _match_detection_batch(
        predictions: np.ndarray, targets: np.ndarray, iou_levels: np.ndarray
    ) -> np.ndarray:
        correct = np.zeros((predictions.shape[0], iou_levels.shape[0])).astype(bool)
        iou = box_iou_batch(targets[:, :4], predictions[:, :4])

        correct_class = targets[:, 4:5] == predictions[:, 4]

        for i in range(len(iou_levels)):
            x = np.where((iou >= iou_levels[i]) & correct_class)

            if x[0].shape[0]:
                _X1 = np.vstack([x[0], x[1]]).T
                _x2 = iou[x[0], x[1]][:, None]
                matches = np.concatenate([_X1, _x2], axis=1)  # [label, detect, iou]
                matches = MeanAveragePrecision._drop_extra_matches(matches)
                correct[matches[:, 1].astype(int), i] = True
        return correct, iou_levels

    @classmethod
    def _validate_input_tensors(
        cls, predictions: List[np.ndarray], targets: List[np.ndarray]
    ):
        """
        Checks for shape consistency of input tensors.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) and targets ({len(targets)}) must be equal."
            )
        if len(predictions) > 0:
            if not isinstance(predictions[0], np.ndarray) or not isinstance(
                targets[0], np.ndarray
            ):
                raise ValueError(
                    f"Predictions and targets must be lists of numpy arrays. Got {type(predictions[0])} and {type(targets[0])} instead."
                )
            if predictions[0].shape[1] != 6:
                raise ValueError(
                    f"Predictions must have shape (N, 6). Got {predictions[0].shape} instead."
                )
            if targets[0].shape[1] != 5:
                raise ValueError(
                    f"Targets must have shape (N, 5). Got {targets[0].shape} instead."
                )

    @staticmethod
    def _drop_extra_matches(matches: np.ndarray) -> np.ndarray:
        """
        Deduplicate matches. If there are multiple matches for the same true or predicted box,
        only the one with the highest IoU is kept.
        """
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches

    @staticmethod
    def compute_ap(recall, precision):
        """Compute the average precision using 101-point interpolation (COCO), given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
        return ap

    @staticmethod
    def ap_per_class(matches: np.ndarray, prediction_confidence: np.ndarray, prediction_class_ids: np.ndarray, true_batch_class_ids: np.ndarray, EPS=1e-16):
        """
        Args:
            matches (np.ndarray): matches between predictions and targets
            prediction_confidence (np.ndarray): confidence values of predictions
            prediction_class_ids (np.ndarray): class ids values of predictions
            true_batch_class_ids (np.ndarray): class ids values of targets
            EPS: constant to avoid divide by zero
        Returns:
            precision, recall, f1_score, average_precisions, unique_classes
        """
        sorted_ids = np.argsort(-prediction_confidence)
        matches = matches[sorted_ids]

        unique_classes, class_counts = np.unique(true_batch_class_ids, return_counts=True)
        num_classes = unique_classes.shape[0]  # number of classes, number of detections

        average_precisions = np.zeros((num_classes, matches.shape[1]))

        for ci, c in enumerate(unique_classes):
            i = prediction_class_ids == c
            num_targets = class_counts[ci]
            num_predictions = i.sum()
            if num_targets == 0 or num_predictions == 0:
                continue

            _false_positives = (1 - matches[i]).cumsum(0)
            _true_positives = matches[i].cumsum(0)

            recall = _true_positives / (num_targets + EPS)  # recall curve

            precision = _true_positives / (_true_positives + _false_positives)  # precision curve

            # AP from recall-precision curve
            for j in range(matches.shape[1]):
                average_precisions[ci, j] = MeanAveragePrecision.compute_ap(recall[:, j], precision[:, j])

        return average_precisions

    @staticmethod
    def smooth(y, f=0.05):
        # Box filter of fraction f
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
