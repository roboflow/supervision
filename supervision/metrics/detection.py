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
    Data class containing information about classification results in form of a confusion matrix.
    Attributes:
        matrix: An array of shape (len(classes) + 1, len(classes) + 1) containing the number of TP, FP, FN and TN for each class.
        classes: all known class names.
        conf_threshold: detection confidence threshold between 0 and 1. Detections with lower confidence will be excluded from the matrix.
        iou_threshold: detection iou threshold between 0 and 1. Detections with lower iou will be classified as FP.
    """

    matrix: np.ndarray
    classes: List[str]
    conf_threshold: float
    iou_threshold: float

    @classmethod
    def from_matrix(
        cls,
        matrix,
        conf_threshold: float,
        iou_threshold: float,
        classes: Optional[List[str]],
    ) -> ConfusionMatrix:
        """
        Create ConfusionMatrix from matrix.

        Args:
            matrix: confusion matrix.
            classes:  all known classes.
            conf_threshold:  detection confidence threshold between 0 and 1. Detections with lower confidence will be excluded.
            iou_threshold:  detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.
        """
        classes = (
            classes if classes is not None else [str(x) for x in (range(len(matrix)))]
        )
        return cls(
            matrix=matrix,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

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
            targets: Detections objects from ground-truth.
            predictions: Detections objects predicted by the model.
            classes:  all known classes.
            conf_threshold:  detection confidence threshold between 0 and 1. Detections with lower confidence will be excluded.
            iou_threshold:  detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.


        Example:
        ```
        >>> from supervision.metrics.detection import ConfusionMatrix

        >>> target = [
        ...     Detections(xyxy=array([
        ...     [ 0.0, 0.0, 3.0, 3.0 ],
        ...     [ 2.0, 2.0, 5.0, 5.0 ],
        ...     [ 6.0, 1.0, 8.0, 3.0 ],
        ...     ]), confidence=array([ 1.0, 1.0, 1.0, 1.0 ]), class_id=array([1, 1,  2])),
        ...     Detections(xyxy=array([
        ...     [ 1.0, 1.0, 2.0, 2.0 ],
        ...     ]), confidence=array([ 1.0 ]), class_id=array([2]))
        ... ]
        >>> predictions = [
        ...     Detections(
        ...         xyxy=array([
        ...     [ 0.0, 0.0, 3.0, 3.0 ],
        ...     [ 0.1, 0.1, 3.0, 3.0 ],
        ...     [ 6.0, 1.0, 8.0, 3.0 ],
        ...     [ 1.0, 6.0, 2.0, 7.0 ],
        ...     ]),
        ...     confidence=array([ 0.9, 0.9, 0.8, 0.8 ]),
        ...     class_id=array([1, 0, 1, 1])
        ...     ),
        ...     Detections(
        ...         xyxy=array([
        ...     [ 1.0, 1.0, 2.0, 2.0 ]
        ...     ]),
        ...     confidence=array([ 0.8 ]),
        ...     class_id=array([2])
        ...     )
        ... ]

        >>> confusion_matrix = ConfusionMatrix.from_detections(
        ...     predictions=predictions,
        ...     target=target,
        ...     num_classes=3
        ... )

        >>> confusion_matrix.matrix
        ... array([
        ...     [0., 0., 0., 0.],
        ...     [0., 1., 0., 1.],
        ...     [0., 1., 1., 0.],
        ...     [1., 1., 0., 0.]
        ... ])
        ```
        Source: https://github.com/SkalskiP/onemetric/blob/master/onemetric/cv/object_detection/confusion_matrix.py
        """
        # TODO: add validation for inputs

        num_classes = len(classes)
        matrix = np.zeros((num_classes + 1, num_classes + 1))
        for true_batch, detection_batch in zip(targets, predictions):
            matrix += ConfusionMatrix._evaluate_detection_batch(
                true_detections=true_batch,
                pred_detections=detection_batch,
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
    def _evaluate_detection_batch(
        true_detections: Detections,
        pred_detections: Detections,
        num_classes: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> np.ndarray:
        """
        Calculate confusion matrix for a batch of detections for a single image.

        Args:
            See ConfusionMatrix.from_detections

        Returns:
            confusion matrix based on a single image.
        """
        result_matrix = np.zeros((num_classes + 1, num_classes + 1))
        detection_batch_filtered = pred_detections[
            pred_detections.confidence > conf_threshold
        ]
        true_classes = true_detections.class_id
        detection_classes = detection_batch_filtered.class_id
        true_boxes = true_detections.xyxy
        detection_boxes = detection_batch_filtered.xyxy

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
            dataset: an annotated dataset.
            callback: a function that takes an image as input and returns detections.
            conf_threshold: see ConfusionMatrix.from_detections.
            iou_threshold: see ConfusionMatrix.from_detections.
        """
        predictions = []
        targets = []
        for img_name, img in dataset.images.items():
            predictions.append(callback(img))
            targets.append(dataset.annotations[img_name])
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
        class_names: Optional[List[str]] = None,
        do_normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
    ) -> matplotlib.figure.Figure:
        """
        Create confusion matrix plot and save it at selected location.

        Args:
            save_path: save location of confusion matrix plot.
            title: title displayed at the top of the confusion matrix plot. Default `None`.
            class_names: custom classes to be displayed on the plot. If not provided, original classes will be used.
            do_normalize: chart will display fraction of detections in a given class instead of absolute numbers.
            figsize: size of the plot.
        """

        array = self.matrix.copy()

        if do_normalize:
            eps = 1e-8
            array = array / (array.sum(0).reshape(1, -1) + eps)

        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(figsize=figsize, tight_layout=True, facecolor="white")

        class_names = class_names if class_names is not None else self.classes
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
                    ax.text(
                        j,
                        i,
                        f"{array[i, j]:.2f}" if do_normalize else f"{array[i, j]:.0f}",
                        ha="center",
                        va="center",
                        color="white",
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
