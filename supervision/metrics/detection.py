from dataclasses import dataclass
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib import rcParams

import supervision as sv
from supervision.detection.utils import box_iou_batch


@dataclass
class ConfusionMatrix:
    matrix: np.ndarray
    classes: List[str]
    num_classes: int
    conf_threshold: float
    iou_threshold: float

    @classmethod
    def from_detections(
        cls,
        predictions: List[sv.Detections],
        target: List[sv.Detections],
        classes: List[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> "ConfusionMatrix":
        """
        Calculate confusion matrix based on ground-true and detected objects across all images in concerned dataset.

        Args:
            target: representing ground-truth objects across all images in concerned dataset. Each element of `target` list describe single image and has `shape = (N, 5)` where `N` is number of ground-truth objects. Each row is an sv.Detections object
            predictions: representing detected objects across all images in concerned dataset. Each element of `detection_batches` list describe single image and has `shape = (M, 1)` where `M` is number of detected objects. Each row is an sv.Detections object
            classes:  all known classes.
            conf_threshold:  detection confidence threshold between 0 and 1. Detections with lower confidence will be excluded.
            iou_threshold:  detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.

        Returns:
            confusion_matrix: `ConfusionMatrix` object raw confusion matrix 2d `np.ndarray`.

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
        """
        # TODO: add validation for inputs
        
        num_classes = len(classes)
        matrix = np.zeros((num_classes + 1, num_classes + 1))
        for true_batch, detection_batch in zip(target, predictions):
            matrix += ConfusionMatrix._evaluate_detection_batch(
                true_detections=true_batch,
                pred_detections=detection_batch,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        return cls(
            matrix=matrix,
            num_classes=num_classes,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    @staticmethod
    def _evaluate_detection_batch(
        true_detections: sv.Detections,
        pred_detections: sv.Detections,
        num_classes: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> np.ndarray:

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
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches

    def plot(
        self,
        target_path: str,
        title: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        """
        Create confusion matrix plot and save it at selected location.

        Args:
            target_path: `str` selected target location of confusion matrix plot.
            title: `Optional[str]` title displayed at the top of the confusion matrix plot. Default `None`.
            class_names: `Optional[List[str]]` list of class names detected my model. If non given class indexes will be used. Default `None`.
            normalize: `bool` if set to `False` chart will display absolute number of detections falling into given category. Otherwise percentage of detections will be displayed.
        """
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Verdana"]

        array = self.matrix.copy()

        if normalize:
            array = array / (array.sum(0).reshape(1, self.num_classes + 1) + EPSILON)

        array[array < 0.005] = np.nan

        fig = plt.figure(figsize=(12, 10), tight_layout=True, facecolor="white")
        sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)

        labels = (
            class_names is not None
            and (0 < len(class_names) < 99)
            and len(class_names) == self.num_classes
        )
        x_tick_labels = class_names + ["FN"] if labels else "auto"
        y_tick_labels = class_names + ["FP"] if labels else "auto"
        sn.heatmap(
            array,
            annot=self.num_classes < 30,
            annot_kws={"size": 8},
            fmt=".2f",
            square=True,
            vmin=0,
            cmap="Blues",
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
        ).set_facecolor((1, 1, 1))

        if title:
            fig.axes[0].set_title(title, fontsize=20)

        fig.axes[0].set_xlabel("Predicted")
        fig.axes[0].set_ylabel("True")
        fig.axes[0].set_facecolor("white")
        fig.savefig(
            target_path, dpi=250, facecolor=fig.get_facecolor(), transparent=True
        )

    @classmethod
    def benchmark(
        cls,
        dataset: sv.DetectionDataset,
        callback: Callable[[np.ndarray], sv.Detections],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> "ConfusionMatrix":
        pass
