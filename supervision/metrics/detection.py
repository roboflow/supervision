from dataclasses import dataclass
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
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
        targets: List[sv.Detections],
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

    @classmethod
    def benchmark(
        cls,
        dataset: sv.DetectionDataset,
        callback: Callable[[np.ndarray], sv.Detections],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> "ConfusionMatrix":
        """
        Create confusion matrix from dataset and callback function.

        Args:
            dataset: an annotated dataset.
            callback: a function that takes an image as input and returns detections.
            conf_threshold: see ConfusionMatrix.from_detections.
            iou_threshold: see ConfusionMatrix.from_detections.

        Returns:
            ConfusionMatrix object.
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
        save_path: str,
        title: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        """
        Create confusion matrix plot and save it at selected location.

        Args:
            target_path: save location of confusion matrix plot.
            title: title displayed at the top of the confusion matrix plot. Default `None`.
            class_names: list of class names detected my model. If non given class indexes will be used. Default `None`.
            normalize: chart will display absolute number of detections falling into given category. Otherwise percentage of detections will be displayed.
        """

        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Verdana"]

        array = self.matrix.copy()

        # num_classes = self.num_classes + 1
        num_classes = 21
        class_names = class_names[:20]
        array = array[:20, :20]
        array[0, 0] = 10
        array[1, 8] = 15
        array[2, 9] = 20
        array[3, 10] = 25
        array[4, 11] = 10
        array[5, 12] = 15
        array[8, 12] = 35

        if normalize:
            eps = 1e-8
            array = array / (array.sum(0).reshape(1, -1) + eps)

        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(figsize=(12, 10), tight_layout=True, facecolor="white")

        labels = (
            class_names is not None
            and (0 < len(class_names) < 99)
            and len(class_names) == self.num_classes
        )
        x_tick_labels = class_names + ["FN"] if labels else None
        y_tick_labels = class_names + ["FP"] if labels else None
        im = ax.imshow(array, cmap="Blues")

        cbar_kw = {}
        cbarlabel = ""
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        cbar.mappable.set_clim(vmin=0, vmax=np.nanmax(array))

        if x_tick_labels is None:
            tick_interval = 2
        else:
            tick_interval = 1
        ax.set_xticks(np.arange(0, num_classes, tick_interval), labels=x_tick_labels)
        ax.set_yticks(np.arange(0, num_classes, tick_interval), labels=y_tick_labels)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="default")

        labelsize = 10 if self.num_classes < 50 else 8
        ax.tick_params(axis="both", which="both", labelsize=labelsize)

        if (num_classes) < 30:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{array[i, j]:.2f}" if normalize else f"{array[i, j]:.0f}",
                        ha="center",
                        va="center",
                        color="white",
                    )

        if title:
            ax.set_title(title, fontsize=20)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_facecolor("white")
        fig.savefig(save_path, dpi=250, facecolor=fig.get_facecolor(), transparent=True)
