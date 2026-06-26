from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deprecate import TargetMode, deprecated, deprecated_class, void

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.core import DetectionDataset
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    box_iou_batch,
    oriented_box_iou_batch,
)
from supervision.metrics.core import MetricTarget


def _assert_supported_target(metric_target: MetricTarget) -> None:
    if metric_target == MetricTarget.MASKS:
        raise ValueError(
            "MetricTarget.MASKS is not currently supported for ConfusionMatrix."
        )


def detections_to_tensor(
    detections: Detections,
    with_confidence: bool = False,
    metric_target: MetricTarget = MetricTarget.BOXES,
) -> npt.NDArray[np.float32]:
    """
    Convert Supervision Detections to a numpy tensor for metric computation.

    Args:
        detections: Detections/Targets in the format of sv.Detections.
        with_confidence: Whether to include confidence as the last column.
        metric_target: The type of detection data to use.
            Supports `MetricTarget.BOXES` and
            `MetricTarget.ORIENTED_BOUNDING_BOXES`.

    Returns:
        Detections as a float32 numpy array. Shape depends on `metric_target`
        and `with_confidence`:

        | `metric_target`                        | `with_confidence` | shape     |
        |----------------------------------------|-------------------|-----------|
        | `MetricTarget.BOXES`                   | `False`           | `(N, 5)`  |
        | `MetricTarget.BOXES`                   | `True`            | `(N, 6)`  |
        | `MetricTarget.ORIENTED_BOUNDING_BOXES` | `False`           | `(N, 9)`  |
        | `MetricTarget.ORIENTED_BOUNDING_BOXES` | `True`            | `(N, 10)` |

        Column layout:

        - `BOXES`: ``[x_min, y_min, x_max, y_max, class_id [, confidence]]``
        - `ORIENTED_BOUNDING_BOXES`:
          ``[x1, y1, x2, y2, x3, y3, x4, y4, class_id [, confidence]]``

    Raises:
        ValueError: If `metric_target` is `MetricTarget.MASKS`.
        ValueError: If `detections.class_id` is `None`.
        ValueError: If `with_confidence=True` and `detections.confidence` is `None`.
        ValueError: If `metric_target` is `MetricTarget.ORIENTED_BOUNDING_BOXES`
            and `detections.data` does not contain `ORIENTED_BOX_COORDINATES`,
            or if the stored array does not have exactly `N * 8` elements.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.metrics.core import MetricTarget
        >>> from supervision.config import ORIENTED_BOX_COORDINATES
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        ...     class_id=np.array([0]),
        ...     confidence=np.array([0.9]),
        ... )
        >>> tensor = detections_to_tensor(detections, with_confidence=True)
        >>> tensor.shape
        (1, 6)
        >>> obb_coords = np.array([[0, 0, 10, 0, 10, 10, 0, 10]], dtype=np.float32)
        >>> det_obb = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        ...     class_id=np.array([0]),
        ...     data={ORIENTED_BOX_COORDINATES: obb_coords},
        ... )
        >>> tensor_obb = detections_to_tensor(
        ...     det_obb, metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        ... )
        >>> tensor_obb.shape
        (1, 9)

        ```
    """
    _assert_supported_target(metric_target)

    if detections.class_id is None:
        raise ValueError(
            "ConfusionMatrix can only be calculated for Detections with class_id"
        )

    if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
        obb = detections.data.get(ORIENTED_BOX_COORDINATES)
        if obb is None:
            if len(detections) > 0:
                raise ValueError(
                    "ORIENTED_BOUNDING_BOXES requested, but "
                    f"{ORIENTED_BOX_COORDINATES} is missing from detections.data"
                )
            box_data = np.empty((0, 8), dtype=np.float32)
        else:
            obb_arr = np.asarray(obb, dtype=np.float32)
            # Normalize (N, 4, 2) → (N, 8) as produced by from_ultralytics.
            if obb_arr.ndim == 3 and obb_arr.shape[1:] == (4, 2):
                obb_arr = obb_arr.reshape(-1, 8)
            if obb_arr.size != len(detections) * 8:
                raise ValueError(
                    f"Expected {ORIENTED_BOX_COORDINATES} to contain "
                    f"{len(detections) * 8} elements "
                    f"(N={len(detections)} detections x 8 coordinates), "
                    f"but got {obb_arr.size}. "
                    "Each OBB must be stored as [x1, y1, x2, y2, x3, y3, x4, y4]."
                )
            box_data = obb_arr.reshape(-1, 8)
    else:
        box_data = detections.xyxy

    arrays_to_concat = [
        box_data,
        np.expand_dims(detections.class_id.astype(np.float32), 1),
    ]

    if with_confidence:
        if detections.confidence is None:
            raise ValueError(
                "ConfusionMatrix can only be calculated for Detections with confidence"
            )
        arrays_to_concat.append(np.expand_dims(detections.confidence, 1))

    result: npt.NDArray[np.float32] = np.concatenate(arrays_to_concat, axis=1)
    return result


def _validate_input_tensors(
    predictions: list[npt.NDArray[np.float32]],
    targets: list[npt.NDArray[np.float32]],
    metric_target: MetricTarget = MetricTarget.BOXES,
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
                "Predictions and targets must be lists of numpy arrays. "
                f"Got {type(predictions[0])} and {type(targets[0])} instead."
            )

        expected_pred_cols = (
            10 if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES else 6
        )
        expected_target_cols = (
            9 if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES else 5
        )

        if predictions[0].shape[1] != expected_pred_cols:
            raise ValueError(
                f"Predictions must have shape (N, {expected_pred_cols}). "
                f"Got {predictions[0].shape} instead."
            )
        if targets[0].shape[1] != expected_target_cols:
            raise ValueError(
                f"Targets must have shape (N, {expected_target_cols}). "
                f"Got {targets[0].shape} instead."
            )


def _split_detections_by_outcome(
    predictions: Detections,
    targets: Detections,
    conf_threshold: float,
    iou_threshold: float,
    metric_target: MetricTarget = MetricTarget.BOXES,
) -> tuple[Detections, Detections, Detections]:
    """
    Split detections into true positives, false positives, and false negatives.

    Matching follows the same attribution logic as
    ``ConfusionMatrix.evaluate_detection_batch``:
    - matches are computed globally across classes
    - same-class matches are prioritized
    - higher-IoU matches are preferred
    - each prediction and target can be matched at most once

    Cross-class spatial matches are treated as:
    - false positives for the prediction
    - false negatives for the target

    Args:
        predictions: Predicted detections for a single image.
        targets: Ground-truth detections for a single image.
        conf_threshold: Confidence threshold; predictions below this are excluded.
        iou_threshold: IoU threshold; candidate pairs below this are not matched.
        metric_target: Coordinate representation to use for IoU computation.
            Use ``MetricTarget.ORIENTED_BOUNDING_BOXES`` for rotated-box datasets.

    Returns:
        A 3-tuple ``(true_positives, false_positives, false_negatives)`` where
        each element is a ``Detections`` instance sliced from the input arrays.
    """

    if predictions.class_id is None:
        raise ValueError("Predictions must contain class_id values.")

    if targets.class_id is None:
        raise ValueError("Targets must contain class_id values.")

    target_class_ids = targets.class_id

    if predictions.confidence is None:
        filtered_predictions = predictions
    else:
        filtered_predictions = cast(
            Detections,
            predictions[predictions.confidence >= conf_threshold],
        )

    filtered_prediction_class_ids = filtered_predictions.class_id
    if filtered_prediction_class_ids is None:
        raise ValueError("Predictions must contain class_id values.")

    prediction_count = len(filtered_predictions)
    target_count = len(targets)

    tp_indices: list[int] = []
    fp_indices: list[int] = []
    fn_indices: list[int] = []

    if prediction_count == 0:
        fn_indices = list(range(target_count))
        return (
            cast(Detections, filtered_predictions[tp_indices]),
            cast(Detections, filtered_predictions[fp_indices]),
            cast(Detections, targets[fn_indices]),
        )

    if target_count == 0:
        fp_indices = list(range(prediction_count))
        return (
            cast(Detections, filtered_predictions[tp_indices]),
            cast(Detections, filtered_predictions[fp_indices]),
            cast(Detections, targets[fn_indices]),
        )

    # IoU computation mirrors evaluate_detection_batch — keep in sync if either changes.
    if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
        iou_matrix = oriented_box_iou_batch(
            boxes_true=np.asarray(
                targets.data[ORIENTED_BOX_COORDINATES], dtype=np.float32
            ).reshape(len(targets), 8),
            boxes_detection=np.asarray(
                filtered_predictions.data[ORIENTED_BOX_COORDINATES], dtype=np.float32
            ).reshape(len(filtered_predictions), 8),
        )
    else:
        iou_matrix = box_iou_batch(
            boxes_true=targets.xyxy,
            boxes_detection=filtered_predictions.xyxy,
        )

    target_candidate_indices, prediction_candidate_indices = np.where(
        iou_matrix > iou_threshold
    )

    matched_predictions: npt.NDArray[np.bool_] = np.zeros(prediction_count, dtype=bool)
    matched_targets: npt.NDArray[np.bool_] = np.zeros(target_count, dtype=bool)

    cross_class_prediction_indices: list[int] = []
    cross_class_target_indices: list[int] = []

    if len(target_candidate_indices) > 0:
        candidate_ious = iou_matrix[
            target_candidate_indices,
            prediction_candidate_indices,
        ]

        same_class_candidates = (
            target_class_ids[target_candidate_indices]
            == filtered_prediction_class_ids[prediction_candidate_indices]
        )

        candidate_order = np.lexsort(
            (
                -candidate_ious,
                ~same_class_candidates,
            )
        )

        for candidate_index in candidate_order:
            target_index = int(target_candidate_indices[candidate_index])
            prediction_index = int(prediction_candidate_indices[candidate_index])

            if matched_predictions[prediction_index] or matched_targets[target_index]:
                continue

            matched_predictions[prediction_index] = True
            matched_targets[target_index] = True

            prediction_class = filtered_prediction_class_ids[prediction_index]
            target_class = target_class_ids[target_index]

            if prediction_class == target_class:
                tp_indices.append(prediction_index)
            else:
                cross_class_prediction_indices.append(prediction_index)
                cross_class_target_indices.append(target_index)

    fp_indices.extend(np.flatnonzero(~matched_predictions).tolist())
    fn_indices.extend(np.flatnonzero(~matched_targets).tolist())

    fp_indices.extend(cross_class_prediction_indices)
    fn_indices.extend(cross_class_target_indices)

    return (
        cast(Detections, filtered_predictions[tp_indices]),
        cast(Detections, filtered_predictions[fp_indices]),
        cast(Detections, targets[fn_indices]),
    )


def _build_error_labels(
    detections: Detections,
    class_names: list[str] | None,
) -> list[str]:
    """Build per-detection label strings for annotation panels.

    Produces labels like ``"cat 0.95"`` (class name + confidence when available)
    or numeric class-id strings when ``class_names`` is ``None``.

    Args:
        detections: Detections whose labels to build.
        class_names: Optional list mapping class integer ids to name strings.

    Returns:
        List of label strings, one per detection. Returns empty strings when
        ``detections.class_id`` is ``None``.
    """
    if detections.class_id is None:
        return [""] * len(detections)

    labels: list[str] = []
    for index, class_id in enumerate(detections.class_id):
        if class_names is not None and 0 <= int(class_id) < len(class_names):
            class_label = class_names[int(class_id)]
        else:
            class_label = str(int(class_id))

        confidence = ""
        if detections.confidence is not None:
            confidence = f" {detections.confidence[index]:.2f}"

        labels.append(f"{class_label}{confidence}")

    return labels


def _get_annotation_parameters(
    scene: npt.NDArray[np.uint8],
) -> tuple[int, float, int, int, int]:
    """Compute adaptive annotation parameters scaled to the panel size.

    Args:
        scene: The image panel for which to compute parameters.

    Returns:
        A 5-tuple ``(box_thickness, text_scale, text_thickness, text_padding,
        font_size)`` where all values are ``int`` except ``text_scale`` (``float``).
    """
    height, width = scene.shape[:2]
    panel_size = max(min(height, width), 1)
    grid_factor = 2

    font_size = max(18, round(panel_size / (26 * grid_factor)))
    box_thickness = max(2, round(font_size / 5))
    text_scale = float(max(1.0, font_size / 20.0))
    text_thickness = max(1, round(font_size / 15.0))
    text_padding = max(6, round(font_size / 3))

    return box_thickness, text_scale, text_thickness, text_padding, font_size


def _annotate_detection_panel(
    scene: npt.NDArray[np.uint8],
    detections: Detections,
    title: str,
    class_names: list[str] | None,
    annotation_parameters: tuple[int, float, int, int, int],
) -> npt.NDArray[np.uint8]:
    """Render detections onto a copy of ``scene`` with a title overlay.

    Args:
        scene: Source image panel (not mutated).
        detections: Detections to annotate on the panel.
        title: Text label rendered in the top-left corner of the panel.
        class_names: Optional list mapping class integer ids to name strings.
        annotation_parameters: Pre-computed parameters from
            ``_get_annotation_parameters``.

    Returns:
        Annotated copy of ``scene`` as a ``np.uint8`` array.
    """
    import cv2  # lazy: only needed when save_directory_path is set

    from supervision.annotators.core import BoxAnnotator, LabelAnnotator
    from supervision.annotators.utils import ColorLookup
    from supervision.draw.color import ColorPalette

    panel = scene.copy()

    box_thickness, text_scale, text_thickness, text_padding, font_size = (
        annotation_parameters
    )

    if len(detections) > 0:
        box_annotator = BoxAnnotator(
            color=ColorPalette.DEFAULT,
            color_lookup=ColorLookup.CLASS,
            thickness=box_thickness,
        )
        label_annotator = LabelAnnotator(
            color=ColorPalette.DEFAULT,
            color_lookup=ColorLookup.CLASS,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
        )
        labels = _build_error_labels(detections, class_names)
        panel = box_annotator.annotate(panel, detections)
        panel = label_annotator.annotate(panel, detections, labels=labels)

    title_scale = float(max(1.0, font_size / 18.0))
    title_thickness = max(2, round(font_size / 8))
    panel_height, panel_width = panel.shape[:2]
    (title_width, title_height), title_baseline = cv2.getTextSize(
        title,
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        title_thickness,
    )
    title_x = max(0, min(text_padding, panel_width - title_width - 1))
    title_y = max(title_height + text_padding, 0)
    title_y = min(title_y, max(panel_height - title_baseline - 1, 0))

    cv2.putText(
        panel,
        title,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (240, 240, 240),
        title_thickness,
        cv2.LINE_AA,
    )
    return panel


def _save_detection_validation_visualization(
    scene: npt.NDArray[np.uint8],
    predictions: Detections,
    targets: Detections,
    save_path: Path,
    conf_threshold: float,
    iou_threshold: float,
    class_names: list[str] | None,
    metric_target: MetricTarget = MetricTarget.BOXES,
) -> None:
    """Build and save a 2x2 GT/TP/FP/FN mosaic for one image.

    Splits ``predictions`` into true-positive, false-positive, and false-negative
    groups using the same matching logic as
    ``ConfusionMatrix.evaluate_detection_batch``, renders four annotation panels,
    concatenates them into a 2x2 grid, and writes the result to ``save_path``.

    A ``UserWarning`` is emitted if ``cv2.imwrite`` fails (e.g. unsupported
    extension or permission error); the benchmark loop continues regardless.

    Args:
        scene: The original image for this dataset entry.
        predictions: Raw model predictions for ``scene``.
        targets: Ground-truth annotations for ``scene``.
        save_path: Destination file path for the mosaic image.
        conf_threshold: Confidence threshold forwarded to
            ``_split_detections_by_outcome``.
        iou_threshold: IoU threshold forwarded to ``_split_detections_by_outcome``.
        class_names: Optional list mapping class integer ids to name strings.
        metric_target: Coordinate representation used for IoU matching.
    """
    import cv2  # lazy: only needed when save_directory_path is set

    tp_predictions, fp_predictions, fn_targets = _split_detections_by_outcome(
        predictions=predictions,
        targets=targets,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        metric_target=metric_target,
    )

    annotation_parameters = _get_annotation_parameters(scene)

    gt_panel = _annotate_detection_panel(
        scene=scene,
        detections=targets,
        title="Ground Truth",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
    )
    tp_panel = _annotate_detection_panel(
        scene=scene,
        detections=tp_predictions,
        title="True Positives",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
    )
    fp_panel = _annotate_detection_panel(
        scene=scene,
        detections=fp_predictions,
        title="False Positives",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
    )
    fn_panel = _annotate_detection_panel(
        scene=scene,
        detections=fn_targets,
        title="False Negatives",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
    )

    top_row = np.concatenate((gt_panel, tp_panel), axis=1)
    bottom_row = np.concatenate((fp_panel, fn_panel), axis=1)
    result = np.concatenate((top_row, bottom_row), axis=0)

    panel_height = result.shape[0] // 2
    panel_width = result.shape[1] // 2
    divider_thickness = max(1, min(8, min(panel_height, panel_width) // 32))

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1] - 1, result.shape[0] - 1),
        (255, 255, 255),
        thickness=divider_thickness,
    )

    center_x = result.shape[1] // 2
    center_y = result.shape[0] // 2
    cv2.line(
        result,
        (center_x, 0),
        (center_x, result.shape[0] - 1),
        (255, 255, 255),
        divider_thickness,
    )
    cv2.line(
        result,
        (0, center_y),
        (result.shape[1] - 1, center_y),
        (255, 255, 255),
        divider_thickness,
    )

    write_success = cv2.imwrite(str(save_path), result)
    if not write_success:
        warnings.warn(
            f"Failed to write validation image to '{save_path}'.",
            UserWarning,
            stacklevel=2,
        )


@deprecated(  # type: ignore[untyped-decorator]
    target=_validate_input_tensors,
    deprecated_in="0.29.0",
    remove_in="0.32.0",
)
def validate_input_tensors(
    predictions: list[npt.NDArray[np.float32]],
    targets: list[npt.NDArray[np.float32]],
) -> None:
    void(predictions, targets)


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
        metric_target: The type of detection data used for IoU computation.
            Informational metadata set by `from_detections` and `from_tensors`.
            Excluded from `__eq__` comparisons — two `ConfusionMatrix` instances
            with identical `matrix`, `classes`, `conf_threshold`, and
            `iou_threshold` compare as equal regardless of `metric_target`.
    """

    matrix: npt.NDArray[np.int32]
    classes: list[str]
    conf_threshold: float
    iou_threshold: float
    metric_target: MetricTarget = MetricTarget.BOXES

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented
        return (
            np.array_equal(self.matrix, other.matrix)
            and self.classes == other.classes
            and self.conf_threshold == other.conf_threshold
            and self.iou_threshold == other.iou_threshold
        )

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_detections(
        cls,
        predictions: list[Detections],
        targets: list[Detections],
        classes: list[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        metric_target: MetricTarget = MetricTarget.BOXES,
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
            metric_target: The type of detection data to use.
                Supports `MetricTarget.BOXES` (default) and
                `MetricTarget.ORIENTED_BOUNDING_BOXES`. When using
                `MetricTarget.ORIENTED_BOUNDING_BOXES`, each `Detections`
                object must include OBB coordinates in
                `detections.data[ORIENTED_BOX_COORDINATES]` as a float32
                array of shape `(N, 8)` (flat) or `(N, 4, 2)` (as stored by
                `from_ultralytics`); both are normalised to `(N, 8)` internally.
                `MetricTarget.MASKS` is not supported.

        Returns:
            New instance of ConfusionMatrix.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> targets = [
            ...     sv.Detections(
            ...         xyxy=np.array([[0, 0, 10, 10], [50, 50, 60, 60]]),
            ...         class_id=np.array([0, 0])
            ...     )
            ... ]
            >>> predictions = [
            ...     sv.Detections(
            ...         xyxy=np.array([[0, 0, 10, 10], [100, 100, 110, 110]]),
            ...         class_id=np.array([0, 0]),
            ...         confidence=np.array([0.9, 0.8])
            ...     )
            ... ]
            >>> confusion_matrix = sv.ConfusionMatrix.from_detections(
            ...     predictions=predictions,
            ...     targets=targets,
            ...     classes=['person']
            ... )
            >>> confusion_matrix.matrix
            array([[1., 1.],
                   [1., 0.]])

            ```
        """
        prediction_tensors = []
        target_tensors = []
        for prediction, target in zip(predictions, targets):
            prediction_tensors.append(
                detections_to_tensor(
                    prediction, with_confidence=True, metric_target=metric_target
                )
            )
            target_tensors.append(
                detections_to_tensor(
                    target, with_confidence=False, metric_target=metric_target
                )
            )
        return cls.from_tensors(
            predictions=prediction_tensors,
            targets=target_tensors,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            metric_target=metric_target,
        )

    @classmethod
    def from_tensors(
        cls,
        predictions: list[npt.NDArray[np.float32]],
        targets: list[npt.NDArray[np.float32]],
        classes: list[str],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        metric_target: MetricTarget = MetricTarget.BOXES,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix based on predicted and ground-truth detections.

        Args:
            predictions: Each element of the list describes a single
                image and has `shape = (M, 6)` or `shape = (M, 10)` depending on
                `metric_target`.
                If `MetricTarget.BOXES`, each row is in
                `(x_min, y_min, x_max, y_max, class, conf)` format.
                If `MetricTarget.ORIENTED_BOUNDING_BOXES`, each row is in
                `(x1, y1, x2, y2, x3, y3, x4, y4, class, conf)` format.
            targets: Each element of the list describes a single
                image and has `shape = (N, 5)` or `shape = (N, 9)` depending on
                `metric_target`.
                If `MetricTarget.BOXES`, each row is in
                `(x_min, y_min, x_max, y_max, class)` format.
                If `MetricTarget.ORIENTED_BOUNDING_BOXES`, each row is in
                `(x1, y1, x2, y2, x3, y3, x4, y4, class)` format.
            classes: Model class names.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection iou threshold between `0` and `1`.
                Detections with lower iou will be classified as `FP`.
            metric_target: The type of detection data to use.
                Determines expected tensor shapes (see Args above for column
                layouts). `MetricTarget.MASKS` is not supported.

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
        _assert_supported_target(metric_target)
        _validate_input_tensors(predictions, targets, metric_target=metric_target)

        num_classes = len(classes)
        matrix = np.zeros((num_classes + 1, num_classes + 1))
        for true_batch, detection_batch in zip(targets, predictions):
            matrix += cls.evaluate_detection_batch(
                predictions=detection_batch,
                targets=true_batch,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                metric_target=metric_target,
            )
        return cls(
            matrix=matrix,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            metric_target=metric_target,
        )

    @staticmethod
    def evaluate_detection_batch(
        predictions: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        num_classes: int,
        conf_threshold: float,
        iou_threshold: float,
        metric_target: MetricTarget = MetricTarget.BOXES,
    ) -> npt.NDArray[np.int32]:
        """
        Calculate confusion matrix for a batch of detections for a single image.

        Args:
            predictions: Batch prediction. Describes a single image and
                has `shape = (M, 6)` or `shape = (M, 10)` depending on
                `metric_target`.
                If `MetricTarget.BOXES`, each row is in
                `(x_min, y_min, x_max, y_max, class, conf)` format.
                If `MetricTarget.ORIENTED_BOUNDING_BOXES`, each row is in
                `(x1, y1, x2, y2, x3, y3, x4, y4, class, conf)` format.
            targets: Batch target labels. Describes a single image and
                has `shape = (N, 5)` or `shape = (N, 9)` depending on
                `metric_target`.
                If `MetricTarget.BOXES`, each row is in
                `(x_min, y_min, x_max, y_max, class)` format.
                If `MetricTarget.ORIENTED_BOUNDING_BOXES`, each row is in
                `(x1, y1, x2, y2, x3, y3, x4, y4, class)` format.
            num_classes: Number of classes.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection iou threshold between `0` and `1`.
                Detections with lower iou will be classified as `FP`.
            metric_target: The type of detection data to use.
                Determines IoU function (`box_iou_batch` vs
                `oriented_box_iou_batch`) and coordinate column count.
                `MetricTarget.MASKS` is not supported.

        Returns:
            Confusion matrix based on a single image.
        """
        _assert_supported_target(metric_target)

        expected_pred_cols = (
            10 if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES else 6
        )
        expected_target_cols = (
            9 if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES else 5
        )
        if predictions.ndim != 2 or predictions.shape[1] != expected_pred_cols:
            raise ValueError(
                f"Predictions must have shape (M, {expected_pred_cols}). "
                f"Got {predictions.shape} instead."
            )
        if targets.ndim != 2 or targets.shape[1] != expected_target_cols:
            raise ValueError(
                f"Targets must have shape (N, {expected_target_cols}). "
                f"Got {targets.shape} instead."
            )

        result_matrix = np.zeros((num_classes + 1, num_classes + 1))

        # Filter predictions by confidence threshold
        coords_dim = 8 if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES else 4
        class_id_idx = coords_dim
        conf_idx = coords_dim + 1

        confidence = predictions[:, conf_idx]
        detection_batch_filtered = predictions[confidence >= conf_threshold]

        if len(detection_batch_filtered) == 0:
            # No detections pass confidence threshold - all GT are FN
            true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
            for gt_class in true_classes:
                result_matrix[gt_class, num_classes] += 1
            return result_matrix

        if len(targets) == 0:
            # No ground truth - all detections are FP
            detection_classes = np.array(
                detection_batch_filtered[:, class_id_idx], dtype=np.int16
            )
            for det_class in detection_classes:
                result_matrix[num_classes, det_class] += 1
            return result_matrix

        true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
        detection_classes = np.array(
            detection_batch_filtered[:, class_id_idx], dtype=np.int16
        )
        true_boxes = targets[:, :coords_dim]
        detection_boxes = detection_batch_filtered[:, :coords_dim]

        # Calculate IoU matrix
        if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            iou_batch = oriented_box_iou_batch(
                boxes_true=true_boxes, boxes_detection=detection_boxes
            )
        else:
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
        metric_target: MetricTarget = MetricTarget.BOXES,
        *,
        save_directory_path: str | Path | None = None,
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
            save_directory_path: Optional directory where per-image validation
                result grids are saved using the original image filenames. Images
                are written directly to this directory (no subdirectory is added).
                When ``None`` (default), no images are saved.
            metric_target: The type of detection data to use.
                Supports `MetricTarget.BOXES` and
                `MetricTarget.ORIENTED_BOUNDING_BOXES`. Passed through to
                `from_detections`. `MetricTarget.MASKS` is not supported.

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
        if save_directory_path is not None:
            save_directory = Path(save_directory_path)
            save_directory.mkdir(parents=True, exist_ok=True)

        predictions, targets = [], []
        for index, (image_name, image, annotation) in enumerate(dataset):
            predictions_batch = callback(image)
            predictions.append(predictions_batch)
            targets.append(annotation)

            if save_directory_path is not None:
                if isinstance(image_name, Path):
                    image_filename = image_name.name
                elif isinstance(image_name, str):
                    image_filename = Path(image_name).name
                else:
                    image_filename = f"image_{index:06d}.jpg"

                if Path(image_filename).suffix == "":
                    image_filename = f"{image_filename}.jpg"

                save_path = save_directory / image_filename
                if save_path.exists():
                    warnings.warn(
                        f"Validation image '{image_filename}' already exists at "
                        f"'{save_path}' and will be overwritten.",
                        UserWarning,
                        stacklevel=2,
                    )
                _save_detection_validation_visualization(
                    scene=image,
                    predictions=predictions_batch,
                    targets=annotation,
                    save_path=save_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    class_names=dataset.classes,
                    metric_target=metric_target,
                )
        return cls.from_detections(
            predictions=predictions,
            targets=targets,
            classes=dataset.classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            metric_target=metric_target,
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


@deprecated_class(
    target=TargetMode.NOTIFY,
    deprecated_in="0.27.0",
    remove_in="0.31.0",
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
        _validate_input_tensors(predictions, targets)
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
