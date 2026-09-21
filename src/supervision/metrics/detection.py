from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from deprecate import (  # type: ignore[import-untyped,unused-ignore]
    deprecated,
    void,
)

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.core import DetectionDataset
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.metrics.core import MetricTarget

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _assert_tensor_target(metric_target: MetricTarget) -> None:
    """Reject metric targets that cannot be laid out as detection tensor rows."""
    if metric_target == MetricTarget.MASKS:
        raise ValueError(
            "MetricTarget.MASKS cannot be represented as a detection tensor. Use "
            "`ConfusionMatrix.from_detections` or `ConfusionMatrix.benchmark`, "
            "which compute mask IoU directly from `Detections.mask`."
        )


def _detections_masks(
    detections: Detections, role: str
) -> npt.NDArray[np.bool_] | CompactMask:
    """Return the masks of `detections`, raising if they are missing."""
    if detections.mask is None:
        raise ValueError(
            f"ConfusionMatrix with `MetricTarget.MASKS` requires {role} to include "
            "masks."
        )
    return detections.mask


def _mask_iou_batch_for_matching(
    targets: Detections, predictions: Detections
) -> npt.NDArray[np.floating]:
    """Pairwise mask IoU between non-empty `targets` (rows) and `predictions` (columns),
    validating that both carry masks of one resolution."""
    target_masks = _detections_masks(targets, "targets")
    prediction_masks = _detections_masks(predictions, "predictions")
    target_resolution = tuple(target_masks.shape[1:])
    prediction_resolution = tuple(prediction_masks.shape[1:])
    if target_resolution != prediction_resolution:
        raise ValueError(
            "ConfusionMatrix with `MetricTarget.MASKS` requires predictions and "
            "targets to share one mask resolution, got prediction masks of shape "
            f"{prediction_resolution} and target masks of shape "
            f"{target_resolution}."
        )
    return mask_iou_batch(target_masks, prediction_masks)


def _validated_class_ids(
    values: npt.NDArray[np.number],
    num_classes: int,
    role: str,
) -> npt.NDArray[np.int64]:
    """Return class ids that are safe to use as confusion-matrix indexes."""
    if values.size == 0:
        return np.asarray(values, dtype=np.int64)

    finite = np.isfinite(values)
    integral = values == np.floor(values)
    if not np.all(finite & integral):
        raise ValueError(f"{role} class ids must be finite integers.")

    class_ids = values.astype(np.int64)
    invalid = (class_ids < 0) | (class_ids >= num_classes)
    if np.any(invalid):
        invalid_values = np.unique(class_ids[invalid]).tolist()
        raise ValueError(
            f"{role} class ids must be in [0, {num_classes - 1}], got {invalid_values}."
        )
    return class_ids


def _confusion_matrix_from_iou(
    iou_batch: npt.NDArray[np.floating],
    true_classes: npt.NDArray[np.int64],
    detection_classes: npt.NDArray[np.int64],
    num_classes: int,
    iou_threshold: float,
) -> npt.NDArray[np.int32]:
    """Build one image's confusion matrix from a target x detection IoU matrix.

    Matching is greedy: candidate pairs above `iou_threshold` are visited
    same-class first, then by descending IoU, and each target and detection is
    matched at most once. A cross-class match counts as a misclassification,
    unmatched targets as `FN`, and unmatched detections as `FP`.

    Args:
        iou_batch: IoU matrix of shape `(N, M)` for `N` targets and `M`
            detections.
        true_classes: Validated class id of each target, shape `(N,)`.
        detection_classes: Validated class id of each detection, shape `(M,)`.
        num_classes: Number of classes.
        iou_threshold: Candidate pairs at or below this IoU are not matched.

    Returns:
        Confusion matrix of shape `(num_classes + 1, num_classes + 1)`.
    """
    result_matrix: npt.NDArray[np.int32] = np.zeros(
        (num_classes + 1, num_classes + 1), dtype=np.int32
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


def _evaluate_mask_batch(
    predictions: Detections,
    targets: Detections,
    num_classes: int,
    conf_threshold: float,
    iou_threshold: float,
) -> npt.NDArray[np.int32]:
    """Calculate the confusion matrix of a single image from mask IoU.

    Counterpart of `ConfusionMatrix.evaluate_detection_batch` for
    `MetricTarget.MASKS`: masks cannot be laid out as tensor rows, so the
    matching reads `Detections.mask` directly. Dense `(N, H, W)` arrays and
    `CompactMask` are both accepted.

    Args:
        predictions: Predicted detections for a single image. Must carry
            `class_id`, `confidence` and, unless empty, `mask`.
        targets: Ground-truth detections for a single image. Must carry
            `class_id` and, unless empty, `mask`.
        num_classes: Number of classes.
        conf_threshold: Detection confidence threshold between `0` and `1`.
            Detections with lower confidence will be excluded.
        iou_threshold: Detection IoU threshold between `0` and `1`.
            Detections with lower IoU will be classified as `FP`.

    Returns:
        Confusion matrix based on a single image.
    """
    if predictions.class_id is None or targets.class_id is None:
        raise ValueError(
            "ConfusionMatrix can only be calculated for Detections with class_id"
        )
    if predictions.confidence is None:
        raise ValueError(
            "ConfusionMatrix can only be calculated for Detections with confidence"
        )

    keep = np.asarray(predictions.confidence, dtype=np.float32) >= conf_threshold
    filtered_predictions = predictions.select(keep)

    true_classes = _validated_class_ids(targets.class_id, num_classes, "Target")
    detection_classes = _validated_class_ids(
        predictions.class_id[keep], num_classes, "Prediction"
    )

    result_matrix: npt.NDArray[np.int32] = np.zeros(
        (num_classes + 1, num_classes + 1), dtype=np.int32
    )
    if len(filtered_predictions) == 0:
        for gt_class in true_classes:
            result_matrix[gt_class, num_classes] += 1
        return result_matrix
    if len(targets) == 0:
        for det_class in detection_classes:
            result_matrix[num_classes, det_class] += 1
        return result_matrix

    iou_batch = _mask_iou_batch_for_matching(targets, filtered_predictions)
    return _confusion_matrix_from_iou(
        iou_batch=iou_batch,
        true_classes=true_classes,
        detection_classes=detection_classes,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
    )


def detections_to_tensor(
    detections: Detections,
    with_confidence: bool = False,
    metric_target: MetricTarget = MetricTarget.BOXES,
) -> npt.NDArray[np.float32]:
    """Convert Supervision Detections to a numpy tensor for metric computation.

    Args:
        detections: Detections/Targets in the format of sv.Detections.
        with_confidence: Whether to include confidence as the last column.
        metric_target: The type of detection data to use.
            Supports `MetricTarget.BOXES` and
            `MetricTarget.ORIENTED_BOUNDING_BOXES`. Masks have no tensor row
            layout, so `MetricTarget.MASKS` is rejected here; use
            `ConfusionMatrix.from_detections` for masks.

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
        ValueError: If `metric_target` is `MetricTarget.MASKS`, which has no
            tensor representation.
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
    _assert_tensor_target(metric_target)

    if detections.class_id is None:
        raise ValueError(
            "ConfusionMatrix can only be calculated for Detections with class_id"
        )

    box_data: npt.NDArray[np.float32]
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
        box_data = np.asarray(detections.xyxy, dtype=np.float32)

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
    """Checks for shape consistency of input tensors."""
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
    """Split detections into true positives, false positives, and false negatives.

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
        metric_target: Detection data to use for IoU computation.
            Use ``MetricTarget.ORIENTED_BOUNDING_BOXES`` for rotated-box datasets
            and ``MetricTarget.MASKS`` for instance segmentation, where both
            ``predictions`` and ``targets`` must carry ``mask``.

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
        prediction_confidence = np.asarray(predictions.confidence, dtype=np.float32)
        filtered_predictions = predictions.select(
            prediction_confidence >= conf_threshold
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
            filtered_predictions.select(tp_indices),
            filtered_predictions.select(fp_indices),
            targets.select(fn_indices),
        )

    if target_count == 0:
        fp_indices = list(range(prediction_count))
        return (
            filtered_predictions.select(tp_indices),
            filtered_predictions.select(fp_indices),
            targets.select(fn_indices),
        )

    # IoU computation mirrors evaluate_detection_batch / _evaluate_mask_batch —
    # keep in sync if either changes.
    iou_matrix: npt.NDArray[np.floating]
    if metric_target == MetricTarget.MASKS:
        iou_matrix = _mask_iou_batch_for_matching(targets, filtered_predictions)
    elif metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
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
        filtered_predictions.select(tp_indices),
        filtered_predictions.select(fp_indices),
        targets.select(fn_indices),
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
    metric_target: MetricTarget = MetricTarget.BOXES,
) -> npt.NDArray[np.uint8]:
    """Render detections onto a copy of ``scene`` with a title overlay.

    Args:
        scene: Source image panel (not mutated).
        detections: Detections to annotate on the panel.
        title: Text label rendered in the top-left corner of the panel.
        class_names: Optional list mapping class integer ids to name strings.
        annotation_parameters: Pre-computed parameters from
            ``_get_annotation_parameters``.
        metric_target: Detection data the matrix was matched on. With
            ``MetricTarget.MASKS`` the masks are filled under the boxes, so the
            panel shows the geometry the outcome was decided on.

    Returns:
        Annotated copy of ``scene`` as a ``np.uint8`` array.
    """
    from supervision import (
        _cv2 as cv2,  # lazy: only needed when save_directory_path is set
    )
    from supervision.annotators.core import (
        BoxAnnotator,
        LabelAnnotator,
        MaskAnnotator,
    )
    from supervision.annotators.utils import ColorLookup
    from supervision.draw.color import ColorPalette

    panel = scene.copy()

    box_thickness, text_scale, text_thickness, text_padding, font_size = (
        annotation_parameters
    )

    if len(detections) > 0:
        if metric_target == MetricTarget.MASKS and detections.mask is not None:
            mask_annotator = MaskAnnotator(
                color=ColorPalette.DEFAULT,
                color_lookup=ColorLookup.CLASS,
            )
            panel = mask_annotator.annotate(panel, detections)
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
        panel = cast(
            npt.NDArray[np.uint8],
            label_annotator.annotate(cast(Any, panel), detections, labels=labels),
        )

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
    panel_array: npt.NDArray[np.uint8] = panel
    return panel_array


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
        metric_target: Detection data used for IoU matching; with
            ``MetricTarget.MASKS`` the panels also fill each mask.
    """
    from supervision import (
        _cv2 as cv2,  # lazy: only needed when save_directory_path is set
    )

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
        metric_target=metric_target,
    )
    tp_panel = _annotate_detection_panel(
        scene=scene,
        detections=tp_predictions,
        title="True Positives",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
        metric_target=metric_target,
    )
    fp_panel = _annotate_detection_panel(
        scene=scene,
        detections=fp_predictions,
        title="False Positives",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
        metric_target=metric_target,
    )
    fn_panel = _annotate_detection_panel(
        scene=scene,
        detections=fn_targets,
        title="False Negatives",
        class_names=class_names,
        annotation_parameters=annotation_parameters,
        metric_target=metric_target,
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
    """Confusion matrix for object detection and instance segmentation tasks.

    Attributes:
        matrix: An 2D `np.ndarray` of shape `(len(classes) + 1, len(classes) + 1)`
            containing the number of `TP`, `FP`, `FN` and `TN` for each class.
        classes: Model class names.
        conf_threshold: Detection confidence threshold between `0` and `1`.
            Detections with lower confidence will be excluded from the matrix.
        iou_threshold: Detection IoU threshold between `0` and `1`.
            Detections with lower IoU will be classified as `FP`.
        metric_target: The type of detection data used for IoU computation:
            `MetricTarget.BOXES`, `MetricTarget.ORIENTED_BOUNDING_BOXES` or
            `MetricTarget.MASKS`. Informational metadata set by
            `from_detections`, `from_tensors` and `benchmark`.
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
        """Calculate confusion matrix based on predicted and ground-truth detections.

        Args:
            targets: Detections objects from ground-truth.
            predictions: Detections objects predicted by the model.
            classes: Model class names.
            conf_threshold: Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
            iou_threshold: Detection IoU threshold between `0` and `1`.
                Detections with lower IoU will be classified as `FP`.
            metric_target: The type of detection data to use.
                Supports `MetricTarget.BOXES` (default),
                `MetricTarget.ORIENTED_BOUNDING_BOXES` and `MetricTarget.MASKS`.
                When using `MetricTarget.ORIENTED_BOUNDING_BOXES`, each
                `Detections` object must include OBB coordinates in
                `detections.data[ORIENTED_BOX_COORDINATES]` as a float32
                array of shape `(N, 8)` (flat) or `(N, 4, 2)` (as stored by
                `from_ultralytics`); both are normalised to `(N, 8)` internally.
                When using `MetricTarget.MASKS`, every non-empty `Detections`
                object must carry `mask`, either a dense `(N, H, W)` boolean
                array or a `CompactMask`, and predictions and targets of one
                image must share the mask resolution. IoU is then computed on
                the masks, so two instances that share a box but not a shape
                are not matched.

        Returns:
            New instance of ConfusionMatrix.

        Raises:
            ValueError: If `predictions` and `targets` differ in length, if any
                `Detections` lacks `class_id`, if a prediction lacks
                `confidence`, or if `MetricTarget.MASKS` is requested and a
                non-empty `Detections` lacks `mask` or the masks of one image
                differ in resolution.

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
            array([[1, 1],
                   [1, 0]], dtype=int32)

            ```

            Instance segmentation is scored on the masks. Both instances below
            share a box; only the one whose shape overlaps counts as a match:

            ```pycon
            >>> from supervision.metrics import MetricTarget
            >>> target_masks = np.zeros((2, 20, 20), dtype=bool)
            >>> target_masks[0, 2:8, 2:8] = True
            >>> target_masks[1, 12:18, 12:18] = True
            >>> predicted_masks = np.zeros((2, 20, 20), dtype=bool)
            >>> predicted_masks[0, 2:8, 2:8] = True
            >>> predicted_masks[1, 12:18, 12:14] = True
            >>> targets = [
            ...     sv.Detections(
            ...         xyxy=sv.mask_to_xyxy(target_masks),
            ...         mask=target_masks,
            ...         class_id=np.array([0, 0]),
            ...     )
            ... ]
            >>> predictions = [
            ...     sv.Detections(
            ...         xyxy=sv.mask_to_xyxy(target_masks),
            ...         mask=predicted_masks,
            ...         class_id=np.array([0, 0]),
            ...         confidence=np.array([0.9, 0.8]),
            ...     )
            ... ]
            >>> sv.ConfusionMatrix.from_detections(
            ...     predictions=predictions,
            ...     targets=targets,
            ...     classes=['cell'],
            ...     metric_target=MetricTarget.MASKS,
            ... ).matrix
            array([[1, 1],
                   [1, 0]], dtype=int32)

            ```
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) and "
                f"targets ({len(targets)}) must be equal."
            )

        if metric_target == MetricTarget.MASKS:
            num_classes = len(classes)
            matrix: npt.NDArray[np.int32] = np.zeros(
                (num_classes + 1, num_classes + 1), dtype=np.int32
            )
            for prediction, target in zip(predictions, targets):
                matrix += _evaluate_mask_batch(
                    predictions=prediction,
                    targets=target,
                    num_classes=num_classes,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            return cls(
                matrix=matrix,
                classes=classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                metric_target=metric_target,
            )

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
        """Calculate confusion matrix based on predicted and ground-truth detections.

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
                layouts). Masks have no tensor row layout, so
                `MetricTarget.MASKS` is rejected here; use `from_detections`
                for masks.

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
            array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=int32)

            ```
        """
        _assert_tensor_target(metric_target)
        _validate_input_tensors(predictions, targets, metric_target=metric_target)

        num_classes = len(classes)
        matrix: npt.NDArray[np.int32] = np.zeros(
            (num_classes + 1, num_classes + 1), dtype=np.int32
        )
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
        """Calculate confusion matrix for a batch of detections for a single image.

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
                `oriented_box_iou_batch`) and coordinate column count. Masks
                have no tensor row layout, so `MetricTarget.MASKS` is rejected
                here; use `from_detections` for masks.

        Returns:
            Confusion matrix based on a single image.
        """
        _assert_tensor_target(metric_target)

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

        result_matrix: npt.NDArray[np.int32] = np.zeros(
            (num_classes + 1, num_classes + 1), dtype=np.int32
        )

        # Filter predictions by confidence threshold
        coords_dim = 8 if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES else 4
        class_id_idx = coords_dim
        conf_idx = coords_dim + 1

        confidence = predictions[:, conf_idx]
        detection_batch_filtered = predictions[confidence >= conf_threshold]

        if len(detection_batch_filtered) == 0:
            true_classes = _validated_class_ids(
                targets[:, class_id_idx], num_classes, "Target"
            )
            for gt_class in true_classes:
                result_matrix[gt_class, num_classes] += 1
            return result_matrix

        if len(targets) == 0:
            detection_classes = _validated_class_ids(
                detection_batch_filtered[:, class_id_idx], num_classes, "Prediction"
            )
            for det_class in detection_classes:
                result_matrix[num_classes, det_class] += 1
            return result_matrix

        true_classes = _validated_class_ids(
            targets[:, class_id_idx], num_classes, "Target"
        )
        detection_classes = _validated_class_ids(
            detection_batch_filtered[:, class_id_idx], num_classes, "Prediction"
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

        return _confusion_matrix_from_iou(
            iou_batch=iou_batch,
            true_classes=true_classes,
            detection_classes=detection_classes,
            num_classes=num_classes,
            iou_threshold=iou_threshold,
        )

    @staticmethod
    def _drop_extra_matches(
        matches: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Deduplicate matches.

        If there are multiple matches for the same true or predicted box, only the one
        with the highest IoU is kept.
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
        """Calculate confusion matrix from dataset and callback function.

        Args:
            dataset: Detection or instance segmentation dataset used for
                evaluation.
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
                Supports `MetricTarget.BOXES`,
                `MetricTarget.ORIENTED_BOUNDING_BOXES` and `MetricTarget.MASKS`.
                Passed through to `from_detections`. With `MetricTarget.MASKS`
                the dataset annotations and the callback's detections must
                carry `mask`, and the saved validation grids fill each mask.

        Returns:
            New instance of ConfusionMatrix.

        Example:
            ```python
            import supervision as sv
            from rfdetr import RFDETRMedium

            dataset = sv.DetectionDataset.from_yolo(...)

            model = RFDETRMedium()
            def callback(image: np.ndarray) -> sv.Detections:
                return model.predict(image[:, :, ::-1])

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
    ) -> Figure:
        """Create confusion matrix plot and save it at selected location.

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
        from matplotlib import pyplot as plt

        # Cast to float so that the NaN masking below never hits an integer
        # matrix (assigning NaN into an int array raises ValueError).
        array = self.matrix.astype(np.float64)

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
        cbar.mappable.set_clim(vmin=0, vmax=float(np.nanmax(array)))

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
