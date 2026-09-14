"""Provide reusable greedy one-to-one matching for ``Detections`` collections.

Purpose:
    Expose the metrics package's highest-IoU-first matching primitive as a
    public, index-oriented operation without coupling callers to metric result
    objects.
Scope:
    Matches two in-memory ``Detections`` instances using bounding-box IoU and,
    unless requested otherwise, equal class identifiers. Class-aware matching
    requires both collections to provide class IDs. It does not rank by
    confidence, mutate inputs, calculate metrics, or perform model inference.
Usage:
    Import ``match_detections`` through ``supervision`` and pass two detection
    sets plus an IoU threshold in the closed range from zero to one.
Outputs:
    Returns matched index pairs and unmatched indices with stable NumPy integer
    dtypes, suitable for slicing the original detection collections.
Failure:
    Raises ``ValueError`` for an IoU threshold outside the closed valid range;
    class-aware matching without class IDs; and malformed detection arrays as
    defined by ``Detections`` and the underlying IoU utility.
Used by:
    Public ``sv.match_detections`` callers and metrics utilities sharing the
    greedy matching policy.
"""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    _validate_iou_threshold,
    box_iou_batch,
)


def _greedy_match(
    iou: npt.NDArray[np.float32],
    matched_indices: tuple[npt.NDArray[np.intp], ...],
) -> Iterator[tuple[int, int]]:
    """Yield (target_idx, pred_idx) pairs in greedy highest-IoU-first one-to-one order.

    Candidate pairs are sorted by descending IoU and assigned one-to-one: a pair
    is accepted only when neither the target nor the prediction has been matched.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> iou = np.array([[1.0, 0.667], [0.333, 0.538]], dtype=np.float32)
        >>> matched_indices = np.where(iou >= 0.5)
        >>> list(_greedy_match(iou, matched_indices))
        [(0, 0), (1, 1)]

        ```
    """
    target_idx = matched_indices[0]
    pred_idx = matched_indices[1]
    iou_values = iou[matched_indices]
    order = np.argsort(-iou_values, kind="stable")
    matched_targets: set[int] = set()
    matched_preds: set[int] = set()
    for t, p in zip(target_idx[order].tolist(), pred_idx[order].tolist()):
        if t not in matched_targets and p not in matched_preds:
            matched_targets.add(t)
            matched_preds.add(p)
            yield t, p


def _match_detection_batch_with_target_indices(
    predictions_classes: npt.NDArray[np.int32],
    target_classes: npt.NDArray[np.int32],
    iou: npt.NDArray[np.float32],
    iou_thresholds: npt.NDArray[np.float32],
    target_scored_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int32]]:
    """Match predictions to targets and retain target indices per IoU threshold.

    When ``target_scored_mask`` is provided, scored targets are matched first and
    only predictions left unmatched may then match unscored (ignored) targets.
    This mirrors COCO evaluation, where detections prefer non-ignored ground
    truth, so an out-of-bucket target can never steal a prediction from an
    in-bucket one.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> predictions_classes = np.array([0], dtype=np.int32)
        >>> target_classes = np.array([0], dtype=np.int32)
        >>> iou = np.array([[1.0]], dtype=np.float32)
        >>> thresholds = np.array([0.5], dtype=np.float32)
        >>> correct, matched = _match_detection_batch_with_target_indices(
        ...     predictions_classes,
        ...     target_classes,
        ...     iou,
        ...     thresholds,
        ... )
        >>> correct.tolist(), matched.tolist()
        ([[True]], [[0]])

        ```
    """
    num_predictions = predictions_classes.shape[0]
    num_iou_levels = iou_thresholds.shape[0]
    correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
    matched_targets = np.full((num_predictions, num_iou_levels), -1, dtype=np.int32)
    correct_class = target_classes[:, None] == predictions_classes

    for i, iou_level in enumerate(iou_thresholds):
        candidate_pairs = (iou >= iou_level) & correct_class
        if target_scored_mask is None:
            match_rounds = [candidate_pairs]
        else:
            match_rounds = [
                candidate_pairs & target_scored_mask[:, None],
                candidate_pairs & ~target_scored_mask[:, None],
            ]

        for round_pairs in match_rounds:
            unmatched_predictions = matched_targets[:, i] < 0
            matched_indices = np.where(round_pairs & unmatched_predictions)

            for target_idx, prediction_idx in _greedy_match(iou, matched_indices):
                correct[prediction_idx, i] = True
                matched_targets[prediction_idx, i] = target_idx

    return correct, matched_targets


def match_detections(
    detections_a: Detections,
    detections_b: Detections,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Match detections from two sources into one-to-one pairs.

    The assignment is greedy and highest-IoU-first, identical to the matcher
    used by the metrics modules: each detection from ``detections_a`` can match
    at most one detection from ``detections_b``, and vice versa. It does not
    compute a globally optimal assignment. Pairs below ``iou_threshold`` are
    never matched, and by default a pair also requires equal ``class_id``
    values. ``confidence`` is ignored; callers that want metric-style score
    ordering should sort their detections first.

    Args:
        detections_a (Detections): First set of detections.
        detections_b (Detections): Second set of detections.
        iou_threshold (float, optional): Minimum IoU required for a pair.
            Defaults to 0.5.
        class_agnostic (bool, optional): When True, matching ignores
            ``class_id`` and uses IoU only. Defaults to False.

    Raises:
        ValueError: If `iou_threshold` is outside `[0, 1]`, or class-aware
            matching is requested without class IDs on both inputs.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of
        ``(matched_pairs, unmatched_a, unmatched_b)`` where ``matched_pairs``
        has shape ``(M, 2)`` with column 0 indexing ``detections_a`` and
        column 1 indexing ``detections_b``; ``unmatched_a`` and
        ``unmatched_b`` hold the remaining indices of each side.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> a = Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        ...     class_id=np.array([0]),
        ... )
        >>> b = Detections(
        ...     xyxy=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        ...     class_id=np.array([0, 1]),
        ... )
        >>> matched_pairs, unmatched_a, unmatched_b = match_detections(a, b)
        >>> matched_pairs.tolist()
        [[0, 0]]
        >>> unmatched_a.tolist()
        []
        >>> unmatched_b.tolist()
        [1]

        ```
    """
    _validate_iou_threshold(iou_threshold)
    if not class_agnostic and (
        detections_a.class_id is None or detections_b.class_id is None
    ):
        raise ValueError(
            "Both detections must provide class_id when class_agnostic is False."
        )

    if len(detections_a) == 0 or len(detections_b) == 0:
        matched_pairs = np.empty((0, 2), dtype=np.int64)
        unmatched_a = np.arange(len(detections_a), dtype=np.int64)
        unmatched_b = np.arange(len(detections_b), dtype=np.int64)
        return matched_pairs, unmatched_a, unmatched_b

    iou = box_iou_batch(detections_a.xyxy, detections_b.xyxy)
    candidates = iou >= iou_threshold
    if not class_agnostic:
        class_ids_a = detections_a.class_id
        class_ids_b = detections_b.class_id
        assert class_ids_a is not None
        assert class_ids_b is not None
        candidates &= class_ids_a[:, None] == class_ids_b[None, :]

    matched_indices = np.where(candidates)
    pairs = list(_greedy_match(iou, matched_indices))
    matched_pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if matched_pairs.shape[0] == 0:
        matched_a = np.empty(0, dtype=np.int64)
        matched_b = np.empty(0, dtype=np.int64)
    else:
        matched_a = matched_pairs[:, 0]
        matched_b = matched_pairs[:, 1]

    unmatched_a = np.setdiff1d(np.arange(len(detections_a), dtype=np.int64), matched_a)
    unmatched_b = np.setdiff1d(np.arange(len(detections_b), dtype=np.int64), matched_b)
    return matched_pairs, unmatched_a, unmatched_b
