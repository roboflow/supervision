from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


def _greedy_match(
    iou: npt.NDArray[np.float32],
    matched_indices: tuple[npt.NDArray[np.intp], ...],
) -> Iterator[tuple[int, int]]:
    """Yield (target_idx, pred_idx) pairs in greedy highest-IoU-first one-to-one order.

    Candidate pairs are sorted by descending IoU and assigned one-to-one: a pair
    is accepted only when neither the target nor the prediction has been matched.

    Examples:
        >>> import numpy as np
        >>> iou = np.array([[1.0, 0.667], [0.333, 0.538]], dtype=np.float32)
        >>> matched_indices = np.where(iou >= 0.5)
        >>> list(_greedy_match(iou, matched_indices))
        [(0, 0), (1, 1)]
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
