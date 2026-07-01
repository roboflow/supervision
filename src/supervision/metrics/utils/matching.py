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
