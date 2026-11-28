"""Non-maximum suppression, merging and fusion of overlapping predictions."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void

from supervision.detection.compact_mask import CompactMask, _compact_mask_union
from supervision.detection.utils._fusion import _box_weighted_box_fusion
from supervision.detection.utils.iou import (
    OverlapMetric,
    _count_dtype,
    _count_pixels,
    _is_mergeable_overlap,
    _overlap_from_areas,
    _pack_masks,
    _use_packed,
    _validate_iou_threshold,
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.utils.internal import (
    _DEPRECATION_TEMPLATE_KEYWORD,
    _DEPRECATION_WARNING_STREAM,
    warn_deprecated,
)

_Candidate = TypeVar("_Candidate")


class FusionBoxWeights(Enum):
    """Enum specifying how members of a cluster are weighted when fusing them.

    Attributes:
        CONFIDENCE: Each member counts in proportion to its confidence, as in the
            Weighted Box Fusion paper. A cluster whose members all have zero
            confidence counts them equally.
        UNIFORM: Members have equal geometry weights before source weighting.
            The confidence formula for a fixed cluster is unchanged, but the
            geometry can change its membership.

    Example:
        ```pycon
        >>> FusionBoxWeights.from_value("UNIFORM")
        <FusionBoxWeights.UNIFORM: 'uniform'>

        ```
    """

    CONFIDENCE = "confidence"
    UNIFORM = "uniform"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string values of all members."""
        return list(map(lambda member: member.value, cls))

    @classmethod
    def from_value(cls, value: FusionBoxWeights | str) -> FusionBoxWeights:
        """Return the member for an enum member or a case-insensitive string.

        Args:
            value: A member of this enum or its string value.

        Returns:
            The matching member.

        Raises:
            ValueError: If `value` is not a member, a known string or a string.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
        )


#: Arguments shared by every deprecated array-level NMS, soft-NMS and Weighted Box
#: Fusion kernel: the release that deprecates them in favour of the `Detections`
#: methods, the release that removes them (three minor releases after the next
#: minor), the warning stream, and the docstring notice pydeprecate injects.
_ARRAY_KERNEL_DEPRECATED_DETAILS: dict[str, Any] = {
    "deprecated_in": "0.33.0",
    "remove_in": "0.36.0",
    "stream": _DEPRECATION_WARNING_STREAM,
    "update_docstring": True,
    "docstring_style": "mkdocs",
}


#: Recipe appended to the NMS kernels' warning, which returned a keep mask that
#: callers used to recover the surviving row indices.
_KEEP_INDICES_NOTE = (
    "To keep row indices, set `detections.data['index'] = "
    "np.arange(len(detections))` before `with_nms` and read `data['index']` "
    "afterwards."
)

_CLASSLESS_ARRAY_NOTE = "For five-column predictions, pass class_agnostic=True."


def _array_kernel_deprecation(replacement: str, note: str = "") -> str:
    """Build the warning template of a deprecated array-level kernel.

    Args:
        replacement: Markdown text naming the `Detections` call to use instead.
        note: Optional extra sentence appended to the warning.

    Returns:
        A template for pydeprecate's message keyword, with its `source_name`,
            `deprecated_in` and `remove_in` placeholders.
    """
    template = (
        "`%(source_name)s` is deprecated in `%(deprecated_in)s` and will be "
        f"removed in `%(remove_in)s`. Use {replacement} instead."
    )
    return " ".join(part for part in (template, _CLASSLESS_ARRAY_NOTE, note) if part)


def _mask_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """Perform NMS on segmentation predictions and return the keep mask.

    IoU is computed exactly on the full-resolution masks for both dense and
    :class:`~supervision.detection.compact_mask.CompactMask` inputs. The
    behaviour of the deprecated public
    :func:`~supervision.detection.utils.nms.mask_non_max_suppression`.

    Args:
        predictions: Rows `(x_min, y_min, x_max, y_max, score[, class])`, shape
            `(N, 5)` or `(N, 6)`.
        masks: Binary masks aligned with `predictions`, shape `(N, H, W)`.
        iou_threshold: Overlap above which a lower-scoring mask is suppressed.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        A boolean array, `True` for each prediction that survives.
    """
    _validate_iou_threshold(iou_threshold)
    rows, columns = predictions.shape

    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    sort_index = predictions[:, 4].argsort()[::-1]
    predictions = predictions[sort_index]
    masks = masks[sort_index]

    ious = mask_iou_batch(masks, masks, overlap_metric)
    categories = predictions[:, 5]

    keep = np.ones(rows, dtype=bool)
    for row_idx in range(rows):
        if keep[row_idx]:
            condition = (ious[row_idx] > iou_threshold) & (
                categories[row_idx] == categories
            )
            keep[row_idx + 1 :] = np.where(
                condition[row_idx + 1 :], False, keep[row_idx + 1 :]
            )

    return cast(npt.NDArray[np.bool_], keep[sort_index.argsort()])


@deprecated(  # type: ignore[untyped-decorator]
    target=_mask_non_max_suppression,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    args_mapping={"mask_dimension": None},
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(..., mask=masks).with_nms(threshold=..., "
            "overlap_metric=..., overlap_on='mask')`",
            _KEEP_INDICES_NOTE,
        )
    },
)
def mask_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> npt.NDArray[np.bool_]:
    """Perform Non-Maximum Suppression (NMS) on segmentation predictions."""
    return void(  # type: ignore[no-any-return]
        predictions, masks, iou_threshold, overlap_metric, mask_dimension
    )


def _mask_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Perform Soft-NMS on segmentation predictions and return decayed scores.

    Each confidence is rescaled by `exp(-iou**2 / sigma)` for every higher-scoring,
    same-category overlap. IoU is computed exactly on the full-resolution masks for
    both dense and :class:`~supervision.detection.compact_mask.CompactMask` inputs.
    The behaviour of the deprecated public
    :func:`~supervision.detection.utils.nms.mask_soft_non_max_suppression`.

    Args:
        predictions: Rows `(x_min, y_min, x_max, y_max, score[, class])`, shape
            `(N, 5)` or `(N, 6)`.
        masks: Binary masks aligned with `predictions`, shape `(N, H, W)`.
        sigma: Strength of the confidence decay; must be greater than `0`.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        The decayed confidence scores, in the order of `predictions`.
    """
    _validate_sigma(sigma)
    rows, columns = predictions.shape

    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    sort_index = predictions[:, 4].argsort()[::-1]
    predictions = predictions[sort_index]
    masks = masks[sort_index]

    ious = mask_iou_batch(masks, masks, overlap_metric)
    categories = predictions[:, 5]

    decayed = _soft_nms_decay_from_iou_matrix(
        ious, categories, predictions[:, 4], sigma
    )
    return decayed[sort_index.argsort()]


@deprecated(  # type: ignore[untyped-decorator]
    target=_mask_soft_non_max_suppression,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    args_mapping={"mask_dimension": None},
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(..., mask=masks).with_soft_nms(sigma=..., "
            "overlap_metric=..., overlap_on='mask')`"
        )
    },
)
def mask_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> npt.NDArray[np.floating]:
    """Perform Soft Non-Maximum Suppression (Soft-NMS) on segmentation predictions."""
    return void(  # type: ignore[no-any-return]
        predictions, masks, sigma, overlap_metric, mask_dimension
    )


def _sort_for_nms(
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.number]]:
    """Sort indices by descending score and align category IDs for NMS."""
    sort_index = np.flip(scores.argsort())
    if categories is None:
        sorted_categories: npt.NDArray[np.number] = np.zeros(len(scores), dtype=np.intp)
    else:
        sorted_categories = categories[sort_index]
    return sort_index, sorted_categories


def _prepare_predictions_for_nms(
    predictions: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Add an agnostic class column when missing, sort by descending score.

    Returns the score-descending sort index, the reordered predictions, and the category
    vector for the loop callers to consume.
    """
    rows, columns = predictions.shape
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]
    categories = predictions[:, 5]
    return sort_index, predictions, categories


def _nms_loop_from_iou_matrix(
    ious: npt.NDArray[np.floating],
    categories: npt.NDArray[np.number],
    iou_threshold: float,
) -> npt.NDArray[np.bool_]:
    """Greedy NMS suppression loop given a precomputed pairwise IoU matrix.

    Assumes `ious` is square with row/column order matching `categories`. Detections
    sharing a category whose IoU exceeds `iou_threshold` are dropped in favour of the
    higher-confidence entry.
    """
    rows = len(ious)
    ious = ious - np.eye(rows)
    keep: npt.NDArray[np.bool_] = np.ones(rows, dtype=bool)
    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition
    return keep


def _validate_sigma(sigma: float) -> None:
    """Raise `ValueError` when a Soft-NMS `sigma` is not strictly positive."""
    if not sigma > 0:
        raise ValueError(f"Value of `sigma` must be greater than 0, {sigma} given.")


def _soft_nms_decay_from_iou_matrix(
    ious: npt.NDArray[np.floating],
    categories: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    sigma: float,
) -> npt.NDArray[np.floating]:
    """Vectorized Gaussian Soft-NMS confidence decay given a precomputed IoU matrix.

    Assumes `ious`, `categories`, and `scores` are all sorted by descending score
    (as produced by
    :func:`~supervision.detection.utils.nms._prepare_predictions_for_nms`),
    and that `ious` is square with
    row/column order matching `categories`. Each detection's score is decayed once
    per higher-scoring, same-category detection that precedes it — equivalent to
    the reference single-pass (no re-sort) Soft-NMS loop, computed as a single
    vectorized closed form: `decayed[j] = scores[j] * exp(-sum_i(ious[i, j]**2) /
    sigma)` summed over same-category `i < j`.
    """
    rows = len(ious)
    same_category = categories[:, None] == categories[None, :]
    precedes = np.triu(np.ones((rows, rows), dtype=bool), k=1)
    weighted_iou_sq = np.where(precedes & same_category, ious**2, 0.0)
    decay_exponent = weighted_iou_sq.sum(axis=0)
    return cast(npt.NDArray[np.floating], scores * np.exp(-decay_exponent / sigma))


def _box_non_max_suppression_from_arrays(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """Perform NMS directly on box, score, and optional category arrays."""
    _validate_iou_threshold(iou_threshold)
    sort_index, sorted_categories = _sort_for_nms(scores, categories)
    sorted_boxes = boxes[sort_index]
    ious = box_iou_batch(sorted_boxes, sorted_boxes, overlap_metric)
    keep = _nms_loop_from_iou_matrix(ious, sorted_categories, iou_threshold)
    result: npt.NDArray[np.bool_] = keep[sort_index.argsort()]
    return result


def _box_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """Perform NMS on `(n, 5)` or `(n, 6)` box predictions and return a keep mask.

    The behaviour of the deprecated public
    :func:`~supervision.detection.utils.nms.box_non_max_suppression`.

    Args:
        predictions: Rows `(x_min, y_min, x_max, y_max, score[, class])`.
        iou_threshold: Overlap above which a lower-scoring box is suppressed.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        A boolean array, `True` for each prediction that survives.
    """
    categories = predictions[:, 5] if predictions.shape[1] > 5 else None
    return _box_non_max_suppression_from_arrays(
        boxes=predictions[:, :4],
        scores=predictions[:, 4],
        categories=categories,
        iou_threshold=iou_threshold,
        overlap_metric=overlap_metric,
    )


@deprecated(  # type: ignore[untyped-decorator]
    target=_box_non_max_suppression,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(xyxy=..., confidence=..., class_id=...)"
            ".with_nms(threshold=..., overlap_metric=...)`",
            _KEEP_INDICES_NOTE,
        )
    },
)
def box_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """Perform Non-Maximum Suppression (NMS) on object detection predictions."""
    return void(predictions, iou_threshold, overlap_metric)  # type: ignore[no-any-return]


def _box_soft_non_max_suppression_from_arrays(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Perform Soft-NMS directly on box, score, and optional category arrays."""
    _validate_sigma(sigma)
    sort_index, sorted_categories = _sort_for_nms(scores, categories)
    sorted_boxes = boxes[sort_index]
    ious = box_iou_batch(sorted_boxes, sorted_boxes, overlap_metric)
    score_dtype = (
        np.result_type(boxes.dtype, scores.dtype, np.float64)
        if categories is None
        else np.result_type(boxes.dtype, scores.dtype, categories.dtype)
    )
    decayed = _soft_nms_decay_from_iou_matrix(
        ious,
        sorted_categories,
        scores[sort_index].astype(score_dtype, copy=False),
        sigma,
    )
    result_scores: npt.NDArray[np.floating] = decayed[sort_index.argsort()]
    return result_scores


def _box_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Perform Soft-NMS on `(n, 5)` or `(n, 6)` box predictions.

    The behaviour of the deprecated public
    :func:`~supervision.detection.utils.nms.box_soft_non_max_suppression`.

    Args:
        predictions: Rows `(x_min, y_min, x_max, y_max, score[, class])`.
        sigma: Strength of the confidence decay; must be greater than `0`.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        The decayed confidence scores, in the order of `predictions`.
    """
    categories = predictions[:, 5] if predictions.shape[1] > 5 else None
    return _box_soft_non_max_suppression_from_arrays(
        boxes=predictions[:, :4],
        scores=predictions[:, 4],
        categories=categories,
        sigma=sigma,
        overlap_metric=overlap_metric,
    )


@deprecated(  # type: ignore[untyped-decorator]
    target=_box_soft_non_max_suppression,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(xyxy=..., confidence=..., class_id=...)"
            ".with_soft_nms(sigma=..., overlap_metric=...)`"
        )
    },
)
def box_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Perform Soft Non-Maximum Suppression (Soft-NMS) on object detection
    predictions."""
    return void(predictions, sigma, overlap_metric)  # type: ignore[no-any-return]


def _group_overlapping_masks(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[np.bool_],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Apply greedy version of non-maximum merging to avoid detecting too many.

    Args:
        predictions: An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        masks: A 3D array of binary masks corresponding to
            the predictions.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged.
            Each group may have 1 or more elements.
    """
    if _use_packed(masks):
        return _group_overlapping_packed_masks(
            predictions, masks, iou_threshold, overlap_metric
        )
    return _greedy_nmm_via_mask_candidate(
        predictions, masks, iou_threshold, overlap_metric
    )


def _greedy_merge_groups(
    scores: npt.NDArray[np.floating],
    iou_threshold: float,
    start: Callable[[int], _Candidate],
    overlap: Callable[[npt.NDArray[np.int_], _Candidate], npt.NDArray[np.floating]],
    union: Callable[[_Candidate, npt.NDArray[np.int_]], _Candidate],
) -> list[list[int]]:
    """Internal function.

    Greedy non-maximum merging shared by the dense, bit-packed and compact mask
    representations: take the best remaining prediction, repeatedly absorb every
    remaining prediction that overlaps the growing union of the group, and start
    a new group when none does. Only how a candidate is built, compared and
    grown differs by representation, so the grouping rule lives here once.

    Args:
        scores: Confidence of each prediction, shape `(n,)`.
        iou_threshold: Minimum overlap, in the closed range from 0 to 1, for a
            prediction to join the group.
        start: Builds the candidate for the best remaining prediction index.
        overlap: Returns the overlaps, shape `(len(order),)`, of the predictions
            at `order` against the candidate.
        union: Grows the candidate with the predictions at the given indices.

    Returns:
        Groups of prediction indices to be merged.
            Each group may have 1 or more elements.
    """
    merge_groups: list[list[int]] = []
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        candidate = start(idx)
        merge_group = [idx]
        while len(order) > 0:
            above_threshold = _is_mergeable_overlap(
                overlap(order, candidate), iou_threshold
            )
            if not above_threshold.any():
                break
            above_idx = order[above_threshold]
            candidate = union(candidate, above_idx)
            merge_group.extend(np.flip(above_idx).tolist())
            order = order[~above_threshold]
        merge_groups.append(merge_group)
    return merge_groups


def _group_overlapping_packed_masks(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[np.bool_],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Internal function.

    Bit-packed equivalent of `_group_overlapping_masks` for boolean masks on
    NumPy >= 2.0. Every mask is packed once and the merge candidate stays packed,
    so each step is a popcount over words instead of a fresh float copy of the
    masks. Counts are cast to the same float dtype as `mask_iou_batch` uses, so
    the groups are identical.

    Args:
        predictions: An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        masks: A 3D array of binary masks corresponding to
            the predictions.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged.
            Each group may have 1 or more elements.

    Raises:
        ValueError: If more than one mask is given and `masks` is not 3D.
    """
    if len(masks) > 1 and masks.ndim != 3:
        # The dense route validates this inside `mask_iou_batch`; keep the error.
        raise ValueError(
            f"masks must be 3D (N, H, W); got ndim={masks.ndim}. "
            "Dense masks are compared as (N, H, W) arrays."
        )

    words = _pack_masks(masks)
    count_dtype = _count_dtype(int(np.prod(masks.shape[1:])))
    areas = _count_pixels(words).astype(count_dtype)

    def start(idx: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating]]:
        """Return the packed words and area of the prediction at `idx`."""
        return words[idx : idx + 1], areas[idx : idx + 1]

    def overlap(
        order: npt.NDArray[np.int_],
        candidate: tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating]],
    ) -> npt.NDArray[np.floating]:
        """Return the overlaps of the predictions at `order` with the candidate."""
        candidate_words, candidate_area = candidate
        intersection_area = _count_pixels(words[order] & candidate_words)
        return _overlap_from_areas(
            intersection_area.astype(count_dtype)[:, None],
            areas[order],
            candidate_area,
            overlap_metric,
        ).flatten()

    def union(
        candidate: tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating]],
        above_idx: npt.NDArray[np.int_],
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating]]:
        """Grow the candidate with the predictions at `above_idx`."""
        merged = candidate[0] | np.bitwise_or.reduce(
            words[above_idx], axis=0, keepdims=True
        )
        return merged, _count_pixels(merged).astype(count_dtype)

    return _greedy_merge_groups(predictions[:, 4], iou_threshold, start, overlap, union)


def mask_non_max_merge(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    *args: Any,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    mask_dimension: int = 640,
) -> list[list[int]]:
    """Perform Non-Maximum Merging (NMM) on segmentation predictions.

    Args:
        predictions: A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks: A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum merging.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of masks (e.g., IoU, IoS).
        mask_dimension: Deprecated in `0.30.0`, removed in `0.33.0`. No longer
            used; the parameter is silently ignored. Passing `mask_dimension`
            positionally emits a deprecation warning.

    Returns:
        A list of groups of prediction indices. Each inner list contains
            the indices of predictions whose masks overlap above `iou_threshold`
            according to the chosen `overlap_metric`, and should be merged or
            kept together as a single detection by non-maximum merging.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.
        TypeError: If more than five positional arguments are passed.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> masks = np.zeros((2, 4, 4), dtype=bool)
        >>> masks[:, :2, :2] = True
        >>> sv.mask_non_max_merge(predictions, masks, iou_threshold=0.5)
        [[0, 1]]

        ```
    """

    _validate_iou_threshold(iou_threshold)
    if len(args) > 2:
        raise TypeError(
            "mask_non_max_merge accepts at most five positional arguments. "
            "Pass overlap_metric and mask_dimension by keyword."
        )
    if args:
        warn_deprecated(
            "Passing `overlap_metric` or `mask_dimension` positionally to "
            "`mask_non_max_merge` is deprecated in `0.30.0` and will be removed "
            "in `0.33.0`. Pass them by keyword instead."
        )
        first = args[0]
        if isinstance(first, OverlapMetric):
            overlap_metric = first
        else:
            mask_dimension = cast(int, first)
        if len(args) == 2:
            second = args[1]
            if isinstance(first, OverlapMetric):
                mask_dimension = cast(int, second)
            else:
                overlap_metric = cast(OverlapMetric, second)

    del mask_dimension

    def group_within(global_indices: npt.NDArray[np.int_]) -> list[list[int]]:
        if isinstance(masks, CompactMask):
            return _group_overlapping_masks_pairwise(
                predictions[global_indices],
                masks[global_indices],
                iou_threshold,
                overlap_metric,
            )
        return _group_overlapping_masks(
            predictions[global_indices],
            masks[global_indices],
            iou_threshold,
            overlap_metric,
        )

    return _non_max_merge_per_category(predictions, group_within)


def _update_mask_candidate(
    masks: npt.NDArray[Any] | CompactMask,
    candidate: npt.NDArray[Any] | CompactMask,
    above_idx: npt.NDArray[np.int_],
) -> npt.NDArray[Any] | CompactMask:
    """Union newly matched masks into the candidate, retaining compact storage."""
    if isinstance(masks, CompactMask):
        compact_candidate = cast(CompactMask, candidate)
        # `above_idx` is one-dimensional, so indexing returns a CompactMask.
        matched_masks = cast(CompactMask, masks[above_idx])
        return _compact_mask_union([matched_masks, compact_candidate])
    dense_candidate = cast(npt.NDArray[Any], candidate)
    dense_union: npt.NDArray[Any] = np.logical_or.reduce(
        np.concatenate([masks[above_idx], dense_candidate]),
        axis=0,
        keepdims=True,
    )
    return dense_union


def _greedy_nmm_via_mask_candidate(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Group masks by exact overlap while updating the merged candidate union."""
    return _greedy_merge_groups(
        predictions[:, 4],
        iou_threshold,
        lambda idx: masks[[idx]],
        lambda order, candidate: mask_iou_batch(
            masks[order], candidate, overlap_metric
        ).flatten(),
        lambda candidate, above_idx: _update_mask_candidate(
            masks, candidate, above_idx
        ),
    )


def _group_overlapping_masks_pairwise(
    predictions: npt.NDArray[np.floating],
    masks: npt.NDArray[Any] | CompactMask,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    return _greedy_nmm_via_mask_candidate(
        predictions, masks, iou_threshold, overlap_metric
    )


def _non_max_merge_by_categories(
    row_count: int,
    categories: npt.NDArray[np.number] | None,
    group_within: Callable[[npt.NDArray[np.int_]], list[list[int]]],
) -> list[list[int]]:
    """Dispatch NMM grouping per category and map indices back globally."""
    if categories is None:
        global_indices = np.arange(row_count, dtype=int)
        return [
            global_indices[group].tolist() for group in group_within(global_indices)
        ]

    merge_groups: list[list[int]] = []
    for category_id in np.unique(categories):
        curr_indices = np.where(categories == category_id)[0]
        for local_group in group_within(curr_indices):
            merge_groups.append(curr_indices[local_group].tolist())

    for merge_group in merge_groups:
        if len(merge_group) == 0:
            raise ValueError(
                f"Empty group detected when non-max-merging detections: {merge_groups}"
            )
    return merge_groups


def _non_max_merge_per_category(
    predictions: npt.NDArray[np.floating],
    group_within: Callable[[npt.NDArray[np.int_]], list[list[int]]],
) -> list[list[int]]:
    """Dispatch NMM grouping per class, then translate local indices back to the global
    row positions of ``predictions``.

    ``group_within(global_indices)`` must return merge groups expressed in terms of
    *positions inside `global_indices`*, not absolute row positions. When
    ``predictions`` has no class column, a single pass over all rows is performed
    instead of per-category iteration.
    """
    categories = None if predictions.shape[1] == 5 else predictions[:, 5]
    return _non_max_merge_by_categories(len(predictions), categories, group_within)


def _group_overlapping_box_arrays(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Apply greedy non-maximum merging directly on box and score arrays."""
    merge_groups: list[list[int]] = []
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        ious = box_iou_batch(
            boxes[order],
            boxes[idx : idx + 1],
            overlap_metric,
        ).flatten()
        above_threshold = _is_mergeable_overlap(ious, iou_threshold)
        merge_group = [idx, *np.flip(order[above_threshold]).tolist()]
        merge_groups.append(merge_group)
        order = order[~above_threshold]
    return merge_groups


def _group_overlapping_boxes(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.

    Args:
        predictions: An array of shape `(n, 5)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            and the confidence scores.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged.
            Each group may have 1 or more elements.
    """
    return _group_overlapping_box_arrays(
        boxes=predictions[:, :4],
        scores=predictions[:, 4],
        iou_threshold=iou_threshold,
        overlap_metric=overlap_metric,
    )


def _box_non_max_merge_from_arrays(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Perform greedy NMM per category directly on box, score, and category arrays."""
    _validate_iou_threshold(iou_threshold)

    def group_within(global_indices: npt.NDArray[np.int_]) -> list[list[int]]:
        return _group_overlapping_box_arrays(
            boxes=boxes[global_indices],
            scores=scores[global_indices],
            iou_threshold=iou_threshold,
            overlap_metric=overlap_metric,
        )

    return _non_max_merge_by_categories(len(boxes), categories, group_within)


def box_non_max_merge(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Apply greedy version of non-maximum merging per category to avoid detecting too
    many overlapping bounding boxes for a given object.

    Args:
        predictions: An array of shape `(n, 5)` or `(n, 6)`
            containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`,
            the confidence scores and class_ids. Omit class_id column to allow
            detections of different classes to be merged.
        iou_threshold: The intersection-over-union threshold
            to use for non-maximum suppression. Defaults to 0.5.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of boxes (e.g., IoU, IoS).

    Returns:
        list[list[int]]: Groups of prediction indices be merged.
            Each group may have 1 or more elements.

    Raises:
        ValueError: If `iou_threshold` is not within the closed range
            from `0` to `1`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> predictions = np.array([
        ...     [0, 0, 4, 4, 0.9, 0],
        ...     [0, 0, 4, 4, 0.8, 0],
        ... ])
        >>> sv.box_non_max_merge(predictions, iou_threshold=0.5)
        [[0, 1]]

        ```
    """
    categories = predictions[:, 5] if predictions.shape[1] > 5 else None
    return _box_non_max_merge_from_arrays(
        boxes=predictions[:, :4],
        scores=predictions[:, 4],
        categories=categories,
        iou_threshold=iou_threshold,
        overlap_metric=overlap_metric,
    )


@deprecated(  # type: ignore[untyped-decorator]
    target=_box_weighted_box_fusion,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(xyxy=..., confidence=..., class_id=...)"
            ".with_fusion(threshold=..., overlap_metric=..., num_models=...)`"
        )
    },
)
def box_weighted_box_fusion(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    num_models: int | None = None,
) -> npt.NDArray[np.floating]:
    """Fuse overlapping object detection predictions with Weighted Box Fusion."""
    return void(  # type: ignore[no-any-return]
        predictions, iou_threshold, overlap_metric, num_models
    )


def _validate_oriented_nms_inputs(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
) -> None:
    """Check that predictions and oriented boxes have valid, matching shapes."""
    for name, arr in (("predictions", predictions), ("oriented_boxes", oriented_boxes)):
        if name == "predictions":
            if arr.ndim != 2 or arr.shape[1] not in (5, 6):
                raise ValueError(
                    f"`{name}` has shape {arr.shape}; expected (N, 5) or (N, 6)."
                )
            continue
        if arr.ndim == 3 and arr.shape[1:] != (4, 2):
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 4, 2) "
                f"— each box must have exactly 4 corners with (x, y) coordinates."
            )
        elif arr.ndim == 2 and arr.shape[1] != 8:
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 8) for flat "
                f"YOLO format or (N, 4, 2) for corner format."
            )
        elif arr.ndim not in (2, 3):
            raise ValueError(
                f"`{name}` must be 2-D (N, 8) or 3-D (N, 4, 2), got shape {arr.shape}."
            )
    if len(predictions) != len(oriented_boxes):
        raise ValueError(
            f"`predictions` and `oriented_boxes` must have the same length, "
            f"got {len(predictions)} and {len(oriented_boxes)}."
        )


def _oriented_box_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """Perform NMS on oriented bounding box predictions and return a keep mask.

    Overlap is computed on the four corners of each box. The behaviour of the
    deprecated public
    :func:`~supervision.detection.utils.nms.oriented_box_non_max_suppression`.

    Args:
        predictions: Rows `(x_min, y_min, x_max, y_max, score[, class])`; only the
            score and optional class are read.
        oriented_boxes: Corners of each box, shape `(N, 4, 2)`, aligned with
            `predictions`.
        iou_threshold: Overlap above which a lower-scoring box is suppressed.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        A boolean array, `True` for each prediction that survives.
    """
    _validate_iou_threshold(iou_threshold)
    _validate_oriented_nms_inputs(predictions, oriented_boxes)
    sort_index, _, categories = _prepare_predictions_for_nms(predictions)
    oriented_boxes = oriented_boxes[sort_index]
    # same object intentional — triggers upper-triangle optimization
    ious = oriented_box_iou_batch(oriented_boxes, oriented_boxes, overlap_metric)
    keep = _nms_loop_from_iou_matrix(ious, categories, iou_threshold)
    result: npt.NDArray[np.bool_] = keep[sort_index.argsort()]
    return result


@deprecated(  # type: ignore[untyped-decorator]
    target=_oriented_box_non_max_suppression,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(..., data={'xyxyxyxy': corners}).with_nms("
            "threshold=..., overlap_metric=..., overlap_on='box')`",
            _KEEP_INDICES_NOTE,
        )
    },
)
def oriented_box_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.bool_]:
    """Perform Non-Maximum Suppression on oriented bounding box predictions."""
    return void(  # type: ignore[no-any-return]
        predictions, oriented_boxes, iou_threshold, overlap_metric
    )


def _oriented_box_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Perform Soft-NMS on oriented bounding box predictions.

    Overlap is computed on the four corners of each box. The behaviour of the
    deprecated public
    :func:`~supervision.detection.utils.nms.oriented_box_soft_non_max_suppression`.

    Args:
        predictions: Rows `(x_min, y_min, x_max, y_max, score[, class])`; only the
            score and optional class are read.
        oriented_boxes: Corners of each box, shape `(N, 4, 2)`, aligned with
            `predictions`.
        sigma: Strength of the confidence decay; must be greater than `0`.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        The decayed confidence scores, in the order of `predictions`.
    """
    _validate_sigma(sigma)
    _validate_oriented_nms_inputs(predictions, oriented_boxes)
    sort_index, predictions, categories = _prepare_predictions_for_nms(predictions)
    oriented_boxes = oriented_boxes[sort_index]
    ious = oriented_box_iou_batch(oriented_boxes, oriented_boxes, overlap_metric)
    decayed = _soft_nms_decay_from_iou_matrix(
        ious, categories, predictions[:, 4], sigma
    )
    result_scores: npt.NDArray[np.floating] = decayed[sort_index.argsort()]
    return result_scores


@deprecated(  # type: ignore[untyped-decorator]
    target=_oriented_box_soft_non_max_suppression,
    **_ARRAY_KERNEL_DEPRECATED_DETAILS,
    **{
        _DEPRECATION_TEMPLATE_KEYWORD: _array_kernel_deprecation(
            "`sv.Detections(..., data={'xyxyxyxy': corners}).with_soft_nms("
            "sigma=..., overlap_metric=..., overlap_on='box')`"
        )
    },
)
def oriented_box_soft_non_max_suppression(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    sigma: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> npt.NDArray[np.floating]:
    """Perform Soft Non-Maximum Suppression on oriented bounding box predictions."""
    return void(  # type: ignore[no-any-return]
        predictions, oriented_boxes, sigma, overlap_metric
    )


def _group_overlapping_oriented_boxes(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Greedy non-maximum merging on oriented boxes.

    Mirrors
    :func:`~supervision.detection.utils.nms._group_overlapping_boxes`
    but uses :func:`~supervision.detection.utils.iou.oriented_box_iou_batch`.
    """
    merge_groups: list[list[int]] = []
    scores = predictions[:, 4]
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        ious = oriented_box_iou_batch(
            oriented_boxes[order],
            oriented_boxes[idx][None, ...],
            overlap_metric,
        ).flatten()
        above_threshold = _is_mergeable_overlap(ious, iou_threshold)
        merge_group = [idx, *np.flip(order[above_threshold]).tolist()]
        merge_groups.append(merge_group)
        order = order[~above_threshold]
    return merge_groups


def oriented_box_non_max_merge(
    predictions: npt.NDArray[np.floating],
    oriented_boxes: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> list[list[int]]:
    """Perform Non-Maximum Merging on oriented bounding box predictions, grouped per
    category.

    Mirrors :func:`~supervision.detection.utils.nms.box_non_max_merge`
    but uses oriented-box IoU, so groups of rotated detections sharing the same
    body — rather than the same
    axis-aligned bounding box — are merged.

    Args:
        predictions: An array of shape ``(n, 5)`` or ``(n, 6)`` containing
            the axis-aligned coordinates ``[x1, y1, x2, y2]``, confidence
            scores, and optionally class ids. Only the score and optional
            class are used by the grouping logic; overlap is computed on
            ``oriented_boxes``.
        oriented_boxes: Array of shape ``(N, 4, 2)`` containing the four
            ``(x, y)`` corners of each oriented box.
        iou_threshold: The intersection-over-union threshold to use for
            non-maximum merging.
        overlap_metric: Metric used to compute the degree of overlap
            between pairs of oriented boxes (e.g., IoU, IoS).

    Returns:
        Groups of prediction indices to be merged. Each group may have 1
            or more elements.

    Raises:
        ValueError: If ``iou_threshold`` is not within the closed range
            from 0 to 1.
        ValueError: If ``predictions`` and ``oriented_boxes`` have
            mismatched lengths.
        ValueError: If ``predictions`` or ``oriented_boxes`` has an invalid shape.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> oriented_boxes = np.array([
        ...     [[10, 10], [50, 10], [50, 30], [10, 30]],
        ...     [[11, 11], [51, 11], [51, 31], [11, 31]],
        ... ], dtype=np.float32)
        >>> predictions = np.array([
        ...     [10, 10, 50, 30, 0.9, 0],
        ...     [11, 11, 51, 31, 0.8, 0],
        ... ], dtype=np.float32)
        >>> groups = sv.oriented_box_non_max_merge(
        ...     predictions=predictions,
        ...     oriented_boxes=oriented_boxes,
        ...     iou_threshold=0.5,
        ... )
        >>> len(groups)
        1

        ```
    """
    for name, arr in (("predictions", predictions), ("oriented_boxes", oriented_boxes)):
        if name == "predictions":
            if arr.ndim != 2 or arr.shape[1] not in (5, 6):
                raise ValueError(
                    f"`{name}` has shape {arr.shape}; expected (N, 5) or (N, 6)."
                )
            continue
        if arr.ndim == 3 and arr.shape[1:] != (4, 2):
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 4, 2) "
                f"— each box must have exactly 4 corners with (x, y) coordinates."
            )
        elif arr.ndim == 2 and arr.shape[1] != 8:
            raise ValueError(
                f"`{name}` has shape {arr.shape}; expected (N, 8) for flat "
                f"YOLO format or (N, 4, 2) for corner format."
            )
        elif arr.ndim not in (2, 3):
            raise ValueError(
                f"`{name}` must be 2-D (N, 8) or 3-D (N, 4, 2), got shape {arr.shape}."
            )
    if len(predictions) != len(oriented_boxes):
        raise ValueError(
            f"`predictions` and `oriented_boxes` must have the same length, "
            f"got {len(predictions)} and {len(oriented_boxes)}."
        )
    _validate_iou_threshold(iou_threshold)

    def group_within(global_indices: npt.NDArray[np.int_]) -> list[list[int]]:
        return _group_overlapping_oriented_boxes(
            predictions[global_indices],
            oriented_boxes[global_indices],
            iou_threshold,
            overlap_metric,
        )

    return _non_max_merge_per_category(predictions, group_within)
