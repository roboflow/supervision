"""Non-maximum suppression, merging and fusion of overlapping predictions."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from fractions import Fraction
from typing import Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void

from supervision.detection.compact_mask import CompactMask, _compact_mask_union
from supervision.detection.utils._obb_frame import (
    _offsets_from_origin,
    corners_to_frame,
    frame_to_corners,
    seed_frame,
)
from supervision.detection.utils.iou import (
    OverlapMetric,
    _box_overlap_float64,
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

#: Weighting of the member boxes in the Weighted Box Fusion mean: by confidence
#: (the paper's rule) or equally.
_FusionBoxWeights = Literal["confidence", "uniform"]

#: Largest ratio between the biggest and the smallest source weight. Beyond it a
#: weighted confidence can underflow, so the weights are rejected.
_MAX_SOURCE_WEIGHT_RATIO = 2.0**100


class FusionBoxWeights(Enum):
    """Enum specifying how members of a cluster are weighted when fusing them.

    Attributes:
        CONFIDENCE: Each member counts in proportion to its confidence, as in the
            Weighted Box Fusion paper. A cluster whose members all have zero
            confidence counts them equally.
        UNIFORM: Every member counts the same. The fused confidence is not affected.

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


def _validate_num_models(num_models: int | None) -> None:
    """Reject a Weighted Box Fusion model count that is not a positive integer.

    ``None`` passes and means no rescaling. A ``bool`` is rejected even though it
    is an ``int`` subclass, because ``True`` silently standing for one model
    would hide a caller mistake.

    Args:
        num_models: The model count passed to
            :func:`~supervision.detection.utils.nms.box_weighted_box_fusion`.

    Raises:
        ValueError: If `num_models` is not `None` and not a positive integer.
    """
    if num_models is None:
        return
    is_integer = isinstance(num_models, (int, np.integer)) and not isinstance(
        num_models, bool
    )
    if not is_integer or num_models < 1:
        raise ValueError(
            f"num_models must be a positive integer or None, got {num_models!r}. "
            "Pass the number of models whose predictions were concatenated, "
            "or leave it as None to keep the plain mean confidence."
        )


def _prepare_fusion_sources(
    sources: npt.ArrayLike | None,
    source_weights: npt.ArrayLike | None,
    num_models: int | None,
    count: int,
) -> tuple[npt.NDArray[np.integer] | None, npt.NDArray[np.float64] | None]:
    """Validate the source ids and source weights of a Weighted Box Fusion.

    Args:
        sources: Source (model or pass) id of each row, or `None` to keep the
            clustering free of sources.
        source_weights: Weight of each source id, indexed by id, or `None`.
        num_models: Number of sources. Ids are checked against it when it is given,
            and `source_weights` needs it.
        count: Number of rows that `sources` must label.

    Returns:
        The ids as an integer array, and the weights as a float64 array, or `None`
            for each that is not given. Weights that are all equal give the same
            fusion as no weights, so they come back as `None`.

    Raises:
        ValueError: If `source_weights` is given without `sources` or `num_models`,
            `sources` is not one integer id per row (in `[0, num_models)` when
            `num_models` is given), or `source_weights` is not one finite positive
            weight per source whose largest value is at most `2**100` times the
            smallest.
    """
    if sources is None:
        if source_weights is not None:
            raise ValueError("source_weights requires sources to be given.")
        return None, None
    source_ids = np.asarray(sources)
    if source_ids.shape != (count,):
        raise ValueError(
            f"sources must hold one id per detection, shape ({count},), got "
            f"shape {source_ids.shape}."
        )
    if count:
        if not np.issubdtype(source_ids.dtype, np.integer):
            raise ValueError(f"sources must be integers, got {source_ids.dtype}.")
        if num_models is not None and (
            source_ids.min() < 0 or source_ids.max() >= num_models
        ):
            raise ValueError(
                f"sources must lie in [0, num_models) = [0, {num_models}), got "
                f"values from {source_ids.min()} to {source_ids.max()}."
            )
    if source_weights is None:
        return source_ids, None
    if num_models is None:
        raise ValueError("source_weights requires num_models, one weight per source.")
    weights = np.asarray(source_weights, dtype=np.float64)
    if weights.shape != (num_models,):
        raise ValueError(
            f"source_weights must hold one weight per source, shape ({num_models},), "
            f"got shape {weights.shape}."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("source_weights must be finite and strictly positive.")
    if weights.max() / weights.min() > _MAX_SOURCE_WEIGHT_RATIO:
        raise ValueError(
            "The largest source weight must be at most 2**100 times the smallest."
        )
    if np.all(weights == weights[0]):
        return source_ids, None
    return source_ids, weights


def _validate_fusion_predictions_shape(predictions: npt.NDArray[np.number]) -> None:
    """Reject Weighted Box Fusion predictions not shaped `(n, 5)` or `(n, 6)`.

    Without this check a `(n, 4)` array or a single 1-D row fails later with an
    opaque ``IndexError``, and a `(n, 7)` array is silently cut down to six
    columns. Empty input must carry five or six columns as well, so the empty
    result keeps the same column count as a fused one.

    Args:
        predictions: The predictions passed to
            :func:`~supervision.detection.utils.nms.box_weighted_box_fusion`.

    Raises:
        ValueError: If `predictions` is not a 2-D array with 5 or 6 columns.
    """
    if predictions.ndim != 2 or predictions.shape[1] not in (5, 6):
        raise ValueError(
            "Predictions passed to Weighted Box Fusion must have shape (n, 5) "
            "or (n, 6), one [x1, y1, x2, y2, confidence] row per box with an "
            f"optional class id column, got shape {predictions.shape}. Stack the "
            "boxes, confidences and class ids into one 2-D array before fusing."
        )


def _validate_fusion_confidences(confidences: npt.NDArray[np.number]) -> None:
    """Reject Weighted Box Fusion confidences that are negative or not finite.

    Confidences weight the fused coordinates, so a negative score turns the
    weighted mean into an extrapolation outside every member box, and a NaN or
    infinite score makes the fused box or score NaN. Zero is allowed: a cluster
    whose scores are all zero is fused with a plain average instead.

    Args:
        confidences: The confidence column of the predictions passed to
            :func:`~supervision.detection.utils.nms.box_weighted_box_fusion`,
            shape `(n,)`.

    Raises:
        ValueError: If any confidence is negative, NaN, or infinite.
    """
    invalid = ~np.isfinite(confidences) | (confidences < 0)
    if invalid.any():
        invalid_rows = np.flatnonzero(invalid)
        raise ValueError(
            "Confidences passed to Weighted Box Fusion must be finite and "
            f"non-negative, got {confidences[invalid_rows[:5]].tolist()} at rows "
            f"{invalid_rows[:5].tolist()}. Drop or clip those predictions before "
            "fusing them."
        )


def _validate_fusion_boxes(boxes: npt.NDArray[np.number]) -> None:
    """Reject Weighted Box Fusion boxes with non-finite or reversed corners.

    A box with a NaN coordinate overlaps nothing, so it would come back unfused
    as a NaN box, and an infinite coordinate makes the overlap arithmetic NaN. A
    box with `x2 < x1` or `y2 < y1` is not a box at all. Zero-width and
    zero-height boxes (`x2 == x1` or `y2 == y1`) are accepted: they overlap
    nothing and come back unfused.

    Args:
        boxes: The `[x1, y1, x2, y2]` columns of the predictions passed to
            :func:`~supervision.detection.utils.nms.box_weighted_box_fusion`,
            shape `(n, 4)`.

    Raises:
        ValueError: If any box has a NaN or infinite coordinate.
        ValueError: If any box has `x2 < x1` or `y2 < y1`.
    """
    finite = np.isfinite(boxes).all(axis=1)
    # A NaN corner fails both comparisons, so it is flagged even without `finite`.
    ordered = (boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])
    invalid = ~(finite & ordered)
    if invalid.any():
        invalid_rows = np.flatnonzero(invalid)
        raise ValueError(
            "Boxes passed to Weighted Box Fusion must have finite "
            "[x1, y1, x2, y2] coordinates with x1 <= x2 and y1 <= y2, got "
            f"{boxes[invalid_rows[:5]].tolist()} at rows "
            f"{invalid_rows[:5].tolist()}. Drop those predictions or fix their "
            "corners before fusing them."
        )


def _validate_fusion_class_ids(class_ids: npt.NDArray[np.number]) -> None:
    """Reject Weighted Box Fusion class ids that are NaN or infinite.

    Boxes are fused only within a class, and :func:`numpy.unique` collapses every
    NaN into one group, so NaN class ids would be fused together and come back
    with a NaN class. Fractional and negative class ids are not rejected: they
    are grouped by their exact value.

    Args:
        class_ids: The class id column of the predictions passed to
            :func:`~supervision.detection.utils.nms.box_weighted_box_fusion`,
            shape `(n,)`.

    Raises:
        ValueError: If any class id is NaN or infinite.
    """
    invalid = ~np.isfinite(class_ids)
    if invalid.any():
        invalid_rows = np.flatnonzero(invalid)
        raise ValueError(
            "Class ids passed to Weighted Box Fusion must be finite, got "
            f"{class_ids[invalid_rows[:5]].tolist()} at rows "
            f"{invalid_rows[:5].tolist()}. Drop those predictions or assign them "
            "a class id before fusing them."
        )


def _best_matching_cluster(
    fused_boxes: npt.NDArray[np.float64],
    box: npt.NDArray[np.float64],
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    eligible: npt.NDArray[np.bool_] | None = None,
) -> int:
    """Pick the cluster a box should join during Weighted Box Fusion.

    The box is compared with every cluster's current fused box in one batched
    overlap call, and the cluster with the highest overlap wins (the first one
    on ties). The box joins it only when that overlap is strictly greater than
    ``iou_threshold``, as in the WBF paper and its reference implementation.

    Args:
        fused_boxes: Current fused boxes of the existing clusters, shape `(k, 4)`
            in `[x1, y1, x2, y2]` format; `k` may be `0`.
        box: The candidate box, shape `(4,)` in `[x1, y1, x2, y2]` format.
        iou_threshold: Overlap that the best match must strictly exceed.
        overlap_metric: Metric used to compute the overlap.
        eligible: Optional mask of shape `(k,)` of the clusters the box may join.
            It is applied before the best match is picked, so an ineligible
            cluster never hides an eligible runner-up.

    Returns:
        Index of the cluster to join, or `-1` when there is no cluster yet or
            no eligible overlap exceeds `iou_threshold`.
    """
    if len(fused_boxes) == 0:
        return -1
    overlaps = _box_overlap_float64(fused_boxes, box[np.newaxis, :], overlap_metric)[
        :, 0
    ]
    return _pick_cluster(overlaps, iou_threshold, eligible)


def _pick_cluster(
    overlaps: npt.NDArray[np.floating],
    iou_threshold: float,
    eligible: npt.NDArray[np.bool_] | None = None,
) -> int:
    """Pick the best overlapping cluster that a box may join.

    Args:
        overlaps: Overlap of the box with every existing cluster's fused box,
            shape `(k,)` with `k >= 1`.
        iou_threshold: Overlap that the best match must strictly exceed.
        eligible: Optional mask of shape `(k,)` of the clusters the box may join,
            applied before the best match is picked.

    Returns:
        Index of the cluster to join, the first one on ties, or `-1` when no
            eligible overlap exceeds `iou_threshold`.
    """
    if eligible is not None:
        overlaps = np.where(eligible, overlaps, -np.inf)
    best_cluster = int(overlaps.argmax())
    if overlaps[best_cluster] > iou_threshold:
        return best_cluster
    return -1


def _fuse_single_class_boxes(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    sources: npt.NDArray[np.integer] | None = None,
    box_weights: _FusionBoxWeights = "confidence",
    source_factors: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Cluster and fuse the boxes of one class with Weighted Box Fusion.

    Boxes are visited in descending score order; a stable sort keeps equal
    scores in input order. Each box joins the cluster chosen by
    :func:`~supervision.detection.utils.nms._best_matching_cluster` or
    starts a new one. After every join, the cluster's fused box is recomputed
    from running sums, so later boxes are matched against its updated position
    rather than against the cluster's first box. The fused box is the
    confidence-weighted average of the cluster's boxes, or their plain average
    when the cluster's scores sum to zero.

    The running sums use each score divided by the power of two that brings the
    cluster's first, highest score into `[0.5, 1)`, and the mean score is scaled
    back at the end. Sums of finite scores near the float64 limit therefore stay
    finite. When coordinate sums can overflow or scaling can underflow a
    positive score, exact arithmetic accumulates coordinates until division.

    Args:
        boxes: Boxes of a single class, shape `(n, 4)` in `[x1, y1, x2, y2]`
            format, with `n >= 1`.
        scores: Confidence scores aligned with `boxes`, shape `(n,)`.
        iou_threshold: Overlap that a box must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        sources: Optional source id per box, shape `(n,)`. A box may join only a
            cluster that holds no box of its source yet; the check runs before
            the best match is picked, so a full best cluster never hides an
            eligible runner-up.
        box_weights: `"confidence"` averages the member boxes weighted by their
            scores, `"uniform"` gives every member the same weight. The mean score
            is the same in both cases.
        source_factors: Optional weight in `(0, 1]` of the source of each box,
            shape `(n,)`. It multiplies the member's box weight; the returned mean
            score ignores it, and the caller weights the score itself.

    Returns:
        A tuple `(fused_boxes, mean_scores, cluster_sizes, box_clusters)` with
            shapes `(k, 4)`, `(k,)`, `(k,)` and `(n,)`. The first three hold one
            entry per cluster in creation order; `box_clusters[i]` is the index of
            the cluster that input box `i` joined.
    """
    boxes_float = boxes.astype(np.float64)
    scores_float = scores.astype(np.float64)
    box_count = len(boxes_float)
    box_clusters = np.empty(box_count, dtype=np.int64)
    fused_boxes = np.empty((box_count, 4), dtype=np.float64)
    cluster_origins = np.empty((box_count, 4), dtype=np.float64)
    # int32 matches the exponent dtype of np.frexp and the C int that np.ldexp
    # accepts on every platform, including Windows where C long is 32-bit.
    cluster_exponents = np.zeros(box_count, dtype=np.int32)
    weighted_offset_sums = np.zeros((box_count, 4), dtype=np.float64)
    offset_sums = np.zeros((box_count, 4), dtype=np.float64)
    # The score accumulator drives the mean score; the geometry accumulator drives
    # the fused box. They hold the same numbers for confidence weights, so that
    # default stays bit-identical, and differ for uniform weights, where the box
    # weight is 1 and must not change the score.
    weight_sums = np.zeros(box_count, dtype=np.float64)
    geometry_weight_sums = np.zeros(box_count, dtype=np.float64)
    uniform_boxes = box_weights == "uniform"
    cluster_sizes = np.zeros(box_count, dtype=np.int64)
    cluster_count = 0
    if sources is None:
        source_codes = None
        source_in_cluster = None
    else:
        _, source_codes = np.unique(sources, return_inverse=True)
        source_in_cluster = np.zeros((box_count, source_codes.max() + 1), dtype=bool)
    limits = np.finfo(np.float64)
    # A box's weight is its score times its source factor (the factor is 1 without
    # source weights), so the smallest effective weight drives the range check.
    effective_scores = (
        scores_float if source_factors is None else scores_float * source_factors
    )
    smallest_score = np.min(
        effective_scores, where=effective_scores > 0, initial=np.inf
    )
    weight_underflowed = bool(np.any((scores_float > 0) & (effective_scores == 0)))
    # Differences can span twice the largest magnitude, and all offsets can
    # add in the same direction. Scaling can also round a small positive weight
    # before it is multiplied by a large offset. Keep the existing float
    # accumulation for ordinary ranges to preserve rounding at IoU thresholds.
    # Uniform weights equal the source factor, which is 1 without source weights,
    # so only a small factor or the coordinate range can trigger it.
    exact_totals: dict[int, tuple[Fraction, list[Fraction]]] | None = None
    if (
        np.max(np.abs(boxes_float)) > limits.max / (2 * box_count)
        or (
            not uniform_boxes
            and (
                smallest_score <= (2 * limits.tiny) * scores_float.max()
                or weight_underflowed
            )
        )
        or (
            uniform_boxes
            and source_factors is not None
            and source_factors.min() <= 2 * limits.tiny
        )
    ):
        exact_totals = {}

    # WBF is order dependent: every box is matched against fused boxes that the
    # higher-scoring boxes before it have already moved, so the visit itself is
    # sequential; only the overlap against all current clusters is vectorized.
    for box_index in np.argsort(-scores_float, kind="stable"):
        box = boxes_float[box_index]
        score = scores_float[box_index]
        eligible = None
        if source_in_cluster is not None and source_codes is not None:
            eligible = ~source_in_cluster[:cluster_count, source_codes[box_index]]
        cluster = _best_matching_cluster(
            fused_boxes[:cluster_count], box, iou_threshold, overlap_metric, eligible
        )
        if cluster < 0:
            cluster = cluster_count
            cluster_count += 1
            # A new cluster starts exactly at its first box, so a later box
            # whose overlap equals the threshold is not tipped over it by
            # rounding in the weighted average.
            fused_boxes[cluster] = box
            cluster_origins[cluster] = box
            # The first box has the cluster's highest score, so scaling every
            # member by this power of two keeps all weights within [0, 1) and
            # their sums finite even for scores near the float64 limit. A zero
            # score gives exponent 0, leaving all-zero clusters unscaled.
            cluster_exponents[cluster] = np.frexp(score)[1]

        box_clusters[box_index] = cluster
        if source_in_cluster is not None and source_codes is not None:
            source_in_cluster[cluster, source_codes[box_index]] = True
        weight = np.ldexp(score, -cluster_exponents[cluster])
        weight_sums[cluster] += weight
        factor = 1.0 if source_factors is None else float(source_factors[box_index])
        geometry_weight = factor if uniform_boxes else weight * factor
        geometry_weight_sums[cluster] += geometry_weight
        cluster_sizes[cluster] += 1
        if exact_totals is not None:
            # Divide the exact weighted coordinates before converting to
            # float64. A cluster that starts with zero weight can only receive
            # other zero-score boxes because scores are visited in descending order.
            if uniform_boxes:
                exact_weight = Fraction(factor)
            elif geometry_weight_sums[cluster]:
                exact_weight = Fraction(float(score)) * Fraction(factor)
            else:
                exact_weight = Fraction(1)
            total, sums = exact_totals.get(cluster, (Fraction(), [Fraction()] * 4))
            total += exact_weight
            sums = [
                value + exact_weight * Fraction(float(coordinate))
                for value, coordinate in zip(sums, box)
            ]
            exact_totals[cluster] = (total, sums)
            if cluster_sizes[cluster] > 1:
                fused_boxes[cluster] = [float(value / total) for value in sums]
            continue

        # Sum offsets from the cluster's first box instead of absolute
        # coordinates: with large coordinates the absolute sums lose low bits,
        # which would shift a cluster of identical boxes and split it.
        offset = box - cluster_origins[cluster]
        weighted_offset_sums[cluster] += geometry_weight * offset
        offset_sums[cluster] += offset
        if cluster_sizes[cluster] == 1:
            continue

        # Fall back to a plain average when the weights sum to zero (all-zero
        # scores), so the weighted mean stays defined instead of dividing by 0.
        if geometry_weight_sums[cluster] > 0:
            mean_offset = weighted_offset_sums[cluster] / geometry_weight_sums[cluster]
        else:
            mean_offset = offset_sums[cluster] / cluster_sizes[cluster]
        fused_boxes[cluster] = cluster_origins[cluster] + mean_offset

    sizes = cluster_sizes[:cluster_count]
    # Every weight is below 1, so their mean is too, and undoing the scale keeps
    # the score within the float64 range even for the largest finite scores.
    mean_scores = np.ldexp(
        weight_sums[:cluster_count] / sizes, cluster_exponents[:cluster_count]
    )
    return fused_boxes[:cluster_count], mean_scores, sizes, box_clusters


def _box_weighted_fusion_clusters(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    num_models: int | None = None,
    output_dtype: npt.DTypeLike = np.float64,
    sources: npt.NDArray[np.integer] | None = None,
    box_weights: _FusionBoxWeights = "confidence",
    source_weights: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Cluster and fuse boxes with Weighted Box Fusion and report the membership.

    Boxes are fused within their class, or across all of them when `categories` is
    `None`. Fusion runs in float64 so the strict threshold comparison does not
    move; the fused arrays are then cast to `output_dtype` and stably sorted by
    descending fused score, so scores that only become equal after the cast keep
    the deterministic tie order (class id ascending, then cluster creation
    order). `labels` is remapped after that sort, so it indexes the returned
    rows.

    Boxes must be finite and ordered, scores finite and non-negative, and any
    class IDs finite.

    Args:
        boxes: Boxes in `[x1, y1, x2, y2]` format, shape `(n, 4)`.
        scores: Confidence scores aligned with `boxes`, shape `(n,)`.
        categories: Class ids aligned with `boxes`, shape `(n,)`, or `None` to fuse
            across classes.
        iou_threshold: Overlap that a box must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        num_models: When set, each fused score is multiplied by
            `min(cluster_size, num_models) / num_models`.
        output_dtype: Floating dtype of the returned boxes and scores.
        sources: Optional source id per box, shape `(n,)`; a cluster holds at most
            one box per source. With `None` the clustering is unchanged.
        box_weights: `"confidence"` (default) or `"uniform"` weighting of the
            member boxes in the fused box; the fused score is unaffected.
        source_weights: Optional weight of each source id, indexed by id, with one
            weight per model (`num_models` is required). It scales each member's
            weight in the fused box, and the fused score becomes the weighted sum of
            the members' scores over the sum of all source weights.

    Raises:
        ValueError: If the box weights, sources or any input values are invalid.

    Returns:
        A tuple `(fused_boxes, fused_scores, labels)` with shapes `(m, 4)`, `(m,)`
            and `(n,)`, sorted by descending fused score. `labels[i]` is the row,
            in the returned arrays, of the cluster that input box `i` joined.
    """
    if box_weights not in ("confidence", "uniform"):
        raise ValueError(
            f"box_weights must be 'confidence' or 'uniform', got {box_weights!r}."
        )
    box_count = len(boxes)
    sources, weights = _prepare_fusion_sources(
        sources, source_weights, num_models, box_count
    )
    if box_count == 0:
        return (
            np.empty((0, 4), dtype=output_dtype),
            np.empty(0, dtype=output_dtype),
            np.empty(0, dtype=np.int64),
        )
    _validate_fusion_confidences(scores)
    _validate_fusion_boxes(boxes)
    return _fuse_by_class(
        boxes,
        scores,
        categories,
        iou_threshold,
        overlap_metric,
        num_models,
        output_dtype,
        output_dtype,
        sources,
        box_weights,
        _fuse_single_class_boxes,
        weights,
    )[:3]


def _fuse_by_class(
    geometry: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    num_models: int | None,
    score_dtype: npt.DTypeLike,
    geometry_dtype: npt.DTypeLike,
    sources: npt.NDArray[np.integer] | None,
    box_weights: _FusionBoxWeights,
    fuse_single_class: Callable[..., Any],
    source_weights: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.int64], npt.NDArray[np.intp]
]:
    """Run a single-class fusion kernel per class and order the fused rows.

    Shared by the axis-aligned, oriented and mask fusion: `fuse_single_class` fuses
    the rows of one class and returns `(fused_geometry, mean_scores, cluster_sizes,
    box_clusters)`. The fused rows are cast, concatenated and stably sorted by
    descending fused score, and `labels` is remapped to the sorted rows.

    With `source_weights` the kernel receives each row's source weight, scaled so the
    largest is 1, as an extra argument that multiplies the member weight, and the
    fused score of a cluster is the weighted sum of its members' scores divided by
    the sum of all source weights. That replaces the `num_models` rescale, which it
    equals when every weight is 1. A cluster holds at most one row per source, so
    the sum never exceeds the largest member score and stays finite.

    Args:
        geometry: Validated boxes `(n, 4)` or corners `(n, 4, 2)` with `n >= 1`.
        scores: Validated confidence scores, shape `(n,)`.
        categories: Class ids, shape `(n,)`, or `None` to fuse across classes.
        iou_threshold: Overlap that a box must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        num_models: When set, rescales each fused score by
            `min(cluster_size, num_models) / num_models`.
        score_dtype: Floating dtype of the returned scores.
        geometry_dtype: Floating dtype of the returned fused geometry.
        sources: Optional source id per row, shape `(n,)`.
        box_weights: `"confidence"` or `"uniform"` member weighting.
        fuse_single_class: Fusion kernel of one class.
        source_weights: Optional weight per source id, indexed by id; needs `sources`
            and `num_models` equal to its length (see `_prepare_fusion_sources`).

    Returns:
        A tuple `(fused_geometry, fused_scores, labels, order)` sorted by descending
            fused score; `labels[i]` is the returned row of the cluster input row `i`
            joined, and `order[r]` is the position, in class-then-creation order, of
            the cluster that returned row `r` holds, for callers that carry
            per-cluster data of their own.
    """
    box_count = len(geometry)
    if categories is None:
        categories = np.zeros(box_count)
    else:
        _validate_fusion_class_ids(categories)
    unique_class_ids, class_positions = np.unique(categories, return_inverse=True)
    source_factors = None
    factor_total = 1.0
    if source_weights is not None and sources is not None:
        source_factors = source_weights / source_weights.max()
        factor_total = float(source_factors.sum())

    geometry_per_class: list[npt.NDArray[np.float64]] = []
    scores_per_class: list[npt.NDArray[np.float64]] = []
    # Cluster index across all classes, in class-then-creation order, per input row.
    cluster_of_box = np.empty(box_count, dtype=np.int64)
    cluster_offset = 0
    for class_position in range(len(unique_class_ids)):
        in_class = np.flatnonzero(class_positions == class_position)
        class_sources = None if sources is None else sources[in_class]
        class_factors = (
            None
            if source_factors is None or class_sources is None
            else source_factors[class_sources]
        )
        fused_geometry, fused_scores, cluster_sizes, local_clusters = fuse_single_class(
            geometry[in_class],
            scores[in_class],
            iou_threshold,
            overlap_metric,
            class_sources,
            box_weights,
            class_factors,
        )
        if class_factors is not None:
            # Each member contributes `factor / total` of its score, so the sum over
            # a cluster, which holds one member per source, cannot exceed its largest
            # score.
            fused_scores = np.bincount(
                local_clusters,
                weights=scores[in_class].astype(np.float64)
                * class_factors
                / factor_total,
                minlength=len(fused_scores),
            )
        elif num_models is not None:
            # Paper rescale: a cluster backed by fewer boxes than there are
            # models is demoted in proportion to the models that missed it.
            rescale = np.minimum(cluster_sizes, num_models) / num_models
            fused_scores = fused_scores * rescale
        cluster_of_box[in_class] = cluster_offset + local_clusters
        cluster_offset += len(fused_scores)
        geometry_per_class.append(fused_geometry)
        scores_per_class.append(fused_scores)

    # The cast comes before sorting, so scores that only become equal when rounded
    # (e.g. to float32) still follow the tie order.
    cast_geometry = np.concatenate(geometry_per_class).astype(
        geometry_dtype, copy=False
    )
    cast_scores = np.concatenate(scores_per_class).astype(score_dtype, copy=False)
    # A stable sort on the negated score keeps equal scores in matching order
    # (class id ascending, then cluster creation order), so ties come back
    # deterministic instead of reversed or scrambled by an unstable sort.
    descending_score = np.argsort(-cast_scores, kind="stable")
    rank_of_cluster = np.empty(len(descending_score), dtype=np.int64)
    rank_of_cluster[descending_score] = np.arange(len(descending_score))
    return (
        cast_geometry[descending_score],
        cast_scores[descending_score],
        rank_of_cluster[cluster_of_box],
        descending_score,
    )


def _validate_fusion_oriented_boxes(corners: npt.NDArray[np.number]) -> None:
    """Reject oriented boxes with a wrong shape or non-finite corners.

    A NaN corner overlaps nothing, so it would come back unfused as a NaN box, and an
    infinite corner makes the overlap arithmetic NaN.

    Args:
        corners: Oriented boxes as `((x1, y1), (x2, y2), (x3, y3), (x4, y4))`,
            shape `(n, 4, 2)`.

    Raises:
        ValueError: If `corners` does not have shape `(n, 4, 2)`.
        ValueError: If any corner coordinate is NaN or infinite.
    """
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError(
            "Oriented boxes passed to Weighted Box Fusion must have shape "
            f"(n, 4, 2), got {corners.shape}."
        )
    invalid = ~np.isfinite(corners).all(axis=(1, 2))
    if invalid.any():
        invalid_rows = np.flatnonzero(invalid)
        raise ValueError(
            "Oriented boxes passed to Weighted Box Fusion must have finite "
            f"corner coordinates, got {corners[invalid_rows[:5]].tolist()} at rows "
            f"{invalid_rows[:5].tolist()}. Drop those predictions or fix their "
            "corners before fusing them."
        )


def _fuse_single_class_oriented_boxes(
    corners: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    sources: npt.NDArray[np.integer] | None = None,
    box_weights: _FusionBoxWeights = "confidence",
    source_factors: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Cluster and fuse the oriented boxes of one class with Weighted Box Fusion.

    Follows :func:`~supervision.detection.utils.nms._fuse_single_class_boxes`: boxes
    are visited in descending score order, each joins the cluster whose current
    fused box it overlaps most (when above the threshold) or starts a new one, and
    the overlap is measured with
    :func:`~supervision.detection.utils.iou.oriented_box_iou_batch` against the
    running fused polygon.

    Each cluster has the frame of its seed, the first and highest-scoring member
    (see :mod:`supervision.detection.utils._obb_frame`). Every member's corners are
    projected onto the seed's axes, and the fused box is the rectangle whose
    per-axis minima and maxima are the confidence-weighted means of the members'
    minima and maxima, or their plain means when the weights sum to zero, rotated
    back. A cluster of one box keeps its input corners.

    Scores are scaled by the power of two that brings the seed's score into
    `[0.5, 1)`, as in the axis-aligned kernel, so sums of finite scores stay finite.
    Unlike that kernel there is no exact-arithmetic path for coordinates or scores
    near the float64 limits: coordinates are offsets from the seed's first corner,
    so ordinary large coordinates (such as 1e9) are safe.

    Args:
        corners: Oriented boxes of a single class, shape `(n, 4, 2)` with `n >= 1`.
        scores: Confidence scores aligned with `corners`, shape `(n,)`.
        iou_threshold: Overlap that a box must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        sources: Optional source id per box, shape `(n,)`; a cluster holds at most one
            box per source, checked before the best match is picked.
        box_weights: `"confidence"` weights the member extents by their scores,
            `"uniform"` gives every member the same weight; the mean score is the
            same in both cases.
        source_factors: Optional weight in `(0, 1]` of the source of each box, shape
            `(n,)`, that multiplies the member's extent weight. The returned mean
            score ignores it, and the caller weights the score itself.

    Returns:
        A tuple `(fused_corners, mean_scores, cluster_sizes, box_clusters)` with
            shapes `(k, 4, 2)`, `(k,)`, `(k,)` and `(n,)`, in float64 except for the
            integer sizes and clusters. The first three hold one entry per cluster
            in creation order; `box_clusters[i]` is the cluster input box `i` joined.
    """
    corners_float = corners.astype(np.float64)
    # At this scale even float64 spacing reaches one pixel. Match in each
    # seed's translated coordinate system so absolute rounding cannot change IoU.
    use_local_overlap = np.issubdtype(corners.dtype, np.integer) and (
        np.any(corners >= 2**52) or np.any(corners <= -(2**52))
    )
    scores_float = scores.astype(np.float64)
    box_count = len(corners)
    box_clusters = np.empty(box_count, dtype=np.int64)
    fused_corners = np.empty((box_count, 4, 2), dtype=np.float64)
    cluster_origins = np.empty((box_count, 2), dtype=corners.dtype)
    frame_cos = np.zeros(box_count, dtype=np.float64)
    frame_sin = np.zeros(box_count, dtype=np.float64)
    cluster_exponents = np.zeros(box_count, dtype=np.int32)
    weighted_min_sums = np.zeros((box_count, 2), dtype=np.float64)
    weighted_max_sums = np.zeros((box_count, 2), dtype=np.float64)
    min_sums = np.zeros((box_count, 2), dtype=np.float64)
    max_sums = np.zeros((box_count, 2), dtype=np.float64)
    # The score accumulator drives the mean score; the geometry accumulator drives
    # the fused box, so uniform weights cannot change the score.
    weight_sums = np.zeros(box_count, dtype=np.float64)
    geometry_weight_sums = np.zeros(box_count, dtype=np.float64)
    uniform_boxes = box_weights == "uniform"
    cluster_sizes = np.zeros(box_count, dtype=np.int64)
    cluster_count = 0
    zero_origin = np.zeros(2, dtype=np.float64)
    if sources is None:
        source_codes = None
        source_in_cluster = None
    else:
        _, source_codes = np.unique(sources, return_inverse=True)
        source_in_cluster = np.zeros((box_count, source_codes.max() + 1), dtype=bool)

    # Like the axis-aligned kernel, the visit is sequential because every box is
    # matched against fused boxes that earlier boxes have already moved.
    for box_index in np.argsort(-scores_float, kind="stable"):
        box = corners[box_index]
        score = scores_float[box_index]
        cluster = -1
        overlaps: npt.NDArray[np.floating]
        if cluster_count and use_local_overlap:
            candidates = _offsets_from_origin(
                box, cluster_origins[:cluster_count, np.newaxis, :]
            )
            overlaps = np.zeros(cluster_count, dtype=np.float64)
            running = fused_corners[:cluster_count]
            # Only compare pairs whose local envelopes have positive overlap.
            possible = np.all(
                np.minimum(running.max(axis=1), candidates.max(axis=1))
                > np.maximum(running.min(axis=1), candidates.min(axis=1)),
                axis=1,
            )
            for index in np.flatnonzero(possible):
                overlaps[index] = oriented_box_iou_batch(
                    running[index : index + 1],
                    candidates[index : index + 1],
                    overlap_metric,
                )[0, 0]
        elif cluster_count:
            overlaps = oriented_box_iou_batch(
                fused_corners[:cluster_count],
                corners_float[box_index][np.newaxis],
                overlap_metric,
            )[:, 0]
        if cluster_count:
            eligible = None
            if source_in_cluster is not None and source_codes is not None:
                eligible = ~source_in_cluster[:cluster_count, source_codes[box_index]]
            cluster = _pick_cluster(overlaps, iou_threshold, eligible)
        if cluster < 0:
            cluster = cluster_count
            cluster_count += 1
            # A new cluster starts exactly at its seed's corners, so an unfused box
            # comes back value-identical and a later box whose overlap equals the
            # threshold is not tipped over it by rounding.
            cluster_origins[cluster] = box[0]
            if use_local_overlap:
                fused_corners[cluster] = _offsets_from_origin(box, box[0])
            else:
                fused_corners[cluster] = corners_float[box_index]
            frame_cos[cluster], frame_sin[cluster] = seed_frame(box)
            cluster_exponents[cluster] = np.frexp(score)[1]

        box_clusters[box_index] = cluster
        if source_in_cluster is not None and source_codes is not None:
            source_in_cluster[cluster, source_codes[box_index]] = True
        weight = np.ldexp(score, -cluster_exponents[cluster])
        weight_sums[cluster] += weight
        factor = 1.0 if source_factors is None else float(source_factors[box_index])
        geometry_weight = factor if uniform_boxes else weight * factor
        geometry_weight_sums[cluster] += geometry_weight
        cluster_sizes[cluster] += 1

        local = corners_to_frame(
            box, cluster_origins[cluster], frame_cos[cluster], frame_sin[cluster]
        )
        lowest, highest = local.min(axis=0), local.max(axis=0)
        weighted_min_sums[cluster] += geometry_weight * lowest
        weighted_max_sums[cluster] += geometry_weight * highest
        min_sums[cluster] += lowest
        max_sums[cluster] += highest
        if cluster_sizes[cluster] == 1:
            continue

        # Fall back to a plain average when the weights sum to zero (all-zero
        # scores), so the weighted mean stays defined instead of dividing by 0.
        if geometry_weight_sums[cluster] > 0:
            mean_lowest = weighted_min_sums[cluster] / geometry_weight_sums[cluster]
            mean_highest = weighted_max_sums[cluster] / geometry_weight_sums[cluster]
        else:
            mean_lowest = min_sums[cluster] / cluster_sizes[cluster]
            mean_highest = max_sums[cluster] / cluster_sizes[cluster]
        rectangle = np.array(
            [
                [mean_lowest[0], mean_lowest[1]],
                [mean_highest[0], mean_lowest[1]],
                [mean_highest[0], mean_highest[1]],
                [mean_lowest[0], mean_highest[1]],
            ]
        )
        origin = zero_origin if use_local_overlap else cluster_origins[cluster]
        fused_corners[cluster] = frame_to_corners(
            rectangle, origin, frame_cos[cluster], frame_sin[cluster]
        )

    sizes = cluster_sizes[:cluster_count]
    mean_scores = np.ldexp(
        weight_sums[:cluster_count] / sizes, cluster_exponents[:cluster_count]
    )
    output = fused_corners[:cluster_count]
    if use_local_overlap:
        origins = cluster_origins[:cluster_count, np.newaxis, :]
        # Multiples of 2048 fit exactly in float64 throughout the 64-bit integer
        # range. Add the low bits to the local offsets before the final rounding.
        low_bits = origins % 2048
        output += low_bits
        output += origins - low_bits
        # Use the original cast for singletons, avoiding a second rounding.
        singleton_rows = sizes[box_clusters] == 1
        output[box_clusters[singleton_rows]] = corners_float[singleton_rows]
    return output, mean_scores, sizes, box_clusters


def _oriented_box_weighted_fusion_clusters(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    num_models: int | None = None,
    output_dtype: npt.DTypeLike = np.float64,
    sources: npt.NDArray[np.integer] | None = None,
    box_weights: _FusionBoxWeights = "confidence",
    source_weights: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Cluster and fuse oriented boxes with Weighted Box Fusion and report membership.

    The oriented counterpart of
    :func:`~supervision.detection.utils.nms._box_weighted_fusion_clusters`: the same
    per-class clustering, `num_models` rescale and stable descending-score order,
    with each cluster fused in the frame of its seed.

    Args:
        boxes: Oriented boxes as corner arrays, shape `(n, 4, 2)`.
        scores: Confidence scores aligned with `boxes`, shape `(n,)`.
        categories: Class ids aligned with `boxes`, shape `(n,)`, or `None` to fuse
            across classes.
        iou_threshold: Overlap that a box must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        num_models: When set, each fused score is multiplied by
            `min(cluster_size, num_models) / num_models`.
        output_dtype: Floating dtype of the returned scores. The fused corners are
            always float64.
        sources: Optional source id per box, shape `(n,)`; a cluster holds at most
            one box per source.
        box_weights: `"confidence"` (default) or `"uniform"` weighting of the
            member extents in the fused box; the fused score is unaffected.
        source_weights: Optional weight of each source id, indexed by id, with one
            weight per model (`num_models` is required); see
            :func:`~supervision.detection.utils.nms._box_weighted_fusion_clusters`.

    Raises:
        ValueError: If the box weights, sources or any input values are invalid.

    Returns:
        A tuple `(fused_corners, fused_scores, labels)` with shapes `(m, 4, 2)`,
            `(m,)` and `(n,)`, sorted by descending fused score. A cluster of one
            box has its input corners, as float64. `labels[i]` is the row, in the
            returned arrays, of the cluster that input box `i` joined.
    """
    if box_weights not in ("confidence", "uniform"):
        raise ValueError(
            f"box_weights must be 'confidence' or 'uniform', got {box_weights!r}."
        )
    sources, weights = _prepare_fusion_sources(
        sources, source_weights, num_models, len(boxes)
    )
    if len(boxes) == 0:
        return (
            np.empty((0, 4, 2), dtype=np.float64),
            np.empty(0, dtype=output_dtype),
            np.empty(0, dtype=np.int64),
        )
    _validate_fusion_confidences(scores)
    _validate_fusion_oriented_boxes(boxes)
    return _fuse_by_class(
        boxes,
        scores,
        categories,
        iou_threshold,
        overlap_metric,
        num_models,
        output_dtype,
        np.float64,
        sources,
        box_weights,
        _fuse_single_class_oriented_boxes,
        weights,
    )[:3]


def _box_weighted_box_fusion(
    predictions: npt.NDArray[np.floating],
    iou_threshold: float = 0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
    num_models: int | None = None,
) -> npt.NDArray[np.floating]:
    """Fuse `(n, 5)` or `(n, 6)` prediction rows with Weighted Box Fusion.

    Validates the arguments and the prediction rows, runs
    :func:`~supervision.detection.utils.nms._box_weighted_fusion_clusters`
    and returns the fused rows in the input's column layout. The public
    :func:`~supervision.detection.utils.nms.box_weighted_box_fusion`
    documents the behaviour.

    Args:
        predictions: Rows `[x1, y1, x2, y2, confidence]` with an optional class id
            column.
        iou_threshold: Overlap that a box must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        num_models: Number of models whose predictions were concatenated, or
            `None` to keep the plain mean confidence.

    Returns:
        Fused rows, ordered by descending fused confidence, of dtype
            `numpy.result_type(predictions.dtype, numpy.float32)`.
    """
    _validate_iou_threshold(iou_threshold)
    _validate_num_models(num_models)
    # Resolve the metric before any early return: with empty input or one box
    # per class no overlap is ever computed, so a bad value would pass silently.
    overlap_metric = OverlapMetric.from_value(overlap_metric)
    _validate_fusion_predictions_shape(predictions)
    # One output dtype for the empty and the fused path: float inputs keep their
    # precision, while integer inputs become floating point because fused
    # coordinates and scores are averages.
    output_dtype = np.result_type(predictions.dtype, np.float32)
    if len(predictions) == 0:
        empty_result: npt.NDArray[np.floating] = predictions.astype(output_dtype)
        return empty_result
    has_class_column = predictions.shape[1] > 5
    class_ids = predictions[:, 5] if has_class_column else None
    fused_boxes, fused_scores, labels = _box_weighted_fusion_clusters(
        predictions[:, :4],
        predictions[:, 4],
        class_ids,
        iou_threshold,
        overlap_metric,
        num_models,
        output_dtype,
    )
    columns = [fused_boxes, fused_scores[:, np.newaxis]]
    if class_ids is not None:
        # Every member of a cluster shares its class, so any member names it.
        fused_class_ids = np.empty(len(fused_scores), dtype=np.float64)
        fused_class_ids[labels] = class_ids
        columns.append(fused_class_ids.astype(output_dtype)[:, np.newaxis])
    result: npt.NDArray[np.floating] = np.hstack(columns)
    return result


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
