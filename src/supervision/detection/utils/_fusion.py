"""Weighted Box Fusion kernels for axis-aligned and oriented boxes.

The array-level kernels behind
:meth:`~supervision.detection.core.Detections.with_fusion`: clustering against a
running fused geometry, the shared per-class driver, and the validation of sources
and source weights. The mask kernel, which reuses the driver, lives in
``_mask_fusion``.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from supervision.detection.utils._obb_frame import (
    _offsets_from_origin,
    corners_to_frame,
    frame_to_corners,
    seed_frame,
)
from supervision.detection.utils.iou import (
    OverlapMetric,
    _validate_iou_threshold,
    oriented_box_iou_batch,
)

#: Weighting of the member boxes in the Weighted Box Fusion mean: by confidence
#: (the paper's rule) or equally.
_FusionBoxWeights = Literal["confidence", "uniform"]

#: Largest ratio between the biggest and the smallest source weight. Beyond it a
#: weighted confidence can underflow, so the weights are rejected.
_MAX_SOURCE_WEIGHT_RATIO = 2.0**100


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


def _box_overlap_one_to_many(
    boxes: npt.NDArray[np.float64],
    box: npt.NDArray[np.float64],
    overlap_metric: OverlapMetric,
) -> npt.NDArray[np.float64]:
    """Return the overlap of one box with every row of `boxes`.

    Equal to `_box_overlap_float64(boxes, box[np.newaxis, :], overlap_metric)[:, 0]`,
    bit for bit, for float64 input: it runs the same float64 operations in the same
    order. It is the one-against-many form that Weighted Box Fusion calls once per
    visited box, so it skips the general function's per-call setup (metric
    parsing, dtype checks, transposes and temporary allocations), which costs more
    than the arithmetic when the number of clusters is small.

    Args:
        boxes: Boxes in `[x1, y1, x2, y2]` format, shape `(k, 4)` and float64;
            `k` may be `0`.
        box: The candidate box, shape `(4,)` and float64.
        overlap_metric: `OverlapMetric.IOU` or `OverlapMetric.IOS`.

    Returns:
        The overlaps, shape `(k,)`; zero where the normalising area is not
            positive.

    Raises:
        ValueError: If `overlap_metric` is neither IOU nor IOS.
    """
    x_min, y_min, x_max, y_max = boxes.T
    box_x_min, box_y_min, box_x_max, box_y_max = box
    # Intersection extents, clipped at zero. `np.maximum(..., 0.0)` returns what
    # `np.clip(..., 0.0, None)` does and avoids its Python-level wrapper.
    width = np.minimum(x_max, box_x_max)
    width -= np.maximum(x_min, box_x_min)
    np.maximum(width, 0.0, out=width)
    height = np.minimum(y_max, box_y_max)
    height -= np.maximum(y_min, box_y_min)
    np.maximum(height, 0.0, out=height)
    area_inter = width * height
    area_true = (x_max - x_min) * (y_max - y_min)
    area_det = (box_x_max - box_x_min) * (box_y_max - box_y_min)
    if overlap_metric == OverlapMetric.IOU:
        # Same order as the general function: (a + b) - intersection.
        area_norm = area_true + area_det
        area_norm -= area_inter
    elif overlap_metric == OverlapMetric.IOS:
        area_norm = np.minimum(area_true, area_det)
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )
    overlaps: npt.NDArray[np.float64] = np.zeros_like(area_inter)
    np.divide(area_inter, area_norm, out=overlaps, where=area_norm > 0)
    return overlaps


def _best_matching_cluster(
    fused_boxes: npt.NDArray[np.float64],
    box: npt.NDArray[np.float64],
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    eligible: npt.NDArray[np.bool_] | None = None,
) -> int:
    """Pick the cluster a box should join during Weighted Box Fusion.

    The box is compared with every cluster's current fused box in one batched
    overlap computation, and the cluster with the highest overlap wins (the first
    one on ties). The box joins it only when that overlap is strictly greater than
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
    overlaps = _box_overlap_one_to_many(fused_boxes, box, overlap_metric)
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
    :func:`~supervision.detection.utils._fusion._best_matching_cluster` or
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
        factor = 1.0
        # The seed has the highest score. A zero total means the entire cluster
        # votes equally, even in uniform mode with unequal source weights.
        if source_factors is not None and weight_sums[cluster] > 0:
            factor = float(source_factors[box_index])
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
        if (
            class_factors is not None
            and source_weights is not None
            and class_sources is not None
        ):
            class_scores = scores[in_class].astype(np.float64)
            contributions = class_scores * (class_factors / factor_total)
            limits = np.finfo(np.float64)
            if class_scores.max() > limits.max / 2 or np.any(
                (class_scores > 0) & (contributions <= limits.tiny)
            ):
                # Rounded terms can overflow in the sum or vanish individually.
                # Accumulate the original dyadic weights and round once per cluster.
                exact_weights = [Fraction(float(weight)) for weight in source_weights]
                total_weight = sum(exact_weights, Fraction())
                totals = [Fraction() for _ in fused_scores]
                for cluster, score, source in zip(
                    local_clusters, class_scores, class_sources
                ):
                    totals[cluster] += Fraction(float(score)) * exact_weights[source]
                fused_scores = np.array(
                    [float(total / total_weight) for total in totals]
                )
            else:
                fused_scores = np.bincount(
                    local_clusters, weights=contributions, minlength=len(fused_scores)
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

    Follows :func:`~supervision.detection.utils._fusion._fuse_single_class_boxes`: boxes
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
        factor = 1.0
        # The seed has the highest score. A zero total means the entire cluster
        # votes equally, even in uniform mode with unequal source weights.
        if source_factors is not None and weight_sums[cluster] > 0:
            factor = float(source_factors[box_index])
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
    :func:`~supervision.detection.utils._fusion._box_weighted_fusion_clusters`: the same
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
            :func:`~supervision.detection.utils._fusion._box_weighted_fusion_clusters`.

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
    :func:`~supervision.detection.utils._fusion._box_weighted_fusion_clusters`
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
