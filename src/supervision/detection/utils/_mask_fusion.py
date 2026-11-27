"""Fuse detection masks using confidence-weighted votes.

Vote grids are allocated only for merged clusters and span the union of their members'
foreground bounds. This workspace can reach image size for large clusters. Compact
singleton masks stay encoded between comparisons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision.detection.compact_mask import CompactMask, _rle_to_column_intervals
from supervision.detection.utils.converters import _mask_to_rle_counts
from supervision.detection.utils.iou import OverlapMetric, _overlap_from_areas
from supervision.detection.utils.nms import (
    _fuse_by_class,
    _FusionBoxWeights,
    _pick_cluster,
    _prepare_fusion_sources,
    _validate_fusion_boxes,
    _validate_fusion_confidences,
)

_MAX_INT64 = int(np.iinfo(np.int64).max)
_VOTE_BATCH_SIZE = 4096


def _binary_weight(weight: float) -> tuple[int, int]:
    """Express a nonnegative float as an odd mantissa times a power of two."""
    numerator, denominator = weight.as_integer_ratio()
    if numerator == 0:
        return 0, 0
    trailing_zeros = (numerator & -numerator).bit_length() - 1
    return numerator >> trailing_zeros, trailing_zeros - (denominator.bit_length() - 1)


def _compact_runs(
    masks: CompactMask, row: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Express foreground intervals as half-open runs in the image's column order."""
    columns, starts, stops = _rle_to_column_intervals(
        masks._rles[row], int(masks._crop_shapes[row, 0]), masks.offsets[row]
    ).T
    origins = columns * masks.image_shape[0]
    return origins + starts, origins + stops


def _run_intersection(
    candidate: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    seed: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> int:
    """Count shared foreground pixels without decoding either compact crop."""
    starts, stops = seed
    if len(starts) == 0:
        return 0
    lengths = stops - starts
    covered_before = np.cumsum(lengths) - lengths

    def covered_before_pixel(pixels: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Count foreground pixels strictly before each requested position."""
        indices = np.maximum(np.searchsorted(starts, pixels, side="right") - 1, 0)
        return cast(
            npt.NDArray[np.int64],
            covered_before[indices]
            + np.clip(pixels - starts[indices], 0, lengths[indices]),
        )

    return int(
        (covered_before_pixel(candidate[1]) - covered_before_pixel(candidate[0])).sum()
    )


def _wide_mask_vote(
    masks: npt.NDArray[Any] | CompactMask,
    members: list[tuple[int, float]],
    box: list[int],
) -> npt.NDArray[np.bool_]:
    """Replay unusually wide exact votes in batches with bounded integer storage."""
    weights = [_binary_weight(weight) for _, weight in members]
    exponent = min(exp for mantissa, exp in weights if mantissa)
    votes = [
        mantissa << (exp - exponent) if mantissa else 0 for mantissa, exp in weights
    ]
    cutoff = (sum(votes) + 1) // 2
    left, top, right, bottom = box
    width = right - left
    fused = np.empty((bottom - top, width), dtype=bool)
    flat = fused.ravel()
    compact_runs = []
    if isinstance(masks, CompactMask):
        for row, _ in members:
            starts, stops = _compact_runs(masks, row)
            compact_runs.append((np.append(starts, 0), stops))
    for start in range(0, len(flat), _VOTE_BATCH_SIZE):
        positions = np.arange(start, min(len(flat), start + _VOTE_BATCH_SIZE))
        ys, xs = top + positions // width, left + positions % width
        batch = np.zeros(len(positions), dtype=object)
        for i, (row, _) in enumerate(members):
            if isinstance(masks, CompactMask):
                starts, stops = compact_runs[i]
                pixels = xs * masks.image_shape[0] + ys
                indices = np.searchsorted(stops, pixels, side="right")
                foreground = (indices < len(stops)) & (pixels >= starts[indices])
            else:
                foreground = masks[row, ys, xs].astype(bool, copy=False)
            np.add(batch, votes[i], out=batch, where=foreground)
        flat[positions] = batch >= cutoff
    return fused


@dataclass(frozen=True)
class _FusedMask:
    """A fused mask: a boolean crop and the image position of its top-left pixel.

    Attributes:
        x: Column of the crop's left edge in the image.
        y: Row of the crop's top edge in the image.
        crop: Boolean array of shape `(height, width)`.
    """

    x: int
    y: int
    crop: npt.NDArray[np.bool_]


@dataclass
class _MaskCluster:
    """Running state of one fused cluster of masks.

    Attributes:
        box: `[x_min, y_min, x_max, y_max]` of the union of the members' crops, with
            exclusive maximum.
        votes: Fixed-width integer votes per pixel, or None for wide replay.
        total: Sum of the members' integer vote weights.
        exponent: Common binary exponent of the vote weights.
        seed_crop: Crop of the seed, the first and highest-confidence member.
        seed_xy: Image position of `seed_crop`.
        fused: The cluster's current fused mask over `box`.
        area: Pixel count of `fused`.
    """

    box: list[int]
    votes: npt.NDArray[np.int64] | None
    total: int
    exponent: int
    seed_crop: npt.NDArray[np.bool_]
    seed_xy: tuple[int, int]
    fused: npt.NDArray[np.bool_]
    area: int

    @classmethod
    def start(
        cls, crop: npt.NDArray[np.bool_], x: int, y: int, weight: float
    ) -> _MaskCluster:
        """Start a cluster whose fused mask is exactly its seed's mask."""
        height, width = crop.shape
        mantissa, exponent = _binary_weight(weight)
        return cls(
            box=[x, y, x + width, y + height],
            votes=mantissa * crop.astype(np.int64),
            total=mantissa,
            exponent=exponent,
            seed_crop=crop,
            seed_xy=(x, y),
            fused=crop,
            area=int(np.count_nonzero(crop)),
        )

    def add(
        self,
        crop: npt.NDArray[np.bool_],
        x: int,
        y: int,
        weight: float,
        wide_vote: Callable[[list[int]], npt.NDArray[np.bool_]] | None = None,
    ) -> None:
        """Add a member and recompute the fused mask.

        A pixel is kept when its votes are at least half the total weight, so ties
        keep the pixel. If no pixel survives, the seed's mask is used instead.

        Args:
            crop: The member's boolean crop.
            x: Column of the crop's left edge in the image.
            y: Row of the crop's top edge in the image.
            weight: The member's vote weight.
            wide_vote: Replays all members when their exact weights exceed int64.
        """
        mantissa, exponent = _binary_weight(weight)
        if mantissa == 0:
            return
        common_exponent = min(self.exponent, exponent)
        shift = self.exponent - common_exponent
        vote = mantissa << (exponent - common_exponent)
        total = (self.total << shift) + vote
        height, width = crop.shape
        left, top = min(self.box[0], x), min(self.box[1], y)
        right = max(self.box[2], x + width)
        bottom = max(self.box[3], y + height)
        bounds = [left, top, right, bottom]
        if self.votes is None or total > _MAX_INT64:
            # Avoid retaining Python integers per pixel for wide score ranges.
            assert wide_vote is not None
            self.votes = None
            self.fused = wide_vote(bounds)
        else:
            if shift and self.total:
                self.votes *= 1 << shift
            if bounds != self.box:
                grown = np.zeros((bottom - top, right - left), dtype=np.int64)
                old_top, old_left = self.box[1] - top, self.box[0] - left
                old_height = self.box[3] - self.box[1]
                old_width = self.box[2] - self.box[0]
                grown[
                    old_top : old_top + old_height, old_left : old_left + old_width
                ] = self.votes
                self.votes = grown
            view = self.votes[y - top : y - top + height, x - left : x - left + width]
            np.add(view, vote, out=view, where=crop)
            self.fused = self.votes >= (total + 1) // 2
        self.total = total
        self.exponent = common_exponent
        self.box = bounds
        if not self.fused.any():
            seed_height, seed_width = self.seed_crop.shape
            seed_x, seed_y = self.seed_xy
            self.fused = np.zeros_like(self.fused)
            self.fused[
                seed_y - top : seed_y - top + seed_height,
                seed_x - left : seed_x - left + seed_width,
            ] = self.seed_crop
        self.area = int(np.count_nonzero(self.fused))


def _mask_crop(
    masks: npt.NDArray[Any] | CompactMask, index: int
) -> tuple[npt.NDArray[np.bool_], int, int]:
    """Return the tight boolean crop of one mask and the image position of its corner.

    Tight bounds exclude false padding, and an empty mask gives a `0 x 0` crop.

    Args:
        masks: Dense `(N, H, W)` masks or a `CompactMask`.
        index: Row of the mask.

    Returns:
        A tuple `(crop, x, y)` with the crop and its top-left corner.
    """
    compact = isinstance(masks, CompactMask)
    if isinstance(masks, CompactMask):
        x, y = (int(value) for value in masks.offsets[index])
        mask = masks.crop(index)
    else:
        x = y = 0
        mask = np.asarray(masks[index], dtype=np.bool_)
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.bool_), 0, 0
    columns = np.flatnonzero(mask.any(axis=0))
    crop = mask[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    if compact and crop.shape != mask.shape:
        # A merged cluster may retain the seed. Release the decoded padding.
        crop = crop.copy()
    return crop, x + int(columns[0]), y + int(rows[0])


def _overlap_with_clusters(
    clusters: list[_MaskCluster | None],
    cluster_boxes: npt.NDArray[np.int64],
    seed_rows: npt.NDArray[np.int64],
    seed_areas: npt.NDArray[np.int64],
    masks: npt.NDArray[Any] | CompactMask,
    row: int,
    crop: npt.NDArray[np.bool_],
    x: int,
    y: int,
    overlap_metric: OverlapMetric,
) -> npt.NDArray[np.float64]:
    """Measure the overlap of a mask with each cluster's current fused mask.

    Only the clusters whose bounds intersect the mask are compared, and each
    comparison reads just the intersection of the two crops.

    Args:
        clusters: Merged clusters, with None for singletons.
        cluster_boxes: Bounds of the clusters, shape `(k, 4)`.
        seed_rows: Source rows for singleton representatives.
        seed_areas: Pixel counts of the seed masks.
        masks: Source masks used to read singleton representatives.
        row: Source row of the candidate mask.
        crop: The candidate's boolean crop.
        x: Column of the crop's left edge in the image.
        y: Row of the crop's top edge in the image.
        overlap_metric: Metric used to compute the overlap.

    Returns:
        The overlap with every cluster, shape `(k,)`; zero without intersection.
    """
    height, width = crop.shape
    overlaps = np.zeros(len(clusters), dtype=np.float64)
    if height == 0 or width == 0:
        return overlaps
    intersects = (
        (cluster_boxes[:, 0] < x + width)
        & (cluster_boxes[:, 2] > x)
        & (cluster_boxes[:, 1] < y + height)
        & (cluster_boxes[:, 3] > y)
    )
    candidate_area = np.array([float(np.count_nonzero(crop))])
    candidate_runs = None
    for index in np.flatnonzero(intersects):
        cluster = clusters[index]
        cx, cy, cx2, cy2 = cluster_boxes[index]
        left, top = max(cx, x), max(cy, y)
        right, bottom = min(cx2, x + width), min(cy2, y + height)
        area = int(seed_areas[index]) if cluster is None else cluster.area
        member = crop[top - y : bottom - y, left - x : right - x]
        if area == (cx2 - cx) * (cy2 - cy):
            pixels = int(np.count_nonzero(member))
        elif cluster is None and isinstance(masks, CompactMask):
            if candidate_runs is None:
                candidate_runs = _compact_runs(masks, row)
            pixels = _run_intersection(
                candidate_runs, _compact_runs(masks, int(seed_rows[index]))
            )
        else:
            if cluster is None:
                seed_row = int(seed_rows[index])
                dense_masks = cast(npt.NDArray[Any], masks)
                fused = dense_masks[seed_row, top:bottom, left:right].astype(
                    bool, copy=False
                )
            else:
                fused = cluster.fused[top - cy : bottom - cy, left - cx : right - cx]
            pixels = int(np.count_nonzero(fused & member))
        intersection = np.array([[float(pixels)]])
        overlaps[index] = _overlap_from_areas(
            intersection,
            np.array([float(area)]),
            candidate_area,
            overlap_metric,
        )[0, 0]
    return overlaps


def _fuse_single_class_masks(
    masks: npt.NDArray[Any] | CompactMask,
    boxes: npt.NDArray[np.float64],
    row_ids: npt.NDArray[np.int64],
    scores: npt.NDArray[np.number],
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    sources: npt.NDArray[np.integer] | None,
    box_weights: _FusionBoxWeights,
    fused_masks: list[_FusedMask | None],
    source_factors: npt.NDArray[np.float64] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Cluster and fuse the masks of one class with Weighted Box Fusion.

    Masks are visited in descending score order; a stable sort keeps equal scores in
    input order. Each mask joins the cluster whose current fused mask it overlaps
    most, when that overlap strictly exceeds the threshold, or starts a new one.
    Scores are normalized for the mean confidence and the ordinary box mean.
    Pixel votes use the original scores. An all-zero cluster or
    `box_weights="uniform"` gives members equal weight in the vote and box mean.
    With `source_factors` every member's vote and box weight is multiplied by the
    factor of its source, except in an all-zero cluster, whose members stay equal.

    Args:
        masks: Dense `(N, H, W)` masks or a `CompactMask`, for all classes.
        boxes: `xyxy` boxes of all rows, shape `(N, 4)`.
        row_ids: Rows of `masks` and `boxes` that belong to this class.
        scores: Confidence scores of those rows.
        iou_threshold: Overlap that a mask must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        sources: Optional source id of each of those rows; a cluster holds at most
            one mask per source.
        box_weights: `"confidence"` or `"uniform"` member weighting.
        fused_masks: Output list that receives, per cluster in creation order, its
            fused mask, or `None` for a cluster of one mask.
        source_factors: Optional weight in `(0, 1]` of the source of each of those
            rows. The returned mean score ignores it, and the caller weights the
            score itself.

    Returns:
        A tuple `(fused_xyxy, mean_scores, cluster_sizes, mask_clusters)` with shapes
            `(k, 4)`, `(k,)`, `(k,)` and `(n,)`, where `mask_clusters[i]` is the
            cluster that the `i`-th row of this class joined.
    """
    scores_float = scores.astype(np.float64)
    count = len(row_ids)
    visit_order = np.argsort(-scores_float, kind="stable")
    if iou_threshold == 1.0:
        # Strict overlap cannot beat one. Preserve the usual singleton order
        # and skip materializing masks, including full-frame compact crops.
        labels = np.empty(count, dtype=np.int64)
        labels[visit_order] = np.arange(count)
        fused_masks.extend([None] * count)
        return (
            boxes[row_ids[visit_order]],
            scores_float[visit_order],
            np.ones(count, dtype=np.int64),
            labels,
        )

    clusters: list[_MaskCluster | None] = []
    cluster_members: list[list[tuple[int, float]]] = []
    cluster_boxes = np.empty((count, 4), dtype=np.int64)
    seed_rows = np.empty(count, dtype=np.int64)
    seed_areas = np.empty(count, dtype=np.int64)
    mask_clusters = np.empty(count, dtype=np.int64)
    origins = np.empty((count, 4), dtype=np.float64)
    exponents = np.zeros(count, dtype=np.int32)
    unit_weights = np.zeros(count, dtype=bool)
    weight_sums = np.zeros(count, dtype=np.float64)
    weighted_offsets = np.zeros((count, 4), dtype=np.float64)
    geometry_weight_sums = np.zeros(count, dtype=np.float64)
    sizes = np.zeros(count, dtype=np.int64)
    # Whether a cluster's members take their source factor: an all-zero cluster
    # votes equally and ignores it.
    applies_factor = np.zeros(count, dtype=bool)
    uniform = box_weights == "uniform"
    limits = np.finfo(np.float64)
    effective_scores = (
        scores_float if source_factors is None else scores_float * source_factors
    )
    smallest_score = np.min(
        effective_scores, where=effective_scores > 0, initial=np.inf
    )
    weight_underflowed = bool(np.any((scores_float > 0) & (effective_scores == 0)))
    coordinates = np.abs(boxes[row_ids])
    smallest_coordinate = np.min(coordinates, where=coordinates > 0, initial=np.inf)
    # The offset sum can overflow or lose a weight before multiplication.
    # Use the same conservative range guard as axis-aligned box fusion.
    exact_boxes = bool(
        np.max(coordinates) > limits.max / (2 * count)
        or smallest_coordinate <= 2 * limits.tiny
        or (
            not uniform
            and (
                smallest_score <= (2 * limits.tiny) * scores_float.max()
                or weight_underflowed
            )
        )
        or (
            uniform
            and source_factors is not None
            and source_factors.min() <= 2 * limits.tiny
        )
    )

    def member_factor(position: int, cluster: int) -> float:
        """Return the source factor of a member, 1 when its cluster ignores it."""
        if source_factors is None or not applies_factor[cluster]:
            return 1.0
        return float(source_factors[position])

    if sources is None:
        source_codes = None
        source_in_cluster = None
    else:
        _, source_codes = np.unique(sources, return_inverse=True)
        source_in_cluster = np.zeros((count, source_codes.max() + 1), dtype=bool)

    for position in visit_order:
        row = int(row_ids[position])
        crop, x, y = _mask_crop(masks, row)
        score = scores_float[position]
        cluster = -1
        factor = 1.0
        if clusters:
            eligible = None
            if source_in_cluster is not None and source_codes is not None:
                eligible = ~source_in_cluster[: len(clusters), source_codes[position]]
            overlaps = _overlap_with_clusters(
                clusters,
                cluster_boxes[: len(clusters)],
                seed_rows,
                seed_areas,
                masks,
                row,
                crop,
                x,
                y,
                overlap_metric,
            )
            cluster = _pick_cluster(overlaps, iou_threshold, eligible)
        if cluster < 0:
            cluster = len(clusters)
            origins[cluster] = boxes[row_ids[position]]
            exponents[cluster] = np.frexp(score)[1]
            # The seed has the highest score, so a zero seed weight means every
            # member of the cluster has none and the vote becomes unweighted.
            unit_weights[cluster] = uniform or score == 0
            applies_factor[cluster] = uniform or score > 0
            factor = member_factor(position, cluster)
            # Keep a reference to the input row instead of a decoded mask or
            # vote grid. A singleton's pixels are only needed where bboxes meet.
            height, width = crop.shape
            cluster_boxes[cluster] = [x, y, x + width, y + height]
            seed_rows[cluster] = row
            seed_areas[cluster] = np.count_nonzero(crop)
            clusters.append(None)
            cluster_members.append(
                [(row, (1.0 if unit_weights[cluster] else float(score)) * factor)]
            )
        else:
            members = cluster_members[cluster]
            factor = member_factor(position, cluster)
            vote_weight = (1.0 if unit_weights[cluster] else float(score)) * factor
            members.append((row, vote_weight))
            merged = clusters[cluster]
            if merged is None:
                seed_row, seed_weight = members[0]
                seed, sx, sy = _mask_crop(masks, seed_row)
                merged = _MaskCluster.start(seed, sx, sy, seed_weight)
                clusters[cluster] = merged
            merged.add(
                crop, x, y, vote_weight, partial(_wide_mask_vote, masks, members)
            )
            cluster_boxes[cluster] = merged.box
        weight = float(np.ldexp(score, -exponents[cluster]))
        mask_clusters[position] = cluster
        if source_in_cluster is not None and source_codes is not None:
            source_in_cluster[cluster, source_codes[position]] = True
        weight_sums[cluster] += weight
        sizes[cluster] += 1
        if not exact_boxes:
            geometry_weight = (1.0 if unit_weights[cluster] else weight) * factor
            geometry_weight_sums[cluster] += geometry_weight
            weighted_offsets[cluster] += geometry_weight * (
                boxes[row] - origins[cluster]
            )

    cluster_count = len(clusters)
    if exact_boxes:
        fused_xyxy = origins[:cluster_count].copy()
        by_cluster = np.argsort(mask_clusters, kind="stable")
        cluster_ends = np.cumsum(sizes[:cluster_count])
        for c in np.flatnonzero(sizes[:cluster_count] > 1):
            member_positions = by_cluster[cluster_ends[c] - sizes[c] : cluster_ends[c]]
            weights = [
                (Fraction(1) if unit_weights[c] else Fraction(float(scores_float[i])))
                * Fraction(member_factor(int(i), int(c)))
                for i in member_positions
            ]
            total = sum(weights)
            for column in range(4):
                coordinates = boxes[row_ids[member_positions], column]
                weighted_sum = sum(
                    weight * Fraction(float(value))
                    for weight, value in zip(weights, coordinates)
                )
                fused_xyxy[c, column] = float(weighted_sum / total)
    else:
        fused_xyxy = origins[:cluster_count] + (
            weighted_offsets[:cluster_count]
            / geometry_weight_sums[:cluster_count, None]
        )
    fused_masks.extend(
        None if cl is None else _FusedMask(x=cl.box[0], y=cl.box[1], crop=cl.fused)
        for cl in clusters
    )
    mean_scores = np.ldexp(
        weight_sums[:cluster_count] / sizes[:cluster_count],
        exponents[:cluster_count],
    )
    return fused_xyxy, mean_scores, sizes[:cluster_count], mask_clusters


def _mask_weighted_fusion_clusters(
    boxes: npt.NDArray[np.number],
    scores: npt.NDArray[np.number],
    categories: npt.NDArray[np.number] | None,
    iou_threshold: float,
    overlap_metric: OverlapMetric,
    masks: npt.NDArray[Any] | CompactMask,
    num_models: int | None = None,
    output_dtype: npt.DTypeLike = np.float64,
    sources: npt.NDArray[np.integer] | None = None,
    box_weights: _FusionBoxWeights = "confidence",
    source_weights: npt.ArrayLike | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[np.int64],
    list[_FusedMask | None],
]:
    """Cluster and fuse segmentation masks with Weighted Box Fusion.

    The mask counterpart of
    :func:`~supervision.detection.utils.nms._box_weighted_fusion_clusters`: masks
    are clustered within their class by overlap with the cluster's running fused
    mask, the mean score is rescaled by `num_models` as for boxes, and the fused
    rows are ordered by descending fused score. The fused mask of a cluster of two
    or more is the confidence-weighted vote of its members: a pixel is kept when
    its votes are at least half the cluster's total weight, ties keep the pixel,
    and all-zero weights vote equally. The seed's mask is used if the vote leaves no
    pixel. The fused `xyxy` is the confidence-weighted mean of the members' boxes,
    or the seed's box for a cluster of one.

    Args:
        boxes: `xyxy` boxes aligned with `masks`, shape `(n, 4)`.
        scores: Confidence scores aligned with `masks`, shape `(n,)`.
        categories: Class ids, shape `(n,)`, or `None` to fuse across classes.
        iou_threshold: Overlap that a mask must strictly exceed to join a cluster.
        overlap_metric: Metric used to compute the overlap.
        masks: Dense `(n, H, W)` masks or a `CompactMask`.
        num_models: When set, each fused score is multiplied by
            `min(cluster_size, num_models) / num_models`.
        output_dtype: Floating dtype of the returned boxes and scores.
        sources: Optional source id per mask, shape `(n,)`; a cluster holds at most
            one mask per source.
        box_weights: `"confidence"` (default) or `"uniform"` weighting of the
            members in the vote and the box mean; the fused score is unaffected.
        source_weights: Optional weight of each source id, indexed by id, with one
            weight per model (`num_models` is required). It scales each member's
            vote and box weight, and the fused score becomes the weighted sum of
            the members' scores over the sum of all source weights.

    Raises:
        ValueError: If the box weights, sources, masks or any input values are
            invalid.

    Returns:
        A tuple `(fused_xyxy, fused_scores, labels, fused_masks)`. The first two have
            shapes `(m, 4)` and `(m,)` and are sorted by descending fused score;
            `labels[i]` is the returned row of the cluster that input `i` joined;
            `fused_masks[r]` is the fused mask of row `r`, or `None` when that
            cluster holds a single mask.
    """
    if box_weights not in ("confidence", "uniform"):
        raise ValueError(
            f"box_weights must be 'confidence' or 'uniform', got {box_weights!r}."
        )
    count = len(boxes)
    if len(masks) != count or (
        not isinstance(masks, CompactMask) and np.ndim(masks) != 3
    ):
        raise ValueError(
            "Masks passed to Weighted Box Fusion must be an (n, H, W) array or a "
            f"CompactMask with one mask per box, got {count} boxes and "
            f"{len(masks)} masks."
        )
    sources, weights = _prepare_fusion_sources(
        sources, source_weights, num_models, count
    )
    if count == 0:
        return (
            np.empty((0, 4), dtype=output_dtype),
            np.empty(0, dtype=output_dtype),
            np.empty(0, dtype=np.int64),
            [],
        )
    _validate_fusion_confidences(scores)
    _validate_fusion_boxes(boxes)
    boxes_float = boxes.astype(np.float64)
    per_class_masks: list[_FusedMask | None] = []

    def fuse_class(
        row_ids: npt.NDArray[np.int64],
        class_scores: npt.NDArray[np.number],
        threshold: float,
        metric: OverlapMetric,
        class_sources: npt.NDArray[np.integer] | None,
        member_weights: _FusionBoxWeights,
        class_factors: npt.NDArray[np.float64] | None,
    ) -> Any:
        """Fuse the rows of one class, collecting their fused masks in order."""
        return _fuse_single_class_masks(
            masks,
            boxes_float,
            row_ids,
            class_scores,
            threshold,
            metric,
            class_sources,
            member_weights,
            per_class_masks,
            class_factors,
        )

    fused_xyxy, fused_scores, labels, order = _fuse_by_class(
        np.arange(count, dtype=np.int64),
        scores,
        categories,
        iou_threshold,
        overlap_metric,
        num_models,
        output_dtype,
        output_dtype,
        sources,
        box_weights,
        fuse_class,
        weights,
    )
    return fused_xyxy, fused_scores, labels, [per_class_masks[i] for i in order]


def _paint_fused_masks(
    masks: npt.NDArray[Any] | CompactMask,
    winner_rows: npt.NDArray[np.intp],
    fused_masks: list[_FusedMask | None],
) -> npt.NDArray[Any] | CompactMask:
    """Build the masks of a fusion result from the winners and the fused crops.

    A cluster of one mask keeps its winner's mask object unchanged; a larger cluster
    gets its fused mask. A `CompactMask` stays compact, and a dense array stays
    dense with the dtype of the input.

    Args:
        masks: The input dense `(N, H, W)` masks or `CompactMask`.
        winner_rows: Input row of each cluster's winner.
        fused_masks: Fused mask per result row, `None` for single-mask clusters.

    Returns:
        The masks of the result, one per cluster.
    """
    if isinstance(masks, CompactMask):
        winners = cast(CompactMask, masks[winner_rows])
        rles = list(winners._rles)
        crop_shapes = winners._crop_shapes.copy()
        offsets = winners._offsets.copy()
        for row, fused in enumerate(fused_masks):
            if fused is not None:
                rles[row] = _mask_to_rle_counts(fused.crop)
                crop_shapes[row] = fused.crop.shape
                offsets[row] = (fused.x, fused.y)
        return CompactMask(rles, crop_shapes, offsets, masks.image_shape)
    result = masks[winner_rows]
    for row, fused in enumerate(fused_masks):
        if fused is not None:
            height, width = fused.crop.shape
            result[row] = False
            result[row, fused.y : fused.y + height, fused.x : fused.x + width] = (
                fused.crop
            )
    return result
