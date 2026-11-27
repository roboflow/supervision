"""Weighted Box Fusion of segmentation masks.

Each cluster keeps a running fused mask: a confidence-weighted vote of its members, held
in a buffer that spans only the union of the members' bounding boxes. Memory therefore
scales with the size of the objects, never with the image size or with the number of
clusters times the image area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.converters import _mask_to_rle_counts
from supervision.detection.utils.iou import OverlapMetric, _overlap_from_areas
from supervision.detection.utils.nms import (
    _fuse_by_class,
    _FusionBoxWeights,
    _pick_cluster,
    _validate_fusion_boxes,
    _validate_fusion_confidences,
)


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
        votes: Weighted votes per pixel of `box`.
        total: Sum of the members' vote weights.
        seed_crop: Crop of the seed, the first and highest-confidence member.
        seed_xy: Image position of `seed_crop`.
        fused: The cluster's current fused mask over `box`.
        area: Pixel count of `fused`.
    """

    box: list[int]
    votes: npt.NDArray[np.float64]
    total: float
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
        return cls(
            box=[x, y, x + width, y + height],
            votes=weight * crop.astype(np.float64),
            total=weight,
            seed_crop=crop,
            seed_xy=(x, y),
            fused=crop,
            area=int(np.count_nonzero(crop)),
        )

    def add(self, crop: npt.NDArray[np.bool_], x: int, y: int, weight: float) -> None:
        """Add a member and recompute the fused mask.

        A pixel is kept when its votes are at least half the total weight, so ties
        keep the pixel. If no pixel survives, the seed's mask is used instead.

        Args:
            crop: The member's boolean crop.
            x: Column of the crop's left edge in the image.
            y: Row of the crop's top edge in the image.
            weight: The member's vote weight.
        """
        height, width = crop.shape
        left, top = min(self.box[0], x), min(self.box[1], y)
        right = max(self.box[2], x + width)
        bottom = max(self.box[3], y + height)
        if [left, top, right, bottom] != self.box:
            grown = np.zeros((bottom - top, right - left), dtype=np.float64)
            old_top, old_left = self.box[1] - top, self.box[0] - left
            old_height = self.box[3] - self.box[1]
            old_width = self.box[2] - self.box[0]
            grown[old_top : old_top + old_height, old_left : old_left + old_width] = (
                self.votes
            )
            self.votes = grown
            self.box = [left, top, right, bottom]
        self.votes[y - top : y - top + height, x - left : x - left + width] += (
            weight * crop
        )
        self.total += weight
        self.fused = self.votes * 2 >= self.total
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

    A `CompactMask` crop is decoded as stored. A dense mask is cropped to the
    bounds of its true pixels; an empty mask gives a `0 x 0` crop.

    Args:
        masks: Dense `(N, H, W)` masks or a `CompactMask`.
        index: Row of the mask.

    Returns:
        A tuple `(crop, x, y)` with the crop and its top-left corner.
    """
    if isinstance(masks, CompactMask):
        x, y = (int(value) for value in masks.offsets[index])
        return masks.crop(index), x, y
    mask = np.asarray(masks[index], dtype=np.bool_)
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.bool_), 0, 0
    columns = np.flatnonzero(mask.any(axis=0))
    crop = mask[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    return crop, int(columns[0]), int(rows[0])


def _overlap_with_clusters(
    clusters: list[_MaskCluster],
    cluster_boxes: npt.NDArray[np.int64],
    crop: npt.NDArray[np.bool_],
    x: int,
    y: int,
    overlap_metric: OverlapMetric,
) -> npt.NDArray[np.float64]:
    """Measure the overlap of a mask with each cluster's current fused mask.

    Only the clusters whose bounds intersect the mask are compared, and each
    comparison reads just the intersection of the two crops.

    Args:
        clusters: The existing clusters.
        cluster_boxes: Bounds of the clusters, shape `(k, 4)`.
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
    for index in np.flatnonzero(intersects):
        cluster = clusters[index]
        left, top = max(cluster.box[0], x), max(cluster.box[1], y)
        right = min(cluster.box[2], x + width)
        bottom = min(cluster.box[3], y + height)
        fused = cluster.fused[
            top - cluster.box[1] : bottom - cluster.box[1],
            left - cluster.box[0] : right - cluster.box[0],
        ]
        member = crop[top - y : bottom - y, left - x : right - x]
        intersection = np.array([[float(np.count_nonzero(fused & member))]])
        overlaps[index] = _overlap_from_areas(
            intersection,
            np.array([float(cluster.area)]),
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
    Weights are each score divided by the power of two that brings the seed's score
    into `[0.5, 1)`; a cluster whose seed has a zero score weighs its members
    equally. `box_weights="uniform"` also weighs every member equally in the vote
    and in the `xyxy` mean, leaving the mean score unchanged.

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

    Returns:
        A tuple `(fused_xyxy, mean_scores, cluster_sizes, mask_clusters)` with shapes
            `(k, 4)`, `(k,)`, `(k,)` and `(n,)`, where `mask_clusters[i]` is the
            cluster that the `i`-th row of this class joined.
    """
    scores_float = scores.astype(np.float64)
    count = len(row_ids)
    clusters: list[_MaskCluster] = []
    cluster_boxes = np.empty((count, 4), dtype=np.int64)
    mask_clusters = np.empty(count, dtype=np.int64)
    origins = np.empty((count, 4), dtype=np.float64)
    exponents = np.zeros(count, dtype=np.int32)
    unit_weights = np.zeros(count, dtype=bool)
    weight_sums = np.zeros(count, dtype=np.float64)
    weighted_offsets = np.zeros((count, 4), dtype=np.float64)
    geometry_weight_sums = np.zeros(count, dtype=np.float64)
    sizes = np.zeros(count, dtype=np.int64)
    uniform = box_weights == "uniform"
    if sources is None:
        source_codes = None
        source_in_cluster = None
    else:
        _, source_codes = np.unique(sources, return_inverse=True)
        source_in_cluster = np.zeros((count, source_codes.max() + 1), dtype=bool)

    for position in np.argsort(-scores_float, kind="stable"):
        crop, x, y = _mask_crop(masks, int(row_ids[position]))
        score = scores_float[position]
        cluster = -1
        if clusters:
            eligible = None
            if source_in_cluster is not None and source_codes is not None:
                eligible = ~source_in_cluster[: len(clusters), source_codes[position]]
            overlaps = _overlap_with_clusters(
                clusters, cluster_boxes[: len(clusters)], crop, x, y, overlap_metric
            )
            cluster = _pick_cluster(overlaps, iou_threshold, eligible)
        if cluster < 0:
            cluster = len(clusters)
            origins[cluster] = boxes[row_ids[position]]
            exponents[cluster] = np.frexp(score)[1]
            # The seed has the highest score, so a zero seed weight means every
            # member of the cluster has none and the vote becomes unweighted.
            unit_weights[cluster] = uniform or score == 0
        weight = float(np.ldexp(score, -exponents[cluster]))
        vote_weight = 1.0 if unit_weights[cluster] else weight
        if cluster == len(clusters):
            clusters.append(_MaskCluster.start(crop, x, y, vote_weight))
        else:
            clusters[cluster].add(crop, x, y, vote_weight)
        cluster_boxes[cluster] = clusters[cluster].box
        mask_clusters[position] = cluster
        if source_in_cluster is not None and source_codes is not None:
            source_in_cluster[cluster, source_codes[position]] = True
        weight_sums[cluster] += weight
        geometry_weight_sums[cluster] += vote_weight
        sizes[cluster] += 1
        weighted_offsets[cluster] += vote_weight * (
            boxes[row_ids[position]] - origins[cluster]
        )

    cluster_count = len(clusters)
    fused_xyxy = origins[:cluster_count] + (
        weighted_offsets[:cluster_count] / geometry_weight_sums[:cluster_count, None]
    )
    fused_masks.extend(
        None if sizes[c] == 1 else _FusedMask(x=cl.box[0], y=cl.box[1], crop=cl.fused)
        for c, cl in enumerate(clusters)
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

    Raises:
        ValueError: If the box weights, masks or any input values are invalid.

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
        weights: _FusionBoxWeights,
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
            weights,
            per_class_masks,
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
    result = np.array(masks[winner_rows])
    for row, fused in enumerate(fused_masks):
        if fused is not None:
            height, width = fused.crop.shape
            result[row] = False
            result[row, fused.y : fused.y + height, fused.x : fused.x + width] = (
                fused.crop
            )
    return result
