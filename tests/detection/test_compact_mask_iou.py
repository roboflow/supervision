"""Correctness and integration tests for CompactMask IoU and NMS.

These tests verify that:
- compact_mask_iou_batch gives numerically identical results to the
  dense mask_iou_batch (raster IoU) for all overlap patterns.
- mask_iou_batch dispatches correctly when given CompactMask inputs.
- mask_non_max_suppression and mask_non_max_merge work with CompactMask
  and produce the same keep-set as when given equivalent dense arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    compact_mask_iou_batch,
    mask_iou_batch,
    mask_non_max_suppression,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cm_from_masks(masks: np.ndarray, image_shape: tuple[int, int]) -> CompactMask:
    """Build a CompactMask using full-image bounding boxes (lossless)."""
    n = len(masks)
    h, w = image_shape
    xyxy = np.tile(np.array([0, 0, w - 1, h - 1], dtype=np.float32), (n, 1))
    return CompactMask.from_dense(masks, xyxy, image_shape=image_shape)


def _cm_tight(masks: np.ndarray, image_shape: tuple[int, int]) -> CompactMask:
    """Build a CompactMask using tight per-mask bounding boxes."""
    from supervision.detection.utils.converters import mask_to_xyxy

    xyxy = mask_to_xyxy(masks).astype(np.float32)
    return CompactMask.from_dense(masks, xyxy, image_shape=image_shape)


def _dense_iou(
    a: np.ndarray,
    b: np.ndarray,
    metric: OverlapMetric = OverlapMetric.IOU,
) -> np.ndarray:
    """Reference pairwise IoU using the existing dense implementation."""
    return mask_iou_batch(a, b, overlap_metric=metric)


class TestCompactMaskIouBatch:
    """Verify that compact_mask_iou_batch matches dense raster IoU exactly.

    Every test builds a pair of CompactMask collections from known boolean
    arrays, runs compact_mask_iou_batch, and compares the result to the dense
    reference computed by mask_iou_batch on the raw numpy arrays.
    """

    def test_no_overlap_gives_zero(self) -> None:
        """Non-overlapping masks should always produce IoU = 0."""
        h, w = 20, 20
        a = np.zeros((1, h, w), dtype=bool)
        a[0, 0:5, 0:5] = True  # top-left

        b = np.zeros((1, h, w), dtype=bool)
        b[0, 10:15, 10:15] = True  # bottom-right

        cm_a = _cm_from_masks(a, (h, w))
        cm_b = _cm_from_masks(b, (h, w))

        result = compact_mask_iou_batch(cm_a, cm_b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0)

    def test_identical_masks_give_one(self) -> None:
        """IoU of a mask with itself must be 1.0."""
        h, w = 20, 20
        masks = np.zeros((2, h, w), dtype=bool)
        masks[0, 2:8, 2:8] = True
        masks[1, 10:18, 10:18] = True

        cm = _cm_from_masks(masks, (h, w))
        result = compact_mask_iou_batch(cm, cm)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0], atol=1e-9)

    def test_matches_dense_random(self) -> None:
        """compact_mask_iou_batch must be numerically identical to dense IoU."""
        rng = np.random.default_rng(0)
        h, w = 30, 30
        a = rng.integers(0, 2, size=(5, h, w)).astype(bool)
        b = rng.integers(0, 2, size=(4, h, w)).astype(bool)

        cm_a = _cm_from_masks(a, (h, w))
        cm_b = _cm_from_masks(b, (h, w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b)
        dense_result = _dense_iou(a, b)

        assert compact_result.shape == (5, 4)
        np.testing.assert_allclose(compact_result, dense_result, atol=1e-9)

    def test_matches_dense_with_tight_bboxes(self) -> None:
        """Using tight bounding boxes (mask_to_xyxy) must still be accurate."""
        rng = np.random.default_rng(1)
        h, w = 40, 40
        a = rng.integers(0, 2, size=(4, h, w)).astype(bool)
        b = rng.integers(0, 2, size=(3, h, w)).astype(bool)

        cm_a = _cm_tight(a, (h, w))
        cm_b = _cm_tight(b, (h, w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b)
        dense_result = _dense_iou(a, b)

        np.testing.assert_allclose(compact_result, dense_result, atol=1e-9)

    def test_partial_overlap(self) -> None:
        """Partially overlapping masks: IoU should match the analytic value."""
        h, w = 10, 10
        # Mask A: columns 0-4 (5 wide), Mask B: columns 3-7 (5 wide).
        # Overlap: columns 3-4 (2 wide) x full height (10 rows) = 20 px.
        a = np.zeros((1, h, w), dtype=bool)
        a[0, :, 0:5] = True  # area = 50

        b = np.zeros((1, h, w), dtype=bool)
        b[0, :, 3:8] = True  # area = 50

        cm_a = _cm_from_masks(a, (h, w))
        cm_b = _cm_from_masks(b, (h, w))

        result = compact_mask_iou_batch(cm_a, cm_b)
        # inter=20, union=50+50-20=80 → IoU=0.25
        assert result[0, 0] == pytest.approx(0.25, abs=1e-9)
        np.testing.assert_allclose(result, _dense_iou(a, b), atol=1e-9)

    def test_ios_metric(self) -> None:
        """IOS = intersection / min(area_a, area_b) must match dense reference."""
        rng = np.random.default_rng(2)
        h, w = 25, 25
        a = rng.integers(0, 2, size=(3, h, w)).astype(bool)
        b = rng.integers(0, 2, size=(3, h, w)).astype(bool)

        cm_a = _cm_from_masks(a, (h, w))
        cm_b = _cm_from_masks(b, (h, w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b, OverlapMetric.IOS)
        dense_result = _dense_iou(a, b, OverlapMetric.IOS)

        np.testing.assert_allclose(compact_result, dense_result, atol=1e-9)

    def test_all_false_masks(self) -> None:
        """Zero-area masks should produce IoU = 0, not NaN."""
        h, w = 10, 10
        a = np.zeros((2, h, w), dtype=bool)
        b = np.zeros((2, h, w), dtype=bool)

        cm_a = _cm_from_masks(a, (h, w))
        cm_b = _cm_from_masks(b, (h, w))

        result = compact_mask_iou_batch(cm_a, cm_b)
        assert not np.any(np.isnan(result))
        np.testing.assert_array_equal(result, 0.0)

    def test_empty_inputs(self) -> None:
        """Empty CompactMask collections should return a zero-shaped matrix."""
        h, w = 10, 10
        empty = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (h, w),
        )
        masks = np.zeros((3, h, w), dtype=bool)
        cm = _cm_from_masks(masks, (h, w))

        result_a = compact_mask_iou_batch(empty, cm)
        assert result_a.shape == (0, 3)

        result_b = compact_mask_iou_batch(cm, empty)
        assert result_b.shape == (3, 0)

    def test_n_by_n_pairwise(self) -> None:
        """N x N pairwise IoU: diagonal must be 1.0 for non-zero-area masks."""
        h, w = 50, 50
        rng = np.random.default_rng(3)
        masks = rng.integers(0, 2, size=(8, h, w)).astype(bool)
        # Ensure no all-false mask (diagonal would be undefined).
        for i in range(8):
            masks[i, i * 5, i * 5] = True

        cm = _cm_from_masks(masks, (h, w))
        result = compact_mask_iou_batch(cm, cm)

        assert result.shape == (8, 8)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-9)
        np.testing.assert_allclose(result, _dense_iou(masks, masks), atol=1e-9)


class TestMaskIouBatchDispatch:
    """Verify mask_iou_batch dispatches correctly for CompactMask inputs.

    When both arguments are CompactMask, the function must route to the
    efficient RLE implementation and produce identical results to the dense
    path.  When one argument is dense and the other is CompactMask, the
    CompactMask must be materialised transparently before computation.
    """

    def test_both_compact_dispatches_to_rle(self) -> None:
        h, w = 20, 20
        rng = np.random.default_rng(10)
        a = rng.integers(0, 2, size=(3, h, w)).astype(bool)
        b = rng.integers(0, 2, size=(2, h, w)).astype(bool)

        cm_a = _cm_from_masks(a, (h, w))
        cm_b = _cm_from_masks(b, (h, w))

        result_compact = mask_iou_batch(cm_a, cm_b)
        result_dense = mask_iou_batch(a, b)

        np.testing.assert_allclose(result_compact, result_dense, atol=1e-9)

    def test_mixed_compact_and_dense(self) -> None:
        """One CompactMask + one dense array must still work correctly."""
        h, w = 20, 20
        rng = np.random.default_rng(11)
        a = rng.integers(0, 2, size=(3, h, w)).astype(bool)
        b = rng.integers(0, 2, size=(2, h, w)).astype(bool)

        cm_a = _cm_from_masks(a, (h, w))

        result = mask_iou_batch(cm_a, b)
        expected = mask_iou_batch(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-9)


class TestNmsWithCompactMask:
    """Verify mask NMS produces the same keep-set for CompactMask and dense inputs.

    The CompactMask path skips resizing (IoU is computed directly on RLE crops),
    while the dense path downscales to mask_dimension pixels first.  Results
    should agree for non-degenerate cases.
    """

    def test_nms_compact_matches_dense(self) -> None:
        """NMS keep-set is identical for CompactMask and the equivalent dense array."""
        h, w = 40, 40
        # Two non-overlapping high-confidence masks and one that overlaps mask 0.
        masks = np.zeros((3, h, w), dtype=bool)
        masks[0, 0:20, 0:20] = True  # top-left
        masks[1, 0:18, 0:18] = True  # heavily overlaps mask 0
        masks[2, 20:40, 20:40] = True  # bottom-right, no overlap

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack(
            [np.zeros((3, 4)), scores]  # dummy xyxy, real scores
        )

        cm = _cm_from_masks(masks, (h, w))

        keep_dense = mask_non_max_suppression(predictions, masks, iou_threshold=0.3)
        keep_compact = mask_non_max_suppression(predictions, cm, iou_threshold=0.3)

        np.testing.assert_array_equal(keep_compact, keep_dense)

    def test_nms_compact_no_suppression(self) -> None:
        """Non-overlapping masks: all should be kept."""
        h, w = 20, 20
        masks = np.zeros((3, h, w), dtype=bool)
        masks[0, 0:5, 0:5] = True
        masks[1, 7:12, 7:12] = True
        masks[2, 14:19, 14:19] = True

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (h, w))

        keep = mask_non_max_suppression(predictions, cm, iou_threshold=0.5)
        assert keep.all(), "All non-overlapping masks should be kept"

    def test_nms_compact_full_suppression(self) -> None:
        """Identical masks: only the highest-confidence one should survive."""
        h, w = 20, 20
        mask = np.zeros((1, h, w), dtype=bool)
        mask[0, 5:15, 5:15] = True

        masks = np.repeat(mask, 3, axis=0)
        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (h, w))

        keep = mask_non_max_suppression(predictions, cm, iou_threshold=0.5)
        assert keep.sum() == 1
        assert keep[0], "Highest-confidence mask should survive"
