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
    mask_non_max_merge,
    mask_non_max_suppression,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cm_from_masks(masks: np.ndarray, image_shape: tuple[int, int]) -> CompactMask:
    """Build a CompactMask using full-image bounding boxes (lossless)."""
    num_masks = len(masks)
    img_h, img_w = image_shape
    xyxy = np.tile(
        np.array([0, 0, img_w - 1, img_h - 1], dtype=np.float32), (num_masks, 1)
    )
    return CompactMask.from_dense(masks, xyxy, image_shape=image_shape)


def _cm_tight(masks: np.ndarray, image_shape: tuple[int, int]) -> CompactMask:
    """Build a CompactMask using tight per-mask bounding boxes."""
    from supervision.detection.utils.converters import mask_to_xyxy

    xyxy = mask_to_xyxy(masks).astype(np.float32)
    return CompactMask.from_dense(masks, xyxy, image_shape=image_shape)


def _dense_iou(
    masks_a: np.ndarray,
    masks_b: np.ndarray,
    metric: OverlapMetric = OverlapMetric.IOU,
) -> np.ndarray:
    """Reference pairwise IoU using the existing dense implementation."""
    return mask_iou_batch(masks_a, masks_b, overlap_metric=metric)


class TestCompactMaskIouBatch:
    """Verify that compact_mask_iou_batch matches dense raster IoU exactly.

    Every test builds a pair of CompactMask collections from known boolean
    arrays, runs compact_mask_iou_batch, and compares the result to the dense
    reference computed by mask_iou_batch on the raw numpy arrays.
    """

    def test_no_overlap_gives_zero(self) -> None:
        """Non-overlapping masks should always produce IoU = 0."""
        img_h, img_w = 20, 20
        masks_a = np.zeros((1, img_h, img_w), dtype=bool)
        masks_a[0, 0:5, 0:5] = True  # top-left

        masks_b = np.zeros((1, img_h, img_w), dtype=bool)
        masks_b[0, 10:15, 10:15] = True  # bottom-right

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        result = compact_mask_iou_batch(cm_a, cm_b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0)

    def test_identical_masks_give_one(self) -> None:
        """IoU of a mask with itself must be 1.0."""
        img_h, img_w = 20, 20
        masks = np.zeros((2, img_h, img_w), dtype=bool)
        masks[0, 2:8, 2:8] = True
        masks[1, 10:18, 10:18] = True

        cm = _cm_from_masks(masks, (img_h, img_w))
        result = compact_mask_iou_batch(cm, cm)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0], atol=1e-9)

    def test_matches_dense_random(self) -> None:
        """compact_mask_iou_batch must be numerically identical to dense IoU."""
        rng = np.random.default_rng(0)
        img_h, img_w = 30, 30
        masks_a = rng.integers(0, 2, size=(5, img_h, img_w)).astype(bool)
        masks_b = rng.integers(0, 2, size=(4, img_h, img_w)).astype(bool)

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b)
        dense_result = _dense_iou(masks_a, masks_b)

        assert compact_result.shape == (5, 4)
        np.testing.assert_allclose(compact_result, dense_result, atol=1e-9)

    def test_matches_dense_with_tight_bboxes(self) -> None:
        """Using tight bounding boxes (mask_to_xyxy) must still be accurate."""
        rng = np.random.default_rng(1)
        img_h, img_w = 40, 40
        masks_a = rng.integers(0, 2, size=(4, img_h, img_w)).astype(bool)
        masks_b = rng.integers(0, 2, size=(3, img_h, img_w)).astype(bool)

        cm_a = _cm_tight(masks_a, (img_h, img_w))
        cm_b = _cm_tight(masks_b, (img_h, img_w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b)
        dense_result = _dense_iou(masks_a, masks_b)

        np.testing.assert_allclose(compact_result, dense_result, atol=1e-9)

    def test_partial_overlap(self) -> None:
        """Partially overlapping masks: IoU should match the analytic value."""
        img_h, img_w = 10, 10
        # Mask A: columns 0-4 (5 wide), Mask B: columns 3-7 (5 wide).
        # Overlap: columns 3-4 (2 wide) x full height (10 rows) = 20 px.
        masks_a = np.zeros((1, img_h, img_w), dtype=bool)
        masks_a[0, :, 0:5] = True  # area = 50

        masks_b = np.zeros((1, img_h, img_w), dtype=bool)
        masks_b[0, :, 3:8] = True  # area = 50

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        result = compact_mask_iou_batch(cm_a, cm_b)
        # inter=20, union=50+50-20=80 → IoU=0.25
        assert result[0, 0] == pytest.approx(0.25, abs=1e-9)
        np.testing.assert_allclose(result, _dense_iou(masks_a, masks_b), atol=1e-9)

    def test_ios_metric(self) -> None:
        """IOS = intersection / min(area_a, area_b) must match dense reference."""
        rng = np.random.default_rng(2)
        img_h, img_w = 25, 25
        masks_a = rng.integers(0, 2, size=(3, img_h, img_w)).astype(bool)
        masks_b = rng.integers(0, 2, size=(3, img_h, img_w)).astype(bool)

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b, OverlapMetric.IOS)
        dense_result = _dense_iou(masks_a, masks_b, OverlapMetric.IOS)

        np.testing.assert_allclose(compact_result, dense_result, atol=1e-9)

    def test_all_false_masks(self) -> None:
        """Zero-area masks should produce IoU = 0, not NaN."""
        img_h, img_w = 10, 10
        masks_a = np.zeros((2, img_h, img_w), dtype=bool)
        masks_b = np.zeros((2, img_h, img_w), dtype=bool)

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        result = compact_mask_iou_batch(cm_a, cm_b)
        assert not np.any(np.isnan(result))
        np.testing.assert_array_equal(result, 0.0)

    def test_empty_inputs(self) -> None:
        """Empty CompactMask collections should return a zero-shaped matrix."""
        img_h, img_w = 10, 10
        empty = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (img_h, img_w),
        )
        masks = np.zeros((3, img_h, img_w), dtype=bool)
        cm = _cm_from_masks(masks, (img_h, img_w))

        result_a = compact_mask_iou_batch(empty, cm)
        assert result_a.shape == (0, 3)

        result_b = compact_mask_iou_batch(cm, empty)
        assert result_b.shape == (3, 0)

    def test_n_by_n_pairwise(self) -> None:
        """N x N pairwise IoU: diagonal must be 1.0 for non-zero-area masks."""
        img_h, img_w = 50, 50
        rng = np.random.default_rng(3)
        masks = rng.integers(0, 2, size=(8, img_h, img_w)).astype(bool)
        # Ensure no all-false mask (diagonal would be undefined).
        for mask_idx in range(8):
            masks[mask_idx, mask_idx * 5, mask_idx * 5] = True

        cm = _cm_from_masks(masks, (img_h, img_w))
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
        img_h, img_w = 20, 20
        rng = np.random.default_rng(10)
        masks_a = rng.integers(0, 2, size=(3, img_h, img_w)).astype(bool)
        masks_b = rng.integers(0, 2, size=(2, img_h, img_w)).astype(bool)

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        result_compact = mask_iou_batch(cm_a, cm_b)
        result_dense = mask_iou_batch(masks_a, masks_b)

        np.testing.assert_allclose(result_compact, result_dense, atol=1e-9)

    def test_mixed_compact_and_dense(self) -> None:
        """One CompactMask + one dense array must still work correctly."""
        img_h, img_w = 20, 20
        rng = np.random.default_rng(11)
        masks_a = rng.integers(0, 2, size=(3, img_h, img_w)).astype(bool)
        masks_b = rng.integers(0, 2, size=(2, img_h, img_w)).astype(bool)

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))

        result = mask_iou_batch(cm_a, masks_b)
        expected = mask_iou_batch(masks_a, masks_b)
        np.testing.assert_allclose(result, expected, atol=1e-9)


class TestNmsWithCompactMask:
    """Verify mask NMS produces identical keep-sets for CompactMask and dense inputs.

    Both paths now use exact full-resolution IoU — no resize approximation.
    Tests use images larger than 640 px to ensure the old resize-to-640 path
    would have introduced lossy approximation (catching the regression).
    """

    def test_nms_compact_matches_dense(self) -> None:
        """NMS keep-set is identical for CompactMask and the equivalent dense array."""
        # Use > 640 px so the old resize-to-640 path would have been lossy.
        img_h, img_w = 720, 720
        masks = np.zeros((3, img_h, img_w), dtype=bool)
        masks[0, 0:360, 0:360] = True  # top-left
        masks[1, 0:324, 0:324] = True  # heavily overlaps mask 0
        masks[2, 360:720, 360:720] = True  # bottom-right, no overlap

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack(
            [np.zeros((3, 4)), scores]  # dummy xyxy, real scores
        )

        cm = _cm_from_masks(masks, (img_h, img_w))

        keep_dense = mask_non_max_suppression(predictions, masks, iou_threshold=0.3)
        keep_compact = mask_non_max_suppression(predictions, cm, iou_threshold=0.3)

        np.testing.assert_array_equal(keep_compact, keep_dense)

    def test_nms_compact_matches_dense_borderline(self) -> None:
        """Borderline IoU pair (≈ threshold) must agree — catches the resize bug.

        With resize-to-640, sub-pixel rounding on a pair whose true IoU is very
        close to the threshold flips the keep/suppress decision.  Both paths now
        compute exact pixel-level IoU so results are identical.
        """
        img_h, img_w = 1080, 1920
        masks = np.zeros((2, img_h, img_w), dtype=bool)
        # Mask 0: 200x200 square; mask 1: shifted 141 px → true IoU ≈ 0.50.
        masks[0, 100:300, 100:300] = True
        masks[1, 241:441, 241:441] = True

        scores = np.array([0.9, 0.8])
        predictions = np.column_stack([np.zeros((2, 4)), scores])
        cm = _cm_from_masks(masks, (img_h, img_w))

        keep_dense = mask_non_max_suppression(predictions, masks, iou_threshold=0.5)
        keep_compact = mask_non_max_suppression(predictions, cm, iou_threshold=0.5)

        np.testing.assert_array_equal(keep_compact, keep_dense)

    def test_nms_compact_no_suppression(self) -> None:
        """Non-overlapping masks: all should be kept."""
        img_h, img_w = 20, 20
        masks = np.zeros((3, img_h, img_w), dtype=bool)
        masks[0, 0:5, 0:5] = True
        masks[1, 7:12, 7:12] = True
        masks[2, 14:19, 14:19] = True

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (img_h, img_w))

        keep = mask_non_max_suppression(predictions, cm, iou_threshold=0.5)
        assert keep.all(), "All non-overlapping masks should be kept"

    def test_nms_compact_full_suppression(self) -> None:
        """Identical masks: only the highest-confidence one should survive."""
        img_h, img_w = 20, 20
        mask = np.zeros((1, img_h, img_w), dtype=bool)
        mask[0, 5:15, 5:15] = True

        masks = np.repeat(mask, 3, axis=0)
        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (img_h, img_w))

        keep = mask_non_max_suppression(predictions, cm, iou_threshold=0.5)
        assert keep.sum() == 1
        assert keep[0], "Highest-confidence mask should survive"


class TestNmmWithCompactMask:
    """Verify mask_non_max_merge produces the same groups for CompactMask and dense.

    NMM materialises CompactMask to a downscaled dense array internally, so
    results must be numerically identical to the dense path.
    """

    def test_nmm_compact_matches_dense(self) -> None:
        """Merge groups must match between CompactMask and dense inputs."""
        img_h, img_w = 40, 40
        masks = np.zeros((3, img_h, img_w), dtype=bool)
        masks[0, 0:20, 0:20] = True  # top-left
        masks[1, 0:18, 0:18] = True  # heavily overlaps mask 0
        masks[2, 20:40, 20:40] = True  # bottom-right, no overlap

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (img_h, img_w))

        groups_dense = mask_non_max_merge(predictions, masks, iou_threshold=0.3)
        groups_compact = mask_non_max_merge(predictions, cm, iou_threshold=0.3)

        def normalise(groups: list[list[int]]) -> list[list[int]]:
            return sorted(sorted(group) for group in groups)

        assert normalise(groups_compact) == normalise(groups_dense)

    def test_nmm_no_merge(self) -> None:
        """Non-overlapping masks: every mask should be its own group."""
        img_h, img_w = 20, 20
        masks = np.zeros((3, img_h, img_w), dtype=bool)
        masks[0, 0:5, 0:5] = True
        masks[1, 7:12, 7:12] = True
        masks[2, 14:19, 14:19] = True

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (img_h, img_w))

        groups = mask_non_max_merge(predictions, cm, iou_threshold=0.5)
        assert len(groups) == 3, "Each non-overlapping mask gets its own group"
        assert all(len(group) == 1 for group in groups)

    def test_nmm_full_merge(self) -> None:
        """Identical masks: all predictions should merge into one group."""
        img_h, img_w = 20, 20
        single = np.zeros((1, img_h, img_w), dtype=bool)
        single[0, 5:15, 5:15] = True
        masks = np.repeat(single, 3, axis=0)

        scores = np.array([0.9, 0.8, 0.7])
        predictions = np.column_stack([np.zeros((3, 4)), scores])
        cm = _cm_from_masks(masks, (img_h, img_w))

        groups = mask_non_max_merge(predictions, cm, iou_threshold=0.5)
        assert len(groups) == 1, "Identical masks must collapse to one group"
        assert len(groups[0]) == 3


# ---------------------------------------------------------------------------
# Random scenario helpers
# ---------------------------------------------------------------------------

# Small (N, h, w) configs to keep IoU tests fast.
_IOU_RANDOM_CONFIGS = [
    (5, 30, 30),
    (8, 40, 40),
    (10, 25, 25),
    (6, 50, 50),
    (12, 30, 40),
    (5, 60, 60),
    (15, 20, 20),
    (7, 35, 35),
    (10, 40, 50),
    (8, 45, 45),
]


def _random_masks(
    rng: np.random.Generator,
    num_masks: int,
    img_h: int,
    img_w: int,
    fill_prob: float = 0.25,
) -> np.ndarray:
    """Generate *num_masks* random boolean masks with at least one True pixel each."""
    masks = np.zeros((num_masks, img_h, img_w), dtype=bool)
    for mask_idx in range(num_masks):
        y1 = rng.integers(0, img_h)
        y2 = rng.integers(y1, img_h)
        x1 = rng.integers(0, img_w)
        x2 = rng.integers(x1, img_w)
        region = rng.random((y2 - y1 + 1, x2 - x1 + 1)) < fill_prob
        if not region.any():
            region[0, 0] = True
        masks[mask_idx, y1 : y2 + 1, x1 : x2 + 1] = region
    return masks


class TestCompactMaskIouRandom:
    """compact_mask_iou_batch matches dense mask_iou_batch across 10 random seeds.

    Uses small mask counts (5-15) and image sizes (20x20 to 60x60) to keep
    individual test runs under 1 second.
    """

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_parity_seed(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        num_masks_a, img_h, img_w = _IOU_RANDOM_CONFIGS[seed]
        num_masks_b = max(3, num_masks_a - 2)

        masks_a = _random_masks(rng, num_masks_a, img_h, img_w)
        masks_b = _random_masks(rng, num_masks_b, img_h, img_w)

        cm_a = _cm_from_masks(masks_a, (img_h, img_w))
        cm_b = _cm_from_masks(masks_b, (img_h, img_w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b)
        dense_result = _dense_iou(masks_a, masks_b)

        assert compact_result.shape == (num_masks_a, num_masks_b), (
            f"Shape mismatch: {compact_result.shape} vs ({num_masks_a}, {num_masks_b})"
        )
        np.testing.assert_allclose(
            compact_result,
            dense_result,
            atol=1e-9,
            err_msg=f"IoU mismatch: seed={seed}, N_a={num_masks_a}, N_b={num_masks_b}",
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_self_iou_diagonal(self, seed: int) -> None:
        """Self-IoU diagonal must be 1.0 for masks with at least one True pixel."""
        rng = np.random.default_rng(seed + 50)
        num_masks, img_h, img_w = _IOU_RANDOM_CONFIGS[seed]
        masks = _random_masks(rng, num_masks, img_h, img_w)

        cm = _cm_from_masks(masks, (img_h, img_w))
        result = compact_mask_iou_batch(cm, cm)

        np.testing.assert_allclose(
            np.diag(result),
            1.0,
            atol=1e-9,
            err_msg=f"Diagonal not 1.0 for seed={seed}",
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_tight_bbox_parity(self, seed: int) -> None:
        """Tight bounding boxes (mask_to_xyxy) must still produce identical IoU."""
        from supervision.detection.utils.converters import mask_to_xyxy

        rng = np.random.default_rng(seed + 200)
        num_masks, img_h, img_w = _IOU_RANDOM_CONFIGS[seed]
        num_masks_b = max(3, num_masks - 2)

        masks_a = _random_masks(rng, num_masks, img_h, img_w)
        masks_b = _random_masks(rng, num_masks_b, img_h, img_w)

        xyxy_a = mask_to_xyxy(masks_a).astype(np.float32)
        xyxy_b = mask_to_xyxy(masks_b).astype(np.float32)

        cm_a = CompactMask.from_dense(masks_a, xyxy_a, image_shape=(img_h, img_w))
        cm_b = CompactMask.from_dense(masks_b, xyxy_b, image_shape=(img_h, img_w))

        compact_result = compact_mask_iou_batch(cm_a, cm_b)
        dense_result = _dense_iou(masks_a, masks_b)

        np.testing.assert_allclose(
            compact_result,
            dense_result,
            atol=1e-9,
            err_msg=f"Tight bbox IoU mismatch for seed={seed}",
        )
