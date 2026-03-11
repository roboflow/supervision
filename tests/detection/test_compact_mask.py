"""Unit tests for CompactMask and its private RLE helpers."""

from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.compact_mask import (
    CompactMask,
    _rle_area,
    _rle_decode,
    _rle_encode,
)
from supervision.detection.utils.converters import mask_to_xyxy
from supervision.detection.utils.masks import (
    calculate_masks_centroids,
    contains_holes,
    contains_multiple_segments,
    move_masks,
)


def _make_cm(masks: np.ndarray, image_shape: tuple[int, int]) -> CompactMask:
    """Build a CompactMask whose crops equal the full bounding-box extents."""
    num_masks = len(masks)
    img_h, img_w = image_shape
    xyxy = np.tile(np.array([0, 0, img_w, img_h], dtype=np.float32), (num_masks, 1))
    return CompactMask.from_dense(masks, xyxy, image_shape=image_shape)


class TestRleHelpers:
    """Tests for _rle_encode, _rle_decode, and _rle_area.

    Verifies that the private RLE encoding round-trips correctly for a range
    of mask shapes (all-False, all-True, diagonal, L-shape, checkerboard,
    single-pixel, and empty), and that _rle_area matches np.sum on the
    original boolean array.
    """

    @pytest.mark.parametrize(
        ("mask_2d", "description"),
        [
            (np.zeros((5, 5), dtype=bool), "all-False"),
            (np.ones((5, 5), dtype=bool), "all-True"),
            (np.eye(4, dtype=bool), "diagonal"),
            (
                np.array([[True, True, False], [True, False, False]], dtype=bool),
                "L-shape",
            ),
            (
                np.indices((4, 4)).sum(axis=0) % 2 == 0,
                "checkerboard",
            ),
            (np.zeros((1, 1), dtype=bool), "single-pixel-False"),
            (np.ones((1, 1), dtype=bool), "single-pixel-True"),
            (np.zeros((0, 0), dtype=bool), "empty"),
        ],
    )
    def test_encode_decode_round_trip(
        self, mask_2d: np.ndarray, description: str
    ) -> None:
        if mask_2d.size == 0:
            rle = _rle_encode(mask_2d)
            assert _rle_area(rle) == 0
            return

        rle = _rle_encode(mask_2d)
        assert rle.dtype == np.int32, "RLE must be int32"
        reconstructed = _rle_decode(rle, mask_2d.shape[0], mask_2d.shape[1])
        np.testing.assert_array_equal(
            reconstructed, mask_2d, err_msg=f"Round-trip failed for: {description}"
        )

    @pytest.mark.parametrize(
        "mask_2d",
        [
            np.zeros((6, 6), dtype=bool),
            np.ones((6, 6), dtype=bool),
            np.eye(6, dtype=bool),
            np.array([[True, False, True], [False, True, False]], dtype=bool),
        ],
    )
    def test_area_matches_numpy_sum(self, mask_2d: np.ndarray) -> None:
        rle = _rle_encode(mask_2d)
        assert _rle_area(rle) == int(np.sum(mask_2d))


class TestFromDenseToDense:
    """Tests for CompactMask.from_dense and to_dense.

    Verifies that the from_dense → to_dense round-trip is lossless when the
    bounding boxes span the full image (no True pixels fall outside the crop).
    Covers N=0 (empty), N=1 (single mask), and N=5 (several random masks).
    """

    @pytest.mark.parametrize(
        ("num_masks", "image_shape"),
        [
            (0, (50, 50)),
            (1, (50, 50)),
            (5, (50, 50)),
        ],
    )
    def test_round_trip(self, num_masks: int, image_shape: tuple[int, int]) -> None:
        rng = np.random.default_rng(42)
        img_h, img_w = image_shape
        masks = rng.integers(0, 2, size=(num_masks, img_h, img_w)).astype(bool)
        cm = _make_cm(masks, image_shape)
        np.testing.assert_array_equal(cm.to_dense(), masks)

    def test_round_trip_with_mask_to_xyxy(self) -> None:
        """Round-trip must be lossless with inclusive xyxy from mask_to_xyxy."""
        img_h, img_w = 12, 14
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 3:7, 4:9] = True  # non-full-image object

        xyxy = mask_to_xyxy(masks).astype(np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        np.testing.assert_array_equal(cm.to_dense(), masks)


class TestGetItem:
    """Tests for CompactMask.__getitem__.

    Covers four indexing modes:
    - Integer index → dense (H, W) np.ndarray with correct shape and dtype.
    - List of indices → new CompactMask with the selected detections.
    - Slice → new CompactMask with the sliced detections.
    - Boolean ndarray → new CompactMask filtered by the boolean selector.
    """

    def test_int_returns_2d_dense(self) -> None:
        img_h, img_w = 30, 40
        rng = np.random.default_rng(0)
        masks = rng.integers(0, 2, size=(3, img_h, img_w)).astype(bool)
        cm = _make_cm(masks, (img_h, img_w))

        result = cm[1]
        assert isinstance(result, np.ndarray)
        assert result.shape == (img_h, img_w)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, masks[1])

    def test_list_returns_compact_mask(self) -> None:
        img_h, img_w = 20, 20
        masks = np.zeros((4, img_h, img_w), dtype=bool)
        for mask_idx in range(4):
            masks[
                mask_idx,
                mask_idx * 2 : mask_idx * 2 + 2,
                mask_idx * 2 : mask_idx * 2 + 2,
            ] = True
        cm = _make_cm(masks, (img_h, img_w))

        subset = cm[[0, 2]]
        assert isinstance(subset, CompactMask)
        assert len(subset) == 2
        np.testing.assert_array_equal(subset[0], masks[0])
        np.testing.assert_array_equal(subset[1], masks[2])

    def test_slice_returns_compact_mask(self) -> None:
        img_h, img_w = 20, 20
        masks = np.zeros((5, img_h, img_w), dtype=bool)
        cm = _make_cm(masks, (img_h, img_w))

        subset = cm[1:4]
        assert isinstance(subset, CompactMask)
        assert len(subset) == 3

    def test_bool_ndarray(self) -> None:
        img_h, img_w = 15, 15
        rng = np.random.default_rng(7)
        masks = rng.integers(0, 2, size=(4, img_h, img_w)).astype(bool)
        cm = _make_cm(masks, (img_h, img_w))

        selector = np.array([True, False, True, False])
        subset = cm[selector]
        assert isinstance(subset, CompactMask)
        assert len(subset) == 2
        np.testing.assert_array_equal(subset[0], masks[0])
        np.testing.assert_array_equal(subset[1], masks[2])

    def test_bool_list(self) -> None:
        """Python list[bool] should behave like boolean masking."""
        img_h, img_w = 15, 15
        rng = np.random.default_rng(8)
        masks = rng.integers(0, 2, size=(4, img_h, img_w)).astype(bool)
        cm = _make_cm(masks, (img_h, img_w))

        subset = cm[[True, False, True, False]]
        assert isinstance(subset, CompactMask)
        assert len(subset) == 2
        np.testing.assert_array_equal(subset[0], masks[0])
        np.testing.assert_array_equal(subset[1], masks[2])


class TestProperties:
    """Tests for len, shape, dtype, and area properties.

    Verifies that the shape tuple follows the (N, H, W) dense convention,
    dtype is always bool, and area returns per-mask True-pixel counts that
    match np.sum on the corresponding dense masks.
    """

    def test_len(self) -> None:
        masks = np.zeros((3, 10, 10), dtype=bool)
        cm = _make_cm(masks, (10, 10))
        assert len(cm) == 3

    def test_shape(self) -> None:
        masks = np.zeros((3, 10, 10), dtype=bool)
        cm = _make_cm(masks, (10, 10))
        assert cm.shape == (3, 10, 10)

    def test_shape_empty(self) -> None:
        cm = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (480, 640),
        )
        assert cm.shape == (0, 480, 640)

    def test_dtype(self) -> None:
        cm = _make_cm(np.zeros((1, 5, 5), dtype=bool), (5, 5))
        assert cm.dtype == np.dtype(bool)

    def test_area_matches_dense(self) -> None:
        img_h, img_w = 20, 20
        rng = np.random.default_rng(3)
        masks = rng.integers(0, 2, size=(4, img_h, img_w)).astype(bool)
        cm = _make_cm(masks, (img_h, img_w))

        expected = np.array([mask.sum() for mask in masks])
        np.testing.assert_array_equal(cm.area, expected)

    def test_area_empty(self) -> None:
        cm = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (10, 10),
        )
        assert cm.area.shape == (0,)


class TestCrop:
    """Tests for CompactMask.crop.

    Verifies that crop(index) returns an array shaped (crop_h, crop_w)
    containing only the pixels within the bounding box, without allocating
    the full (H, W) image.
    """

    def test_returns_crop_shape(self) -> None:
        img_h, img_w = 50, 60
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 10:30, 5:25] = True  # 20 x 20 region
        xyxy = np.array([[5, 10, 24, 29]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        crop = cm.crop(0)
        assert crop.shape == (20, 20)
        assert crop.all()  # the entire crop should be True


class TestArrayProtocol:
    """Tests for the __array__ protocol.

    Verifies that np.asarray(cm) materialises the full (N, H, W) dense array
    and that optional dtype casting (e.g. to uint8) is correctly applied.
    """

    def test_array_protocol(self) -> None:
        img_h, img_w = 10, 10
        rng = np.random.default_rng(9)
        masks = rng.integers(0, 2, size=(2, img_h, img_w)).astype(bool)
        cm = _make_cm(masks, (img_h, img_w))

        arr = np.asarray(cm)
        assert arr.shape == (2, img_h, img_w)
        np.testing.assert_array_equal(arr, masks)

    def test_dtype_cast(self) -> None:
        masks = np.ones((1, 5, 5), dtype=bool)
        cm = _make_cm(masks, (5, 5))
        arr = np.asarray(cm, dtype=np.uint8)
        assert arr.dtype == np.uint8
        assert arr.sum() == 25


class TestMerge:
    """Tests for CompactMask.merge.

    Verifies that multiple CompactMask instances with the same image_shape
    can be concatenated into a single CompactMask, that merging with an empty
    instance works, that an empty input list raises ValueError, and that
    mismatched image shapes raise ValueError.
    """

    def test_merge(self) -> None:
        img_h, img_w = 20, 20
        masks1 = np.zeros((2, img_h, img_w), dtype=bool)
        masks2 = np.zeros((3, img_h, img_w), dtype=bool)
        cm1 = _make_cm(masks1, (img_h, img_w))
        cm2 = _make_cm(masks2, (img_h, img_w))

        merged = CompactMask.merge([cm1, cm2])
        assert len(merged) == 5
        assert merged.shape == (5, img_h, img_w)
        np.testing.assert_array_equal(
            merged.to_dense(), np.concatenate([masks1, masks2], axis=0)
        )

    def test_merge_with_empty(self) -> None:
        img_h, img_w = 10, 10
        empty_cm = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (img_h, img_w),
        )
        masks = np.zeros((2, img_h, img_w), dtype=bool)
        cm = _make_cm(masks, (img_h, img_w))

        merged = CompactMask.merge([empty_cm, cm])
        assert len(merged) == 2

    def test_merge_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            CompactMask.merge([])

    def test_merge_mismatched_image_shape_raises(self) -> None:
        cm1 = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (10, 10),
        )
        cm2 = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (20, 20),
        )
        with pytest.raises(ValueError, match="image shapes"):
            CompactMask.merge([cm1, cm2])


class TestEquality:
    """Tests for CompactMask.__eq__.

    Verifies element-wise equality between two CompactMask instances and
    between a CompactMask and an equivalent dense (N, H, W) boolean array.
    """

    def test_eq_identical(self) -> None:
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[0, 2:5, 2:5] = True
        cm1 = _make_cm(masks, (10, 10))
        cm2 = _make_cm(masks, (10, 10))
        assert cm1 == cm2

    def test_eq_different(self) -> None:
        masks_a = np.zeros((2, 10, 10), dtype=bool)
        masks_a[0, 2:5, 2:5] = True
        masks_b = np.zeros((2, 10, 10), dtype=bool)
        masks_b[1, 6:9, 6:9] = True
        cm1 = _make_cm(masks_a, (10, 10))
        cm2 = _make_cm(masks_b, (10, 10))
        assert not (cm1 == cm2)

    def test_eq_with_dense_array(self) -> None:
        masks = np.zeros((1, 8, 8), dtype=bool)
        masks[0, 1:4, 1:4] = True
        cm = _make_cm(masks, (8, 8))
        assert cm == masks


class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs.

    Covers: zero-area bounding box (x1 == x2), masks that reach the image
    edge, xyxy values beyond image dimensions (clamped silently), empty
    CompactMask (N=0), sum axis compatibility with area, and with_offset for
    use by InferenceSlicer.
    """

    def test_zero_area_mask_clipped_to_1x1(self) -> None:
        """An invalid bounding box should not crash from_dense."""
        masks = np.zeros((1, 10, 10), dtype=bool)
        xyxy = np.array([[6, 5, 5, 8]], dtype=np.float32)
        with DoesNotRaise():
            cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
        assert len(cm) == 1

    def test_mask_at_image_boundary(self) -> None:
        img_h, img_w = 20, 20
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 15:20, 15:20] = True
        xyxy = np.array([[15, 15, 19, 19]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        np.testing.assert_array_equal(cm.to_dense(), masks)

    def test_xyxy_beyond_image_clipped(self) -> None:
        """xyxy values beyond the image boundary should be clipped silently."""
        img_h, img_w = 10, 10
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 5:10, 5:10] = True
        xyxy = np.array([[5, 5, 999, 999]], dtype=np.float32)
        with DoesNotRaise():
            cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        np.testing.assert_array_equal(cm.to_dense(), masks)

    def test_empty_compact_mask_to_dense(self) -> None:
        cm = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (50, 60),
        )
        dense = cm.to_dense()
        assert dense.shape == (0, 50, 60)
        assert dense.dtype == bool

    def test_sum_axis_1_2_equals_area(self) -> None:
        rng = np.random.default_rng(11)
        masks = rng.integers(0, 2, size=(4, 15, 15)).astype(bool)
        cm = _make_cm(masks, (15, 15))
        np.testing.assert_array_equal(cm.sum(axis=(1, 2)), cm.area)

    def test_with_offset(self) -> None:
        img_h, img_w = 20, 20
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 5:10, 5:10] = True
        xyxy = np.array([[5, 5, 9, 9]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        cm2 = cm.with_offset(100, 200, new_image_shape=(400, 400))
        assert cm2.offsets[0].tolist() == [105, 205]
        assert cm2._image_shape == (400, 400)
        np.testing.assert_array_equal(cm2.crop(0), cm.crop(0))

    def test_with_offset_clips_partial_overlap_like_move_masks(self) -> None:
        """with_offset must clip partial out-of-frame translations like move_masks."""
        img_h, img_w = 10, 10
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 2:6, 3:8] = True
        xyxy = np.array([[3, 2, 7, 5]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        dx, dy = -4, 3
        cm_shifted = cm.with_offset(dx=dx, dy=dy, new_image_shape=(img_h, img_w))
        expected = move_masks(
            masks=masks,
            offset=np.array([dx, dy], dtype=np.int32),
            resolution_wh=(img_w, img_h),
        )

        np.testing.assert_array_equal(cm_shifted.to_dense(), expected)

    def test_with_offset_clips_full_outside_like_move_masks(self) -> None:
        """Masks shifted fully outside should remain valid and decode to all-False."""
        img_h, img_w = 10, 10
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 2:6, 2:6] = True
        xyxy = np.array([[2, 2, 5, 5]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        dx, dy = 100, 100
        cm_shifted = cm.with_offset(dx=dx, dy=dy, new_image_shape=(img_h, img_w))
        expected = move_masks(
            masks=masks,
            offset=np.array([dx, dy], dtype=np.int32),
            resolution_wh=(img_w, img_h),
        )

        np.testing.assert_array_equal(cm_shifted.to_dense(), expected)

    def test_repack_tightens_loose_bbox(self) -> None:
        """repack() shrinks the crop to the minimal True-pixel rectangle."""
        img_h, img_w = 20, 20
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 5:10, 6:12] = True  # True block at (5,6)-(9,11)

        # Deliberately loose bbox covers full image.
        xyxy = np.array([[0, 0, img_w - 1, img_h - 1]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        # Before repack: crop is the full 20x20 image.
        assert cm._crop_shapes[0].tolist() == [20, 20]

        repacked = cm.repack()

        # After repack: crop is exactly the True block.
        assert repacked.offsets[0].tolist() == [6, 5]  # (x1, y1)
        assert repacked._crop_shapes[0].tolist() == [5, 6]  # (h, w)
        # Pixel content must be identical to the original.
        np.testing.assert_array_equal(repacked.to_dense(), masks)

    def test_repack_preserves_all_false_mask(self) -> None:
        """repack() normalises an all-False mask to a 1x1 crop."""
        img_h, img_w = 10, 10
        masks = np.zeros((2, img_h, img_w), dtype=bool)
        masks[1, 3:6, 3:6] = True  # only mask 1 is non-empty

        xyxy = np.array([[0, 0, 9, 9], [0, 0, 9, 9]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        repacked = cm.repack()

        assert repacked._crop_shapes[0].tolist() == [1, 1]  # normalised
        assert repacked._crop_shapes[1].tolist() == [3, 3]  # tight True block
        np.testing.assert_array_equal(repacked.to_dense(), masks)

    def test_repack_empty_collection(self) -> None:
        """repack() on an empty CompactMask returns another empty CompactMask."""
        cm = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (10, 10),
        )
        repacked = cm.repack()
        assert len(repacked) == 0
        assert repacked._image_shape == (10, 10)

    def test_repack_already_tight(self) -> None:
        """repack() is a no-op when bboxes are already tight."""
        img_h, img_w = 15, 15
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        masks[0, 4:9, 3:8] = True

        # Tight bbox.
        xyxy = np.array([[3, 4, 7, 8]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        repacked = cm.repack()

        np.testing.assert_array_equal(repacked.offsets, cm.offsets)
        np.testing.assert_array_equal(repacked._crop_shapes, cm._crop_shapes)
        np.testing.assert_array_equal(repacked.to_dense(), masks)


class TestCalculateMasksCentroidsCompact:
    """Verify calculate_masks_centroids gives identical results for CompactMask.

    The function has a dedicated CompactMask branch that computes centroids
    per-crop.  Results must match the dense path to within integer rounding.
    """

    def test_centroids_compact_matches_dense(self) -> None:
        """Centroid coordinates must be numerically identical for dense and compact."""
        rng = np.random.default_rng(42)
        img_h, img_w = 30, 30
        masks = rng.integers(0, 2, size=(5, img_h, img_w)).astype(bool)
        # Ensure each mask has at least one True pixel.
        for mask_idx in range(5):
            masks[mask_idx, mask_idx * 5, mask_idx * 5] = True

        cm = _make_cm(masks, (img_h, img_w))

        centroids_dense = calculate_masks_centroids(masks)
        centroids_compact = calculate_masks_centroids(cm)

        np.testing.assert_array_equal(centroids_compact, centroids_dense)

    def test_centroids_empty_mask(self) -> None:
        """All-zero masks should return centroid (0, 0) — same as dense."""
        img_h, img_w = 10, 10
        masks = np.zeros((3, img_h, img_w), dtype=bool)
        cm = _make_cm(masks, (img_h, img_w))

        centroids_dense = calculate_masks_centroids(masks)
        centroids_compact = calculate_masks_centroids(cm)

        np.testing.assert_array_equal(centroids_compact, centroids_dense)

    def test_centroids_empty_mask_with_tight_bbox(self) -> None:
        """All-zero tight crops must still return centroid (0, 0)."""
        img_h, img_w = 10, 10
        masks = np.zeros((1, img_h, img_w), dtype=bool)
        xyxy = np.array([[3, 4, 7, 8]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        centroids_dense = calculate_masks_centroids(masks)
        centroids_compact = calculate_masks_centroids(cm)

        np.testing.assert_array_equal(centroids_compact, centroids_dense)

    def test_centroids_zero_masks_returns_empty(self) -> None:
        """Empty CompactMask (0 objects) must return shape (0, 2)."""
        empty_cm = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (10, 10),
        )
        result = calculate_masks_centroids(empty_cm)
        assert result.shape == (0, 2)


class TestContainsHolesCompact:
    """Verify contains_holes result is unchanged after CompactMask roundtrip.

    contains_holes works on a 2D boolean mask.  Encoding then decoding via
    CompactMask must preserve pixel topology so that the function returns
    the same result as on the original array.
    """

    @pytest.mark.parametrize(
        ("mask_2d", "expected"),
        [
            # simple foreground blob — no holes
            (
                np.array(
                    [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
                    dtype=bool,
                ),
                False,
            ),
            # ring shape — has one hole
            (
                np.array(
                    [[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                    dtype=bool,
                ),
                True,
            ),
            # all-False — no holes
            (np.zeros((6, 6), dtype=bool), False),
            # all-True — no holes
            (np.ones((6, 6), dtype=bool), False),
        ],
    )
    def test_contains_holes_compact_roundtrip(
        self, mask_2d: np.ndarray, expected: bool
    ) -> None:
        """contains_holes must agree after CompactMask encode→decode."""
        img_h, img_w = mask_2d.shape
        masks = mask_2d[np.newaxis]  # (1, H, W)
        cm = _make_cm(masks, (img_h, img_w))

        decoded = cm.to_dense()[0]
        assert contains_holes(decoded) == expected
        assert contains_holes(decoded) == contains_holes(mask_2d)


class TestContainsMultipleSegmentsCompact:
    """Verify contains_multiple_segments result survives CompactMask roundtrip.

    Encoding and decoding must preserve connected-component topology so
    that the multi-segment predicate returns the same value.
    """

    @pytest.mark.parametrize(
        ("mask_2d", "connectivity", "expected"),
        [
            # single contiguous blob — not multi-segment
            (
                np.array(
                    [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
                    dtype=bool,
                ),
                4,
                False,
            ),
            # two separate blobs — multi-segment
            (
                np.array(
                    [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
                    dtype=bool,
                ),
                4,
                True,
            ),
            # diagonal touch — single segment under 8-connectivity
            (
                np.array(
                    [[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]],
                    dtype=bool,
                ),
                8,
                False,
            ),
            # all-False — not multi-segment
            (np.zeros((6, 6), dtype=bool), 4, False),
        ],
    )
    def test_contains_multiple_segments_compact_roundtrip(
        self, mask_2d: np.ndarray, connectivity: int, expected: bool
    ) -> None:
        """contains_multiple_segments must agree after CompactMask encode→decode."""
        img_h, img_w = mask_2d.shape
        masks = mask_2d[np.newaxis]  # (1, H, W)
        cm = _make_cm(masks, (img_h, img_w))

        decoded = cm.to_dense()[0]
        result = contains_multiple_segments(decoded, connectivity=connectivity)
        assert result == expected
        assert result == contains_multiple_segments(mask_2d, connectivity=connectivity)


# ---------------------------------------------------------------------------
# Random scenario helpers
# ---------------------------------------------------------------------------

# Varying (N, image_h, image_w) combinations for random tests.
_RANDOM_CONFIGS = [
    (1, 50, 50),
    (5, 50, 50),
    (5, 200, 300),
    (20, 100, 150),
    (20, 200, 300),
    (50, 50, 50),
    (5, 1080, 1920),
    (1, 1080, 1920),
    (20, 480, 640),
    (50, 100, 100),
]


def _random_masks_and_xyxy(
    rng: np.random.Generator,
    num_masks: int,
    img_h: int,
    img_w: int,
    fill_prob: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate *num_masks* random boolean masks with matching tight xyxy boxes.

    Each mask is built by filling a random sub-rectangle with Bernoulli noise at
    ``fill_prob``, then computing tight bounding boxes via ``mask_to_xyxy``.
    This guarantees every mask has at least one True pixel (for non-degenerate
    bounding boxes).
    """
    masks = np.zeros((num_masks, img_h, img_w), dtype=bool)
    for mask_idx in range(num_masks):
        y1 = rng.integers(0, img_h)
        y2 = rng.integers(y1, img_h)
        x1 = rng.integers(0, img_w)
        x2 = rng.integers(x1, img_w)
        region = rng.random((y2 - y1 + 1, x2 - x1 + 1)) < fill_prob
        # Ensure at least one True pixel.
        if not region.any():
            region[0, 0] = True
        masks[mask_idx, y1 : y2 + 1, x1 : x2 + 1] = region

    xyxy = mask_to_xyxy(masks).astype(np.float32)
    return masks, xyxy


class TestCompactMaskRoundtripRandom:
    """from_dense -> to_dense pixel equality across 10 random seeds.

    Uses tight bounding boxes so the round-trip must be lossless (all True
    pixels lie strictly within the crop).
    """

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_parity_seed(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        np.testing.assert_array_equal(
            cm.to_dense(),
            masks,
            err_msg=(
                f"Round-trip failed for seed={seed}, "
                f"N={num_masks}, shape=({img_h},{img_w})"
            ),
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_shape_and_len(self, seed: int) -> None:
        """len() and .shape must agree with the dense array."""
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        assert len(cm) == num_masks
        assert cm.shape == (num_masks, img_h, img_w)

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_individual_mask_access(self, seed: int) -> None:
        """cm[i] must equal masks[i] for every index."""
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        for mask_idx in range(num_masks):
            np.testing.assert_array_equal(
                cm[mask_idx],
                masks[mask_idx],
                err_msg=f"cm[{mask_idx}] mismatch for seed={seed}",
            )


class TestCompactMaskAreaRandom:
    """area from CompactMask equals dense .sum(axis=(1,2)) across 10 seeds."""

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_parity_seed(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        expected_area = masks.sum(axis=(1, 2))
        np.testing.assert_array_equal(
            cm.area,
            expected_area,
            err_msg=(
                f"Area mismatch for seed={seed}, N={num_masks}, shape=({img_h},{img_w})"
            ),
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_sum_axis_matches_area(self, seed: int) -> None:
        """cm.sum(axis=(1,2)) must equal cm.area (the fast path)."""
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))
        np.testing.assert_array_equal(cm.sum(axis=(1, 2)), cm.area)


class TestCompactMaskFilterRandom:
    """Boolean filter on CompactMask matches dense fancy indexing across 10 seeds."""

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_parity_seed(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        selector = rng.random(num_masks) > 0.5
        # Guarantee at least one True in the selector so we test non-empty subsets.
        if not selector.any():
            selector[0] = True

        subset_cm = cm[selector]
        subset_dense = masks[selector]

        assert isinstance(subset_cm, CompactMask)
        assert len(subset_cm) == int(selector.sum())
        np.testing.assert_array_equal(
            subset_cm.to_dense(),
            subset_dense,
            err_msg=f"Boolean filter mismatch for seed={seed}",
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_list_index(self, seed: int) -> None:
        """Integer list indexing must match dense fancy indexing."""
        rng = np.random.default_rng(seed)
        num_masks, img_h, img_w = _RANDOM_CONFIGS[seed]
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        num_selected = min(num_masks, max(1, rng.integers(1, num_masks + 1)))
        indices = sorted(
            rng.choice(num_masks, size=num_selected, replace=False).tolist()
        )

        subset_cm = cm[indices]
        subset_dense = masks[indices]
        np.testing.assert_array_equal(
            subset_cm.to_dense(),
            subset_dense,
            err_msg=f"List index mismatch for seed={seed}, indices={indices}",
        )


class TestCompactMaskWithOffsetRandom:
    """with_offset roundtrip matches move_masks across 10 random seeds."""

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_parity_seed(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Use smaller images to keep move_masks fast.
        num_masks = rng.integers(1, 10)
        img_h, img_w = int(rng.integers(30, 80)), int(rng.integers(30, 80))
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        # Random offset that may push some masks partially or fully off-frame.
        dx = int(rng.integers(-img_w, img_w))
        dy = int(rng.integers(-img_h, img_h))

        cm_shifted = cm.with_offset(dx=dx, dy=dy, new_image_shape=(img_h, img_w))
        expected = move_masks(
            masks=masks,
            offset=np.array([dx, dy], dtype=np.int32),
            resolution_wh=(img_w, img_h),
        )

        np.testing.assert_array_equal(
            cm_shifted.to_dense(),
            expected,
            err_msg=(
                f"with_offset mismatch for seed={seed}, "
                f"dx={dx}, dy={dy}, shape=({img_h},{img_w})"
            ),
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_offset_into_larger_canvas(self, seed: int) -> None:
        """Offset into a larger destination image must preserve pixels."""
        rng = np.random.default_rng(seed + 100)
        num_masks = rng.integers(1, 8)
        img_h, img_w = int(rng.integers(20, 50)), int(rng.integers(20, 50))
        masks, xyxy = _random_masks_and_xyxy(rng, num_masks, img_h, img_w)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(img_h, img_w))

        new_h, new_w = img_h * 2, img_w * 2
        dx = int(rng.integers(0, img_w))
        dy = int(rng.integers(0, img_h))

        cm_shifted = cm.with_offset(dx=dx, dy=dy, new_image_shape=(new_h, new_w))
        dense_shifted = cm_shifted.to_dense()

        assert dense_shifted.shape == (num_masks, new_h, new_w)
        # Manually place each original mask into the larger canvas.
        expected = np.zeros((num_masks, new_h, new_w), dtype=bool)
        for mask_idx in range(num_masks):
            expected[mask_idx, dy : dy + img_h, dx : dx + img_w] |= masks[mask_idx]

        np.testing.assert_array_equal(
            dense_shifted,
            expected,
            err_msg=f"Larger canvas offset mismatch for seed={seed}",
        )
