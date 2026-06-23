"""Integration tests: CompactMask <-> Detections, annotators, merge."""

from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

import supervision as sv
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections


def _full_xyxy(n: int, h: int, w: int) -> np.ndarray:
    """N boxes covering the whole image (ensures crop == full mask)."""
    return np.tile(np.array([0, 0, w, h], dtype=np.float32), (n, 1))


def _make_compact_detections(
    n: int, h: int = 40, w: int = 40
) -> tuple[Detections, np.ndarray]:
    """Detections with a CompactMask backed by full-image bounding boxes.

    Using full-image xyxy means all True pixels are within the crop region,
    so from_dense -> to_dense is lossless.
    """
    rng = np.random.default_rng(42)
    masks = rng.integers(0, 2, size=(n, h, w)).astype(bool)
    xyxy = _full_xyxy(n, h, w)
    cm = CompactMask.from_dense(masks, xyxy, image_shape=(h, w))
    det = Detections(
        xyxy=xyxy,
        mask=cm,
        confidence=np.ones(n, dtype=np.float32) * 0.9,
        class_id=np.arange(n),
    )
    return det, masks


class TestConstruction:
    """Tests for building Detections with a CompactMask.

    Verifies that a CompactMask is accepted as a valid mask argument and that
    the validator raises ValueError when the mask length does not match the
    number of bounding boxes.
    """

    def test_detections_construction_with_compact_mask(self) -> None:
        with DoesNotRaise():
            det, _ = _make_compact_detections(3)
        assert isinstance(det.mask, CompactMask)
        assert len(det) == 3

    def test_detections_compact_mask_validation_mismatch(self) -> None:
        n, h, w = 3, 20, 20
        xyxy = _full_xyxy(n, h, w)
        masks_wrong_n = np.zeros((n + 1, h, w), dtype=bool)
        cm = CompactMask.from_dense(masks_wrong_n, _full_xyxy(n + 1, h, w), (h, w))
        with pytest.raises(ValueError, match="mask must contain"):
            Detections(xyxy=xyxy, mask=cm)


class TestFiltering:
    """Tests for Detections.__getitem__ with a CompactMask.

    Verifies that integer, slice, and boolean-array indexing all preserve the
    CompactMask type and return the correct subset of masks.
    """

    def test_int_wraps_to_compact_mask(self) -> None:
        det, _ = _make_compact_detections(3)
        # Detections converts int to [int] internally -> subset has 1 element
        subset = det[1]
        assert isinstance(subset.mask, CompactMask)
        assert len(subset) == 1

    def test_slice_preserves_compact_mask(self) -> None:
        det, masks = _make_compact_detections(4)
        subset = det[1:3]
        assert isinstance(subset.mask, CompactMask)
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.mask.to_dense(), masks[1:3])

    def test_bool_array_preserves_compact_mask(self) -> None:
        det, masks = _make_compact_detections(4)
        selector = np.array([True, False, True, False])
        subset = det[selector]
        assert isinstance(subset.mask, CompactMask)
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.mask.to_dense(), masks[[0, 2]])


class TestIteration:
    """Tests for iterating over Detections with a CompactMask.

    Verifies that each iteration step yields a 2-D boolean (H, W) array
    identical to the corresponding dense mask, so downstream code that
    iterates over detections needs no changes.
    """

    def test_iter_yields_2d_dense(self) -> None:
        h, w = 20, 20
        det, masks = _make_compact_detections(3, h, w)
        for i, (_, mask_2d, *_) in enumerate(det):
            assert mask_2d is not None
            assert isinstance(mask_2d, np.ndarray)
            assert mask_2d.shape == (h, w)
            assert mask_2d.dtype == bool
            np.testing.assert_array_equal(mask_2d, masks[i])


class TestEquality:
    """Tests for Detections.__eq__ mixing CompactMask and dense arrays.

    Verifies that a Detections object backed by a CompactMask compares equal
    to an otherwise identical Detections object backed by a dense ndarray.
    """

    def test_compact_vs_dense(self) -> None:
        h, w = 20, 20
        det_compact, masks = _make_compact_detections(2, h, w)
        xyxy = det_compact.xyxy.copy()
        det_dense = Detections(
            xyxy=xyxy,
            mask=masks,
            confidence=np.ones(2, dtype=np.float32) * 0.9,
            class_id=np.arange(2),
        )
        assert det_compact == det_dense


class TestArea:
    """Tests for the Detections.area property with a CompactMask.

    Verifies that the fast CompactMask path in Detections.area returns the
    same per-detection pixel counts as summing the equivalent dense array.
    """

    def test_compact_matches_dense(self) -> None:
        det_compact, masks = _make_compact_detections(3)
        expected_area = np.array([m.sum() for m in masks])
        np.testing.assert_array_equal(det_compact.area, expected_area)


class TestMerge:
    """Tests for merging Detections objects that contain CompactMask instances.

    Covers three scenarios:
    - All-compact merge: result is a CompactMask.
    - Mixed compact + dense: result falls back to a dense ndarray.
    - Inner pair merge (merge_inner_detection_object_pair): used during NMS-like
      operations, each input must contain exactly one detection.
    """

    def test_all_compact(self) -> None:
        h, w = 30, 30
        det1, masks1 = _make_compact_detections(2, h, w)

        rng = np.random.default_rng(7)
        masks2 = rng.integers(0, 2, size=(3, h, w)).astype(bool)
        xyxy2 = _full_xyxy(3, h, w)
        cm2 = CompactMask.from_dense(masks2, xyxy2, (h, w))
        det2 = Detections(
            xyxy=xyxy2,
            mask=cm2,
            confidence=np.ones(3, dtype=np.float32) * 0.8,
            class_id=np.arange(3),
        )

        merged = Detections.merge([det1, det2])
        assert isinstance(merged.mask, CompactMask)
        assert len(merged) == 5
        expected = np.concatenate([masks1, masks2], axis=0)
        np.testing.assert_array_equal(merged.mask.to_dense(), expected)

    def test_mixed_compact_and_dense(self) -> None:
        """Merging a CompactMask with a dense ndarray falls back to dense."""
        h, w = 20, 20
        det_compact, _ = _make_compact_detections(2, h, w)
        masks_dense = np.zeros((1, h, w), dtype=bool)
        xyxy_dense = _full_xyxy(1, h, w)
        det_dense = Detections(
            xyxy=xyxy_dense,
            mask=masks_dense,
            confidence=np.array([0.5], dtype=np.float32),
            class_id=np.array([0]),
        )

        merged = Detections.merge([det_compact, det_dense])
        assert isinstance(merged.mask, np.ndarray)
        assert merged.mask.shape == (3, h, w)

    def test_inner_pair_with_compact(self) -> None:
        from supervision.detection.core import merge_inner_detection_object_pair

        h, w = 20, 20
        masks_a = np.zeros((1, h, w), dtype=bool)
        masks_a[0, 0:5, 0:5] = True
        xyxy_a = _full_xyxy(1, h, w)
        cm_a = CompactMask.from_dense(masks_a, xyxy_a, (h, w))
        det_a = Detections(
            xyxy=xyxy_a,
            mask=cm_a,
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([1]),
        )

        masks_b = np.zeros((1, h, w), dtype=bool)
        masks_b[0, 5:10, 5:10] = True
        xyxy_b = _full_xyxy(1, h, w)
        cm_b = CompactMask.from_dense(masks_b, xyxy_b, (h, w))
        det_b = Detections(
            xyxy=xyxy_b,
            mask=cm_b,
            confidence=np.array([0.7], dtype=np.float32),
            class_id=np.array([1]),
        )

        with DoesNotRaise():
            result = merge_inner_detection_object_pair(det_a, det_b)
        assert len(result) == 1


class TestAnnotators:
    """Tests for annotators that consume CompactMask via Detections.

    Verifies that MaskAnnotator and PolygonAnnotator produce pixel-identical
    output when given Detections backed by a CompactMask versus the equivalent
    dense ndarray, confirming that the annotators are transparent to the mask
    representation.
    """

    def test_mask_annotator(self) -> None:
        h, w = 40, 40
        det_compact, masks = _make_compact_detections(2, h, w)
        det_dense = Detections(
            xyxy=det_compact.xyxy.copy(),
            mask=masks,
            confidence=det_compact.confidence.copy(),
            class_id=det_compact.class_id.copy(),
        )

        image = np.zeros((h, w, 3), dtype=np.uint8)
        annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        annotated_compact = annotator.annotate(image.copy(), det_compact)
        annotated_dense = annotator.annotate(image.copy(), det_dense)

        np.testing.assert_array_equal(
            annotated_compact,
            annotated_dense,
            err_msg="MaskAnnotator output differs between CompactMask and dense mask",
        )

    def test_polygon_annotator(self) -> None:
        h, w = 40, 40
        # Use solid rectangular masks for stable polygon results.
        masks = np.zeros((2, h, w), dtype=bool)
        masks[0, 5:15, 5:15] = True
        masks[1, 20:30, 20:30] = True
        xyxy = _full_xyxy(2, h, w)
        cm = CompactMask.from_dense(masks, xyxy, (h, w))

        det_compact = Detections(xyxy=xyxy, mask=cm, class_id=np.array([0, 1]))
        det_dense = Detections(xyxy=xyxy, mask=masks, class_id=np.array([0, 1]))

        image = np.zeros((h, w, 3), dtype=np.uint8)
        annotator = sv.PolygonAnnotator(color_lookup=sv.ColorLookup.INDEX)

        annotated_compact = annotator.annotate(image.copy(), det_compact)
        annotated_dense = annotator.annotate(image.copy(), det_dense)

        np.testing.assert_array_equal(annotated_compact, annotated_dense)
