"""Exact compact unions and NMM without full-frame mask materialization."""

from unittest.mock import patch

import numpy as np
import pytest

from supervision import Detections, InferenceSlicer, OverlapFilter, OverlapMetric
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import _merge_detection_group
from supervision.detection.utils.converters import mask_to_xyxy


class TestCompactMaskUnion:
    """Union crop RLEs without decoding either crops or image-sized arrays."""

    @pytest.mark.parametrize("count", [0, 1, 7])
    @pytest.mark.parametrize("density", [0.0, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("loose_crops", [False, True])
    def test_matches_dense_reference(
        self, count: int, density: float, loose_crops: bool
    ) -> None:
        """Empty, solid and fragmented masks agree with the NumPy union."""
        rng = np.random.default_rng(104)
        dense = np.zeros((count, 31, 43), dtype=bool)
        for index in range(count):
            y, x = rng.integers(0, 20, size=2)
            dense[index, y : y + 11, x : x + 17] = rng.random((11, 17)) < density
        boxes = mask_to_xyxy(dense)
        if loose_crops:
            boxes = np.tile([0, 0, 42, 30], (count, 1))
        compact = CompactMask.from_dense(dense, boxes, (31, 43))
        original_rles = [rle.copy() for rle in compact._rles]
        expected = np.any(dense, axis=0, keepdims=True)

        with (
            patch.object(
                CompactMask, "to_dense", side_effect=AssertionError("full image")
            ),
            patch.object(
                CompactMask, "crop", side_effect=AssertionError("crop decode")
            ),
        ):
            result = compact._union()

        np.testing.assert_array_equal(result.to_dense(), expected)
        assert result.image_shape == (31, 43)
        assert len(result) == 1
        assert result._rles[0].dtype == np.int32
        for original, current in zip(original_rles, compact._rles):
            np.testing.assert_array_equal(current, original)
        if expected.any():
            np.testing.assert_array_equal(result.bbox_xyxy, mask_to_xyxy(expected))

    def test_encodes_distant_pixels_with_long_background_runs(self) -> None:
        """A union crop larger than int32 stays sparse and preserves RLE parity."""
        compact = CompactMask(
            rles=[np.array([0, 1], dtype=np.int32)] * 2,
            crop_shapes=np.ones((2, 2), dtype=np.int32),
            offsets=np.array([[0, 0], [99_999, 99_999]], dtype=np.int32),
            image_shape=(100_000, 100_000),
        )

        with (
            patch.object(
                CompactMask, "to_dense", side_effect=AssertionError("full image")
            ),
            patch.object(
                CompactMask, "crop", side_effect=AssertionError("crop decode")
            ),
        ):
            result = compact._union()

        rle = result._rles[0]
        assert rle.dtype == np.int32
        assert np.all(rle >= 0)
        assert int(rle.sum(dtype=np.int64)) == 10_000_000_000
        assert result.area.tolist() == [2]
        assert len(rle) < 20
        ends = np.cumsum(rle, dtype=np.int64)
        nonempty_foreground = (np.arange(len(rle)) % 2 == 1) & (rle > 0)
        np.testing.assert_array_equal(ends[nonempty_foreground], [1, 10_000_000_000])

    def test_splits_a_foreground_run_larger_than_int32(self) -> None:
        """Adjacent solid crops form a huge foreground run without pixel allocation."""
        compact = CompactMask(
            rles=[np.array([0, 1_250_000_000], dtype=np.int32)] * 2,
            crop_shapes=np.array([[50_000, 25_000]] * 2, dtype=np.int32),
            offsets=np.array([[0, 0], [25_000, 0]], dtype=np.int32),
            image_shape=(50_000, 50_000),
        )

        with (
            patch.object(
                CompactMask, "to_dense", side_effect=AssertionError("full image")
            ),
            patch.object(
                CompactMask, "crop", side_effect=AssertionError("crop decode")
            ),
        ):
            result = compact._union()

        assert result.area.tolist() == [2_500_000_000]
        assert result._rles[0].tolist() == [0, 2_147_483_647, 0, 352_516_353]
        assert result._crop_shapes.tolist() == [[50_000, 50_000]]


class TestCompactMaskNmmUnion:
    """Compact and dense NMM produce identical groups, masks and metadata."""

    @pytest.mark.parametrize("overlap_metric", [OverlapMetric.IOU, OverlapMetric.IOS])
    @pytest.mark.parametrize("class_agnostic", [False, True])
    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 1.0])
    def test_matches_dense_public_output(
        self, overlap_metric: OverlapMetric, class_agnostic: bool, threshold: float
    ) -> None:
        """Iterative unions preserve threshold, class, score and winner semantics."""
        dense = np.zeros((6, 14, 25), dtype=bool)
        dense[0, 2:8, 2:8] = True
        dense[1, 2:8, 5:11] = True
        dense[2, 2:8, 8:14] = True
        dense[3, 2:8, 5:11] = True
        dense[4, 9:12, 19:23] = True
        # Intentionally smaller detector boxes must not clip the stored masks.
        detections = Detections(
            xyxy=np.array([[3, 3, 5, 5]] * 6, dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.7, 0.8, 0.6, 0.5]),
            class_id=np.array([0, 0, 0, 1, 1, 0]),
            tracker_id=np.arange(6),
            mask=dense,
            data={"label": np.array(["a", "b", "c", "d", "e", "f"])},
            metadata={"camera": "north"},
        )
        expected = detections.with_nmm(
            threshold=threshold,
            class_agnostic=class_agnostic,
            overlap_metric=overlap_metric,
        )
        compact = detections.to_compact_masks()

        with patch.object(
            CompactMask, "to_dense", side_effect=AssertionError("full image")
        ):
            result = compact.with_nmm(
                threshold=threshold,
                class_agnostic=class_agnostic,
                overlap_metric=overlap_metric,
            )

        assert isinstance(result.mask, CompactMask)
        np.testing.assert_array_equal(result.mask.to_dense(), expected.mask)
        np.testing.assert_array_equal(result.xyxy, expected.xyxy)
        np.testing.assert_array_equal(result.class_id, expected.class_id)
        np.testing.assert_array_equal(result.tracker_id, expected.tracker_id)
        np.testing.assert_array_equal(result.confidence, expected.confidence)
        np.testing.assert_array_equal(result.data["label"], expected.data["label"])
        assert result.metadata == expected.metadata

    def test_final_detection_union_never_decodes(self) -> None:
        """Final group reduction independently avoids full-frame and crop decoding."""
        dense = np.zeros((2, 10, 12), dtype=bool)
        dense[0, 2:5, 1:6] = True
        dense[1, 3:7, 3:9] = True
        detections = Detections(
            xyxy=mask_to_xyxy(dense).astype(float),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 0]),
            mask=CompactMask.from_dense(dense, mask_to_xyxy(dense), (10, 12)),
        )

        with (
            patch.object(
                CompactMask, "to_dense", side_effect=AssertionError("full image")
            ),
            patch.object(
                CompactMask, "crop", side_effect=AssertionError("crop decode")
            ),
        ):
            result = _merge_detection_group([detections[0], detections[1]])

        assert isinstance(result.mask, CompactMask)
        np.testing.assert_array_equal(
            result.mask.to_dense(), np.any(dense, axis=0)[None]
        )

    def test_large_canvas_keeps_only_local_mask_storage(self) -> None:
        """Tiny duplicates on a 100000-square canvas merge without full images."""
        compact = CompactMask(
            rles=[np.array([0, 64], dtype=np.int32)] * 3,
            crop_shapes=np.array([[8, 8]] * 3, dtype=np.int32),
            offsets=np.array(
                [[90_000, 80_000], [90_002, 80_000], [12, 20]], dtype=np.int32
            ),
            image_shape=(100_000, 100_000),
        )
        detections = Detections(
            xyxy=compact.bbox_xyxy.astype(float),
            confidence=np.array([0.9, 0.8, 0.7]),
            class_id=np.array([0, 0, 0]),
            mask=compact,
        )

        with patch.object(
            CompactMask, "to_dense", side_effect=AssertionError("full image")
        ):
            result = detections.with_nmm(threshold=0.5)

        assert isinstance(result.mask, CompactMask)
        assert result.mask.image_shape == (100_000, 100_000)
        assert result.mask.area.tolist() == [80, 64]
        assert result.mask._crop_shapes.tolist() == [[8, 10], [8, 8]]

    def test_slicer_preserves_compact_masks_through_nmm(self) -> None:
        """Duplicate tile predictions are relocated and merged without dense images."""
        image = np.zeros((16, 24, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(24)

        def callback(tile: np.ndarray) -> Detections:
            """Detect the same world-space object in each overlapping tile."""
            origin_x = int(tile[0, 0, 0])
            mask = np.zeros((1, *tile.shape[:2]), dtype=bool)
            mask[0, 4:9, 10 - origin_x : 14 - origin_x] = True
            return Detections(
                xyxy=mask_to_xyxy(mask).astype(float),
                mask=mask,
                confidence=np.array([0.9]),
                class_id=np.array([0]),
            )

        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=(16, 16),
            overlap_wh=(8, 0),
            overlap_filter=OverlapFilter.NON_MAX_MERGE,
            compact_masks=True,
            thread_workers=1,
        )
        expected = np.zeros((1, 16, 24), dtype=bool)
        expected[0, 4:9, 10:14] = True

        with patch.object(
            CompactMask, "to_dense", side_effect=AssertionError("full image")
        ):
            result = slicer(image)

        assert isinstance(result.mask, CompactMask)
        assert len(result) == 1
        np.testing.assert_array_equal(result.mask.to_dense(), expected)
