"""Exact compact unions and NMM without full-frame mask materialization."""

from unittest.mock import patch

import numpy as np
import pytest

from supervision import Detections, InferenceSlicer, OverlapFilter, OverlapMetric
from supervision.detection.compact_mask import (
    CompactMask,
    _compact_mask_union,
    _should_union_densely,
)
from supervision.detection.core import _merge_detection_group
from supervision.detection.utils.converters import _rle_counts_to_mask, mask_to_xyxy


class TestCompactMaskUnion:
    """Union sparse crop RLEs without decoding, matching a dense reference.

    Fragmented (checkerboard) cases in this class exercise the dense fallback, which
    does decode crop-local buffers by design; the guards below only assert that a *full-
    image-sized* array is never decoded.
    """

    @pytest.mark.parametrize("count", [0, 1, 7])
    @pytest.mark.parametrize(
        "density", [0.0, 0.1, 0.5, 1.0, pytest.param(None, id="checkerboard")]
    )
    @pytest.mark.parametrize("loose_crops", [False, True])
    @pytest.mark.parametrize(
        "block_offset",
        [
            pytest.param(None, id="random-position"),
            pytest.param((0, 0), id="top-left-edge"),
            pytest.param((0, 26), id="top-right-edge"),
            pytest.param((20, 0), id="bottom-left-edge"),
            pytest.param((20, 26), id="bottom-right-edge"),
        ],
    )
    def test_matches_dense_reference(
        self,
        count: int,
        density: float | None,
        loose_crops: bool,
        block_offset: tuple[int, int] | None,
    ) -> None:
        """Match NumPy unions for random and edge-pinned blocks in tight/loose crops."""
        rng = np.random.default_rng(104)
        dense = np.zeros((count, 31, 43), dtype=bool)
        for index in range(count):
            y, x = rng.integers(0, 20, size=2) if block_offset is None else block_offset
            if density is None:
                crop = (np.indices((11, 17)).sum(axis=0) + index) % 2 == 0
            else:
                crop = rng.random((11, 17)) < density
            dense[index, y : y + 11, x : x + 17] = crop
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
            result = _compact_mask_union([compact])

        np.testing.assert_array_equal(result.to_dense(), expected)
        assert result.image_shape == (31, 43)
        assert len(result) == 1
        assert result._rles[0].dtype == np.int32
        for original, current in zip(original_rles, compact._rles):
            np.testing.assert_array_equal(current, original)
        if expected.any():
            np.testing.assert_array_equal(result.bbox_xyxy, mask_to_xyxy(expected))

    @pytest.mark.parametrize(
        ("rectangles", "expected_crop_shapes"),
        [
            pytest.param(
                [(2, 3, 3, 4), (8, 6, 9, 7)],
                [[1, 1], [1, 1]],
                id="single-pixel-crops",
            ),
            pytest.param(
                [(2, 1, 5, 3), (4, 2, 7, 7), (1, 6, 4, 9)],
                [[2, 3], [5, 3], [3, 3]],
                id="different-crop-heights",
            ),
            pytest.param(
                [(2, 1, 5, 3), None, (4, 2, 7, 7)],
                [[2, 3], [1, 1], [5, 3]],
                id="foreground-with-all-background-mask",
            ),
        ],
    )
    def test_matches_dense_reference_for_explicit_crops(
        self,
        rectangles: list[tuple[int, int, int, int] | None],
        expected_crop_shapes: list[list[int]],
    ) -> None:
        """Match NumPy unions for single pixels, unequal heights and empty masks."""
        dense = np.zeros((len(rectangles), 10, 12), dtype=bool)
        for index, rectangle in enumerate(rectangles):
            if rectangle is not None:
                x1, y1, x2, y2 = rectangle
                dense[index, y1:y2, x1:x2] = True
        compact = CompactMask.from_dense(dense, mask_to_xyxy(dense), (10, 12))
        np.testing.assert_array_equal(compact._crop_shapes, expected_crop_shapes)
        expected = np.any(dense, axis=0, keepdims=True)

        with (
            patch.object(
                CompactMask, "to_dense", side_effect=AssertionError("full image")
            ),
            patch.object(
                CompactMask, "crop", side_effect=AssertionError("crop decode")
            ),
        ):
            result = _compact_mask_union([compact])

        np.testing.assert_array_equal(result.to_dense(), expected)
        np.testing.assert_array_equal(result.bbox_xyxy, mask_to_xyxy(expected))
        np.testing.assert_array_equal(result.area, expected.sum(axis=(1, 2)))

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
            result = _compact_mask_union([compact])

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
            result = _compact_mask_union([compact])

        assert result.area.tolist() == [2_500_000_000]
        assert result._rles[0].tolist() == [0, 2_147_483_647, 0, 352_516_353]
        assert result._crop_shapes.tolist() == [[50_000, 50_000]]

    @pytest.mark.parametrize("operation", ["to_dense", "crop", "repack"])
    @pytest.mark.parametrize(
        ("rle", "expected_crop"),
        [
            pytest.param(
                [0, 5, 0, 5, 0, 2],
                [[True] * 4] * 3,
                id="split-foreground",
            ),
            pytest.param(
                [0, 1, 5, 0, 5, 1],
                [
                    [True, False, False, False],
                    [False, False, False, False],
                    [False, False, False, True],
                ],
                id="split-background",
            ),
        ],
    )
    def test_preserves_split_run_pixels(
        self, operation: str, rle: list[int], expected_crop: list[list[bool]]
    ) -> None:
        """Zero-length separators preserve pixels across decoding and repacking."""
        compact = CompactMask(
            rles=[np.array(rle, dtype=np.int32)],
            crop_shapes=np.array([[3, 4]], dtype=np.int32),
            offsets=np.array([[2, 1]], dtype=np.int32),
            image_shape=(6, 8),
        )
        expected = np.zeros((1, 6, 8), dtype=bool)
        expected[0, 1:4, 2:6] = expected_crop
        if operation == "crop":
            expected = np.array(expected_crop, dtype=bool)

        if operation == "crop":
            result = compact.crop(0)
        elif operation == "repack":
            result = compact.repack().to_dense()
        else:
            result = compact.to_dense()

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.bool_

    @pytest.mark.parametrize(
        ("second_offset_y", "expected_rle"),
        [
            pytest.param(1, [0, 2], id="exact-touch"),
            pytest.param(2, [0, 1, 1, 1], id="one-pixel-gap"),
        ],
    )
    def test_canonicalizes_interval_boundaries(
        self, second_offset_y: int, expected_rle: list[int]
    ) -> None:
        """Merge touching intervals while retaining a real one-pixel background gap.

        Two 1x1 crops give a run-count ratio of 2.0 (four runs over two pixels), which
        would otherwise dispatch to the dense fallback and never exercise this
        function's interval boundary-merge logic at all. Pin the interval path
        explicitly so the test still means what it says.
        """
        compact = CompactMask(
            rles=[np.array([0, 1], dtype=np.int32)] * 2,
            crop_shapes=np.ones((2, 2), dtype=np.int32),
            offsets=np.array([[0, 0], [0, second_offset_y]], dtype=np.int32),
            image_shape=(3, 1),
        )

        with patch(
            "supervision.detection.compact_mask._should_union_densely",
            return_value=False,
        ):
            result = _compact_mask_union([compact])

        assert result._rles[0].tolist() == expected_rle
        assert result._crop_shapes.tolist() == [[second_offset_y + 1, 1]]


class TestUnionDenseFallback:
    """Fragmented masks dispatch to a bbox-local dense union, sparse ones to RLE."""

    @staticmethod
    def _make_compact(pattern: str, size: int) -> CompactMask:
        """Build a single (size, size) mask, either fully solid or checkerboard."""
        crop = np.ones((size, size), dtype=bool)
        if pattern == "checkerboard":
            crop = np.indices(crop.shape).sum(axis=0) % 2 == 0
        return CompactMask.from_dense(
            crop[None], np.array([[0, 0, size - 1, size - 1]]), (size, size)
        )

    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            pytest.param("solid", False, id="solid-stays-interval"),
            pytest.param("checkerboard", True, id="checkerboard-goes-dense"),
        ],
    )
    def test_should_union_densely_reads_fragmentation(
        self, pattern: str, expected: bool
    ) -> None:
        """`_should_union_densely` flags checkerboard fragmentation, not solid fill.

        Both masks fit well under the memory cap, so the run-count ratio alone
        must decide: one run for a solid crop, near one run per pixel for the
        checkerboard, at the same bbox size.
        """
        compact = self._make_compact(pattern, size=40)

        result = _should_union_densely([compact], bbox_width=40, bbox_height=40)

        assert result is expected

    def test_should_union_densely_respects_memory_cap(self) -> None:
        """A huge bbox skips the dense fallback even when fully fragmented.

        Two single-pixel masks at opposite corners of a 100000x100000 image have a run-
        count ratio of 1.0 (maximally "fragmented" by that metric alone), but the bbox
        they span is far past the memory cap, so falling back to a dense array would
        reintroduce the full-canvas materialization this union function exists to avoid.
        """
        compact = CompactMask(
            rles=[np.array([0, 1], dtype=np.int32)] * 2,
            crop_shapes=np.ones((2, 2), dtype=np.int32),
            offsets=np.array([[0, 0], [99_999, 99_999]], dtype=np.int32),
            image_shape=(100_000, 100_000),
        )

        result = _should_union_densely(
            [compact], bbox_width=100_000, bbox_height=100_000
        )

        assert result is False

    def test_dense_fallback_matches_interval_result(self) -> None:
        """Dense and interval paths agree on a checkerboard union's pixels and crop.

        Forces each path in turn via monkeypatching so a future change to the dispatch
        threshold cannot silently make both paths run the same way without anyone
        noticing the comparison stopped being meaningful. A spy on the dense path's own
        decode call proves each branch actually ran as forced, rather than both patches
        being inert and the comparison passing vacuously.
        """
        compact = self._make_compact("checkerboard", size=20)
        expected_dense = compact.to_dense()

        with (
            patch(
                "supervision.detection.compact_mask._should_union_densely",
                return_value=True,
            ),
            patch(
                "supervision.detection.compact_mask._rle_counts_to_mask",
                wraps=_rle_counts_to_mask,
            ) as decode_spy,
        ):
            via_dense = _compact_mask_union([compact])
            decode_spy.assert_called_once()
        with (
            patch(
                "supervision.detection.compact_mask._should_union_densely",
                return_value=False,
            ),
            patch(
                "supervision.detection.compact_mask._rle_counts_to_mask",
                wraps=_rle_counts_to_mask,
            ) as decode_spy,
        ):
            via_interval = _compact_mask_union([compact])
            decode_spy.assert_not_called()

        np.testing.assert_array_equal(via_dense.to_dense(), expected_dense)
        np.testing.assert_array_equal(via_interval.to_dense(), expected_dense)
        np.testing.assert_array_equal(via_dense.bbox_xyxy, via_interval.bbox_xyxy)
        assert via_dense._crop_shapes.tolist() == via_interval._crop_shapes.tolist()


class TestCompactMaskNmmUnion:
    """Compact and dense NMM produce identical groups, masks and metadata."""

    @pytest.mark.parametrize("overlap_metric", [OverlapMetric.IOU, OverlapMetric.IOS])
    @pytest.mark.parametrize("class_agnostic", [False, True])
    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 1.0])
    def test_matches_dense_public_output(
        self, overlap_metric: OverlapMetric, class_agnostic: bool, threshold: float
    ) -> None:
        """NMM preserves groups and metadata, including an all-background mask."""
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
            data={"label": np.array(["a", "b", "c", "d", "e", "all-background"])},
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
