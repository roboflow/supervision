"""Integration tests for InferenceSlicer with compact_masks=True.

Verifies that with compact_masks=True:
- Masks stay as CompactMask throughout the pipeline (no dense materialisation).
- NMS is computed via RLE IoU (no resize, no dense (N,H,W) alloc).
- Final detections are pixel-identical to the compact_masks=False path.
"""

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import supervision as sv
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections


def _fake_seg_callback(tile: np.ndarray) -> Detections:
    """Return two non-overlapping segmentation detections for any tile."""
    h, w = tile.shape[:2]
    masks = np.zeros((2, h, w), dtype=bool)
    masks[0, : h // 3, : w // 3] = True
    masks[1, h // 2 :, w // 2 :] = True
    xyxy = np.array([[0, 0, w // 3, h // 3], [w // 2, h // 2, w, h]], dtype=np.float32)
    return Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 1]),
    )


class TestInferenceSlicerCompactMasks:
    """Tests that compact_masks=True keeps masks in RLE form end-to-end.

    The pipeline inside InferenceSlicer goes:
      callback → CompactMask.from_dense (tile coords)
               → with_offset (full-image coords)
               → CompactMask.merge (all tiles)
               → mask_non_max_suppression → compact_mask_iou_batch (RLE IoU)

    None of those steps materialise a full (N, H, W) dense array.
    """

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("compact_masks", [False, True])
    def test_reuses_dense_callback_buffers(
        self, batch_size: int, compact_masks: bool
    ) -> None:
        """A callback can reset its retained dense masks on every tile or batch."""
        image = np.zeros((4, 16, 3), dtype=np.uint8)
        for tile_index in range(4):
            image[:, tile_index * 4 : (tile_index + 1) * 4] = tile_index
        buffers = [
            Detections(
                xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
                mask=np.zeros((1, 4, 4), dtype=bool),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
            )
            for _ in range(batch_size)
        ]
        source_masks = [buffer.mask for buffer in buffers]

        def fill_buffer(tile: npt.NDArray[np.generic], index: int) -> Detections:
            """Reset one reusable mask and mark the pixel selected by the tile."""
            detection = buffers[index]
            mask = cast(npt.NDArray[np.bool_], detection.mask)
            mask.fill(False)
            mask[0, 0, int(tile[0, 0, 0])] = True
            return detection

        def callback(tile: npt.NDArray[np.generic]) -> Detections:
            """Return the same output buffer for each tile."""
            return fill_buffer(tile, 0)

        def batch_callback(tiles: list[npt.NDArray[np.generic]]) -> list[Detections]:
            """Return the same output buffers for each batch."""
            for index, tile in enumerate(tiles):
                fill_buffer(tile, index)
            return buffers

        slicer = sv.InferenceSlicer(
            callback=callback if batch_size == 1 else batch_callback,
            slice_wh=4,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=compact_masks,
            batch_size=batch_size,
        )
        expected_masks = np.zeros((4, 4, 16), dtype=bool)
        expected_masks[np.arange(4), 0, np.arange(4) * 5] = True

        result = slicer(image)

        np.testing.assert_array_equal(np.asarray(result.mask), expected_masks)
        for buffer, source_mask in zip(buffers, source_masks):
            assert buffer.mask is source_mask

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("thread_workers", [1, 2])
    def test_preserves_callback_mask_objects(
        self, batch_size: int, thread_workers: int
    ) -> None:
        """Compact output leaves each retained callback result's dense mask intact."""
        retained: list[tuple[Detections, npt.NDArray[np.bool_]]] = []

        def callback(tile: npt.NDArray[np.generic]) -> Detections:
            """Retain each dense callback result and its original mask."""
            detections = _fake_seg_callback(tile)
            assert isinstance(detections.mask, np.ndarray)
            retained.append((detections, detections.mask))
            return detections

        def batch_callback(tiles: list[npt.NDArray[np.generic]]) -> list[Detections]:
            """Retain each result from the batch callback."""
            return [callback(tile) for tile in tiles]

        slicer = sv.InferenceSlicer(
            callback=callback if batch_size == 1 else batch_callback,
            slice_wh=4,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=True,
            batch_size=batch_size,
            thread_workers=thread_workers,
        )

        result = slicer(np.zeros((8, 8, 3), dtype=np.uint8))

        assert isinstance(result.mask, CompactMask)
        assert len(retained) == 4
        for detections, source_mask in retained:
            assert detections.mask is source_mask

    def test_compact_masks_flag_converts_dense_to_compact(self) -> None:
        """Masks returned from callback are CompactMask after _run_callback."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        slicer = sv.InferenceSlicer(
            callback=_fake_seg_callback,
            slice_wh=200,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=True,
        )
        result = slicer(image)
        assert isinstance(result.mask, CompactMask), (
            f"compact_masks=True must produce a CompactMask, got {type(result.mask)}"
        )

    def test_compact_masks_false_keeps_dense(self) -> None:
        """Default (compact_masks=False) keeps dense ndarray masks."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        slicer = sv.InferenceSlicer(
            callback=_fake_seg_callback,
            slice_wh=200,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=False,
        )
        result = slicer(image)
        assert isinstance(result.mask, np.ndarray)
        assert not isinstance(result.mask, CompactMask)

    def test_compact_and_dense_pipelines_give_same_masks(self) -> None:
        """compact_masks=True and False must produce pixel-identical final masks."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)

        slicer_dense = sv.InferenceSlicer(
            callback=_fake_seg_callback,
            slice_wh=150,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.3,
            compact_masks=False,
        )
        slicer_compact = sv.InferenceSlicer(
            callback=_fake_seg_callback,
            slice_wh=150,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.3,
            compact_masks=True,
        )

        det_dense = slicer_dense(image)
        det_compact = slicer_compact(image)

        assert len(det_dense) == len(det_compact)

        dense_masks = det_dense.mask
        assert dense_masks is not None
        compact_masks_arr = np.asarray(det_compact.mask)

        # Sort both by xyxy to align order (NMS order may differ).
        def _sort_key(d: Detections) -> np.ndarray:
            return d.xyxy[:, 0] * 10000 + d.xyxy[:, 1]

        order_d = np.argsort(_sort_key(det_dense))
        order_c = np.argsort(_sort_key(det_compact))

        np.testing.assert_array_equal(
            dense_masks[order_d],
            compact_masks_arr[order_c],
            err_msg="compact_masks pipeline produced different mask pixels than dense",
        )

    def test_compact_masks_preserve_pixels_outside_detector_box(self) -> None:
        """compact_masks=True crops to the full tile, so mask pixels outside the
        detection xyxy box (but inside the tile) are preserved."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        def callback(tile: np.ndarray) -> Detections:
            h, w = tile.shape[:2]
            masks = np.zeros((1, h, w), dtype=bool)
            masks[0, 0, 0] = True
            masks[0, h - 1, w - 1] = True
            return Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                mask=masks,
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0]),
            )

        dense = sv.InferenceSlicer(
            callback=callback,
            slice_wh=100,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=False,
        )(image)
        compact = sv.InferenceSlicer(
            callback=callback,
            slice_wh=100,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=True,
        )(image)

        assert isinstance(compact.mask, CompactMask)
        np.testing.assert_array_equal(compact.mask.to_dense(), dense.mask)

    def test_nms_with_overlapping_tiles_uses_rle_iou(self) -> None:
        """With overlapping tiles, NMS must suppress duplicates using RLE IoU."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)

        call_count = 0

        def counting_callback(tile: np.ndarray) -> Detections:
            nonlocal call_count
            call_count += 1
            return _fake_seg_callback(tile)

        slicer = sv.InferenceSlicer(
            callback=counting_callback,
            slice_wh=200,
            overlap_wh=100,  # heavy overlap → many duplicate detections
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.3,
            compact_masks=True,
        )
        result = slicer(image)

        assert call_count > 1, "Should have run on multiple tiles"
        assert isinstance(result.mask, CompactMask), (
            "Result mask must remain CompactMask after cross-tile NMS"
        )

    def test_no_mask_callback_unaffected(self) -> None:
        """compact_masks=True must not crash when callback returns no masks."""

        def box_only_callback(tile: np.ndarray) -> Detections:
            h, w = tile.shape[:2]
            return Detections(
                xyxy=np.array([[0, 0, w // 2, h // 2]], dtype=np.float32),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
            )

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        slicer = sv.InferenceSlicer(
            callback=box_only_callback,
            slice_wh=200,
            overlap_wh=0,
            overlap_filter=sv.OverlapFilter.NONE,
            compact_masks=True,
        )
        result = slicer(image)
        assert result.mask is None
