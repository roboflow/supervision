"""Lossless reduced-grid COCO RLE parsing and compact mask export."""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import supervision as sv
from supervision import _cv2 as cv2
from supervision.detection.utils import internal


def _mask(shape: tuple[int, int], pattern: str) -> np.ndarray:
    """Build masks including full frames and foreground touching image edges."""
    height, width = shape
    if pattern == "full":
        return np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=bool)
    if pattern == "rectangle":
        mask[
            height // 4 : max(height // 4 + 1, 3 * height // 4),
            width // 4 : max(width // 4 + 1, 3 * width // 4),
        ] = True
    elif pattern == "edge":
        mask[-1, :] = True
        mask[:, -1] = True
    elif pattern == "fragmented":
        mask = np.random.default_rng(42).random(shape) > 0.7
    return mask


def _rle(mask: np.ndarray, kind: str = "str") -> dict[str, Any]:
    """Encode a fixture into one of the supported COCO count representations."""
    counts = sv.mask_to_rle(mask, compressed=kind != "list")
    if kind == "bytes":
        counts = counts.encode()
    return {"size": list(mask.shape), "counts": counts}


def _response(mask: np.ndarray, shape: tuple[int, int], kind: str) -> dict[str, Any]:
    """Wrap a mask with a detector box deliberately smaller than its foreground."""
    return {
        "image": {"height": shape[0], "width": shape[1]},
        "predictions": [
            {
                "x": 20,
                "y": 20,
                "width": 4,
                "height": 4,
                "class": "object",
                "class_id": 0,
                "confidence": 0.9,
                "rle": _rle(mask, kind),
            }
        ],
    }


class TestCompactMaskFromCocoRleResized:
    """Full-image resizing preserves source sampling and mask boundaries."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((1, 1), id="single-pixel"),
            pytest.param((3, 5), id="tiny"),
            pytest.param((12, 16), id="small"),
            pytest.param((178, 218), id="model-grid"),
        ],
    )
    @pytest.mark.parametrize(
        "pattern", ["empty", "full", "rectangle", "edge", "fragmented"]
    )
    @pytest.mark.parametrize("scale", ["same", "four", "uneven", "down"])
    def test_matches_nearest_neighbor_pixels(
        self, shape: tuple[int, int], pattern: str, scale: str
    ) -> None:
        """Identity, integer, fractional and downscaled grids retain exact pixels."""
        height, width = shape
        target = {
            "same": shape,
            "four": (4 * height, 4 * width),
            "uneven": (max(1, 3 * height - 1), 3 * width + 1),
            "down": (max(1, height // 2), max(1, width // 2)),
        }[scale]
        mask = _mask(shape, pattern)
        expected = cv2.resize(
            mask.astype(np.uint8),
            (target[1], target[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        result = sv.CompactMask.from_coco_rle_resized([_rle(mask)], image_shape=target)

        np.testing.assert_array_equal(result.to_dense()[0], expected)
        assert result.shape == (1, *target)
        np.testing.assert_array_equal(result.offsets, [[0, 0]])

    def test_accepts_empty_batch(self) -> None:
        """An empty carrier retains its declared target canvas."""
        result = sv.CompactMask.from_coco_rle_resized([], image_shape=(48, 64))

        assert result.shape == (0, 48, 64)
        assert result.to_coco_rle() == []

    def test_handles_mixed_source_grids(self) -> None:
        """Each RLE is resized from its own grid while preserving row order."""
        masks = [_mask((12, 16), "rectangle"), _mask((7, 9), "edge")]
        expected = np.stack(
            [
                cv2.resize(
                    m.astype(np.uint8), (64, 48), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for m in masks
            ]
        )

        result = sv.CompactMask.from_coco_rle_resized(
            [_rle(m) for m in masks], image_shape=(48, 64)
        )

        np.testing.assert_array_equal(result.to_dense(), expected)

    @pytest.mark.parametrize(
        ("bad", "message"),
        [
            pytest.param(
                {"size": [0, 4], "counts": [0]},
                "positive integers",
                id="zero-dimension",
            ),
            pytest.param(
                {"size": [12, 16], "counts": [-1, 193]},
                "non-negative",
                id="negative-run",
            ),
            pytest.param(
                {"size": [12, 16], "counts": [1]},
                "cover the source grid",
                id="wrong-area",
            ),
            pytest.param(
                {"size": [12, 16], "counts": []}, "cannot be empty", id="empty-counts"
            ),
            pytest.param({"size": [12, 16]}, "size and counts", id="missing-counts"),
            pytest.param(
                {"size": [1.5, 4], "counts": [6]},
                "positive integers",
                id="fractional-dimension",
            ),
            pytest.param(
                {"size": None, "counts": [6]}, "integer dimensions", id="missing-shape"
            ),
            pytest.param(
                {"size": [2, 3], "counts": [2**32 + 6]},
                "Invalid COCO RLE counts",
                id="run-overflow",
            ),
        ],
    )
    def test_rejects_invalid_rle(self, bad: dict[str, Any], message: str) -> None:
        """Malformed dimensions and counts cannot enter the resize path."""
        with pytest.raises(ValueError, match=message):
            sv.CompactMask.from_coco_rle_resized([bad], image_shape=(48, 64))

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((0, 4), id="zero"),
            pytest.param((-1, 4), id="negative"),
            pytest.param((32769, 4), id="oversized"),
            pytest.param((True, 4), id="boolean"),
        ],
    )
    def test_rejects_invalid_target(self, shape: tuple[int, int]) -> None:
        """Even empty batches validate target dimensions before construction."""
        with pytest.raises(ValueError, match="positive integers"):
            sv.CompactMask.from_coco_rle_resized([], image_shape=shape)


class TestCompactMaskToCocoRle:
    """COCO export includes crop offsets and requires no dense materialization."""

    @pytest.mark.parametrize(
        "pattern", ["empty", "full", "rectangle", "edge", "fragmented"]
    )
    @pytest.mark.parametrize(
        "box",
        [
            pytest.param([0, 0, 15, 11], id="full-frame"),
            pytest.param([3, 2, 13, 9], id="offset-crop"),
            pytest.param([15, 11, 15, 11], id="corner-pixel"),
        ],
    )
    @pytest.mark.parametrize("compressed", [True, False])
    def test_round_trips_stored_foreground(
        self, pattern: str, box: list[int], compressed: bool
    ) -> None:
        """Exported masks match stored pixels, including zeros outside the crop."""
        mask = _mask((12, 16), pattern)
        compact = sv.CompactMask.from_dense(mask[None], np.array([box]), (12, 16))
        expected = compact.to_dense()[0]

        with patch.object(
            sv.CompactMask, "to_dense", side_effect=AssertionError("dense export")
        ):
            encoded = compact.to_coco_rle(compressed=compressed)[0]

        assert encoded["size"] == [12, 16]
        np.testing.assert_array_equal(
            sv.rle_to_mask(encoded["counts"], (16, 12)), expected
        )

    def test_preserves_sliced_detection_order(self) -> None:
        """Filtering and reordering masks keeps exports aligned with detections."""
        masks = np.stack([_mask((12, 16), "rectangle"), _mask((12, 16), "edge")])
        compact = sv.CompactMask.from_coco_rle_resized(
            [_rle(m) for m in masks], image_shape=(12, 16)
        )
        selected = compact[[1, 0, 1]]

        encoded = selected.to_coco_rle()

        actual = np.stack([sv.rle_to_mask(r["counts"], (16, 12)) for r in encoded])
        np.testing.assert_array_equal(actual, masks[[1, 0, 1]])

    def test_emits_known_coco_counts(self) -> None:
        """Offset export follows COCO column-major order independently of decoding."""
        compact = sv.CompactMask(
            [np.array([0, 1], dtype=np.int32)],
            np.array([[1, 1]], dtype=np.int32),
            np.array([[2, 1]], dtype=np.int32),
            (4, 5),
        )

        encoded = compact.to_coco_rle(compressed=False)

        assert encoded == [{"size": [4, 5], "counts": [9, 1, 10]}]


class TestDetectionsFromReducedRle:
    """Compact parsing preserves masks and overlays without dense conversion."""

    @pytest.mark.parametrize("kind", ["str", "bytes", "list"])
    @pytest.mark.parametrize(
        "target",
        [
            pytest.param((48, 64), id="integer-scale"),
            pytest.param((43, 61), id="fractional-scale"),
        ],
    )
    def test_avoids_dense_materialization(
        self, kind: str, target: tuple[int, int]
    ) -> None:
        """Reduced RLE stays compact and keeps pixels outside the detector box."""
        result = _response(_mask((12, 16), "edge"), target, kind)
        expected = sv.Detections.from_inference(result)

        with (
            patch.object(
                internal, "rle_to_mask", side_effect=AssertionError("dense decode")
            ),
            patch.object(
                sv.CompactMask, "from_dense", side_effect=AssertionError("dense encode")
            ),
        ):
            actual = sv.Detections.from_inference(result, compact_masks=True)

        assert isinstance(actual.mask, sv.CompactMask)
        np.testing.assert_array_equal(actual.xyxy, expected.xyxy)
        np.testing.assert_array_equal(actual.mask.to_dense(), expected.mask)

    def test_preserves_mask_annotation(self) -> None:
        """Compact and dense masks produce identical full-frame overlays."""
        result = _response(_mask((12, 16), "edge"), (48, 64), "str")
        dense = sv.Detections.from_inference(result)
        compact = sv.Detections.from_inference(result, compact_masks=True)
        scene = np.zeros((48, 64, 3), dtype=np.uint8)
        expected = sv.MaskAnnotator().annotate(scene.copy(), dense)

        actual = sv.MaskAnnotator().annotate(scene.copy(), compact)

        np.testing.assert_array_equal(actual, expected)
