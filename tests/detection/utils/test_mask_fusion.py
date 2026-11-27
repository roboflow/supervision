"""Verify mask fusion clustering and mask painting."""

from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils._mask_fusion import (
    _FusedMask,
    _mask_weighted_fusion_clusters,
    _MaskCluster,
    _paint_fused_masks,
)
from supervision.detection.utils.iou import OverlapMetric
from supervision.detection.utils.nms import _box_weighted_fusion_clusters


def _rect_masks(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Boolean masks, one filled `(x_min, y_min, x_max, y_max)` rectangle each."""
    masks = np.zeros((len(rectangles), *shape), dtype=bool)
    for index, (x_min, y_min, x_max, y_max) in enumerate(rectangles):
        masks[index, y_min:y_max, x_min:x_max] = True
    return masks


def _boxes_of(rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    """`xyxy` boxes of the rectangles, as floats."""
    return np.array(rectangles, dtype=np.float64)


def _paint(fused: _FusedMask, shape: tuple[int, int]) -> np.ndarray:
    """Place a fused crop on an empty image."""
    image = np.zeros(shape, dtype=bool)
    height, width = fused.crop.shape
    image[fused.y : fused.y + height, fused.x : fused.x + width] = fused.crop
    return image


class TestMaskWeightedFusionClusters:
    """The mask cluster kernel mirrors the box kernel's ordering and bookkeeping."""

    def test_labels_and_scores_match_the_box_kernel_on_rectangles(self) -> None:
        """Rectangular masks cluster exactly like their boxes, across two classes."""
        rectangles = [(0, 0, 10, 10), (1, 1, 11, 11), (0, 0, 10, 10), (30, 30, 40, 40)]
        masks = _rect_masks((50, 50), rectangles)
        boxes = _boxes_of(rectangles)
        scores = np.array([0.9, 0.8, 0.9, 0.9])
        class_ids = np.array([0, 0, 1, 0])
        _, expected_scores, expected_labels = _box_weighted_fusion_clusters(
            boxes, scores, class_ids, 0.5, OverlapMetric.IOU
        )

        _, fused_scores, labels, fused_masks = _mask_weighted_fusion_clusters(
            boxes, scores, class_ids, 0.5, OverlapMetric.IOU, masks
        )

        assert np.allclose(fused_scores, expected_scores)
        assert np.array_equal(labels, expected_labels)
        assert len(fused_masks) == len(fused_scores)

    def test_single_mask_clusters_have_no_fused_mask(self) -> None:
        """Only clusters of two or more masks carry a fused mask."""
        rectangles = [(0, 0, 10, 10), (1, 0, 11, 10), (30, 30, 40, 40)]
        masks = _rect_masks((50, 50), rectangles)

        _, _, labels, fused_masks = _mask_weighted_fusion_clusters(
            _boxes_of(rectangles),
            np.array([0.9, 0.8, 0.7]),
            None,
            0.5,
            OverlapMetric.IOU,
            masks,
        )

        assert fused_masks[labels[0]] is not None
        assert fused_masks[labels[2]] is None

    @pytest.mark.parametrize("seed", range(5))
    def test_compact_masks_give_the_dense_result(self, seed: int) -> None:
        """`CompactMask` and dense inputs produce identical clusters and masks."""
        rng = np.random.default_rng(seed)
        rectangles = []
        for _ in range(8):
            x, y = rng.integers(0, 60, 2)
            size = rng.integers(15, 30, 2)
            for _jitter in range(3):
                dx, dy = rng.integers(-2, 3, 2)
                rectangles.append(
                    (
                        int(x + dx),
                        int(y + dy),
                        int(x + dx + size[0]),
                        int(y + dy + size[1]),
                    )
                )
        masks = _rect_masks((100, 100), rectangles)
        boxes = _boxes_of(rectangles)
        scores = rng.uniform(0.1, 1.0, len(rectangles))
        compact = CompactMask.from_dense(masks, boxes - [0, 0, 1, 1], (100, 100))

        dense_result = _mask_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU, masks
        )
        compact_result = _mask_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU, compact
        )

        assert np.array_equal(dense_result[2], compact_result[2])
        for dense_mask, compact_mask in zip(dense_result[3], compact_result[3]):
            assert (dense_mask is None) == (compact_mask is None)
            if dense_mask is not None and compact_mask is not None:
                np.testing.assert_array_equal(
                    _paint(dense_mask, (100, 100)), _paint(compact_mask, (100, 100))
                )

    def test_one_mask_per_source_in_a_cluster(self) -> None:
        """A mask never joins a cluster that already holds a mask of its source."""
        rectangles = [(0, 0, 10, 10), (1, 1, 11, 11), (2, 2, 12, 12)]
        masks = _rect_masks((20, 20), rectangles)

        _, _, labels, _ = _mask_weighted_fusion_clusters(
            _boxes_of(rectangles),
            np.array([0.9, 0.8, 0.7]),
            None,
            0.3,
            OverlapMetric.IOU,
            masks,
            sources=np.array([0, 0, 1]),
        )

        assert labels[0] != labels[1]

    def test_uniform_weights_average_the_boxes_but_keep_the_score(self) -> None:
        """Uniform weights give the plain mean box, leaving the mean score alone."""
        rectangles = [(0, 0, 10, 10), (2, 0, 12, 10)]
        masks = _rect_masks((20, 20), rectangles)

        fused_xyxy, fused_scores, _, _ = _mask_weighted_fusion_clusters(
            _boxes_of(rectangles),
            np.array([0.9, 0.1]),
            None,
            0.3,
            OverlapMetric.IOU,
            masks,
            box_weights="uniform",
        )

        assert np.allclose(fused_xyxy, [[1, 0, 11, 10]])
        assert np.allclose(fused_scores, [0.5])

    def test_empty_input_returns_empty_results(self) -> None:
        """No masks give empty boxes, scores, labels and fused masks."""
        fused_xyxy, fused_scores, labels, fused_masks = _mask_weighted_fusion_clusters(
            np.empty((0, 4)),
            np.empty(0),
            None,
            0.5,
            OverlapMetric.IOU,
            np.empty((0, 10, 10), dtype=bool),
        )

        assert fused_xyxy.shape == (0, 4)
        assert len(fused_scores) == len(labels) == len(fused_masks) == 0

    def test_empty_masks_stay_unfused(self) -> None:
        """A mask with no true pixel overlaps nothing and comes back alone."""
        masks = _rect_masks((10, 10), [(0, 0, 5, 5), (0, 0, 0, 0)])

        _, _, labels, fused_masks = _mask_weighted_fusion_clusters(
            _boxes_of([(0, 0, 5, 5), (0, 0, 0, 0)]),
            np.array([0.9, 0.8]),
            None,
            0.5,
            OverlapMetric.IOU,
            masks,
        )

        assert labels[0] != labels[1]
        assert fused_masks == [None, None]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param(
                {"box_weights": "median"}, "box_weights must be", id="weights"
            ),
            pytest.param(
                {"masks": np.zeros((3, 10, 10), dtype=bool)},
                "one mask per box",
                id="wrong-count",
            ),
            pytest.param(
                {"masks": np.zeros((2, 10), dtype=bool)},
                "one mask per box",
                id="not-3d",
            ),
        ],
    )
    def test_invalid_arguments_raise(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        """Unknown weights and masks that do not match the boxes are rejected."""
        arguments: dict[str, object] = {
            "boxes": _boxes_of([(0, 0, 5, 5), (1, 1, 6, 6)]),
            "scores": np.array([0.9, 0.8]),
            "categories": None,
            "iou_threshold": 0.5,
            "overlap_metric": OverlapMetric.IOU,
            "masks": _rect_masks((10, 10), [(0, 0, 5, 5), (1, 1, 6, 6)]),
        }
        arguments.update(kwargs)

        with pytest.raises(ValueError, match=message):
            _mask_weighted_fusion_clusters(**arguments)  # type: ignore[arg-type]


class TestMaskCluster:
    """A cluster keeps a vote over the union of its members' crops."""

    def test_fused_mask_falls_back_to_the_seed_when_no_pixel_survives(self) -> None:
        """If every pixel has under half the weight, the seed's mask is used."""
        seed = np.ones((2, 2), dtype=bool)
        cluster = _MaskCluster.start(seed, 0, 0, 0.4)

        cluster.add(np.ones((2, 2), dtype=bool), 4, 0, 0.4)
        cluster.add(np.ones((2, 2), dtype=bool), 8, 0, 0.4)

        assert cluster.box == [0, 0, 10, 2]
        np.testing.assert_array_equal(cluster.fused[:, :2], seed)
        assert not cluster.fused[:, 2:].any()
        assert cluster.area == 4

    def test_buffer_grows_to_the_union_of_the_member_bounds(self) -> None:
        """Adding a member outside the bounds extends the vote buffer in place."""
        cluster = _MaskCluster.start(np.ones((2, 2), dtype=bool), 5, 5, 1.0)

        cluster.add(np.ones((3, 3), dtype=bool), 2, 1, 1.0)

        assert cluster.box == [2, 1, 7, 7]
        assert cluster.votes is not None
        assert cluster.votes.shape == (6, 5)
        assert cluster.votes.sum() == 2 * 2 + 3 * 3


class TestPaintFusedMasks:
    """The result masks keep unfused winners and paint the fused crops."""

    def test_dense_masks_are_painted_and_other_rows_are_copied(self) -> None:
        """A fused row is replaced by its crop; an unfused row is its winner's mask."""
        masks = _rect_masks((10, 10), [(0, 0, 4, 4), (6, 6, 9, 9)])
        fused = _FusedMask(x=1, y=1, crop=np.ones((2, 3), dtype=bool))

        result = _paint_fused_masks(masks, np.array([0, 1]), [fused, None])

        expected = np.zeros((10, 10), dtype=bool)
        expected[1:3, 1:4] = True
        np.testing.assert_array_equal(result[0], expected)
        np.testing.assert_array_equal(result[1], masks[1])
        assert result is not masks

    def test_compact_masks_stay_compact(self) -> None:
        """A `CompactMask` input gives a `CompactMask` with the fused crop."""
        masks = _rect_masks((10, 10), [(0, 0, 4, 4), (6, 6, 9, 9)])
        compact = CompactMask.from_dense(
            masks, np.array([[0, 0, 3, 3], [6, 6, 8, 8]], dtype=np.float32), (10, 10)
        )
        fused = _FusedMask(x=1, y=1, crop=np.ones((2, 3), dtype=bool))

        result = _paint_fused_masks(compact, np.array([0, 1]), [fused, None])

        assert isinstance(result, CompactMask)
        expected = np.zeros((2, 10, 10), dtype=bool)
        expected[0, 1:3, 1:4] = True
        expected[1] = masks[1]
        np.testing.assert_array_equal(result.to_dense(), expected)
