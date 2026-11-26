"""Tests for `Detections.with_nms`, `with_soft_nms` and `with_nmm`."""

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import (
    Detections,
)
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
)
from tests.detection.helpers import _make_obb_detections, _rotated_rect


class TestDetectionsOverlapValidation:
    """`with_nms` and `with_nmm` require confidence and class IDs by default."""

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("with_nms", id="with_nms"),
            pytest.param("with_nmm", id="with_nmm"),
        ],
    )
    def test_requires_confidence(self, method: str) -> None:
        """Missing confidence raises a descriptive `ValueError`."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )

        with pytest.raises(ValueError, match="Detections confidence must be given"):
            getattr(detections, method)(threshold=0.5)

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("with_nms", id="with_nms"),
            pytest.param("with_nmm", id="with_nmm"),
        ],
    )
    def test_requires_class_id_when_not_class_agnostic(self, method: str) -> None:
        """Missing class IDs raise a descriptive `ValueError`."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
        )

        with pytest.raises(ValueError, match="Detections class_id must be given"):
            getattr(detections, method)(threshold=0.5)


class TestDetectionsWithSoftNms:
    """`Detections.with_soft_nms` — decay semantics, dispatch, non-mutation."""

    def test_requires_confidence(self) -> None:
        """Missing confidence raises a descriptive `ValueError`."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )

        with pytest.raises(ValueError, match="Detections confidence must be given"):
            detections.with_soft_nms()

    def test_requires_class_id_when_not_class_agnostic(self) -> None:
        """Missing class IDs raise a descriptive `ValueError`."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
        )

        with pytest.raises(ValueError, match="Detections class_id must be given"):
            detections.with_soft_nms()

    def test_rejects_non_positive_sigma(self) -> None:
        """`sigma` must be strictly greater than 0."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )

        with pytest.raises(ValueError, match="sigma"):
            detections.with_soft_nms(sigma=0)

    def test_returns_self_for_empty_detections(self) -> None:
        """Empty input short-circuits without dispatching to Soft-NMS."""
        detections = Detections.empty()

        result = detections.with_soft_nms()

        assert result is detections

    def test_does_not_mutate_original(self) -> None:
        """The source `Detections` confidence is untouched after the call."""
        detections = Detections(
            xyxy=np.array([[10, 10, 40, 40], [15, 15, 40, 40]], dtype=np.float32),
            confidence=np.array([0.8, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )
        assert detections.confidence is not None
        original_confidence = detections.confidence.copy()

        result = detections.with_soft_nms(sigma=0.2)

        assert result is not detections
        np.testing.assert_array_equal(detections.confidence, original_confidence)
        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.07176137, 0.9], decimal=5)

    def test_box_path_class_agnostic(self) -> None:
        """`class_agnostic=True` skips the class_id requirement for boxes."""
        detections = Detections(
            xyxy=np.array([[10, 10, 40, 40], [15, 15, 40, 40]], dtype=np.float32),
            confidence=np.array([0.8, 0.9], dtype=np.float32),
        )

        result = detections.with_soft_nms(sigma=0.2, class_agnostic=True)

        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.07176137, 0.9], decimal=5)

    def test_mask_path_dispatch(self) -> None:
        """Mask data present routes through `mask_soft_non_max_suppression`."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[:, :2, :2] = True
        detections = Detections(
            xyxy=np.array([[0, 0, 4, 4], [0, 0, 4, 4]], dtype=np.float32),
            confidence=np.array([0.9, 0.8], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
            mask=masks,
        )

        result = detections.with_soft_nms(sigma=0.5)

        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.9, 0.10826823], decimal=5)
        assert result.mask is not detections.mask

    def test_single_mask_self_decay_class_agnostic(self) -> None:
        """A single class-agnostic mask detection is left undecayed (no peer)."""
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[:, :2, :2] = True
        detections = Detections(
            xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
            confidence=np.array([0.8], dtype=np.float32),
            mask=masks,
        )

        result = detections.with_soft_nms(sigma=0.5, class_agnostic=True)

        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.8], decimal=5)

    def test_score_threshold_filters_to_subset(self) -> None:
        """`score_threshold` drops decayed detections, producing a real subset."""
        detections = Detections(
            xyxy=np.array([[10, 10, 40, 40], [15, 15, 40, 40]], dtype=np.float32),
            confidence=np.array([0.8, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        result = detections.with_soft_nms(sigma=0.2, score_threshold=0.5)

        assert len(result) == 1
        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.9], decimal=5)

    def test_uses_obb_iou_when_oriented_box_coordinates_present(self) -> None:
        """Edge-touching OBBs keep their score although their envelopes overlap."""
        quad_a = _rotated_rect(0, 0, 10 * np.sqrt(2), 10 * np.sqrt(2), 45)
        quad_b = _rotated_rect(10, 10, 10 * np.sqrt(2), 10 * np.sqrt(2), 45)
        detections = _make_obb_detections([quad_a, quad_b], [0.9, 0.8], [0, 0])

        result = detections.with_soft_nms(sigma=0.5)

        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.9, 0.8], decimal=4)
        assert len(detections.with_soft_nms(sigma=0.5, score_threshold=0.79)) == 2

    def test_overlapping_obbs_are_decayed(self) -> None:
        """OBBs whose rotated bodies overlap are still decayed."""
        quad = _rotated_rect(50, 50, 100, 10, 30)
        detections = _make_obb_detections([quad, quad], [0.9, 0.8], [0, 0])

        result = detections.with_soft_nms(sigma=0.5)

        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.9, 0.10826823], decimal=4)

    def test_mask_takes_precedence_over_oriented_box(self) -> None:
        """With both a mask and OBB corners, the mask IoU drives the decay."""
        quad_a = _rotated_rect(0, 0, 10 * np.sqrt(2), 10 * np.sqrt(2), 45)
        quad_b = _rotated_rect(10, 10, 10 * np.sqrt(2), 10 * np.sqrt(2), 45)
        detections = _make_obb_detections([quad_a, quad_b], [0.9, 0.8], [0, 0])
        detections.mask = np.ones((2, 4, 4), dtype=bool)

        result = detections.with_soft_nms(sigma=0.5)

        assert result.confidence is not None
        np.testing.assert_almost_equal(result.confidence, [0.9, 0.10826823], decimal=4)

    def test_none_score_threshold_keeps_all_detections(self) -> None:
        """Default `score_threshold=None` never drops a detection."""
        detections = Detections(
            xyxy=np.array([[10, 10, 40, 40], [15, 15, 40, 40]], dtype=np.float32),
            confidence=np.array([0.8, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        result = detections.with_soft_nms(sigma=0.2)

        assert len(result) == 2

    @pytest.mark.parametrize(
        ("dtype", "origin"),
        [
            pytest.param(np.int64, 2**60, id="int64-2pow60"),
            pytest.param(np.uint64, 2**60, id="uint64-2pow60"),
        ],
    )
    def test_narrow_boxes_at_large_64bit_integer_origin(
        self, dtype: type[np.integer], origin: int
    ) -> None:
        """Soft-NMS decays scores identically to origin=0 and keeps integer xyxy."""
        boxes_zero = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=dtype)
        boxes_large = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [origin + 1, origin + 1, origin + 11, origin + 11],
            ],
            dtype=dtype,
        )
        confidence = np.array([0.9, 0.7], dtype=np.float32)
        class_id = np.array([0, 0], dtype=int)

        expected = Detections(
            xyxy=boxes_zero, confidence=confidence, class_id=class_id
        ).with_soft_nms(sigma=0.5)
        result = Detections(
            xyxy=boxes_large, confidence=confidence, class_id=class_id
        ).with_soft_nms(sigma=0.5)

        assert len(result) == 2
        assert result.confidence is not None
        assert expected.confidence is not None
        np.testing.assert_allclose(result.confidence, expected.confidence)
        assert result.xyxy.dtype == dtype
        np.testing.assert_array_equal(result.xyxy, boxes_large)

    @pytest.mark.parametrize("class_agnostic", [False, True])
    def test_float32_boxes_and_scores_with_int64_class_id_keep_float64_decay_precision(
        self, class_agnostic: bool
    ) -> None:
        """Float32 boxes/scores with int64 class_id retain float64 decay precision."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9, 0.01], dtype=np.float32),
            class_id=np.array([0, 0], dtype=np.int64),
        )

        result = detections.with_soft_nms(
            sigma=0.01, score_threshold=0, class_agnostic=class_agnostic
        )

        assert len(result) == 2
        assert result.confidence is not None
        assert result.confidence.dtype == np.float64
        assert result.confidence[1] > 0.0


class TestDetectionsWithNMS:
    """NMS behaviour tests for 64-bit integer coordinates."""

    @pytest.mark.parametrize(
        ("dtype", "origin"),
        [
            pytest.param(np.int64, 2**60, id="int64-2pow60"),
            pytest.param(np.uint64, 2**60, id="uint64-2pow60"),
        ],
    )
    def test_narrow_boxes_at_large_64bit_integer_origin(
        self, dtype: type[np.integer], origin: int
    ) -> None:
        """NMS suppresses overlapping narrow 64-bit boxes and keeps exact xyxy."""
        boxes = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [origin + 1, origin + 1, origin + 11, origin + 11],
            ],
            dtype=dtype,
        )
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.7], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        result = detections.with_nms(threshold=0.5)

        assert len(result) == 1
        assert result.xyxy.dtype == dtype
        np.testing.assert_array_equal(result.xyxy, boxes[:1])


def _make_rectangle_detections(kind: str, rectangles: list[list[int]]) -> Detections:
    """Build same-class detections of `kind` from `[x1, y1, x2, y2]` pixel rectangles.

    Rectangles cover pixels `x1:x2` by `y1:y2`. `kind` selects the geometry: `box`,
    `dense-mask`, `compact-mask` or `obb`. Confidences descend with the index.
    """
    boxes = np.array(rectangles, dtype=np.float32)
    detections = Detections(
        xyxy=boxes,
        confidence=np.linspace(0.9, 0.5, len(rectangles), dtype=np.float32),
        class_id=np.zeros(len(rectangles), dtype=int),
    )
    if kind == "box":
        return detections
    if kind == "obb":
        x1, y1, x2, y2 = boxes.T
        detections.data[ORIENTED_BOX_COORDINATES] = np.stack(
            [
                np.stack([x1, y1], axis=1),
                np.stack([x2, y1], axis=1),
                np.stack([x2, y2], axis=1),
                np.stack([x1, y2], axis=1),
            ],
            axis=1,
        )
        return detections

    height = width = int(boxes.max()) + 1
    masks = np.zeros((len(rectangles), height, width), dtype=bool)
    for index, (left, top, right, bottom) in enumerate(rectangles):
        masks[index, top:bottom, left:right] = True
    if kind == "dense-mask":
        detections.mask = masks
    else:
        detections.mask = CompactMask.from_dense(
            masks=masks,
            xyxy=np.tile([0, 0, width - 1, height - 1], (len(rectangles), 1)),
            image_shape=(height, width),
        )
    return detections


class TestDetectionsWithNMM:
    """NMM-specific behaviour tests for `Detections.with_nmm`."""

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param(1e9, id="large-origin"),
            pytest.param(0.123456789123, id="decimal-extrema"),
        ],
    )
    @pytest.mark.parametrize(
        "class_agnostic",
        [
            pytest.param(False, id="class-aware"),
            pytest.param(True, id="class-agnostic"),
        ],
    )
    def test_float64_union(self, origin: float, class_agnostic: bool) -> None:
        """Merging retains representable extrema and leaves singleton groups intact."""
        boxes = np.array(
            [
                [origin, origin, origin + 20, origin + 20],
                [origin + 2, origin + 2, origin + 22, origin + 22],
                [origin + 100, origin + 100, origin + 120, origin + 120],
            ],
            dtype=np.float64,
        )
        before = boxes.copy()
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.8, 0.7]),
            class_id=np.array([1, 2 if class_agnostic else 1, 3]),
            tracker_id=np.array([10, 20, 30]),
        )
        result = detections.with_nmm(class_agnostic=class_agnostic)
        assert len(result) == 2
        assert result.xyxy.dtype == np.float64
        assert result.tracker_id is not None
        assert result.class_id is not None
        merged = result[result.tracker_id == 10]
        singleton = result[result.tracker_id == 30]
        assert isinstance(merged, Detections)
        assert isinstance(singleton, Detections)
        assert len(merged) == len(singleton) == 1
        np.testing.assert_array_equal(
            merged.xyxy, [[origin, origin, origin + 22, origin + 22]]
        )
        np.testing.assert_array_equal(singleton.xyxy, boxes[2:])
        np.testing.assert_array_equal(merged.class_id, [1])
        np.testing.assert_array_equal(singleton.class_id, [3])
        np.testing.assert_array_equal(detections.xyxy, before)

    def test_float32_union_unchanged(self) -> None:
        """Ordinary float32 boxes keep exact union coordinates and output dtype."""
        boxes = np.array([[0, 0, 20, 20], [2, 2, 22, 22]], dtype=np.float32)
        result = Detections(
            xyxy=boxes, confidence=np.array([0.9, 0.8]), class_id=np.array([1, 1])
        ).with_nmm()
        assert len(result) == 1
        assert result.xyxy.dtype == np.float32
        np.testing.assert_array_equal(result.xyxy, [[0, 0, 22, 22]])

    def test_class_aware_keeps_different_classes(self) -> None:
        """Overlapping detections in different classes remain separate."""
        boxes = np.array(
            [[1e9, 1e9, 1e9 + 20, 1e9 + 20], [1e9 + 2, 1e9 + 2, 1e9 + 22, 1e9 + 22]]
        )
        result = Detections(
            xyxy=boxes, confidence=np.array([0.9, 0.8]), class_id=np.array([1, 2])
        ).with_nmm()
        assert len(result) == 2
        np.testing.assert_array_equal(result.xyxy, boxes)

    @pytest.mark.parametrize(
        "compact", [pytest.param(False, id="dense"), pytest.param(True, id="compact")]
    )
    def test_mask_union_retains_float64_boxes(self, compact: bool) -> None:
        """Dense and compact masks keep their union alongside float64 coordinates."""
        boxes = np.array([[2, 2, 12, 12], [3, 3, 13, 13]], dtype=np.float64)
        masks = np.zeros((2, 16, 16), dtype=bool)
        masks[0, 2:12, 2:12] = True
        masks[1, 3:13, 3:13] = True
        mask = (
            CompactMask.from_dense(masks, boxes, image_shape=(16, 16))
            if compact
            else masks
        )
        result = Detections(
            xyxy=boxes,
            mask=mask,
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([1, 1]),
        ).with_nmm()
        assert len(result) == 1
        assert result.xyxy.dtype == np.float64
        np.testing.assert_array_equal(result.xyxy, [[2, 2, 13, 13]])
        assert result.mask is not None
        actual = (
            result.mask.to_dense()
            if isinstance(result.mask, CompactMask)
            else result.mask
        )
        np.testing.assert_array_equal(actual, masks.any(axis=0, keepdims=True))
        np.testing.assert_array_equal(result.class_id, [1])
        assert result.confidence is not None
        np.testing.assert_allclose(result.confidence, [0.85])

    @pytest.mark.parametrize(
        ("dtype", "extent"),
        [
            pytest.param(np.int8, 100, id="int8"),
            pytest.param(np.int16, 30000, id="int16"),
            pytest.param(np.int32, 2000000000, id="int32"),
            pytest.param(np.int64, 2**62, id="int64"),
            pytest.param(np.float16, 60000, id="float16"),
            pytest.param(np.float32, 3e38, id="float32"),
            pytest.param(np.float64, 60000, id="float64"),
        ],
    )
    @pytest.mark.parametrize("vertical", [False, True])
    def test_confidence_uses_unwrapped_box_extents(
        self, dtype: type[np.number], extent: float, vertical: bool
    ) -> None:
        """NMM weights scores by geometric areas even for narrow coordinate dtypes."""
        boxes = np.array(
            [[-extent, 0, extent, 1], [-extent / 2, 0, extent, 1]], dtype=dtype
        )
        if vertical:
            boxes = boxes[:, [1, 0, 3, 2]]
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.6]),
            class_id=np.array([0, 0]),
        )

        with np.errstate(over="raise", invalid="raise"):
            result = detections.with_nmm(threshold=0.5)

        assert len(result) == 1
        assert result.confidence is not None
        np.testing.assert_allclose(result.confidence, [(2 * 0.9 + 1.5 * 0.6) / 3.5])
        np.testing.assert_array_equal(detections.xyxy, boxes)

    @pytest.mark.parametrize(
        (
            "corners",
            "confidence",
            "class_ids",
            "iou_threshold",
            "class_agnostic",
            "overlap_metric",
            "expected_corners",
            "expected_confidence",
            "exception",
        ),
        [
            pytest.param(
                [
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                    [[11, 11], [51, 11], [51, 31], [11, 31]],
                ],
                [0.9, 0.85],
                [0, 0],
                0.5,
                False,
                OverlapMetric.IOU,
                [[[10, 10], [51, 10], [51, 31], [10, 31]]],
                [0.875],
                DoesNotRaise(),
                id="axis-aligned-merge",
            ),
            pytest.param(
                [
                    _rotated_rect(50, 50, 40, 10, 45).tolist(),
                    _rotated_rect(55, 55, 40, 10, 45).tolist(),
                ],
                [0.9, 0.8],
                [0, 0],
                0.3,
                False,
                OverlapMetric.IOU,
                [[[39.39, 32.32], [72.68, 65.61], [65.61, 72.68], [32.32, 39.39]]],
                [0.85],
                DoesNotRaise(),
                id="rotated-45deg-merge",
            ),
            pytest.param(
                [
                    [[0, 0], [20, 0], [20, 10], [0, 10]],
                    [[5, 5], [25, 5], [25, 15], [5, 15]],
                    [[10, 0], [30, 0], [30, 10], [10, 10]],
                ],
                [0.9, 0.8, 0.7],
                [0, 0, 0],
                0.2,
                False,
                OverlapMetric.IOU,
                [[[0, 0], [30, 0], [30, 15], [0, 15]]],
                [0.8],
                DoesNotRaise(),
                id="three-group-merge",
            ),
            pytest.param(
                [
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                ],
                [0.9],
                [0],
                0.5,
                False,
                OverlapMetric.IOU,
                [[[10, 10], [50, 10], [50, 30], [10, 30]]],
                [0.9],
                DoesNotRaise(),
                id="single-passthrough",
            ),
            pytest.param(
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 5], [35, 5], [35, 25], [5, 25]],
                ],
                [0.9, 0.8],
                [0, 1],
                0.3,
                True,
                OverlapMetric.IOU,
                [[[0, 0], [35, 0], [35, 25], [0, 25]]],
                [0.85],
                DoesNotRaise(),
                id="class-agnostic",
            ),
            pytest.param(
                [
                    [[0, 0], [40, 0], [40, 30], [0, 30]],
                    [[10, 10], [30, 10], [30, 20], [10, 20]],
                ],
                [0.9, 0.8],
                [0, 0],
                0.3,
                False,
                OverlapMetric.IOS,
                [[[0, 0], [40, 0], [40, 30], [0, 30]]],
                [0.885714],
                DoesNotRaise(),
                id="ios-metric",
            ),
            pytest.param(
                [
                    _rotated_rect(50, 50, 40, 15, 30).tolist(),
                    _rotated_rect(55, 50, 40, 15, -15).tolist(),
                ],
                [0.9, 0.7],
                [0, 0],
                0.2,
                False,
                OverlapMetric.IOU,
                [[[43.65, 20.99], [81.56, 42.88], [62.12, 76.56], [24.21, 54.68]]],
                [0.813652],
                DoesNotRaise(),
                id="mixed-angle-merge",
            ),
            pytest.param(
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 5], [35, 5], [35, 25], [5, 25]],
                    [[200, 200], [240, 200], [240, 220], [200, 220]],
                    [[205, 205], [245, 205], [245, 225], [205, 225]],
                ],
                [0.9, 0.7, 0.85, 0.6],
                [0, 0, 0, 0],
                0.2,
                False,
                OverlapMetric.IOU,
                [
                    [[0, 0], [35, 0], [35, 25], [0, 25]],
                    [[200, 200], [245, 200], [245, 225], [200, 225]],
                ],
                [0.8, 0.725],
                DoesNotRaise(),
                id="two-separate-groups",
            ),
            pytest.param(
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 10], [25, 10], [25, 10], [5, 10]],
                ],
                [0.9, 0.7],
                [0, 0],
                0.01,
                False,
                OverlapMetric.IOU,
                # A zero-area (collinear) OBB scores IoU 0 (see
                # test_degenerate_boxes_score_zero), so it cannot group and the
                # two detections are not merged.
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 10], [25, 10], [25, 10], [5, 10]],
                ],
                [0.9, 0.7],
                DoesNotRaise(),
                id="degenerate-collinear-obb",
            ),
            pytest.param(
                None,
                [0.9, 0.8],
                [0, 0],
                0.4,
                False,
                OverlapMetric.IOU,
                None,
                None,
                pytest.raises(ValueError, match="corners must have shape"),
                id="flat-n8-raises",
            ),
        ],
    )
    def test_obb_nmm_merge(
        self,
        corners: list[list[list[float]]] | None,
        confidence: list[float],
        class_ids: list[int],
        iou_threshold: float,
        class_agnostic: bool,
        overlap_metric: OverlapMetric,
        expected_corners: list[list[list[float]]] | None,
        expected_confidence: list[float] | None,
        exception: DoesNotRaise,
    ) -> None:
        """OBB NMM produces correct geometry and confidence."""
        if corners is None:
            xyxy = np.array(
                [[0, 0, 30, 20], [5, 5, 35, 25]],
                dtype=np.float32,
            )
            flat = np.array(
                [
                    [0, 0, 30, 0, 30, 20, 0, 20],
                    [5, 5, 35, 5, 35, 25, 5, 25],
                ],
                dtype=np.float32,
            )
            detections = Detections(
                xyxy=xyxy,
                confidence=np.array(confidence, dtype=np.float32),
                class_id=np.array(class_ids),
                data={ORIENTED_BOX_COORDINATES: flat},
            )
        else:
            corner_arrays = [np.array(corner, dtype=np.float32) for corner in corners]
            detections = _make_obb_detections(corner_arrays, confidence, class_ids)

        with exception:
            result = detections.with_nmm(
                threshold=iou_threshold,
                class_agnostic=class_agnostic,
                overlap_metric=overlap_metric,
            )

            assert expected_confidence is not None
            assert expected_corners is not None
            assert len(result) == len(expected_confidence)
            assert result.confidence is not None
            for i, exp_c in enumerate(expected_confidence):
                assert result.confidence[i] == pytest.approx(exp_c, abs=1e-3)
            result_corners = result.data[ORIENTED_BOX_COORDINATES]
            expected_corner_array = np.array(expected_corners, dtype=np.float32)
            assert np.allclose(
                result_corners,
                expected_corner_array,
                atol=0.5,
            )

    def test_obb_nmm_matches_aabb_for_axis_aligned(self) -> None:
        """Axis-aligned OBB NMM produces same envelope as AABB NMM."""
        xyxy = np.array([[0, 0, 30, 20], [5, 5, 35, 25]], dtype=np.float32)
        confidence = np.array([0.9, 0.5], dtype=np.float32)
        class_id = np.array([0, 0])

        aabb_detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        obb_detections = _make_obb_detections(
            [
                np.array(
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    dtype=np.float32,
                ),
                np.array(
                    [[5, 5], [35, 5], [35, 25], [5, 25]],
                    dtype=np.float32,
                ),
            ],
            confidence.tolist(),
            class_id.tolist(),
        )

        aabb_result = aabb_detections.with_nmm(threshold=0.4)
        obb_result = obb_detections.with_nmm(threshold=0.4)

        assert len(aabb_result) == 1
        assert len(obb_result) == 1
        assert np.allclose(aabb_result.xyxy, obb_result.xyxy, atol=1e-4)

    def test_staircase_obb_merge_within_union(self) -> None:
        """Diagonal staircase OBBs: merged AABB equals axis-aligned union."""
        quads = [
            np.array(
                [[0, 0], [20, 0], [20, 20], [0, 20]],
                dtype=np.float32,
            ),
            np.array(
                [[12, 12], [32, 12], [32, 32], [12, 32]],
                dtype=np.float32,
            ),
            np.array(
                [[24, 24], [44, 24], [44, 44], [24, 44]],
                dtype=np.float32,
            ),
        ]
        detections = _make_obb_detections(quads, [0.7, 0.9, 0.8], [0, 0, 0])

        result = detections.with_nmm(threshold=0.05)

        assert len(result) == 1
        assert np.allclose(result.xyxy, [[0.0, 0.0, 44.0, 44.0]], atol=0.5)

    def test_obb_nmm_empty_detections(self) -> None:
        """Empty OBB detections return empty result."""
        dets = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
            data={ORIENTED_BOX_COORDINATES: np.empty((0, 4, 2), dtype=np.float32)},
        )

        result = dets.with_nmm(threshold=0.5)

        assert len(result) == 0

    def test_compact_mask_nmm_preserves_full_frame_union(self) -> None:
        """CompactMask NMM keeps full-frame mask pixels after merging."""
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[0, 1, 1] = True
        masks[0, 8, 8] = True
        masks[1, 1, 1] = True
        masks[1, 7, 7] = True
        compact_mask = CompactMask.from_dense(
            masks=masks,
            xyxy=np.array([[0, 0, 9, 9], [0, 0, 9, 9]], dtype=np.float32),
            image_shape=(10, 10),
        )
        detections = Detections(
            xyxy=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            mask=compact_mask,
            confidence=np.array([0.9, 0.8], dtype=np.float32),
            class_id=np.array([0, 0]),
        )

        result = detections.with_nmm(threshold=0.1)

        assert len(result) == 1
        assert isinstance(result.mask, CompactMask)
        assert result.mask.bbox_xyxy.tolist() == [[1, 1, 8, 8]]
        result_mask = result.mask.to_dense()[0]
        assert result_mask[1, 1]
        assert result_mask[7, 7]
        assert result_mask[8, 8]

    @pytest.mark.parametrize(
        ("dtype", "origin"),
        [
            pytest.param(np.int64, 2**60, id="int64-2pow60"),
            pytest.param(np.int64, -(2**63), id="int64-min"),
            pytest.param(np.uint64, 2**60, id="uint64-2pow60"),
            pytest.param(np.uint64, 2**64 - 20, id="uint64-max"),
        ],
    )
    @pytest.mark.parametrize(
        "class_agnostic",
        [
            pytest.param(False, id="class-aware"),
            pytest.param(True, id="class-agnostic"),
        ],
    )
    @pytest.mark.parametrize(
        "overlap_metric",
        [
            pytest.param(OverlapMetric.IOU, id="iou"),
            pytest.param(OverlapMetric.IOS, id="ios"),
        ],
    )
    def test_narrow_boxes_at_large_64bit_integer_origins(
        self,
        dtype: type[np.integer],
        origin: int,
        class_agnostic: bool,
        overlap_metric: OverlapMetric,
    ) -> None:
        """NMM merges overlapping narrow 64-bit boxes identically to origin=0."""
        boxes_zero = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=dtype)
        boxes_large = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [origin + 1, origin + 1, origin + 11, origin + 11],
            ],
            dtype=dtype,
        )
        confidence = np.array([0.9, 0.7], dtype=np.float32)
        class_id = np.array([0, 0], dtype=int)

        expected = Detections(
            xyxy=boxes_zero, confidence=confidence, class_id=class_id
        ).with_nmm(
            threshold=0.5,
            class_agnostic=class_agnostic,
            overlap_metric=overlap_metric,
        )
        result = Detections(
            xyxy=boxes_large, confidence=confidence, class_id=class_id
        ).with_nmm(
            threshold=0.5,
            class_agnostic=class_agnostic,
            overlap_metric=overlap_metric,
        )

        assert len(result) == 1
        assert result.confidence is not None
        assert expected.confidence is not None
        np.testing.assert_allclose(result.confidence, expected.confidence)
        assert result.xyxy.dtype == dtype
        np.testing.assert_array_equal(
            result.xyxy,
            np.array([[origin, origin, origin + 11, origin + 11]], dtype=dtype),
        )

    @pytest.mark.parametrize(
        ("dtype", "origin"),
        [
            pytest.param(np.int64, 2**60, id="int64"),
            pytest.param(np.uint64, 2**60, id="uint64"),
        ],
    )
    def test_distinct_area_narrow_boxes_merge_with_exact_area_weights(
        self, dtype: type[np.integer], origin: int
    ) -> None:
        """Distinct-area narrow 64-bit boxes merge with exact area-weighted score."""
        boxes = np.array(
            [
                [origin, origin, origin + 20, origin + 10],
                [origin, origin, origin + 10, origin + 10],
            ],
            dtype=dtype,
        )
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.6], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        result = detections.with_nmm(threshold=0.5)

        assert len(result) == 1
        assert result.confidence is not None
        np.testing.assert_allclose(result.confidence, [0.8])
        assert result.xyxy.dtype == dtype
        np.testing.assert_array_equal(result.xyxy, boxes[:1])

    def test_multi_origin_int64_clusters_merge_separately(self) -> None:
        """Clusters at negative and positive large int64 origins merge separately."""
        neg = -(2**60)
        pos = 2**60
        boxes = np.array(
            [
                [neg, 0, neg + 10, 10],
                [neg + 1, 1, neg + 11, 11],
                [pos, 0, pos + 10, 10],
                [pos + 1, 1, pos + 11, 11],
            ],
            dtype=np.int64,
        )
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.7, 0.85, 0.65], dtype=np.float32),
            class_id=np.array([0, 0, 0, 0], dtype=int),
        )

        result = detections.with_nmm(threshold=0.5)

        assert len(result) == 2
        assert result.xyxy.dtype == np.int64
        np.testing.assert_array_equal(
            result.xyxy,
            np.array(
                [[neg, 0, neg + 11, 11], [pos, 0, pos + 11, 11]],
                dtype=np.int64,
            ),
        )

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(np.int64, id="int64"), pytest.param(np.uint64, id="uint64")],
    )
    @pytest.mark.parametrize(
        "shift_x",
        [
            pytest.param(10, id="edge-touching"),
            pytest.param(20, id="disjoint"),
        ],
    )
    def test_disjoint_and_edge_touching_narrow_boxes_do_not_merge(
        self, dtype: type[np.integer], shift_x: int
    ) -> None:
        """Edge-touching and disjoint narrow 64-bit boxes stay separate."""
        origin = 2**60
        boxes = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [origin + shift_x, origin, origin + shift_x + 10, origin + 10],
            ],
            dtype=dtype,
        )
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.7], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        result = detections.with_nmm(threshold=0.5)

        assert len(result) == 2
        assert result.xyxy.dtype == dtype
        np.testing.assert_array_equal(result.xyxy, boxes)

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(np.int64, id="int64"), pytest.param(np.uint64, id="uint64")],
    )
    def test_merged_pair_and_singleton_at_large_origin_preserve_integer_xyxy(
        self, dtype: type[np.integer]
    ) -> None:
        """Merged pairs and disjoint singletons both keep exact 64-bit integer xyxy."""
        origin = 2**60
        boxes = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [origin + 1, origin + 1, origin + 11, origin + 11],
                [origin + 100, origin + 100, origin + 110, origin + 110],
            ],
            dtype=dtype,
        )
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.7, 0.5], dtype=np.float32),
            class_id=np.array([0, 0, 0], dtype=int),
        )

        result = detections.with_nmm(threshold=0.5)

        expected_xyxy = np.array(
            [
                [origin, origin, origin + 11, origin + 11],
                [origin + 100, origin + 100, origin + 110, origin + 110],
            ],
            dtype=dtype,
        )
        assert len(result) == 2
        assert result.xyxy.dtype == dtype
        np.testing.assert_array_equal(result.xyxy, expected_xyxy)

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(np.int64, id="int64"), pytest.param(np.uint64, id="uint64")],
    )
    @pytest.mark.parametrize(
        ("class_agnostic", "expected_len"),
        [
            pytest.param(False, 2, id="class-aware"),
            pytest.param(True, 1, id="class-agnostic"),
        ],
    )
    def test_different_class_overlapping_narrow_boxes_at_large_origin(
        self,
        dtype: type[np.integer],
        class_agnostic: bool,
        expected_len: int,
    ) -> None:
        """Different-class narrow 64-bit boxes merge only when class-agnostic."""
        origin = 2**60
        boxes = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [origin + 1, origin + 1, origin + 11, origin + 11],
            ],
            dtype=dtype,
        )
        detections = Detections(
            xyxy=boxes,
            confidence=np.array([0.9, 0.7], dtype=np.float32),
            class_id=np.array([0, 1], dtype=int),
        )

        result = detections.with_nmm(threshold=0.5, class_agnostic=class_agnostic)

        assert len(result) == expected_len

    @pytest.mark.parametrize("kind", ["box", "dense-mask", "compact-mask", "obb"])
    def test_zero_threshold_keeps_disjoint_detections_unchanged(
        self, kind: str
    ) -> None:
        """Disjoint same-class detections are not merged at threshold 0."""
        detections = _make_rectangle_detections(
            kind, [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]]
        )

        result = detections.with_nmm(threshold=0.0)

        assert len(result) == 3
        np.testing.assert_array_equal(result.xyxy, detections.xyxy)
        assert result.confidence is not None
        assert detections.confidence is not None
        np.testing.assert_allclose(result.confidence, detections.confidence)

    @pytest.mark.parametrize("kind", ["box", "dense-mask", "compact-mask", "obb"])
    def test_zero_threshold_merges_detections_with_any_overlap(self, kind: str) -> None:
        """A pair sharing a single pixel of area is still merged at threshold 0."""
        detections = _make_rectangle_detections(
            kind, [[0, 0, 10, 10], [9, 9, 19, 19], [100, 100, 110, 110]]
        )

        result = detections.with_nmm(threshold=0.0)

        assert len(result) == 2
        assert [0, 0, 19, 19] in result.xyxy.tolist()
        assert [100, 100, 110, 110] in result.xyxy.tolist()
