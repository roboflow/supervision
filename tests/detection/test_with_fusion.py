"""Exercise the public detection fusion API across supported geometries."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils._fusion import _box_weighted_box_fusion
from supervision.detection.utils.iou import OverlapMetric, OverlapOn
from supervision.detection.utils.nms import FusionBoxWeights


def _pair(**overrides: object) -> Detections:
    """Two overlapping boxes of class 0 that fuse into one detection."""
    fields: dict[str, object] = {
        "xyxy": np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float64),
        "confidence": np.array([0.9, 0.8]),
        "class_id": np.array([0, 0]),
    }
    fields.update(overrides)
    return Detections(**fields)  # type: ignore[arg-type]


def _axis_corners(xyxy: np.ndarray) -> np.ndarray:
    """Corners (top-left, top-right, bottom-right, bottom-left) of axis-aligned
    boxes."""
    x1, y1, x2, y2 = xyxy.T
    return np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )


def _rotated_box(
    width: float,
    height: float,
    degrees: float,
    origin: tuple[float, float],
    shift: float = 0.0,
) -> np.ndarray:
    """Corners of a `width` x `height` box turned by `degrees` around `origin`.

    `shift` moves the box along its own first edge before the rotation.
    """
    angle = np.deg2rad(degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    base = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    shifted = base + np.array([shift, 0.0])
    return shifted @ rotation.T + np.asarray(origin)


def _oriented(
    corners: np.ndarray, confidence: list[float], **overrides: object
) -> Detections:
    """Oriented-box detections of class 0 whose `xyxy` is the corners' envelope."""
    points = corners.reshape(len(corners), 4, 2)
    xyxy = np.concatenate([points.min(axis=1), points.max(axis=1)], axis=1)
    fields: dict[str, object] = {
        "xyxy": xyxy,
        "confidence": np.array(confidence),
        "class_id": np.zeros(len(corners), dtype=int),
        "data": {ORIENTED_BOX_COORDINATES: corners},
    }
    fields.update(overrides)
    return Detections(**fields)  # type: ignore[arg-type]


def _rect_masks(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Boolean masks, one filled `(x_min, y_min, x_max, y_max)` rectangle each."""
    masks = np.zeros((len(rectangles), *shape), dtype=bool)
    for index, (x_min, y_min, x_max, y_max) in enumerate(rectangles):
        masks[index, y_min:y_max, x_min:x_max] = True
    return masks


def _with_masks(
    masks: np.ndarray, confidence: list[float], **overrides: object
) -> Detections:
    """Detections of class 0 whose boxes are the bounds of the given masks."""
    boxes = []
    for mask in masks:
        rows, columns = np.nonzero(mask)
        boxes.append([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1])
    fields: dict[str, object] = {
        "xyxy": np.array(boxes, dtype=np.float64),
        "mask": masks,
        "confidence": np.array(confidence),
        "class_id": np.zeros(len(masks), dtype=int),
    }
    fields.update(overrides)
    return Detections(**fields)  # type: ignore[arg-type]


def _masked_pair() -> Detections:
    """Two overlapping boxes with rectangular masks on a 20x20 image."""
    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, 0:10, 0:10] = True
    masks[1, 1:11, 1:11] = True
    return _pair(mask=masks)


class TestWithFusion:
    """`Detections.with_fusion` fuses overlapping detections into one."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            pytest.param(
                {"confidence": np.array([0.9, -0.4])},
                "Confidences .* must be finite and non-negative",
                id="negative-confidence",
            ),
            pytest.param(
                {"confidence": np.array([0.9, np.nan])},
                "Confidences .* must be finite and non-negative",
                id="nonfinite-confidence",
            ),
            pytest.param(
                {"xyxy": np.array([[10.0, 0, 0, 10], [1.0, 1, 11, 11]])},
                "Boxes .* must have finite",
                id="reversed-corners",
            ),
            pytest.param(
                {"class_id": np.array([0, np.nan])},
                "Class ids .* must be finite",
                id="nonfinite-class-id",
            ),
        ],
    )
    def test_rejects_invalid_operands(
        self, overrides: dict[str, np.ndarray], message: str
    ) -> None:
        """Reject inputs the legacy array fusion API cannot cluster."""
        detections = _pair(**overrides)

        with pytest.raises(ValueError, match=message):
            detections.with_fusion()

    def test_fuses_overlapping_boxes_with_confidence_weights(self) -> None:
        """The fused box is the confidence-weighted mean; the score is the mean."""
        detections = _pair()

        fused = detections.with_fusion(threshold=0.5)

        weighted_corner = 0.8 / 1.7
        assert len(fused) == 1
        assert np.allclose(
            fused.xyxy,
            [
                [
                    weighted_corner,
                    weighted_corner,
                    10 + weighted_corner,
                    10 + weighted_corner,
                ]
            ],
        )
        assert np.allclose(fused.confidence, [0.85])

    def test_matches_box_weighted_box_fusion_on_stacked_predictions(self) -> None:
        """The method reproduces the array kernel's rows for the same inputs."""
        xyxy = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [2, 2, 12, 12]],
            dtype=np.float64,
        )
        confidence = np.array([0.9, 0.8, 0.7, 0.6])
        class_id = np.array([0, 0, 1, 1])
        detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        predictions = np.column_stack([xyxy, confidence, class_id])

        fused = detections.with_fusion(num_models=2)
        expected = _box_weighted_box_fusion(predictions, num_models=2)

        assert np.array_equal(fused.xyxy, expected[:, :4])
        assert np.array_equal(fused.confidence, expected[:, 4])
        assert np.array_equal(fused.class_id, expected[:, 5].astype(int))

    def test_class_aware_fusion_keeps_classes_apart(self) -> None:
        """Overlapping boxes of different classes stay separate detections."""
        detections = _pair(class_id=np.array([0, 1]))

        fused = detections.with_fusion()

        assert len(fused) == 2
        assert fused.class_id.tolist() == [0, 1]

    def test_class_agnostic_fusion_ignores_class_id(self) -> None:
        """Overlapping boxes of different classes fuse; the winner names the class."""
        detections = _pair(class_id=np.array([3, 5]))

        fused = detections.with_fusion(class_agnostic=True)

        assert len(fused) == 1
        assert fused.class_id.tolist() == [3]

    def test_class_agnostic_fusion_needs_no_class_id(self) -> None:
        """Without `class_id`, class-agnostic fusion still runs."""
        detections = _pair(class_id=None)

        fused = detections.with_fusion(class_agnostic=True)

        assert len(fused) == 1
        assert fused.class_id is None

    def test_winner_metadata_comes_from_highest_confidence_member(self) -> None:
        """`tracker_id` and `data` are the cluster seed's, not the later member's."""
        detections = _pair(
            confidence=np.array([0.5, 0.9]),
            tracker_id=np.array([7, 8]),
            data={"class_name": np.array(["low", "high"])},
        )

        fused = detections.with_fusion()

        assert fused.tracker_id.tolist() == [8]
        assert fused["class_name"].tolist() == ["high"]

    def test_tied_confidences_pick_the_first_member_as_winner(self) -> None:
        """With equal confidence the earlier input row supplies the metadata."""
        detections = _pair(
            confidence=np.array([0.7, 0.7]),
            data={"class_name": np.array(["first", "second"])},
        )

        fused = detections.with_fusion()

        assert fused["class_name"].tolist() == ["first"]

    def test_returns_rows_ordered_by_descending_confidence(self) -> None:
        """The fused detections are sorted by fused confidence, not input order."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float64),
            confidence=np.array([0.4, 0.9]),
            class_id=np.array([0, 0]),
        )

        fused = detections.with_fusion()

        assert fused.confidence.tolist() == [0.9, 0.4]
        assert fused.xyxy[0].tolist() == [50, 50, 60, 60]

    def test_num_models_demotes_boxes_missed_by_some_models(self) -> None:
        """A cluster backed by fewer detections than models is scaled down."""
        detections = Detections(
            xyxy=np.array(
                [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float64
            ),
            confidence=np.array([0.9, 0.8, 0.7]),
            class_id=np.array([0, 0, 0]),
        )

        fused = detections.with_fusion(num_models=2)

        assert np.allclose(fused.confidence, [0.85, 0.35])

    def test_overlap_metric_accepts_strings(self) -> None:
        """IoS fuses a small box nested in a large one that IoU keeps apart."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10], [0, 0, 4, 4]], dtype=np.float64),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 0]),
        )

        by_iou = detections.with_fusion(overlap_metric=OverlapMetric.IOU)
        by_ios = detections.with_fusion(overlap_metric="ios")

        assert len(by_iou) == 2
        assert len(by_ios) == 1

    @pytest.mark.parametrize(
        ("class_agnostic", "class_id", "expected_dtype"),
        [
            pytest.param(
                True, np.array([0, 0]), np.float32, id="agnostic-float32-stays-float32"
            ),
            pytest.param(
                False, np.array([0, 0]), np.float64, id="class-aware-int-class-id"
            ),
        ],
    )
    def test_output_dtype_follows_the_stacked_predictions(
        self,
        class_agnostic: bool,
        class_id: np.ndarray,
        expected_dtype: type[np.floating],
    ) -> None:
        """`xyxy` and `confidence` use the dtype stacking the inputs would give."""
        detections = _pair(
            xyxy=np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
            confidence=np.array([0.9, 0.8], dtype=np.float32),
            class_id=class_id,
        )

        fused = detections.with_fusion(class_agnostic=class_agnostic)

        assert fused.xyxy.dtype == expected_dtype
        assert fused.confidence.dtype == expected_dtype

    def test_integer_boxes_become_floating_point(self) -> None:
        """Fused coordinates are averages, so integer input widens to float."""
        detections = _pair(xyxy=np.array([[0, 0, 10, 10], [1, 1, 11, 11]]))

        fused = detections.with_fusion()

        assert np.issubdtype(fused.xyxy.dtype, np.floating)

    def test_empty_detections_are_returned_unchanged(self) -> None:
        """Empty input has nothing to fuse."""
        detections = Detections.empty()

        fused = detections.with_fusion()

        assert fused is detections

    def test_empty_detections_still_validate_arguments(self) -> None:
        """An invalid `overlap_on` raises even when there is nothing to fuse."""
        with pytest.raises(ValueError, match="Invalid value: polygon"):
            Detections.empty().with_fusion(overlap_on="polygon")

    def test_requires_confidence(self) -> None:
        """Fusion weights by confidence, so it cannot run without it."""
        detections = _pair(confidence=None)

        with pytest.raises(ValueError, match="confidence must be given"):
            detections.with_fusion()

    def test_requires_class_id_unless_class_agnostic(self) -> None:
        """Class-aware fusion needs class ids and names the escape hatch."""
        detections = _pair(class_id=None)

        with pytest.raises(ValueError, match="class_agnostic=True"):
            detections.with_fusion()

    @pytest.mark.parametrize(
        ("keyword", "value", "match"),
        [
            pytest.param("threshold", 1.5, "iou_threshold|threshold", id="threshold"),
            pytest.param("num_models", 0, "num_models", id="num-models"),
        ],
    )
    def test_rejects_invalid_arguments(
        self, keyword: str, value: float, match: str
    ) -> None:
        """Out-of-range `threshold` and `num_models` raise `ValueError`."""
        with pytest.raises(ValueError, match=match):
            _pair().with_fusion(**{keyword: value})


class TestWithFusionGeometry:
    """`with_fusion` rejects detections whose geometry is ambiguous or missing."""

    def test_mask_overlap_without_masks_raises(self) -> None:
        """`overlap_on='mask'` needs masks, as for the other methods."""
        with pytest.raises(ValueError, match=r"requires `Detections\.mask`"):
            _pair().with_fusion(overlap_on="mask")

    def test_masks_and_oriented_boxes_together_raise(self) -> None:
        """Carrying both geometries is ambiguous and always rejected."""
        corners = np.zeros((2, 4, 2), dtype=np.float64)
        detections = _masked_pair()
        detections.data[ORIENTED_BOX_COORDINATES] = corners

        with pytest.raises(ValueError, match="both masks and oriented boxes"):
            detections.with_fusion(overlap_on="box")


class TestWithFusionMasks:
    """With `overlap_on='box'`, a fused detection carries the union of its masks."""

    def test_dense_masks_are_united(self) -> None:
        """The fused mask is the logical OR of the cluster's masks."""
        detections = _masked_pair()

        fused = detections.with_fusion(overlap_on="box")

        expected = np.logical_or(detections.mask[0], detections.mask[1])
        assert fused.mask.shape == (1, 20, 20)
        assert np.array_equal(fused.mask[0], expected)

    def test_compact_masks_stay_compact_and_match_dense(self) -> None:
        """`CompactMask` input yields a `CompactMask` equal to the dense union."""
        dense = _masked_pair()
        compact = dense.to_compact_masks()

        fused = compact.with_fusion(overlap_on="box")

        assert isinstance(fused.mask, CompactMask)
        expected = dense.with_fusion(overlap_on="box").mask
        assert np.array_equal(fused.mask.to_dense(), expected)

    def test_compact_mask_subclass_keeps_its_selection_path(self) -> None:
        """Fuse compact masks without reconstructing a Detections subclass."""

        class CompactDetections(Detections):
            def __init__(self, source: Detections) -> None:
                """Initialize a compact detection subclass from existing rows."""
                compact = source.to_compact_masks()
                super().__init__(
                    xyxy=compact.xyxy,
                    mask=compact.mask,
                    confidence=compact.confidence,
                    class_id=compact.class_id,
                )

        dense = _masked_pair()
        detections = CompactDetections(dense)

        fused = detections.with_fusion(overlap_on="box")

        assert isinstance(fused.mask, CompactMask)
        assert np.array_equal(
            fused.mask.to_dense(), dense.with_fusion(overlap_on="box").mask
        )

    def test_unfused_detection_keeps_its_mask(self) -> None:
        """A detection that fuses with nothing keeps its mask unchanged."""
        masks = np.zeros((2, 20, 20), dtype=bool)
        masks[0, 0:5, 0:5] = True
        masks[1, 10:15, 10:15] = True
        detections = Detections(
            xyxy=np.array([[0, 0, 5, 5], [10, 10, 15, 15]], dtype=np.float64),
            mask=masks,
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 0]),
        )

        fused = detections.with_fusion(overlap_on="box")

        assert np.array_equal(fused.mask, masks)

    @pytest.mark.parametrize("compact", [False, True])
    def test_reordered_clusters_keep_their_masks(self, compact: bool) -> None:
        """Keep each union with its output row and leave input masks unchanged."""
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[0, :10, :10] = True
        masks[1, 1:11, 1:11] = True
        masks[2, 15:20, 15:20] = True
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10], [1, 1, 11, 11], [15, 15, 20, 20]]),
            mask=masks.copy(),
            confidence=np.array([0.9, 0.5, 0.8]),
            class_id=np.zeros(3, dtype=int),
        )
        detections = detections.to_compact_masks() if compact else detections

        fused = detections.with_fusion(overlap_on="box")

        output_masks = (
            fused.mask.to_dense() if isinstance(fused.mask, CompactMask) else fused.mask
        )
        input_masks = (
            detections.mask.to_dense()
            if isinstance(detections.mask, CompactMask)
            else detections.mask
        )
        np.testing.assert_allclose(fused.confidence, [0.8, 0.7])
        np.testing.assert_array_equal(output_masks, [masks[2], masks[0] | masks[1]])
        np.testing.assert_array_equal(input_masks, masks)


class TestWithFusionOrientedBoxes:
    """Oriented boxes fuse in the frame of each cluster's highest-confidence member."""

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_angle_zero_boxes_match_axis_aligned_fusion(self, threshold: float) -> None:
        """Unrotated boxes fuse to the same boxes and scores as without corners."""
        rng = np.random.default_rng(0)
        centers = rng.uniform(0, 300, (40, 2))
        sizes = rng.uniform(30, 90, (40, 2))
        # Duplicate each box with a small jitter so clusters have 2 or more members.
        xyxy = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
        xyxy = np.concatenate([xyxy, xyxy + rng.normal(0, 2, xyxy.shape)])
        confidence = rng.uniform(0.1, 1.0, len(xyxy))
        class_id = np.zeros(len(xyxy), dtype=int)
        axis = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        oriented = _oriented(_axis_corners(xyxy), list(confidence))

        expected = axis.with_fusion(threshold=threshold)
        result = oriented.with_fusion(threshold=threshold)

        np.testing.assert_allclose(result.xyxy, expected.xyxy, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.confidence, expected.confidence)

    def test_threshold_tie_follows_the_float32_polygon_overlap(self) -> None:
        """A pair whose overlap equals the threshold in float64 fuses as oriented boxes.

        Oriented overlap runs in float32 polygon arithmetic, so a pair that the axis-
        aligned float64 path leaves apart (overlap equal to the threshold) can end just
        above it. The fusion of the two paths is only equal away from ties.
        """
        xyxy = np.array([[27.2, 0.1, 41.8, 47.8], [36.5, 1.7, 43.9, 9.8]])
        threshold = 0.06017408855809258
        axis = Detections(
            xyxy=xyxy, confidence=np.array([0.9, 0.8]), class_id=np.array([0, 0])
        )
        oriented = _oriented(_axis_corners(xyxy), [0.9, 0.8])

        axis_count = len(axis.with_fusion(threshold=threshold))
        oriented_count = len(oriented.with_fusion(threshold=threshold))

        assert (axis_count, oriented_count) == (2, 1)

    def test_rotated_boxes_are_fused_in_the_seed_frame(self) -> None:
        """The fused box is the weighted mean of the extremes along the seed's axes."""
        seed = _rotated_box(100, 40, 30, (500, 500))
        other = _rotated_box(100, 40, 30, (500, 500), shift=6)
        detections = _oriented(np.stack([seed, other]), [0.9, 0.6])
        angle = np.deg2rad(30)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        # Local x extremes: (0 * 0.9 + 6 * 0.6) / 1.5 and (100 * 0.9 + 106 * 0.6) / 1.5.
        local = np.array([[2.4, 0], [102.4, 0], [102.4, 40], [2.4, 40]])
        expected = local @ rotation.T + 500

        result = detections.with_fusion()

        assert len(result) == 1
        np.testing.assert_allclose(
            result.data[ORIENTED_BOX_COORDINATES][0], expected, atol=1e-9
        )
        np.testing.assert_allclose(result.confidence, [0.75])
        envelope = np.concatenate([expected.min(axis=0), expected.max(axis=0)])
        np.testing.assert_allclose(result.xyxy[0], envelope)

    def test_fused_corners_form_a_rectangle(self) -> None:
        """A cluster of rotated boxes fuses into a rectangle with right angles."""
        corners = np.stack(
            [
                _rotated_box(80, 30, 20, (100, 100)),
                _rotated_box(84, 28, 20, (102, 99), shift=3),
                _rotated_box(78, 32, 20, (99, 101), shift=-2),
            ]
        )
        detections = _oriented(corners, [0.9, 0.7, 0.5])

        fused = detections.with_fusion().data[ORIENTED_BOX_COORDINATES][0]

        edges = np.roll(fused, -1, axis=0) - fused
        dots = (edges * np.roll(edges, -1, axis=0)).sum(axis=1)
        np.testing.assert_allclose(dots, 0, atol=1e-9)

    @pytest.mark.parametrize(
        "corners",
        [
            pytest.param(_rotated_box(60, 20, 35, (50, 50)), id="rotated"),
            pytest.param(
                np.array([[40, 10], [40, 50], [10, 50], [10, 10]], dtype=float),
                id="unrotated-other-corner-order",
            ),
            pytest.param(
                np.array([[0, 0], [30, 0], [24, 10], [6, 10]], dtype=float),
                id="non-rectangular-quad",
            ),
        ],
    )
    def test_unfused_box_comes_back_unchanged(self, corners: np.ndarray) -> None:
        """A detection that joins no cluster keeps its corners and `xyxy`."""
        far_away = _axis_corners(np.array([[500.0, 500, 520, 520]]))[0]
        detections = _oriented(np.stack([corners, far_away]), [0.9, 0.8])

        result = detections.with_fusion()

        np.testing.assert_array_equal(result.data[ORIENTED_BOX_COORDINATES][0], corners)
        np.testing.assert_array_equal(result.xyxy[0], detections.xyxy[0])

    @pytest.mark.parametrize(
        ("dtype", "offset"),
        [
            pytest.param(np.float32, 1000, id="float32"),
            pytest.param(np.float64, 10**9, id="float64-large"),
            pytest.param(np.int64, 2**53 - 1000, id="int64-near-2-pow-53"),
        ],
    )
    def test_unfused_corners_keep_their_value_in_float64(
        self, dtype: type, offset: int
    ) -> None:
        """Float inputs and integers up to 2**53 round-trip into a float64 array."""
        corners = (_axis_corners(np.array([[0.0, 0, 40, 30]])) + offset).astype(dtype)
        xyxy = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
        detections = Detections(
            xyxy=xyxy.astype(np.float64),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        result = detections.with_fusion()

        fused = result.data[ORIENTED_BOX_COORDINATES]
        assert fused.dtype == np.float64
        np.testing.assert_array_equal(fused, corners.astype(np.float64))

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((4, 2), id="corner-layout"),
            pytest.param((8,), id="flat-layout"),
        ],
    )
    def test_fused_corners_keep_the_input_layout(self, shape: tuple[int, ...]) -> None:
        """Corners come back as `(N, 4, 2)` or flat `(N, 8)`, as they went in."""
        corners = np.stack(
            [
                _rotated_box(80, 30, 20, (100, 100)),
                _rotated_box(80, 30, 20, (101, 100)),
                _rotated_box(80, 30, 20, (400, 400)),
            ]
        )
        detections = _oriented(corners.reshape(3, *shape), [0.9, 0.8, 0.7])

        result = detections.with_fusion()

        assert result.data[ORIENTED_BOX_COORDINATES].shape == (2, *shape)

    def test_large_coordinates_stay_finite_and_match_the_local_result(self) -> None:
        """Coordinates near 1e9 give the same fused box as near the origin."""
        local = np.stack(
            [_rotated_box(80, 30, 20, (0, 0)), _rotated_box(80, 30, 20, (2, 1))]
        )
        near_origin = _oriented(local, [0.9, 0.8]).with_fusion()

        far = _oriented(local + 1e9, [0.9, 0.8]).with_fusion()

        fused = far.data[ORIENTED_BOX_COORDINATES]
        assert np.isfinite(fused).all()
        np.testing.assert_allclose(
            fused - 1e9, near_origin.data[ORIENTED_BOX_COORDINATES], atol=1e-6
        )

    def test_zero_confidence_cluster_uses_the_plain_average(self) -> None:
        """When every confidence is zero the extents are averaged without weights."""
        xyxy = np.array([[0.0, 0, 100, 100], [4, 2, 104, 98], [2, 6, 102, 106]])
        detections = _oriented(_axis_corners(xyxy), [0.0, 0.0, 0.0])

        result = detections.with_fusion()

        assert len(result) == 1
        np.testing.assert_allclose(result.xyxy[0], xyxy.mean(axis=0))

    def test_winner_metadata_and_class_handling(self) -> None:
        """The fused detection takes class, tracker and data from the best member."""
        corners = np.stack(
            [_rotated_box(80, 30, 20, (100, 100)), _rotated_box(80, 30, 20, (101, 100))]
        )
        detections = _oriented(
            corners,
            [0.5, 0.9],
            class_id=np.array([3, 7]),
            tracker_id=np.array([11, 22]),
        )
        detections.data["name"] = np.array(["low", "high"])

        class_agnostic = detections.with_fusion(class_agnostic=True)
        class_aware = detections.with_fusion()

        assert class_agnostic.class_id.tolist() == [7]
        assert class_agnostic.tracker_id.tolist() == [22]
        assert class_agnostic.data["name"].tolist() == ["high"]
        assert len(class_aware) == 2

    def test_num_models_rescales_the_confidence(self) -> None:
        """A cluster backed by fewer detections than models is demoted."""
        corners = np.stack(
            [_rotated_box(80, 30, 20, (100, 100)), _rotated_box(80, 30, 20, (101, 100))]
        )
        detections = _oriented(corners, [0.9, 0.7])

        result = detections.with_fusion(num_models=4)

        np.testing.assert_allclose(result.confidence, [0.8 * 2 / 4])

    def test_explicit_box_overlap_matches_the_default(self) -> None:
        """`overlap_on="box"` fuses oriented detections the same way as `None`."""
        corners = np.stack(
            [_rotated_box(80, 30, 20, (100, 100)), _rotated_box(80, 30, 20, (101, 100))]
        )
        detections = _oriented(corners, [0.9, 0.7])

        default = detections.with_fusion()
        explicit = detections.with_fusion(overlap_on="box")

        np.testing.assert_array_equal(
            default.data[ORIENTED_BOX_COORDINATES],
            explicit.data[ORIENTED_BOX_COORDINATES],
        )

    @pytest.mark.parametrize(
        ("corners", "message"),
        [
            pytest.param(
                np.full((2, 4, 2), np.nan),
                "finite corner coordinates",
                id="nan",
            ),
            pytest.param(
                np.full((2, 4, 2), np.inf),
                "finite corner coordinates",
                id="inf",
            ),
            pytest.param(np.zeros((2, 3, 2)), r"shape \(n, 4, 2\)", id="three-corners"),
        ],
    )
    def test_invalid_corners_raise(self, corners: np.ndarray, message: str) -> None:
        """Non-finite or wrongly shaped corners are rejected."""
        detections = _pair(data={ORIENTED_BOX_COORDINATES: corners})

        with pytest.raises(ValueError, match=message):
            detections.with_fusion()

    @pytest.mark.parametrize(
        ("width", "shift", "threshold", "expected"),
        [
            pytest.param(1, 0, 0.5, 1, id="identical-unit-boxes"),
            pytest.param(2, 1, 0.4, 2, id="one-pixel-overlap"),
            pytest.param(2, 1, 0.3, 1, id="overlap-above-threshold"),
        ],
    )
    def test_integer_overlap_survives_large_origin(
        self, width: int, shift: int, threshold: float, expected: int
    ) -> None:
        """Integer translation does not change cluster membership."""
        local = np.array([[0, 0, width, 2], [shift, 0, width + shift, 2]])
        corners = _axis_corners(local + np.array([2**53, 0, 2**53, 0]))

        result = _oriented(corners, [0.9, 0.8]).with_fusion(threshold=threshold)

        assert len(result) == expected

    def test_reversed_oriented_bounds_raise(self) -> None:
        """Oriented fusion rejects malformed axis-aligned bounds."""
        corners = _axis_corners(np.array([[0, 0, 10, 10]]))
        detections = _oriented(corners, [0.9], xyxy=np.array([[10, 0, 0, 10]]))

        with pytest.raises(ValueError, match="x1 <= x2"):
            detections.with_fusion()

    def test_fused_integer_corners_keep_representable_edge(self) -> None:
        """World coordinates round after the local offsets are added."""
        origin = 2**53 + 1
        xyxy = np.array([[origin, 0, origin + 1, 2]] * 2)
        detections = _oriented(_axis_corners(xyxy), [0.9, 0.8])

        result = detections.with_fusion()

        assert len(result) == 1
        assert result.xyxy[0, 2] == float(origin + 1)

    @pytest.mark.skipif(
        np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
        reason="requires extended precision",
    )
    def test_singleton_and_nmm_keep_extended_origin(self) -> None:
        """The output cast and NMM frame preserve an extended-precision origin."""
        origin = np.longdouble(2**53) + 1
        xyxy = np.array([[origin, 0, origin + 8, 8]], dtype=np.longdouble)
        corners = _axis_corners(xyxy)
        singleton = _oriented(corners, [0.9])
        pair = _oriented(np.repeat(corners, 2, axis=0), [0.9, 0.8])

        fused = singleton.with_fusion()
        merged = pair.with_nmm()

        np.testing.assert_array_equal(fused.xyxy, xyxy)
        np.testing.assert_array_equal(merged.data[ORIENTED_BOX_COORDINATES], corners)

    @pytest.mark.parametrize(
        ("winner", "extent", "expected"),
        [
            pytest.param(
                0,
                3.25,
                [[-3.25, -3.25], [3.25, -3.25], [3.25, 3.25], [-3.25, 3.25]],
                id="square-wins",
            ),
            pytest.param(
                1, 4.5, [[0, -4.5], [4.5, 0], [0, 4.5], [-4.5, 0]], id="diamond-wins"
            ),
        ],
    )
    def test_mixed_angle_extrema_follow_the_winner(
        self, winner: int, extent: float, expected: list[list[float]]
    ) -> None:
        """Mixed angles produce the weighted extents measured in the winner's frame."""
        square = np.array([[-3, -3], [3, -3], [3, 3], [-3, 3]])
        diamond = np.array([[0, -4], [4, 0], [0, 4], [-4, 0]])
        corners = np.stack([square, diamond])[[1 - winner, winner]]

        result = _oriented(corners, [0.25, 0.75]).with_fusion()

        assert len(result) == 1
        np.testing.assert_allclose(
            result.data[ORIENTED_BOX_COORDINATES][0], expected, atol=1e-12
        )
        np.testing.assert_allclose(result.xyxy, [[-extent, -extent, extent, extent]])


class EncodedMasks(CompactMask):
    """Count materializations of the original stored masks."""

    decodes = 0

    def crop(self, index: int) -> npt.NDArray[np.bool_]:
        """Count reads through the decoded crop API."""
        self.decodes += 1
        return super().crop(index)


class TestWithFusionMaskOverlap:
    """Masks are fused by overlap of the masks, with a confidence-weighted vote."""

    def test_equal_confidence_pair_fuses_to_the_union(self) -> None:
        """Two members of equal confidence keep every pixel that either covers."""
        masks = _rect_masks((20, 30), [(0, 0, 10, 10), (4, 0, 14, 10)])
        detections = _with_masks(masks, [0.8, 0.8])

        result = detections.with_fusion(threshold=0.3)

        assert len(result) == 1
        np.testing.assert_array_equal(result.mask[0], masks.any(axis=0))
        np.testing.assert_allclose(result.xyxy[0], [2, 0, 12, 10])
        np.testing.assert_allclose(result.confidence, [0.8])

    def test_unequal_confidence_pair_keeps_the_winner_mask(self) -> None:
        """A pixel only the weaker member covers has too little weight to stay."""
        masks = _rect_masks((20, 30), [(0, 0, 10, 10), (4, 0, 14, 10)])
        detections = _with_masks(masks, [0.9, 0.6])

        result = detections.with_fusion(threshold=0.3)

        np.testing.assert_array_equal(result.mask[0], masks[0])
        np.testing.assert_allclose(result.confidence, [0.75])

    def test_a_pixel_at_exactly_half_the_weight_is_kept(self) -> None:
        """Votes equal to half the total weight keep the pixel."""
        masks = _rect_masks((10, 10), [(0, 0, 5, 5), (3, 0, 8, 5), (3, 0, 8, 5)])
        detections = _with_masks(masks, [0.5, 0.3, 0.2])

        result = detections.with_fusion(threshold=0.2)

        assert len(result) == 1
        np.testing.assert_array_equal(result.mask[0], masks.any(axis=0))

    @pytest.mark.parametrize("compact", [False, True])
    @pytest.mark.parametrize(
        ("scores", "expected"),
        [
            pytest.param([0.75, 0.75, 0.1, 0.1], [True, True, True], id="tie"),
            pytest.param(
                [0.75, 0.75, 0.1, 0.1, 0.05],
                [True, True, False],
                id="later-member-joins",
            ),
        ],
    )
    def test_rounded_total_does_not_drop_exact_ties(
        self, compact: bool, scores: list[float], expected: list[bool]
    ) -> None:
        """Equal weighted fringes survive and admit a later mask by overlap."""
        masks = _rect_masks((1, 3), [(0, 0, 2, 1), (1, 0, 3, 1)] * 2 + [(0, 0, 1, 1)])[
            : len(scores)
        ]
        stored_masks = (
            CompactMask.from_dense(
                masks, np.tile([0, 0, 2, 0], (len(scores), 1)), (1, 3)
            )
            if compact
            else masks
        )
        detections = _with_masks(masks, scores, mask=stored_masks)

        result = detections.with_fusion(threshold=0.25)

        np.testing.assert_array_equal(np.asarray(result.mask), [[expected]])

    def test_adjacent_confidences_do_not_create_a_false_tie(self) -> None:
        """A runner-up-only pixel stays below half even for adjacent floats."""
        masks = _rect_masks((1, 3), [(0, 0, 2, 1), (1, 0, 3, 1)])
        detections = _with_masks(masks, [float(np.nextafter(0.75, 1.0)), 0.75])

        result = detections.with_fusion(threshold=0.25)

        np.testing.assert_array_equal(result.mask, masks[:1])

    @pytest.mark.parametrize("compact", [False, True])
    def test_subnormal_confidence_can_break_a_half_weight_tie(
        self, compact: bool
    ) -> None:
        """A positive subnormal vote breaks a tie across multiple vote batches."""
        masks = _rect_masks((70, 100), [(0, 0, 70, 70)] + [(30, 0, 100, 70)] * 3)
        stored_masks = (
            CompactMask.from_dense(masks, np.tile([0, 0, 99, 69], (4, 1)), (70, 100))
            if compact
            else masks
        )
        scores = [1.0, 0.5, 0.5, float(np.nextafter(0.0, 1.0))]
        detections = _with_masks(masks, scores, mask=stored_masks)

        result = detections.with_fusion(threshold=0.25)

        np.testing.assert_array_equal(np.asarray(result.mask), masks[1:2])

    def test_finite_extreme_boxes_have_a_finite_weighted_mean(self) -> None:
        """Mask overlap allows finite boxes whose coordinate difference overflows."""
        masks = np.ones((2, 1, 1), dtype=bool)
        detections = _with_masks(
            masks,
            [1.0, 1.0],
            xyxy=np.array([[-1e308, 0, -1e308, 1], [1e308, 0, 1e308, 1]]),
        )

        result = detections.with_fusion()

        np.testing.assert_array_equal(result.xyxy, [[0.0, 0.0, 0.0, 1.0]])

    def test_maximum_threshold_keeps_compact_singletons_without_decoding(self) -> None:
        """A threshold of one admits no merge and needs no decoded masks."""
        masks = np.ones((2, 4, 4), dtype=bool)
        compact = EncodedMasks.from_dense(masks, np.tile([0, 0, 3, 3], (2, 1)), (4, 4))
        detections = _with_masks(masks, [0.6, 0.8], mask=compact)

        result = detections.with_fusion(threshold=1.0)

        assert compact.decodes == 0
        np.testing.assert_array_equal(result.confidence, [0.8, 0.6])
        np.testing.assert_array_equal(np.asarray(result.mask), masks)

    def test_compact_singleton_overlaps_do_not_redecode_seeds(self) -> None:
        """Overlapping bounds with disjoint foreground keep independent encoded rows."""
        masks = np.zeros((4, 10, 10), dtype=bool)
        ids = np.arange(4)
        masks[ids, ids, ids] = True
        masks[ids, ids + 6, ids + 6] = True
        compact = EncodedMasks.from_dense(
            masks, np.tile([0, 0, 9, 9], (4, 1)), (10, 10)
        )
        detections = _with_masks(masks, [0.9, 0.8, 0.7, 0.6], mask=compact)

        result = detections.with_fusion()

        assert compact.decodes <= 4
        np.testing.assert_array_equal(np.asarray(result.mask), masks)

    def test_subclass_select_receives_original_masks(self) -> None:
        """Selection overrides can use input masks and contribute metadata."""

        class MaskAwareDetections(Detections):
            """Record the source mask area during winner selection."""

            def select(
                self,
                index: int | np.integer | slice | list[int] | npt.NDArray[np.generic],
            ) -> Detections:
                """Read masks before delegating to the base selection."""
                assert self.mask is not None
                result = super().select(index)
                result.metadata["selected_mask_area"] = int(np.asarray(self.mask).sum())
                return result

        masks = np.ones((2, 4, 4), dtype=bool)
        detections = MaskAwareDetections(
            xyxy=np.tile([0, 0, 4, 4], (2, 1)),
            mask=masks,
            confidence=np.array([0.8, 0.6]),
            class_id=np.zeros(2, dtype=int),
        )

        result = detections.with_fusion()

        assert result.metadata["selected_mask_area"] == 32
        np.testing.assert_array_equal(result.mask, masks[:1])

    def test_zero_confidence_members_vote_equally(self) -> None:
        """With every confidence zero a pixel needs at least half of the members."""
        masks = _rect_masks((10, 10), [(0, 0, 6, 10), (2, 0, 8, 10), (2, 0, 8, 10)])
        detections = _with_masks(masks, [0.0, 0.0, 0.0])

        result = detections.with_fusion(threshold=0.3)

        np.testing.assert_array_equal(
            result.mask[0], _rect_masks((10, 10), [(2, 0, 8, 10)])[0]
        )

    def test_unfused_detection_keeps_its_mask_and_box(self) -> None:
        """A detection that joins no cluster comes back unchanged."""
        masks = _rect_masks((20, 20), [(0, 0, 6, 6), (12, 12, 18, 18)])
        detections = _with_masks(masks, [0.9, 0.8])

        result = detections.with_fusion()

        np.testing.assert_array_equal(result.mask, masks)
        np.testing.assert_array_equal(result.xyxy, detections.xyxy)

    def test_masks_decide_the_overlap_not_the_boxes(self) -> None:
        """Overlapping boxes of disjoint masks stay apart, unlike box fusion."""
        masks = _rect_masks((10, 10), [(0, 0, 5, 10), (5, 0, 10, 10)])
        detections = _with_masks(masks, [0.9, 0.8])
        detections.xyxy = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)

        by_mask = detections.with_fusion()
        by_box = detections.with_fusion(overlap_on="box")

        assert (len(by_mask), len(by_box)) == (2, 1)

    @pytest.mark.parametrize(
        ("metric", "expected_count"),
        [
            pytest.param(OverlapMetric.IOU, 2, id="iou"),
            pytest.param(OverlapMetric.IOS, 1, id="ios"),
        ],
    )
    def test_overlap_metric_applies_to_masks(
        self, metric: OverlapMetric, expected_count: int
    ) -> None:
        """A small mask inside a large one joins it under IoS but not under IoU."""
        masks = _rect_masks((20, 20), [(0, 0, 20, 20), (2, 2, 6, 6)])
        detections = _with_masks(masks, [0.9, 0.8])

        result = detections.with_fusion(overlap_metric=metric)

        assert len(result) == expected_count

    def test_explicit_mask_overlap_matches_the_default(self) -> None:
        """`overlap_on="mask"` fuses masked detections the way `None` does."""
        masks = _rect_masks((20, 30), [(0, 0, 10, 10), (4, 0, 14, 10)])
        detections = _with_masks(masks, [0.8, 0.8])

        default = detections.with_fusion(threshold=0.3)
        explicit = detections.with_fusion(threshold=0.3, overlap_on=OverlapOn.MASK)

        np.testing.assert_array_equal(default.mask, explicit.mask)
        np.testing.assert_array_equal(default.xyxy, explicit.xyxy)

    def test_winner_metadata_class_agnostic_and_num_models(self) -> None:
        """Metadata comes from the best member, and `num_models` demotes the score."""
        masks = _rect_masks((20, 30), [(0, 0, 10, 10), (4, 0, 14, 10)])
        detections = _with_masks(
            masks,
            [0.5, 0.9],
            class_id=np.array([3, 7]),
            tracker_id=np.array([11, 22]),
        )

        agnostic = detections.with_fusion(
            threshold=0.3, class_agnostic=True, num_models=4
        )
        class_aware = detections.with_fusion(threshold=0.3)

        assert agnostic.class_id.tolist() == [7]
        assert agnostic.tracker_id.tolist() == [22]
        np.testing.assert_allclose(agnostic.confidence, [0.7 * 2 / 4])
        assert len(class_aware) == 2

    def test_compact_masks_stay_compact_and_match_dense(self) -> None:
        """A `CompactMask` input gives a `CompactMask` holding the dense result."""
        masks = _rect_masks(
            (40, 40), [(2, 2, 14, 14), (6, 2, 18, 14), (28, 28, 38, 38)]
        )
        dense = _with_masks(masks, [0.9, 0.7, 0.6])
        compact = _with_masks(
            CompactMask.from_dense(masks, dense.xyxy - [0, 0, 1, 1], (40, 40)),
            [0.9, 0.7, 0.6],
        )

        expected = dense.with_fusion(threshold=0.3)
        result = compact.with_fusion(threshold=0.3)

        assert isinstance(result.mask, CompactMask)
        np.testing.assert_array_equal(result.mask.to_dense(), expected.mask)
        np.testing.assert_array_equal(result.xyxy, expected.xyxy)

    def test_masks_touching_the_image_border_are_painted_in_place(self) -> None:
        """A fused mask at the image corner keeps its position and shape."""
        masks = _rect_masks((12, 12), [(0, 0, 6, 6), (0, 0, 6, 6), (0, 0, 6, 6)])
        detections = _with_masks(masks, [0.9, 0.8, 0.7])

        result = detections.with_fusion()

        assert result.mask.shape == (1, 12, 12)
        np.testing.assert_array_equal(result.mask[0], masks[0])

    def test_masks_and_oriented_boxes_still_raise(self) -> None:
        """Carrying both geometries stays ambiguous, whatever `overlap_on` says."""
        masks = _rect_masks((10, 10), [(0, 0, 6, 6), (1, 1, 7, 7)])
        detections = _with_masks(
            masks, [0.9, 0.8], data={ORIENTED_BOX_COORDINATES: np.zeros((2, 4, 2))}
        )

        with pytest.raises(ValueError, match="both masks and oriented boxes"):
            detections.with_fusion(overlap_on="mask")


def _fusion_detections(
    kind: str,
    rectangles: list[tuple[int, int, int, int]],
    confidence: list[float],
    **overrides: object,
) -> Detections:
    """Detections of one geometry (`box`, `oriented` or `mask`) from rectangles."""
    boxes = np.array(rectangles, dtype=np.float64)
    if kind == "oriented":
        return _oriented(_axis_corners(boxes), confidence, **overrides)
    if kind == "mask":
        return _with_masks(_rect_masks((30, 30), rectangles), confidence, **overrides)
    fields: dict[str, object] = {
        "xyxy": boxes,
        "confidence": np.array(confidence),
        "class_id": np.zeros(len(boxes), dtype=int),
    }
    fields.update(overrides)
    return Detections(**fields)  # type: ignore[arg-type]


_GEOMETRIES = [
    pytest.param("box", id="box"),
    pytest.param("oriented", id="oriented"),
    pytest.param("mask", id="mask"),
]

#: Two models: model 0 found the object twice, model 1 once; the second copy of
#: model 0 cannot join the cluster that already holds model 0's box.
_TWO_MODELS = [(0, 0, 10, 10), (1, 1, 11, 11), (0, 0, 10, 10)]


class TestWithFusionSources:
    """`sources` keeps one detection per model in a cluster."""

    def test_near_copies_of_one_model_do_not_inflate_the_cluster(self) -> None:
        """Four models with a 0.02 near-copy each fuse to 0.8, not the 0.41 mean."""
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (8, 1))
        confidence = np.array([0.8, 0.02] * 4)
        detections = Detections(
            xyxy=boxes, confidence=confidence, class_id=np.zeros(8, dtype=int)
        )

        without_sources = detections.with_fusion(num_models=4)
        with_sources = detections.with_fusion(
            num_models=4, sources=np.repeat(np.arange(4), 2)
        )

        np.testing.assert_allclose(without_sources.confidence, [0.41])
        np.testing.assert_allclose(with_sources.confidence, [0.8, 0.02])

    def test_a_model_that_missed_the_object_counts_as_zero(self) -> None:
        """Three of four models seeing an object give the sum over four models."""
        detections = Detections(
            xyxy=np.tile([0.0, 0.0, 10.0, 10.0], (3, 1)),
            confidence=np.array([0.9, 0.6, 0.3]),
            class_id=np.zeros(3, dtype=int),
        )

        result = detections.with_fusion(sources=[0, 1, 2], num_models=4)

        np.testing.assert_allclose(result.confidence, [0.45])

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    def test_each_geometry_keeps_one_detection_per_model(self, kind: str) -> None:
        """Boxes, oriented boxes and masks all leave the near-copy of a model apart."""
        detections = _fusion_detections(kind, _TWO_MODELS, [0.9, 0.8, 0.05])

        result = detections.with_fusion(sources=[0, 1, 0], num_models=2)

        np.testing.assert_allclose(result.confidence, [0.85, 0.025])

    def test_sources_may_be_a_list_and_class_agnostic(self) -> None:
        """A plain list of ids works, and `class_agnostic` ignores class ids."""
        detections = _pair(class_id=np.array([0, 1]))

        result = detections.with_fusion(
            class_agnostic=True, sources=[0, 1], num_models=2
        )

        assert len(result) == 1
        np.testing.assert_allclose(result.confidence, [0.85])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param(
                {"sources": [0, 1]}, "num_models is required", id="without-num-models"
            ),
            pytest.param(
                {"sources": [0, 2], "num_models": 2},
                "must lie in",
                id="id-out-of-range",
            ),
            pytest.param(
                {"sources": [-1, 0], "num_models": 2}, "must lie in", id="negative-id"
            ),
            pytest.param(
                {"sources": [0, 1, 1], "num_models": 2},
                "one id per detection",
                id="wrong-length",
            ),
            pytest.param(
                {"sources": [0.0, 1.0], "num_models": 2},
                "must be integers",
                id="float-ids",
            ),
        ],
    )
    def test_invalid_sources_raise(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        """Ids must be integers in `[0, num_models)`, one per detection."""
        with pytest.raises(ValueError, match=message):
            _pair().with_fusion(**kwargs)  # type: ignore[arg-type]

    def test_sources_of_empty_detections_are_still_checked(self) -> None:
        """An empty input still rejects a `sources` array that is not empty."""
        with pytest.raises(ValueError, match="one id per detection"):
            Detections.empty().with_fusion(sources=[0], num_models=1)


class TestWithFusionBoxWeights:
    """`box_weights` chooses how the members of a cluster are averaged."""

    @pytest.mark.parametrize(
        "box_weights",
        [
            pytest.param("uniform", id="string"),
            pytest.param("UNIFORM", id="upper-case-string"),
            pytest.param(FusionBoxWeights.UNIFORM, id="enum"),
        ],
    )
    def test_uniform_weights_average_boxes_and_keep_the_mean_score(
        self, box_weights: FusionBoxWeights | str
    ) -> None:
        """Equal weights give the plain mean box, and the score stays the mean."""
        detections = _fusion_detections(
            "box", [(0, 0, 10, 10), (2, 0, 12, 10)], [0.9, 0.1]
        )

        result = detections.with_fusion(box_weights=box_weights)

        np.testing.assert_allclose(result.xyxy, [[1, 0, 11, 10]])
        np.testing.assert_allclose(result.confidence, [0.5])

    def test_confidence_weights_are_the_default(self) -> None:
        """Without `box_weights` the boxes are averaged by their confidence."""
        detections = _fusion_detections(
            "box", [(0, 0, 10, 10), (2, 0, 12, 10)], [0.9, 0.1]
        )

        result = detections.with_fusion()

        np.testing.assert_allclose(result.xyxy, [[0.2, 0, 10.2, 10]])

    def test_unknown_box_weights_raise(self) -> None:
        """A string that names no member is rejected, even for empty input."""
        with pytest.raises(ValueError, match="Invalid value"):
            Detections.empty().with_fusion(box_weights="median")

    def test_enum_members_are_listed(self) -> None:
        """The enum lists its string values."""
        assert FusionBoxWeights.list() == ["confidence", "uniform"]


class TestWithFusionSourceWeights:
    """`source_weights` makes a model count more or less in the fused result."""

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    def test_weights_scale_the_confidence(self, kind: str) -> None:
        """The fused confidence is the weighted sum over the sum of the weights."""
        detections = _fusion_detections(kind, _TWO_MODELS, [0.9, 0.8, 0.05])

        result = detections.with_fusion(
            sources=[0, 1, 0], num_models=2, source_weights=[3.0, 1.0]
        )

        np.testing.assert_allclose(result.confidence, [0.875, 0.0375])

    def test_weights_scale_the_box_mean(self) -> None:
        """Each member counts by its confidence times its source's weight."""
        detections = _pair()

        result = detections.with_fusion(
            sources=[0, 1], num_models=2, source_weights=[3.0, 1.0]
        )

        # weights 0.9 and 0.8 / 3: x1 = (0.8 / 3) / (0.9 + 0.8 / 3)
        np.testing.assert_allclose(result.xyxy[0, 0], 0.8 / 3 / (0.9 + 0.8 / 3))

    def test_uniform_boxes_use_only_the_source_weights(self) -> None:
        """With uniform weights the box mean is weighted by the source weights."""
        result = _pair().with_fusion(
            sources=[0, 1],
            num_models=2,
            box_weights="uniform",
            source_weights=[3.0, 1.0],
        )

        np.testing.assert_allclose(result.xyxy[0, 0], 0.25)
        np.testing.assert_allclose(result.confidence, [0.875])

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    def test_equal_weights_match_no_weights(self, kind: str) -> None:
        """Equal weights, whatever their size, give the unweighted result."""
        detections = _fusion_detections(kind, _TWO_MODELS, [0.9, 0.8, 0.05])
        expected = detections.with_fusion(sources=[0, 1, 0], num_models=2)

        result = detections.with_fusion(
            sources=[0, 1, 0], num_models=2, source_weights=[2.5, 2.5]
        )

        np.testing.assert_array_equal(result.xyxy, expected.xyxy)
        np.testing.assert_array_equal(result.confidence, expected.confidence)

    def test_a_down_weighted_model_loses_its_mask_pixels(self) -> None:
        """A weak model's own pixels leave the vote, so the fused mask is the strong
        one."""
        masks = _rect_masks((20, 20), [(0, 0, 10, 10), (2, 0, 12, 10)])
        detections = _with_masks(masks, [0.6, 0.6])

        union = detections.with_fusion(sources=[0, 1], num_models=2)
        weighted = detections.with_fusion(
            sources=[0, 1], num_models=2, source_weights=[1.0, 0.2]
        )

        np.testing.assert_array_equal(union.mask[0], masks.any(axis=0))
        np.testing.assert_array_equal(weighted.mask[0], masks[0])

    def test_oriented_boxes_match_axis_aligned_boxes_at_angle_zero(self) -> None:
        """Unrotated oriented boxes fuse like boxes under source weights."""
        rectangles = [(0, 0, 10, 10), (2, 0, 12, 10), (1, 1, 11, 11)]
        confidence = [0.9, 0.7, 0.4]
        options = {
            "sources": [0, 1, 2],
            "num_models": 3,
            "source_weights": [1.0, 4.0, 2.0],
        }

        boxes = _fusion_detections("box", rectangles, confidence).with_fusion(**options)
        oriented = _fusion_detections("oriented", rectangles, confidence).with_fusion(
            **options
        )

        np.testing.assert_allclose(oriented.xyxy, boxes.xyxy, atol=1e-9)
        np.testing.assert_allclose(oriented.confidence, boxes.confidence)

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    def test_zero_confidence_clusters_ignore_the_weights(self, kind: str) -> None:
        """Members that all have zero confidence stay equal, so the box is their
        mean."""
        detections = _fusion_detections(
            kind, [(0, 0, 10, 10), (2, 0, 12, 10)], [0.0, 0.0]
        )

        result = detections.with_fusion(
            sources=[0, 1], num_models=2, source_weights=[3.0, 1.0]
        )

        np.testing.assert_allclose(result.xyxy, [[1, 0, 11, 10]])
        np.testing.assert_allclose(result.confidence, [0.0])

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    @pytest.mark.parametrize(
        ("score", "weights"),
        [
            pytest.param(
                float(np.finfo(np.float64).max), [2.0, 1.0], id="finite-maximum"
            ),
            pytest.param(float(np.nextafter(0.0, 1.0)), [1.0, 0.75], id="subnormal"),
        ],
    )
    def test_equal_extreme_confidences_keep_their_value(
        self, kind: str, score: float, weights: list[float]
    ) -> None:
        """A weighted mean of equal confidences keeps their exact value."""
        detections = _fusion_detections(kind, [(0, 0, 10, 10)] * 2, [score] * 2)

        result = detections.with_fusion(
            sources=[0, 1], num_models=2, source_weights=weights
        )

        np.testing.assert_array_equal(result.confidence, [score])

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    def test_uniform_zero_confidence_clusters_vote_equally(self, kind: str) -> None:
        """A zero-score seed makes both geometry and mask votes uniform."""
        rectangles = [(0, 0, 10, 10), (2, 0, 12, 10)]
        detections = _fusion_detections(kind, rectangles, [0.0, 0.0])

        result = detections.with_fusion(
            sources=[0, 1],
            num_models=2,
            source_weights=[3.0, 1.0],
            box_weights="uniform",
        )

        np.testing.assert_allclose(result.xyxy, [[1, 0, 11, 10]])
        if_mask = _rect_masks((30, 30), [(0, 0, 12, 10)]) if kind == "mask" else None
        np.testing.assert_array_equal(result.mask, if_mask)

    @pytest.mark.parametrize("compact", [False, True])
    def test_exact_weight_tie_keeps_pixels_and_attracts_a_later_member(
        self, compact: bool
    ) -> None:
        """Tied fringe pixels participate in overlap with the next detection."""
        masks = np.array(
            [[[1, 1, 0]], [[1, 1, 0]], [[0, 1, 1]], [[1, 0, 0]]], dtype=bool
        )
        detections = _with_masks(masks, [1.0, 1.0, 1.0, 0.5])
        detections = detections.to_compact_masks() if compact else detections
        options = {
            "threshold": 0.3,
            "sources": [0, 1, 2, 3],
            "num_models": 4,
            "source_weights": [1, 2, 3, 1],
        }

        groups = detections.fusion_groups(**options)  # type: ignore[arg-type]
        fused = detections.with_fusion(**options)  # type: ignore[arg-type]

        assert groups == [[0, 1, 2, 3]]
        np.testing.assert_array_equal(np.asarray(fused.mask), [[[True, True, False]]])

    @pytest.mark.parametrize("compact", [False, True])
    def test_wide_vote_preserves_exact_ties(self, compact: bool) -> None:
        """Two exact products wider than int64 keep both masks' fringes."""
        masks = _rect_masks((12, 14), [(0, 0, 10, 10), (2, 0, 12, 10)])
        detections = _with_masks(masks, [0.9, 0.8])
        detections = detections.to_compact_masks() if compact else detections

        result = detections.with_fusion(
            sources=[0, 1], num_models=2, source_weights=[0.8, 0.9]
        )

        np.testing.assert_array_equal(np.asarray(result.mask), masks.any(axis=0)[None])

    def test_the_widest_allowed_ratio_stays_finite(self) -> None:
        """A weight 2**100 times smaller barely moves the result and stays finite."""
        result = _pair().with_fusion(
            sources=[0, 1], num_models=2, source_weights=[1.0, 2.0**-100]
        )

        assert np.isfinite(result.xyxy).all()
        np.testing.assert_allclose(result.xyxy, [[0, 0, 10, 10]], atol=1e-9)
        np.testing.assert_allclose(result.confidence, [0.9])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param(
                {"sources": None, "source_weights": [1.0, 2.0]},
                "requires sources",
                id="without-sources",
            ),
            pytest.param(
                {"source_weights": [1.0]}, "one weight per source", id="wrong-length"
            ),
            pytest.param(
                {"source_weights": [1.0, 0.0]}, "strictly positive", id="zero"
            ),
            pytest.param(
                {"source_weights": [1.0, -1.0]}, "strictly positive", id="negative"
            ),
            pytest.param({"source_weights": [1.0, np.nan]}, "finite", id="nan"),
            pytest.param({"source_weights": [1.0, np.inf]}, "finite", id="infinite"),
            pytest.param(
                {"source_weights": [1.0, 2.0**-101]}, "2\\*\\*100", id="ratio-too-large"
            ),
        ],
    )
    def test_invalid_source_weights_raise(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        """Weights must be one finite positive value per source, within the ratio."""
        options: dict[str, object] = {"sources": [0, 1], "num_models": 2}
        options.update(kwargs)

        with pytest.raises(ValueError, match=message):
            _pair().with_fusion(**options)  # type: ignore[arg-type]


class TestFusionGroups:
    """`fusion_groups` lists the members of each fused detection, seed first."""

    def test_groups_follow_the_fused_rows_with_the_seed_first(self) -> None:
        """Each group lists its seed, then the other members best first."""
        detections = _fusion_detections(
            "box", [(0, 0, 10, 10), (50, 50, 60, 60), (1, 1, 11, 11)], [0.7, 0.9, 0.8]
        )

        assert detections.fusion_groups() == [[1], [2, 0]]

    def test_equal_confidence_keeps_the_input_order(self) -> None:
        """On a confidence tie the earlier row is the seed."""
        detections = _fusion_detections("box", [(0, 0, 10, 10)] * 3, [0.5, 0.5, 0.5])

        assert detections.fusion_groups() == [[0, 1, 2]]

    def test_empty_detections_have_no_groups(self) -> None:
        """Nothing to fuse gives no groups."""
        assert Detections.empty().fusion_groups() == []

    @pytest.mark.parametrize("kind", _GEOMETRIES)
    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({}, id="default"),
            pytest.param({"box_weights": "uniform"}, id="uniform"),
            pytest.param({"sources": [0, 1, 0], "num_models": 2}, id="sources"),
            pytest.param(
                {"sources": [0, 1, 0], "num_models": 2, "source_weights": [3.0, 1.0]},
                id="source-weights",
            ),
        ],
    )
    def test_groups_describe_the_rows_that_with_fusion_returns(
        self, kind: str, options: dict[str, object]
    ) -> None:
        """Group `i` holds the members of fused row `i`, led by its winner."""
        detections = _fusion_detections(
            kind, _TWO_MODELS, [0.9, 0.8, 0.05], tracker_id=np.arange(3)
        )

        groups = detections.fusion_groups(**options)  # type: ignore[arg-type]
        fused = detections.with_fusion(**options)  # type: ignore[arg-type]

        assert len(groups) == len(fused)
        assert [group[0] for group in groups] == fused.tracker_id.tolist()
        assert sorted(row for group in groups for row in group) == [0, 1, 2]

    def test_groups_reject_what_with_fusion_rejects(self) -> None:
        """Both methods share one validation."""
        with pytest.raises(ValueError, match="requires sources"):
            _pair().fusion_groups(source_weights=[1.0, 2.0])


class TestDetectionsFuse:
    """`Detections.fuse` labels each input as a source and fuses them."""

    def test_matches_merging_with_explicit_sources(self) -> None:
        """Fusing the list equals `merge` plus `with_fusion` with the sources."""
        model_a = _fusion_detections("box", [(0, 0, 10, 10)], [0.9])
        model_b = _fusion_detections(
            "box", [(1, 1, 11, 11), (50, 50, 60, 60)], [0.8, 0.7]
        )
        expected = Detections.merge([model_a, model_b]).with_fusion(
            sources=[0, 1, 1], num_models=2
        )

        result = Detections.fuse([model_a, model_b])

        np.testing.assert_array_equal(result.xyxy, expected.xyxy)
        np.testing.assert_array_equal(result.confidence, expected.confidence)

    def test_an_empty_input_counts_as_a_model_that_found_nothing(self) -> None:
        """An empty input still lowers the confidence, unlike in `merge`."""
        model = _fusion_detections("box", [(0, 0, 10, 10)], [0.9])

        result = Detections.fuse([model, Detections.empty()])

        np.testing.assert_allclose(result.confidence, [0.45])

    def test_source_weights_are_given_per_input(self) -> None:
        """One weight per input list reaches the fusion."""
        model_a = _fusion_detections("box", [(0, 0, 10, 10)], [0.9])
        model_b = _fusion_detections("box", [(1, 1, 11, 11)], [0.8])

        result = Detections.fuse([model_a, model_b], source_weights=[3.0, 1.0])

        np.testing.assert_allclose(result.confidence, [0.875])

    def test_fields_missing_from_one_input_are_dropped_from_all(self) -> None:
        """Optional fields and `data` keys survive only if every input has them."""
        model_a = _fusion_detections(
            "box",
            [(0, 0, 10, 10)],
            [0.9],
            tracker_id=np.array([7]),
            data={"name": np.array(["a"]), "extra": np.array([1])},
        )
        model_b = _fusion_detections(
            "box", [(1, 1, 11, 11)], [0.8], data={"name": np.array(["b"])}
        )

        result = Detections.fuse([model_a, model_b])

        assert result.tracker_id is None
        assert list(result.data) == ["name"]
        assert result.data["name"].tolist() == ["a"]

    def test_masks_are_fused_when_every_input_has_them(self) -> None:
        """Mask detections fuse by mask overlap."""
        masks = _rect_masks((20, 20), [(0, 0, 10, 10), (2, 0, 12, 10)])
        model_a = _with_masks(masks[:1], [0.6])
        model_b = _with_masks(masks[1:], [0.6])

        result = Detections.fuse([model_a, model_b])

        np.testing.assert_array_equal(result.mask[0], masks.any(axis=0))

    def test_mixed_masks_preserve_foreground_outside_detector_boxes(self) -> None:
        """The representation does not affect pixels, membership, or confidence."""
        masks = np.array([[[False, False, True]]])
        dense = _with_masks(masks, [1.0], xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]))
        compact = dense.to_compact_masks()

        result = Detections.fuse([dense, compact])

        assert isinstance(result.mask, CompactMask)
        np.testing.assert_array_equal(result.mask.to_dense(), masks)
        np.testing.assert_array_equal(result.confidence, [1.0])

    @pytest.mark.parametrize("flat_first", [False, True])
    def test_fuse_aligns_shared_data_and_oriented_corner_layouts(
        self, flat_first: bool
    ) -> None:
        """Both oriented layouts can be mixed while preserving the first layout."""
        corners = _axis_corners(np.array([[0.0, 0.0, 10.0, 10.0]]))
        flat = corners.reshape(1, 8)
        first, second = (flat, corners) if flat_first else (corners, flat)
        a = _oriented(
            first, [0.9], data={ORIENTED_BOX_COORDINATES: first, "name": ["a"]}
        )
        b = _oriented(
            second,
            [0.8],
            data={ORIENTED_BOX_COORDINATES: second, "name": np.array(["b"])},
        )

        result = Detections.fuse([a, b])

        actual_corners = np.asarray(result.data[ORIENTED_BOX_COORDINATES])
        assert actual_corners.shape == first.shape
        np.testing.assert_allclose(actual_corners, first, atol=1e-12)
        np.testing.assert_array_equal(result.data["name"], ["a"])

    def test_fuse_retains_only_shared_equal_collection_metadata(self) -> None:
        """Unequal or one-sided metadata is dropped without modifying the inputs."""
        a = _pair(metadata={"image": "scene", "model": "a", "extra": 1})
        b = _pair(metadata={"image": "scene", "model": "b"})

        result = Detections.fuse([a, b])

        assert result.metadata == {"image": "scene"}
        assert a.metadata == {"image": "scene", "model": "a", "extra": 1}
        assert b.metadata == {"image": "scene", "model": "b"}

    def test_all_empty_inputs_give_empty_detections(self) -> None:
        """Nothing to fuse is empty, after the arguments were validated."""
        result = Detections.fuse([Detections.empty(), Detections.empty()])

        assert len(result) == 0

    def test_an_empty_list_raises(self) -> None:
        """At least one input is needed to count the models."""
        with pytest.raises(ValueError, match="at least one"):
            Detections.fuse([])
