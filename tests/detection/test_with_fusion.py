from __future__ import annotations

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.iou import OverlapMetric, OverlapOn
from supervision.detection.utils.nms import _box_weighted_box_fusion


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
    """`with_fusion` rejects the geometries it cannot fuse yet."""

    def test_mask_detections_raise_unless_fused_by_box(self) -> None:
        """Detections with masks need `overlap_on='box'` for now."""
        detections = _masked_pair()

        with pytest.raises(ValueError, match="overlap_on='box'"):
            detections.with_fusion()

    def test_explicit_mask_overlap_raises(self) -> None:
        """Requesting mask overlap explicitly is not supported yet either."""
        detections = _masked_pair()

        with pytest.raises(ValueError, match="overlap_on='box'"):
            detections.with_fusion(overlap_on=OverlapOn.MASK)

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
