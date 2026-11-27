from __future__ import annotations

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    OverlapOn,
    _box_weighted_box_fusion,
)


def _pair(**overrides: object) -> Detections:
    """Two overlapping boxes of class 0 that fuse into one detection."""
    fields: dict[str, object] = {
        "xyxy": np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float64),
        "confidence": np.array([0.9, 0.8]),
        "class_id": np.array([0, 0]),
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

    @pytest.mark.parametrize("overlap_on", [None, "box"])
    def test_oriented_boxes_raise(self, overlap_on: str | None) -> None:
        """Oriented-box detections cannot be fused yet."""
        corners = np.array(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[1, 1], [11, 1], [11, 11], [1, 11]],
            ],
            dtype=np.float64,
        )
        detections = _pair(data={ORIENTED_BOX_COORDINATES: corners})

        with pytest.raises(ValueError, match="oriented boxes"):
            detections.with_fusion(overlap_on=overlap_on)

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
