import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget
from supervision.metrics.mean_average_precision import (
    EvaluationDataset,
    MeanAveragePrecision,
)


def _mask_detections(
    row_slice: slice, confidence: bool = False, mask_shape: tuple[int, int] = (32, 32)
) -> Detections:
    """Build single-detection `Detections` with a mask filling the given rows."""
    mask = np.zeros((1, *mask_shape), dtype=bool)
    mask[0, row_slice, :] = True
    return Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]) if confidence else None,
        mask=mask,
    )


def _obb_detections(corners: list[list[int]], confidence: bool = False) -> Detections:
    """Build single-detection `Detections` with the given oriented box corners."""
    return Detections(
        xyxy=np.array([[0, 0, 30, 30]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]) if confidence else None,
        data={ORIENTED_BOX_COORDINATES: np.array([corners], dtype=np.float32)},
    )


class TestMeanAveragePrecision:
    def test_single_perfect_detection(self, detections_50_50, targets_50_50):
        """Test that single perfect detection gets 1.0 mAP (not 0.0 due to ID=0 bug)"""
        metric = MeanAveragePrecision()
        metric.update([detections_50_50], [targets_50_50])
        result = metric.compute()

        # Should be perfect 1.0 mAP, not 0.0 due to ID=0 bug
        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_multiple_perfect_detections(self):
        """Test that multiple perfect detections get 1.0 mAP"""
        # Multiple perfect detections in one image
        detections = Detections(
            xyxy=np.array(
                [[10, 10, 50, 50], [100, 100, 140, 140], [200, 200, 240, 240]],
                dtype=np.float64,
            ),
            class_id=np.array([0, 0, 0]),
            confidence=np.array([0.9, 0.9, 0.9]),
        )

        metric = MeanAveragePrecision()
        metric.update([detections], [detections])
        result = metric.compute()

        # Should be perfect 1.0 mAP
        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_perfect_non_square_oriented_boxes_get_full_map(self):
        """Perfect non-square OBB predictions score full mAP via OBB IoU."""
        obb = np.array(
            [[[10, 0], [0, 1], [30, 4], [40, 3]]],
            dtype=np.float32,
        )
        detections = Detections(
            xyxy=np.array([[0, 0, 40, 4]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            data={ORIENTED_BOX_COORDINATES: obb},
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 40, 4]], dtype=np.float64),
            class_id=np.array([0]),
            data={ORIENTED_BOX_COORDINATES: obb},
        )

        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )
        metric.update([detections], [targets])
        result = metric.compute()

        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_batch_updates_perfect_detections(self, detections_50_50, targets_50_50):
        """Test that batch updates with perfect detections get 1.0 mAP"""
        metric = MeanAveragePrecision()
        # Add 3 batch updates
        metric.update([detections_50_50], [targets_50_50])
        metric.update([detections_50_50], [targets_50_50])
        metric.update([detections_50_50], [targets_50_50])
        result = metric.compute()

        # Should be perfect 1.0 mAP across all batches
        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_scenario_1_success_case_imperfect_match(self):
        """Scenario 1: Success Case with imperfect match"""
        # Small object (class 0) - area = 30*30 = 900 < 1024
        small_perfect = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object (class 1) - area = 50*50 = 2500 (between 1024 and 9216)
        medium_target = Detections(
            xyxy=np.array([[10, 10, 60, 60]], dtype=np.float64),
            class_id=np.array([1]),
            data={"area": np.array([2500])},
        )
        medium_pred = Detections(
            xyxy=np.array([[12, 12, 60, 60]], dtype=np.float64),  # Slightly off
            class_id=np.array([1]),
            confidence=np.array([0.9]),
            data={"area": np.array([2304])},  # 48*48
        )

        # Large objects (classes 0, 1, 2) - area = 100*100 = 10000 > 9216
        large_targets = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            data={"area": np.array([10000, 10000, 10000])},
        )
        large_preds = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            confidence=np.array([0.9, 0.9, 0.9]),
            data={"area": np.array([10000, 10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_perfect], [small_perfect])
        metric.update([medium_pred], [medium_target])
        metric.update([large_preds], [large_targets])
        result = metric.compute()

        # Should be close to 0.9 (slightly less than perfect due to medium object)
        assert 0.85 < result.map50_95 < 0.98  # Adjusted upper bound
        assert (
            result.medium_objects.map50_95 < 1.0
        )  # Medium should be less than perfect

    def test_scenario_2_missed_detection(self):
        """Scenario 2: GT Present, No Prediction (Missed Detection)"""
        # Small object - area = 30*30 = 900 < 1024
        small_detection = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object - area = 50*50 = 2500 (between 1024 and 9216) - missed
        medium_target = Detections(
            xyxy=np.array([[10, 10, 60, 60]], dtype=np.float64),
            class_id=np.array([1]),
            data={"area": np.array([2500])},
        )
        no_medium_pred = Detections.empty()

        # Large objects - area = 100*100 = 10000 > 9216
        large_detections = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            confidence=np.array([0.9, 0.9, 0.9]),
            data={"area": np.array([10000, 10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_detection], [small_detection])
        metric.update([no_medium_pred], [medium_target])
        metric.update([large_detections], [large_detections])
        result = metric.compute()

        # Medium objects should have 0.0 mAP (missed detection)
        assert abs(result.medium_objects.map50_95 - 0.0) < 1e-6

    def test_scenario_3_false_positive(self):
        """Scenario 3: No GT, Prediction Present (False Positive)"""
        # Small object - area = 30*30 = 900 < 1024
        small_detection = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object - area = 50*50 = 2500 - false positive (no GT)
        medium_pred = Detections(
            xyxy=np.array([[12, 12, 62, 62]], dtype=np.float64),
            class_id=np.array([1]),
            confidence=np.array([0.9]),
            data={"area": np.array([2500])},
        )
        no_medium_target = Detections.empty()

        # Large objects - area = 100*100 = 10000 > 9216
        large_detections = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            confidence=np.array([0.9, 0.9, 0.9]),
            data={"area": np.array([10000, 10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_detection], [small_detection])
        metric.update([medium_pred], [no_medium_target])
        metric.update([large_detections], [large_detections])
        result = metric.compute()

        # Medium objects should have -1 mAP (false positive, matching pycocotools)
        assert result.medium_objects.map50_95 == -1

    def test_scenario_4_no_data(self):
        """Scenario 4: No GT, No Prediction (Category has no data)"""
        # Small object - area = 30*30 = 900 < 1024
        small_detection = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object - no data at all
        no_medium = Detections.empty()

        # Large objects - area = 100*100 = 10000 > 9216
        # only classes 0 and 2 (no class 1)
        large_targets = Detections(
            xyxy=np.array(
                [
                    [10, 10, 110, 110],
                    [120, 120, 220, 220],
                ],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0]),
            data={"area": np.array([10000, 10000])},
        )
        large_preds = Detections(
            xyxy=np.array(
                [
                    [10, 10, 110, 110],
                    [120, 120, 220, 220],
                ],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0]),
            confidence=np.array([0.9, 0.9]),
            data={"area": np.array([10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_detection], [small_detection])
        metric.update([no_medium], [no_medium])
        metric.update([large_preds], [large_targets])
        result = metric.compute()

        # Should NOT have negative mAP values for overall
        assert result.map50_95 >= 0.0
        # Medium objects should have -1 mAP (no data, matching pycocotools)
        assert result.medium_objects.map50_95 == -1

    def test_scenario_5_only_one_class_present(self):
        """Scenario 5: Only 1 of 3 Classes Present (Perfect Match)"""
        # Only class 0 objects with perfect matches
        detections_class_0 = [
            Detections(
                xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
                class_id=np.array([0]),
                confidence=np.array([0.95]),
            ),
            Detections(
                xyxy=np.array([[20, 20, 230, 130]], dtype=np.float64),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
            ),
        ]

        metric = MeanAveragePrecision()
        for det in detections_class_0:
            metric.update([det], [det])

        result = metric.compute()

        # Should be 1.0 mAP (perfect match for the only class present)
        assert abs(result.map50_95 - 1.0) < 1e-6
        assert abs(result.map50 - 1.0) < 1e-6
        assert abs(result.map75 - 1.0) < 1e-6

    def test_mixed_classes_with_missing_detections(
        self, detections_50_50, targets_50_50
    ):
        """Test mixed scenario with some classes having no detections"""
        # Class 1: GT exists but no prediction
        class_1_target = Detections(
            xyxy=np.array([[60, 60, 100, 100]], dtype=np.float64),
            class_id=np.array([1]),
        )
        class_1_pred = Detections.empty()

        # Class 2: Prediction exists but no GT (false positive)
        class_2_pred = Detections(
            xyxy=np.array([[110, 110, 150, 150]], dtype=np.float64),
            class_id=np.array([2]),
            confidence=np.array([0.8]),
        )
        class_2_target = Detections.empty()

        metric = MeanAveragePrecision()
        metric.update([detections_50_50], [targets_50_50])
        metric.update([class_1_pred], [class_1_target])
        metric.update([class_2_pred], [class_2_target])
        result = metric.compute()

        # Should not have negative mAP
        assert result.map50_95 >= 0.0
        # Should be less than 1.0 due to missed detection and false positive
        assert result.map50_95 < 1.0

    def test_empty_predictions_and_targets(self):
        """Test completely empty predictions and targets"""
        metric = MeanAveragePrecision()
        metric.update([Detections.empty()], [Detections.empty()])
        result = metric.compute()

        # Should return -1 for no data (matching pycocotools behavior)
        assert result.map50_95 == -1
        assert result.map50 == -1
        assert result.map75 == -1

        # All object size categories should also be -1
        assert result.small_objects.map50_95 == -1
        assert result.medium_objects.map50_95 == -1
        assert result.large_objects.map50_95 == -1


class TestMeanAveragePrecisionMasks:
    @pytest.mark.parametrize(
        ("prediction_rows", "target_rows", "expected_map50"),
        [
            pytest.param(slice(0, 16), slice(0, 16), 1.0, id="matching-masks"),
            pytest.param(slice(0, 16), slice(16, 32), 0.0, id="disjoint-masks"),
        ],
    )
    def test_map50_follows_mask_overlap(
        self, prediction_rows: slice, target_rows: slice, expected_map50: float
    ) -> None:
        """With MASKS target, map50 must reflect mask IoU, not identical boxes."""
        predictions = _mask_detections(prediction_rows, confidence=True)
        targets = _mask_detections(target_rows)
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(expected_map50, abs=1e-6)

    def test_missing_masks_raise(self) -> None:
        """With MASKS target, detections without masks must raise ValueError."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        targets = _mask_detections(slice(0, 16))
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)
        metric.update([predictions], [targets])

        with pytest.raises(ValueError, match="MASKS"):
            metric.compute()

    def test_mask_pixel_count_drives_size_buckets(self) -> None:
        """With MASKS target, object size buckets use mask area, not bbox area."""
        # bbox area is 100*100 = 10000 (large), mask area is 30*30 = 900 (small)
        mask = np.zeros((1, 120, 120), dtype=bool)
        mask[0, 10:40, 10:40] = True
        predictions = Detections(
            xyxy=np.array([[0, 0, 100, 100]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=mask,
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 100, 100]], dtype=np.float64),
            class_id=np.array([0]),
            mask=mask.copy(),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.small_objects.map50 == pytest.approx(1.0, abs=1e-6)
        assert result.large_objects.map50 == -1

    def test_boxes_target_ignores_masks(self) -> None:
        """With default BOXES target, disjoint masks must not affect the score."""
        predictions = _mask_detections(slice(0, 16), confidence=True)
        targets = _mask_detections(slice(16, 32))
        metric = MeanAveragePrecision()

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionOrientedBoundingBoxes:
    @pytest.mark.parametrize(
        ("prediction_corners", "target_corners", "expected_map50"),
        [
            pytest.param(
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                1.0,
                id="matching-obb",
            ),
            pytest.param(
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 20], [30, 20], [30, 30], [20, 30]],
                0.0,
                id="disjoint-obb",
            ),
        ],
    )
    def test_map50_follows_oriented_box_overlap(
        self,
        prediction_corners: list[list[int]],
        target_corners: list[list[int]],
        expected_map50: float,
    ) -> None:
        """With OBB target, map50 must reflect OBB IoU, not identical boxes."""
        predictions = _obb_detections(prediction_corners, confidence=True)
        targets = _obb_detections(target_corners)
        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(expected_map50, abs=1e-6)

    def test_missing_oriented_boxes_raise(self) -> None:
        """With OBB target, detections without OBB data must raise ValueError."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 30, 30]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        targets = _obb_detections([[0, 0], [10, 0], [10, 10], [0, 10]])
        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )
        metric.update([predictions], [targets])

        with pytest.raises(ValueError, match=ORIENTED_BOX_COORDINATES):
            metric.compute()

    def test_cross_matched_obb_orients_iou_correctly(self) -> None:
        """2x2 cross-match: pred0->target1, pred1->target0 must both score as TP.

        A transposed (gt, dt) matrix would yield 0 IoU for every pair; map50=0.
        Passing asserts the (dt, gt) orientation is correct end-to-end.
        """
        box_tl = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        box_br = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32)
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64),
            class_id=np.array([0, 0]),
            data={ORIENTED_BOX_COORDINATES: np.stack([box_tl, box_br])},
        )
        # Predictions deliberately swapped: pred0 matches target1, pred1 matches target0
        predictions = Detections(
            xyxy=np.array([[20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            data={ORIENTED_BOX_COORDINATES: np.stack([box_br, box_tl])},
        )
        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionMasksCrowdBranch:
    """Tests for the crowd-aware Jaccard path in _mask_iou_with_jaccard."""

    def test_crowd_gt_ignores_contained_detection(self) -> None:
        """Detection inside a crowd GT is ignored (not FP) with Jaccard crowd IoU.

        Without Jaccard: small pred's standard IoU with crowd GT is 0.25 < 0.5,
        so pred0 is a FP, which reduces map50. With Jaccard: IoU = 1.0, pred0 is
        matched to crowd and ignored, so only pred1 (perfect TP) is scored -> map50=1.0.
        """
        mask_normal = np.zeros((1, 32, 32), dtype=bool)
        mask_normal[0, :16, :] = True  # normal GT: top half
        mask_crowd = np.ones((1, 32, 32), dtype=bool)  # crowd GT: full image

        targets = Detections(
            xyxy=np.array([[0, 0, 32, 16], [0, 0, 32, 32]], dtype=np.float64),
            class_id=np.array([0, 0]),
            mask=np.concatenate([mask_normal, mask_crowd]),
            data={"iscrowd": np.array([0, 1], dtype=np.int64)},
        )
        # pred0 (conf=0.9): bottom quarter - inside crowd, no overlap with normal GT
        mask_pred0 = np.zeros((1, 32, 32), dtype=bool)
        mask_pred0[0, 16:24, :] = True
        # pred1 (conf=0.8): exact match with normal GT
        mask_pred1 = np.zeros((1, 32, 32), dtype=bool)
        mask_pred1[0, :16, :] = True

        predictions = Detections(
            xyxy=np.array([[0, 16, 32, 24], [0, 0, 32, 16]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            mask=np.concatenate([mask_pred0, mask_pred1]),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionIgnoreFlag:
    """Tests for explicit target ignore flags in COCO-style evaluation."""

    def test_user_ignore_flag_excludes_target_from_scoring(self) -> None:
        """Targets marked ignored by the user must not count as normal GT."""
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0]),
            data={"ignore": np.array([1], dtype=np.int64)},
        )
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        metric = MeanAveragePrecision()

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(-1.0, abs=1e-6)

    def test_normal_gt_matched_correctly_alongside_crowd_gt(self) -> None:
        """Normal GT is matched and scored when a crowd GT is also present."""
        mask_normal = np.zeros((1, 32, 32), dtype=bool)
        mask_normal[0, :16, :] = True
        mask_crowd = np.ones((1, 32, 32), dtype=bool)

        targets = Detections(
            xyxy=np.array([[0, 0, 32, 16], [0, 0, 32, 32]], dtype=np.float64),
            class_id=np.array([0, 0]),
            mask=np.concatenate([mask_normal, mask_crowd]),
            data={"iscrowd": np.array([0, 1], dtype=np.int64)},
        )
        mask_pred = np.zeros((1, 32, 32), dtype=bool)
        mask_pred[0, :16, :] = True  # exact match with normal GT

        predictions = Detections(
            xyxy=np.array([[0, 0, 32, 16]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=mask_pred,
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionMasksOrientation:
    """Tests that the (dt, gt) IoU-matrix orientation is correct end-to-end."""

    def test_cross_matched_masks_orient_iou_correctly(self) -> None:
        """2x2 cross-match: pred0->target1, pred1->target0 must both score as TP.

        A transposed (gt, dt) matrix would yield 0 IoU for every pair; map50=0.
        Passing asserts the (dt, gt) orientation is correct end-to-end.
        """
        top_mask = np.zeros((1, 32, 32), dtype=bool)
        top_mask[0, :16, :] = True
        bottom_mask = np.zeros((1, 32, 32), dtype=bool)
        bottom_mask[0, 16:, :] = True

        # target0 = top half, target1 = bottom half
        targets = Detections(
            xyxy=np.array([[0, 0, 32, 16], [0, 16, 32, 32]], dtype=np.float64),
            class_id=np.array([0, 0]),
            mask=np.concatenate([top_mask, bottom_mask]),
        )
        # Predictions deliberately swapped: pred0=bottom, pred1=top
        predictions = Detections(
            xyxy=np.array([[0, 16, 32, 32], [0, 0, 32, 16]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            mask=np.concatenate([bottom_mask, top_mask]),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestEvaluationDatasetLoadPredictions:
    """Tests for `EvaluationDataset.load_predictions` input validation."""

    def test_unknown_image_id_raises_value_error(self) -> None:
        """Predictions referencing an unknown image id raise ValueError."""
        dataset = EvaluationDataset(
            targets={
                "images": [{"id": 1}],
                "annotations": [],
                "categories": [{"id": 1}],
            }
        )
        predictions = [{"image_id": 999, "category_id": 1, "bbox": [0, 0, 1, 1]}]

        with pytest.raises(ValueError, match="current coco set"):
            dataset.load_predictions(predictions)
