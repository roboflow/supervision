import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.f1_score import F1Score
from tests.helpers import assert_almost_equal


class TestF1Score:
    @pytest.fixture
    def predictions_multiple_classes(self):
        return Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0, matches target
                    [60, 60, 100, 100],  # class 1, matches target
                    [120, 120, 130, 130],  # class 1, false positive
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.8, 0.7]),
            class_id=np.array([0, 1, 1]),
        )

    @pytest.fixture
    def targets_multiple_classes(self):
        return Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0
                    [60, 60, 100, 100],  # class 1
                ],
                dtype=np.float32,
            ),
            class_id=np.array([0, 1]),
        )

    def test_initialization_default(self):
        """Test that F1Score can be initialized with default parameters"""
        metric = F1Score()
        assert metric._metric_target == MetricTarget.BOXES
        assert metric.averaging_method == AveragingMethod.WEIGHTED
        assert metric._predictions_list == []
        assert metric._targets_list == []

    def test_initialization_custom(self):
        """Test that F1Score can be initialized with custom parameters"""
        metric = F1Score(
            metric_target=MetricTarget.MASKS,
            averaging_method=AveragingMethod.MACRO,
        )
        assert metric._metric_target == MetricTarget.MASKS
        assert metric.averaging_method == AveragingMethod.MACRO

    def test_reset(self, dummy_prediction):
        """Test that reset() clears all stored data"""
        metric = F1Score()

        # Add some dummy data
        metric.update(dummy_prediction, dummy_prediction)

        # Verify data was added
        assert len(metric._predictions_list) == 1
        assert len(metric._targets_list) == 1

        # Reset and verify lists are empty
        metric.reset()
        assert metric._predictions_list == []
        assert metric._targets_list == []

    def test_perfect_match(self, detections_50_50, targets_50_50):
        """Test F1 score with perfect matching predictions and targets"""
        metric = F1Score()
        result = metric.update(detections_50_50, targets_50_50).compute()

        # Perfect match should give F1 = 1.0
        # TP = 1, FP = 0, FN = 0
        # Precision = TP / (TP + FP) = 1 / 1 = 1.0
        # Recall = TP / (TP + FN) = 1 / 1 = 1.0
        # F1 = 2 * (P * R) / (P + R) = 2 * 1.0 / 2 = 1.0
        assert result.f1_50 == 1.0
        assert result.f1_75 == 1.0
        assert len(result.matched_classes) == 1
        assert result.matched_classes[0] == 0

    def test_no_overlap(self, predictions_no_overlap, targets_no_overlap):
        """Test F1 score with predictions that don't overlap with targets"""
        metric = F1Score()
        result = metric.update(predictions_no_overlap, targets_no_overlap).compute()

        # No overlap means TP=0, FP=1, FN=1
        # Precision = 0 / 1 = 0.0
        # Recall = 0 / 1 = 0.0
        # F1 = 2 * (0 * 0) / (0 + 0) = 0 / 0 = 0.0
        assert result.f1_50 == 0.0
        assert result.f1_75 == 0.0

    def test_empty_predictions(self, targets_50_50):
        """Test F1 score with empty predictions but existing targets"""
        predictions = Detections.empty()

        metric = F1Score()
        result = metric.update(predictions, targets_50_50).compute()

        # No predictions: TP=0, FP=0, FN=1
        # Precision = 0 / 0 = 0 (by convention)
        # Recall = 0 / 1 = 0.0
        # F1 = 0.0
        assert result.f1_50 == 0.0
        assert result.f1_75 == 0.0

    def test_empty_targets(self, detections_50_50):
        """Test F1 score with predictions but no targets"""
        targets = Detections.empty()

        metric = F1Score()
        result = metric.update(detections_50_50, targets).compute()

        # No targets: TP=0, FP=1, FN=0
        # Precision = 0 / 1 = 0.0
        # Recall = 0 / 0 = 0 (by convention)
        # F1 = 0.0
        assert result.f1_50 == 0.0
        assert result.f1_75 == 0.0

    def test_false_positives_on_background_image_counted(self):
        """Predictions on an image with no targets must count as false positives."""
        predictions_with_gt = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        targets_with_gt = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
        )
        background_predictions = Detections(
            xyxy=np.array([[20, 0, 25, 5], [40, 0, 45, 5], [60, 0, 65, 5]], np.float32),
            class_id=np.array([0, 0, 0]),
            confidence=np.array([0.9, 0.9, 0.9]),
        )

        metric = F1Score(averaging_method=AveragingMethod.MICRO)
        result = metric.update(
            [predictions_with_gt, background_predictions],
            [targets_with_gt, Detections.empty()],
        ).compute()

        # TP=1, FP=3, FN=0 -> F1 = 2*1 / (2*1 + 3 + 0) = 0.4
        assert result.f1_50 == pytest.approx(0.4)
        assert 0 in result.matched_classes

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            pytest.param(
                AveragingMethod.MICRO, 0.5, id="micro-counts-absent-class-fps"
            ),
            pytest.param(
                AveragingMethod.MACRO, 0.5, id="macro-counts-absent-class-fps"
            ),
            pytest.param(
                AveragingMethod.WEIGHTED,
                1.0,
                id="weighted-absent-class-fps-not-counted-by-design",
            ),
        ],
    )
    def test_false_positives_of_absent_class_counted(self, method, expected):
        """Predictions of absent class count as FPs under MICRO/MACRO; WEIGHTED
        excludes them by design (GT support=0 → weight=0, consistent with sklearn)."""
        predictions = Detections(
            xyxy=np.array(
                [[0, 0, 10, 10], [100, 0, 110, 10], [120, 0, 130, 10]], np.float32
            ),
            class_id=np.array([0, 1, 1]),  # class 1 never appears in the targets
            confidence=np.array([0.9, 0.8, 0.7]),
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
        )

        metric = F1Score(averaging_method=method)
        result = metric.update(predictions, targets).compute()

        # MICRO: TP=1, FP=2, FN=0 -> F1 = 2/(2+2) = 0.5
        # MACRO: mean([F1_class0=1.0, F1_class1=0.0]) = 0.5
        # WEIGHTED: class_1 weight=0 -> only class 0 contributes -> 1.0
        assert result.f1_50 == pytest.approx(expected)

    def test_false_positives_on_background_image_weighted_returns_zero(self):
        """WEIGHTED F1 is 0.0 when all images are background (no GT anywhere)."""
        background_predictions = Detections(
            xyxy=np.array([[20, 0, 25, 5], [40, 0, 45, 5]], np.float32),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
        )

        metric = F1Score(averaging_method=AveragingMethod.WEIGHTED)
        result = metric.update(background_predictions, Detections.empty()).compute()

        # No GT support anywhere -> class_counts.sum() == 0 -> returns 0.0
        assert result.f1_50 == 0.0

    def test_single_class_mixed_results(
        self, predictions_confidence_ranking, targets_50_50
    ):
        """Test F1 score calculation with mixed precision and recall"""
        metric = F1Score()
        result = metric.update(predictions_confidence_ranking, targets_50_50).compute()

        # TP=1, FP=1, FN=0
        # Precision = TP / (TP + FP) = 1 / 2 = 0.5
        # Recall = TP / (TP + FN) = 1 / 1 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3 ≈ 0.6667
        expected_f1 = 2.0 / 3.0
        assert_almost_equal(result.f1_50, expected_f1)
        assert_almost_equal(result.f1_75, expected_f1)

    def test_precision_recall_imbalance(
        self, detections_50_50, targets_two_objects_class_0
    ):
        """Test F1 score with different precision and recall scenarios"""
        metric = F1Score()
        result = metric.update(detections_50_50, targets_two_objects_class_0).compute()

        # TP=1, FP=0, FN=1
        # Precision = TP / (TP + FP) = 1 / 1 = 1.0
        # Recall = TP / (TP + FN) = 1 / 2 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 2/3 ≈ 0.6667
        expected_f1 = 2.0 / 3.0
        assert_almost_equal(result.f1_50, expected_f1)
        assert_almost_equal(result.f1_75, expected_f1)

    def test_multiple_classes(
        self, predictions_multiple_classes, targets_multiple_classes
    ):
        """Test F1 score calculation for multiple classes"""
        metric = F1Score()
        result = metric.update(
            predictions_multiple_classes, targets_multiple_classes
        ).compute()

        # Class 0: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0 (weight=1)
        # Class 1: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3 (weight=1)
        # Weighted avg: (1*1.0 + 1*2/3) / (1+1) = (1 + 2/3) / 2 = 5/6 ≈ 0.8333
        expected_f1 = (1.0 + 2.0 / 3.0) / 2.0
        assert_almost_equal(result.f1_50, expected_f1)
        assert len(result.matched_classes) == 2
        assert 0 in result.matched_classes
        assert 1 in result.matched_classes

    def test_different_iou_thresholds(self, predictions_iou_064, targets_iou_064):
        """Test F1 score at different IoU thresholds"""
        metric = F1Score()
        result = metric.update(predictions_iou_064, targets_iou_064).compute()

        # IoU = 0.64 > 0.5 but < 0.75
        # At IoU 0.5: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        # At IoU 0.75: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0
        assert result.f1_50 == 1.0
        assert result.f1_75 == 0.0

    def test_confidence_ranking(self, predictions_confidence_ranking, targets_50_50):
        """Test that F1 score respects confidence ranking"""
        metric = F1Score()
        result = metric.update(predictions_confidence_ranking, targets_50_50).compute()

        # Higher confidence prediction should match the target
        # TP=1, FP=1, FN=0
        # Precision = 1/2 = 0.5, Recall = 1/1 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
        expected_f1 = 2.0 / 3.0
        assert_almost_equal(result.f1_50, expected_f1)

    def test_list_inputs(
        self, detections_50_50, targets_50_50, prediction_class_1, target_class_1
    ):
        """Test F1 score with list inputs"""
        metric = F1Score()
        result = metric.update(
            [detections_50_50, prediction_class_1], [targets_50_50, target_class_1]
        ).compute()

        # Perfect matches for both
        assert result.f1_50 == 1.0
        assert result.f1_75 == 1.0

    def test_mismatched_list_lengths(self, detections_50_50, targets_50_50):
        """Test that mismatched prediction/target list lengths raise error"""
        metric = F1Score()

        # Should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="number of predictions"):
            metric.update([detections_50_50], [targets_50_50, targets_50_50])

    @pytest.mark.parametrize(
        "missing_attribute",
        ["predictions_class_id", "targets_class_id", "predictions_confidence"],
    )
    def test_compute_value_error_for_missing_required_fields(
        self, missing_attribute
    ) -> None:
        """Test compute raises ValueError when required fields are missing."""
        metric = F1Score()
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        class_id = np.array([0], dtype=np.int32)
        confidence = np.array([0.9], dtype=np.float32)

        predictions = Detections(
            xyxy=boxes,
            confidence=confidence,
            class_id=class_id,
        )
        targets = Detections(
            xyxy=boxes,
            class_id=class_id,
        )

        if missing_attribute == "predictions_class_id":
            predictions = Detections(
                xyxy=boxes,
                confidence=confidence,
            )
        elif missing_attribute == "targets_class_id":
            targets = Detections(xyxy=boxes)
        else:
            predictions = Detections(
                xyxy=boxes,
                class_id=class_id,
            )

        with pytest.raises(ValueError, match="F1Score metric requires"):
            metric.update(predictions, targets).compute()

    @pytest.mark.parametrize(
        "averaging_method",
        [AveragingMethod.MACRO, AveragingMethod.MICRO, AveragingMethod.WEIGHTED],
    )
    def test_averaging_methods(self, averaging_method, detections_50_50, targets_50_50):
        """Test different averaging methods"""
        metric = F1Score(averaging_method=averaging_method)
        result = metric.update(detections_50_50, targets_50_50).compute()

        # Perfect match should give 1.0 regardless of averaging method
        assert result.f1_50 == 1.0
        assert result.averaging_method == averaging_method

    def test_macro_averaging(
        self, predictions_multiple_classes, targets_multiple_classes
    ):
        """Test MACRO averaging with specific example"""
        metric = F1Score(averaging_method=AveragingMethod.MACRO)
        result = metric.update(
            predictions_multiple_classes, targets_multiple_classes
        ).compute()

        # Macro average: (1.0 + 2/3) / 2 = 5/6
        expected_f1 = (1.0 + 2.0 / 3.0) / 2.0
        assert_almost_equal(result.f1_50, expected_f1)

    def test_micro_averaging(
        self, predictions_multiple_classes, targets_multiple_classes
    ):
        """Test MICRO averaging with specific example"""
        metric = F1Score(averaging_method=AveragingMethod.MICRO)
        result = metric.update(
            predictions_multiple_classes, targets_multiple_classes
        ).compute()

        # Micro F1: 4/5 = 0.8
        expected_f1 = 0.8
        assert_almost_equal(result.f1_50, expected_f1)

    def test_weighted_averaging(
        self, predictions_multiple_classes, targets_multiple_classes
    ):
        """Test WEIGHTED averaging with specific example"""
        metric = F1Score(averaging_method=AveragingMethod.WEIGHTED)
        result = metric.update(
            predictions_multiple_classes, targets_multiple_classes
        ).compute()

        # Weighted average: 5/6
        expected_f1 = 5.0 / 6.0
        assert_almost_equal(result.f1_50, expected_f1)

    def test_greedy_matching_two_valid_pairs(self):
        """Greedy matching finds both TPs; np.unique style missed the second pair.

        IoU matrix: [[1.0, 0.667], [0.333, 0.538]]. At iou>=0.5 the optimal
        assignment is T0<->P0 and T1<->P1 (2 TPs, F1=1.0).
        """
        preds = Detections(
            xyxy=np.array([[40, 60, 380, 470], [108, 60, 448, 470]], dtype=np.float32),
            confidence=np.array([0.95, 0.90]),
            class_id=np.array([0, 0]),
        )
        targets = Detections(
            xyxy=np.array([[40, 60, 380, 470], [210, 60, 550, 470]], dtype=np.float32),
            class_id=np.array([0, 0]),
        )

        result = F1Score().update(preds, targets).compute()

        assert result.f1_50 == 1.0
