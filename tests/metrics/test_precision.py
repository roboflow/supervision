import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.precision import Precision


class TestPrecision:
    @pytest.fixture
    def predictions_multiple_classes(self):
        return Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0, matches target
                    [60, 60, 100, 100],  # class 0, matches target
                    [200, 200, 240, 240],  # class 1, matches target
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.8, 0.7]),
            class_id=np.array([0, 0, 1]),
        )

    @pytest.fixture
    def targets_multiple_classes(self):
        return Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0
                    [60, 60, 100, 100],  # class 0
                    [200, 200, 240, 240],  # class 1
                ],
                dtype=np.float32,
            ),
            class_id=np.array([0, 0, 1]),
        )

    def test_initialization_default(self):
        """Test that Precision can be initialized with default parameters"""
        metric = Precision()
        assert metric._metric_target == MetricTarget.BOXES
        assert metric.averaging_method == AveragingMethod.WEIGHTED
        assert metric._predictions_list == []
        assert metric._targets_list == []

    def test_initialization_custom(self):
        """Test that Precision can be initialized with custom parameters"""
        metric = Precision(
            metric_target=MetricTarget.MASKS,
            averaging_method=AveragingMethod.MACRO,
        )
        assert metric._metric_target == MetricTarget.MASKS
        assert metric.averaging_method == AveragingMethod.MACRO

    def test_reset(self, dummy_prediction):
        """Test that reset() clears all stored data"""
        metric = Precision()

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
        """Test precision with perfect matching predictions and targets"""
        metric = Precision()
        result = metric.update(detections_50_50, targets_50_50).compute()

        # Perfect match should give precision = 1.0
        # TP = 1, FP = 0 -> precision = TP / (TP + FP) = 1 / 1 = 1.0
        # TP = 1, FP = 0 -> precision = TP / (TP + FP) = 1 / 1 = 1.0
        assert result.precision_at_50 == 1.0
        assert result.precision_at_75 == 1.0
        assert len(result.matched_classes) == 1
        assert result.matched_classes[0] == 0

    def test_no_overlap(self, predictions_no_overlap, targets_no_overlap):
        """Test precision with predictions that don't overlap with targets"""
        metric = Precision()
        result = metric.update(predictions_no_overlap, targets_no_overlap).compute()

        # No overlap means no TP, only FP
        # TP = 0, FP = 1 -> precision = TP / (TP + FP) = 0 / 1 = 0.0
        assert result.precision_at_50 == 0.0
        assert result.precision_at_75 == 0.0

    def test_empty_predictions(self, targets_50_50):
        """Test precision with empty predictions but existing targets"""
        predictions = Detections.empty()

        metric = Precision()
        result = metric.update(predictions, targets_50_50).compute()

        # No predictions means TP = 0, FP = 0 -> precision = 0 / 0 = 0
        assert result.precision_at_50 == 0.0
        assert result.precision_at_75 == 0.0

    def test_empty_targets(self, detections_50_50):
        """Test precision with predictions but no targets"""
        targets = Detections.empty()

        metric = Precision()
        result = metric.update(detections_50_50, targets).compute()

        # All predictions are false positives
        # TP = 0, FP = 1 -> precision = 0 / 1 = 0.0
        assert result.precision_at_50 == 0.0
        assert result.precision_at_75 == 0.0

    def test_single_class(self, predictions_confidence_ranking, targets_50_50):
        """Test precision calculation for single class with mixed results"""
        metric = Precision()
        result = metric.update(predictions_confidence_ranking, targets_50_50).compute()

        # TP = 1 (first prediction), FP = 1 (second prediction)
        # precision = TP / (TP + FP) = 1 / 2 = 0.5
        assert result.precision_at_50 == 0.5
        assert result.precision_at_75 == 0.5

    def test_multiple_classes(
        self, predictions_multiple_classes, targets_multiple_classes
    ):
        """Test precision calculation for multiple classes"""
        metric = Precision()
        result = metric.update(
            predictions_multiple_classes, targets_multiple_classes
        ).compute()

        # All predictions match targets perfectly
        # Class 0: TP=2, FP=0 -> precision=1.0 (weight=2)
        # Class 1: TP=1, FP=0 -> precision=1.0 (weight=1)
        # Weighted avg: (2*1.0 + 1*1.0) / (2+1) = 3/3 = 1.0
        assert result.precision_at_50 == 1.0
        assert result.precision_at_75 == 1.0
        assert len(result.matched_classes) == 2
        assert 0 in result.matched_classes
        assert 1 in result.matched_classes

    def test_different_iou_thresholds(self, predictions_iou_064, targets_iou_064):
        """Test precision at different IoU thresholds"""
        metric = Precision()
        result = metric.update(predictions_iou_064, targets_iou_064).compute()

        # IoU = 0.64 > 0.5 but < 0.75
        # Should match at IoU 0.5 but not at 0.75
        assert result.precision_at_50 == 1.0  # TP=1, FP=0
        assert result.precision_at_75 == 0.0  # TP=0, FP=1

    def test_confidence_ranking(self, predictions_confidence_ranking, targets_50_50):
        """Test that predictions are ranked by confidence"""
        metric = Precision()
        result = metric.update(predictions_confidence_ranking, targets_50_50).compute()

        # Higher confidence prediction should match first
        # TP = 1, FP = 1 -> precision = 0.5
        assert result.precision_at_50 == 0.5

    def test_list_inputs(
        self, detections_50_50, targets_50_50, prediction_class_1, target_class_1
    ):
        """Test precision with list inputs"""
        metric = Precision()
        result = metric.update(
            [detections_50_50, prediction_class_1], [targets_50_50, target_class_1]
        ).compute()

        # Perfect matches for both
        assert result.precision_at_50 == 1.0
        assert result.precision_at_75 == 1.0

    def test_mismatched_list_lengths(self, detections_50_50, targets_50_50):
        """Test that mismatched prediction/target list lengths raise error"""
        metric = Precision()

        # Should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="number of predictions"):
            metric.update([detections_50_50], [targets_50_50, targets_50_50])

    @pytest.mark.parametrize(
        "averaging_method",
        [AveragingMethod.MACRO, AveragingMethod.MICRO, AveragingMethod.WEIGHTED],
    )
    def test_averaging_methods(self, averaging_method, detections_50_50, targets_50_50):
        """Test different averaging methods"""
        metric = Precision(averaging_method=averaging_method)
        result = metric.update(detections_50_50, targets_50_50).compute()

        # Perfect match should give 1.0 regardless of averaging method
        assert result.precision_at_50 == 1.0
        assert result.averaging_method == averaging_method
