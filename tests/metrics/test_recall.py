import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.recall import Recall
from tests.helpers import assert_almost_equal


class TestRecall:
    @pytest.fixture
    def predictions_multiple_classes(self):
        return Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0, matches first target
                    [200, 200, 240, 240],  # class 1, matches target
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 1]),
        )

    @pytest.fixture
    def targets_multiple_classes(self):
        return Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0, matched
                    [60, 60, 100, 100],  # class 0, missed
                    [200, 200, 240, 240],  # class 1, matched
                ],
                dtype=np.float32,
            ),
            class_id=np.array([0, 0, 1]),
        )

    def test_initialization_default(self):
        """Test that Recall can be initialized with default parameters"""
        metric = Recall()
        assert metric._metric_target == MetricTarget.BOXES
        assert metric.averaging_method == AveragingMethod.WEIGHTED
        assert metric._predictions_list == []
        assert metric._targets_list == []

    def test_initialization_custom(self):
        """Test that Recall can be initialized with custom parameters"""
        metric = Recall(
            metric_target=MetricTarget.MASKS,
            averaging_method=AveragingMethod.MACRO,
        )
        assert metric._metric_target == MetricTarget.MASKS
        assert metric.averaging_method == AveragingMethod.MACRO

    def test_reset(self, dummy_prediction):
        """Test that reset() clears all stored data"""
        metric = Recall()

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
        """Test recall with perfect matching predictions and targets"""
        metric = Recall()
        result = metric.update(detections_50_50, targets_50_50).compute()

        # Perfect match should give recall = 1.0
        # TP = 1, FN = 0 -> recall = TP / (TP + FN) = 1 / 1 = 1.0
        assert result.recall_at_50 == 1.0
        assert result.recall_at_75 == 1.0
        assert len(result.matched_classes) == 1
        assert result.matched_classes[0] == 0

    def test_no_overlap(self, predictions_no_overlap, targets_no_overlap):
        """Test recall with predictions that don't overlap with targets"""
        metric = Recall()
        result = metric.update(predictions_no_overlap, targets_no_overlap).compute()

        # No overlap means no TP, only FN
        # TP = 0, FN = 1 -> recall = TP / (TP + FN) = 0 / 1 = 0.0
        assert result.recall_at_50 == 0.0
        assert result.recall_at_75 == 0.0

    def test_empty_predictions(self, targets_50_50):
        """Test recall with empty predictions but existing targets"""
        predictions = Detections.empty()

        metric = Recall()
        result = metric.update(predictions, targets_50_50).compute()

        # No predictions means TP = 0, FN = 1 -> recall = 0 / 1 = 0.0
        assert result.recall_at_50 == 0.0
        assert result.recall_at_75 == 0.0

    def test_empty_targets(self, detections_50_50):
        """Test recall with predictions but no targets"""
        targets = Detections.empty()

        metric = Recall()
        result = metric.update(detections_50_50, targets).compute()

        # No targets means TP = 0, FN = 0 -> recall = 0 / 0 = 0
        assert result.recall_at_50 == 0.0
        assert result.recall_at_75 == 0.0

    def test_single_class_missed_detections(
        self, detections_50_50, targets_two_objects_class_0
    ):
        """Test recall calculation with some missed detections"""
        metric = Recall()
        result = metric.update(detections_50_50, targets_two_objects_class_0).compute()

        # TP = 1 (first target matched), FN = 1 (second target missed)
        # recall = TP / (TP + FN) = 1 / 2 = 0.5
        assert_almost_equal(result.recall_at_50, 0.5)
        assert_almost_equal(result.recall_at_75, 0.5)

    def test_multiple_classes(
        self, predictions_multiple_classes, targets_multiple_classes
    ):
        """Test recall calculation for multiple classes"""
        metric = Recall()
        result = metric.update(
            predictions_multiple_classes, targets_multiple_classes
        ).compute()

        # Class 0: TP=1, FN=1 -> recall=0.5 (weight=2)
        # Class 1: TP=1, FN=0 -> recall=1.0 (weight=1)
        # Weighted avg: (2*0.5 + 1*1.0) / (2+1) = 2.0/3 = 0.6667
        expected_recall = (2 * 0.5 + 1 * 1.0) / (2 + 1)
        assert_almost_equal(result.recall_at_50, expected_recall)
        assert_almost_equal(result.recall_at_75, expected_recall)
        assert len(result.matched_classes) == 2
        assert 0 in result.matched_classes
        assert 1 in result.matched_classes

    def test_different_iou_thresholds(self, predictions_iou_064, targets_iou_064):
        """Test recall at different IoU thresholds"""
        metric = Recall()
        result = metric.update(predictions_iou_064, targets_iou_064).compute()

        # IoU = 0.64 > 0.5 but < 0.75
        # Should match at IoU 0.5 but not at 0.75
        assert result.recall_at_50 == 1.0  # TP=1, FN=0
        assert result.recall_at_75 == 0.0  # TP=0, FN=1

    def test_confidence_ranking(self, predictions_confidence_ranking, targets_50_50):
        """Test that higher confidence predictions are preferred for matching"""
        metric = Recall()
        result = metric.update(predictions_confidence_ranking, targets_50_50).compute()

        # Target should be matched (by higher confidence prediction)
        # TP = 1, FN = 0 -> recall = 1.0
        assert result.recall_at_50 == 1.0

    def test_multiple_predictions_one_target(
        self, predictions_confidence_ranking, targets_50_50
    ):
        """Test recall when multiple predictions compete for one target"""
        metric = Recall()
        result = metric.update(predictions_confidence_ranking, targets_50_50).compute()

        # Target should be matched exactly once
        # TP = 1, FN = 0 -> recall = 1.0
        assert result.recall_at_50 == 1.0

    def test_list_inputs(
        self, detections_50_50, targets_50_50, prediction_class_1, target_class_1
    ):
        """Test recall with list inputs"""
        metric = Recall()
        result = metric.update(
            [detections_50_50, prediction_class_1], [targets_50_50, target_class_1]
        ).compute()

        # Perfect matches for both
        assert result.recall_at_50 == 1.0
        assert result.recall_at_75 == 1.0

    def test_mismatched_list_lengths(self, detections_50_50, targets_50_50):
        """Test that mismatched prediction/target list lengths raise error"""
        metric = Recall()

        # Should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="number of predictions"):
            metric.update([detections_50_50], [targets_50_50, targets_50_50])

    @pytest.mark.parametrize(
        "averaging_method",
        [AveragingMethod.MACRO, AveragingMethod.MICRO, AveragingMethod.WEIGHTED],
    )
    def test_averaging_methods(self, averaging_method, detections_50_50, targets_50_50):
        """Test different averaging methods"""
        metric = Recall(averaging_method=averaging_method)
        result = metric.update(detections_50_50, targets_50_50).compute()

        # Perfect match should give 1.0 regardless of averaging method
        assert result.recall_at_50 == 1.0
        assert result.averaging_method == averaging_method

    def test_macro_averaging(self):
        """Test MACRO averaging with specific example"""
        # Class 0: 1/2 targets matched -> recall = 0.5
        # Class 1: 1/1 targets matched -> recall = 1.0
        # Macro average: (0.5 + 1.0) / 2 = 0.75

        predictions = Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # matches class 0 target 1
                    [200, 200, 240, 240],  # matches class 1 target
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 1]),
        )

        targets = Detections(
            xyxy=np.array(
                [
                    [10, 10, 50, 50],  # class 0, matched
                    [60, 60, 100, 100],  # class 0, missed
                    [200, 200, 240, 240],  # class 1, matched
                ],
                dtype=np.float32,
            ),
            class_id=np.array([0, 0, 1]),
        )

        metric = Recall(averaging_method=AveragingMethod.MACRO)
        result = metric.update(predictions, targets).compute()

        # Macro average: (0.5 + 1.0) / 2 = 0.75
        assert result.recall_at_50 == 0.75
