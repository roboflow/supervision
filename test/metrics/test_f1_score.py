"""
Tests for F1Score metric
"""

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.f1_score import F1Score


def test_f1_score_initialization_default():
    """Test that F1Score can be initialized with default parameters"""
    metric = F1Score()
    assert metric._metric_target == MetricTarget.BOXES
    assert metric.averaging_method == AveragingMethod.WEIGHTED
    assert metric._predictions_list == []
    assert metric._targets_list == []


def test_f1_score_initialization_custom():
    """Test that F1Score can be initialized with custom parameters"""
    metric = F1Score(
        metric_target=MetricTarget.MASKS,
        averaging_method=AveragingMethod.MACRO,
    )
    assert metric._metric_target == MetricTarget.MASKS
    assert metric.averaging_method == AveragingMethod.MACRO


def test_f1_score_reset():
    """Test that reset() clears all stored data"""
    metric = F1Score()

    # Add some dummy data
    dummy_prediction = Detections(
        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
        confidence=np.array([0.8]),
        class_id=np.array([0]),
    )
    metric.update(dummy_prediction, dummy_prediction)

    # Verify data was added
    assert len(metric._predictions_list) == 1
    assert len(metric._targets_list) == 1

    # Reset and verify lists are empty
    metric.reset()
    assert metric._predictions_list == []
    assert metric._targets_list == []


def test_f1_score_perfect_match():
    """Test F1 score with perfect matching predictions and targets"""
    # Create identical predictions and targets
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # Perfect match should give F1 = 1.0
    # TP = 1, FP = 0, FN = 0
    # Precision = TP / (TP + FP) = 1 / 1 = 1.0
    # Recall = TP / (TP + FN) = 1 / 1 = 1.0
    # F1 = 2 * (P * R) / (P + R) = 2 * 1.0 / 2 = 1.0
    assert result.f1_50 == 1.0
    assert result.f1_75 == 1.0
    assert len(result.matched_classes) == 1
    assert result.matched_classes[0] == 0


def test_f1_score_no_overlap():
    """Test F1 score with predictions that don't overlap with targets"""
    # Predictions and targets are completely separate
    predictions = Detections(
        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.array([[100, 100, 110, 110]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # No overlap means TP=0, FP=1, FN=1
    # Precision = 0 / 1 = 0.0
    # Recall = 0 / 1 = 0.0
    # F1 = 2 * (0 * 0) / (0 + 0) = 0 / 0 = 0.0
    assert result.f1_50 == 0.0
    assert result.f1_75 == 0.0


def test_f1_score_empty_predictions():
    """Test F1 score with empty predictions but existing targets"""
    predictions = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # No predictions: TP=0, FP=0, FN=1
    # Precision = 0 / 0 = 0 (by convention)
    # Recall = 0 / 1 = 0.0
    # F1 = 0.0
    assert result.f1_50 == 0.0
    assert result.f1_75 == 0.0


def test_f1_score_empty_targets():
    """Test F1 score with predictions but no targets"""
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # No targets: TP=0, FP=1, FN=0
    # Precision = 0 / 1 = 0.0
    # Recall = 0 / 0 = 0 (by convention)
    # F1 = 0.0
    assert result.f1_50 == 0.0
    assert result.f1_75 == 0.0


def test_f1_score_single_class_mixed_results():
    """Test F1 score calculation with mixed precision and recall"""
    # Two predictions, one target
    # One prediction matches, one doesn't
    predictions = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # matches target - TP
                [100, 100, 110, 110],  # doesn't match - FP
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # TP=1, FP=1, FN=0
    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    # Recall = TP / (TP + FN) = 1 / 1 = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3 ≈ 0.6667
    expected_f1 = 2.0 / 3.0
    assert abs(result.f1_50 - expected_f1) < 1e-6
    assert abs(result.f1_75 - expected_f1) < 1e-6


def test_f1_score_precision_recall_imbalance():
    """Test F1 score with different precision and recall scenarios"""
    # Two targets, one prediction that matches first target
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # matched
                [100, 100, 110, 110],  # missed
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # TP=1, FP=0, FN=1
    # Precision = TP / (TP + FP) = 1 / 1 = 1.0
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 2/3 ≈ 0.6667
    expected_f1 = 2.0 / 3.0
    assert abs(result.f1_50 - expected_f1) < 1e-6
    assert abs(result.f1_75 - expected_f1) < 1e-6


def test_f1_score_multiple_classes():
    """Test F1 score calculation for multiple classes"""
    # Class 0: 1 prediction, 1 target (perfect match)
    # Class 1: 2 predictions, 1 target (one correct, one FP)
    predictions = Detections(
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

    targets = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # class 0
                [60, 60, 100, 100],  # class 1
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # Class 0: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0 (weight=1)
    # Class 1: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3 (weight=1)
    # Weighted avg: (1*1.0 + 1*2/3) / (1+1) = (1 + 2/3) / 2 = 5/6 ≈ 0.8333
    expected_f1 = (1.0 + 2.0 / 3.0) / 2.0
    assert abs(result.f1_50 - expected_f1) < 1e-6
    assert len(result.matched_classes) == 2
    assert 0 in result.matched_classes
    assert 1 in result.matched_classes


def test_f1_score_different_iou_thresholds():
    """Test F1 score at different IoU thresholds"""
    # Prediction partially overlaps with target
    # Target: [10, 10, 60, 60] = 50x50
    # Prediction: [15, 15, 55, 55] = 40x40
    # Intersection: [15, 15, 55, 55] = 40x40 = 1600
    # Union: 2500 + 1600 - 1600 = 2500
    # IoU = 1600 / 2500 = 0.64

    predictions = Detections(
        xyxy=np.array([[15, 15, 55, 55]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # IoU = 0.64 > 0.5 but < 0.75
    # At IoU 0.5: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    # At IoU 0.75: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0
    assert result.f1_50 == 1.0
    assert result.f1_75 == 0.0


def test_f1_score_confidence_ranking():
    """Test that F1 score respects confidence ranking"""
    # Two predictions for one target, higher confidence should win
    predictions = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # low confidence, perfect match
                [11, 11, 49, 49],  # high confidence, good match
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.6, 0.9]),  # second has higher confidence
        class_id=np.array([0, 0]),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()
    result = metric.update(predictions, targets).compute()

    # Higher confidence prediction should match the target
    # TP=1, FP=1, FN=0
    # Precision = 1/2 = 0.5, Recall = 1/1 = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
    expected_f1 = 2.0 / 3.0
    assert abs(result.f1_50 - expected_f1) < 1e-6


def test_f1_score_list_inputs():
    """Test F1 score with list inputs"""
    # Test with lists of Detections
    pred1 = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    pred2 = Detections(
        xyxy=np.array([[60, 60, 100, 100]], dtype=np.float32),
        confidence=np.array([0.8]),
        class_id=np.array([1]),
    )

    target1 = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )
    target2 = Detections(
        xyxy=np.array([[60, 60, 100, 100]], dtype=np.float32),
        class_id=np.array([1]),
    )

    metric = F1Score()
    result = metric.update([pred1, pred2], [target1, target2]).compute()

    # Perfect matches for both
    assert result.f1_50 == 1.0
    assert result.f1_75 == 1.0


def test_f1_score_mismatched_list_lengths():
    """Test that mismatched prediction/target list lengths raise error"""
    pred = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    target = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = F1Score()

    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        metric.update([pred], [target, target])


@pytest.mark.parametrize(
    "averaging_method",
    [AveragingMethod.MACRO, AveragingMethod.MICRO, AveragingMethod.WEIGHTED],
)
def test_f1_score_averaging_methods(averaging_method):
    """Test different averaging methods"""
    metric = F1Score(averaging_method=averaging_method)

    # Test with simple perfect match case
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    result = metric.update(predictions, targets).compute()

    # Perfect match should give 1.0 regardless of averaging method
    assert result.f1_50 == 1.0
    assert result.averaging_method == averaging_method


def test_f1_score_macro_averaging():
    """Test MACRO averaging with specific example"""
    # Class 0: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    # Class 1: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3
    # Macro average: (1.0 + 2/3) / 2 = 5/6

    predictions = Detections(
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

    targets = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # class 0
                [60, 60, 100, 100],  # class 1
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
    )

    metric = F1Score(averaging_method=AveragingMethod.MACRO)
    result = metric.update(predictions, targets).compute()

    # Macro average: (1.0 + 2/3) / 2 = 5/6
    expected_f1 = (1.0 + 2.0 / 3.0) / 2.0
    assert abs(result.f1_50 - expected_f1) < 1e-6


def test_f1_score_micro_averaging():
    """Test MICRO averaging with specific example"""
    # Micro averaging pools all TP, FP, FN across classes
    # Total: TP=2, FP=1, FN=0
    # P = 2/3, R = 2/2 = 1.0, F1 = 2*(2/3*1.0)/(2/3 + 1.0) = (4/3)/(5/3) = 4/5 = 0.8

    predictions = Detections(
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

    targets = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # class 0
                [60, 60, 100, 100],  # class 1
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
    )

    metric = F1Score(averaging_method=AveragingMethod.MICRO)
    result = metric.update(predictions, targets).compute()

    # Micro F1: 4/5 = 0.8
    expected_f1 = 0.8
    assert abs(result.f1_50 - expected_f1) < 1e-6


def test_f1_score_weighted_averaging():
    """Test WEIGHTED averaging with specific example"""
    # Class 0: F1=1.0 (weight=1)
    # Class 1: F1=2/3 (weight=1)
    # Weighted average: (1*1.0 + 1*2/3) / (1+1) = 5/6

    predictions = Detections(
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

    targets = Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # class 0
                [60, 60, 100, 100],  # class 1
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
    )

    metric = F1Score(averaging_method=AveragingMethod.WEIGHTED)
    result = metric.update(predictions, targets).compute()

    # Weighted average: 5/6
    expected_f1 = 5.0 / 6.0
    assert abs(result.f1_50 - expected_f1) < 1e-6
