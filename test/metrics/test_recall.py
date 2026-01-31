"""
Tests for Recall metric
"""

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.recall import Recall


def test_recall_initialization_default():
    """Test that Recall can be initialized with default parameters"""
    metric = Recall()
    assert metric._metric_target == MetricTarget.BOXES
    assert metric.averaging_method == AveragingMethod.WEIGHTED
    assert metric._predictions_list == []
    assert metric._targets_list == []


def test_recall_initialization_custom():
    """Test that Recall can be initialized with custom parameters"""
    metric = Recall(
        metric_target=MetricTarget.MASKS,
        averaging_method=AveragingMethod.MACRO,
    )
    assert metric._metric_target == MetricTarget.MASKS
    assert metric.averaging_method == AveragingMethod.MACRO


def test_recall_reset():
    """Test that reset() clears all stored data"""
    metric = Recall()
    
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


def test_recall_perfect_match():
    """Test recall with perfect matching predictions and targets"""
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
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # Perfect match should give recall = 1.0
    # TP = 1, FN = 0 -> recall = TP / (TP + FN) = 1 / 1 = 1.0
    assert result.recall_at_50 == 1.0
    assert result.recall_at_75 == 1.0
    assert len(result.matched_classes) == 1
    assert result.matched_classes[0] == 0


def test_recall_no_overlap():
    """Test recall with predictions that don't overlap with targets"""
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
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # No overlap means no TP, only FN
    # TP = 0, FN = 1 -> recall = TP / (TP + FN) = 0 / 1 = 0.0
    assert result.recall_at_50 == 0.0
    assert result.recall_at_75 == 0.0


def test_recall_empty_predictions():
    """Test recall with empty predictions but existing targets"""
    predictions = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )
    
    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # No predictions means TP = 0, FN = 1 -> recall = 0 / 1 = 0.0
    assert result.recall_at_50 == 0.0
    assert result.recall_at_75 == 0.0


def test_recall_empty_targets():
    """Test recall with predictions but no targets"""
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    
    targets = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # No targets means TP = 0, FN = 0 -> recall = 0 / 0 = 0
    assert result.recall_at_50 == 0.0
    assert result.recall_at_75 == 0.0


def test_recall_single_class_missed_detections():
    """Test recall calculation with some missed detections"""
    # One prediction that matches one of two targets
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    
    # Two targets, only first one will be matched
    targets = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],    # matches prediction
            [100, 100, 110, 110] # missed by prediction
        ], dtype=np.float32),
        class_id=np.array([0, 0]),
    )
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # TP = 1 (first target matched), FN = 1 (second target missed)
    # recall = TP / (TP + FN) = 1 / 2 = 0.5
    assert result.recall_at_50 == 0.5
    assert result.recall_at_75 == 0.5


def test_recall_multiple_classes():
    """Test recall calculation for multiple classes"""
    # Class 0: 1 prediction matches 1 of 2 targets
    # Class 1: 1 prediction matches 1 of 1 targets
    predictions = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],   # class 0, matches first target
            [200, 200, 240, 240] # class 1, matches target
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 1]),
    )
    
    targets = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],     # class 0, matched
            [60, 60, 100, 100],   # class 0, missed  
            [200, 200, 240, 240]  # class 1, matched
        ], dtype=np.float32),
        class_id=np.array([0, 0, 1]),
    )
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # Class 0: TP=1, FN=1 -> recall=0.5 (weight=2)
    # Class 1: TP=1, FN=0 -> recall=1.0 (weight=1)
    # Weighted avg: (2*0.5 + 1*1.0) / (2+1) = 2.0/3 = 0.6667
    expected_recall = (2 * 0.5 + 1 * 1.0) / (2 + 1)
    assert abs(result.recall_at_50 - expected_recall) < 1e-6
    assert abs(result.recall_at_75 - expected_recall) < 1e-6
    assert len(result.matched_classes) == 2
    assert 0 in result.matched_classes
    assert 1 in result.matched_classes


def test_recall_different_iou_thresholds():
    """Test recall at different IoU thresholds"""
    # Prediction slightly overlaps with target
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
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # IoU = 0.64 > 0.5 but < 0.75
    # Should match at IoU 0.5 but not at 0.75
    assert result.recall_at_50 == 1.0  # TP=1, FN=0
    assert result.recall_at_75 == 0.0  # TP=0, FN=1


def test_recall_confidence_ranking():
    """Test that higher confidence predictions are preferred for matching"""
    # Two predictions for one target, higher confidence should win
    predictions = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],   # low confidence, perfect match
            [11, 11, 49, 49],   # high confidence, good match
        ], dtype=np.float32),
        confidence=np.array([0.6, 0.9]),  # second has higher confidence
        class_id=np.array([0, 0]),
    )
    
    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # Target should be matched (by higher confidence prediction)
    # TP = 1, FN = 0 -> recall = 1.0
    assert result.recall_at_50 == 1.0


def test_recall_multiple_predictions_one_target():
    """Test recall when multiple predictions compete for one target"""
    # Two predictions, one target - only best match should count
    predictions = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],   # perfect match
            [12, 12, 52, 52],   # good match but slightly offset
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
    )
    
    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )
    
    metric = Recall()
    result = metric.update(predictions, targets).compute()
    
    # Target should be matched exactly once
    # TP = 1, FN = 0 -> recall = 1.0
    assert result.recall_at_50 == 1.0


def test_recall_list_inputs():
    """Test recall with list inputs"""
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
    
    metric = Recall()
    result = metric.update([pred1, pred2], [target1, target2]).compute()
    
    # Perfect matches for both
    assert result.recall_at_50 == 1.0
    assert result.recall_at_75 == 1.0


def test_recall_mismatched_list_lengths():
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
    
    metric = Recall()
    
    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        metric.update([pred], [target, target])


@pytest.mark.parametrize("averaging_method", [
    AveragingMethod.MACRO,
    AveragingMethod.MICRO, 
    AveragingMethod.WEIGHTED
])
def test_recall_averaging_methods(averaging_method):
    """Test different averaging methods"""
    metric = Recall(averaging_method=averaging_method)
    
    # Test with simple case
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
    assert result.recall_at_50 == 1.0
    assert result.averaging_method == averaging_method


def test_recall_macro_averaging():
    """Test MACRO averaging with specific example"""
    # Class 0: 1/2 targets matched -> recall = 0.5
    # Class 1: 1/1 targets matched -> recall = 1.0
    # Macro average: (0.5 + 1.0) / 2 = 0.75
    
    predictions = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],     # matches class 0 target 1
            [200, 200, 240, 240]  # matches class 1 target
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 1]),
    )
    
    targets = Detections(
        xyxy=np.array([
            [10, 10, 50, 50],     # class 0, matched
            [60, 60, 100, 100],   # class 0, missed
            [200, 200, 240, 240]  # class 1, matched
        ], dtype=np.float32),
        class_id=np.array([0, 0, 1]),
    )
    
    metric = Recall(averaging_method=AveragingMethod.MACRO)
    result = metric.update(predictions, targets).compute()
    
    # Macro average: (0.5 + 1.0) / 2 = 0.75
    assert abs(result.recall_at_50 - 0.75) < 1e-6