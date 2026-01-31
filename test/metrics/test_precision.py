"""
Tests for Precision metric
"""

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.precision import Precision


def test_precision_initialization_default():
    """Test that Precision can be initialized with default parameters"""
    metric = Precision()
    assert metric._metric_target == MetricTarget.BOXES
    assert metric.averaging_method == AveragingMethod.WEIGHTED
    assert metric._predictions_list == []
    assert metric._targets_list == []


def test_precision_initialization_custom():
    """Test that Precision can be initialized with custom parameters"""
    metric = Precision(
        metric_target=MetricTarget.MASKS,
        averaging_method=AveragingMethod.MACRO,
    )
    assert metric._metric_target == MetricTarget.MASKS
    assert metric.averaging_method == AveragingMethod.MACRO


def test_precision_reset():
    """Test that reset() clears all stored data"""
    metric = Precision()

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


def test_precision_perfect_match():
    """Test precision with perfect matching predictions and targets"""
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

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # Perfect match should give precision = 1.0
    # TP = 1, FP = 0 -> precision = TP / (TP + FP) = 1 / 1 = 1.0
    assert result.precision_at_50 == 1.0
    assert result.precision_at_75 == 1.0
    assert len(result.matched_classes) == 1
    assert result.matched_classes[0] == 0


def test_precision_no_overlap():
    """Test precision with predictions that don't overlap with targets"""
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

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # No overlap means no TP, only FP
    # TP = 0, FP = 1 -> precision = TP / (TP + FP) = 0 / 1 = 0.0
    assert result.precision_at_50 == 0.0
    assert result.precision_at_75 == 0.0


def test_precision_empty_predictions():
    """Test precision with empty predictions but existing targets"""
    predictions = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )

    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # No predictions means TP = 0, FP = 0 -> precision = 0 / 0 = 0
    assert result.precision_at_50 == 0.0
    assert result.precision_at_75 == 0.0


def test_precision_empty_targets():
    """Test precision with predictions but no targets"""
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    targets = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # All predictions are false positives
    # TP = 0, FP = 1 -> precision = 0 / 1 = 0.0
    assert result.precision_at_50 == 0.0
    assert result.precision_at_75 == 0.0


def test_precision_single_class():
    """Test precision calculation for single class with mixed results"""
    # Two predictions, one matches target, one doesn't
    predictions = Detections(
        xyxy=np.array([[10, 10, 50, 50], [100, 100, 110, 110]], dtype=np.float32),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
    )

    # Only one target that matches first prediction
    targets = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # TP = 1 (first prediction), FP = 1 (second prediction)
    # precision = TP / (TP + FP) = 1 / 2 = 0.5
    assert result.precision_at_50 == 0.5
    assert result.precision_at_75 == 0.5


def test_precision_multiple_classes():
    """Test precision calculation for multiple classes"""
    # Class 0: 2 predictions, 2 targets
    # Class 1: 1 prediction, 1 target
    predictions = Detections(
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

    targets = Detections(
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

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # All predictions match targets perfectly
    # Class 0: TP=2, FP=0 -> precision=1.0 (weight=2)
    # Class 1: TP=1, FP=0 -> precision=1.0 (weight=1)
    # Weighted avg: (2*1.0 + 1*1.0) / (2+1) = 3/3 = 1.0
    assert result.precision_at_50 == 1.0
    assert result.precision_at_75 == 1.0
    assert len(result.matched_classes) == 2
    assert 0 in result.matched_classes
    assert 1 in result.matched_classes


def test_precision_different_iou_thresholds():
    """Test precision at different IoU thresholds"""
    # Prediction slightly overlaps with target (IoU ~ 0.64)
    # Box areas: pred = 40x40 = 1600, target = 50x50 = 2500
    # Intersection: 30x30 = 900
    # Union: 1600 + 2500 - 900 = 3200
    # IoU = 900 / 3200 = 0.28125 (too low for any threshold)

    # Let's create a better overlap: IoU ~ 0.7
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

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # IoU = 0.64 > 0.5 but < 0.75
    # Should match at IoU 0.5 but not at 0.75
    assert result.precision_at_50 == 1.0  # TP=1, FP=0
    assert result.precision_at_75 == 0.0  # TP=0, FP=1


def test_precision_confidence_ranking():
    """Test that predictions are ranked by confidence"""
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

    metric = Precision()
    result = metric.update(predictions, targets).compute()

    # Higher confidence prediction should match first
    # TP = 1, FP = 1 -> precision = 0.5
    assert result.precision_at_50 == 0.5


def test_precision_list_inputs():
    """Test precision with list inputs"""
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

    metric = Precision()
    result = metric.update([pred1, pred2], [target1, target2]).compute()

    # Perfect matches for both
    assert result.precision_at_50 == 1.0
    assert result.precision_at_75 == 1.0


def test_precision_mismatched_list_lengths():
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

    metric = Precision()

    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        metric.update([pred], [target, target])


@pytest.mark.parametrize(
    "averaging_method",
    [AveragingMethod.MACRO, AveragingMethod.MICRO, AveragingMethod.WEIGHTED],
)
def test_precision_averaging_methods(averaging_method):
    """Test different averaging methods"""
    metric = Precision(averaging_method=averaging_method)

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
    assert result.precision_at_50 == 1.0
    assert result.averaging_method == averaging_method
