"""
Tests for Mean Average Precision ID=0 bug fix
"""

import numpy as np

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision


def test_single_perfect_detection():
    """Test that single perfect detection gets 1.0 mAP (not 0.0 due to ID=0 bug)"""
    # Perfect detection (identical prediction and target)
    detection = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )

    metric = MeanAveragePrecision()
    metric.update([detection], [detection])
    result = metric.compute()

    # Should be perfect 1.0 mAP, not 0.0 due to ID=0 bug
    assert abs(result.map50_95 - 1.0) < 1e-6


def test_multiple_perfect_detections():
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


def test_batch_updates_perfect_detections():
    """Test that batch updates with perfect detections get 1.0 mAP"""
    # Single perfect detection for multiple batch updates
    detection = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )

    metric = MeanAveragePrecision()
    # Add 3 batch updates
    metric.update([detection], [detection])
    metric.update([detection], [detection])
    metric.update([detection], [detection])
    result = metric.compute()

    # Should be perfect 1.0 mAP across all batches
    assert abs(result.map50_95 - 1.0) < 1e-6


def test_scenario_1_success_case_imperfect_match():
    """Scenario 1: Success Case with imperfect match"""
    # Small object (class 0) - area = 30*30 = 900 < 1024
    small_perfect = Detections(
        xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.95]),
        data={"area": np.array([900])}
    )
    
    # Medium object (class 1) - area = 50*50 = 2500 (between 1024 and 9216)
    medium_target = Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float64),
        class_id=np.array([1]),
        data={"area": np.array([2500])}
    )
    medium_pred = Detections(
        xyxy=np.array([[12, 12, 60, 60]], dtype=np.float64),  # Slightly off
        class_id=np.array([1]),
        confidence=np.array([0.9]),
        data={"area": np.array([2304])}  # 48*48
    )
    
    # Large objects (classes 0, 1, 2) - area = 100*100 = 10000 > 9216
    large_targets = Detections(
        xyxy=np.array([
            [10, 10, 110, 110],
            [120, 120, 220, 220],
            [230, 230, 330, 330]
        ], dtype=np.float64),
        class_id=np.array([2, 0, 1]),
        data={"area": np.array([10000, 10000, 10000])}
    )
    large_preds = Detections(
        xyxy=np.array([
            [10, 10, 110, 110],
            [120, 120, 220, 220],
            [230, 230, 330, 330]
        ], dtype=np.float64),
        class_id=np.array([2, 0, 1]),
        confidence=np.array([0.9, 0.9, 0.9]),
        data={"area": np.array([10000, 10000, 10000])}
    )
    
    metric = MeanAveragePrecision()
    metric.update([small_perfect], [small_perfect])
    metric.update([medium_pred], [medium_target])
    metric.update([large_preds], [large_targets])
    result = metric.compute()
    
    # Should be close to 0.9 (slightly less than perfect due to medium object)
    assert 0.85 < result.map50_95 < 0.98  # Adjusted upper bound
    assert result.medium_objects.map50_95 < 1.0  # Medium should be less than perfect


def test_scenario_2_missed_detection():
    """Scenario 2: GT Present, No Prediction (Missed Detection)"""
    # Small object - area = 30*30 = 900 < 1024
    small_detection = Detections(
        xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.95]),
        data={"area": np.array([900])}
    )
    
    # Medium object - area = 50*50 = 2500 (between 1024 and 9216) - no prediction (missed)
    medium_target = Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float64),
        class_id=np.array([1]),
        data={"area": np.array([2500])}
    )
    no_medium_pred = Detections.empty()
    
    # Large objects - area = 100*100 = 10000 > 9216
    large_detections = Detections(
        xyxy=np.array([
            [10, 10, 110, 110],
            [120, 120, 220, 220],
            [230, 230, 330, 330]
        ], dtype=np.float64),
        class_id=np.array([2, 0, 1]),
        confidence=np.array([0.9, 0.9, 0.9]),
        data={"area": np.array([10000, 10000, 10000])}
    )
    
    metric = MeanAveragePrecision()
    metric.update([small_detection], [small_detection])
    metric.update([no_medium_pred], [medium_target])
    metric.update([large_detections], [large_detections])
    result = metric.compute()
    
    # Medium objects should have 0.0 mAP (missed detection)
    assert abs(result.medium_objects.map50_95 - 0.0) < 1e-6


def test_scenario_3_false_positive():
    """Scenario 3: No GT, Prediction Present (False Positive)"""
    # Small object - area = 30*30 = 900 < 1024
    small_detection = Detections(
        xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.95]),
        data={"area": np.array([900])}
    )
    
    # Medium object - area = 50*50 = 2500 - false positive (no GT)
    medium_pred = Detections(
        xyxy=np.array([[12, 12, 62, 62]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.9]),
        data={"area": np.array([2500])}
    )
    no_medium_target = Detections.empty()
    
    # Large objects - area = 100*100 = 10000 > 9216
    large_detections = Detections(
        xyxy=np.array([
            [10, 10, 110, 110],
            [120, 120, 220, 220],
            [230, 230, 330, 330]
        ], dtype=np.float64),
        class_id=np.array([2, 0, 1]),
        confidence=np.array([0.9, 0.9, 0.9]),
        data={"area": np.array([10000, 10000, 10000])}
    )
    
    metric = MeanAveragePrecision()
    metric.update([small_detection], [small_detection])
    metric.update([medium_pred], [no_medium_target])
    metric.update([large_detections], [large_detections])
    result = metric.compute()
    
    # Medium objects should have -1 mAP (false positive, matching pycocotools)
    assert result.medium_objects.map50_95 == -1


def test_scenario_4_no_data():
    """Scenario 4: No GT, No Prediction (Category has no data)"""
    # Small object - area = 30*30 = 900 < 1024
    small_detection = Detections(
        xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.95]),
        data={"area": np.array([900])}
    )
    
    # Medium object - no data at all
    no_medium = Detections.empty()
    
    # Large objects - area = 100*100 = 10000 > 9216 - only classes 0 and 2 (no class 1)
    large_targets = Detections(
        xyxy=np.array([
            [10, 10, 110, 110],
            [120, 120, 220, 220],
        ], dtype=np.float64),
        class_id=np.array([2, 0]),
        data={"area": np.array([10000, 10000])}
    )
    large_preds = Detections(
        xyxy=np.array([
            [10, 10, 110, 110],
            [120, 120, 220, 220],
        ], dtype=np.float64),
        class_id=np.array([2, 0]),
        confidence=np.array([0.9, 0.9]),
        data={"area": np.array([10000, 10000])}
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


def test_scenario_5_only_one_class_present():
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


def test_mixed_classes_with_missing_detections():
    """Test mixed scenario with some classes having no detections"""
    # Class 0: Perfect detection
    class_0_det = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )
    
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
    metric.update([class_0_det], [class_0_det])
    metric.update([class_1_pred], [class_1_target])
    metric.update([class_2_pred], [class_2_target])
    result = metric.compute()
    
    # Should not have negative mAP
    assert result.map50_95 >= 0.0
    # Should be less than 1.0 due to missed detection and false positive
    assert result.map50_95 < 1.0


def test_empty_predictions_and_targets():
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
