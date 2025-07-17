"""
Tests for Mean Average Precision ID=0 bug fix
"""
import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision


def test_single_perfect_detection():
    """Test that single perfect detection gets 1.0 mAP (not 0.0 due to ID=0 bug)"""
    # Perfect detection (identical prediction and target)
    detection = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9])
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
        xyxy=np.array([
            [10, 10, 50, 50],
            [100, 100, 140, 140],
            [200, 200, 240, 240]
        ], dtype=np.float64),
        class_id=np.array([0, 0, 0]),
        confidence=np.array([0.9, 0.9, 0.9])
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
        confidence=np.array([0.9])
    )
    
    metric = MeanAveragePrecision()
    # Add 3 batch updates
    metric.update([detection], [detection])
    metric.update([detection], [detection])
    metric.update([detection], [detection])
    result = metric.compute()
    
    # Should be perfect 1.0 mAP across all batches
    assert abs(result.map50_95 - 1.0) < 1e-6 