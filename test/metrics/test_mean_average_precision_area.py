from __future__ import annotations

import numpy as np

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class TestMeanAveragePrecisionArea:
    """Test area calculation in MeanAveragePrecision."""

    def test_area_calculated_from_bbox_when_data_empty(self):
        """Test that area is calculated from bbox when data is empty (normal case)."""
        # Create detections with empty data (normal case)
        gt = Detections(
            xyxy=np.array([
                [10, 10, 40, 40],      # Small: 30x30 = 900
                [100, 100, 200, 150],  # Medium: 100x50 = 5000
                [300, 300, 500, 400]   # Large: 200x100 = 20000
            ], dtype=np.float32),
            class_id=np.array([0, 0, 0]),
            confidence=np.array([1.0, 1.0, 1.0])
        )
        
        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=gt.class_id.copy(),
            confidence=np.array([0.9, 0.9, 0.9])
        )
        
        # Verify data is empty (normal case)
        assert gt.data == {}
        assert pred.data == {}
        
        # Create mAP metric and test area calculation
        map_metric = MeanAveragePrecision()
        map_metric.update([pred], [gt])
        
        # Check that areas were calculated correctly from bbox
        prepared_targets = map_metric._prepare_targets(map_metric._targets_list)
        
        areas = [ann["area"] for ann in prepared_targets["annotations"]]
        expected_areas = [900.0, 5000.0, 20000.0]  # width * height for each bbox
        
        assert np.allclose(areas, expected_areas, rtol=1e-05, atol=1e-08), f"Expected {expected_areas}, got {areas}"
        
        # Verify mAP works correctly (no -1.0 for medium/large objects)
        result = map_metric.compute()
        assert result.medium_objects.map50 >= 0.0, "Medium objects should have valid mAP"
        assert result.large_objects.map50 >= 0.0, "Large objects should have valid mAP"

    def test_area_preserved_when_provided_in_data(self):
        """Test that area from data is preserved when provided (COCO case)."""
        # Create detections with area in data (COCO style)
        gt = Detections(
            xyxy=np.array([[100, 100, 200, 150]], dtype=np.float32),  # Would be 5000
            class_id=np.array([0]),
            confidence=np.array([1.0])
        )
        
        # Add custom area to data (different from calculated)
        gt.data = {"area": np.array([3000.0])}
        
        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=gt.class_id.copy(),
            confidence=np.array([0.9])
        )
        pred.data = {"area": np.array([3000.0])}
        
        # Test area calculation
        map_metric = MeanAveragePrecision()
        map_metric.update([pred], [gt])
        
        # Check that provided area is used (not calculated)
        prepared_targets = map_metric._prepare_targets(map_metric._targets_list)
        used_area = prepared_targets["annotations"][0]["area"]
        
        assert np.allclose(used_area, 3000.0, rtol=1e-05, atol=1e-08), f"Should use provided area 3000.0, got {used_area}"
        
        # Verify it's different from what would be calculated
        calculated_area = (200 - 100) * (150 - 100)  # 100 * 50 = 5000
        assert not np.allclose(used_area, calculated_area, rtol=1e-05, atol=1e-08), "Should use provided area, not calculated"

    def test_mixed_area_sources(self):
        """Test mix of detections with and without area in data."""
        # Create detections where some have area in data, others don't
        gt1 = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float32),  # 900
            class_id=np.array([0])
        )
        # No area in data - should be calculated
        
        gt2 = Detections(
            xyxy=np.array([[100, 100, 200, 150]], dtype=np.float32),  # 5000
            class_id=np.array([1])
        )
        # Add area in data - should be preserved
        gt2.data = {"area": np.array([3000.0])}
        
        pred1 = Detections(
            xyxy=gt1.xyxy.copy(),
            class_id=gt1.class_id.copy(),
            confidence=np.array([0.9])
        )
        
        pred2 = Detections(
            xyxy=gt2.xyxy.copy(),
            class_id=gt2.class_id.copy(),
            confidence=np.array([0.8])
        )
        pred2.data = {"area": np.array([3000.0])}
        
        # Test area calculation for mixed sources
        map_metric = MeanAveragePrecision()
        map_metric.update([pred1, pred2], [gt1, gt2])
        
        prepared_targets = map_metric._prepare_targets(map_metric._targets_list)
        areas = [ann["area"] for ann in prepared_targets["annotations"]]
        
        expected_areas = [900.0, 3000.0]  # calculated, then provided
        assert np.allclose(areas, expected_areas, rtol=1e-05, atol=1e-08), f"Expected {expected_areas}, got {areas}"

    def test_size_specific_map_works_correctly(self):
        """Test that size-specific mAP works correctly with area fix."""
        # Create detections with one object of each size
        gt = Detections(
            xyxy=np.array([
                [10, 10, 40, 40],      # Small: 30x30 = 900 < 1024
                [100, 100, 200, 150],  # Medium: 100x50 = 5000 (1024 <= x < 9216)
                [300, 300, 500, 400]   # Large: 200x100 = 20000 >= 9216
            ], dtype=np.float32),
            class_id=np.array([0, 0, 0])
        )
        
        # Perfect predictions
        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=gt.class_id.copy(),
            confidence=np.array([0.9, 0.9, 0.9])
        )
        
        # Test mAP calculation
        map_metric = MeanAveragePrecision()
        map_metric.update([pred], [gt])
        result = map_metric.compute()
        
        # All size categories should have valid results (not -1.0)
        assert result.small_objects.map50 >= 0.0, "Small objects should have valid mAP"
        assert result.medium_objects.map50 >= 0.0, "Medium objects should have valid mAP"
        assert result.large_objects.map50 >= 0.0, "Large objects should have valid mAP"
        
        # Perfect matches should yield high mAP for medium and large
        assert result.medium_objects.map50 > 0.9, "Perfect medium matches should have high mAP"
        assert result.large_objects.map50 > 0.9, "Perfect large matches should have high mAP"

    def test_area_uses_detections_property(self):
        """Test that area calculation uses Detections.area property correctly."""
        # Create detection
        gt = Detections(
            xyxy=np.array([[100, 100, 200, 150]], dtype=np.float32),
            class_id=np.array([0])
        )
        
        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=gt.class_id.copy(),
            confidence=np.array([0.9])
        )
        
        # Test that internal calculation matches Detections.area property
        map_metric = MeanAveragePrecision()
        map_metric.update([pred], [gt])
        
        prepared_targets = map_metric._prepare_targets(map_metric._targets_list)
        used_area = prepared_targets["annotations"][0]["area"]
        expected_area = gt.area[0]
        
        assert np.allclose(used_area, expected_area, rtol=1e-05, atol=1e-08), f"Should use Detections.area property {expected_area}, got {used_area}" 