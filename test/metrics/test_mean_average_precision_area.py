from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class TestMeanAveragePrecisionArea:
    """Test area calculation in MeanAveragePrecision."""

    @pytest.mark.parametrize(
        "xyxy, expected_areas, expected_size_maps",
        [
            (
                np.array(
                    [
                        [10, 10, 40, 40],  # Small: 900
                        [100, 100, 200, 150],  # Medium: 5000
                        [300, 300, 500, 400],  # Large: 20000
                    ],
                    dtype=np.float32,
                ),
                [900.0, 5000.0, 20000.0],
                {"small": True, "medium": True, "large": True},
            ),
            (
                np.array([[0, 0, 10, 10]], dtype=np.float32),  # Small: 100
                [100.0],
                {"small": True, "medium": False, "large": False},
            ),
            (
                np.array([[0, 0, 50, 50]], dtype=np.float32),  # Medium: 2500
                [2500.0],
                {"small": False, "medium": True, "large": False},
            ),
            (
                np.array([[0, 0, 100, 100]], dtype=np.float32),  # Large: 10000
                [10000.0],
                {"small": False, "medium": False, "large": True},
            ),
        ],
    )
    def test_area_calculation_and_size_specific_map(
        self, xyxy, expected_areas, expected_size_maps
    ):
        """Test area calculation and size-specific mAP functionality."""
        gt = Detections(
            xyxy=xyxy,
            class_id=np.arange(len(xyxy)),
        )
        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=gt.class_id.copy(),
            confidence=np.full(len(xyxy), 0.9),
        )

        map_metric = MeanAveragePrecision()
        map_metric.update([pred], [gt])

        # Test area calculation
        prepared_targets = map_metric._prepare_targets(map_metric._targets_list)
        areas = [ann["area"] for ann in prepared_targets["annotations"]]
        assert np.allclose(areas, expected_areas), (
            f"Expected {expected_areas}, got {areas}"
        )

        # Test size-specific mAP
        result = map_metric.compute()

        if expected_size_maps["small"]:
            assert result.small_objects.map50 > 0.9, (
                "Small objects should have high mAP"
            )
        else:
            assert result.small_objects.map50 == -1.0, (
                "Small objects should have no data"
            )

        if expected_size_maps["medium"]:
            assert result.medium_objects.map50 > 0.9, (
                "Medium objects should have high mAP"
            )
        else:
            assert result.medium_objects.map50 == -1.0, (
                "Medium objects should have no data"
            )

        if expected_size_maps["large"]:
            assert result.large_objects.map50 > 0.9, (
                "Large objects should have high mAP"
            )
        else:
            assert result.large_objects.map50 == -1.0, (
                "Large objects should have no data"
            )

    def test_area_preserved_from_data(self):
        """Test that area from data field is preserved (COCO case)."""
        gt = Detections(
            xyxy=np.array(
                [[100, 100, 200, 150]], dtype=np.float32
            ),  # Would calculate to 5000
            class_id=np.array([0]),
        )
        # Override with custom area
        gt.data = {"area": np.array([3000.0])}

        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=gt.class_id.copy(),
            confidence=np.array([0.9]),
        )
        pred.data = {"area": np.array([3000.0])}

        map_metric = MeanAveragePrecision()
        map_metric.update([pred], [gt])

        prepared_targets = map_metric._prepare_targets(map_metric._targets_list)
        used_area = prepared_targets["annotations"][0]["area"]

        assert np.allclose(used_area, 3000.0), (
            f"Should use provided area 3000.0, got {used_area}"
        )

        # Verify it's different from what would be calculated
        calculated_area = (200 - 100) * (150 - 100)  # 100 * 50 = 5000
        assert not np.allclose(used_area, calculated_area), (
            "Should use provided area, not calculated"
        )
