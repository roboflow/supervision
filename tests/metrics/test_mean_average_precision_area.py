import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class TestMeanAveragePrecisionArea:
    """Test area calculation in MeanAveragePrecision."""

    @pytest.mark.parametrize(
        ("xyxy", "expected_areas", "expected_size_maps"),
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
        self,
        xyxy: np.ndarray,
        expected_areas: list[float],
        expected_size_maps: dict[str, bool],
    ) -> None:
        """Test area calculation and size-specific mAP functionality."""
        class_id = np.arange(len(xyxy))
        gt = Detections(xyxy=xyxy, class_id=class_id)
        pred = Detections(
            xyxy=gt.xyxy.copy(),
            class_id=class_id.copy(),
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
        assert result.small_objects is not None
        assert result.medium_objects is not None
        assert result.large_objects is not None

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

    def test_area_preserved_from_data(self) -> None:
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
            class_id=np.array([0]),
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


class TestMeanAveragePrecisionCoordinateDtypes:
    """Coordinate storage dtypes must not change box evaluation results."""

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param(
                np.array([[0, 0, 512, 512]], dtype=np.float16), id="float16-area"
            ),
            pytest.param(
                np.array([[0, 0, 512, 512]], dtype=np.uint16), id="uint16-area"
            ),
            pytest.param(
                np.array([[0, 0, 50000, 50000]], dtype=np.int32), id="int32-area"
            ),
            pytest.param(
                np.array([[-20000, -20000, 20000, 20000]], dtype=np.int16),
                id="int16-extent",
            ),
            pytest.param(
                np.array([[-40000, -40000, 40000, 40000]], dtype=np.float16),
                id="float16-extent",
            ),
            pytest.param(
                np.array([[0, 0, 512, 512]], dtype=np.float32), id="float32-control"
            ),
            pytest.param(
                np.array([[0, 0, 512, 512]], dtype=np.float64), id="float64-control"
            ),
            pytest.param(
                np.array([[0, 0, 512, 512]], dtype=np.int64), id="int64-control"
            ),
        ],
    )
    def test_perfect_large_box_keeps_its_area_and_score(self, xyxy: np.ndarray) -> None:
        """Perfect boxes retain their large-object score across dtypes."""
        target = Detections(xyxy=xyxy, class_id=np.array([0]))
        prediction = Detections(
            xyxy=xyxy.copy(), class_id=np.array([0]), confidence=np.array([0.9])
        )
        original_xyxy = xyxy.copy()
        metric = MeanAveragePrecision().update(prediction, target)

        result = metric.compute()

        assert result.small_objects is not None
        assert result.medium_objects is not None
        assert result.large_objects is not None
        assert result.map50_95 == pytest.approx(1.0)
        assert result.large_objects.map50_95 == pytest.approx(1.0)
        assert result.small_objects.map50_95 == -1.0
        assert result.medium_objects.map50_95 == -1.0
        np.testing.assert_array_equal(target.xyxy, original_xyxy)
        np.testing.assert_array_equal(prediction.xyxy, original_xyxy)
        assert target.xyxy.dtype == xyxy.dtype
        assert prediction.xyxy.dtype == xyxy.dtype

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.float16, id="float16"),
            pytest.param(np.uint16, id="uint16"),
            pytest.param(np.float32, id="float32-control"),
        ],
    )
    def test_large_false_positive_is_ignored_only_in_small_bucket(
        self, dtype: type[np.generic]
    ) -> None:
        """A large false positive counts overall but not in the small bucket."""
        target = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=dtype), class_id=np.array([0])
        )
        predictions = Detections(
            xyxy=np.array([[100, 100, 612, 612], [0, 0, 10, 10]], dtype=dtype),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
        )
        metric = MeanAveragePrecision().update(predictions, target)

        result = metric.compute()

        assert result.small_objects is not None
        assert result.map50_95 == pytest.approx(0.5)
        assert result.small_objects.map50_95 == pytest.approx(1.0)
