import numpy as np
import pytest

from supervision.config import AREA_DATA_FIELD
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils._typing import _DetectionDataType
from supervision.detection.utils.converters import mask_to_xyxy
from supervision.metrics.core import MetricTarget
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


class TestCompactMaskAreaParity:
    """Compact mask evaluation preserves dense object-size results."""

    @pytest.mark.parametrize(
        ("compact_targets", "compact_predictions"),
        [
            pytest.param(True, True, id="both-compact"),
            pytest.param(True, False, id="compact-targets"),
            pytest.param(False, True, id="compact-predictions"),
        ],
    )
    @pytest.mark.parametrize(
        ("side", "target_area"),
        [
            pytest.param(31, None, id="below-small-boundary"),
            pytest.param(32, None, id="small-boundary"),
            pytest.param(33, None, id="above-small-boundary"),
            pytest.param(95, None, id="below-large-boundary"),
            pytest.param(96, None, id="large-boundary"),
            pytest.param(97, None, id="above-large-boundary"),
            pytest.param(20, 1024.0, id="target-area-override"),
        ],
    )
    def test_size_bucket_scores_match_dense_masks(
        self,
        compact_targets: bool,
        compact_predictions: bool,
        side: int,
        target_area: float | None,
    ) -> None:
        """Preserve area buckets for boundaries, metadata, and false positives."""
        target_masks = np.zeros((1, 220, 220), dtype=bool)
        target_masks[0, :side, :side] = True
        prediction_masks = np.zeros((2, 220, 220), dtype=bool)
        prediction_masks[0, 110 : 110 + side, 110 : 110 + side] = True
        prediction_masks[1, :side, :side] = True
        target_boxes = mask_to_xyxy(target_masks)
        prediction_boxes = mask_to_xyxy(prediction_masks)
        target_data: _DetectionDataType = (
            {AREA_DATA_FIELD: np.array([target_area])}
            if target_area is not None
            else {}
        )
        targets = Detections(
            xyxy=target_boxes,
            mask=target_masks,
            class_id=np.array([0]),
            data=target_data,
        )
        predictions = Detections(
            xyxy=prediction_boxes,
            mask=prediction_masks,
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
        )
        dense = (
            MeanAveragePrecision(metric_target=MetricTarget.MASKS)
            .update(predictions, targets)
            .compute()
        )
        targets.mask = (
            CompactMask.from_dense(target_masks, target_boxes, (220, 220))
            if compact_targets
            else target_masks
        )
        predictions.mask = (
            CompactMask.from_dense(prediction_masks, prediction_boxes, (220, 220))
            if compact_predictions
            else prediction_masks
        )

        result = (
            MeanAveragePrecision(metric_target=MetricTarget.MASKS)
            .update(predictions, targets)
            .compute()
        )

        assert result.map50_95 == pytest.approx(0.5)
        assert dense.map50_95 == pytest.approx(0.5)
        assert result.small_objects is not None
        assert dense.small_objects is not None
        assert result.medium_objects is not None
        assert dense.medium_objects is not None
        assert result.large_objects is not None
        assert dense.large_objects is not None
        np.testing.assert_array_equal(
            result.small_objects.mAP_scores, dense.small_objects.mAP_scores
        )
        np.testing.assert_array_equal(
            result.medium_objects.mAP_scores, dense.medium_objects.mAP_scores
        )
        np.testing.assert_array_equal(
            result.large_objects.mAP_scores, dense.large_objects.mAP_scores
        )
