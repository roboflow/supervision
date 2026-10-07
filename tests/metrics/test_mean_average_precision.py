import numpy as np
import pytest

import supervision.metrics.mean_average_precision as mean_average_precision
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget
from supervision.metrics.mean_average_precision import (
    EvaluationDataset,
    MeanAveragePrecision,
)


def _mask_detections(
    row_slice: slice, confidence: bool = False, mask_shape: tuple[int, int] = (32, 32)
) -> Detections:
    """Build single-detection `Detections` with a mask filling the given rows."""
    mask = np.zeros((1, *mask_shape), dtype=bool)
    mask[0, row_slice, :] = True
    return Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]) if confidence else None,
        mask=mask,
    )


def _obb_detections(corners: list[list[int]], confidence: bool = False) -> Detections:
    """Build single-detection `Detections` with the given oriented box corners."""
    return Detections(
        xyxy=np.array([[0, 0, 30, 30]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]) if confidence else None,
        data={ORIENTED_BOX_COORDINATES: np.array([corners], dtype=np.float32)},
    )


def _agnostic_pair(
    prediction_class_id: int | None, target_class_id: int | None
) -> tuple[Detections, Detections]:
    """Build a perfectly matching prediction and target with optional class IDs.

    Boxes, masks and oriented boxes all describe the same square, so any metric target
    can evaluate the pair. Each call returns fresh arrays, so a second call yields an
    untouched reference copy.
    """
    prediction_class_ids = (
        np.array([prediction_class_id]) if prediction_class_id is not None else None
    )
    target_class_ids = (
        np.array([target_class_id]) if target_class_id is not None else None
    )
    oriented_boxes = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32)
    predictions = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
        confidence=np.array([0.9]),
        class_id=prediction_class_ids,
        mask=np.ones((1, 10, 10), dtype=bool),
        data={ORIENTED_BOX_COORDINATES: oriented_boxes},
    )
    targets = Detections(
        xyxy=predictions.xyxy.copy(),
        class_id=target_class_ids,
        mask=predictions.mask.copy(),
        data={ORIENTED_BOX_COORDINATES: oriented_boxes.copy()},
    )
    return predictions, targets


def _square_detections(
    offsets: tuple[int, int, int],
    class_id: np.ndarray | None,
    confidence: np.ndarray | None,
) -> Detections:
    """Build one 10x10 square whose box, mask and oriented box shift independently.

    `offsets` holds the shift in pixels of the box, the mask and the oriented box, so a
    test can break only the geometry that its metric target reads.
    """
    box_offset, mask_offset, oriented_box_offset = offsets
    mask = np.zeros((1, 40, 40), dtype=bool)
    mask[0, mask_offset : mask_offset + 10, mask_offset : mask_offset + 10] = True
    corners = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32)
    return Detections(
        xyxy=np.array(
            [[box_offset, box_offset, box_offset + 10, box_offset + 10]],
            dtype=np.float64,
        ),
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        data={ORIENTED_BOX_COORDINATES: corners + oriented_box_offset},
    )


class TestMeanAveragePrecision:
    @pytest.mark.parametrize(
        ("prediction_class_id", "target_class_id", "class_mapping", "expected_class"),
        [
            pytest.param(None, 3, None, -1, id="unlabeled-predictions"),
            pytest.param(7, None, None, -1, id="unlabeled-targets"),
            pytest.param(None, None, None, 0, id="both-unlabeled"),
            pytest.param(7, 3, None, -1, id="both-labeled"),
            pytest.param(None, 3, {-1: 9}, 9, id="mapped-unlabeled-predictions"),
            pytest.param(7, None, {-1: 9}, 9, id="mapped-unlabeled-targets"),
            pytest.param(None, None, {-1: 9}, 0, id="both-unlabeled-unused-mapping"),
            pytest.param(7, 3, {-1: 9}, 9, id="mapped-labeled"),
        ],
    )
    @pytest.mark.parametrize(
        "metric_target",
        [
            pytest.param(MetricTarget.BOXES, id="boxes"),
            pytest.param(MetricTarget.MASKS, id="masks"),
            pytest.param(MetricTarget.ORIENTED_BOUNDING_BOXES, id="oriented-boxes"),
        ],
    )
    def test_class_agnostic_missing_class_ids(
        self,
        prediction_class_id: int | None,
        target_class_id: int | None,
        metric_target: MetricTarget,
        class_mapping: dict[int, int] | None,
        expected_class: int,
    ) -> None:
        """Perfect geometry scores full mAP regardless of class ID presence."""
        predictions, targets = _agnostic_pair(prediction_class_id, target_class_id)
        original_predictions, original_targets = _agnostic_pair(
            prediction_class_id, target_class_id
        )
        metric = MeanAveragePrecision(
            metric_target=metric_target,
            class_agnostic=True,
            class_mapping=class_mapping,
        )

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [expected_class])
        np.testing.assert_array_equal(
            predictions.class_id, original_predictions.class_id
        )
        np.testing.assert_array_equal(targets.class_id, original_targets.class_id)

    def test_class_agnostic_unlabeled_inputs_ignore_unused_class_mapping(self) -> None:
        """Unused class mappings do not change all-unlabeled evaluation."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            confidence=np.array([0.9]),
        )
        targets = Detections(xyxy=predictions.xyxy.copy())
        metric = MeanAveragePrecision(class_agnostic=True, class_mapping={0: 4})

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [0])

    def test_class_agnostic_normalizes_across_updates(self) -> None:
        """Labeled and unlabeled images share one class across update calls."""
        unlabeled = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            confidence=np.array([0.9]),
        )
        labeled = Detections(
            xyxy=unlabeled.xyxy.copy(),
            class_id=np.array([3]),
            confidence=np.array([0.9]),
        )
        metric = MeanAveragePrecision(class_agnostic=True)
        metric.update(unlabeled, unlabeled).update(labeled, labeled)

        result = metric.compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])

    def test_class_agnostic_state_sequence(self) -> None:
        """Repeated compute, update after compute and reset keep the right class."""
        unlabeled_predictions, unlabeled_targets = _agnostic_pair(None, None)
        labeled_predictions, labeled_targets = _agnostic_pair(3, 4)
        metric = MeanAveragePrecision(class_agnostic=True)

        unlabeled_result = metric.update(
            unlabeled_predictions, unlabeled_targets
        ).compute()
        mixed_result = metric.update(labeled_predictions, labeled_targets).compute()
        repeated_result = metric.compute()
        metric.reset()
        reset_result = metric.update(unlabeled_predictions, unlabeled_targets).compute()

        np.testing.assert_array_equal(unlabeled_result.matched_classes, [0])
        np.testing.assert_array_equal(mixed_result.matched_classes, [-1])
        assert mixed_result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(repeated_result.matched_classes, [-1])
        assert repeated_result.map50_95 == pytest.approx(mixed_result.map50_95)
        np.testing.assert_array_equal(reset_result.matched_classes, [0])
        assert unlabeled_predictions.class_id is None
        assert unlabeled_targets.class_id is None

    def test_class_agnostic_crossed_labeling_across_images(self) -> None:
        """Images that label opposite sides still share one class in one update."""
        first_predictions, first_targets = _agnostic_pair(None, 1)
        second_predictions, second_targets = _agnostic_pair(5, None)
        metric = MeanAveragePrecision(class_agnostic=True)

        result = metric.update(
            [first_predictions, second_predictions], [first_targets, second_targets]
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])

    def test_class_agnostic_crossed_labeling_with_image_indices(self) -> None:
        """Custom image indices do not change how mixed labeling is normalized."""
        first_predictions, first_targets = _agnostic_pair(None, 1)
        second_predictions, second_targets = _agnostic_pair(5, None)
        metric = MeanAveragePrecision(class_agnostic=True, image_indices=[5, 9])

        result = metric.update(
            [first_predictions, second_predictions], [first_targets, second_targets]
        ).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])

    def test_class_agnostic_empty_labeled_predictions_keep_targets_unlabeled(
        self,
    ) -> None:
        """Labeled but empty predictions do not move unlabeled targets to class -1."""
        _, targets = _agnostic_pair(None, None)
        metric = MeanAveragePrecision(class_agnostic=True)

        result = metric.update(Detections.empty(), targets).compute()

        assert result.map50_95 == pytest.approx(0.0)
        np.testing.assert_array_equal(result.matched_classes, [0])

    @pytest.mark.parametrize(
        "class_id_dtype",
        [
            pytest.param(np.uint8, id="uint8"),
            pytest.param(np.uint32, id="uint32"),
        ],
    )
    def test_class_agnostic_unsigned_class_ids(self, class_id_dtype: type) -> None:
        """Unsigned class IDs are relabeled to -1 without overflow or wrap-around."""
        predictions = _square_detections(
            (0, 0, 0), np.array([3], dtype=class_id_dtype), np.array([0.9])
        )
        targets = _square_detections(
            (0, 0, 0), np.array([4], dtype=class_id_dtype), None
        )
        metric = MeanAveragePrecision(class_agnostic=True)

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])
        assert predictions.class_id is not None
        assert predictions.class_id.dtype == class_id_dtype
        np.testing.assert_array_equal(predictions.class_id, [3])

    @pytest.mark.parametrize(
        ("prediction_class_id", "target_class_id"),
        [
            pytest.param(None, np.array([3]), id="unlabeled-predictions"),
            pytest.param(np.array([7]), None, id="unlabeled-targets"),
        ],
    )
    @pytest.mark.parametrize(
        ("metric_target", "target_offsets"),
        [
            pytest.param(MetricTarget.BOXES, (20, 0, 0), id="boxes"),
            pytest.param(MetricTarget.MASKS, (0, 20, 0), id="masks"),
            pytest.param(
                MetricTarget.ORIENTED_BOUNDING_BOXES, (0, 0, 20), id="oriented-boxes"
            ),
        ],
    )
    def test_class_agnostic_mismatched_geometry_scores_zero(
        self,
        prediction_class_id: np.ndarray | None,
        target_class_id: np.ndarray | None,
        metric_target: MetricTarget,
        target_offsets: tuple[int, int, int],
    ) -> None:
        """Mixed-label detections that miss the target geometry score zero mAP."""
        predictions = _square_detections(
            (0, 0, 0), prediction_class_id, np.array([0.9])
        )
        targets = _square_detections(target_offsets, target_class_id, None)
        metric = MeanAveragePrecision(metric_target=metric_target, class_agnostic=True)

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(0.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])

    def test_class_aware_unlabeled_predictions_stay_unmatched(self) -> None:
        """Without `class_agnostic`, unlabeled predictions stay unmatched."""
        predictions, targets = _agnostic_pair(None, 3)
        metric = MeanAveragePrecision()

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(0.0)
        np.testing.assert_array_equal(result.matched_classes, [3])

    def test_class_agnostic_multiple_detections_with_mixed_labeling(self) -> None:
        """Several detections per image merge into one class."""
        xyxy = np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float64
        )
        predictions = Detections(xyxy=xyxy, confidence=np.array([0.9, 0.8, 0.7]))
        targets = Detections(xyxy=xyxy.copy(), class_id=np.array([1, 2, 3]))
        metric = MeanAveragePrecision(class_agnostic=True)

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])

    def test_class_agnostic_mixed_labeling_without_confidence(self) -> None:
        """Predictions without confidence still merge into the labeled class."""
        predictions, targets = _agnostic_pair(None, 3)
        predictions.confidence = None
        metric = MeanAveragePrecision(class_agnostic=True)

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(1.0)
        np.testing.assert_array_equal(result.matched_classes, [-1])

    def test_single_perfect_detection(self, detections_50_50, targets_50_50):
        """Test that single perfect detection gets 1.0 mAP (not 0.0 due to ID=0 bug)"""
        metric = MeanAveragePrecision()
        metric.update([detections_50_50], [targets_50_50])
        result = metric.compute()

        # Should be perfect 1.0 mAP, not 0.0 due to ID=0 bug
        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_multiple_perfect_detections(self):
        """Test that multiple perfect detections get 1.0 mAP."""
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

    def test_perfect_non_square_oriented_boxes_get_full_map(self):
        """Perfect non-square OBB predictions score full mAP via OBB IoU."""
        obb = np.array(
            [[[10, 0], [0, 1], [30, 4], [40, 3]]],
            dtype=np.float32,
        )
        detections = Detections(
            xyxy=np.array([[0, 0, 40, 4]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            data={ORIENTED_BOX_COORDINATES: obb},
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 40, 4]], dtype=np.float64),
            class_id=np.array([0]),
            data={ORIENTED_BOX_COORDINATES: obb},
        )

        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )
        metric.update([detections], [targets])
        result = metric.compute()

        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_batch_updates_perfect_detections(self, detections_50_50, targets_50_50):
        """Test that batch updates with perfect detections get 1.0 mAP."""
        metric = MeanAveragePrecision()
        # Add 3 batch updates
        metric.update([detections_50_50], [targets_50_50])
        metric.update([detections_50_50], [targets_50_50])
        metric.update([detections_50_50], [targets_50_50])
        result = metric.compute()

        # Should be perfect 1.0 mAP across all batches
        assert abs(result.map50_95 - 1.0) < 1e-6

    def test_scenario_1_success_case_imperfect_match(self):
        """Scenario 1: Success Case with imperfect match."""
        # Small object (class 0) - area = 30*30 = 900 < 1024
        small_perfect = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object (class 1) - area = 50*50 = 2500 (between 1024 and 9216)
        medium_target = Detections(
            xyxy=np.array([[10, 10, 60, 60]], dtype=np.float64),
            class_id=np.array([1]),
            data={"area": np.array([2500])},
        )
        medium_pred = Detections(
            xyxy=np.array([[12, 12, 60, 60]], dtype=np.float64),  # Slightly off
            class_id=np.array([1]),
            confidence=np.array([0.9]),
            data={"area": np.array([2304])},  # 48*48
        )

        # Large objects (classes 0, 1, 2) - area = 100*100 = 10000 > 9216
        large_targets = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            data={"area": np.array([10000, 10000, 10000])},
        )
        large_preds = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            confidence=np.array([0.9, 0.9, 0.9]),
            data={"area": np.array([10000, 10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_perfect], [small_perfect])
        metric.update([medium_pred], [medium_target])
        metric.update([large_preds], [large_targets])
        result = metric.compute()

        # Should be close to 0.9 (slightly less than perfect due to medium object)
        assert 0.85 < result.map50_95 < 0.98  # Adjusted upper bound
        assert (
            result.medium_objects.map50_95 < 1.0
        )  # Medium should be less than perfect

    def test_scenario_2_missed_detection(self):
        """Scenario 2: GT Present, No Prediction (Missed Detection)"""
        # Small object - area = 30*30 = 900 < 1024
        small_detection = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object - area = 50*50 = 2500 (between 1024 and 9216) - missed
        medium_target = Detections(
            xyxy=np.array([[10, 10, 60, 60]], dtype=np.float64),
            class_id=np.array([1]),
            data={"area": np.array([2500])},
        )
        no_medium_pred = Detections.empty()

        # Large objects - area = 100*100 = 10000 > 9216
        large_detections = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            confidence=np.array([0.9, 0.9, 0.9]),
            data={"area": np.array([10000, 10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_detection], [small_detection])
        metric.update([no_medium_pred], [medium_target])
        metric.update([large_detections], [large_detections])
        result = metric.compute()

        # Medium objects should have 0.0 mAP (missed detection)
        assert abs(result.medium_objects.map50_95 - 0.0) < 1e-6

    def test_scenario_3_false_positive(self):
        """Scenario 3: No GT, Prediction Present (False Positive)"""
        # Small object - area = 30*30 = 900 < 1024
        small_detection = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object - area = 50*50 = 2500 - false positive (no GT)
        medium_pred = Detections(
            xyxy=np.array([[12, 12, 62, 62]], dtype=np.float64),
            class_id=np.array([1]),
            confidence=np.array([0.9]),
            data={"area": np.array([2500])},
        )
        no_medium_target = Detections.empty()

        # Large objects - area = 100*100 = 10000 > 9216
        large_detections = Detections(
            xyxy=np.array(
                [[10, 10, 110, 110], [120, 120, 220, 220], [230, 230, 330, 330]],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0, 1]),
            confidence=np.array([0.9, 0.9, 0.9]),
            data={"area": np.array([10000, 10000, 10000])},
        )

        metric = MeanAveragePrecision()
        metric.update([small_detection], [small_detection])
        metric.update([medium_pred], [no_medium_target])
        metric.update([large_detections], [large_detections])
        result = metric.compute()

        # Medium objects should have -1 mAP (false positive, matching pycocotools)
        assert result.medium_objects.map50_95 == -1

    def test_scenario_4_no_data(self):
        """Scenario 4: No GT, No Prediction (Category has no data)"""
        # Small object - area = 30*30 = 900 < 1024
        small_detection = Detections(
            xyxy=np.array([[10, 10, 40, 40]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={"area": np.array([900])},
        )

        # Medium object - no data at all
        no_medium = Detections.empty()

        # Large objects - area = 100*100 = 10000 > 9216
        # only classes 0 and 2 (no class 1)
        large_targets = Detections(
            xyxy=np.array(
                [
                    [10, 10, 110, 110],
                    [120, 120, 220, 220],
                ],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0]),
            data={"area": np.array([10000, 10000])},
        )
        large_preds = Detections(
            xyxy=np.array(
                [
                    [10, 10, 110, 110],
                    [120, 120, 220, 220],
                ],
                dtype=np.float64,
            ),
            class_id=np.array([2, 0]),
            confidence=np.array([0.9, 0.9]),
            data={"area": np.array([10000, 10000])},
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

    def test_scenario_5_only_one_class_present(self):
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

    def test_mixed_classes_with_missing_detections(
        self, detections_50_50, targets_50_50
    ):
        """Test mixed scenario with some classes having no detections."""
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
        metric.update([detections_50_50], [targets_50_50])
        metric.update([class_1_pred], [class_1_target])
        metric.update([class_2_pred], [class_2_target])
        result = metric.compute()

        # Should not have negative mAP
        assert result.map50_95 >= 0.0
        # Should be less than 1.0 due to missed detection and false positive
        assert result.map50_95 < 1.0

    def test_empty_predictions_and_targets(self):
        """Test completely empty predictions and targets."""
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


class TestMeanAveragePrecisionMasks:
    @pytest.mark.parametrize(
        ("prediction_rows", "target_rows", "expected_map50"),
        [
            pytest.param(slice(0, 16), slice(0, 16), 1.0, id="matching-masks"),
            pytest.param(slice(0, 16), slice(16, 32), 0.0, id="disjoint-masks"),
        ],
    )
    def test_map50_follows_mask_overlap(
        self, prediction_rows: slice, target_rows: slice, expected_map50: float
    ) -> None:
        """With MASKS target, map50 must reflect mask IoU, not identical boxes."""
        predictions = _mask_detections(prediction_rows, confidence=True)
        targets = _mask_detections(target_rows)
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(expected_map50, abs=1e-6)

    def test_missing_masks_raise(self) -> None:
        """With MASKS target, detections without masks must raise ValueError."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        targets = _mask_detections(slice(0, 16))
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)
        metric.update([predictions], [targets])

        with pytest.raises(ValueError, match="MASKS"):
            metric.compute()

    def test_mask_pixel_count_drives_size_buckets(self) -> None:
        """With MASKS target, object size buckets use mask area, not bbox area."""
        # bbox area is 100*100 = 10000 (large), mask area is 30*30 = 900 (small)
        mask = np.zeros((1, 120, 120), dtype=bool)
        mask[0, 10:40, 10:40] = True
        predictions = Detections(
            xyxy=np.array([[0, 0, 100, 100]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=mask,
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 100, 100]], dtype=np.float64),
            class_id=np.array([0]),
            mask=mask.copy(),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.small_objects.map50 == pytest.approx(1.0, abs=1e-6)
        assert result.large_objects.map50 == -1

    def test_boxes_target_ignores_masks(self) -> None:
        """With default BOXES target, disjoint masks must not affect the score."""
        predictions = _mask_detections(slice(0, 16), confidence=True)
        targets = _mask_detections(slice(16, 32))
        metric = MeanAveragePrecision()

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionOrientedBoundingBoxes:
    @pytest.mark.parametrize(
        ("prediction_corners", "target_corners", "expected_map50"),
        [
            pytest.param(
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                1.0,
                id="matching-obb",
            ),
            pytest.param(
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 20], [30, 20], [30, 30], [20, 30]],
                0.0,
                id="disjoint-obb",
            ),
        ],
    )
    def test_map50_follows_oriented_box_overlap(
        self,
        prediction_corners: list[list[int]],
        target_corners: list[list[int]],
        expected_map50: float,
    ) -> None:
        """With OBB target, map50 must reflect OBB IoU, not identical boxes."""
        predictions = _obb_detections(prediction_corners, confidence=True)
        targets = _obb_detections(target_corners)
        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(expected_map50, abs=1e-6)

    def test_missing_oriented_boxes_raise(self) -> None:
        """With OBB target, detections without OBB data must raise ValueError."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 30, 30]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        targets = _obb_detections([[0, 0], [10, 0], [10, 10], [0, 10]])
        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )
        metric.update([predictions], [targets])

        with pytest.raises(ValueError, match=ORIENTED_BOX_COORDINATES):
            metric.compute()

    def test_cross_matched_obb_orients_iou_correctly(self) -> None:
        """2x2 cross-match: pred0->target1, pred1->target0 must both score as TP.

        A transposed (gt, dt) matrix would yield 0 IoU for every pair; map50=0. Passing
        asserts the (dt, gt) orientation is correct end-to-end.
        """
        box_tl = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        box_br = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32)
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64),
            class_id=np.array([0, 0]),
            data={ORIENTED_BOX_COORDINATES: np.stack([box_tl, box_br])},
        )
        # Predictions deliberately swapped: pred0 matches target1, pred1 matches target0
        predictions = Detections(
            xyxy=np.array([[20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            data={ORIENTED_BOX_COORDINATES: np.stack([box_br, box_tl])},
        )
        metric = MeanAveragePrecision(
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES
        )

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionMasksCrowdBranch:
    """Tests for the crowd-aware Jaccard path in _mask_iou_with_jaccard."""

    def test_crowd_gt_ignores_contained_detection(self) -> None:
        """Detection inside a crowd GT is ignored (not FP) with Jaccard crowd IoU.

        Without Jaccard: small pred's standard IoU with crowd GT is 0.25 < 0.5,
        so pred0 is a FP, which reduces map50. With Jaccard: IoU = 1.0, pred0 is
        matched to crowd and ignored, so only pred1 (perfect TP) is scored -> map50=1.0.
        """
        mask_normal = np.zeros((1, 32, 32), dtype=bool)
        mask_normal[0, :16, :] = True  # normal GT: top half
        mask_crowd = np.ones((1, 32, 32), dtype=bool)  # crowd GT: full image

        targets = Detections(
            xyxy=np.array([[0, 0, 32, 16], [0, 0, 32, 32]], dtype=np.float64),
            class_id=np.array([0, 0]),
            mask=np.concatenate([mask_normal, mask_crowd]),
            data={"iscrowd": np.array([0, 1], dtype=np.int64)},
        )
        # pred0 (conf=0.9): bottom quarter - inside crowd, no overlap with normal GT
        mask_pred0 = np.zeros((1, 32, 32), dtype=bool)
        mask_pred0[0, 16:24, :] = True
        # pred1 (conf=0.8): exact match with normal GT
        mask_pred1 = np.zeros((1, 32, 32), dtype=bool)
        mask_pred1[0, :16, :] = True

        predictions = Detections(
            xyxy=np.array([[0, 16, 32, 24], [0, 0, 32, 16]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            mask=np.concatenate([mask_pred0, mask_pred1]),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMaskIouWithJaccard:
    """Tests for dense-mask IoU validation and temporary-buffer bounds."""

    def test_rejects_equal_area_masks_with_different_shapes(self) -> None:
        """Masks with equal pixel counts but different height and width must fail."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 6, 2]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=np.ones((1, 2, 6), dtype=bool),
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 4, 3]], dtype=np.float64),
            class_id=np.array([0]),
            mask=np.ones((1, 3, 4), dtype=bool),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        with pytest.raises(ValueError, match="spatial dimensions"):
            metric.update(predictions, targets).compute()

    def test_does_not_stack_all_ground_truth_masks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The evaluator must materialize ground truths in bounded chunks."""
        masks_true = [
            np.ones((4, 4), dtype=bool),
            np.zeros((4, 4), dtype=bool),
        ]
        masks_detection = [np.ones((4, 4), dtype=bool)]
        original_asarray = np.asarray
        ground_truth_mask_ids = {id(mask) for mask in masks_true}
        chunk_sizes: list[int] = []

        def track_ground_truth_chunks(
            masks: object, *args: object, **kwargs: object
        ) -> np.ndarray:
            """Record each materialized ground-truth chunk."""
            if (
                isinstance(masks, list)
                and masks
                and all(id(mask) in ground_truth_mask_ids for mask in masks)
            ):
                chunk_sizes.append(len(masks))
            return original_asarray(masks, *args, **kwargs)

        monkeypatch.setattr(mean_average_precision, "_MASK_IOU_GT_BUFFER_BYTES", 176)
        monkeypatch.setattr(
            mean_average_precision.np, "asarray", track_ground_truth_chunks
        )

        iou = mean_average_precision._mask_iou_with_jaccard(
            masks_true, masks_detection, [False, False]
        )

        assert chunk_sizes == [1, 1]
        np.testing.assert_allclose(iou, np.array([[1.0, 0.0]]))

    def test_crowd_iou_is_applied_in_each_ground_truth_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Crowd columns retain their special denominator across GT chunks."""
        masks_true = [
            np.array([[True, False], [False, False]]),
            np.ones((2, 2), dtype=bool),
        ]
        masks_detection = [np.array([[False, False], [False, True]])]
        monkeypatch.setattr(mean_average_precision, "_MASK_IOU_GT_BUFFER_BYTES", 56)

        iou = mean_average_precision._mask_iou_with_jaccard(
            masks_true, masks_detection, [False, True]
        )

        np.testing.assert_allclose(iou, np.array([[0.0, 1.0]]))


class TestMeanAveragePrecisionIgnoreFlag:
    """Tests for explicit target ignore flags in COCO-style evaluation."""

    def test_user_ignore_flag_excludes_target_from_scoring(self) -> None:
        """Targets marked ignored by the user must not count as normal GT."""
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0]),
            data={"ignore": np.array([1], dtype=np.int64)},
        )
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        metric = MeanAveragePrecision()

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(-1.0, abs=1e-6)

    def test_normal_gt_matched_correctly_alongside_crowd_gt(self) -> None:
        """Normal GT is matched and scored when a crowd GT is also present."""
        mask_normal = np.zeros((1, 32, 32), dtype=bool)
        mask_normal[0, :16, :] = True
        mask_crowd = np.ones((1, 32, 32), dtype=bool)

        targets = Detections(
            xyxy=np.array([[0, 0, 32, 16], [0, 0, 32, 32]], dtype=np.float64),
            class_id=np.array([0, 0]),
            mask=np.concatenate([mask_normal, mask_crowd]),
            data={"iscrowd": np.array([0, 1], dtype=np.int64)},
        )
        mask_pred = np.zeros((1, 32, 32), dtype=bool)
        mask_pred[0, :16, :] = True  # exact match with normal GT

        predictions = Detections(
            xyxy=np.array([[0, 0, 32, 16]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=mask_pred,
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestMeanAveragePrecisionMasksOrientation:
    """Tests that the (dt, gt) IoU-matrix orientation is correct end-to-end."""

    def test_cross_matched_masks_orient_iou_correctly(self) -> None:
        """2x2 cross-match: pred0->target1, pred1->target0 must both score as TP.

        A transposed (gt, dt) matrix would yield 0 IoU for every pair; map50=0. Passing
        asserts the (dt, gt) orientation is correct end-to-end.
        """
        top_mask = np.zeros((1, 32, 32), dtype=bool)
        top_mask[0, :16, :] = True
        bottom_mask = np.zeros((1, 32, 32), dtype=bool)
        bottom_mask[0, 16:, :] = True

        # target0 = top half, target1 = bottom half
        targets = Detections(
            xyxy=np.array([[0, 0, 32, 16], [0, 16, 32, 32]], dtype=np.float64),
            class_id=np.array([0, 0]),
            mask=np.concatenate([top_mask, bottom_mask]),
        )
        # Predictions deliberately swapped: pred0=bottom, pred1=top
        predictions = Detections(
            xyxy=np.array([[0, 16, 32, 32], [0, 0, 32, 16]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            mask=np.concatenate([bottom_mask, top_mask]),
        )
        metric = MeanAveragePrecision(metric_target=MetricTarget.MASKS)

        result = metric.update([predictions], [targets]).compute()

        assert result.map50 == pytest.approx(1.0, abs=1e-6)


class TestEvaluationDatasetLoadPredictions:
    """Tests for `EvaluationDataset.load_predictions` input validation."""

    @pytest.mark.parametrize(
        ("known_image_ids", "prediction_image_ids"),
        [
            pytest.param([1], [999], id="all-unknown-ids"),
            pytest.param([1, 2], [1, 999], id="mixed-known-and-unknown-ids"),
            pytest.param([], [1], id="empty-dataset-with-nonempty-predictions"),
        ],
    )
    def test_unknown_image_id_raises_value_error(
        self, known_image_ids: list[int], prediction_image_ids: list[int]
    ) -> None:
        """Predictions referencing any unknown image id raise ValueError."""
        dataset = EvaluationDataset(
            targets={
                "images": [{"id": image_id} for image_id in known_image_ids],
                "annotations": [],
                "categories": [{"id": 1}],
            }
        )
        predictions = [
            {"image_id": image_id, "category_id": 1, "bbox": [0, 0, 1, 1]}
            for image_id in prediction_image_ids
        ]

        with pytest.raises(ValueError, match="current coco set"):
            dataset.load_predictions(predictions)

    def test_predictions_subset_of_known_ids_does_not_raise(self) -> None:
        """Predictions referencing only a subset of known image ids are accepted."""
        dataset = EvaluationDataset(
            targets={
                "images": [{"id": 1}, {"id": 2}],
                "annotations": [],
                "categories": [{"id": 1}],
            }
        )
        predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}]

        result = dataset.load_predictions(predictions)

        loaded_annotations = result.get_annotations([1])
        assert len(loaded_annotations) == 1
        assert loaded_annotations[0]["image_id"] == 1
        assert loaded_annotations[0]["category_id"] == 1
        assert loaded_annotations[0]["bbox"] == [0, 0, 1, 1]


class TestMeanAveragePrecisionPycocotoolsParity:
    """Scores match pycocotools where float32 rounding would move a threshold."""

    @pytest.mark.parametrize(
        "num_objects",
        [
            pytest.param(1, id="single-object"),
            pytest.param(2, id="two-objects"),
        ],
    )
    def test_perfect_predictions_score_exactly_one(self, num_objects: int) -> None:
        """Perfect predictions score 1.0, not 1.0 minus the precision epsilon."""
        # Arrange
        xyxy = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(num_objects)])
        targets = Detections(xyxy=xyxy, class_id=np.zeros(num_objects, dtype=int))
        predictions = Detections(
            xyxy=xyxy,
            class_id=np.zeros(num_objects, dtype=int),
            confidence=np.full(num_objects, 0.9),
        )

        # Act
        result = MeanAveragePrecision().update(predictions, targets).compute()

        # Assert
        # pycocotools 2.0.11 gives 1.0 within 3e-16; a float32 epsilon gave
        # 0.99999988.
        assert result.map50_95 == 1.0
        assert result.map50 == 1.0
        assert result.map75 == 1.0

    def test_recall_landing_on_a_recall_threshold_matches_pycocotools(self) -> None:
        """Recall 0.7 of 10 targets samples precision where pycocotools does."""
        targets_xyxy = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(10)])
        false_positives_xyxy = np.array(
            [[i * 20, 100, i * 20 + 10, 110] for i in range(10)]
        )
        # Seven hits, ten misses, then an eighth hit: recall reaches exactly 0.7
        # before the precision drops.
        predictions_xyxy = np.vstack(
            [targets_xyxy[:7], false_positives_xyxy, targets_xyxy[7:8]]
        )
        targets = Detections(xyxy=targets_xyxy, class_id=np.zeros(10, dtype=int))
        predictions = Detections(
            xyxy=predictions_xyxy,
            class_id=np.zeros(len(predictions_xyxy), dtype=int),
            confidence=np.linspace(0.99, 0.5, len(predictions_xyxy)),
        )

        result = MeanAveragePrecision().update(predictions, targets).compute()

        # pycocotools 2.0.11 `COCOeval(..., "bbox")` stats[1] for the same data.
        assert result.map50 == pytest.approx(0.7414741474147416, abs=1e-6)

    @pytest.mark.parametrize(
        ("prediction_width", "expected_map50_95"),
        [(65, 0.4), (70, 0.5), (90, 0.9), (95, 1.0)],
    )
    def test_iou_landing_on_an_iou_threshold_matches_pycocotools(
        self, prediction_width: int, expected_map50_95: float
    ) -> None:
        """An IoU equal to a threshold counts as a match at it, as in pycocotools."""
        targets = Detections(xyxy=np.array([[0, 0, 100, 10]]), class_id=np.array([0]))
        predictions = Detections(
            xyxy=np.array([[0, 0, prediction_width, 10]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )

        result = MeanAveragePrecision().update(predictions, targets).compute()

        # IoU is prediction_width / 100; pycocotools 2.0.11 gives the same stats[0].
        assert result.map50_95 == pytest.approx(expected_map50_95)

    @pytest.mark.parametrize(
        ("prediction_width", "expected_map50_95"),
        [(65, 0.4), (70, 0.5), (90, 0.9), (95, 1.0)],
    )
    def test_mask_iou_landing_on_an_iou_threshold_matches_pycocotools(
        self, prediction_width: int, expected_map50_95: float
    ) -> None:
        """A mask IoU equal to a threshold matches at it, as in pycocotools."""
        target_mask = np.zeros((1, 20, 120), dtype=bool)
        target_mask[0, :10, :100] = True
        prediction_mask = np.zeros((1, 20, 120), dtype=bool)
        prediction_mask[0, :10, :prediction_width] = True
        targets = Detections(
            xyxy=np.array([[0, 0, 100, 10]]), mask=target_mask, class_id=np.array([0])
        )
        predictions = Detections(
            xyxy=np.array([[0, 0, prediction_width, 10]]),
            mask=prediction_mask,
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )

        result = (
            MeanAveragePrecision(metric_target=MetricTarget.MASKS)
            .update(predictions, targets)
            .compute()
        )

        # Mask IoU is prediction_width / 100; pycocotools 2.0.11 ("segm") gives the
        # same stats[0].
        assert result.map50_95 == pytest.approx(expected_map50_95)


def test_box_map_uses_100_max_detections_by_default() -> None:
    """Box mAP reads the 100-detection slice, so a match ranked 12th counts.

    Eleven higher-scored false positives push the only true positive past the 1 and 10
    detection limits, so AP is `1/12` only at 100 max detections.
    """
    false_positives = np.array(
        [[200 + 20 * i, 0, 210 + 20 * i, 10] for i in range(11)], dtype=float
    )
    predictions = Detections(
        xyxy=np.vstack([false_positives, [[0, 0, 50, 50]]]),
        class_id=np.zeros(12, dtype=int),
        confidence=np.r_[np.linspace(0.99, 0.9, 11), 0.5],
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=float), class_id=np.array([0])
    )

    result = MeanAveragePrecision().update(predictions, targets).compute()

    assert result.map50_95 == pytest.approx(1 / 12, abs=1e-6)
    assert result.map50 == pytest.approx(1 / 12, abs=1e-6)


def test_box_map_plot_shows_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Box mAP `plot` draws the bars and hands them to `plt.show` once."""
    from matplotlib import pyplot as plt

    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    detections = Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=float),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )
    result = MeanAveragePrecision().update(detections, detections).compute()

    result.plot()
    plt.close("all")

    assert shown == [True]
