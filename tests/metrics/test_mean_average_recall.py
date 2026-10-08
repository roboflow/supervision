import numpy as np
import pytest

from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.metrics import MeanAverageRecall, MetricTarget


@pytest.fixture
def complex_scenario_targets():
    """Ground truth for complex multi-image scenario.

    15 images with varying object counts and classes.
    Total: class_0=17, class_1=19 objects.
    """
    return [
        # img 0 (2 GT: c0, c1)
        np.array(
            [
                [100, 120, 260, 400, 1.0, 0],
                [500, 200, 760, 640, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 1 (3 GT: c0, c0, c1)
        np.array(
            [
                [50, 60, 180, 300, 1.0, 0],
                [210, 70, 340, 310, 1.0, 0],
                [400, 90, 620, 360, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 2 (1 GT: c1)
        np.array(
            [
                [320, 200, 540, 520, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 3 (4 GT: c0, c1, c0, c1)
        np.array(
            [
                [100, 100, 240, 340, 1.0, 0],
                [260, 110, 410, 350, 1.0, 1],
                [430, 120, 580, 360, 1.0, 0],
                [600, 130, 760, 370, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 4 (2 GT: c0, c0)
        np.array(
            [
                [120, 400, 260, 700, 1.0, 0],
                [300, 420, 480, 720, 1.0, 0],
            ],
            dtype=np.float32,
        ),
        # img 5 (3 GT: c1, c1, c1)
        np.array(
            [
                [50, 50, 200, 260, 1.0, 1],
                [230, 60, 380, 270, 1.0, 1],
                [410, 70, 560, 280, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 6 (1 GT: c0)
        np.array(
            [
                [600, 60, 780, 300, 1.0, 0],
            ],
            dtype=np.float32,
        ),
        # img 7 (5 GT: c0, c1, c1, c0, c1)
        np.array(
            [
                [60, 360, 180, 600, 1.0, 0],
                [200, 350, 340, 590, 1.0, 1],
                [360, 340, 500, 580, 1.0, 1],
                [520, 330, 660, 570, 1.0, 0],
                [680, 320, 820, 560, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 8 (2 GT: c1, c1)
        np.array(
            [
                [100, 100, 220, 300, 1.0, 1],
                [260, 110, 380, 310, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 9 (1 GT: c0)
        np.array(
            [
                [420, 400, 600, 700, 1.0, 0],
            ],
            dtype=np.float32,
        ),
        # img 10 (4 GT: c0, c1, c1, c0)
        np.array(
            [
                [50, 500, 180, 760, 1.0, 0],
                [200, 500, 350, 760, 1.0, 1],
                [370, 500, 520, 760, 1.0, 1],
                [540, 500, 690, 760, 1.0, 0],
            ],
            dtype=np.float32,
        ),
        # img 11 (2 GT: c1, c0)
        np.array(
            [
                [150, 150, 300, 420, 1.0, 1],
                [330, 160, 480, 430, 1.0, 0],
            ],
            dtype=np.float32,
        ),
        # img 12 (3 GT: c0, c1, c1)
        np.array(
            [
                [600, 200, 760, 460, 1.0, 0],
                [100, 220, 240, 480, 1.0, 1],
                [260, 230, 400, 490, 1.0, 1],
            ],
            dtype=np.float32,
        ),
        # img 13 (1 GT: c0)
        np.array(
            [
                [50, 50, 190, 250, 1.0, 0],
            ],
            dtype=np.float32,
        ),
        # img 14 (2 GT: c1, c0)
        np.array(
            [
                [420, 80, 560, 300, 1.0, 1],
                [580, 90, 730, 310, 1.0, 0],
            ],
            dtype=np.float32,
        ),
    ]


@pytest.fixture
def complex_scenario_predictions():
    """Predictions for complex multi-image scenario.

    15 images with varying detection quality:
    - True positives, false positives, false negatives
    - Class mismatches and IoU variations
    - Different confidence levels
    """
    return [
        # img 0: 2 TP + 1 class mismatch FP
        np.array(
            [
                [102, 118, 258, 398, 0.94, 0],  # TP (c0)
                [500, 200, 760, 640, 0.90, 1],  # TP (c1)
                [100, 120, 260, 400, 0.55, 1],  # FP (class mismatch)
            ],
            dtype=np.float32,
        ),
        # img 1: TPs for two c0, miss c1 (FN) + background FP
        np.array(
            [
                [50, 60, 180, 300, 0.91, 0],  # TP (c0)
                [210, 70, 340, 310, 0.88, 0],  # TP (c0)
                [600, 400, 720, 560, 0.42, 1],  # FP (no GT nearby)
            ],
            dtype=np.float32,
        ),
        # img 2: Low-IoU (miss) + random FP
        np.array(
            [
                [300, 180, 500, 430, 0.83, 1],  # Low IoU (shifted, suppose < threshold)
                [50, 50, 140, 140, 0.30, 0],  # FP
            ],
            dtype=np.float32,
        ),
        # img 3: Only match two (others FN) + one mismatch
        np.array(
            [
                [100, 100, 240, 340, 0.90, 0],  # TP (c0)
                [260, 110, 410, 350, 0.87, 1],  # TP (c1)
                [430, 120, 580, 360, 0.70, 1],  # FP (class mismatch; GT is c0)
            ],
            dtype=np.float32,
        ),
        # img 4: No predictions (2 FN)
        np.array([], dtype=np.float32).reshape(0, 6),
        # img 5: All three matched + class mismatch
        np.array(
            [
                [50, 50, 200, 260, 0.95, 1],  # TP (c1)
                [230, 60, 380, 270, 0.92, 1],  # TP (c1)
                [410, 70, 560, 280, 0.90, 1],  # TP (c1)
                [50, 50, 200, 260, 0.40, 0],  # FP (class mismatch)
            ],
            dtype=np.float32,
        ),
        # img 6: Wrong class over GT (0 recall)
        np.array(
            [
                [600, 60, 780, 300, 0.89, 1],  # FP (class mismatch)
            ],
            dtype=np.float32,
        ),
        # img 7: 3 TP, 1 miss (only 3/5 recalled)
        np.array(
            [
                [60, 360, 180, 600, 0.93, 0],  # TP (c0)
                [200, 350, 340, 590, 0.90, 1],  # TP (c1)
                [360, 340, 500, 580, 0.88, 1],  # TP (c1)
                [520, 330, 660, 570, 0.50, 1],  # FP (class mismatch; GT is c0)
            ],
            dtype=np.float32,
        ),
        # img 8: 2 TP
        np.array(
            [
                [100, 100, 220, 300, 0.96, 1],  # TP
                [262, 112, 378, 308, 0.89, 1],  # TP
            ],
            dtype=np.float32,
        ),
        # img 9: 1 TP + 1 FP
        np.array(
            [
                [418, 398, 602, 702, 0.86, 0],  # TP
                [100, 100, 140, 160, 0.33, 1],  # FP
            ],
            dtype=np.float32,
        ),
        # img 10: Perfect (all 4 TP)
        np.array(
            [
                [50, 500, 180, 760, 0.94, 0],  # TP
                [200, 500, 350, 760, 0.93, 1],  # TP
                [370, 500, 520, 760, 0.92, 1],  # TP
                [540, 500, 690, 760, 0.91, 0],  # TP
            ],
            dtype=np.float32,
        ),
        # img 11: 1 TP, 1 low IoU (FN remains) + FP
        np.array(
            [
                [150, 150, 300, 420, 0.90, 1],  # TP (c1)
                [
                    332,
                    162,
                    478,
                    428,
                    0.58,
                    0,
                ],  # TP? (slight shift) treat as TP if IoU high enough; assume OK
                [148, 148, 298, 415, 0.52, 0],  # FP (class mismatch over c1)
            ],
            dtype=np.float32,
        ),
        # img 12: 2 TP + 1 miss (one c1 missed)
        np.array(
            [
                [600, 200, 760, 460, 0.92, 0],  # TP
                [100, 220, 240, 480, 0.90, 1],  # TP
                [260, 230, 400, 490, 0.40, 0],  # FP (class mismatch; GT is c1)
            ],
            dtype=np.float32,
        ),
        # img 13: No predictions (1 FN)
        np.array([], dtype=np.float32).reshape(0, 6),
        # img 14: Class swapped (0 recall) + one correct + one FP
        np.array(
            [
                [420, 80, 560, 300, 0.88, 0],  # FP (class mismatch; GT is c1)
                [580, 90, 730, 310, 0.86, 1],  # FP (class mismatch; GT is c0)
            ],
            dtype=np.float32,
        ),
    ]


@pytest.fixture
def two_class_two_image_detections():
    """Scenario: 2 images, each with one perfect detection per class.

    The most confident class differs between the images, so an image-wide
    detection limit would keep a different class in each image.

    Returns:
        tuple: `(predictions, targets)`
            - Image 1: `class_0` (conf=0.9) > `class_1` (conf=0.8)
            - Image 2: `class_1` (conf=0.95) > `class_0` (conf=0.7)
    """
    targets = [
        Detections(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int32),
        ),
        Detections(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int32),
        ),
    ]

    predictions = [
        Detections(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32),
            confidence=np.array([0.9, 0.8], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int32),
        ),
        Detections(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32),
            confidence=np.array([0.7, 0.95], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int32),
        ),
    ]

    return predictions, targets


@pytest.fixture
def three_class_single_image_detections():
    """Scenario: 1 image with 3 classes, one perfect detection per class.

    An image-wide detection limit would count only the most confident of them,
    while COCO counts the top detection of every class.

    Returns:
        tuple: `(predictions, targets)`
            - Single image with 3 perfect detections
            - Confidences: `[0.9, 0.8, 0.7]` for classes `[0, 1, 2]`
    """
    targets = [
        Detections(
            xyxy=np.array(
                [[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]],
                dtype=np.float32,
            ),
            class_id=np.array([0, 1, 2], dtype=np.int32),
        )
    ]

    predictions = [
        Detections(
            xyxy=np.array(
                [[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            class_id=np.array([0, 1, 2], dtype=np.int32),
        )
    ]

    return predictions, targets


@pytest.mark.parametrize(
    "missing_attribute",
    ["predictions_class_id", "targets_class_id", "predictions_confidence"],
)
def test_compute_value_error_for_missing_required_fields_after_update(
    missing_attribute,
) -> None:
    """Raises ValueError when required detection fields are missing."""
    metric = MeanAverageRecall()
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    class_id = np.array([0], dtype=np.int32)
    confidence = np.array([0.9], dtype=np.float32)

    predictions = Detections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_id,
    )
    targets = Detections(
        xyxy=boxes,
        class_id=class_id,
    )

    if missing_attribute == "predictions_class_id":
        predictions = Detections(
            xyxy=boxes,
            confidence=confidence,
        )
    elif missing_attribute == "targets_class_id":
        targets = Detections(xyxy=boxes)
    else:
        predictions = Detections(
            xyxy=boxes,
            class_id=class_id,
        )

    with pytest.raises(ValueError, match="MeanAverageRecall metric requires"):
        metric.update(predictions, targets).compute()


def test_mask_content_preserves_compact_mask() -> None:
    """CompactMask inputs stay compact for mask IoU."""
    dense_mask = np.zeros((1, 4, 5), dtype=bool)
    dense_mask[0, 1:3, 1:4] = True
    xyxy = np.array([[1, 1, 4, 3]], dtype=np.float64)
    compact_mask = CompactMask.from_dense(
        dense_mask, xyxy=xyxy, image_shape=dense_mask.shape[1:]
    )
    detections = Detections(xyxy=xyxy, mask=compact_mask)
    metric = MeanAverageRecall(metric_target=MetricTarget.MASKS)

    content = metric._detections_content(detections)

    assert content is compact_mask


def test_compute_with_compact_mask_matches_dense() -> None:
    """MeanAverageRecall.compute() yields same recall_scores for CompactMask."""
    masks = np.zeros((1, 50, 50), dtype=bool)
    masks[0, 10:20, 10:20] = True
    xyxy = np.array([[10, 10, 19, 19]], dtype=np.float64)
    cm = CompactMask.from_dense(masks, xyxy, image_shape=(50, 50))
    det_dense = Detections(
        xyxy=xyxy, mask=masks, confidence=np.array([0.9]), class_id=np.array([0])
    )
    det_compact = Detections(
        xyxy=xyxy, mask=cm, confidence=np.array([0.9]), class_id=np.array([0])
    )
    metric = MeanAverageRecall(metric_target=MetricTarget.MASKS)

    r_dense = metric.update(det_dense, det_dense).compute()
    metric.reset()
    r_compact = metric.update(det_compact, det_compact).compute()

    np.testing.assert_allclose(r_dense.recall_scores, r_compact.recall_scores)


def test_single_perfect_detection() -> None:
    """Test that a single perfect detection yields 1.0 recall."""
    detections = Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)
    metric.update([detections], [detections])
    result = metric.compute()

    # For a single GT, if it's recalled, the score is 1.0 across all K
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(result.recall_scores, expected, decimal=6)


def test_recall_per_class_keeps_each_max_detection_cutoff() -> None:
    """Per-class recall must expose @1, @10 and @100 instead of only @100."""
    predictions = Detections(
        xyxy=np.array(
            [[0, 0, 10, 10], [100, 100, 110, 110], [20, 20, 30, 30]],
            dtype=np.float32,
        ),
        confidence=np.array([0.5, 0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 1, 1], dtype=np.int32),
    )
    targets = Detections(
        xyxy=np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30]],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1], dtype=np.int32),
    )
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)

    result = metric.update([predictions], [targets]).compute()

    # Class 1 ranks a false positive first, so only its top-2 contain the true
    # positive. Class 0 is recalled at every cutoff although it scores lowest.
    assert result.recall_per_class.shape == (3, 2, 10)
    np.testing.assert_allclose(result.recall_per_class[0, :, 0], [1.0, 0.0])
    np.testing.assert_allclose(result.recall_per_class[1, :, 0], [1.0, 1.0])
    np.testing.assert_allclose(result.recall_per_class[2, :, 0], [1.0, 1.0])


def test_empty_inputs_keep_max_detection_axis() -> None:
    """Empty inputs must keep mAR result shapes aligned with max detections."""
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)

    result = metric.update([Detections.empty()], [Detections.empty()]).compute()

    assert result.recall_scores.shape == result.max_detections.shape
    assert result.recall_per_class.shape == (
        result.max_detections.shape[0],
        0,
        result.iou_thresholds.shape[0],
    )
    np.testing.assert_allclose(
        result.recall_scores,
        np.zeros(result.max_detections.shape[0]),
    )
    assert result.mAR_at_1 == 0.0
    assert result.mAR_at_10 == 0.0
    assert result.mAR_at_100 == 0.0
    assert result.matched_classes.shape == (0,)


def test_medium_bucket_scores_target_matched_small_prediction() -> None:
    """Medium-object mAR keeps valid matches even if the prediction is small."""
    predictions = Detections(
        xyxy=np.array([[0, 0, 31, 31]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 32, 32]], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )

    result = (
        MeanAverageRecall(metric_target=MetricTarget.BOXES)
        .update(
            [predictions],
            [targets],
        )
        .compute()
    )

    assert result.medium_objects is not None
    assert result.medium_objects.mAR_at_1 == pytest.approx(0.9)
    assert result.medium_objects.mAR_at_10 == pytest.approx(0.9)
    assert result.medium_objects.mAR_at_100 == pytest.approx(0.9)


@pytest.mark.parametrize(
    "missing_attribute",
    ["predictions_class_id", "targets_class_id", "predictions_confidence"],
)
def test_compute_value_error_for_missing_required_fields(missing_attribute) -> None:
    """Test compute raises ValueError when required fields are missing."""
    metric = MeanAverageRecall()
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    class_id = np.array([0], dtype=np.int32)
    confidence = np.array([0.9], dtype=np.float32)

    predictions = Detections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_id,
    )
    targets = Detections(
        xyxy=boxes,
        class_id=class_id,
    )

    if missing_attribute == "predictions_class_id":
        predictions = Detections(
            xyxy=boxes,
            confidence=confidence,
        )
    elif missing_attribute == "targets_class_id":
        targets = Detections(xyxy=boxes)
    else:
        predictions = Detections(
            xyxy=boxes,
            class_id=class_id,
        )

    with pytest.raises(ValueError, match="MeanAverageRecall metric requires"):
        metric.update(predictions, targets).compute()


def test_complex_integration_scenario(
    complex_scenario_predictions, complex_scenario_targets
) -> None:
    """Test integration scenario with multiple images and varying performance."""

    def mock_detections_list(boxes_list):
        return [
            Detections(
                xyxy=boxes[:, :4],
                confidence=boxes[:, 4],
                class_id=boxes[:, 5].astype(int),
            )
            for boxes in boxes_list
        ]

    predictions_list = mock_detections_list(complex_scenario_predictions)
    targets_list = mock_detections_list(complex_scenario_targets)

    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)
    metric.update(predictions_list, targets_list)
    result = metric.compute()

    # Expected mAR at K = 1, 10, 100; equal to pycocotools AR@1, AR@10, AR@100
    # on the same data, because the limit applies per image and per class.
    expected_result = np.array([0.44845201, 0.63622291, 0.63622291])

    np.testing.assert_almost_equal(result.recall_scores, expected_result, decimal=6)


def test_mar_at_k_applies_limit_per_class_across_images(
    two_class_two_image_detections,
) -> None:
    """`mAR @ K` keeps the top detection of each class in each image, as COCO does."""
    predictions, targets = two_class_two_image_detections
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)

    result = metric.update(predictions, targets).compute()

    # Every class has one detection per image, so it is the top detection of its
    # class at K=1 even when another class scores higher in that image.
    np.testing.assert_allclose(result.recall_scores, [1.0, 1.0, 1.0])


def test_mar_at_k_applies_limit_per_class_within_one_image(
    three_class_single_image_detections,
) -> None:
    """With one detection per class in one image, `mAR @ 1` already reaches 1.0."""
    predictions, targets = three_class_single_image_detections
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)

    result = metric.update(predictions, targets).compute()

    # pycocotools reports AR@1 = 1.0 for this data; an image-wide limit would
    # keep only the 0.9 detection and give 1/3.
    np.testing.assert_allclose(result.recall_scores, [1.0, 1.0, 1.0])


def test_mar_at_k_limits_each_class_separately_in_one_image() -> None:
    """`mAR @ 1` keeps one detection per class, so classes with two targets get half."""
    boxes = np.array(
        [[0, 0, 10, 10], [20, 20, 30, 30], [100, 0, 110, 10], [120, 20, 130, 30]],
        dtype=np.float32,
    )
    class_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    targets = Detections(xyxy=boxes, class_id=class_ids)
    predictions = Detections(
        xyxy=boxes,
        confidence=np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        class_id=class_ids,
    )
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)

    result = metric.update(predictions, targets).compute()

    # pycocotools reports AR@1 = 0.5 and AR@10 = AR@100 = 1.0. An image-wide limit
    # would count only class 0's top detection and give 0.25 at K=1.
    np.testing.assert_allclose(result.recall_scores, [0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    ("sorted_class_ids", "expected_ranks"),
    [
        pytest.param([0, 1, 0, 0, 1], [0, 0, 1, 2, 1], id="interleaved-classes"),
        pytest.param([3, 3, 3], [0, 1, 2], id="single-class"),
        pytest.param([2, 0, 1], [0, 0, 0], id="all-distinct-classes"),
        pytest.param([5], [0], id="single-prediction"),
    ],
)
def test_rank_within_class_counts_predictions_of_the_same_class(
    sorted_class_ids: list[int], expected_ranks: list[int]
) -> None:
    """Each prediction is ranked only among predictions of its own class."""
    class_ids = np.array(sorted_class_ids, dtype=np.int32)

    ranks = MeanAverageRecall._rank_within_class(class_ids)

    assert ranks.tolist() == expected_ranks


def test_dataset_split_integration(yolo_dataset_two_classes) -> None:
    """Test mAR with a roboflow-format dataset loaded from disk.

    Uses a synthetic YOLO-format dataset loaded via DetectionDataset.from_yolo()
    to validate that the mAR metric works correctly with dataset splits - an
    important real-world use case.

    Scenarios tested:
    - Multiple images with varying object counts
    - Two classes with different distributions
    - Predictions with different confidence levels
    - mAR @ K correctly limits detections per class in each image
    """
    from supervision import DetectionDataset

    dataset_info = yolo_dataset_two_classes
    rng = np.random.default_rng(42)  # Match fixture seed for offset generation

    # Load dataset from YOLO format
    dataset = DetectionDataset.from_yolo(
        images_directory_path=dataset_info["images_dir"],
        annotations_directory_path=dataset_info["labels_dir"],
        data_yaml_path=dataset_info["data_yaml_path"],
    )

    assert len(dataset) == dataset_info["num_images"]
    assert dataset.classes == ["class_0", "class_1"]

    # Create predictions and targets from loaded dataset
    predictions_list = []
    targets_list = []

    for idx, (img_path, img, gt_detections) in enumerate(dataset):
        targets_list.append(gt_detections)

        # Create predictions based on GT with small offsets
        if len(gt_detections) > 0:
            pred_xyxy = gt_detections.xyxy.copy().astype(np.float32)
            # Add small random offset (±3 pixels)
            offset = rng.integers(-3, 4, pred_xyxy.shape).astype(np.float32)
            pred_xyxy = np.clip(pred_xyxy + offset, 0, 640)

            # Generate decreasing confidence scores
            num_preds = len(pred_xyxy)
            confidences = np.linspace(0.95, 0.65, num_preds, dtype=np.float32)

            predictions_list.append(
                Detections(
                    xyxy=pred_xyxy,
                    confidence=confidences,
                    class_id=gt_detections.class_id.copy(),
                )
            )
        else:
            predictions_list.append(Detections.empty())

    # Calculate mAR
    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)
    metric.update(predictions_list, targets_list)
    result = metric.compute()

    # Expected behavior validation
    expected_min_mar_at_100 = 0.8  # High recall with small offsets

    # Verify expected behavior
    assert 0.0 <= result.mAR_at_1 <= 1.0
    assert 0.0 <= result.mAR_at_10 <= 1.0
    assert 0.0 <= result.mAR_at_100 <= 1.0

    # mAR should increase with more detections considered
    assert result.mAR_at_1 <= result.mAR_at_10
    assert result.mAR_at_10 <= result.mAR_at_100

    # With good predictions (small offsets), expect high recall
    assert result.mAR_at_100 > expected_min_mar_at_100

    # mAR@1 should be lower than mAR@10 because images hold several objects of the
    # same class, and K=1 keeps only the top detection of each class
    assert result.mAR_at_1 < result.mAR_at_10


def test_mar_at_k_ignores_predictions_ranked_below_each_limit() -> None:
    """A prediction outside each top-K cutoff cannot take a target inside it."""
    targets = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        class_id=np.array([0]),
    )
    intermediate_predictions = np.tile(
        np.array([[20, 20, 30, 30]], dtype=np.float32), (9, 1)
    )
    predictions = Detections(
        xyxy=np.vstack(
            [
                np.array([[0, 0, 10, 14]], dtype=np.float32),
                intermediate_predictions,
                np.array([[0, 0, 10, 10]], dtype=np.float32),
            ]
        ),
        confidence=np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1],
            dtype=np.float32,
        ),
        class_id=np.zeros(11, dtype=int),
    )

    result = MeanAverageRecall().update(predictions, targets).compute()

    # The tighter duplicate ranks 11th, so it must not displace the top prediction
    # at K=1 or K=10. Its IoU of 100/140 matches five of ten thresholds.
    assert result.mAR_at_1 == pytest.approx(0.5)
    assert result.mAR_at_10 == pytest.approx(0.5)
    assert result.mAR_at_100 == pytest.approx(1.0)


def test_greedy_matching_two_valid_pairs():
    """Greedy matching finds both TPs; np.unique style missed the second pair.

    IoU matrix: [[1.0, 0.667], [0.333, 0.538]]. At iou>=0.5 the optimal
    assignment is T0<->P0 and T1<->P1. mAR@100 at iou=0.5 is 1.0.
    """
    preds = Detections(
        xyxy=np.array([[40, 60, 380, 470], [108, 60, 448, 470]], dtype=np.float32),
        confidence=np.array([0.95, 0.90]),
        class_id=np.array([0, 0]),
    )
    targets = Detections(
        xyxy=np.array([[40, 60, 380, 470], [210, 60, 550, 470]], dtype=np.float32),
        class_id=np.array([0, 0]),
    )

    result = MeanAverageRecall().update(preds, targets).compute()

    # At iou=0.5 both pairs match (recall=1.0); IoU(T1,P1)=0.538 < 0.55 so only
    # the first threshold has 2 TPs. mAR@100 = (1.0 + 0.5*9) / 10 = 0.55.
    # The buggy np.unique algorithm gave 0.5 (only 1 TP even at iou=0.5).
    assert result.mAR_at_100 == pytest.approx(0.55)
