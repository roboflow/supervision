import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics import MeanAverageRecall, MetricTarget


@pytest.fixture
def complex_scenario_targets():
    """
    Ground truth for complex multi-image scenario.

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
    """
    Predictions for complex multi-image scenario.

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
    """
    Scenario: 2 images with 2 classes with varying confidence levels.

    Tests that `mAR @ K` limits per image (not per class) by creating a case where
    the highest confidence detection differs between images.

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
    """
    Scenario: 1 image with 3 classes - explicit bug reproduction.

    Demonstrates the N x K vs K issue: with 3 classes, the bug would allow
    3 detections for `mAR @ 1` (one per class) instead of just 1.

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


def test_single_perfect_detection():
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


def test_complex_integration_scenario(
    complex_scenario_predictions, complex_scenario_targets
):
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

    # Expected mAR at K = 1, 10, 100
    expected_result = np.array([0.2874613, 0.63622291, 0.63622291])

    np.testing.assert_almost_equal(result.recall_scores, expected_result, decimal=6)


def test_mar_at_k_limits_per_image_not_per_class(two_class_two_image_detections):
    """
    Test that `mAR @ K` limits detections per image, not per class.

    BUG SCENARIO (what was wrong):
    The previous implementation would limit detections per CLASS per image,
    meaning `mAR@1` would take the top-1 prediction for EACH class in each image.
    With 2 classes and `mAR@1`, this incorrectly allowed 2 detections per image.

    This test uses a scenario where the bug would produce different results:
    - 2 images, each with 2 GT objects (one of each class)
    - Predictions perfectly match GT with varying confidences
    - Image 1: `class_0` (conf=0.9) > `class_1` (conf=0.8)
    - Image 2: `class_1` (conf=0.95) > `class_0` (conf=0.7)

    BUGGY BEHAVIOR (if bug were present):
    - `mAR@1` would take top-1 per class → both detections per image count
    - Recall for `class_0`: 2/2 = 1.0
    - Recall for `class_1`: 2/2 = 1.0
    - `mAR@1` would incorrectly = 1.0 (same as `mAR@10`)

    CORRECT BEHAVIOR (with fix):
    - `mAR@1` takes top-1 per image → only highest confidence per image counts
    - Image 1: only `class_0` counts (conf=0.9)
    - Image 2: only `class_1` counts (conf=0.95)
    - Recall for `class_0`: 1/2 = 0.5
    - Recall for `class_1`: 1/2 = 0.5
    - `mAR@1` = 0.5 (correctly < `mAR@10` = 1.0)
    """
    predictions, targets = two_class_two_image_detections

    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)
    metric.update(predictions, targets)
    result = metric.compute()

    # Expected results with correct behavior
    expected_mar_at_1 = 0.5  # Only top detection per image
    expected_mar_at_10 = 1.0  # All detections count
    expected_mar_at_100 = 1.0
    # Note: Bug would produce mAR @ 1 = 1.0

    # Test correct behavior (this would fail with the bug)
    np.testing.assert_almost_equal(result.mAR_at_1, expected_mar_at_1, decimal=6)
    np.testing.assert_almost_equal(result.mAR_at_10, expected_mar_at_10, decimal=6)
    np.testing.assert_almost_equal(result.mAR_at_100, expected_mar_at_100, decimal=6)

    # Critical assertion: mAR @ 1 must be less than mAR @ 10
    # With the bug, both would equal 1.0
    assert result.mAR_at_1 < result.mAR_at_10, (
        f"Bug detected: mAR @ 1 ({result.mAR_at_1}) should be < mAR @ 10 "
        f"({result.mAR_at_10}) when images have multiple objects. "
        "If they're equal, K is being applied per-class instead of per-image."
    )


def test_three_class_single_image_scenario(three_class_single_image_detections):
    """
    Test with 3 classes on single image - explicit N x K bug reproduction.

    THE BUG:
    mAR @ K was limiting detections per class per image, not per image globally.
    This meant with N classes, up to N x K detections could count per image
    instead of just K detections.

    REPRODUCTION SCENARIO:
    Image with 3 GT objects: `[class_0, class_1, class_2]`
    Model predicts all 3 correctly with confidences: `[0.9, 0.8, 0.7]`

    With mAR @ 1 (max 1 detection per image):

    BUGGY: Would take top-1 per class → all 3 detections count
    → Recall per class: `[1/1, 1/1, 1/1]` → mAR @ 1 = 1.0

    CORRECT: Takes top-1 globally → only `class_0` (conf=0.9) counts
    → Recall per class: `[1/1, 0/1, 0/1]` → mAR @ 1 = 0.33

    This test would PASS with the bug (incorrectly) if mAR @ 1 ≈ 1.0
    and PASS with the fix (correctly) if mAR @ 1 ≈ 0.33
    """
    predictions, targets = three_class_single_image_detections

    metric = MeanAverageRecall(metric_target=MetricTarget.BOXES)
    metric.update(predictions, targets)
    result = metric.compute()

    # Expected results with correct behavior
    expected_mar_at_1 = 1.0 / 3.0  # Only highest confidence (class_0) counts
    expected_mar_at_10 = 1.0  # All detections count
    # Note: Bug would produce mAR @ 1 = 1.0 (all 3 counted, one per class)

    # Test correct behavior
    np.testing.assert_almost_equal(result.mAR_at_1, expected_mar_at_1, decimal=6)
    np.testing.assert_almost_equal(result.mAR_at_10, expected_mar_at_10, decimal=6)

    # Sanity check: if this fails, the bug is present
    # Bug would produce mAR @ 1 ≈ 1.0, correct is ≈ 0.333
    assert result.mAR_at_1 < 0.5, (
        f"Bug detected: mAR @ 1 = {result.mAR_at_1:.4f}, expected ≈ 0.333. "
        "The bug would produce mAR @ 1 ≈ 1.0 by counting all detections."
    )


def test_dataset_split_integration(yolo_dataset_two_classes):
    """
    Test mAR with a roboflow-format dataset loaded from disk.

    Uses a synthetic YOLO-format dataset loaded via DetectionDataset.from_yolo()
    to validate that the mAR metric works correctly with dataset splits - an
    important real-world use case.

    Scenarios tested:
    - Multiple images with varying object counts
    - Two classes with different distributions
    - Predictions with different confidence levels
    - mAR @ K correctly limits per image (not per class)
    """
    from supervision import DetectionDataset

    dataset_info = yolo_dataset_two_classes
    np.random.seed(42)  # Match fixture seed for offset generation

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
            offset = np.random.randint(-3, 4, pred_xyxy.shape).astype(np.float32)
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

    # mAR@1 should be significantly lower than mAR@10 for multi-object images
    # This validates that K limits detections per image (not per class)
    assert result.mAR_at_1 < result.mAR_at_10
