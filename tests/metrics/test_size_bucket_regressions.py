import numpy as np
import pytest

from supervision.config import AREA_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics import (
    F1Score,
    MeanAverageRecall,
    MetricTarget,
    Precision,
    Recall,
)


@pytest.mark.parametrize(
    ("metric_cls", "bucket_attr", "score_attrs", "expected"),
    [
        pytest.param(
            Precision,
            "medium_objects",
            ("precision_at_50", "precision_at_75"),
            (1.0, 1.0),
            id="precision",
        ),
        pytest.param(
            Recall,
            "medium_objects",
            ("recall_at_50", "recall_at_75"),
            (1.0, 1.0),
            id="recall",
        ),
        pytest.param(
            F1Score,
            "medium_objects",
            ("f1_50", "f1_75"),
            (1.0, 1.0),
            id="f1",
        ),
        pytest.param(
            MeanAverageRecall,
            "medium_objects",
            ("mAR_at_1", "mAR_at_10", "mAR_at_100"),
            (0.6, 0.6, 0.6),
            id="mar",
        ),
    ],
)
def test_size_bucket_match_is_not_stolen(
    metric_cls, bucket_attr, score_attrs, expected
):
    """Bucketed metrics must keep the in-bucket match instead of stealing it."""
    predictions = Detections(
        xyxy=np.array([[0, 0, 90, 90]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=np.array(
            [[0, 0, 80, 80], [0, 0, 100, 100]],
            dtype=np.float32,
        ),
        class_id=np.array([0, 0], dtype=np.int32),
    )

    result = (
        metric_cls(metric_target=MetricTarget.BOXES)
        .update(predictions, targets)
        .compute()
    )

    bucket_result = getattr(result, bucket_attr)
    assert bucket_result is not None
    for score_attr, score_expected in zip(score_attrs, expected, strict=True):
        assert getattr(bucket_result, score_attr) == pytest.approx(score_expected)


def test_small_bucket_mar_returns_zero_without_bucket_targets() -> None:
    """Bucketed mAR must return zeros instead of NaN when there is no support."""
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
        .update(predictions, targets)
        .compute()
    )

    assert result.small_objects is not None
    np.testing.assert_allclose(result.small_objects.recall_scores, np.zeros(3))
    assert result.small_objects.mAR_at_1 == 0.0
    assert result.small_objects.mAR_at_10 == 0.0
    assert result.small_objects.mAR_at_100 == 0.0


def test_medium_bucket_mar_counts_global_rank_budget() -> None:
    """Bucketed mAR must count out-of-bucket predictions against top-K."""
    predictions = Detections(
        xyxy=np.array(
            [[0, 0, 150, 150], [0, 0, 80, 80]],
            dtype=np.float32,
        ),
        confidence=np.array([0.95, 0.90], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 80, 80]], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )

    result = (
        MeanAverageRecall(metric_target=MetricTarget.BOXES)
        .update(predictions, targets)
        .compute()
    )

    assert result.medium_objects is not None
    assert result.medium_objects.mAR_at_1 == 0.0
    assert result.medium_objects.mAR_at_10 == 1.0
    assert result.medium_objects.mAR_at_100 == 1.0


@pytest.mark.parametrize(
    ("metric_cls", "bucket_attrs", "score_attrs"),
    [
        pytest.param(
            Precision,
            ("medium_objects", "large_objects"),
            ("precision_at_50", "precision_at_75"),
            id="precision",
        ),
        pytest.param(
            Recall,
            ("medium_objects", "large_objects"),
            ("recall_at_50", "recall_at_75"),
            id="recall",
        ),
        pytest.param(
            F1Score,
            ("medium_objects", "large_objects"),
            ("f1_50", "f1_75"),
            id="f1",
        ),
        pytest.param(
            MeanAverageRecall,
            ("medium_objects", "large_objects"),
            ("mAR_at_10", "mAR_at_100"),
            id="mar",
        ),
    ],
)
def test_perfect_detector_scores_full_marks_in_every_bucket(
    metric_cls, bucket_attrs, score_attrs
):
    """Bucketed metrics must score a perfect detector 1.0 in every bucket."""
    xyxy = np.array(
        [[0, 0, 50, 50], [100, 100, 250, 250]],
        dtype=np.float32,
    )
    predictions = Detections(
        xyxy=xyxy.copy(),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=xyxy.copy(),
        class_id=np.array([0, 0], dtype=np.int32),
    )

    result = (
        metric_cls(metric_target=MetricTarget.BOXES)
        .update(predictions, targets)
        .compute()
    )

    for bucket_attr in bucket_attrs:
        bucket_result = getattr(result, bucket_attr)
        assert bucket_result is not None
        for score_attr in score_attrs:
            assert getattr(bucket_result, score_attr) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric_cls", "score_attr", "overall_expected"),
    [
        pytest.param(Precision, "precision_at_50", 0.5, id="precision"),
        pytest.param(F1Score, "f1_50", 2 / 3, id="f1"),
    ],
)
def test_unmatched_out_of_bucket_prediction_does_not_penalize_bucket(
    metric_cls, score_attr, overall_expected
):
    """A stray large false positive lowers overall scores, not the medium bucket."""
    predictions = Detections(
        xyxy=np.array(
            [[0, 0, 50, 50], [200, 200, 350, 350]],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )

    result = (
        metric_cls(metric_target=MetricTarget.BOXES)
        .update(predictions, targets)
        .compute()
    )

    assert getattr(result, score_attr) == pytest.approx(overall_expected)
    assert result.medium_objects is not None
    assert getattr(result.medium_objects, score_attr) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric_cls", "score_attr"),
    [
        pytest.param(Precision, "precision_at_50", id="precision"),
        pytest.param(Recall, "recall_at_50", id="recall"),
        pytest.param(F1Score, "f1_50", id="f1"),
    ],
)
def test_explicit_area_metadata_controls_bucket_assignment(
    metric_cls: type, score_attr: str
) -> None:
    """Explicit area metadata controls bucket assignment for scoring metrics."""
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    predictions = Detections(
        xyxy=box.copy(),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=box.copy(),
        class_id=np.array([0], dtype=np.int32),
        data={AREA_DATA_FIELD: np.array([2500.0], dtype=np.float32)},
    )

    result = (
        metric_cls(metric_target=MetricTarget.BOXES)
        .update(predictions, targets)
        .compute()
    )

    assert result.small_objects is not None
    assert result.medium_objects is not None
    assert getattr(result.small_objects, score_attr) == pytest.approx(0.0)
    assert getattr(result.medium_objects, score_attr) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric_cls", "score_attr"),
    [
        pytest.param(Precision, "precision_at_50", id="precision"),
        pytest.param(Recall, "recall_at_50", id="recall"),
        pytest.param(F1Score, "f1_50", id="f1"),
    ],
)
def test_mask_bucket_ignores_out_of_bucket_predictions(
    metric_cls: type, score_attr: str
) -> None:
    """Mask bucket scoring ignores predictions outside the requested bucket."""
    masks = np.zeros((2, 128, 128), dtype=bool)
    masks[0, :50, :50] = True
    masks[1, 28:, 28:] = True
    target_mask = masks[0:1].copy()
    predictions = Detections(
        xyxy=np.array([[0, 0, 50, 50], [28, 28, 128, 128]], dtype=np.float32),
        mask=masks,
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
        mask=target_mask,
        class_id=np.array([0], dtype=np.int32),
    )

    result = (
        metric_cls(metric_target=MetricTarget.MASKS)
        .update(predictions, targets)
        .compute()
    )

    assert result.medium_objects is not None
    assert getattr(result.medium_objects, score_attr) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric_cls", "score_attr"),
    [
        pytest.param(Precision, "precision_at_50", id="precision"),
        pytest.param(Recall, "recall_at_50", id="recall"),
        pytest.param(F1Score, "f1_50", id="f1"),
    ],
)
def test_obb_bucket_ignores_out_of_bucket_predictions(
    metric_cls: type, score_attr: str
) -> None:
    """OBB bucket scoring ignores predictions outside the requested bucket."""
    target_obb = np.array([[[0, 0], [50, 0], [50, 50], [0, 50]]], dtype=np.float32)
    prediction_obb = np.concatenate(
        [
            target_obb,
            np.array(
                [[[28, 28], [128, 28], [128, 128], [28, 128]]],
                dtype=np.float32,
            ),
        ]
    )
    predictions = Detections(
        xyxy=np.array([[0, 0, 50, 50], [28, 28, 128, 128]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
        data={ORIENTED_BOX_COORDINATES: prediction_obb},
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        data={ORIENTED_BOX_COORDINATES: target_obb},
    )

    result = (
        metric_cls(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
        .update(predictions, targets)
        .compute()
    )

    assert result.medium_objects is not None
    assert getattr(result.medium_objects, score_attr) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric_cls", "missing_side"),
    [
        pytest.param(Precision, "predictions", id="precision-predictions"),
        pytest.param(Precision, "targets", id="precision-targets"),
        pytest.param(Recall, "predictions", id="recall-predictions"),
        pytest.param(Recall, "targets", id="recall-targets"),
        pytest.param(F1Score, "predictions", id="f1-predictions"),
        pytest.param(F1Score, "targets", id="f1-targets"),
        pytest.param(
            MeanAverageRecall,
            "predictions",
            id="mar-predictions",
        ),
        pytest.param(MeanAverageRecall, "targets", id="mar-targets"),
    ],
)
def test_mask_target_requires_masks(metric_cls, missing_side) -> None:
    """Mask-target metrics raise when either side omits masks."""
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 2:8, 2:8] = True

    masked_predictions = Detections(
        xyxy=box,
        mask=mask,
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    masked_targets = Detections(
        xyxy=box,
        mask=mask,
        class_id=np.array([0], dtype=np.int32),
    )

    predictions = masked_predictions
    targets = masked_targets
    if missing_side == "predictions":
        predictions = Detections(
            xyxy=box,
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=np.int32),
        )
    else:
        targets = Detections(xyxy=box, class_id=np.array([0], dtype=np.int32))

    metric = metric_cls(metric_target=MetricTarget.MASKS)

    with pytest.raises(ValueError, match="requires detections to include masks"):
        metric.update(predictions, targets).compute()
