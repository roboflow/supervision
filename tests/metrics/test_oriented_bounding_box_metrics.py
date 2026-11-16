import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget
from supervision.metrics.f1_score import F1Score
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from supervision.metrics.mean_average_recall import MeanAverageRecall
from supervision.metrics.precision import Precision
from supervision.metrics.recall import Recall


def _non_square_obb_detections(confidence: bool = False) -> Detections:
    obb = np.array(
        [[[10, 0], [0, 1], [30, 4], [40, 3]]],
        dtype=np.float32,
    )
    return Detections(
        xyxy=np.array([[0, 0, 40, 4]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]) if confidence else None,
        data={ORIENTED_BOX_COORDINATES: obb},
    )


@pytest.mark.parametrize(
    ("cls_metric", "score_name"),
    [
        (Precision, "precision_at_50"),
        (Recall, "recall_at_50"),
        (F1Score, "f1_50"),
        (MeanAverageRecall, "mAR_at_100"),
    ],
)
def test_perfect_non_square_oriented_boxes_score_as_perfect(
    cls_metric: type,
    score_name: str,
) -> None:
    """Perfect non-square OBB predictions score 1.0 for metrics that use OBB IoU."""
    predictions = _non_square_obb_detections(confidence=True)
    targets = _non_square_obb_detections()

    metric = cls_metric(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
    result = metric.update([predictions], [targets]).compute()

    assert getattr(result, score_name) == pytest.approx(1.0)


def test_mean_average_precision_accepts_obb_metric_target() -> None:
    """MeanAveragePrecision routes metric_target=ORIENTED_BOUNDING_BOXES through
    oriented_box_iou_batch; perfect OBB predictions score 1.0."""
    predictions = _non_square_obb_detections(confidence=True)
    targets = _non_square_obb_detections()

    metric = MeanAveragePrecision(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
    result = metric.update([predictions], [targets]).compute()

    assert result.map50_95 == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric_class", "score_name"),
    [
        pytest.param(Precision, "precision_at_50", id="precision"),
        pytest.param(Recall, "recall_at_50", id="recall"),
        pytest.param(F1Score, "f1_50", id="f1-score"),
        pytest.param(MeanAverageRecall, "mAR_at_100", id="mean-average-recall"),
    ],
)
class TestOrientedBoxMetricsPrecision:
    """Corner precision survives matching and object-size filtering."""

    @pytest.mark.parametrize(
        ("dtype", "origin", "size", "size_name"),
        [
            pytest.param(
                np.float64, 10**8, 50, "medium_objects", id="float64-stitched-origin"
            ),
            pytest.param(
                np.float64, 10**10, 50, "medium_objects", id="float64-area-cancellation"
            ),
            pytest.param(
                np.float16, -40000, 80000, "large_objects", id="float16-large-extent"
            ),
            pytest.param(
                ">f2", -40000, 80000, "large_objects", id="float16-non-native-endian"
            ),
            pytest.param(np.int64, 2**60, 50, "medium_objects", id="int64-origin"),
            pytest.param(np.uint64, 2**60, 50, "medium_objects", id="uint64-origin"),
        ],
    )
    def test_perfect_box_keeps_its_score_and_size(
        self,
        metric_class: type,
        score_name: str,
        dtype: type | str,
        origin: int,
        size: int,
        size_name: str,
    ) -> None:
        """Translated or wide rectangles retain their score and size category."""
        corners: np.ndarray = np.array(
            [
                [
                    [origin, origin],
                    [origin + size, origin],
                    [origin + size, origin + size],
                    [origin, origin + size],
                ]
            ],
            dtype=dtype,
        )
        boxes: np.ndarray = np.array(
            [[origin, origin, origin + size, origin + size]], dtype=dtype
        )
        predictions = Detections(
            xyxy=boxes,
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )
        targets = Detections(
            xyxy=boxes.copy(),
            class_id=np.array([0]),
            data={ORIENTED_BOX_COORDINATES: corners.copy()},
        )

        result = (
            metric_class(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
            .update(predictions, targets)
            .compute()
        )

        assert getattr(result, score_name) == pytest.approx(1.0)
        size_result = getattr(result, size_name)
        assert size_result is not None
        assert getattr(size_result, score_name) == pytest.approx(1.0)
        np.testing.assert_array_equal(
            predictions.data[ORIENTED_BOX_COORDINATES], corners
        )

    def test_subpixel_overlap_at_mosaic_origin_is_not_a_match(
        self, metric_class: type, score_name: str
    ) -> None:
        """A 40%-shifted small rectangle has IoU below one half at a 64k origin."""
        origin, width = 64000.0, 0.01
        corners = np.array([[[0, 0], [width, 0], [width, width], [0, width]]])
        corners += origin
        boxes = np.array([[origin, origin, origin + width, origin + width]])
        predictions = Detections(
            xyxy=boxes + np.array([width * 0.4, 0, width * 0.4, 0]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            data={ORIENTED_BOX_COORDINATES: corners + np.array([width * 0.4, 0])},
        )
        targets = Detections(
            xyxy=boxes,
            class_id=np.array([0]),
            data={ORIENTED_BOX_COORDINATES: corners},
        )

        result = (
            metric_class(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
            .update(predictions, targets)
            .compute()
        )

        assert getattr(result, score_name) == 0.0


def _axis_aligned_detections(
    xyxy: list[float], confidence: float | None = None
) -> Detections:
    """Build one detection whose oriented box is its axis-aligned box.

    Examples:
        ```pycon
        >>> detections = _axis_aligned_detections([0, 0, 10, 10])
        >>> detections.data[ORIENTED_BOX_COORDINATES].tolist()
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]

        ```
    """
    x_min, y_min, x_max, y_max = xyxy
    corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return Detections(
        xyxy=np.array([xyxy], dtype=np.float64),
        class_id=np.array([0]),
        confidence=None if confidence is None else np.array([confidence]),
        data={ORIENTED_BOX_COORDINATES: np.array([corners], dtype=np.float64)},
    )


@pytest.mark.parametrize(
    ("cls_metric", "scores_name"),
    [
        pytest.param(Precision, "precision_scores", id="precision"),
        pytest.param(Recall, "recall_scores", id="recall"),
        pytest.param(F1Score, "f1_scores", id="f1-score"),
        pytest.param(MeanAverageRecall, "recall_per_class", id="mean-average-recall"),
    ],
)
@pytest.mark.parametrize(
    ("threshold", "prediction_box", "target_box"),
    [
        pytest.param(0.55, [0, 0, 11, 20], [0, 0, 20, 20], id="iou-0-55"),
        pytest.param(0.6, [0, 1, 14, 11], [0, 0, 10, 10], id="iou-0-6"),
        pytest.param(0.8, [0, 0, 16, 20], [0, 0, 20, 20], id="iou-0-8"),
        pytest.param(0.85, [0, 0, 17, 20], [0, 0, 20, 20], id="iou-0-85"),
    ],
)
def test_oriented_box_iou_on_a_threshold_matches_like_axis_aligned_box(
    cls_metric: type,
    scores_name: str,
    threshold: float,
    prediction_box: list[float],
    target_box: list[float],
) -> None:
    """Exact threshold IoUs produce matching OBB and box metric scores."""
    predictions = _axis_aligned_detections(prediction_box, confidence=0.9)
    targets = _axis_aligned_detections(target_box)

    boxes_result = (
        cls_metric(metric_target=MetricTarget.BOXES)
        .update([predictions], [targets])
        .compute()
    )
    obb_result = (
        cls_metric(metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES)
        .update([predictions], [targets])
        .compute()
    )

    np.testing.assert_array_equal(
        getattr(obb_result, scores_name), getattr(boxes_result, scores_name)
    )
    threshold_index = np.flatnonzero(
        np.isclose(boxes_result.iou_thresholds, threshold)
    )[0]
    assert getattr(obb_result, scores_name)[..., threshold_index] == pytest.approx(1.0)
