import contextlib
import io

import numpy as np
import numpy.typing as npt
import pytest

from supervision.config import ISCROWD_DATA_FIELD
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.metrics.core import AveragingMethod, MetricTarget
from supervision.metrics.f1_score import F1Score
from supervision.metrics.precision import Precision
from supervision.metrics.recall import Recall
from tests.metrics.coco_reference import evaluate_detections

_RESOLUTION = 120


def _box_masks(
    xyxy: npt.NDArray[np.float64] | npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Rasterize integer boxes into full-image masks."""
    masks = np.zeros((len(xyxy), _RESOLUTION, _RESOLUTION), dtype=bool)
    for index, (x_min, y_min, x_max, y_max) in enumerate(xyxy.astype(int)):
        masks[index, y_min:y_max, x_min:x_max] = True
    return masks


def _boundary_masks(
    boxes: npt.NDArray[np.int64], compact: bool
) -> npt.NDArray[np.bool_] | CompactMask:
    """Build dense or compact masks for the crowd overlap boundary."""
    masks = _box_masks(boxes)
    if compact:
        return CompactMask.from_dense(
            masks, boxes, image_shape=(_RESOLUTION, _RESOLUTION)
        )
    return masks


def _crowd_scene(
    metric_target: MetricTarget,
) -> tuple[Detections, Detections]:
    """One person, one crowd region, and predictions on, inside and across both."""
    target_xyxy = np.array(
        [[5, 5, 35, 45], [50, 10, 110, 70], [10, 70, 40, 100]], dtype=np.float64
    )
    prediction_xyxy = np.array(
        [
            [6, 6, 35, 44],
            [55, 15, 70, 40],
            [80, 30, 100, 60],
            [90, 20, 120, 50],
            [45, 50, 60, 80],
            [12, 72, 38, 98],
        ],
        dtype=np.float64,
    )
    with_masks = metric_target == MetricTarget.MASKS
    targets = Detections(
        xyxy=target_xyxy,
        mask=_box_masks(target_xyxy) if with_masks else None,
        class_id=np.array([0, 0, 1]),
        data={ISCROWD_DATA_FIELD: np.array([0, 1, 0])},
    )
    predictions = Detections(
        xyxy=prediction_xyxy,
        mask=_box_masks(prediction_xyxy) if with_masks else None,
        class_id=np.array([0, 0, 0, 0, 0, 1]),
        confidence=np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5]),
    )
    return predictions, targets


def _coco_precision_and_recall(
    predictions: Detections, targets: Detections, metric_target: MetricTarget
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Count TP, FP and unignored targets from pycocotools' own matching."""
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = evaluate_detections(predictions, targets, metric_target)
    true_positives = np.zeros(10)
    false_positives = np.zeros(10)
    target_count = 0
    for image in evaluator.evalImgs:
        if image is None or image["aRng"] != [0, 1e10] or image["maxDet"] != 100:
            continue
        matched = image["dtMatches"] > 0
        ignored = image["dtIgnore"].astype(bool)
        true_positives += (matched & ~ignored).sum(axis=1)
        false_positives += (~matched & ~ignored).sum(axis=1)
        target_count += int((image["gtIgnore"] == 0).sum())
    precision = true_positives / (true_positives + false_positives)
    return precision, true_positives / target_count


@pytest.mark.parametrize(
    ("metric_class", "scores_field"),
    [
        pytest.param(F1Score, "f1_scores", id="f1"),
        pytest.param(Precision, "precision_scores", id="precision"),
        pytest.param(Recall, "recall_scores", id="recall"),
    ],
)
class TestConfusionMatrixMetricResults:
    def test_mask_scores_follow_logical_boolean_values(
        self,
        metric_class: type[F1Score | Precision | Recall],
        scores_field: str,
    ) -> None:
        """Identical foreground scores perfectly even when true bytes differ."""
        boxes = np.array([[0, 0, 2, 1]], dtype=np.float32)
        class_id = np.array([0])
        predictions = Detections(
            xyxy=boxes,
            mask=np.ones((1, 1, 2), dtype=bool),
            class_id=class_id,
            confidence=np.array([0.9]),
        )
        target_mask = np.array([[[1, 255]]], dtype=np.uint8).view(np.bool_)
        targets = Detections(xyxy=boxes, mask=target_mask, class_id=class_id)

        result = (
            metric_class(metric_target=MetricTarget.MASKS)
            .update(predictions, targets)
            .compute()
        )

        np.testing.assert_array_equal(getattr(result, scores_field), np.ones(10))
        assert result.small_objects is not None
        np.testing.assert_array_equal(
            getattr(result.small_objects, scores_field), np.ones(10)
        )

    def test_bucket_thresholds_have_independent_storage(
        self,
        metric_class: type[F1Score | Precision | Recall],
        scores_field: str,
    ) -> None:
        """Changing a bucket's thresholds leaves overall and sibling results intact."""
        result = metric_class().compute()
        assert result.small_objects is not None
        assert result.medium_objects is not None
        assert result.large_objects is not None

        result.small_objects.iou_thresholds[0] = 0

        assert result.iou_thresholds[0] == 0.5
        assert result.medium_objects.iou_thresholds[0] == 0.5
        assert result.large_objects.iou_thresholds[0] == 0.5


class TestConfusionMatrixMetricIgnoreRegions:
    @pytest.mark.parametrize(
        ("metric_class", "scores_field"),
        [
            pytest.param(Recall, "recall_scores", id="recall"),
            pytest.param(F1Score, "f1_scores", id="f1"),
        ],
    )
    @pytest.mark.parametrize("ignore_field", [ISCROWD_DATA_FIELD, "ignore"])
    def test_ignore_region_is_not_counted_as_a_miss(
        self,
        metric_class: type[Recall | F1Score],
        scores_field: str,
        ignore_field: str,
    ) -> None:
        """A crowd or ignore-flagged target without a prediction is not missed."""
        targets = Detections(
            xyxy=np.array([[0, 0, 50, 50], [100, 0, 200, 100]], dtype=np.float64),
            class_id=np.array([0, 0]),
            data={ignore_field: np.array([0, 1])},
        )
        predictions = Detections(
            xyxy=np.array([[0, 0, 50, 50]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        metric = metric_class()

        result = metric.update(predictions, targets).compute()

        np.testing.assert_allclose(getattr(result, scores_field), 1.0)

    def test_crowd_region_absorbs_every_prediction_inside_it(self) -> None:
        """Several predictions inside one crowd region are all ignored."""
        targets = Detections(
            xyxy=np.array([[0, 0, 20, 20], [100, 0, 200, 100]], dtype=np.float64),
            class_id=np.array([0, 0]),
            data={ISCROWD_DATA_FIELD: np.array([0, 1])},
        )
        predictions = Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [110, 10, 130, 60], [150, 20, 190, 90]],
                dtype=np.float64,
            ),
            class_id=np.array([0, 0, 0]),
            confidence=np.array([0.9, 0.8, 0.7]),
        )

        result = Precision().update(predictions, targets).compute()

        np.testing.assert_allclose(result.precision_scores, 1.0)

    def test_class_with_only_crowd_targets_is_left_out(self) -> None:
        """A class seen only as a crowd region with predictions inside is untracked."""
        targets = Detections(
            xyxy=np.array([[0, 0, 20, 20], [100, 0, 200, 100]], dtype=np.float64),
            class_id=np.array([0, 1]),
            data={ISCROWD_DATA_FIELD: np.array([0, 1])},
        )
        predictions = Detections(
            xyxy=np.array([[0, 0, 20, 20], [110, 10, 130, 60]], dtype=np.float64),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
        )

        metric = Recall(averaging_method=AveragingMethod.MACRO)

        result = metric.update(predictions, targets).compute()

        assert result.matched_classes.tolist() == [0]

    @pytest.mark.parametrize("metric_target", [MetricTarget.BOXES, MetricTarget.MASKS])
    @pytest.mark.parametrize(
        ("metric_class", "scores_field", "reference_index"),
        [
            pytest.param(Precision, "precision_scores", 0, id="precision"),
            pytest.param(Recall, "recall_scores", 1, id="recall"),
        ],
    )
    def test_micro_scores_match_pycocotools(
        self,
        metric_target: MetricTarget,
        metric_class: type[Precision | Recall],
        scores_field: str,
        reference_index: int,
    ) -> None:
        """Micro scores equal the TP, FP and FN counts of pycocotools' matching."""
        predictions, targets = _crowd_scene(metric_target)
        expected = _coco_precision_and_recall(predictions, targets, metric_target)
        metric = metric_class(metric_target, AveragingMethod.MICRO)

        result = metric.update(predictions, targets).compute()

        np.testing.assert_allclose(
            getattr(result, scores_field), expected[reference_index]
        )

    @pytest.mark.parametrize("compact", [False, True])
    def test_crowd_mask_overlap_at_threshold_is_ignored(self, compact: bool) -> None:
        """A crowd overlap exactly at a threshold is ignored for both mask formats."""
        target_boxes = np.array([[0, 0, 10, 10], [0, 10, 10, 16]])
        prediction_boxes = np.array([[0, 0, 10, 10], [0, 10, 10, 20]])
        target_masks = _boundary_masks(target_boxes, compact)
        prediction_masks = _boundary_masks(prediction_boxes, compact)
        targets = Detections(
            xyxy=target_boxes,
            mask=target_masks,
            class_id=np.array([0, 0]),
            data={ISCROWD_DATA_FIELD: np.array([0, 1])},
        )
        predictions = Detections(
            xyxy=prediction_boxes,
            mask=prediction_masks,
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
        )

        result = Precision(MetricTarget.MASKS).update(predictions, targets).compute()

        np.testing.assert_allclose(result.precision_scores[:4], [1.0, 1.0, 1.0, 0.5])

    def test_crowd_box_preserves_large_integer_coordinates(self) -> None:
        """An enclosed narrow integer box at a large origin is ignored."""
        origin = 2**53
        boxes = np.array([[0, 0, 10, 10], [origin, 0, origin + 1, 1]], dtype=np.int64)
        targets = Detections(
            xyxy=boxes,
            class_id=np.array([0, 0]),
            data={ISCROWD_DATA_FIELD: np.array([0, 1])},
        )
        predictions = Detections(
            xyxy=boxes.copy(),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
        )

        result = Precision().update(predictions, targets).compute()

        np.testing.assert_allclose(result.precision_scores, 1.0)

    def test_ignore_and_crowd_matching_uses_confidence_order(self) -> None:
        """An ignore target does not steal a prediction that a crowd can absorb."""
        targets = Detections(
            xyxy=np.array([[30, 0, 40, 10], [0, 0, 10, 10], [5, 0, 10, 10]]),
            class_id=np.array([0, 0, 0]),
            data={
                "ignore": np.array([0, 1, 0]),
                ISCROWD_DATA_FIELD: np.array([0, 0, 1]),
            },
        )
        predictions = Detections(
            xyxy=np.array([[30, 0, 40, 10], [0, 0, 6, 10], [0, 0, 10, 10]]),
            class_id=np.array([0, 0, 0]),
            confidence=np.array([0.99, 0.9, 0.8]),
        )

        result = Precision().update(predictions, targets).compute()

        assert result.precision_scores[0] == 1.0

    def test_crowd_box_preserves_large_mixed_coordinates(self) -> None:
        """Mixed float predictions and integer crowds preserve exact overlap."""
        origin = 2**53
        targets = Detections(
            xyxy=np.array(
                [[0, 0, 10, 10], [origin + 1, 0, origin + 2, 1]], dtype=np.int64
            ),
            class_id=np.array([0, 0]),
            data={ISCROWD_DATA_FIELD: np.array([0, 1])},
        )
        predictions = Detections(
            xyxy=np.array(
                [[0, 0, 10, 10], [origin, 0, origin + 2, 1]], dtype=np.float64
            ),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
        )

        result = Precision().update(predictions, targets).compute()

        np.testing.assert_allclose(result.precision_scores[:2], [1.0, 0.5])
