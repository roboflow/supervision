"""Configured dense-scene limits preserve matching and reference COCO scoring."""

import contextlib
import io

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget
from supervision.metrics.mean_average_precision import (
    COCOEvaluator,
    EvaluationDataset,
    MeanAveragePrecision,
)


def _dense_pair(count: int, classes: int = 1) -> tuple[Detections, Detections]:
    """Build nonoverlapping unit boxes with decreasing scores and repeated classes."""
    x = np.arange(count, dtype=float) * 2
    boxes = np.column_stack((x, np.zeros(count), x + 1, np.ones(count)))
    labels = np.arange(count) % classes
    return (
        Detections(
            xyxy=boxes, class_id=labels, confidence=np.linspace(0.99, 0.5, count)
        ),
        Detections(xyxy=boxes.copy(), class_id=labels.copy()),
    )


def _reference(predictions: Detections, targets: Detections, cap: int) -> np.ndarray:
    """Calculate bbox AP from the selected pycocotools precision slice."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annotations = []
    for index, (box, category) in enumerate(zip(targets.xyxy, targets.class_id)):
        x0, y0, x1, y1 = box
        annotations.append(
            {
                "id": index + 1,
                "image_id": 0,
                "category_id": int(category),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            }
        )
    records = []
    for box, category, score in zip(
        predictions.xyxy, predictions.class_id, predictions.confidence
    ):
        x0, y0, x1, y1 = box
        records.append(
            {
                "image_id": 0,
                "category_id": int(category),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
            }
        )
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = COCO()
        dataset.dataset = {
            "info": {},
            "images": [{"id": 0}],
            "categories": [
                {"id": int(c), "name": str(c)} for c in np.unique(targets.class_id)
            ],
            "annotations": annotations,
        }
        dataset.createIndex()
        evaluator = COCOeval(dataset, dataset.loadRes(records), "bbox")
        evaluator.params.maxDets = [1, 10, cap]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    return np.array([p[p >= 0].mean() for p in precision])


class TestMeanAveragePrecisionLimits:
    @pytest.mark.parametrize("count", [99, 100, 101, 350, 1000])
    @pytest.mark.parametrize("cap", [100, 300, 1000])
    def test_perfect_dense_predictions_match_reference(
        self, count: int, cap: int
    ) -> None:
        """Only predictions above the configured cap lose recall."""
        predictions, targets = _dense_pair(count)
        expected = _reference(predictions, targets, cap)

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, cap))
            .update(predictions, targets)
            .compute()
        )

        np.testing.assert_allclose(result.mAP_scores, expected, atol=1e-7)

    def test_default_is_identical_to_explicit_coco_limits(self) -> None:
        """An explicit standard configuration preserves every reported score."""
        predictions, targets = _dense_pair(350)
        expected = MeanAveragePrecision().update(predictions, targets).compute()

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, 100))
            .update(predictions, targets)
            .compute()
        )

        np.testing.assert_array_equal(result.mAP_scores, expected.mAP_scores)
        assert str(result) == str(expected)
        assert result.map50 == pytest.approx(29 / 101)

    @pytest.mark.parametrize("agnostic", [False, True])
    def test_cap_is_per_category_after_class_mapping(self, agnostic: bool) -> None:
        """Merging five uncapped categories into one capped category loses recall."""
        predictions, targets = _dense_pair(350, classes=5)

        result = (
            MeanAveragePrecision(class_agnostic=agnostic)
            .update(predictions, targets)
            .compute()
        )

        assert result.map50 == pytest.approx(29 / 101 if agnostic else 1.0)

    def test_cap_is_not_shared_between_images(self) -> None:
        """Each image retains its own top predictions."""
        predictions, targets = _dense_pair(100)

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, 100))
            .update([predictions, predictions], [targets, targets])
            .compute()
        )

        assert result.map50 == pytest.approx(1.0)

    @pytest.mark.parametrize("cap", [100, 300, 1000])
    def test_false_positives_and_missing_predictions_match_reference(
        self, cap: int
    ) -> None:
        """A larger cap does not turn imperfect predictions into perfect AP."""
        predictions, targets = _dense_pair(350, classes=2)
        predictions = predictions[np.arange(350) % 7 != 0]
        predictions.xyxy[:20, 1::2] += 4
        expected = _reference(predictions, targets, cap)

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, cap))
            .update(predictions, targets)
            .compute()
        )

        np.testing.assert_allclose(result.mAP_scores, expected, atol=1e-7)
        assert 0 < result.map50 < 1

    @pytest.mark.parametrize(
        "target", [MetricTarget.MASKS, MetricTarget.ORIENTED_BOUNDING_BOXES]
    )
    def test_other_detection_targets_use_the_configured_cap(
        self, target: MetricTarget
    ) -> None:
        """Mask and oriented-box metrics retain more than 100 correct instances."""
        predictions, targets = _dense_pair(121)
        if target == MetricTarget.MASKS:
            masks = np.zeros((121, 2, 242), dtype=bool)
            masks[np.arange(121), 0, np.arange(121) * 2] = True
            predictions.mask = masks
            targets.mask = masks.copy()
        else:
            x0, y0, x1, y1 = predictions.xyxy.T
            corners = np.stack(
                (
                    np.column_stack((x0, y0)),
                    np.column_stack((x1, y0)),
                    np.column_stack((x1, y1)),
                    np.column_stack((x0, y1)),
                ),
                axis=1,
            )
            predictions.data[ORIENTED_BOX_COORDINATES] = corners
            targets.data[ORIENTED_BOX_COORDINATES] = corners.copy()

        result = (
            MeanAveragePrecision(
                metric_target=target, max_detection_thresholds=(1, 10, 300)
            )
            .update(predictions, targets)
            .compute()
        )

        assert result.map50 == pytest.approx(1.0)

    def test_results_and_exports_identify_actual_limits(self) -> None:
        """Overall and size results carry custom limits through every presentation."""
        predictions, targets = _dense_pair(350)

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, 1000))
            .update(predictions, targets)
            .compute()
        )

        assert result.max_detection_thresholds == (1, 10, 1000)
        assert "maxDets=1000" in str(result)
        assert "maxDets=100 ]" not in str(result)
        assert "maxDets=1000" in result._get_plot_details().title
        assert result.to_pandas().attrs["max_detection_thresholds"] == (1, 10, 1000)
        for child in (
            result.small_objects,
            result.medium_objects,
            result.large_objects,
        ):
            assert child.max_detection_thresholds == (1, 10, 1000)
            assert child.to_pandas().attrs["max_detection_thresholds"] == (1, 10, 1000)

    def test_internal_headline_summary_uses_custom_maximum(self) -> None:
        """The internal AP50-95 summary no longer selects a literal 100 slot."""
        predictions, targets = _dense_pair(350)
        metric = MeanAveragePrecision()
        dataset = EvaluationDataset(metric._prepare_targets([targets]))
        evaluator = COCOEvaluator(
            dataset,
            dataset.load_predictions(metric._prepare_predictions([predictions])),
        )
        evaluator.params.max_dets = [1, 10, 1000]
        evaluator.evaluate()

        evaluator._pycocotools_summarize()

        assert evaluator.stats[0] == pytest.approx(1.0)
        assert evaluator.stats[1] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "thresholds",
        [
            pytest.param((), id="empty"),
            pytest.param((1, 100), id="too-short"),
            pytest.param((1, 10, 100, 1000), id="too-long"),
            pytest.param((0, 10, 100), id="zero"),
            pytest.param((-1, 10, 100), id="negative"),
            pytest.param((1, 100, 10), id="unsorted"),
            pytest.param((1, 10, 10), id="duplicate"),
            pytest.param((True, 10, 100), id="boolean"),
            pytest.param((1, 10, 100.0), id="float"),
        ],
    )
    def test_invalid_limits_are_rejected(self, thresholds: tuple) -> None:
        """Require exactly three strictly increasing positive integer limits."""
        with pytest.raises(ValueError, match="max_detection_thresholds"):
            MeanAveragePrecision(max_detection_thresholds=thresholds)

    def test_configuration_is_copied_from_the_callers_list(self) -> None:
        """Later caller mutations do not change the metric being computed."""
        thresholds = [1, 10, 1000]
        metric = MeanAveragePrecision(max_detection_thresholds=thresholds)
        thresholds[-1] = 100
        predictions, targets = _dense_pair(350)

        result = metric.update(predictions, targets).compute()

        assert result.max_detection_thresholds == (1, 10, 1000)
        assert result.map50 == pytest.approx(1.0)

    @pytest.mark.parametrize("scale", [1, 50, 100])
    def test_size_buckets_use_custom_cap(self, scale: int) -> None:
        """All three area buckets use the same configured prediction cap."""
        predictions, targets = _dense_pair(121)
        predictions.xyxy *= scale
        targets.xyxy *= scale

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, 300))
            .update(predictions, targets)
            .compute()
        )

        children = (result.small_objects, result.medium_objects, result.large_objects)
        selected = {1: 0, 50: 1, 100: 2}[scale]
        assert children[selected].map50 == pytest.approx(1.0)
        assert children[selected].max_detection_thresholds == (1, 10, 300)

    def test_tied_scores_match_reference(self) -> None:
        """Stable score sorting agrees with COCO when tied candidates cross the cap."""
        predictions, targets = _dense_pair(121)
        predictions.confidence[:] = 0.9
        predictions.xyxy[::3, 1::2] += 4
        expected = _reference(predictions, targets, 100)

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, 100))
            .update(predictions, targets)
            .compute()
        )

        np.testing.assert_allclose(result.mAP_scores, expected, atol=1e-7)

    def test_reset_preserves_configuration(self) -> None:
        """Reset removes observations and keeps the selected evaluation limits."""
        predictions, targets = _dense_pair(121)
        metric = MeanAveragePrecision(max_detection_thresholds=(1, 10, 300))
        metric.update(predictions, targets)
        metric.reset()

        result = metric.update(predictions, targets).compute()

        assert result.map50 == pytest.approx(1.0)
        assert result.max_detection_thresholds == (1, 10, 300)
        assert len(metric._predictions_list) == 1

    def test_custom_cap_after_explicit_class_mapping(self) -> None:
        """Mapped categories share their cap after merging."""
        predictions, targets = _dense_pair(121, classes=2)

        result = (
            MeanAveragePrecision(
                class_mapping={0: 0, 1: 0}, max_detection_thresholds=(1, 10, 300)
            )
            .update(predictions, targets)
            .compute()
        )

        assert result.map50 == pytest.approx(1.0)
        assert result.matched_classes.tolist() == [0]

    @pytest.mark.parametrize("empty_targets", [False, True])
    def test_empty_predictions_retain_existing_sentinels(
        self, empty_targets: bool
    ) -> None:
        """Custom limits preserve empty prediction and target behavior."""
        _, targets = _dense_pair(121)
        if empty_targets:
            targets = Detections.empty()
        expected = MeanAveragePrecision().update(Detections.empty(), targets).compute()

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 10, 300))
            .update(Detections.empty(), targets)
            .compute()
        )

        np.testing.assert_array_equal(result.mAP_scores, expected.mAP_scores)
        assert result.max_detection_thresholds == (1, 10, 300)

    def test_small_limits_are_supported(self) -> None:
        """Custom limits need not contain COCO's literal 10 or 100 values."""
        predictions, targets = _dense_pair(3)

        result = (
            MeanAveragePrecision(max_detection_thresholds=(1, 2, 3))
            .update(predictions, targets)
            .compute()
        )

        assert result.map50 == pytest.approx(1.0)
        assert "maxDets=3" in str(result)
