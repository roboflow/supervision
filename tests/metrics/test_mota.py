import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.metrics.mota import (
    MOTAResult,
    MOTMetricsResult,
    MOTPResult,
    MultiObjectTrackingAccuracy,
    MultiObjectTrackingPrecision,
    TrackingMetrics,
)


def _make_detections(xyxy, tracker_ids):
    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


def _empty_detections():
    return Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        tracker_id=np.array([], dtype=int),
    )


class TestMOTAInit:
    def test_default_threshold(self):
        metric = MultiObjectTrackingAccuracy()
        assert metric.iou_threshold == 0.5

    def test_custom_threshold(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.75)
        assert metric.iou_threshold == 0.75

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError):
            MultiObjectTrackingAccuracy(iou_threshold=0.0)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError):
            MultiObjectTrackingAccuracy(iou_threshold=-0.1)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError):
            MultiObjectTrackingAccuracy(iou_threshold=1.5)


class TestMOTAPerfectTracking:
    def test_single_frame_perfect_match(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50], [100, 100, 150, 150]], [1, 2])
        pred = _make_detections([[10, 10, 50, 50], [100, 100, 150, 150]], [1, 2])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.mota == 1.0
        assert result.num_false_positives == 0
        assert result.num_false_negatives == 0
        assert result.num_id_switches == 0

    def test_multi_frame_perfect_match(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        for _ in range(5):
            gt = _make_detections([[10, 10, 50, 50]], [1])
            pred = _make_detections([[10, 10, 50, 50]], [1])
            metric.update(gt, pred)
        result = metric.compute()
        assert result.mota == 1.0
        assert result.num_frames == 5


class TestMOTAFalsePositives:
    def test_extra_predictions(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[10, 10, 50, 50], [200, 200, 250, 250]], [1, 99])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.num_false_positives == 1
        assert result.mota == 0.0

    def test_all_false_positives(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[200, 200, 250, 250]], [99])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.mota == -1.0


class TestMOTAFalseNegatives:
    def test_missed_detections(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50], [100, 100, 150, 150]], [1, 2])
        pred = _make_detections([[10, 10, 50, 50]], [1])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.num_false_negatives == 1
        assert result.mota == 0.5

    def test_no_predictions(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _empty_detections()
        metric.update(gt, pred)
        result = metric.compute()
        assert result.mota == 0.0


class TestMOTAIDSwitches:
    def test_identity_switch(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt1 = _make_detections([[10, 10, 50, 50]], [1])
        pred1 = _make_detections([[10, 10, 50, 50]], [10])
        metric.update(gt1, pred1)
        gt2 = _make_detections([[12, 12, 52, 52]], [1])
        pred2 = _make_detections([[12, 12, 52, 52]], [20])
        metric.update(gt2, pred2)
        result = metric.compute()
        assert result.num_id_switches == 1
        assert result.mota == 0.5

    def test_no_switch_same_id(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt1 = _make_detections([[10, 10, 50, 50]], [1])
        pred1 = _make_detections([[10, 10, 50, 50]], [10])
        metric.update(gt1, pred1)
        gt2 = _make_detections([[12, 12, 52, 52]], [1])
        pred2 = _make_detections([[12, 12, 52, 52]], [10])
        metric.update(gt2, pred2)
        result = metric.compute()
        assert result.num_id_switches == 0
        assert result.mota == 1.0


class TestMOTAIoUThreshold:
    def test_low_iou_no_match(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[45, 45, 90, 90]], [1])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.mota == -1.0

    def test_strict_threshold(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.9)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[15, 15, 55, 55]], [1])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.num_false_negatives == 1


class TestMOTAReset:
    def test_reset_clears_state(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[10, 10, 50, 50]], [1])
        metric.update(gt, pred)
        metric.reset()
        with pytest.raises(ValueError):
            metric.compute()


class TestMOTAMissingTrackerID:
    def test_gt_missing(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = Detections(xyxy=np.array([[10, 10, 50, 50]]))
        pred = _make_detections([[10, 10, 50, 50]], [1])
        with pytest.raises(ValueError):
            metric.update(gt, pred)

    def test_pred_missing(self):
        metric = MultiObjectTrackingAccuracy(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = Detections(xyxy=np.array([[10, 10, 50, 50]]))
        with pytest.raises(ValueError):
            metric.update(gt, pred)


class TestMOTPPerfect:
    def test_exact_overlap(self):
        metric = MultiObjectTrackingPrecision(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[10, 10, 50, 50]], [1])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.motp == 1.0

    def test_multi_frame(self):
        metric = MultiObjectTrackingPrecision(iou_threshold=0.5)
        for _ in range(3):
            gt = _make_detections([[0, 0, 100, 100]], [1])
            pred = _make_detections([[0, 0, 100, 100]], [1])
            metric.update(gt, pred)
        result = metric.compute()
        assert result.motp == 1.0
        assert result.num_matches == 3


class TestMOTPPartialOverlap:
    def test_offset_boxes(self):
        metric = MultiObjectTrackingPrecision(iou_threshold=0.3)
        gt = _make_detections([[0, 0, 100, 100]], [1])
        pred = _make_detections([[50, 0, 150, 100]], [1])
        metric.update(gt, pred)
        result = metric.compute()
        assert abs(result.motp - 1.0 / 3.0) < 1e-6

    def test_no_match(self):
        metric = MultiObjectTrackingPrecision(iou_threshold=0.5)
        gt = _make_detections([[0, 0, 100, 100]], [1])
        pred = _make_detections([[200, 200, 300, 300]], [1])
        metric.update(gt, pred)
        with pytest.raises(ValueError):
            metric.compute()


class TestMOTPReset:
    def test_reset(self):
        metric = MultiObjectTrackingPrecision(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[10, 10, 50, 50]], [1])
        metric.update(gt, pred)
        metric.reset()
        with pytest.raises(ValueError):
            metric.compute()


class TestTrackingMetricsPerfect:
    def test_perfect(self):
        metric = TrackingMetrics(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50], [100, 100, 150, 150]], [1, 2])
        pred = _make_detections([[10, 10, 50, 50], [100, 100, 150, 150]], [1, 2])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.mota == 1.0
        assert result.motp == 1.0
        assert result.num_matches == 2


class TestTrackingMetricsMixed:
    def test_fp_and_fn(self):
        metric = TrackingMetrics(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50], [100, 100, 150, 150]], [1, 2])
        pred = _make_detections([[10, 10, 50, 50], [200, 200, 250, 250]], [1, 99])
        metric.update(gt, pred)
        result = metric.compute()
        assert result.num_false_negatives == 1
        assert result.num_false_positives == 1
        assert result.mota == 0.0
        assert result.motp == 1.0

    def test_id_switch(self):
        metric = TrackingMetrics(iou_threshold=0.5)
        gt1 = _make_detections([[10, 10, 50, 50]], [1])
        pred1 = _make_detections([[10, 10, 50, 50]], [10])
        metric.update(gt1, pred1)
        gt2 = _make_detections([[12, 12, 52, 52]], [1])
        pred2 = _make_detections([[12, 12, 52, 52]], [20])
        metric.update(gt2, pred2)
        result = metric.compute()
        assert result.num_id_switches == 1
        assert result.mota == 0.5


class TestTrackingMetricsReset:
    def test_reset(self):
        metric = TrackingMetrics(iou_threshold=0.5)
        gt = _make_detections([[10, 10, 50, 50]], [1])
        pred = _make_detections([[10, 10, 50, 50]], [1])
        metric.update(gt, pred)
        metric.reset()
        with pytest.raises(ValueError):
            metric.compute()


class TestResultStrings:
    def test_mota_str(self):
        r = MOTAResult(
            mota=0.75,
            num_false_positives=5,
            num_false_negatives=10,
            num_id_switches=2,
            num_ground_truth=68,
            num_frames=20,
        )
        assert "0.75" in str(r)

    def test_motp_str(self):
        r = MOTPResult(motp=0.85, total_iou=8.5, num_matches=10, num_frames=5)
        assert "0.85" in str(r)

    def test_combined_str(self):
        r = MOTMetricsResult(
            mota=0.8,
            motp=0.9,
            num_false_positives=3,
            num_false_negatives=5,
            num_id_switches=1,
            num_ground_truth=45,
            num_matches=36,
            num_frames=10,
        )
        assert "0.8" in str(r)
