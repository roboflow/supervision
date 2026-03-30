from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from supervision.detection.core import Detections


@dataclass
class MOTAResult:
    mota: float
    num_false_positives: int
    num_false_negatives: int
    num_id_switches: int
    num_ground_truth: int
    num_frames: int

    def __str__(self):
        return (
            f"MOTAResult(mota={self.mota:.4f}, "
            f"fp={self.num_false_positives}, "
            f"fn={self.num_false_negatives}, "
            f"idsw={self.num_id_switches})"
        )


@dataclass
class MOTPResult:
    motp: float
    total_iou: float
    num_matches: int
    num_frames: int

    def __str__(self):
        return (
            f"MOTPResult(motp={self.motp:.4f}, "
            f"matches={self.num_matches})"
        )


@dataclass
class MOTMetricsResult:
    mota: float
    motp: float
    num_false_positives: int
    num_false_negatives: int
    num_id_switches: int
    num_ground_truth: int
    num_matches: int
    num_frames: int

    def __str__(self):
        return (
            f"MOTMetricsResult(mota={self.mota:.4f}, "
            f"motp={self.motp:.4f}, "
            f"fp={self.num_false_positives}, "
            f"fn={self.num_false_negatives}, "
            f"idsw={self.num_id_switches})"
        )


def _compute_iou_matrix(boxes_a, boxes_b):
    x1 = np.maximum(boxes_a[:, 0][:, None], boxes_b[:, 0][None, :])
    y1 = np.maximum(boxes_a[:, 1][:, None], boxes_b[:, 1][None, :])
    x2 = np.minimum(boxes_a[:, 2][:, None], boxes_b[:, 2][None, :])
    y2 = np.minimum(boxes_a[:, 3][:, None], boxes_b[:, 3][None, :])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def _greedy_match(iou_matrix, threshold):
    matches = []
    matched_gt = set()
    matched_pred = set()
    num_gt, num_pred = iou_matrix.shape
    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        gt_idx = int(flat_idx // num_pred)
        pred_idx = int(flat_idx % num_pred)
        if iou_matrix[gt_idx, pred_idx] < threshold:
            break
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matches.append((gt_idx, pred_idx))
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
    return matched_gt, matched_pred, matches


def _validate_tracker_ids(ground_truth, predictions):
    if ground_truth.tracker_id is None:
        raise ValueError("ground_truth.tracker_id must be set for tracking metrics.")
    if predictions.tracker_id is None:
        raise ValueError("predictions.tracker_id must be set for tracking metrics.")


class MultiObjectTrackingAccuracy:
    def __init__(self, iou_threshold=0.5):
        if not 0 < iou_threshold <= 1:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self._false_positives = 0
        self._false_negatives = 0
        self._id_switches = 0
        self._ground_truth_count = 0
        self._num_frames = 0
        self._last_match = {}

    def update(self, ground_truth, predictions):
        _validate_tracker_ids(ground_truth, predictions)
        self._num_frames += 1
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        self._ground_truth_count += num_gt
        if num_gt == 0 and num_pred == 0:
            return
        if num_gt == 0:
            self._false_positives += num_pred
            return
        if num_pred == 0:
            self._false_negatives += num_gt
            return
        iou_matrix = _compute_iou_matrix(ground_truth.xyxy, predictions.xyxy)
        _, _, matches = _greedy_match(iou_matrix, self.iou_threshold)
        self._false_negatives += num_gt - len(matches)
        self._false_positives += num_pred - len(matches)
        for gt_idx, pred_idx in matches:
            gt_id = int(ground_truth.tracker_id[gt_idx])
            pred_id = int(predictions.tracker_id[pred_idx])
            if gt_id in self._last_match:
                if self._last_match[gt_id] != pred_id:
                    self._id_switches += 1
            self._last_match[gt_id] = pred_id

    def compute(self):
        if self._ground_truth_count == 0:
            raise ValueError("No ground truth objects found. Call update() first.")
        total_errors = self._false_negatives + self._false_positives + self._id_switches
        mota = 1.0 - total_errors / self._ground_truth_count
        return MOTAResult(
            mota=mota,
            num_false_positives=self._false_positives,
            num_false_negatives=self._false_negatives,
            num_id_switches=self._id_switches,
            num_ground_truth=self._ground_truth_count,
            num_frames=self._num_frames,
        )


class MultiObjectTrackingPrecision:
    def __init__(self, iou_threshold=0.5):
        if not 0 < iou_threshold <= 1:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self._total_iou = 0.0
        self._num_matches = 0
        self._num_frames = 0

    def update(self, ground_truth, predictions):
        _validate_tracker_ids(ground_truth, predictions)
        self._num_frames += 1
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        if num_gt == 0 or num_pred == 0:
            return
        iou_matrix = _compute_iou_matrix(ground_truth.xyxy, predictions.xyxy)
        _, _, matches = _greedy_match(iou_matrix, self.iou_threshold)
        for gt_idx, pred_idx in matches:
            self._total_iou += iou_matrix[gt_idx, pred_idx]
            self._num_matches += 1

    def compute(self):
        if self._num_matches == 0:
            raise ValueError("No matches found. Call update() with matching detections first.")
        motp = self._total_iou / self._num_matches
        return MOTPResult(
            motp=motp,
            total_iou=self._total_iou,
            num_matches=self._num_matches,
            num_frames=self._num_frames,
        )


class TrackingMetrics:
    def __init__(self, iou_threshold=0.5):
        if not 0 < iou_threshold <= 1:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self._false_positives = 0
        self._false_negatives = 0
        self._id_switches = 0
        self._ground_truth_count = 0
        self._total_iou = 0.0
        self._num_matches = 0
        self._num_frames = 0
        self._last_match = {}

    def update(self, ground_truth, predictions):
        _validate_tracker_ids(ground_truth, predictions)
        self._num_frames += 1
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        self._ground_truth_count += num_gt
        if num_gt == 0 and num_pred == 0:
            return
        if num_gt == 0:
            self._false_positives += num_pred
            return
        if num_pred == 0:
            self._false_negatives += num_gt
            return
        iou_matrix = _compute_iou_matrix(ground_truth.xyxy, predictions.xyxy)
        _, _, matches = _greedy_match(iou_matrix, self.iou_threshold)
        self._false_negatives += num_gt - len(matches)
        self._false_positives += num_pred - len(matches)
        for gt_idx, pred_idx in matches:
            self._total_iou += iou_matrix[gt_idx, pred_idx]
            self._num_matches += 1
            gt_id = int(ground_truth.tracker_id[gt_idx])
            pred_id = int(predictions.tracker_id[pred_idx])
            if gt_id in self._last_match:
                if self._last_match[gt_id] != pred_id:
                    self._id_switches += 1
            self._last_match[gt_id] = pred_id

    def compute(self):
        if self._ground_truth_count == 0:
            raise ValueError("No ground truth objects found. Call update() first.")
        total_errors = self._false_negatives + self._false_positives + self._id_switches
        mota = 1.0 - total_errors / self._ground_truth_count
        motp = self._total_iou / self._num_matches if self._num_matches > 0 else 0.0
        return MOTMetricsResult(
            mota=mota,
            motp=motp,
            num_false_positives=self._false_positives,
            num_false_negatives=self._false_negatives,
            num_id_switches=self._id_switches,
            num_ground_truth=self._ground_truth_count,
            num_matches=self._num_matches,
            num_frames=self._num_frames,
        )
