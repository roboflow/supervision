# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """
    GATING_THRESHOLD = np.sqrt(kalman_filter.chi2inv95[4])

    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, _lambda=0, ema_alpha=0.9, mc_lambda=0.995):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def camera_update(self, previous_img, current_img):
        for track in self.tracks:
            track.camera_update(previous_img, current_img)

    def update(self, detections, classes, confidences):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                detections[detection_idx], classes[detection_idx], confidences[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], classes[detection_idx].item(), confidences[detection_idx].item())
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This implements the full lambda-based cost-metric. However, in doing so, it disregards
        the possibility to gate the position only which is provided by
        linear_assignment.gate_cost_matrix(). Instead, I gate by everything.
        Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
        distance being normalised, we employ a quick and dirty normalisation based on the
        threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
        that the valid values range 0-1.
        Note also that the authors work with the squared distance. I also sqrt this, so that it
        is more intuitive in terms of values.
        """
        # Compute First the Position-based Cost Matrix
        pos_cost = np.empty([len(track_indices), len(detection_indices)])
        msrs = np.asarray([dets[i].to_xyah() for i in detection_indices])
        for row, track_idx in enumerate(track_indices):
            pos_cost[row, :] = np.sqrt(
                self.kf.gating_distance(
                    tracks[track_idx].mean, tracks[track_idx].covariance, msrs, False
                )
            ) / self.GATING_THRESHOLD
        pos_gate = pos_cost > 1.0
        # Now Compute the Appearance-based Cost Matrix
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )
        app_gate = app_cost > self.metric.matching_threshold
        # Now combine and threshold
        cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
        cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST
        # Return Matrix
        return cost_matrix

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(cost_matrix, tracks, dets, track_indices, detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id, conf):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, class_id, conf, self.n_init, self.max_age, self.ema_alpha,
            detection.feature))
        self._next_id += 1
