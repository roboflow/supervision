from collections import deque
from typing import List

import numpy as np

from supervision.detection.core import Detections
from supervision.tracker.confidence_tracker import matching
from supervision.tracker.confidence_tracker.basetrack import BaseTrack, TrackState
from supervision.tracker.confidence_tracker.kalman_filter import KalmanFilter


class STrack(BaseTrack):
    shared_kalman = KalmanFilter(0.6, 10)

    def __init__(self, tlwh, score, class_id, feat=None, feat_history=50):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(class_id, score)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xywh(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh), self.score
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.score = new_track.score

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh), self.score
        )

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.update_cls(new_track.cls, new_track.score)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


def detections2boxes(detections: Detections) -> np.ndarray:
    """
    Convert Supervision Detections to numpy tensors for further computation.
    Args:
        detections (Detections): Detections/Targets in the format of sv.Detections.
        features (ndarray): The corresponding image features of each detection.
        Has shape [N, num_features]
    Returns:
        (np.ndarray): Detections as numpy tensors as in
            `(x_min, y_min, x_max, y_max, confidence, class_id, feature_vect)` order.
    """
    return np.hstack(
        (
            detections.xyxy,
            detections.confidence[:, np.newaxis],
            detections.class_id[:, np.newaxis],
        )
    )


class ConfTrack:
    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.2,
        new_track_thresh: float = 0.1,
        tent_conf_thresh: float = 0.7,
        minimum_matching_threshold: float = 0.8,
        lost_track_buffer: int = 30,
        proximity_thresh: float = 0.6,
        frame_rate: int = 30,
    ):
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.tent_conf_thresh = tent_conf_thresh

        self.minimum_matching_threshold = minimum_matching_threshold

        # self.buffer_size = int(frame_rate / 30.0 * args.lost_track_buffer)
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.kalman_filter = KalmanFilter(0.6, 10)

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections and
            returns the updated detection results.

        Parameters:
            detections: The new detections to update with.
        Returns:
            Detection: The updated detection results that now include tracking IDs.
        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> model = YOLO(...)
            >>> byte_tracker = sv.ByteTrack()
            >>> annotator = sv.BoxAnnotator()

            >>> def callback(frame: np.ndarray, index: int) -> np.ndarray:
            ...     results = model(frame)[0]
            ...     detections = sv.Detections.from_ultralytics(results)
            ...     detections = byte_tracker.update_with_detections(detections)
            ...     labels = [
            ...         f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            ...         for _, _, confidence, class_id, tracker_id
            ...         in detections
            ...     ]
            ...     return annotator.annotate(scene=frame.copy(),
            ...                               detections=detections, labels=labels)

            >>> sv.process_video(
            ...     source_path='...',
            ...     target_path='...',
            ...     callback=callback
            ... )
            ```
        """
        tensors = detections2boxes(detections)
        # print(f"tensors: {tensors}")
        tracks = self.update_with_tensors(
            # maybe extract features here
            tensors
        )
        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array([track.tlbr for track in tracks], dtype=float)
            detections.class_id = np.array([int(t.cls) for t in tracks], dtype=int)
            detections.tracker_id = np.array(
                [int(t.track_id) for t in tracks], dtype=int
            )
            detections.confidence = np.array([t.score for t in tracks], dtype=float)
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections

    def reset(self):
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        self.frame_id = 0
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        BaseTrack.reset_counter()

    def update_with_tensors(self, tensors: np.ndarray) -> List[STrack]:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Parameters:
            tensors: The new tensors to update with.

        Returns:
            List[STrack]: Updated tracks.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_tracks = []
        removed_stracks = []

        class_ids = tensors[:, 5]
        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        # Remove bad detections
        inds_low = scores > self.track_low_thresh
        bboxes = bboxes[inds_low]
        scores = scores[inds_low]
        class_ids = class_ids[inds_low]

        # Find high threshold detections
        inds_high = scores > self.track_high_thresh
        dets = bboxes[inds_high]
        scores_keep = scores[inds_high]
        class_ids_keep = class_ids[inds_high]

        # Find low threshold detections
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        class_ids_second = class_ids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, cl)
                for (tlbr, s, cl) in zip(dets, scores_keep, class_ids_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_tracks"""
        low_tent = []  # type: list[STrack]
        high_tent = []  # type: list[STrack]
        tracked_tracks = []  # type: list[STrack]
        for track in self.tracked_tracks:
            if not track.is_activated:
                # implement LM from ConfTrack paper
                if track.score < self.tent_conf_thresh:
                    low_tent.append(track)
                else:
                    high_tent.append(track)
            else:
                tracked_tracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_tracks, self.lost_tracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # LM algorithm
        strack_pool = joint_stracks(strack_pool, high_tent)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)

        # Fuse the iou's with the scores
        ious_dists = matching.fuse_score(ious_dists, detections)

        matches, track_conf_remain, det_high_remain = matching.linear_assignment(
            ious_dists, thresh=self.minimum_matching_threshold
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, cl)
                for (tlbr, s, cl) in zip(dets_second, scores_second, class_ids_second)
            ]
        else:
            detections_second = []

        r_tracked_tracks = [
            strack_pool[i]
            for i in track_conf_remain
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_tracks, detections_second)
        matches, track_conf_remain, det_low_remain = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_tracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # implement LM from ConfTrack paper
        """Step 4: low-confidence track matching with high-conf dets"""
        # Associate with high score detection boxes
        stracks_conf_remain = [r_tracked_tracks[i] for i in track_conf_remain]
        ious_dists = matching.iou_distance(low_tent, stracks_conf_remain)
        _, low_tent_valid, _ = matching.linear_assignment(
            ious_dists, thresh=1 - 0.7
        )  # want to get rid of tracks with low iou costs
        stracks_low_tent_valid = [low_tent[i] for i in low_tent_valid]
        stracks_det_high_remain = [detections[i] for i in det_high_remain]
        C_low_ious = matching.iou_distance(
            stracks_low_tent_valid, stracks_det_high_remain
        )

        matches, track_tent_remain, det_high_remain = matching.linear_assignment(
            C_low_ious, thresh=0.3
        )  # need to find this val in ConfTrack paper

        for itracked, idet in matches:
            low_tent[itracked].update(stracks_det_high_remain[idet], self.frame_id)
            activated_starcks.append(low_tent[itracked])

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        for it in track_tent_remain:
            track = stracks_low_tent_valid[it]
            track.mark_removed()
            removed_stracks.append(track)
        # left over confirmed tracks get lost
        for it in track_conf_remain:
            track = r_tracked_tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        """ Step 5: Init new stracks"""
        for inew in det_high_remain:
            track = stracks_det_high_remain[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for inew in det_low_remain:
            track = detections_second[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 6: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_stracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_stracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_stracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = sub_stracks(self.lost_tracks, self.removed_stracks)
        self.removed_tracks.extend(removed_stracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_stracks(
            self.tracked_tracks, self.lost_tracks
        )

        output_stracks = [track for track in self.tracked_tracks]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
