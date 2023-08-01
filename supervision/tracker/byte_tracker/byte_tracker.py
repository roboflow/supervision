from typing import List

import numpy as np

from supervision import Detections
from supervision.detection.utils import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.basetrack import BaseTrack, TrackState
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter


class Strack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
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
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = Strack.shared_kalman.multi_predict(
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
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

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
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
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

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

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


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    """
    Convert Detections into a format that can be consumed by the match_detections_with_tracks function.

    Parameters:
        detections (Detections): An object representing the detected bounding boxes.

    Returns:
        np.ndarray: An array containing the bounding boxes' coordinates (xyxy) and their corresponding confidences.
    """
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


# converts List[strack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[Strack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, tracks: List[Strack]
) -> Detections:
    """
    Match bounding boxes from detections with existing strack objects.

    Parameters:
        detections (Detections): An object representing the detected bounding boxes.
        tracks (List[strack]): A list of strack objects to match with the detections.

    Returns:
        Detections: An object representing the matched tracker IDs for the detections.
    """
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


class ByteTrack:
    def __init__(
        self,
        track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        aspect_ratio_thresh=3.0,
        min_box_area=1.0,
        mot20=False,
        frame_rate=30,
    ):

        """
        Initialize the ByteTrack object.

        Parameters:
            track_thresh (float, optional): Detection confidence threshold for track activation.
            track_buffer (int, optional): Number of frames to buffer when a track is lost.
            match_thresh (float, optional): Threshold for matching tracks with detections.
            aspect_ratio_thresh (float, optional): Threshold for aspect ratio to filter out detections.
            min_box_area (float, optional): Minimum box area threshold to filter out detections.
            mot20 (bool, optional): Set to True for using MOT20 evaluation criteria.
            frame_rate (int, optional): The frame rate of the video.


        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20

        self.tracked_stracks = []  # type: list[Strack]
        self.lost_stracks = []  # type: list[Strack]
        self.removed_stracks = []  # type: list[Strack]

        self.frame_id = 0
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update_from_detections(self, detections, img_info, img_size):
        """
        Updates the strack with the provided results and frame info.

        Parameters:
            detections: The new detections to update with.
            img_info: Image Info.
            img_size: Image Size

        Returns:
            Detection: supervision detection result with track id
        Examples:
            ```python
            from tqdm.notebook import tqdm
            from ultralytics import YOLO

            import supervision as sv
            from supervision.detection.annotate import BoxAnnotator
            from supervision import ByteTrack
            from supervision.utils.video import VideoInfo, VideoSink, get_video_frames_generator

            model = YOLO("yolov5s.pt")
            CLASS_NAMES_DICT = model.model.names
            SOURCE_VIDEO_PATH = "walking.mp4"
            TARGET_VIDEO_PATH = "output.mp4"

            byte_tracker = ByteTrack(
                track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                aspect_ratio_thresh=3.0,
                min_box_area=1.0,
                mot20=False,
                frame_rate=30,
            )
            annotator = BoxAnnotator()

            # create VideoInfo instance
            video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
            # create frame generator
            generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

            # open target video file
            with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
                # loop over video frames
                for frame in tqdm(generator, total=video_info.total_frames):
                    # inference
                    results = model(frame)
                    sv_results = sv.Detections.from_yolov8(results[0])

                    # update tracker
                    detections_res = byte_tracker.update_from_detections(
                        detections=sv_results, img_info=frame.shape, img_size=frame.shape
                    )

                    frame_text_list = []
                    if len(detections_res) > 0:
                        for res in detections_res:
                            confidence = res[2]
                            class_id = res[3]
                            tracker_id = res[4]
                            frame_text = (
                                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                            )
                            frame_text_list.append(frame_text)
                    # draw bbox and add class name with track id
                    frame = annotator.annotate(
                        scene=frame, detections=detections_res, labels=frame_text_list
                    )
                    sink.write_frame(frame)
            ```
        """

        # update tracker
        tracks = self.update_from_numpy(
            output_results=detections2boxes(detections=detections),
            img_info=img_info,
            img_size=img_size,
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        if tracker_id != None:
            detections.tracker_id = np.array(tracker_id)
        return detections

    def update_from_numpy(self, output_results, img_info, img_size):
        """
        Update a matched strack. It uses the numpy results.

        Parameters:
            output_results: The new detections to update with.
            img_info: Image Info.
            img_size: Image Size

        Updates the strack with the provided results and frame info.
        Returns:
            Track_id: track id

        Examples:
            ```python
            from supervision import (
                ByteTrack,
                detections2boxes,
                match_detections_with_tracks,
            )
             # update tracker
            tracks = byte_tracker.update_from_numpy(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )

            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            ```
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                Strack(Strack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s) in zip(dets, scores_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[Strack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        Strack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
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
                Strack(Strack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

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
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
