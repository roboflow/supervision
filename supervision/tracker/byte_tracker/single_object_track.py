from __future__ import annotations
from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt

from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.utils import IdCounter


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    def __init__(
        self,
        tlwh: npt.NDArray[np.float32],
        score: npt.NDArray[np.float32],
        minimum_consecutive_frames: int,
        shared_kalman: KalmanFilter,
        internal_id_counter: IdCounter,
        external_id_counter: IdCounter,
    ):
        self.state = TrackState.New
        self.is_activated = False
        self.start_frame = 0
        self.frame_id = 0

        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.shared_kalman = shared_kalman
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.minimum_consecutive_frames = minimum_consecutive_frames

        self.internal_id_counter = internal_id_counter
        self.external_id_counter = external_id_counter
        self.internal_track_id = self.internal_id_counter.NO_ID
        self.external_track_id = self.external_id_counter.NO_ID

    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks: List[STrack], shared_kalman: KalmanFilter) -> None:
        if len(stracks) > 0:
            multi_mean = []
            multi_covariance = []
            for i, st in enumerate(stracks):
                multi_mean.append(st.mean.copy())
                multi_covariance.append(st.covariance)
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = shared_kalman.multi_predict(
                np.asarray(multi_mean), np.asarray(multi_covariance)
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.internal_track_id = self.internal_id_counter.new_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        if self.minimum_consecutive_frames == 1:
            self.external_track_id = self.external_id_counter.new_id()

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked

        self.frame_id = frame_id
        self.score = new_track.score

    def update(self, new_track: STrack, frame_id: int) -> None:
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
        if self.tracklet_len == self.minimum_consecutive_frames:
            self.is_activated = True
            if self.external_track_id == self.external_id_counter.NO_ID:
                self.external_track_id = self.external_id_counter.new_id()

        self.score = new_track.score

    @property
    def tlwh(self) -> npt.NDArray[np.float32]:
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
    def tlbr(self) -> npt.NDArray[np.float32]:
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh) -> npt.NDArray[np.float32]:
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> npt.NDArray[np.float32]:
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr) -> npt.NDArray[np.float32]:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh) -> npt.NDArray[np.float32]:
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return "OT_{}_({}-{})".format(
            self.internal_track_id, self.start_frame, self.frame_id
        )

