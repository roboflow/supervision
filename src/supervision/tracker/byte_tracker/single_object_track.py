from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.utils import IdCounter


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class _STrackData:
    tlwh: npt.NDArray[np.float32]
    score: float
    minimum_consecutive_frames: int
    internal_id_counter: IdCounter
    external_id_counter: IdCounter
    state: TrackState = TrackState.New
    is_activated: bool = False
    start_frame: int = 0
    frame_id: int = 0
    kalman_filter: KalmanFilter | None = None
    mean: npt.NDArray[np.float32] | None = None
    covariance: npt.NDArray[np.float32] | None = None
    tracklet_len: int = 0
    internal_track_id: int = field(init=False)
    external_track_id: int = field(init=False)

    def __post_init__(self) -> None:
        self.tlwh = np.asarray(self.tlwh, dtype=np.float32)
        self.internal_track_id = self.internal_id_counter.NO_ID
        self.external_track_id = self.external_id_counter.NO_ID


class STrack:
    def __init__(
        self,
        tlwh: npt.NDArray[np.float32],
        score: float,
        minimum_consecutive_frames: int,
        internal_id_counter: IdCounter,
        external_id_counter: IdCounter,
    ):
        self._data = _STrackData(
            tlwh=tlwh,
            score=score,
            minimum_consecutive_frames=minimum_consecutive_frames,
            internal_id_counter=internal_id_counter,
            external_id_counter=external_id_counter,
        )

    @property
    def state(self) -> TrackState:
        return self._data.state

    @state.setter
    def state(self, value: TrackState) -> None:
        self._data.state = value

    @property
    def is_activated(self) -> bool:
        return self._data.is_activated

    @is_activated.setter
    def is_activated(self, value: bool) -> None:
        self._data.is_activated = value

    @property
    def start_frame(self) -> int:
        return self._data.start_frame

    @start_frame.setter
    def start_frame(self, value: int) -> None:
        self._data.start_frame = value

    @property
    def frame_id(self) -> int:
        return self._data.frame_id

    @frame_id.setter
    def frame_id(self, value: int) -> None:
        self._data.frame_id = value

    @property
    def kalman_filter(self) -> KalmanFilter | None:
        return self._data.kalman_filter

    @kalman_filter.setter
    def kalman_filter(self, value: KalmanFilter | None) -> None:
        self._data.kalman_filter = value

    @property
    def mean(self) -> npt.NDArray[np.float32] | None:
        return self._data.mean

    @mean.setter
    def mean(self, value: npt.NDArray[np.float32] | None) -> None:
        self._data.mean = value

    @property
    def covariance(self) -> npt.NDArray[np.float32] | None:
        return self._data.covariance

    @covariance.setter
    def covariance(self, value: npt.NDArray[np.float32] | None) -> None:
        self._data.covariance = value

    @property
    def score(self) -> float:
        return self._data.score

    @score.setter
    def score(self, value: float) -> None:
        self._data.score = value

    @property
    def tracklet_len(self) -> int:
        return self._data.tracklet_len

    @tracklet_len.setter
    def tracklet_len(self, value: int) -> None:
        self._data.tracklet_len = value

    @property
    def minimum_consecutive_frames(self) -> int:
        return self._data.minimum_consecutive_frames

    @property
    def internal_track_id(self) -> int:
        return self._data.internal_track_id

    @internal_track_id.setter
    def internal_track_id(self, value: int) -> None:
        self._data.internal_track_id = value

    @property
    def external_track_id(self) -> int:
        return self._data.external_track_id

    @external_track_id.setter
    def external_track_id(self, value: int) -> None:
        self._data.external_track_id = value

    def predict(self) -> None:
        assert self._data.mean is not None
        assert self._data.covariance is not None
        assert self._data.kalman_filter is not None
        mean_state = self._data.mean.copy()
        if self._data.state != TrackState.Tracked:
            mean_state[7] = 0
        self._data.mean, self._data.covariance = self._data.kalman_filter.predict(
            mean_state, self._data.covariance
        )

    @staticmethod
    def multi_predict(stracks: list[STrack], shared_kalman: KalmanFilter) -> None:
        if len(stracks) > 0:
            multi_mean = []
            multi_covariance = []
            for i, st in enumerate(stracks):
                assert st._data.mean is not None
                assert st._data.covariance is not None
                multi_mean.append(st._data.mean.copy())
                multi_covariance.append(st._data.covariance)
                if st._data.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = shared_kalman.multi_predict(
                np.asarray(multi_mean), np.asarray(multi_covariance)
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i]._data.mean = mean
                stracks[i]._data.covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet"""
        self._data.kalman_filter = kalman_filter
        self._data.internal_track_id = self._data.internal_id_counter.new_id()
        self._data.mean, self._data.covariance = self._data.kalman_filter.initiate(
            self.tlwh_to_xyah(self._data.tlwh)
        )

        self._data.tracklet_len = 0
        self._data.state = TrackState.Tracked
        if frame_id == 1:
            self._data.is_activated = True
            if self._data.minimum_consecutive_frames == 1:
                self._data.external_track_id = self._data.external_id_counter.new_id()

        self._data.frame_id = frame_id
        self._data.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int) -> None:
        assert self._data.kalman_filter is not None
        assert self._data.mean is not None
        assert self._data.covariance is not None
        self._data.mean, self._data.covariance = self._data.kalman_filter.update(
            self._data.mean, self._data.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self._data.tracklet_len = 0
        self._data.state = TrackState.Tracked

        self._data.frame_id = frame_id
        self._data.score = new_track.score

    def update(self, new_track: STrack, frame_id: int) -> None:
        """
        Update a matched track.

        Args:
            new_track: The new track data.
            frame_id: The current frame ID.
        """
        assert self._data.kalman_filter is not None
        assert self._data.mean is not None
        assert self._data.covariance is not None
        self._data.frame_id = frame_id
        self._data.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._data.mean, self._data.covariance = self._data.kalman_filter.update(
            self._data.mean, self._data.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self._data.state = TrackState.Tracked
        if self._data.tracklet_len == self._data.minimum_consecutive_frames:
            self._data.is_activated = True
            if self._data.external_track_id == self._data.external_id_counter.NO_ID:
                self._data.external_track_id = self._data.external_id_counter.new_id()

        self._data.score = new_track.score

    @property
    def tlwh(self) -> npt.NDArray[np.float32]:
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self._data.mean is None:
            return self._data.tlwh.copy()
        ret = self._data.mean[:4].copy()
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
    def tlwh_to_xyah(tlwh: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
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
    def tlbr_to_tlwh(tlbr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self._data.internal_track_id}_({self._data.start_frame}-{self._data.frame_id})"
