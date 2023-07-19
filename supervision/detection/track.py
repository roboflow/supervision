from typing import Optional, Tuple

import numpy as np

from supervision.detection.core import Detections
from supervision.geometry.core import Position


class TrackStorage:
    """
    Trace the object trajectory with Detections with tracker id
    """

    def __init__(
        self,
        position: Position = Position.CENTER,
        max_length: int = 10,
    ):
        """
        Initialize the track storage to store tracked detections from tracker
        Args:
            position [sv.Position]: Trace position whethere mid-point or bottom center point
            max_length [int]: Length of previous detections to annotate
            Detections objects are stored as numpy ndarray with #counter, x, y, class, track_id

        Example:
            ```python
            >>> import supervision as sv
            >>> track_storage = sv.TrackStorage()
            >>> for frame in sv.get_video_frames_generator(source_path='source_video.mp4'):
            >>>     detections = sv.Detections(...)
            >>>     tracked_objects = tracker(...)
            >>>     tracked_detections = sv.Detections(tracked_objects)
            >>>     track_storage.update(tracked_detections)
        """
        self.position = position
        self.max_length = max_length
        self.frame_counter = 0

        self.xy = np.zeros((0, 2))
        self.confidence = np.zeros(0)
        self.class_id = np.zeros(0)
        self.tracker_id = np.zeros(0)
        self.frame_id = np.zeros(0)

    def update(self, frame_counter: int, detections: Detections) -> None:
        n_dets = detections.xyxy.shape[0]
        if n_dets > 0 and detections.tracker_id.shape[0] > 0:
            xy = detections.get_anchor_coordinates(anchor=self.position)

            self.tracker_id = np.append(self.tracker_id, detections.tracker_id, axis=0)

            self.xy = np.append(self.xy, xy, axis=0)
            if detections.class_id is not None:
                self.class_id = np.append(self.class_id, detections.class_id, axis=0)
            if detections.confidence is not None:
                self.confidence = np.append(
                    self.confidence, detections.confidence, axis=0
                )
            self.frame_id = np.append(
                self.frame_id, np.full((n_dets), fill_value=frame_counter), axis=0
            )
            self.frame_counter = frame_counter
            self._remove_lost(tracker_ids=detections.tracker_id)
        self._remove_previous()

    def _remove_previous(self) -> None:
        to_remove_frames = self.frame_counter - self.max_length
        if self.frame_id.shape[0] > 0:
            valid = self.frame_id > to_remove_frames
            self._remove(valid)

    def _remove_lost(self, tracker_ids) -> None:
        if self.tracker_id.shape[0] > 0:
            valid = np.isin(self.tracker_id, tracker_ids)
            self._remove(valid)

    def _remove(self, valid: Optional[Tuple]) -> None:
        if valid.size > 0:
            self.xy = self.xy[valid]
            self.frame_id = self.frame_id[valid]
            self.tracker_id = self.tracker_id[valid]
            if self.confidence.shape[0] > 0:
                self.confidence = self.confidence[valid]
            if self.class_id.shape[0] > 0:
                self.class_id = self.class_id[valid]
