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
        self.storage = np.zeros((0, 5))

    def update(self, frame_counter: int, detections: Detections) -> None:
        if detections.xyxy.shape[0] > 0:
            xyxy = detections.xyxy

            if self.position == Position.CENTER:
                x = (xyxy[:, 0] + xyxy[:, 2]) / 2
                y = (xyxy[:, 1] + xyxy[:, 3]) / 2
            elif self.position == Position.BOTTOM_CENTER:
                x = (xyxy[:, 0] + xyxy[:, 2]) / 2
                y = xyxy[:, 3]

            new_detections = np.zeros(shape=(xyxy.shape[0], 5))
            new_detections[:, 0] = frame_counter
            new_detections[:, 1] = x
            new_detections[:, 2] = y
            new_detections[:, 3] = detections.class_id
            new_detections[:, 4] = detections.tracker_id
            self.storage = np.append(self.storage, new_detections, axis=0)
            self.frame_counter = frame_counter
            self._remove_lost(tracker_ids=detections.tracker_id)
        self._remove_previous()

    def _remove_previous(self) -> None:
        to_remove_frames = self.frame_counter - self.max_length
        if self.storage.shape[0] > 0:
            self.storage = self.storage[
                self.storage[:, 0] > to_remove_frames
                ]

    def _remove_lost(self, tracker_ids) -> None:
        if self.storage.shape[0] > 0:
            self.storage = self.storage[np.isin(self.storage[:, -1], tracker_ids)]
