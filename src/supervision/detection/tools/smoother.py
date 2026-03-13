from __future__ import annotations

import warnings
from collections import defaultdict, deque
from copy import deepcopy
from typing import cast

import numpy as np

from supervision.detection.core import Detections
from supervision.utils.internal import SupervisionWarnings


class DetectionsSmoother:
    """
    A utility class for smoothing detections over multiple frames in video tracking.
    It maintains a history of detections for each track and provides smoothed
    predictions based on these histories.

    <video controls>
        <source
            src="https://media.roboflow.com/supervision-detection-smoothing.mp4"
            type="video/mp4">
    </video>

    !!! warning

        - `DetectionsSmoother` requires the `tracker_id` for each detection. Refer to
          [Roboflow Trackers](/latest/trackers/) for
          information on integrating tracking into your inference pipeline.
        - This class is not compatible with segmentation models.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> smoother = sv.DetectionsSmoother(length=3)
        >>> detections_1 = sv.Detections(
        ...     xyxy=np.array([[0, 0, 10, 10]]),
        ...     confidence=np.array([0.5]),
        ...     tracker_id=np.array([1])
        ... )
        >>> detections_2 = sv.Detections(
        ...     xyxy=np.array([[2, 2, 12, 12]]),
        ...     confidence=np.array([0.7]),
        ...     tracker_id=np.array([1])
        ... )
        >>> smoothed = smoother.update_with_detections(detections_1)
        >>> smoothed.xyxy
        array([[ 0.,  0., 10., 10.]])
        >>> smoothed = smoother.update_with_detections(detections_2)
        >>> smoothed.xyxy
        array([[ 1.,  1., 11., 11.]])
        >>> smoothed.confidence
        array([0.6])

        ```


        ```python
        import supervision as sv

        from ultralytics import YOLO

        video_info = sv.VideoInfo.from_video_path(video_path="<SOURCE_FILE_PATH>")
        frame_generator = sv.get_video_frames_generator(
            source_path="<SOURCE_FILE_PATH>")

        model = YOLO("<MODEL_PATH>")
        tracker = sv.ByteTrack(frame_rate=video_info.fps)
        smoother = sv.DetectionsSmoother()

        box_annotator = sv.BoxAnnotator()

        with sv.VideoSink("<TARGET_FILE_PATH>", video_info=video_info) as sink:
            for frame in frame_generator:
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = tracker.update_with_detections(detections)
                detections = smoother.update_with_detections(detections)

                annotated_frame = box_annotator.annotate(frame.copy(), detections)
                sink.write_frame(annotated_frame)
        ```
    """

    def __init__(self, length: int = 5) -> None:
        """
        Args:
            length: The maximum number of frames to consider for smoothing
                detections. Defaults to 5.
        """
        self.tracks: defaultdict[int, deque[Detections | None]] = defaultdict(
            lambda: deque(maxlen=length)
        )

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the smoother with a new set of detections from a frame.

        Args:
            detections: The detections to add to the smoother.
        """

        if detections.tracker_id is None:
            warnings.warn(
                "Smoothing skipped. DetectionsSmoother requires tracker_id. Refer to "
                "https://supervision.roboflow.com/latest/trackers for more "
                "information.",
                category=SupervisionWarnings,
            )
            return detections

        for detection_idx in range(len(detections)):
            tracker_id_value = detections.tracker_id[detection_idx]
            tracker_id = int(tracker_id_value)

            self.tracks[tracker_id].append(cast(Detections, detections[detection_idx]))

        for track_id in self.tracks.keys():
            if track_id not in detections.tracker_id:
                self.tracks[track_id].append(None)

        for track_id in list(self.tracks.keys()):
            if all([d is None for d in self.tracks[track_id]]):
                del self.tracks[track_id]

        return self.get_smoothed_detections()

    def get_track(self, track_id: int) -> Detections | None:
        track = self.tracks.get(track_id, None)
        if track is None:
            return None

        valid: list[Detections] = [d for d in track if d is not None]
        if len(valid) == 0:
            return None

        ret = deepcopy(valid[0])
        ret.xyxy = np.mean([d.xyxy for d in valid], axis=0)
        ret.confidence = np.mean([d.confidence for d in valid], axis=0)

        return ret

    def get_smoothed_detections(self) -> Detections:
        tracked_detections = []
        for track_id in self.tracks:
            track = self.get_track(track_id)
            if track is not None:
                tracked_detections.append(track)

        detections = Detections.merge(tracked_detections)
        if len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)

        return detections
