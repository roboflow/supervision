from datetime import datetime
from typing import Dict

import numpy as np

import supervision as sv


class FPSBasedTimer:
    """
    A timer that calculates the duration each object has been detected based on frames
    per second (FPS).

    Attributes:
        fps (int): The frame rate of the video stream, used to calculate time durations.
        frame_id (int): The current frame number in the sequence.
        tracker_id2frame_id (Dict[int, int]): Maps each tracker's ID to the frame number
            at which it was first detected.
    """

    def __init__(self, fps: int = 30) -> None:
        """Initializes the FPSBasedTimer with the specified frames per second rate.

        Args:
            fps (int): The frame rate of the video stream. Defaults to 30.
        """
        self.fps = fps
        self.frame_id = 0
        self.tracker_id2frame_id: Dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        """Processes the current frame, updating time durations for each tracker.

        Args:
            detections: The detections for the current frame, including tracker IDs.

        Returns:
            np.ndarray: Time durations (in seconds) for each detected tracker, since
                their first detection.
        """
        self.frame_id += 1
        times = []

        for tracker_id in detections.tracker_id:
            self.tracker_id2frame_id.setdefault(tracker_id, self.frame_id)

            start_frame_id = self.tracker_id2frame_id[tracker_id]
            time_duration = (self.frame_id - start_frame_id) / self.fps
            times.append(time_duration)

        return np.array(times)


class ClockBasedTimer:
    """
    A timer that calculates the duration each object has been detected based on the
    system clock.

    Attributes:
        tracker_id2start_time (Dict[int, datetime]): Maps each tracker's ID to the
            datetime when it was first detected.
    """

    def __init__(self) -> None:
        """Initializes the ClockBasedTimer."""
        self.tracker_id2start_time: Dict[int, datetime] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        """Processes the current frame, updating time durations for each tracker.

        Args:
            detections: The detections for the current frame, including tracker IDs.

        Returns:
            np.ndarray: Time durations (in seconds) for each detected tracker, since
                their first detection.
        """
        current_time = datetime.now()
        times = []

        for tracker_id in detections.tracker_id:
            self.tracker_id2start_time.setdefault(tracker_id, current_time)

            start_time = self.tracker_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        return np.array(times)
