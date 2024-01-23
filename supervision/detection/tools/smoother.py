from collections import deque
from typing import Optional

import numpy as np

from supervision.detection.core import Detections


class DetectionsSmoother:
    """
    Smooth out noise in predictions over time with the `DetectionsSmoother` class.
    This classes uses an existing `Tracker` to track objects over time.
    Detections are averaged out over the `length` most recent frames.

    <video controls>
        <source src="https://media.roboflow.com/supervision/video-examples/smoothed-grocery-example-720.mp4" type="video/mp4">
    </video>
    > _On the left are the model's raw predictions,
    > on the right is the output of DetectionsSmoother._

    !!! warning

        DetectionsSmoother uses the `tracker_id`. Read
        [here](https://supervision.roboflow.com/trackers/) to learn how to plug
        tracking into your inference pipeline.

        Note: DetectionsSmoother is intended for use on Detections without a `mask` field.

    ## Example Usage:

    ```python
    import cv2
    # remember to `pip install inference`
    from inference import InferencePipeline
    import supervision as sv

    box_annotator = sv.BoxAnnotator(color=sv.Color(52, 236, 217))
    byte_tracker = sv.ByteTrack()

    # Initialize the Smoother
    smoother = sv.DetectionsSmoother()

    def render(detections, video_frame):
        # Parse the detections
        detections = sv.Detections.from_roboflow(detections)

        # Run a tracker to link predictions across frames
        detections = byte_tracker.update_with_detections(detections)

        # Record the new frame and get the smoothed predictions
        smoothed_detections = smoother.update_with_detections(detections)

        # Render
        image_smoothed = box_annotator.annotate(
            scene=image.copy(),
            detections=smoothed_detections
        )

        # Visualize
        cv2.imshow("Prediction", image)
        cv2.waitKey(1)


    pipeline = InferencePipeline.init(
        model_id="microsoft-coco/9", # Or put your custom trained model here
        # api_key="YOUR_ROBOFLOW_KEY", # Uncomment and fill if you want to access a
        #                                model that requires auth (or setup a .env file)
        video_reference=0, # Webcam; can also be video path or RTSP stream
        on_prediction=render
    )
    pipeline.start()
    pipeline.join()
    ```
    """

    def __init__(self, length: int = 5) -> None:
        """
        Args:
            length (int): The current count of detected objects within the zone
        """

        self.length = length

        self.current_frame = 0
        self.tracks = {}
        self.track_ends = {}

    def set_length(self, length: int) -> None:
        """
        Sets the number of frames to average out detections over.

        Args:
            length (int): The number of frames to average out detections over.
        """

        self.length = length
        for track_id in self.tracks:
            self.tracks[track_id] = deque(self.tracks[track_id], maxlen=length)

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Adds a new set of predictions to the smoother. Run this with every new
        prediction received from the model.

        Args:
            detections (Detections): The detections to add to the smoother.
        """

        self.current_frame += 1

        for detection_idx in range(len(detections)):
            tracker_id = detections.tracker_id[detection_idx]
            if tracker_id is None:
                # skip detections without a tracker id
                continue

            if self.tracks.get(tracker_id, None) is None:
                self.tracks[tracker_id] = deque(maxlen=self.length)

            self.tracks[tracker_id].append(detections[detection_idx])
            self.track_ends[tracker_id] = self.current_frame

        for track_id in self.tracks:
            track = self.tracks[track_id]
            if self.track_ends[track_id] < self.current_frame:
                # continue tracking for a few frames after the object has left
                # (to prevent flickering in case it comes back)
                track.append(None)

        return self.get_smoothed_detections()

    def get_track(self, track_id: int) -> Optional[Detections]:
        track = self.tracks.get(track_id, None)
        if track is None:
            return None

        track = [d for d in track if d is not None]
        if len(track) == 0:
            return None

        ret = track.copy()[0]
        ret.xyxy = np.mean([d.xyxy for d in track], axis=0)
        ret.confidence = np.mean([d.confidence for d in track], axis=0)

        return ret

    def get_smoothed_detections(self) -> Detections:
        """
        Returns a smoothed set of predictions based on the `length` most recent frames.

        Returns:
            detections (Detections): The smoothed detections.
        """

        tracked_detections = []
        for track_id in self.tracks:
            track = self.get_track(track_id)
            if track is not None:
                tracked_detections.append(track)

        return Detections.merge(tracked_detections)
