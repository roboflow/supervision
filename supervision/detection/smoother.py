from collections import defaultdict
import numpy as np
from supervision.detection.core import Detections

class Smoother:
    """
    A class for smoothing out noise in predictions over time by using a Tracker
    to track objects over time and averaging out the predictions over the
    `length` most recent frames.

    !!! warning

        Smoother utilizes the `tracker_id`. Read
        [here](https://supervision.roboflow.com/trackers/) to learn how to plug
        tracking into your inference pipeline.
    """

    def __init__(
        self,
        length: int = 5
    ) -> None:
        """
        Args:
            length (int): The current count of detected objects within the zone
        """

        self.length = length

        self.current_frame = 0
        self.tracks = NoneDict()
        self.track_starts = NoneDict()
        self.track_ends = NoneDict()

    def tracker_length(self, tracker_id):
        return self.current_frame - self.track_starts[tracker_id]

    def add_frame(self, detections: Detections) -> None:
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

            if self.tracks[tracker_id] is None:
                # initialize a new tracker_id
                self.tracks[tracker_id] = []
                self.track_starts[tracker_id] = self.current_frame
            
            self.tracks[tracker_id].append(detections[detection_idx])
            self.track_ends[tracker_id] = self.current_frame
        
        for track_id in self.tracks:
            track = self.tracks[track_id]
            if self.track_ends[track_id] < self.current_frame:
                # continue tracking for a few frames after the object has left
                # (to prevent flickering in case it comes back)
                track.append(None)
            
            if len(track) > self.length:
                # remove the oldest detection from the track it's too long
                track.pop(0)
    
    def get_track(self, track_id):
        track = self.tracks[track_id]
        if track is None:
            return None

        track = [d for d in track if d is not None]
        if len(track) == 0:
            return None
        
        ret = track[0]
        # set to an average of all the detection boxes
        ret.xyxy = np.mean([d.xyxy for d in track], axis=0)
        ret.confidence = np.mean([d.confidence for d in track], axis=0)
        
        return ret

    def get_smoothed_detections(self):
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

class NoneDict(defaultdict):
    """
    Helper class that returns None instead of raising a KeyError
    when a key is not found.
    """

    def __init__(self, *args, **kwargs):
        super(NoneDict, self).__init__(None, *args, **kwargs)

    def __getitem__(self, key):
        try:
            return super(NoneDict, self).__getitem__(key)
        except KeyError:
            return None