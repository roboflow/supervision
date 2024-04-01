from typing import Dict

import numpy as np
from supervision.detection.core import Detections


class MaxInstanceLimiter:
    """
    A utility class for limiting the number of instances in a video.


    Example:
        ```python
        import supervision as sv

        from ultralytics import YOLO

        video_info = sv.VideoInfo.from_video_path(video_path=<SOURCE_FILE_PATH>)
        frame_generator = sv.get_video_frames_generator(source_path=<SOURCE_FILE_PATH>)

        model = YOLO(<MODEL_PATH>)
        tracker = sv.ByteTrack(frame_rate=video_info.fps)
        smoother = sv.DetectionsSmoother()
        instance_limiter = MaxInstanceLimiter(max_instance_count=22, distance_threshold=45)

        annotator = sv.BoundingBoxAnnotator()

        with sv.VideoSink(<TARGET_FILE_PATH>, video_info=video_info) as sink:
            for frame in frame_generator:
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = tracker.update_with_detections(detections)
                detections = smoother.update_with_detections(detections)
                detections = instance_limiter.update_with_detections(detections)

                annotated_frame = bounding_box_annotator.annotate(frame.copy(), detections)
                sink.write_frame(annotated_frame)
        ```
    """

    def __init__(self, max_instance_count: int = 22, distance_threshold: float = 45.0):
        """
        Args:
            max_instance_count (int): The maximum allowed instances for all classes.
            distance_threshold (float): The maximum distance threshold to consider a missing tracker ID as a match.
                45 is considered a good value for video size: 1280x720, you should adjust this value accordingly
        """
        self.max_instance_count = max_instance_count
        self.distance_threshold = distance_threshold

        # Store the positions of all tracker IDs in all frames
        self.tracker_id_positions: Dict[int: np.ndarray] = {}
        # Store the replacement mapping for abnormal tracker IDs
        self.tracker_id_replacement: Dict[int: int] = {}

    def get_replacement_map(self) -> Dict[int, int]:
        """
        Returns the replacement mapping for abnormal tracker IDs.
        """
        return self.tracker_id_replacement

    @staticmethod
    def line_segment_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculates the distance between the midpoints of two line segments represented by two numpy arrays.
        """
        a_mid = (a[:2] + a[2:]) / 2
        b_mid = (b[:2] + b[2:]) / 2
        return np.linalg.norm(a_mid - b_mid)

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the instance limit with a new set of detections from a frame.

        Args:
            detections (Detections): The detections to add to the instance limit.

        Returns:
            Detections: The updated detections after applying the instance limit.
        """
        if detections.tracker_id is None:
            print(
                "Instance limit skipped. MaxInstanceLimiter requires tracker_id. Refer to "
                "https://supervision.roboflow.com/latest/trackers for more information."
            )
            return detections

        current_tracker_id_positions: Dict[int: np.ndarray] = {}

        # set current tracker_id positions
        existing_tracker_ids = set()
        abnormal_tracker_ids = set()
        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            xyxy = detections.xyxy[detection_idx]
            current_tracker_id_positions[tracker_id] = xyxy
            if tracker_id > self.max_instance_count:
                abnormal_tracker_ids.add(tracker_id)
            else:
                existing_tracker_ids.add(tracker_id)

        missing_tracker_ids = set(range(1, self.max_instance_count)) - existing_tracker_ids

        for tracker_id in self.tracker_id_replacement.keys():
            if tracker_id in abnormal_tracker_ids:
                abnormal_tracker_ids.remove(tracker_id)
        for tracker_id in self.tracker_id_replacement.values():
            if tracker_id in missing_tracker_ids:
                missing_tracker_ids.remove(tracker_id)

        # Replace abnormal tracker IDs with the closest missing tracker IDs
        for abnormal_tracker_id in abnormal_tracker_ids:
            closest_missing_tracker_id = None
            closest_distance = float('inf')
            for missing_tracker_id in missing_tracker_ids:
                if missing_tracker_id in self.tracker_id_positions.keys():
                    distance = self.line_segment_distance(
                        current_tracker_id_positions[abnormal_tracker_id], self.tracker_id_positions[
                            missing_tracker_id])
                    print(f'\t\t\t{abnormal_tracker_id} <---> {missing_tracker_id}: {distance}')
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_missing_tracker_id = missing_tracker_id
            if closest_missing_tracker_id is not None and closest_distance < self.distance_threshold:
                self.tracker_id_replacement[abnormal_tracker_id] = closest_missing_tracker_id
                missing_tracker_ids.remove(closest_missing_tracker_id)

        # Update tracker ID positions
        for tracker_id in current_tracker_id_positions.keys():
            if tracker_id <= self.max_instance_count:
                self.tracker_id_positions[tracker_id] = current_tracker_id_positions[tracker_id]

        # Update or remove abnormal tracker IDs in detections
        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            if tracker_id is None:
                continue

            if tracker_id in self.tracker_id_replacement.keys():
                detections.tracker_id[detection_idx] = self.tracker_id_replacement[tracker_id]
            elif tracker_id > self.max_instance_count:
                detections.tracker_id[detection_idx] = -1

        # remove detections with tracker_id = -1
        detections = detections[detections.tracker_id != -1]

        return detections
