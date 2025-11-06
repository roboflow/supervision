from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class CameraMotionCompensator:
    """
    A class for camera motion compensation, designed to be used with object trackers.

    This class supports two modes of operation:

    1.  Simple Mode: If a `tracker` is provided at initialization,
        `update_with_detections` can be used as a one-step method to get
        compensated tracking results. Ideal for single-tracker scenarios.

    2.  Advanced Mode: If no `tracker` is provided, the user can
        call `update`, `compensate`, and `revert` manually. This is efficient
        for scenarios with multiple trackers (potentially from multiple models),
        as motion is calculated only once per frame and can be reused.

    Example (Simple Mode):
        ```python
        import supervision as sv
        from ultralytics import YOLO

        video_info = sv.VideoInfo.from_video_path(<SOURCE_FILE_PATH>)
        frame_generator = sv.get_video_frames_generator(source_path=<SOURCE_FILE_PATH>)
        model = YOLO(<MODEL_PATH>)
        tracker = sv.ByteTrack()
        cmc = sv.CameraMotionCompensator(tracker=tracker)

        with sv.VideoSink(<TARGET_FILE_PATH>, video_info=video_info) as sink:
            for frame in frame_generator:
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                tracked_detections = cmc.update_with_detections(detections, frame=frame)
                # annotate and save frame
                ...
        ```

    Example (Advanced Mode with Multiple Models and Trackers):
        ```python
        import supervision as sv
        from ultralytics import YOLO

        video_info = sv.VideoInfo.from_video_path(<SOURCE_FILE_PATH>)
        frame_generator = sv.get_video_frames_generator(source_path=<SOURCE_FILE_PATH>)

        person_model = YOLO(<MODEL_PATH>)
        vehicle_model = YOLO(<MODEL_PATH>)

        cmc = sv.CameraMotionCompensator()
        person_tracker = sv.ByteTrack()
        vehicle_tracker = sv.ByteTrack()

        with sv.VideoSink(<TARGET_FILE_PATH>, video_info=video_info) as sink:
            for frame in frame_generator:
                # Calculate motion once per frame
                cmc.update(frame)

                # Process persons
                person_results = person_model(frame)[0]
                person_detections = sv.Detections.from_ultralytics(person_results)
                comp_persons = cmc.compensate(person_detections)
                tracked_persons = person_tracker.update_with_detections(comp_persons)
                final_persons = cmc.revert(tracked_persons)

                # Process vehicles (reusing the same motion calculation)
                vehicle_results = vehicle_model(frame)[0]
                vehicle_detections = sv.Detections.from_ultralytics(vehicle_results)
                comp_vehicles = cmc.compensate(vehicle_detections)
                tracked_vehicles = vehicle_tracker.update_with_detections(comp_vehicles)
                final_vehicles = cmc.revert(tracked_vehicles)

                # Annotate frame with both person and vehicle tracks
                ...
        ```
    """

    def __init__(self, tracker: sv.Tracker | None = None):
        """
        Args:
            tracker (sv.Tracker, optional): The tracker to be wrapped for simple,
                one-step usage. Defaults to None.
        """
        self.tracker = tracker
        self.previous_frame: np.ndarray | None = None
        self.motion_matrix: np.ndarray | None = None

    def reset(self) -> None:
        """
        Resets the internal state of the compensator and the wrapped tracker, if any.
        """
        self.previous_frame = None
        self.motion_matrix = None
        if self.tracker:
            self.tracker.reset()

    @staticmethod
    def _calculate_motion_matrix(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray | None:
        """
        Calculates the motion between two consecutive frames using feature matching.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        if descriptors1 is None or descriptors2 is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)
            return matrix

        return None

    @staticmethod
    def _transform_detections(detections: sv.Detections, matrix: np.ndarray) -> sv.Detections:
        """
        Applies an affine transformation matrix to the bounding boxes of a Detections object.
        """
        if matrix is None or len(detections.xyxy) == 0:
            return detections

        points = detections.xyxy.reshape(-1, 2)
        points_to_transform = np.float32(points).reshape(-1, 1, 2)
        points_transformed = cv2.transform(points_to_transform, matrix)
        new_xyxy = points_transformed.reshape(-1, 4)

        return sv.Detections(
            xyxy=new_xyxy,
            mask=detections.mask.copy() if detections.mask is not None else None,
            confidence=detections.confidence.copy() if detections.confidence is not None else None,
            class_id=detections.class_id.copy() if detections.class_id is not None else None,
            tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
            data={k: v.copy() for k, v in detections.data.items()},
        )

    def update(self, frame: np.ndarray) -> None:
        """
        Updates the compensator with the current frame to calculate camera motion.
        This method should be called once per frame in Decoupled Mode.

        Args:
            frame (np.ndarray): The current video frame.
        """
        self.motion_matrix = None
        if self.previous_frame is not None:
            self.motion_matrix = self._calculate_motion_matrix(self.previous_frame, frame)
        self.previous_frame = frame.copy()

    def compensate(self, detections: sv.Detections) -> sv.Detections:
        """
        Applies inverse motion transformation to detections (for Decoupled Mode).

        Args:
            detections (sv.Detections): The detections to compensate.

        Returns:
            sv.Detections: The compensated detections.
        """
        if self.motion_matrix is None:
            return detections

        try:
            inverse_motion_matrix = cv2.invertAffineTransform(self.motion_matrix)
            return self._transform_detections(detections, inverse_motion_matrix)
        except Exception as e:
            print(f"Warning: Could not invert motion matrix. Error: {e}")
            return detections

    def revert(self, detections: sv.Detections) -> sv.Detections:
        """
        Applies forward motion transformation to detections (for Decoupled Mode).

        Args:
            detections (sv.Detections): The detections to revert.

        Returns:
            sv.Detections: The reverted detections in the current frame's coordinates.
        """
        if self.motion_matrix is None:
            return detections
        return self._transform_detections(detections, self.motion_matrix)

    def update_with_detections(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """
        A convenience method for the simple case of a single, wrapped tracker.
        Requires a tracker to have been provided during initialization.

        Args:
            detections (sv.Detections): The detections from the current frame.
            frame (np.ndarray): The current video frame.

        Returns:
            sv.Detections: The final, compensated tracked detections.

        Raises:
            ValueError: If the compensator was not initialized with a tracker.
        """
        if not self.tracker:
            raise ValueError(
                "A tracker must be provided during initialization to use "
                "update_with_detections. For multi-tracker scenarios, use the "
                "update(), compensate(), and revert() methods manually."
            )

        self.update(frame)
        compensated_detections = self.compensate(detections)
        tracked_detections = self.tracker.update_with_detections(compensated_detections)
        final_detections = self.revert(tracked_detections)
        return final_detections
