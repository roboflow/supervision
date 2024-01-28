import numpy as np
from ultralytics import YOLO

import supervision as sv

model = YOLO("")
# tracker = sv.ByteTrack()

tracker = sv.load_strong_sort()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    # detections = tracker.update_with_detections(detections)
    detections = tracker.update_with_detections(detections, frame)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )
    return annotated_frame


sv.process_video(
    source_path="./test_videos/input.mp4",
    target_path="./test_videos/out2.mp4",
    callback=callback,
)
