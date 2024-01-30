import cv2
import numpy as np
from typing import Generator, Dict, Optional
import argparse
import supervision as sv
# from ultralytics import YOLO
from inference.models.utils import get_roboflow_model
from datetime import datetime


ZONES_COORDINATES = [
    np.array([
        [100, 100],
        [910, 100],
        [910, 980],
        [100, 980]
    ]),
    np.array([
        [1010, 100],
        [1820, 100],
        [1820, 980],
        [1010, 980]
    ])
]


class Timer:

    def __init__(self) -> None:
        self.tracker_id2start_time: Dict[int, datetime] = {}

    def tick(self, detections: sv.Detections) -> None:
        current_time = datetime.now()
        for tracker_id in detections.tracker_id:
            self.tracker_id2start_time.setdefault(tracker_id, current_time)

    def get_time_by_tracker_id(self, tracker_id: int) -> Optional[float]:
        start_time = self.tracker_id2start_time.get(tracker_id, None)
        if start_time is None:
            return None
        return (datetime.now() - start_time).total_seconds()

    def get_time(self, detections: sv.Detections) -> np.ndarray:
        return np.array([
            self.get_time_by_tracker_id(tracker_id)
            for tracker_id
            in detections.tracker_id
        ])

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        self.tick(detections)
        detections['time'] = self.get_time(detections)
        return detections


def get_webcam_frames_generator(
    camera_index: int = 0,
    stride: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Get a generator that yields frames from the specified webcam.

    Args:
        camera_index (int): Index of the webcam to use.
        stride (int): Indicates the interval at which frames are returned,
            skipping stride - 1 frames between each.

    Returns:
        (Generator[np.ndarray, None, None]): A generator that yields the
            frames from the specified webcam.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise Exception(f"Error: Could not open webcam with index {camera_index}.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        yield frame

        for _ in range(stride - 1):
            success = cap.grab()
            if not success:
                break

    cap.release()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counting time duration in zones with YOLOv8 and Supervision"
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Index of the webcam to use"
    )
    parser.add_argument(
        "--source_weights_path",
        default="yolov8m.pt",
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    # model = YOLO(args.source_weights_path)
    model = get_roboflow_model("yolov8m-640")
    fps_monitor = sv.FPSMonitor()

    resolution_wh = (1920, 1080)

    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wh)

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    tracker = sv.ByteTrack()

    zones = [
        sv.PolygonZone(polygon, frame_resolution_wh=resolution_wh)
        for polygon
        in ZONES_COORDINATES
    ]

    timers = [Timer() for _ in zones]

    for frame in get_webcam_frames_generator(args.camera_index):

        fps_monitor.tick()
        fps = fps_monitor()

        # results = model(frame, verbose=False, conf=args.confidence_threshold, device='mps')[0]
        # detections = sv.Detections.from_ultralytics(results)

        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        detections = detections[detections.class_id != 0]
        detections = tracker.update_with_detections(detections=detections)

        annotated_frame = frame.copy()

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"FPS {fps:.2f}",
            text_anchor=sv.Point(110, 50),
            text_scale=text_scale,
            text_thickness=thickness,
            text_color=sv.Color.RED
        )

        for i, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone.polygon,
                color=sv.ColorPalette.LEGACY.by_idx(i + 1),
                thickness=thickness
            )

            detections_in_zone = detections[zone.trigger(detections)]
            detections_in_zone = timers[i].update_with_detections(detections_in_zone)

            labels = [
                f"#{tracker_id} {results.names[class_id]} {time:.2f}s"
                for class_id, tracker_id, time
                in zip(
                    detections_in_zone.class_id,
                    detections_in_zone.tracker_id,
                    detections_in_zone["time"]
                )
            ]

            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame,
                detections=detections_in_zone
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels
            )

        cv2.imshow("Webcam Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
