import argparse
import json
from datetime import datetime
from typing import Generator, Dict, List

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model

import supervision as sv


def load_zones_config(file_path: str) -> List[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. Each polygon is represented as
    a NumPy array of coordinates.

    Args:
    file_path (str): The path to the JSON configuration file.

    Returns:
    List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counting time duration in zones with YOLOv8 and Supervision"
    )
    parser.add_argument(
        "--zone_configuration_path",
        required=True,
        help="Path to the zone configuration JSON file",
        type=str,
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Index of the webcam to use"
    )
    parser.add_argument(
        "--model_id",
        default="yolov8m-640",
        help="Roboflow model ID",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu", "mps"],
        help="Computing device for processing: cuda, cpu, or mps"
    )

    return parser.parse_args()


class Timer:
    """
    A class for tracking and updating time durations associated with object detections.

    Attributes:
        tracker_id2start_time (Dict[int, datetime]): A dictionary mapping tracker IDs
            to the datetime when they were first detected.
    """
    def __init__(self) -> None:
        self.tracker_id2start_time: Dict[int, datetime] = {}

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates detection times based on current detections.

        Args:
            detections (sv.Detections): The current detections with tracker IDs.

        Returns:
            sv.Detections: The updated detections object including the time duration
            each object has been detected.
        """
        current_time = datetime.now()
        times = []

        for tracker_id in detections.tracker_id:
            if tracker_id not in self.tracker_id2start_time:
                self.tracker_id2start_time[tracker_id] = current_time

            start_time = self.tracker_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        detections['time'] = np.array(times)
        return detections


def get_webcam_frames_generator(
    camera_index: int = 0
) -> Generator[np.ndarray, None, None]:
    """
    Get a generator that yields frames from the specified webcam.

    Args:
        camera_index (int): Index of the webcam to use.

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

    cap.release()


class Annotator:

    def __init__(self, thickness: int = 1, text_scale: float = 1.0):
        self.box_annotator = sv.BoundingBoxAnnotator(
            thickness=thickness)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=50,
            position=sv.Position.CENTER)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness)

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        labels = [
            f"#{tracker_id} {time:.2f}s"
            for tracker_id, time
            in zip(
                detections.tracker_id,
                detections["time"]
            )
        ]
        annotated_frame = self.trace_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame


def main():
    args = parse_arguments()
    model = get_roboflow_model(model_id=args.model_id)

    frames_generator = get_webcam_frames_generator(args.camera_index)
    frame = next(frames_generator)
    h, w, _ = frame.shape

    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=(w, h))
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(w, h))
    annotator = Annotator(thickness=thickness, text_scale=text_scale)

    fps_monitor = sv.FPSMonitor()
    tracker = sv.ByteTrack()

    polygons = load_zones_config(args.zone_configuration_path)
    zones = [
        sv.PolygonZone(polygon, frame_resolution_wh=(w, h))
        for polygon
        in polygons
    ]
    timers = [Timer() for _ in zones]

    for frame in frames_generator:

        fps_monitor.tick()
        fps = fps_monitor.fps

        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(
            threshold=args.iou_threshold, class_agnostic=True)
        detections = detections[detections.confidence > args.confidence_threshold]

        # remove detections of class 0 (person)
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
                color=sv.ColorPalette.LEGACY.by_idx(i),
                thickness=thickness
            )

            detections_in_zone = detections[zone.trigger(detections)]
            detections_in_zone = timers[i].update_with_detections(detections_in_zone)

            annotated_frame = annotator.annotate(
                frame=annotated_frame,
                detections=detections_in_zone
            )

        cv2.imshow("Webcam Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
