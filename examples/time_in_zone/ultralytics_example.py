import argparse
import json
from typing import List, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


COLORS = sv.ColorPalette.default()


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


class Timer:

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.frame_id = 0
        self.tacker_id2frame_id: Dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> None:
        self.frame_id += 1
        for tracker_id in detections.tracker_id:
            self.tacker_id2frame_id.setdefault(tracker_id, self.frame_id)

    def get_time_by_tracker_id(self, tracker_id: int) -> Optional[float]:
        start_frame_id = self.tacker_id2frame_id.get(tracker_id, None)
        if start_frame_id is None:
            return None
        return (self.frame_id - start_frame_id) / self.fps

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


if __name__ == "__main__":
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
        "--source_weights_path",
        default="yolov8x.pt",
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )

    args = parser.parse_args()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(
        resolution_wh=video_info.resolution_wh)

    zone_config = load_zones_config(args.zone_configuration_path)
    zones = [
        sv.PolygonZone(polygon, frame_resolution_wh=video_info.resolution_wh)
        for polygon
        in zone_config
    ]
    box_annotators = [
        sv.BoundingBoxAnnotator(
            color=COLORS.by_idx(i + 1),
            thickness=thickness,
        )
        for i
        in range(len(zones))
    ]
    label_annotators = [
        sv.LabelAnnotator(
            color=COLORS.by_idx(i + 1),
            text_scale=text_scale,
            text_thickness=thickness,
        )
        for i
        in range(len(zones))
    ]
    trace_annotators = [
        sv.TraceAnnotator(
            color=COLORS.by_idx(i + 1),
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER
        )
        for i
        in range(len(zones))
    ]
    timers = [
        Timer(fps=video_info.fps)
        for i
        in range(len(zones))
    ]

    model = YOLO(args.source_weights_path)
    tracker = sv.ByteTrack(frame_rate=video_info.fps)

    frames_generator = sv.get_video_frames_generator(args.source_video_path, start=3600, end=4200)

    with sv.VideoSink("data/output-2.mp4", video_info) as sink:
        for frame in tqdm(frames_generator, total=video_info.total_frames):

            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.confidence > args.confidence_threshold]

            in_zone_0 = zones[0].trigger(detections=detections)
            in_zone_1 = zones[1].trigger(detections=detections)
            in_zone_2 = zones[2].trigger(detections=detections)

            detections = detections[in_zone_0 | in_zone_1 | in_zone_2]
            detections = detections.with_nms(threshold=0.9)
            detections = tracker.update_with_detections(detections)

            annotated_frame = frame.copy()

            for i, zone in enumerate(zones):
                detections_in_zone = detections[zones[i].trigger(detections=detections)]
                detections_in_zone = timers[i].update_with_detections(detections_in_zone)
                labels = [
                    f"#{tracker_id} {time:.1f}s {model.model.names[class_id]}" if time is not None else f"#{tracker_id} N/A"
                    for time, tracker_id, class_id
                    in zip(detections_in_zone['time'], detections_in_zone.tracker_id, detections_in_zone.class_id)
                ]
                annotated_frame = sv.draw_polygon(
                    annotated_frame, zone.polygon, color=COLORS.by_idx(i + 1), thickness=thickness)
                annotated_frame = trace_annotators[i].annotate(
                    annotated_frame, detections_in_zone)
                annotated_frame = box_annotators[i].annotate(
                    annotated_frame, detections_in_zone)
                annotated_frame = label_annotators[i].annotate(
                    annotated_frame, detections_in_zone, labels=labels)

            sink.write_frame(annotated_frame)
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
