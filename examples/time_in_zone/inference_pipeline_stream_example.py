import argparse
import json
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


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
        return [np.array(polygon, np.int32) for polygon in data]


def find_in_list(array: np.ndarray, search_list: List[int]) -> np.ndarray:
    """Determines if elements of a numpy array are present in a list.

    Args:
        array (np.ndarray): The numpy array of integers to check.
        search_list (List[int]): The list of integers to search within.

    Returns:
        np.ndarray: A numpy array of booleans, where each boolean indicates whether
        the corresponding element in `array` is found in `search_list`.
    """
    if not search_list:
        return np.ones(array.shape, dtype=bool)
    else:
        return np.isin(array, search_list)


class ClockBasedTimer:
    def __init__(self) -> None:
        self.tracker_id2start_time: Dict[int, datetime] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        current_time = datetime.now()
        times = []

        for tracker_id in detections.tracker_id:
            self.tracker_id2start_time.setdefault(tracker_id, current_time)

            start_time = self.tracker_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        return np.array(times)


class XD:
    def __init__(self, zone_configuration_path: str, classes: List[int]):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = None

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        if self.zones is None:
            resolution_wh = frame.image.shape[1], frame.image.shape[0]
            self.zones = [
                sv.PolygonZone(
                    polygon=polygon,
                    frame_resolution_wh=resolution_wh,
                    triggering_anchors=(sv.Position.CENTER,),
                )
                for polygon in self.polygons
            ]

        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = sv.Detections.from_inference(result)
        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )
        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)


def main(
    rtsp_url: str,
    zone_configuration_path: str,
    model_id: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    xd = XD(zone_configuration_path=zone_configuration_path, classes=classes)

    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=rtsp_url,
        on_prediction=xd.on_prediction,
        confidence=confidence,
        iou_threshold=iou,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using RTSP stream."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        required=True,
        help="Complete RTSP URL for the video stream.",
    )
    parser.add_argument(
        "--model_id", type=str, default="yolov8s-640", help="Roboflow model ID."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        rtsp_url=args.rtsp_url,
        zone_configuration_path=args.zone_configuration_path,
        model_id=args.model_id,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
