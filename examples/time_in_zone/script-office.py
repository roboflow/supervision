import argparse
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv


ZONE_POLYGONS = [
    np.array([[418, 158], [622, 226], [414, 406], [186, 230]])
]

ZONE_COLORS = [
    sv.ColorPalette.default().by_idx(1)
]

DETECTIONS_COLORS = [
    sv.ColorPalette.default().by_idx(0),
    sv.ColorPalette.default().by_idx(1)
]


class Timer:

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.frame_id = 0
        self.tacker_id2frame_id: Dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> None:
        self.frame_id += 1
        for tracker_id in detections.tracker_id:
            self.tacker_id2frame_id.setdefault(tracker_id, self.frame_id)

    def get_time(self, zone_id: int) -> Optional[float]:
        start_frame_id = self.tacker_id2frame_id.get(zone_id, None)
        if start_frame_id is None:
            return None
        return (self.frame_id - start_frame_id) / self.fps


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


def initiate_timers(count: int, fps: int) -> List[Timer]:
    return [Timer(fps=fps) for _ in range(count)]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.tracker = sv.ByteTrack(frame_rate=self.video_info.fps)
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path)
        self.frame_generator = tqdm(frame_generator, total=self.video_info.total_frames)
        self.box_corner_annotation = sv.BoxCornerAnnotator(thickness=2, corner_length=10)
        # self.blur_annotator = sv.BlurAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.CENTER)

        self.zones = initiate_polygon_zones(
            ZONE_POLYGONS, self.video_info.resolution_wh, sv.Position.BOTTOM_CENTER
        )
        self.timers = initiate_timers(len(self.zones), self.video_info.fps)

        self.custom_color_lookup = Optional[np.ndarray]

    def process_video(self):
        print(self.video_info)
        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in self.frame_generator:
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in self.frame_generator:
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = self.tracker.update_with_detections(detections=detections)

        custom_color_lookup = np.zeros(len(detections), dtype=int)
        for index, zone in enumerate(self.zones, start=1):
            in_zone = zone.trigger(detections=detections)
            detections_in_zone = detections[in_zone]
            custom_color_lookup[in_zone] = index
            self.timers[index - 1].tick(detections=detections_in_zone)
        self.custom_color_lookup = custom_color_lookup

        return self.annotate_frame(frame, detections)

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for color, zone in zip(ZONE_COLORS, self.zones):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone.polygon, color)
        # annotated_frame = self.blur_annotator.annotate(
        #     scene=annotated_frame, detections=detections)
        annotated_frame = self.box_corner_annotation.annotate(
            scene=annotated_frame, detections=detections,
            custom_color_lookup=self.custom_color_lookup)
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=detections,
            custom_color_lookup=self.custom_color_lookup)
        labels = [
            f"#{tracker_id}" if zone_index == 0 else f"{self.timers[zone_index - 1].get_time(tracker_id)}s"
            for tracker_id, zone_index
            in zip(detections.tracker_id, self.custom_color_lookup)
        ]
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels,
            custom_color_lookup=self.custom_color_lookup)
        return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time in Zone Analysis with YOLO and ByteTrack"
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
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
