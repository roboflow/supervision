import sys
sys.path.append('/Users/skalskip/Documents/work/inference/')

import argparse
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv

ZONE_POLYGONS = [
    np.array([[956, 742], [104, 757], [328, 330], [944, 310]]),
    np.array([[956, 742], [944, 310], [1560, 300], [1810, 724]]),
]

ZONE_COLORS = [
    sv.ColorPalette.default().by_idx(1),
    sv.ColorPalette.default().by_idx(2)
]

DETECTIONS_COLORS = [
    sv.ColorPalette.default().by_idx(0),
    sv.ColorPalette.default().by_idx(1),
    sv.ColorPalette.default().by_idx(2)
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
        model_id: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = get_roboflow_model(model_id=model_id)

        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path)
        self.frame_generator = tqdm(frame_generator, total=self.video_info.total_frames)
        self.ellipse_annotator = sv.EllipseAnnotator()
        self.trace_annotator = sv.TraceAnnotator(trace_length=100)
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.BOTTOM_CENTER)

        self.zones = initiate_polygon_zones(
            ZONE_POLYGONS, self.video_info.resolution_wh, sv.Position.BOTTOM_CENTER
        )
        self.timers = initiate_timers(len(self.zones), self.video_info.fps)

        self.custom_color_lookup = Optional[np.ndarray]

    def process_video(self):
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.infer(
            frame_rgb, confidence=self.conf_threshold, iou_threshold=self.iou_threshold)[0]

        detections = sv.Detections.from_roboflow(
            result.dict(by_alias=True, exclude_none=True)
        )
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
        annotated_frame = self.ellipse_annotator.annotate(
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
        "--model_id",
        default="coco/5",
        help="Model ID from Roboflow Universe",
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
        model_id=args.model_id,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
