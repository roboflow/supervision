import argparse
from typing import Tuple, List

import cv2
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np

from supervision import Color, Position, draw_polygon, ColorPalette
from supervision.detection.annotate import TraceAnnotator

DEFAULT_COLOR = Color.white()
COLORS = ColorPalette.default()

POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]])
]

DEFAULT_BOX_ANNOTATOR = sv.BoxAnnotator(color=DEFAULT_COLOR)


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: Position = Position.CENTER,
) -> Tuple[List[sv.PolygonZone], List[sv.PolygonZoneAnnotator], List[sv.BoxAnnotator]]:
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]
    zone_annotators = [
        sv.PolygonZoneAnnotator(zone=zones[i], color=COLORS.colors[i], text_scale=1)
        for i
        in range(len(zones))
    ]
    box_annotators = [
        sv.BoxAnnotator(color=COLORS.colors[i])
        for i
        in range(len(zones))
    ]
    return zones, zone_annotators, box_annotators


def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    model = YOLO(source_weights_path)
    tracker = sv.ByteTrack()
    trace_annotator = TraceAnnotator(color=DEFAULT_COLOR)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=2)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    zones, zone_annotators, box_annotators = initiate_polygon_zones(
        polygons=POLYGONS,
        frame_resolution_wh=video_info.resolution_wh,
        triggering_position=Position.CENTER
    )

    if target_video_path:
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames // 2):
                annotated_frame = process_frame(
                    frame, model, tracker, confidence_threshold, iou_threshold, zones,
                    zone_annotators, box_annotators, trace_annotator
                )
                sink.write_frame(frame=annotated_frame)
    else:
        for frame in tqdm(frame_generator, total=video_info.total_frames // 2):
            annotated_frame = process_frame(
                frame, model, tracker, confidence_threshold, iou_threshold, zones,
                zone_annotators, box_annotators, trace_annotator
            )
            cv2.imshow('Processed Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    tracker: sv.ByteTrack,
    confidence_threshold: float,
    iou_threshold: float,
    zones: List[sv.PolygonZone],
    zone_annotators: List[sv.PolygonZoneAnnotator],
    box_annotators: List[sv.BoxAnnotator],
    trace_annotator: TraceAnnotator
) -> np.ndarray:
    results = model(
        frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    detections.class_id = np.zeros(len(detections))
    detections = tracker.update_with_detections(detections)

    annotated_frame = frame.copy()

    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections)

    for i, zone in enumerate(zones):
        is_in_zone = zone.trigger(detections=detections)
        detections_in_zone = detections[is_in_zone]
        detections = detections[~is_in_zone]
        labels = [
            f"#{tracker_id}"
            for _, _, confidence, class_id, tracker_id in detections_in_zone
        ]
        annotated_frame = box_annotators[i].annotate(
            scene=annotated_frame, detections=detections_in_zone, labels=labels)
        annotated_frame = zone_annotators[i].annotate(scene=annotated_frame)
    labels = [
        f"#{tracker_id}"
        for _, _, confidence, class_id, tracker_id in detections
    ]
    annotated_frame = DEFAULT_BOX_ANNOTATOR.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack")

    parser.add_argument(
        "--source_weights_path", required=True,
        help="Path to the source weights file", type=str
    )
    parser.add_argument(
        "--source_video_path", required=True,
        help="Path to the source video file", type=str
    )
    parser.add_argument(
        "--target_video_path", default=None,
        help="Path to the target video file (output)", type=str
    )
    parser.add_argument(
        "--confidence_threshold", default=0.3,
        help="Confidence threshold for the model", type=float
    )
    parser.add_argument(
        "--iou_threshold", default=0.7,
        help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
