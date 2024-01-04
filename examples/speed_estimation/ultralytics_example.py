import argparse
import json
from dataclasses import dataclass
from typing import List

import cv2
from ultralytics import YOLO

import supervision as sv


COLOR = sv.Color.red()


@dataclass
class Line:
    start: sv.Point
    end: sv.Point


@dataclass
class Config:
    lines: List[Line]
    distance: float

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            lines = [
                Line(sv.Point(**line['start']), sv.Point(**line['end']))
                for line
                in data['lines']
            ]
            return cls(lines, data['distance'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Supervision Package"
    )

    parser.add_argument(
        "--lines_configuration_path",
        required=True,
        help="Path to the lines configuration JSON file",
        type=str,
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
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
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float
    )

    args = parser.parse_args()

    config = Config.load(args.lines_configuration_path)
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    model = YOLO(args.source_weights_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(
        resolution_wh=video_info.resolution_wh)

    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=thickness, color=COLOR)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, text_thickness=thickness, color=COLOR,
        text_position=sv.Position.BOTTOM_CENTER)
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness, color=COLOR, trace_length=video_info.fps,
        position=sv.Position.BOTTOM_CENTER)

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections)

        labels = [
            f"#{tracker_id}"
            for _, _, _, _, tracker_id
            in detections
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_line(
            scene=annotated_frame,
            start=config.lines[0].start,
            end=config.lines[0].end,
            color=sv.Color.white(),
            thickness=2)
        annotated_frame = sv.draw_line(
            scene=annotated_frame,
            start=config.lines[1].start,
            end=config.lines[1].end,
            color=sv.Color.white(),
            thickness=2)
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections)
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
