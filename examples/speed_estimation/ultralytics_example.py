import argparse

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

COLOR = sv.Color.red()
ZONE = np.array([
    [1252,  787],
    [2298,  803],
    [5039, 2159],
    [-550, 2159]
])

polygon_zone = sv.PolygonZone(polygon=ZONE, frame_resolution_wh=(3840, 2160))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Supervision Package"
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

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    model = YOLO(args.source_weights_path)
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=args.confidence_threshold)

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
        thickness=thickness, color=COLOR, trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER)

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > args.confidence_threshold]
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        labels = [
            f"#{tracker_id}"
            for _, _, _, _, tracker_id
            in detections
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame,
            polygon=ZONE,
            color=COLOR,
            thickness=thickness)
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
