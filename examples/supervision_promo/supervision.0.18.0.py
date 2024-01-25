import cv2
import argparse
import supervision as sv
from ultralytics import YOLO


LINES_COORDINATES = [
    [sv.Point(x=1354, y=528), sv.Point(x=726, y=754)],
    [sv.Point(x=468, y=381), sv.Point(x=640, y=68)],
    [sv.Point(x=1412, y=366), sv.Point(x=1286, y=62)],
]

line_zones = [
    sv.LineZone(
        start=line_coordinates[0],
        end=line_coordinates[1],
        triggering_anchors=[sv.Position.BOTTOM_CENTER]
    )
    for line_coordinates
    in LINES_COORDINATES
]

line_annotations = [
    sv.LineZoneAnnotator(text_scale=0.7)
    for _ in LINES_COORDINATES
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervision 0.18.0 promo"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=args.confidence_threshold
    )

    thickness = 2
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)

    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=thickness,
        color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.CENTER,
        color_lookup=sv.ColorLookup.TRACK)

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            confidence_filter = detections.confidence > args.confidence_threshold
            class_filter = (detections.class_id == 2) | (detections.class_id == 7)

            detections = detections[confidence_filter & class_filter]
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            labels = [
                f"#{tracker_id} {confidence:.2f}"
                for tracker_id, confidence
                in zip(detections.tracker_id, detections.confidence)
            ]

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            for line_zone, line_annotation in zip(line_zones, line_annotations):
                line_zone.trigger(detections)
                annotated_frame = line_annotation.annotate(
                    frame=annotated_frame, line_counter=line_zone
                )

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()