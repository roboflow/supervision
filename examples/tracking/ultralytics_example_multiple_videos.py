import argparse

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv


def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    tracker: sv.ByteTrack,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    model = YOLO(source_weights_path)

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            labels = [
                    f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, tracker_id, data
                in detections
            ]

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotated_labeled_frame)
    # Reset the tracker after processing the video
    tracker.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack"
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_paths",
        required=True,
        help="Paths to the source video files",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--target_video_paths",
        required=True,
        help="Paths to the target video files (output)",
        nargs="+",
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

    source_video_paths = args.source_video_paths[0].split(',')
    target_video_paths = args.target_video_paths[0].split(',')

    source_video_paths = [path.strip() for path in source_video_paths]
    target_video_paths = [path.strip() for path in target_video_paths]
    tracker = sv.ByteTrack()

    for source_video_path, target_video_path in zip(source_video_paths, target_video_paths):
        process_video(
            source_weights_path=args.source_weights_path,
            source_video_path=source_video_path,
            target_video_path=target_video_path,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            tracker=tracker,
        )