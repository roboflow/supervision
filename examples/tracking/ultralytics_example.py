from tqdm import tqdm
from trackers import ByteTrackTracker
from ultralytics import YOLO

import supervision as sv


def main(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """Video Processing with YOLO and ByteTrackTracker.

    Args:
        source_weights_path: Path to the source weights file
        source_video_path: Path to the source video file
        target_video_path: Path to the target video file (output)
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
    """
    model = YOLO(source_weights_path)

    tracker = ByteTrackTracker(track_activation_threshold=confidence_threshold)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update(detections)
            detections = detections[detections.tracker_id != -1]  # -1 = pending track

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
