import sys

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv


def main(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """
    Video Processing with YOLO and ByteTrack.

    Args:
        source_weights_path: Path to the source weights file
        source_video_path: Path to the source video file
        target_video_path: Path to the target video file (output)
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
    """
    model = YOLO(source_weights_path)

    tracker = sv.ByteTrack()
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
            detections = tracker.update_with_detections(detections)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    try:
        # Try to import jsonargparse for CLI parsing
        from jsonargparse import ArgumentParser
    except ImportError:
        # Fallback if jsonargparse is not installed
        print("Warning: jsonargparse not installed. Using plain positional arguments.")
        if len(sys.argv) < 4:
            raise ValueError("Insufficient arguments provided."
                             "Usage: python ultralytics_example.py "
                             "<source_weights_path> <source_video_path> <target_video_path> "
                             "[confidence_threshold] [iou_threshold]")
        main(
            source_weights_path=sys.argv[1],
            source_video_path=sys.argv[2],
            target_video_path=sys.argv[3],
            confidence_threshold=float(sys.argv[4]) if len(sys.argv) > 4 else 0.3,
            iou_threshold=float(sys.argv[5]) if len(sys.argv) > 5 else 0.7,
        )
    else:
        # Use jsonargparse for automatic CLI if import succeeded
        parser = ArgumentParser()
        parser.add_function_arguments(main)
        args = parser.parse_args()
        main(**vars(args))
