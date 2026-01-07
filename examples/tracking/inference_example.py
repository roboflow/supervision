import os
import sys

from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv


def main(
    source_video_path: str,
    target_video_path: str,
    roboflow_api_key: str,
    model_id: str = "yolov8x-1280",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """
    Video Processing with Inference and ByteTrack.

    Args:
        source_video_path: Path to the source video file
        target_video_path: Path to the target video file (output)
        roboflow_api_key: Roboflow API key
        model_id: Roboflow model ID
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", roboflow_api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )

    model = get_roboflow_model(model_id=model_id, api_key=api_key)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model.infer(
                frame, confidence=confidence_threshold, iou_threshold=iou_threshold
            )[0]
            detections = sv.Detections.from_inference(results)
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
            raise ValueError(
                "Insufficient arguments provided."
                "Usage: python inference_example.py "
                "<source_video_path> <target_video_path> <roboflow_api_key> "
                "[model_id] [confidence_threshold] [iou_threshold]"
            )
        main(
            source_video_path=sys.argv[1],
            target_video_path=sys.argv[2],
            roboflow_api_key=sys.argv[3],
            model_id=sys.argv[4] if len(sys.argv) > 4 else "yolov8x-1280",
            confidence_threshold=float(sys.argv[5]) if len(sys.argv) > 5 else 0.3,
            iou_threshold=float(sys.argv[6]) if len(sys.argv) > 6 else 0.7,
        )
    else:
        # Use jsonargparse for automatic CLI if import succeeded
        parser = ArgumentParser()
        parser.add_function_arguments(main)
        args = parser.parse_args()
        main(**vars(args))
