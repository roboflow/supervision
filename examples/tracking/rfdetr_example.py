from rfdetr import RFDETRMedium
from tqdm import tqdm

import supervision as sv
from supervision import _cv2 as cv2


def main(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """
    Video Processing with RF-DETR and ByteTrack.

    Args:
        source_video_path: Path to the source video file
        target_video_path: Path to the target video file (output)
        device: Computation device ('cpu', 'mps' or 'cuda')
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
    """
    model = RFDETRMedium(device=device)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=confidence_threshold)
            detections = detections.with_nms(threshold=iou_threshold)
            detections = tracker.update_with_detections(detections)

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
