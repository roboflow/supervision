import argparse

import cv2
from ultralytics import YOLO

import supervision as sv
from supervision.assets import VideoAssets, download_assets


def download_video() -> str:
    download_assets(VideoAssets.PEOPLE_WALKING)
    return VideoAssets.PEOPLE_WALKING.value


def heatmap_and_track(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    heatmap_alpha: float = 0.5,
    radius: int = 25,
    track_threshold: float = 0.35,
    track_seconds: int = 5,
    match_threshold: float = 0.99,
) -> None:
    ### instantiate model
    model = YOLO(source_weights_path)

    ### heatmap config
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=heatmap_alpha,
        radius=radius,
        kernel_size=25,
        top_hue=0,
        low_hue=125,
    )

    ### annotation config
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    ### get the video fps
    cap = cv2.VideoCapture(source_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    ### tracker config
    byte_tracker = sv.ByteTrack(
        track_thresh=track_threshold,
        track_buffer=track_seconds * fps,
        match_thresh=match_threshold,
        frame_rate=fps,
    )

    ### video config
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=1
    )

    ### Detect, track, annotate, save
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in frames_generator:
            result = model(
                source=frame,
                classes=[0],  # only person class
                conf=confidence_threshold,
                iou=iou_threshold,
                # show_conf = True,
                # save_txt = True,
                # save_conf = True,
                # save = True,
                device=None,  # use None = CPU, 0 = single GPU, or [0,1] = dual GPU
            )[0]

            detections = sv.Detections.from_ultralytics(result)  # get detections

            detections = byte_tracker.update_with_detections(
                detections
            )  # update tracker

            ### draw heatmap
            annotated_frame = heat_map_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            ### draw other attributes from `detections` object
            labels = [
                f"#{tracker_id}"
                for class_id, tracker_id in zip(
                    detections.class_id, detections.tracker_id
                )
            ]

            label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heatmap and Tracking with Supervision"
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        default=download_video(),
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default="output.mp4",
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.35,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.5,
        help="IOU threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--heatmap_alpha",
        default=0.5,
        help="Opacity of the overlay mask, between 0 and 1",
        type=float,
    )
    parser.add_argument(
        "--radius",
        default=25,
        help="Radius of the heat circle",
        type=float,
    )
    parser.add_argument(
        "--track_threshold",
        default=0.35,
        help="Detection confidence threshold for track activation",
        type=float,
    )
    parser.add_argument(
        "--track_seconds",
        default=5,
        help="Number of seconds to buffer when a track is lost",
        type=int,
    )
    parser.add_argument(
        "--match_threshold",
        default=0.99,
        help="Threshold for matching tracks with detections",
        type=float,
    )

    args = parser.parse_args()

    heatmap_and_track(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        heatmap_alpha=args.heatmap_alpha,
        radius=args.radius,
        track_threshold=args.track_threshold,
        track_seconds=args.track_seconds,
        match_threshold=args.match_threshold,
    )
