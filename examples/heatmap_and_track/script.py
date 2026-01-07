import sys

import cv2
from ultralytics import YOLO

import supervision as sv
from supervision.assets import VideoAssets, download_assets


def download_video() -> str:
    download_assets(VideoAssets.PEOPLE_WALKING)
    return VideoAssets.PEOPLE_WALKING.value


def main(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    heatmap_alpha: float = 0.5,
    radius: int = 25,
    track_activation_threshold: float = 0.35,
    track_seconds: int = 5,
    minimum_matching_threshold: float = 0.99,
) -> None:
    """
    Heatmap and Tracking with Supervision.

    Args:
        source_weights_path: Path to the source weights file
        source_video_path: Path to the source video file
        target_video_path: Path to the target video file (output)
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
        heatmap_alpha: Opacity of the overlay mask, between 0 and 1
        radius: Radius of the heat circle
        track_activation_threshold: Detection confidence threshold for track activation
        track_seconds: Number of seconds to buffer when a track is lost
        minimum_matching_threshold: Threshold for matching tracks with detections
    """
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
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=track_seconds * fps,
        minimum_matching_threshold=minimum_matching_threshold,
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
    try:
        # Try to import jsonargparse for CLI parsing
        from jsonargparse import ArgumentParser
    except ImportError:
        # Fallback if jsonargparse is not installed
        print("Warning: jsonargparse not installed. Using plain positional arguments.")
        if len(sys.argv) < 2:
            script_name = sys.argv[0] if sys.argv else "script.py"
            usage_message = (
                "Insufficient arguments provided.\n\n"
                f"Usage:\n"
                f"  python {script_name} "
                "source_weights_path [source_video_path] [target_video_path] "
                "[confidence_threshold] [iou_threshold] [heatmap_alpha] [radius] "
                "[track_activation_threshold] [track_seconds] "
                "[minimum_matching_threshold]"
            )
            raise ValueError(usage_message)
        main(
            source_weights_path=sys.argv[1],
            source_video_path=sys.argv[2] if len(sys.argv) > 2 else download_video(),
            target_video_path=sys.argv[3] if len(sys.argv) > 3 else "output.mp4",
            confidence_threshold=float(sys.argv[4]) if len(sys.argv) > 4 else 0.35,
            iou_threshold=float(sys.argv[5]) if len(sys.argv) > 5 else 0.5,
            heatmap_alpha=float(sys.argv[6]) if len(sys.argv) > 6 else 0.5,
            radius=int(sys.argv[7]) if len(sys.argv) > 7 else 25,
            track_activation_threshold=float(sys.argv[8])
            if len(sys.argv) > 8
            else 0.35,
            track_seconds=int(sys.argv[9]) if len(sys.argv) > 9 else 5,
            minimum_matching_threshold=float(sys.argv[10])
            if len(sys.argv) > 10
            else 0.99,
        )
    else:
        # Use jsonargparse for automatic CLI if import succeeded
        parser = ArgumentParser()
        parser.add_function_arguments(main)
        args = parser.parse_args()
        main(**vars(args))
