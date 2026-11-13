import cv2
import numpy as np
from trackers import ByteTrackTracker
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


def main(
    zone_configuration_path: str,
    source_video_path: str,
    weights: str = "yolov8s.pt",
    device: str = "cpu",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    classes: list[int] = [],
    output_video_path: str = None,
) -> None:
    """Calculating detections dwell time in zones, using video file.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        source_video_path: Path to the source video file
        weights: Path to the model weights file
        device: Computation device ('cpu', 'mps' or 'cuda')
        confidence_threshold: Confidence level for detections (0 to 1)
        iou_threshold: IOU threshold for non-max suppression
        classes: List of class IDs to track. If empty, all classes are tracked
        output_video_path: Path to save the output video. If not provided, video
            will only be displayed.
    """
    model = YOLO(weights)
    tracker = ByteTrackTracker(
        minimum_iou_threshold=0.5, track_activation_threshold=confidence_threshold
    )
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    # Video writer setup
    video_writer = None
    if output_video_path:
        # Use Twitter-compatible codec - try H264 first, then XVID
        try:
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                video_info.fps,
                (video_info.width, video_info.height),
            )
            print(f"Video output being saved: {output_video_path}")
            print("Using Twitter-compatible H.264 codec")
        except:
            # H264 desteklenmiyorsa XVID kullan
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                video_info.fps,
                (video_info.width, video_info.height),
            )
            print(f"Video output being saved: {output_video_path}")
            print("Using XVID codec (convert with FFmpeg for Twitter)")

    window = sv.ImageWindow("Processed Video")
    for frame in frames_generator:
        results = model(
            frame,
            verbose=False,
            device=device,
            conf=confidence_threshold,
            iou=iou_threshold,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update(detections)
        detections = detections[detections.tracker_id != -1]  # -1 = pending track

        annotated_frame = frame.copy()

        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        window.show(annotated_frame)

        # Save frame to output video if writer is available
        if video_writer is not None:
            video_writer.write(annotated_frame)

        key = window.wait_key(1)
        if not window.is_open or key == "q":
            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"Video successfully saved: {output_video_path}")

    window.close()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
