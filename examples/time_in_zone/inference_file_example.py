import sys

import cv2
import numpy as np
from inference import get_model
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
    model_id: str = "yolov8s-640",
    confidence: float = 0.3,
    iou: float = 0.7,
    classes: list[int] = [],
) -> None:
    """
    Calculating detections dwell time in zones, using video file.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        source_video_path: Path to the source video file
        model_id: Roboflow model ID
        confidence: Confidence level for detections (0 to 1)
        iou: IOU threshold for non-max suppression
        classes: List of class IDs to track. If empty, all classes are tracked
    """
    model = get_model(model_id=model_id)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
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

    for frame in frames_generator:
        results = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update_with_detections(detections)

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

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Try to import jsonargparse for CLI parsing
        from jsonargparse import ArgumentParser
    except ImportError:
        # Fallback if jsonargparse is not installed
        print("Warning: jsonargparse not installed. Using plain positional arguments.")
        if len(sys.argv) < 3:
            raise ValueError(
                "Insufficient arguments provided."
                "Usage: python inference_file_example.py "
                "<zone_configuration_path> <source_video_path> "
                "[model_id] [confidence] [iou] [classes]"
            )
        main(
            zone_configuration_path=sys.argv[1],
            source_video_path=sys.argv[2],
            model_id=sys.argv[3] if len(sys.argv) > 3 else "yolov8s-640",
            confidence=float(sys.argv[4]) if len(sys.argv) > 4 else 0.3,
            iou=float(sys.argv[5]) if len(sys.argv) > 5 else 0.7,
            classes=[int(x) for x in sys.argv[6:]] if len(sys.argv) > 6 else [],
        )
    else:
        # Use jsonargparse for automatic CLI if import succeeded
        parser = ArgumentParser()
        parser.add_function_arguments(main)
        args = parser.parse_args()
        main(**vars(args))
