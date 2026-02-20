import cv2
import numpy as np
from inference import get_model
from utils.general import find_in_list, get_stream_frames_generator, load_zones_config
from utils.timers import ClockBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


def main(
    zone_configuration_path: str,
    rtsp_url: str,
    model_id: str = "rfdetr-medium",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    classes: list[int] = [],
    roboflow_api_key: str = "",
) -> None:
    """
    Calculating detections dwell time in zones, using RTSP stream.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        rtsp_url: Complete RTSP URL for the video stream
        model_id: Roboflow model ID
        confidence_threshold: Confidence level for detections (0 to 1)
        iou_threshold: IOU threshold for non-max suppression
        classes: List of class IDs to track. If empty, all classes are tracked
        roboflow_api_key: Roboflow API key for accessing private models
    """
    model = get_model(model_id=model_id, api_key=roboflow_api_key)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    frames_generator = get_stream_frames_generator(rtsp_url=rtsp_url)
    fps_monitor = sv.FPSMonitor()

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [ClockBasedTimer() for _ in zones]

    for frame in frames_generator:
        fps_monitor.tick()
        fps = fps_monitor.fps

        results = model.infer(
            frame, confidence=confidence_threshold, iou_threshold=iou_threshold
        )[0]
        detections = sv.Detections.from_inference(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

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
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
