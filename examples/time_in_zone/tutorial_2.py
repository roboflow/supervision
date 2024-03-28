import argparse

import cv2

import numpy as np
import supervision as sv
from ultralytics import YOLO
from utils.general import load_zones_config, get_stream_frames_generator
from utils.timers import ClockBasedTimer


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.from_hex("#000000"))


def main(
    rtsp_url: str,
    zone_configuration_path: str,
    model_id: str
) -> None:
    model = YOLO("yolov8x.pt")
    tracker = sv.ByteTrack()
    fps_monitor = sv.FPSMonitor()
    frame_generator = get_stream_frames_generator(rtsp_url=rtsp_url)
    frame = next(frame_generator)
    resolution_wh = frame.shape[1], frame.shape[0]

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=resolution_wh,
            triggering_anchors=(sv.Position.CENTER,)
        )
        for polygon in polygons
    ]

    timers = [ClockBasedTimer() for _ in zones]

    for frame in frame_generator:
        fps_monitor.tick()
        fps = fps_monitor.fps

        results = model(frame, verbose=False, device='mps', conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections=detections)

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
                scene=annotated_frame,
                polygon=zone.polygon,
                color=COLORS.by_idx(idx),
                thickness=2
            )
            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full_like(detections_in_zone.class_id, idx)
            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup
            )
            labels = [
                f"#{tracker_id} {time:.1f}s"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup
            )

        cv2.imshow("Result", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        required=True,
        help="Complete RTSP URL for the video stream.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="yolov8s-640",
        help="Roboflow model ID."
    )
    args = parser.parse_args()

    main(
        rtsp_url=args.rtsp_url,
        zone_configuration_path=args.zone_configuration_path,
        model_id=args.model_id
    )
