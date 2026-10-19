import time

import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from trackers import ByteTrackTracker
from utils.general import find_in_list, load_zones_config

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class CustomSink:
    def __init__(
        self,
        zone_configuration_path: str,
        classes: list[int],
        confidence_threshold: float,
    ) -> None:
        self.classes = classes
        self.tracker = ByteTrackTracker(
            minimum_iou_threshold=0.5, track_activation_threshold=confidence_threshold
        )
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
                # Ignore boxes flickering across the edge for under 3 frames.
                minimum_crossing_threshold=3,
            )
            for polygon in self.polygons
        ]

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = sv.Detections.from_inference(result)
        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update(detections)
        detections = detections[detections.tracker_id != -1]  # -1 = pending track

        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            zone.trigger(detections, timestamp=time.monotonic())
            time_in_zone = zone.time_in_zone(detections)
            # Dwell is NaN until an entry is confirmed, so only confirmed
            # stays are drawn and labelled.
            is_in_zone = ~np.isnan(time_in_zone)
            detections_in_zone = detections[is_in_zone]
            time_in_zone = time_in_zone[is_in_zone]
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(t // 60):02d}:{int(t % 60):02d}"
                for tracker_id, t in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )
        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)


def main(
    zone_configuration_path: str,
    rtsp_url: str,
    model_id: str = "rfdetr-medium",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    classes: list[int] = [],
    roboflow_api_key: str = "",
) -> None:
    """Calculating detections dwell time in zones, using RTSP stream.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        rtsp_url: Complete RTSP URL for the video stream
        model_id: Roboflow model ID
        confidence_threshold: Confidence level for detections (0 to 1)
        iou_threshold: IOU threshold for non-max suppression
        classes: List of class IDs to track. If empty, all classes are tracked
        roboflow_api_key: Roboflow API key for accessing private models
    """
    sink = CustomSink(
        zone_configuration_path=zone_configuration_path,
        classes=classes,
        confidence_threshold=confidence_threshold,
    )

    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=rtsp_url,
        on_prediction=sink.on_prediction,
        confidence=confidence_threshold,
        iou_threshold=iou_threshold,
        api_key=roboflow_api_key,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
