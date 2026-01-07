import sys

import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import ClockBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class CustomSink:
    def __init__(self, zone_configuration_path: str, classes: list[int]):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)

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

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
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
        cv2.waitKey(1)


def main(
    zone_configuration_path: str,
    rtsp_url: str,
    weights: str = "yolov8s.pt",
    device: str = "cpu",
    confidence: float = 0.3,
    iou: float = 0.7,
    classes: list[int] = [],
) -> None:
    """
    Calculating detections dwell time in zones, using RTSP stream.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        rtsp_url: Complete RTSP URL for the video stream
        weights: Path to the model weights file
        device: Computation device ('cpu', 'mps' or 'cuda')
        confidence: Confidence level for detections (0 to 1)
        iou: IOU threshold for non-max suppression
        classes: List of class IDs to track. If empty, all classes are tracked
    """
    model = YOLO(weights)

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        results = model(
            frame.image, verbose=False, conf=confidence, iou=iou, device=device
        )[0]
        return sv.Detections.from_ultralytics(results)

    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes)

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=rtsp_url,
        on_video_frame=inference_callback,
        on_prediction=sink.on_prediction,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


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
                "Usage: python ultralytics_stream_example.py "
                "<zone_configuration_path> <rtsp_url> "
                "[weights] [device] [confidence] [iou] [classes]"
            )
        main(
            zone_configuration_path=sys.argv[1],
            rtsp_url=sys.argv[2],
            weights=sys.argv[3] if len(sys.argv) > 3 else "yolov8s.pt",
            device=sys.argv[4] if len(sys.argv) > 4 else "cpu",
            confidence=float(sys.argv[5]) if len(sys.argv) > 5 else 0.3,
            iou=float(sys.argv[6]) if len(sys.argv) > 6 else 0.7,
            classes=[int(x) for x in sys.argv[7:]] if len(sys.argv) > 7 else [],
        )
    else:
        # Use jsonargparse for automatic CLI if import succeeded
        parser = ArgumentParser()
        parser.add_function_arguments(main)
        args = parser.parse_args()
        main(**vars(args))
