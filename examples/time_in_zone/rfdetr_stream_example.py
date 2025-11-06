from __future__ import annotations

import argparse
from enum import Enum

import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from utils.general import find_in_list, load_zones_config
from utils.timers import ClockBasedTimer

import supervision as sv


class ModelSize(Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    BASE = "base"
    LARGE = "large"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

    @classmethod
    def from_value(cls, value: ModelSize | str) -> ModelSize:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid model size '{value}'. Must be one of {cls.list()}."
                ) from exc
        raise ValueError(
            f"Invalid value type '{type(value)}'. Expected str or ModelSize."
        )


def load_model(checkpoint: ModelSize | str, device: str, resolution: int):
    checkpoint = ModelSize.from_value(checkpoint)
    if checkpoint == ModelSize.NANO:
        return RFDETRNano(device=device, resolution=resolution)
    if checkpoint == ModelSize.SMALL:
        return RFDETRSmall(device=device, resolution=resolution)
    if checkpoint == ModelSize.MEDIUM:
        return RFDETRMedium(device=device, resolution=resolution)
    if checkpoint == ModelSize.BASE:
        return RFDETRBase(device=device, resolution=resolution)
    if checkpoint == ModelSize.LARGE:
        return RFDETRLarge(device=device, resolution=resolution)
    raise RuntimeError("Unhandled checkpoint type.")


def adjust_resolution(checkpoint: ModelSize | str, resolution: int) -> int:
    checkpoint = ModelSize.from_value(checkpoint)
    divisor = (
        32 if checkpoint in {ModelSize.NANO, ModelSize.SMALL, ModelSize.MEDIUM} else 56
    )
    remainder = resolution % divisor
    if remainder == 0:
        return resolution
    lower = resolution - remainder
    upper = lower + divisor
    return lower if resolution - lower < upper - resolution else upper


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
                scene=annotated_frame,
                polygon=zone.polygon,
                color=COLORS.by_idx(idx),
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
    rtsp_url: str,
    zone_configuration_path: str,
    model_size: str,
    device: str,
    confidence: float,
    iou: float,
    classes: list[int],
    resolution: int,
) -> None:
    resolution = adjust_resolution(checkpoint=model_size, resolution=resolution)
    model = load_model(checkpoint=model_size, device=device, resolution=resolution)

    def inference_callback(frames: list[VideoFrame]) -> list[sv.Detections]:
        dets = model.predict(frames[0].image, threshold=confidence)
        return [dets.with_nms(threshold=iou)]

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
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones using an RTSP stream."
    )
    parser.add_argument("--zone_configuration_path", required=True, type=str)
    parser.add_argument("--rtsp_url", required=True, type=str)
    parser.add_argument("--model_size", default="small", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--confidence_threshold", default=0.3, type=float)
    parser.add_argument("--iou_threshold", default=0.7, type=float)
    parser.add_argument("--classes", nargs="*", default=[], type=int)
    parser.add_argument("--resolution", required=True, type=int)
    args = parser.parse_args()
    main(
        rtsp_url=args.rtsp_url,
        zone_configuration_path=args.zone_configuration_path,
        model_size=args.model_size,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
        resolution=args.resolution,
    )
