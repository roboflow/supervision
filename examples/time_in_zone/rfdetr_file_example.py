from __future__ import annotations

import argparse
from enum import Enum

import cv2
import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class ModelSize(Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    BASE = "base"
    LARGE = "large"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_value(cls, value: ModelSize | str) -> ModelSize:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
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

    raise ValueError(
        f"Invalid checkpoint: {checkpoint}. Must be one of: {ModelSize.list()}."
    )


def adjust_resolution(checkpoint: ModelSize | str, resolution: int) -> int:
    checkpoint = ModelSize.from_value(checkpoint)

    if checkpoint in {ModelSize.NANO, ModelSize.SMALL, ModelSize.MEDIUM}:
        divisor = 32
    elif checkpoint in {ModelSize.BASE, ModelSize.LARGE}:
        divisor = 56
    else:
        raise ValueError(
            f"Unknown checkpoint: {checkpoint}. Must be one of: {ModelSize.list()}."
        )

    remainder = resolution % divisor
    if remainder == 0:
        return resolution
    lower = resolution - remainder
    upper = lower + divisor

    if resolution - lower < upper - resolution:
        return lower
    else:
        return upper


def main(
    source_video_path: str,
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
        detections = model.predict(frame, threshold=confidence)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = detections.with_nms(threshold=iou)
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
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Size of RF-DETR model ('nano', 'small', 'medium', 'base' or 'large'). "
        "Default is 'small'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    parser.add_argument(
        "--resolution",
        default=640,
        type=int,
        required=True,
        help="Resolution for the model input.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        model_size=args.model_size,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
        resolution=args.resolution,
    )
