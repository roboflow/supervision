import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv

COLORS = sv.ColorPalette.default()


def load_zones_config(file_path: str) -> List[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. Each polygon is represented as
    a NumPy array of coordinates.

    Args:
    file_path (str): The path to the JSON configuration file.

    Returns:
    List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def initiate_annotators(
    polygons: List[np.ndarray], resolution_wh: Tuple[int, int]
) -> Tuple[
    List[sv.PolygonZone], List[sv.PolygonZoneAnnotator], List[sv.BoundingBoxAnnotator]
]:
    line_thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wh)

    zones = []
    zone_annotators = []
    box_annotators = []

    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=resolution_wh)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=COLORS.by_idx(index),
            thickness=line_thickness,
            text_thickness=line_thickness * 2,
            text_scale=text_scale * 2,
        )
        box_annotator = sv.BoundingBoxAnnotator(
            color=COLORS.by_idx(index), thickness=line_thickness
        )
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        box_annotators.append(box_annotator)

    return zones, zone_annotators, box_annotators


def detect(
    frame: np.ndarray, model: RoboflowInferenceModel, confidence_threshold: float = 0.5
) -> sv.Detections:
    """
    Detect objects in a frame using Inference model, filtering detections by class ID
        and confidence threshold.

    Args:
        frame (np.ndarray): The frame to process, expected to be a NumPy array.
        model (RoboflowInferenceModel): The Inference model used for processing the
            frame.
        confidence_threshold (float, optional): The confidence threshold for filtering
            detections. Default is 0.5.

    Returns:
        sv.Detections: Filtered detections after processing the frame with the Inference
            model.

    Note:
        This function is specifically tailored for an Inference model and assumes class
        ID 0 for filtering.
    """
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    filter_by_class = detections.class_id == 0
    filter_by_confidence = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_confidence]


def annotate(
    frame: np.ndarray,
    zones: List[sv.PolygonZone],
    zone_annotators: List[sv.PolygonZoneAnnotator],
    box_annotators: List[sv.BoundingBoxAnnotator],
    detections: sv.Detections,
) -> np.ndarray:
    """
    Annotate a frame with zone and box annotations based on given detections.

    Args:
        frame (np.ndarray): The original frame to be annotated.
        zones (List[sv.PolygonZone]): A list of polygon zones used for detection.
        zone_annotators (List[sv.PolygonZoneAnnotator]): A list of annotators for
            drawing zone annotations.
        box_annotators (List[sv.BoundingBoxAnnotator]): A list of annotators for
            drawing box annotations.
        detections (sv.Detections): Detections to be used for annotation.

    Returns:
        np.ndarray: The annotated frame.
    """
    annotated_frame = frame.copy()
    for zone, zone_annotator, box_annotator in zip(
        zones, zone_annotators, box_annotators
    ):
        detections_in_zone = detections[zone.trigger(detections=detections)]
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections_in_zone
        )
    return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Counting people in zones with Inference and Supervision"
    )

    parser.add_argument(
        "--zone_configuration_path",
        required=True,
        help="Path to the zone configuration JSON file",
        type=str,
    )
    parser.add_argument(
        "--model_id",
        default="yolov8x-1280",
        help="Roboflow model ID",
        type=str,
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API KEY",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float,
    )

    args = parser.parse_args()

    api_key = args.roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    args.roboflow_api_key = api_key

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    polygons = load_zones_config(args.zone_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )

    model = get_roboflow_model(model_id=args.model_id, api_key=args.roboflow_api_key)

    frames_generator = sv.get_video_frames_generator(args.source_video_path)
    if args.target_video_path is not None:
        with sv.VideoSink(args.target_video_path, video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                detections = detect(frame, model, args.confidence_threshold)
                annotated_frame = annotate(
                    frame=frame,
                    zones=zones,
                    zone_annotators=zone_annotators,
                    box_annotators=box_annotators,
                    detections=detections,
                )
                sink.write_frame(annotated_frame)
    else:
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            detections = detect(frame, model, args.confidence_threshold)
            annotated_frame = annotate(
                frame=frame,
                zones=zones,
                zone_annotators=zone_annotators,
                box_annotators=box_annotators,
                detections=detections,
            )
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
