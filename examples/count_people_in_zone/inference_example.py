import json
import os

import cv2
import numpy as np
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv

COLORS = sv.ColorPalette.DEFAULT


def load_zones_config(file_path: str) -> list[np.ndarray]:
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
    with open(file_path) as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def initiate_annotators(
    polygons: list[np.ndarray], resolution_wh: tuple[int, int]
) -> tuple[list[sv.PolygonZone], list[sv.PolygonZoneAnnotator], list[sv.BoxAnnotator]]:
    line_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    zones = []
    zone_annotators = []
    box_annotators = []

    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=COLORS.by_idx(index),
            thickness=line_thickness,
            text_thickness=line_thickness * 2,
            text_scale=text_scale * 2,
        )
        box_annotator = sv.BoxAnnotator(
            color=COLORS.by_idx(index), thickness=line_thickness
        )
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        box_annotators.append(box_annotator)

    return zones, zone_annotators, box_annotators


def detect(
    frame: np.ndarray,
    model: RoboflowInferenceModel,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.7,
) -> sv.Detections:
    """
    Detect objects in a frame using Inference model, filtering detections by class ID
        and confidence threshold.

    Args:
        frame (np.ndarray): The frame to process, expected to be a NumPy array.
        model (RoboflowInferenceModel): The Inference model used for processing the
            frame.
        confidence_threshold (float): The confidence threshold for filtering
            detections.
        iou_threshold (float): The IoU threshold for non-maximum suppression.

    Returns:
        sv.Detections: Filtered detections after processing the frame with the Inference
            model.

    Note:
        This function is specifically tailored for an Inference model and assumes class
        ID 0 for filtering.
    """
    results = model.infer(frame, confidence=confidence_threshold, iou=iou_threshold)[0]
    detections = sv.Detections.from_inference(results)
    filter_by_class = detections.class_id == 0
    filter_by_confidence = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_confidence]


def annotate(
    frame: np.ndarray,
    zones: list[sv.PolygonZone],
    zone_annotators: list[sv.PolygonZoneAnnotator],
    box_annotators: list[sv.BoxAnnotator],
    detections: sv.Detections,
) -> np.ndarray:
    """
    Annotate a frame with zone and box annotations based on given detections.

    Args:
        frame (np.ndarray): The original frame to be annotated.
        zones (List[sv.PolygonZone]): A list of polygon zones used for detection.
        zone_annotators (List[sv.PolygonZoneAnnotator]): A list of annotators for
            drawing zone annotations.
        box_annotators (List[sv.BoxAnnotator]): A list of annotators for
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


def main(
    zone_configuration_path: str,
    source_video_path: str,
    model_id: str = "yolov8x-1280",
    roboflow_api_key: str | None = None,
    target_video_path: str | None = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
):
    """
    Counting people in zones with Inference and Supervision.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        source_video_path: Path to the source video file
        model_id: Roboflow model ID
        roboflow_api_key: Roboflow API KEY
        target_video_path: Path to the target video file (output)
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
    """
    api_key = roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    roboflow_api_key = api_key

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    polygons = load_zones_config(zone_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )

    model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)

    frames_generator = sv.get_video_frames_generator(source_video_path)
    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                detections = detect(frame, model, confidence_threshold, iou_threshold)
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
            detections = detect(frame, model, confidence_threshold, iou_threshold)
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


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
