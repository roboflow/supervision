import argparse
import json
from typing import List

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Counting time duration in zones with YOLOv8 and Supervision"
    )

    parser.add_argument(
        "--zone_configuration_path",
        required=True,
        help="Path to the zone configuration JSON file",
        type=str,
    )
    parser.add_argument(
        "--source_weights_path",
        default="yolov8x.pt",
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    # parser.add_argument(
    #     "--target_video_path",
    #     default=None,
    #     help="Path to the target video file (output)",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--confidence_threshold",
    #     default=0.3,
    #     help="Confidence threshold for the model",
    #     type=float,
    # )
    # parser.add_argument(
    #     "--iou_threshold",
    #     default=0.7,
    #     help="IOU threshold for the model",
    #     type=float,
    # )

    args = parser.parse_args()

    zone_config = load_zones_config(args.zone_configuration_path)

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    polygons = load_zones_config(args.zone_configuration_path)

    frames_generator = sv.get_video_frames_generator(args.source_video_path)

    for frame in tqdm(frames_generator, total=video_info.fps):

        annotated_frame = frame.copy()

        for i, polygon in enumerate(polygons):
            annotated_frame = sv.draw_polygon(
                annotated_frame, polygon, color=COLORS.by_idx(i), thickness=2)

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
