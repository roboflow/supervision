import numpy as np

from typing import List, Tuple

from supervision.detection.utils import polygon_to_mask, polygon_to_xyxy
from supervision.detection.core import Detections


def parse_box(values: List[str]) -> np.ndarray:
    x_center, y_center, width, height = values
    return np.array([
        float(x_center) - float(width) / 2,
        float(y_center) - float(height) / 2,
        float(x_center) + float(width) / 2,
        float(y_center) + float(height) / 2
    ], dtype=np.float32)


def box_to_polygon(box: np.ndarray) -> np.ndarray:
    return np.array([
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]]
    ])


def parse_polygon(values: List[str]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def polygons_to_masks(polygons: List[np.ndarray], resolution_wh: Tuple[int, int]) -> np.ndarray:
    return np.array([
        polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
        for polygon
        in polygons
    ], dtype=bool)


def yolo_annotations_to_detections(
    lines: List[str],
    resolution_wh: Tuple[int, int],
    force_segmentations: bool
) -> Detections:
    if len(lines) == 0:
        return Detections.empty()
    class_id, relative_xyxy, relative_polygon = [], [], []
    w, h = resolution_wh
    for line in lines:
        values = line.split()
        class_id.append(int(values[0]))
        if len(values) == 5 and not force_segmentations:
            box = parse_box(values=values[1:])
            relative_xyxy.append(box)
        elif len(values) == 5 and force_segmentations:
            box = parse_box(values=values[1:])
            relative_xyxy.append(box)
            relative_polygon.append(box_to_polygon(box=box))
        else:
            polygon = parse_polygon(values=values[1:])
            relative_polygon.append(polygon)
            relative_xyxy.append(polygon_to_xyxy(polygon=polygon))

    class_id = np.array(class_id, dtype=int)
    relative_xyxy = np.array(relative_xyxy, dtype=np.float32)
    xyxy = relative_xyxy * np.array([w, h, w, h], dtype=np.float32)

    if not force_segmentations:
        return Detections(class_id=class_id, xyxy=xyxy)

    polygons = [
        (polygon * np.array(resolution_wh)).astype(int)
        for polygon
        in relative_polygon
    ]
    mask = polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
    return Detections(class_id=class_id, xyxy=xyxy, mask=mask)


def detections_to_yolo_annotations(detections: Detections, resolution_wh: Tuple[int, int]) -> List[str]:
    if len(detections):
        return []

    for xyxy, mask, _, class_id, _ in detections:
        pass
