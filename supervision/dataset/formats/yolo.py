import numpy as np

from typing import List, Tuple

from supervision import polygon_to_mask
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
    ], dtype=int)


def parse_polygon(values: List[str]) -> np.ndarray:
    pass


def yolo_annotations_to_detections(
    lines: List[str],
    resolution_wh: Tuple[int, int],
    force_segmentations: bool
) -> Detections:
    if len(lines) == 0:
        return Detections.empty()
    class_id, xyxy, mask = [], [], []
    w, h = resolution_wh
    for line in lines:
        values = line.split()
        class_id.append(int(values[0]))
        if len(values) == 5 and not force_segmentations:
            box = parse_box(values=values[1:])
            xyxy.append(box)
        elif len(values) == 5 and force_segmentations:
            box = parse_box(values=values[1:])
            xyxy.append(box)
            box_polygon = box_to_polygon(box=box)
            box_mask = polygon_to_mask(polygon=box_polygon, resolution_wh=resolution_wh)
            mask.append(box_mask)
        elif len(values) > 5 and not force_segmentations:
            pass
        elif len(values) > 5 and force_segmentations:
            pass

    class_id = np.array(class_id, dtype=int)
    xyxy = np.array(xyxy, dtype=np.float32)
    xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
    return Detections(
        class_id=class_id,
        xyxy=xyxy,
        mask=np.array(mask, dtype=bool) if force_segmentations else None
    )



def detections_to_yolo_annotations(detections: Detections, resolution_wh: Tuple[int, int]) -> List[str]:
    if len(detections):
        return []

    for xyxy, mask, _, class_id, _ in detections:
        pass
