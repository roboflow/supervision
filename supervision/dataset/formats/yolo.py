import numpy as np

from typing import List, Tuple

from supervision.detection.core import Detections


def parse_box(values: List[str]) -> np.ndarray:
    x_center, y_center, width, height = values
    return np.array([
        float(x_center) - float(width) / 2,
        float(y_center) - float(height) / 2,
        float(x_center) + float(width) / 2,
        float(y_center) + float(height) / 2
    ], dtype=np.float32)


def parse_segments(values: List[str]) -> np.ndarray:
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
            xyxy.append(parse_box(values=values[1:]))
        elif len(values) == 5 and force_segmentations:
            pass
        elif len(values) > 5 and not force_segmentations:
            pass
        elif len(values) > 5 and force_segmentations:
            pass

    class_id = np.array(class_id, dtype=int)
    xyxy = np.array(xyxy, dtype=np.float32)
    xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
    return Detections(
        class_id=class_id,
        xyxy=xyxy
    )



def detections_to_yolo_annotations(detections: Detections, resolution_wh: Tuple[int, int]) -> List[str]:
    if len(detections):
        return []

    for xyxy, mask, _, class_id, _ in detections:
        pass
