from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_mask, polygon_to_xyxy
from supervision.file import list_files_with_extensions, read_txt_file


def _parse_box(values: List[str]) -> np.ndarray:
    x_center, y_center, width, height = values
    return np.array(
        [
            float(x_center) - float(width) / 2,
            float(y_center) - float(height) / 2,
            float(x_center) + float(width) / 2,
            float(y_center) + float(height) / 2,
        ],
        dtype=np.float32,
    )


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    return np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    )


def _parse_polygon(values: List[str]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def _polygons_to_masks(
    polygons: List[np.ndarray], resolution_wh: Tuple[int, int]
) -> np.ndarray:
    return np.array(
        [
            polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
            for polygon in polygons
        ],
        dtype=bool,
    )


def _pair_image_with_annotation(
    image_paths: List[Union[str, Path]], annotation_paths: List[Union[str, Path]]
) -> List[Tuple[Union[str, Path], Union[str, Path]]]:
    image_dict = {p.stem: p for p in image_paths}
    return [
        (image_dict[annotation_path.stem], annotation_path)
        for annotation_path in annotation_paths
        if annotation_path.stem in image_dict
    ]


def _with_mask(lines: List[str]) -> bool:
    return any([len(line.split()) > 5 for line in lines])


def _extract_class_names(file_path: str) -> List[str]:
    names = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.strip().startswith("names:"):
                start_index = lines.index(line) + 1
                break

        for line in lines[start_index:]:
            if line.strip().startswith("-"):
                names.append(line.strip().replace("-", "").strip())
            else:
                break

    return names


def yolo_annotations_to_detections(
    lines: List[str], resolution_wh: Tuple[int, int], with_masks: bool
) -> Detections:
    if len(lines) == 0:
        return Detections.empty()

    class_id, relative_xyxy, relative_polygon = [], [], []
    w, h = resolution_wh
    for line in lines:
        values = line.split()
        class_id.append(int(values[0]))
        if len(values) == 5:
            box = _parse_box(values=values[1:])
            relative_xyxy.append(box)
            if with_masks:
                relative_polygon.append(_box_to_polygon(box=box))
        elif len(values) > 5:
            polygon = _parse_polygon(values=values[1:])
            relative_xyxy.append(polygon_to_xyxy(polygon=polygon))
            if with_masks:
                relative_polygon.append(polygon)

    class_id = np.array(class_id, dtype=int)
    relative_xyxy = np.array(relative_xyxy, dtype=np.float32)
    xyxy = relative_xyxy * np.array([w, h, w, h], dtype=np.float32)

    if not with_masks:
        return Detections(class_id=class_id, xyxy=xyxy)

    polygons = [
        (polygon * np.array(resolution_wh)).astype(int) for polygon in relative_polygon
    ]
    mask = _polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
    return Detections(class_id=class_id, xyxy=xyxy, mask=mask)


def load_yolo_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    image_paths = list_files_with_extensions(
        directory=images_directory_path, extensions=["jpg", "jpeg", "png"]
    )
    annotation_paths = list_files_with_extensions(
        directory=annotations_directory_path, extensions=["txt"]
    )
    path_pairs = _pair_image_with_annotation(
        image_paths=image_paths, annotation_paths=annotation_paths
    )
    classes = _extract_class_names(file_path=data_yaml_path)
    images = {}
    annotations = {}
    for image_path, annotation_path in path_pairs:
        image = cv2.imread(str(image_path))
        lines = read_txt_file(str(annotation_path))
        h, w, _ = image.shape
        resolution_wh = (w, h)

        with_masks = _with_mask(lines=lines)
        with_masks = force_masks if force_masks else with_masks
        annotation = yolo_annotations_to_detections(
            lines=lines, resolution_wh=resolution_wh, with_masks=with_masks
        )

        images[str(image_path)] = image
        annotations[str(image_path)] = annotation
    return classes, images, annotations


# TODO
def detections_to_yolo_annotations(
    detections: Detections, resolution_wh: Tuple[int, int]
) -> List[str]:
    if len(detections):
        return []

    for xyxy, mask, _, class_id, _ in detections:
        pass
