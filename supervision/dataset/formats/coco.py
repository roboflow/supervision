import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from supervision.dataset.ultils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_mask
from supervision.utils.file import (
    save_json_file,
    read_json_file
)

"""
COCO Format Information:
coco: dict[info, licenses, categories, images, annotations]
info: dict []
licenses: dict['id', 'url', 'name']
categories: dict['id', 'name', 'supercategory']
images: dict['id', 'license', 'file_name', 'height', 'width', 'date_captured']
annotations: dict['id', 'image_id', 'category_id', 'bbox', 'area', 'segmentation', 'iscrowd']
bbox are in [x1, y1, w, h] in original scale
"""


def _extract_class_names(annotation_data: dict) -> List[str]:
    names = []
    categories = annotation_data.get("categories", None)
    if categories:
        for cat in categories:
            names.append(cat["name"])
    return names


def _extract_image_info(annotation_data: dict) -> List[dict]:
    image_infos = annotation_data.get("images", None)
    return image_infos


def _extract_image_names(image_infos: dict) -> List[dict]:
    image_names = []
    for image_info in image_infos:
        image_names.append(image_info["file_name"])
    return image_names


def _annotations_dict(annotation_data: dict, with_masks: bool) -> Tuple[np.ndarray, dict]:
    annotations_infos = annotation_data.get("annotations", None)
    # image id, label-id, category_id, bbox[0], bbox[1], bbox[2], bbox[3], segmentaions
    image_id_label_id_pair = np.zeros((0, 2))
    annotations = {}
    for anno in annotations_infos:
        _annotations = {}
        for key, item in anno.items():
            _annotations[key] = anno[key]
        id_pair = np.array([[anno["image_id"], anno["id"]]])
        image_id_label_id_pair = np.append(image_id_label_id_pair, id_pair, axis=0)
        annotations[anno['id']] = _annotations

    return image_id_label_id_pair, annotations


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


def coco_annotations_to_detections(image_annotations: List[dict], with_masks: bool) -> Detections:
    """
    Returns:
        object: detections
    """
    detection = Detections.empty()
    masks = [] if with_masks else None
    class_ids = []
    xyxy = []
    for image_annotation in image_annotations:
        bbox = image_annotation['bbox']
        xyxy.append(bbox)
        class_ids.append(image_annotation['category_id'])
        # _polygons = image_annotation['segmentation']
    xyxy = np.asarray(xyxy)
    if xyxy.shape[0] > 0:
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]
        class_ids = np.asarray(class_ids, dtype=int)
        detection = Detections(xyxy=xyxy, class_id=class_ids)
    return detection


def load_coco_annotations(
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    """
    Loads YOLO annotations and returns class names, images, and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_path (str): The path to the coco json annotation file.
        force_masks (bool, optional): If True, forces masks to be loaded for all annotations, regardless of whether they are present.

    Returns:
        Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]: A tuple containing a list of class names, a dictionary with image names as keys and images as values, and a dictionary with image names as keys and corresponding Detections instances as values.
    """
    annotation_data = read_json_file(file_path=annotations_path)

    classes = _extract_class_names(annotation_data=annotation_data)
    images_infos = _extract_image_info(annotation_data=annotation_data)
    image_id_label_id_pair, annotation_dict = _annotations_dict(annotation_data=annotation_data, with_masks=force_masks)

    images = {}
    annotations = {}

    for images_info in images_infos:
        image_path = os.path.join(images_directory_path, images_info["file_name"])
        image = cv2.imread(str(image_path))

        # Filter annotations based on image id
        per_image_label_ids = image_id_label_id_pair[image_id_label_id_pair[:, 0] == images_info["id"]][:, 1]

        image_annotations = []
        for per_image_label_id in per_image_label_ids:
            image_annotations.append(annotation_dict[int(per_image_label_id)])

        annotation = coco_annotations_to_detections(image_annotations=image_annotations, with_masks=force_masks)

        images[images_info["file_name"]] = image
        annotations[images_info["file_name"]] = annotation
    return classes, images, annotations


def save_coco_annotations(
        annotation_path: str,
        images: Dict[str, np.ndarray],
        annotations: Dict[str, Detections],
        classes: List[str],
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.75,
        licenses: List[dict] = None,
        info: dict = None
) -> None:
    Path(annotation_path).parents[2].mkdir(parents=True, exist_ok=True)
    if not info:
        info = {}
    if not licenses:
        licenses = [{'id': 1, 'url': 'https://creativecommons.org/licenses/by/4.0/', 'name': 'CC BY 4.0'}]

    annotations_data = []
    image_infos = []
    categories = []

    for class_id, class_name in enumerate(classes):
        cate_dict = {'id': class_id, 'name': class_name, "supercategory": "common-objects"}
        categories.append(cate_dict)

    image_id = 0
    label_id = 0
    for image_name, image in images.items():
        image_height, image_width, _ = image.shape

        # TODO: Modify datetime format
        image_info = {'id': image_id, 'license': 1, 'file_name': image_name,
                      'height': image_height, 'width': image_width,
                      'date_captured': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                      }

        image_infos.append(image_info)
        detections = annotations[image_name]
        for xyxy, mask, confidence, class_id, tracker_id in detections:
            polygons = []
            if mask is not None:
                polygons = approximate_mask_with_polygons(
                    mask=mask,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )

            per_label_dict = {}
            per_label_dict['id'] = label_id
            per_label_dict['image_id'] = image_id
            per_label_dict['category_id'] = int(class_id)
            w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
            per_label_dict['bbox'] = [xyxy[0], xyxy[1], w, h]
            per_label_dict['area'] = w * h  # width x height
            per_label_dict['segmentation'] = polygons
            per_label_dict['iscrowd'] = 0
            annotations_data.append(per_label_dict)
            label_id += 1

        image_id += 1

    annotation_dict = {
        'info': info,
        'licenses': licenses,
        'categories': categories,
        'images': image_infos,
        'annotations': annotations_data
    }
    save_json_file(annotation_dict, file_path=annotation_path)
