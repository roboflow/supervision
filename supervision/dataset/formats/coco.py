import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from supervision.dataset.ultils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_mask
from supervision.utils.file import read_json_file, save_json_file


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


def _annotations_dict(annotation_data: dict) -> Tuple[np.ndarray, dict]:
    annotations_infos = annotation_data.get("annotations", None)
    image_id_label_id_pair = np.zeros((0, 2))
    annotations = {}
    for anno in annotations_infos:
        _annotations = {}
        for key, item in anno.items():
            _annotations[key] = anno[key]
        id_pair = np.array([[anno["image_id"], anno["id"]]])
        image_id_label_id_pair = np.append(image_id_label_id_pair, id_pair, axis=0)
        annotations[anno["id"]] = _annotations

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


def coco_annotations_to_detections(
    image_annotations: List[dict], resolution_wh: Tuple[int, int], with_masks: bool
) -> Detections:
    detection = Detections.empty()
    class_ids = []
    xyxy = []
    polygons = []

    for image_annotation in image_annotations:
        bbox = image_annotation["bbox"]
        xyxy.append(bbox)
        class_ids.append(image_annotation["category_id"])
        if with_masks:
            _polygons = image_annotation["segmentation"]
            _polygons = np.asarray(_polygons, dtype=np.int32)
            _polygons = np.reshape(_polygons, (-1, 2))
            polygons.append(_polygons)

    xyxy = np.asarray(xyxy)
    if xyxy.shape[0] > 0:
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]
        class_ids = np.asarray(class_ids, dtype=int)

        if with_masks:
            mask = _polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
            detection = Detections(class_id=class_ids, xyxy=xyxy, mask=mask)
        else:
            detection = Detections(xyxy=xyxy, class_id=class_ids)

    return detection


def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    label_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> Tuple[List[Dict], int]:
    annotations_data = []
    for xyxy, mask, confidence, class_id, tracker_id in detections:
        polygon = []
        if mask is not None:
            polygon = list(
                approximate_mask_with_polygons(
                    mask=mask,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )[0].flatten()
            )

        per_label_dict = {}
        per_label_dict["id"] = label_id
        per_label_dict["image_id"] = image_id
        per_label_dict["category_id"] = int(class_id)
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        per_label_dict["bbox"] = [xyxy[0], xyxy[1], box_width, box_height]
        per_label_dict["area"] = box_width * box_height
        per_label_dict["segmentation"] = polygon
        per_label_dict["iscrowd"] = 0
        annotations_data.append(per_label_dict)
        label_id += 1
    return annotations_data, label_id


def classes_to_coco_category_map(classes: List[str]) -> List[Dict]:
    categories = []
    for class_id, class_name in enumerate(classes):
        cate_dict = {
            "id": class_id,
            "name": class_name,
            "supercategory": "common-objects",
        }
        categories.append(cate_dict)
    return categories


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
    image_id_label_id_pair, annotation_dict = _annotations_dict(
        annotation_data=annotation_data
    )

    images = {}
    annotations = {}

    for images_info in images_infos:
        image_path = os.path.join(images_directory_path, images_info["file_name"])
        image = cv2.imread(str(image_path))

        # Filter annotations based on image id
        per_image_label_ids = image_id_label_id_pair[
            image_id_label_id_pair[:, 0] == images_info["id"]
        ][:, 1]

        image_annotations = []
        for per_image_label_id in per_image_label_ids:
            image_annotations.append(annotation_dict[int(per_image_label_id)])

        w, h = images_info["width"], images_info["height"]

        annotation = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(w, h),
            with_masks=force_masks,
        )

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
    info: dict = None,
) -> None:
    Path(annotation_path).parents[2].mkdir(parents=True, exist_ok=True)
    if not info:
        info = {}
    if not licenses:
        licenses = [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0",
            }
        ]

    annotations_data = []
    image_infos = []
    categories = classes_to_coco_category_map(classes=classes)

    image_id = 0
    label_id = 0
    for image_name, image in images.items():
        image_height, image_width, _ = image.shape

        image_info = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
        }

        image_infos.append(image_info)
        detections = annotations[image_name]

        per_image_labels, label_id = detections_to_coco_annotations(
            detections=detections,
            image_id=image_id,
            label_id=label_id,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        annotations_data.extend(per_image_labels)
        image_id += 1

    annotation_dict = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": image_infos,
        "annotations": annotations_data,
    }
    save_json_file(annotation_dict, file_path=annotation_path)
