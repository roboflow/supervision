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


def coco_categories_to_classes(coco_categories: List[dict]) -> List[str]:
    return [
        category["name"]
        for category in sorted(coco_categories, key=lambda category: category["id"])
        if category["supercategory"] != "none"
    ]


def classes_to_coco_categories(classes: List[str]) -> List[dict]:
    return [
        {
            "id": class_id,
            "name": class_name,
            "supercategory": "common-objects",
        }
        for class_id, class_name in enumerate(classes)
    ]


def group_coco_annotations_by_image_id(
    coco_annotations: List[dict],
) -> Dict[int, List[dict]]:
    annotations = {}
    for annotation in coco_annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(annotation)
    return annotations


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
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> Tuple[List[Dict], int]:
    coco_annotations = []
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

        coco_annotation = {}
        coco_annotation["id"] = annotation_id
        coco_annotation["image_id"] = image_id
        coco_annotation["category_id"] = int(class_id)
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        coco_annotation["bbox"] = [xyxy[0], xyxy[1], box_width, box_height]
        coco_annotation["area"] = box_width * box_height
        coco_annotation["segmentation"] = polygon
        coco_annotation["iscrowd"] = 0
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id


def load_coco_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    """
    Loads COCO annotations and returns class names, images, and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_path (str): The path to the coco json annotation file.
        force_masks (bool, optional): If True, forces masks to be loaded for all annotations, regardless of whether they are present.

    Returns:
        Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]: A tuple containing a list of class names, a dictionary with image names as keys and images as values, and a dictionary with image names as keys and corresponding Detections instances as values.
    """
    coco_data = read_json_file(file_path=annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    coco_images = coco_data["images"]
    coco_annotations_groups = group_coco_annotations_by_image_id(
        coco_annotations=coco_data["annotations"]
    )

    images = {}
    annotations = {}

    for coco_image in coco_images:
        image_name, image_width, image_height = (
            coco_image["file_name"],
            coco_image["width"],
            coco_image["height"],
        )
        image_annotations = coco_annotations_groups.get(coco_image["id"], [])
        image_path = os.path.join(images_directory_path, image_name)

        image = cv2.imread(str(image_path))
        annotation = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(image_width, image_height),
            with_masks=force_masks,
        )

        images[image_name] = image
        annotations[image_name] = annotation

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
    Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
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

    coco_annotations = []
    coco_images = []
    coco_categories = classes_to_coco_categories(classes=classes)

    image_id = 0
    annotation_id = 0
    for image_name, image in images.items():
        image_height, image_width, _ = image.shape

        coco_image = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
        }

        coco_images.append(coco_image)
        detections = annotations[image_name]

        coco_annotation, label_id = detections_to_coco_annotations(
            detections=detections,
            image_id=image_id,
            annotation_id=annotation_id,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        coco_annotations.extend(coco_annotation)
        image_id += 1

    annotation_dict = {
        "info": info,
        "licenses": licenses,
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    save_json_file(annotation_dict, file_path=annotation_path)
