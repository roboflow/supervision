import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from supervision.dataset.utils import (
    approximate_mask_with_polygons,
    map_detections_class_id,
)
from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_mask
from supervision.utils.file import read_json_file, save_json_file


def coco_categories_to_classes(coco_categories: List[dict]) -> List[str]:
    """Converts a list of COCO categories to a list of class names."""
    return [
        category["name"]
        for category in sorted(coco_categories, key=lambda category: category["id"])
    ]


def build_coco_class_index_mapping(
    coco_categories: List[dict], target_classes: List[str]
) -> Dict[int, int]:
    """Builds a mapping from COCO class IDs to target class indices."""
    source_class_to_index = {
        category["name"]: category["id"] for category in coco_categories
    }
    return {
        source_class_to_index[target_class_name]: target_class_index
        for target_class_index, target_class_name in enumerate(target_classes)
    }


def classes_to_coco_categories(classes: List[str]) -> List[dict]:
    """Converts a list of class names to a list of COCO categories."""
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
    """
    Group COCO annotations by image ID.

    This function takes a list of COCO annotations as input and returns a dictionary
    where the keys are image IDs and the values are lists of annotations
    associated with each image ID.

    Args:
        coco_annotations (List[dict]): A list of COCO annotations.

    Returns:
        Dict[int, List[dict]]: A dictionary where the keys are image IDs
        and the values are lists of annotations.
    """
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
    """
    Convert polygons to binary masks.

    This function takes a list of polygons and a resolution as input and converts
    the polygons into binary masks by calling the 'polygon_to_mask' function
    for each polygon.

    Args:
        polygons (List[np.ndarray]): A list of polygons.
        resolution_wh (Tuple[int, int]): The resolution of the masks.

    Returns:
        np.ndarray: An array of binary masks.
    """
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
    if not image_annotations:
        return Detections.empty()

    class_ids = [
        image_annotation["category_id"] for image_annotation in image_annotations
    ]
    xyxy = [image_annotation["bbox"] for image_annotation in image_annotations]
    xyxy = np.asarray(xyxy)
    xyxy[:, 2:4] += xyxy[:, 0:2]

    if with_masks:
        polygons = [
            np.reshape(
                np.asarray(image_annotation["segmentation"], dtype=np.int32), (-1, 2)
            )
            for image_annotation in image_annotations
        ]
        mask = _polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
        return Detections(
            class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask
        )

    return Detections(xyxy=xyxy, class_id=np.asarray(class_ids, dtype=int))


def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> Tuple[List[Dict], int]:
    """
    Convert detections to COCO annotations.

    This function takes in detections, image ID, annotation ID, and other optional
    parameters to generate COCO annotations for each detection. The COCO
    annotations are returned as a list along with the updated annotation ID.

    Args:
        detections (Detections): The detections object.
        image_id (int): The ID of the image.
        annotation_id (int): The ID of the annotation.
        min_image_area_percentage (float, optional): The minimum image area percentage.
        Defaults to 0.0.
        max_image_area_percentage (float, optional): The maximum image area percentage.
        Defaults to 1.0.
        approximation_percentage (float, optional): The approximation percentage.
        Defaults to 0.75.

    Returns:
        Tuple[List[Dict], int]: The COCO annotations and the updated annotation ID.
    """
    coco_annotations = []
    for xyxy, mask, _, class_id, _ in detections:
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
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
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [xyxy[0], xyxy[1], box_width, box_height],
            "area": box_width * box_height,
            "segmentation": [polygon] if polygon else [],
            "iscrowd": 0,
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id


def load_coco_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    """
    Load COCO annotations from a specified directory and annotations file.

    This function reads the annotations data from the JSON file specified by
    'annotations_path', processes the data to extract useful information, and
    returns a tuple containing three items: a list of classes, a dictionary of
    images, and a dictionary of annotations.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_path (str): The path to the annotations file.
        force_masks (bool, optional): Whether to use masks. Defaults to False.

    Returns:
        Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]: A tuple
        containing three items:
            1. A list of classes.
            2. A dictionary of images.
            3. A dictionary of annotations.
    """
    coco_data = read_json_file(file_path=annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    class_index_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )
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

        image = cv2.imread(image_path)
        annotation = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(image_width, image_height),
            with_masks=force_masks,
        )
        annotation = map_detections_class_id(
            source_to_target_mapping=class_index_mapping,
            detections=annotation,
        )

        images[image_path] = image
        annotations[image_path] = annotation

    return classes, images, annotations


def save_coco_annotations(
    annotation_path: str,
    images: Dict[str, np.ndarray],
    annotations: Dict[str, Detections],
    classes: List[str],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    """Save annotations in COCO format to the specified path."""
    Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
    info = {}
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

    image_id, annotation_id = 1, 1
    for image_path, image in images.items():
        image_height, image_width, _ = image.shape
        image_name = f"{Path(image_path).stem}{Path(image_path).suffix}"
        coco_image = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
        }

        coco_images.append(coco_image)
        detections = annotations[image_path]

        coco_annotation, annotation_id = detections_to_coco_annotations(
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
