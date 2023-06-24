import os
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
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


def _annotations_dict_to_numpy(annotation_data: dict) -> np.ndarray:
    annotations_infos = annotation_data.get("annotations", None)
    # TODO: Add segmentation parser
    # image id, label-id, category_id, bbox[0], bbox[1], bbox[2], bbox[3], segmentaions
    annotations = np.zeros((0, 7))
    for anno in annotations_infos:
        _info = np.array([[anno["image_id"], anno["id"], anno["category_id"],
                           anno["bbox"][0], anno["bbox"][1], anno["bbox"][2], anno["bbox"][3]]])
        annotations = np.append(annotations, _info, axis=0)
    return annotations


def coco_annotations_to_detections(coco_annotations: np.ndarray, with_masks: bool) -> Detections:
    """
    Returns:
        object: detections
    """
    detections = Detections.empty()
    if coco_annotations.shape[0] > 0:
        coco_annotations[:, 5] += coco_annotations[:, 3]
        coco_annotations[:, 6] += coco_annotations[:, 4]
        detections = Detections(xyxy=coco_annotations[:, 3:], class_id=coco_annotations[:, 2].astype(int))
    return detections





def load_coco_annotations(
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    """
    Loads YOLO annotations and returns class names, images, and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations__path (str): The path to the coco json annotation file.
        force_masks (bool, optional): If True, forces masks to be loaded for all annotations, regardless of whether they are present.

    Returns:
        Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]: A tuple containing a list of class names, a dictionary with image names as keys and images as values, and a dictionary with image names as keys and corresponding Detections instances as values.
    """
    annotation_data = read_json_file(file_path=annotations_path)

    classes = _extract_class_names(annotation_data=annotation_data)
    images_infos = _extract_image_info(annotation_data=annotation_data)
    annotations_np = _annotations_dict_to_numpy(annotation_data=annotation_data)

    images = {}
    annotations = {}

    for images_info in images_infos:
        image_path = os.path.join(images_directory_path, images_info["file_name"])
        image = cv2.imread(str(image_path))

        # Filter annotations based on image id
        image_annotations = annotations_np[annotations_np[:, 0] == images_info["id"]]

        annotation = coco_annotations_to_detections(coco_annotations=image_annotations, with_masks=False)
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
) -> None:
    # Path(annotation_path).mkdir(parents=True, exist_ok=True)

    license = {'id': 1, 'url': 'https://creativecommons.org/licenses/by/4.0/', 'name': 'CC BY 4.0'}
    licenses = [license]  # TODO: accept as optional parameter

    annotations_data = []
    image_infos = []
    infos = {}  # TODO: accept as optional parameter

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
        # for class_id, x1, y1, x2, y2 in detections.class_id, detections.xyxy:
            per_label_dict = {}
            per_label_dict['id'] = label_id
            per_label_dict['image_id'] = image_id
            per_label_dict['category_id'] = int(class_id)
            w, h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
            per_label_dict['bbox'] = [xyxy[0], xyxy[1], w, h]
            per_label_dict['area'] = w * h  # width x height
            per_label_dict['segmentation'] = []
            per_label_dict['iscrowd'] = 0
            print(per_label_dict)
            annotations_data.append(per_label_dict)
            label_id += 1

        image_id += 1


    annotation_dict = {
        'info': infos,
        'licenses': licenses,
        'categories': categories,
        'images': image_infos,
        'annotations': annotations_data
    }
    save_json_file(annotation_dict, file_path=annotation_path)
