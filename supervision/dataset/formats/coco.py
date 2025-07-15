import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from supervision.dataset.utils import (
    approximate_mask_with_polygons,
    map_detections_class_id,
    mask_to_rle,
    rle_to_mask,
)
from supervision.detection.core import Detections
from supervision.detection.utils import (
    contains_holes,
    contains_multiple_segments,
    polygon_to_mask,
)
from supervision.utils.file import read_json_file, save_json_file

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset


def coco_categories_to_classes(coco_categories: list[dict]) -> list[str]:
    return [
        category["name"]
        for category in sorted(coco_categories, key=lambda category: category["id"])
    ]


def build_coco_class_index_mapping(
    coco_categories: list[dict], target_classes: list[str]
) -> dict[int, int]:
    source_class_to_index = {
        category["name"]: category["id"] for category in coco_categories
    }
    return {
        source_class_to_index[target_class_name]: target_class_index
        for target_class_index, target_class_name in enumerate(target_classes)
    }


def classes_to_coco_categories(classes: list[str]) -> list[dict]:
    return [
        {
            "id": class_id,
            "name": class_name,
            "supercategory": "common-objects",
        }
        for class_id, class_name in enumerate(classes)
    ]


def group_coco_annotations_by_image_id(
    coco_annotations: list[dict],
) -> dict[int, list[dict]]:
    annotations = {}
    for annotation in coco_annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(annotation)
    return annotations


def coco_annotations_to_masks(
    image_annotations: list[dict], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    return np.array(
        [
            rle_to_mask(
                rle=np.array(image_annotation["segmentation"]["counts"]),
                resolution_wh=resolution_wh,
            )
            if image_annotation["iscrowd"]
            else polygon_to_mask(
                polygon=np.reshape(
                    np.asarray(image_annotation["segmentation"], dtype=np.int32),
                    (-1, 2),
                ),
                resolution_wh=resolution_wh,
            )
            for image_annotation in image_annotations
        ],
        dtype=bool,
    )


def coco_annotations_to_detections(
    image_annotations: list[dict],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    use_iscrowd: bool = True,
) -> Detections:
    if not image_annotations:
        return Detections.empty()

    class_ids = [
        image_annotation["category_id"] for image_annotation in image_annotations
    ]
    xyxy = [image_annotation["bbox"] for image_annotation in image_annotations]
    xyxy = np.asarray(xyxy)
    xyxy[:, 2:4] += xyxy[:, 0:2]

    data = dict()
    if use_iscrowd:
        iscrowd = [
            image_annotation["iscrowd"] for image_annotation in image_annotations
        ]
        area = [image_annotation["area"] for image_annotation in image_annotations]
        data = dict(
            iscrowd=np.asarray(iscrowd, dtype=int), area=np.asarray(area, dtype=float)
        )

    if with_masks:
        mask = coco_annotations_to_masks(
            image_annotations=image_annotations, resolution_wh=resolution_wh
        )
    else:
        mask = None

    return Detections(
        class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask, data=data
    )


def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> tuple[list[dict], int]:
    coco_annotations = []
    for xyxy, mask, _, class_id, _, _ in detections:
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        segmentation = []
        iscrowd = 0
        if mask is not None:
            iscrowd = contains_holes(mask=mask) or contains_multiple_segments(mask=mask)

            if iscrowd:
                segmentation = {
                    "counts": mask_to_rle(mask=mask),
                    "size": list(mask.shape[:2]),
                }
            else:
                segmentation = [
                    list(
                        approximate_mask_with_polygons(
                            mask=mask,
                            min_image_area_percentage=min_image_area_percentage,
                            max_image_area_percentage=max_image_area_percentage,
                            approximation_percentage=approximation_percentage,
                        )[0].flatten()
                    )
                ]
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [xyxy[0], xyxy[1], box_width, box_height],
            "area": box_width * box_height,
            "segmentation": segmentation,
            "iscrowd": iscrowd,
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id


def get_coco_class_index_mapping(annotations_path: str) -> dict[int, int]:
    """
    Generates a mapping from sequential class indices to original COCO class ids.

    This function is essential when working with models that expect class ids to be
    zero-indexed and sequential (0 to 79), as opposed to the original COCO
    dataset where category ids are non-contiguous ranging from 1 to 90 but skipping some
    ids.

    Use Cases:
        - Evaluating models trained with COCO-style annotations where class ids
          are sequential ranging from 0 to 79.
        - Ensuring consistent class indexing across training, inference and evaluation,
          when using different tools or datasets with COCO format.
        - Reproducing results from models that assume sequential class ids (0 to 79).

    How it Works:
        - Reads the COCO annotation file in its original format (`annotations_path`).
        - Extracts and sorts all class names by their original COCO id (1 to 90).
        - Builds a mapping from COCO class ids (not sequential with skipped ids) to
          new class ids (sequential ranging from 0 to 79).
        - Returns a dictionary mapping: `{new_class_id: original_COCO_class_id}`.

    Args:
        annotations_path (str): Path to COCO JSON annotations file
        (e.g., `instances_val2017.json`).

    Returns:
        Dict[int, int]: A mapping from new class id (sequential ranging from 0 to 79)
        to original COCO class id (1 to 90 with skipped ids).
    """
    coco_data = read_json_file(annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    class_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )
    return {v: k for k, v in class_mapping.items()}


def load_coco_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
    use_iscrowd: bool = True,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    coco_data = read_json_file(file_path=annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])

    class_index_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )

    coco_images = coco_data["images"]
    coco_annotations_groups = group_coco_annotations_by_image_id(
        coco_annotations=coco_data["annotations"]
    )

    images = []
    annotations = {}

    for coco_image in coco_images:
        image_name, image_width, image_height = (
            coco_image["file_name"],
            coco_image["width"],
            coco_image["height"],
        )
        image_annotations = coco_annotations_groups.get(coco_image["id"], [])
        image_path = os.path.join(images_directory_path, image_name)

        annotation = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(image_width, image_height),
            with_masks=force_masks,
            use_iscrowd=use_iscrowd,
        )

        annotation = map_detections_class_id(
            source_to_target_mapping=class_index_mapping,
            detections=annotation,
        )

        images.append(image_path)
        annotations[image_path] = annotation

    return classes, images, annotations


def save_coco_annotations(
    dataset: "DetectionDataset",
    annotation_path: str,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
    licenses = [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0",
        }
    ]

    coco_annotations = []
    coco_images = []
    coco_categories = classes_to_coco_categories(classes=dataset.classes)

    image_id, annotation_id = 1, 1
    for image_path, image, annotation in dataset:
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
        coco_annotation, annotation_id = detections_to_coco_annotations(
            detections=annotation,
            image_id=image_id,
            annotation_id=annotation_id,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        coco_annotations.extend(coco_annotation)
        image_id += 1

    annotation_dict = {
        "info": {},
        "licenses": licenses,
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    save_json_file(annotation_dict, file_path=annotation_path)
