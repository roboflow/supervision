import copy
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils import (
    approximate_polygon,
    filter_polygons_by_area,
    mask_to_polygons,
)

T = TypeVar("T")


def approximate_mask_with_polygons(
    mask: np.ndarray,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> List[np.ndarray]:
    """
    Approximate a binary mask with polygons.

    This function takes in a binary mask and approximates it with polygons. It
    calculates the area of the input mask and defines the minimum and maximum
    detection area based on the provided image area percentages. It then
    converts the mask into polygons and filters them based on their area,
    keeping only the ones within the defined range. Finally, it approximates
    each polygon using a specified percentage and returns the resulting
    polygons as a list.

    Args:
        mask (np.ndarray): The binary mask to be approximated.
        min_image_area_percentage (float, optional): The minimum image area
            percentage for detection. Defaults to 0.0.
        max_image_area_percentage (float, optional): The maximum image area
            percentage for detection. Defaults to 1.0.
        approximation_percentage (float, optional): The approximation
            percentage for each polygon. Defaults to 0.75.

    Returns:
        List[np.ndarray]: A list of polygons approximating the input mask.
    """
    height, width = mask.shape
    image_area = height * width
    minimum_detection_area = min_image_area_percentage * image_area
    maximum_detection_area = max_image_area_percentage * image_area

    polygons = mask_to_polygons(mask=mask)
    if len(polygons) == 1:
        polygons = filter_polygons_by_area(
            polygons=polygons, min_area=None, max_area=maximum_detection_area
        )
    else:
        polygons = filter_polygons_by_area(
            polygons=polygons,
            min_area=minimum_detection_area,
            max_area=maximum_detection_area,
        )
    return [
        approximate_polygon(polygon=polygon, percentage=approximation_percentage)
        for polygon in polygons
    ]


def merge_class_lists(class_lists: List[List[str]]) -> List[str]:
    """
    Merge multiple lists of class names into a single list of unique class names.

    Args:
        class_lists (List[List[str]]): A list of lists of class names.

    Returns:
        List[str]: A sorted list of unique class names.
    """
    unique_classes = set()

    for class_list in class_lists:
        for class_name in class_list:
            unique_classes.add(class_name.lower())

    return sorted(list(unique_classes))


def build_class_index_mapping(
    source_classes: List[str], target_classes: List[str]
) -> Dict[int, int]:
    """
    Build a dictionary mapping source class indices to target class indices.

    Args:
        source_classes (List[str]): A list of source class names.
        target_classes (List[str]): A list of target class names.

    Returns:
        Dict[int, int]: A dictionary mapping source class indices to target class
            indices.

    Raises:
        ValueError: If a class name in source_classes is not found in target_classes.
    """
    index_mapping = {}

    for i, class_name in enumerate(source_classes):
        if class_name not in target_classes:
            raise ValueError(
                f"Class {class_name} not found in target classes. "
                "source_classes must be a subset of target_classes."
            )
        corresponding_index = target_classes.index(class_name)
        index_mapping[i] = corresponding_index

    return index_mapping


def map_detections_class_id(
    source_to_target_mapping: Dict[int, int], detections: Detections
) -> Detections:
    """
    Map the class IDs of Detections according to the given source-to-target mapping.

    Args:
        source_to_target_mapping (Dict[int, int]): A dictionary mapping source
            class indices to target class indices.
        detections (Detections): The input Detections object.

    Returns:
        Detections: A copy of the input Detections object with modified class IDs.

    Raises:
        ValueError: If detections.class_id is None or not a subset of
        source_to_target_mapping keys.
    """
    if detections.class_id is None:
        raise ValueError("Detections must have class_id attribute.")
    if set(np.unique(detections.class_id)) - set(source_to_target_mapping.keys()):
        raise ValueError(
            "Detections class_id must be a subset of source_to_target_mapping keys."
        )

    detections_copy = copy.deepcopy(detections)

    if len(detections) > 0:
        detections_copy.class_id = np.vectorize(source_to_target_mapping.get)(
            detections_copy.class_id
        )

    return detections_copy


def save_dataset_images(
    images_directory_path: str, images: Dict[str, np.ndarray]
) -> None:
    """
    Creates a directory at the specified path if it doesn't exist and saves the
    images in the directory.

    Args:
        images_directory_path (str): The path to the directory where the images will be
            saved.
        images (Dict[str, np.ndarray]): A dictionary containing image paths as
            keys and corresponding image arrays as values.
    """
    Path(images_directory_path).mkdir(parents=True, exist_ok=True)

    for image_path, image in images.items():
        image_name = Path(image_path).name
        target_image_path = os.path.join(images_directory_path, image_name)
        cv2.imwrite(target_image_path, image)


def train_test_split(
    data: List[T],
    train_ratio: float = 0.8,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[T], List[T]]:
    """
    Splits the data into two parts using the provided train_ratio.

    Args:
        data (List[T]): The data to split.
        train_ratio (float): The ratio of the training set to the entire dataset.
        random_state (Optional[int]): The seed for the random number generator.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        Tuple[List[T], List[T]]: The split data.
    """
    if random_state is not None:
        random.seed(random_state)

    if shuffle:
        random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
