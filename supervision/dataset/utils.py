import copy
import os
import random
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

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
    unique_classes = set()

    for class_list in class_lists:
        for class_name in class_list:
            unique_classes.add(class_name.lower())

    return sorted(list(unique_classes))


def build_class_index_mapping(
    source_classes: List[str], target_classes: List[str]
) -> Dict[int, int]:
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


def rle_to_mask(
    rle: npt.NDArray[np.int_], resolution_wh: Tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """
    Converts run-length encoding (RLE) to a binary mask.

    Args:
        rle (npt.NDArray[np.int_]): The 1D RLE array, the format used in the COCO 
            dataset (column-wise encoding, values of an array with even indices 
            represent the number of pixels assigned as background,
            values of an array with odd indices represent the number of pixels 
            assigned as foreground object).
        resolution_wh (Tuple[int, int]): The width (w) and height (h) 
            of the desired binary mask resolution.

    Returns:
        npt.NDArray[np.bool_]: The generated 2D Boolean mask of shape (h,w), 
            where the foreground object is marked with `True`'s and the rest 
            is filled with `False`'s.

    Raises:
        AssertionError: If the sum of pixels encoded in RLE differs from the
        number of pixels in the expected mask (computed based on resolution_wh).

    Examples:
        rle = [2, 2, 2], resolution_wh = [3, 2] -> mask = [[False, True, False],
                                                           [False, True, False]]
    """
    width, height = resolution_wh

    assert (
        width * height == np.sum(rle)
    ), ("the sum of the number of pixels in the RLE must be the same "
        "as the number of pixels in the expected mask")

    zero_one_values = np.zeros_like(rle)
    zero_one_values[1::2] = 1

    decoded_rle = np.repeat(zero_one_values, rle)
    return decoded_rle.reshape((height, width), order="F")


def mask_to_rle(mask: npt.NDArray[np.bool_]) -> List[int]:
    """
    Converts a binary mask into a run-length encoding (RLE).

    Args:
        mask (npt.NDArray[np.bool_]): 2D binary mask where `True` indicates foreground 
            object and `False` indicates background.

    Returns:
        List[int]: the run-length encoded mask. Values of a list with even indices 
            represent the number of pixels assigned as background (`False`), values 
            of a list with odd indices represent the number of pixels assigned 
            as foreground object (`True`).

    Raises:
        AssertionError: If imput mask is not 2D or is empty.

     Examples:
        mask = [[False, True, True],  -> rle = [2, 4]
                [False, True, True]]

        mask = [[True, True, True],   -> rle = [0, 6]
                [True, True, True]]
    """
    assert mask.ndim == 2, "Input mask must be 2D"
    assert mask.size != 0, "Input mask cannot be empty"

    rle = []
    if mask[0][0] == 1:
        rle = [0]

    for _, group in groupby(mask.ravel(order="F")):
        rle.append(len(list(group)))

    return rle
