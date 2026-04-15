from __future__ import annotations

import copy
import os
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void

from supervision.detection.core import Detections
from supervision.detection.utils.converters import mask_to_polygons
from supervision.detection.utils.converters import (
    mask_to_rle as _mask_to_rle,
)
from supervision.detection.utils.converters import (
    rle_to_mask as _rle_to_mask,
)
from supervision.detection.utils.polygons import (
    approximate_polygon,
    filter_polygons_by_area,
)


@deprecated(target=_mask_to_rle, deprecated_in="0.28.0", remove_in="0.30.0")  # type: ignore[untyped-decorator]
def mask_to_rle(
    mask: npt.NDArray[np.bool_], compressed: bool = False
) -> list[int] | str:
    """Deprecated. Use `supervision.detection.utils.converters.mask_to_rle`."""
    return void(mask, compressed)  # type: ignore[no-any-return]


@deprecated(target=_rle_to_mask, deprecated_in="0.28.0", remove_in="0.30.0")  # type: ignore[untyped-decorator]
def rle_to_mask(
    rle: npt.NDArray[np.integer[Any]] | list[int] | str | bytes,
    resolution_wh: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Deprecated. Use `supervision.detection.utils.converters.rle_to_mask`."""
    return void(rle, resolution_wh)


if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

T = TypeVar("T")


def approximate_mask_with_polygons(
    mask: npt.NDArray[np.bool_],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> list[npt.NDArray[np.number]]:
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


def merge_class_lists(class_lists: list[list[str]]) -> list[str]:
    unique_classes = set()

    for class_list in class_lists:
        for class_name in class_list:
            unique_classes.add(class_name)

    return sorted(list(unique_classes))


def build_class_index_mapping(
    source_classes: list[str], target_classes: list[str]
) -> dict[int, int]:
    """Returns the index map of source classes -> target classes."""
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
    source_to_target_mapping: dict[int, int], detections: Detections
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


def save_dataset_images(dataset: DetectionDataset, images_directory_path: str) -> None:
    Path(images_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path in dataset.image_paths:
        final_path = os.path.join(images_directory_path, Path(image_path).name)
        if image_path in dataset._images_in_memory:
            image = dataset._images_in_memory[image_path]
            cv2.imwrite(final_path, image)
        else:
            shutil.copyfile(image_path, final_path)


def train_test_split(
    data: list[T],
    train_ratio: float = 0.8,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[list[T], list[T]]:
    """
    Splits the data into two parts using the provided train_ratio.

    Args:
        data: The data to split.
        train_ratio: The ratio of the training set to the entire dataset.
        random_state: The seed for the random number generator.
        shuffle: Whether to shuffle the data before splitting.

    Returns:
        The split data.
    """
    if random_state is not None:
        random.seed(random_state)

    if shuffle:
        random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
