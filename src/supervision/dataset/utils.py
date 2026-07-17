from __future__ import annotations

__all__ = ["check_no_basename_collisions", "train_test_split"]

import copy
import os
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void  # type: ignore[import-untyped,unused-ignore]
from tqdm.auto import tqdm

from supervision import _cv2 as cv2
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


@deprecated(target=_mask_to_rle, deprecated_in="0.28.0", remove_in="0.31.0")  # type: ignore[untyped-decorator]
def mask_to_rle(
    mask: npt.NDArray[np.bool_], compressed: bool = False
) -> list[int] | str:
    """Deprecated since 0.28.0.

    Use `supervision.detection.utils.converters.mask_to_rle`.
    """
    return cast(list[int] | str, void(mask, compressed))


@deprecated(target=_rle_to_mask, deprecated_in="0.28.0", remove_in="0.31.0")  # type: ignore[untyped-decorator]
def rle_to_mask(
    rle: npt.NDArray[np.integer] | list[int] | str | bytes,
    resolution_wh: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Deprecated since 0.28.0.

    Use `supervision.detection.utils.converters.rle_to_mask`.
    """
    return cast(npt.NDArray[np.bool_], void(rle, resolution_wh))


if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

T = TypeVar("T")


def approximate_mask_with_polygons(
    mask: npt.NDArray[np.bool_],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.0,
) -> list[npt.NDArray[np.number]]:
    """Filter mask polygons by area and optionally simplify them.

    The default `approximation_percentage=0.0` preserves the original contour
    unless callers explicitly ask for simplification.
    """
    height, width = mask.shape
    image_area = height * width
    minimum_detection_area = min_image_area_percentage * image_area
    maximum_detection_area = max_image_area_percentage * image_area

    polygons = cast(list[npt.NDArray[np.number]], mask_to_polygons(mask=mask))
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


def check_no_basename_collisions(
    image_paths: list[str],
    key: Callable[[str], str],
    output_kind: str,
) -> None:
    """Raise if two image paths would be written to the same output file.

    Dataset image paths may share a basename when they originate from different
    directories (a legal, common state after :meth:`DetectionDataset.merge`).
    Exporting them into a single flat output directory keyed on the basename or
    stem would silently overwrite one file with another and mispair images with
    their annotations. This guard detects such collisions before any file is
    written and names the colliding source paths.

    Args:
        image_paths: The dataset image paths about to be written.
        key: Maps an image path to the output file name it would be written to.
        output_kind: Human-readable description of the output (e.g. ``"image"``
            or ``"YOLO annotation"``) used in the error message.

    Raises:
        ValueError: If two image paths map to the same output file name.

    Examples:
        >>> from pathlib import Path
        >>> from supervision.dataset.utils import check_no_basename_collisions
        >>> check_no_basename_collisions(
        ...     ["a/img.jpg", "b/img.jpg"], lambda p: Path(p).name, "image"
        ... )
        Traceback (most recent call last):
        ...
        ValueError: Cannot export dataset: image paths 'a/img.jpg' and ...
    """
    seen: dict[str, tuple[str, str]] = {}  # casefold(key) → (original name, image_path)
    for image_path in image_paths:
        output_name = key(image_path)
        case_key = output_name.casefold()
        if case_key in seen:
            first_name, first_path = seen[case_key]
            raise ValueError(
                f"Cannot export dataset: image paths {first_path!r} and "
                f"{image_path!r} both map to {output_kind} file {first_name!r}. "
                "Ensure all image basenames are unique before exporting."
            )
        seen[case_key] = (output_name, image_path)


def save_dataset_images(
    dataset: DetectionDataset,
    images_directory_path: str,
    show_progress: bool = False,
) -> None:
    """Save all images from a dataset to a directory.

    Images already in memory are written with ``cv2.imwrite``; images stored
    only as file paths are copied with ``shutil.copyfile``.

    Args:
        dataset: The dataset whose images are saved.
        images_directory_path: Destination directory path; created
            automatically if it does not exist.
        show_progress: If ``True``, display a tqdm progress bar while
            saving images.

    Examples:
        >>> from supervision.dataset.core import DetectionDataset
        >>> from supervision.dataset.utils import save_dataset_images
        >>> dataset = DetectionDataset(classes=["cat"], images={}, annotations={})
        >>> save_dataset_images(dataset, "/tmp/images")
    """
    check_no_basename_collisions(
        image_paths=dataset.image_paths,
        key=lambda image_path: Path(image_path).name,
        output_kind="image",
    )
    Path(images_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(
        dataset.image_paths,
        desc="Saving images",
        disable=not show_progress,
    ):
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
        The split data. The input list is copied and never mutated.

    Examples:
        >>> train, test = train_test_split(
        ...     [1, 2, 3, 4, 5], train_ratio=0.6, random_state=0
        ... )
        >>> len(train), len(test)
        (3, 2)
    """
    rng = random.Random(random_state)  # noqa: S311 — dataset split, not cryptographic
    if shuffle:
        data = list(data)
        rng.shuffle(data)
    else:
        data = list(data)  # copy to guarantee non-mutation

    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
