from __future__ import annotations

import copy
import os
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.detection.utils.converters import mask_to_polygons
from supervision.detection.utils.polygons import (
    approximate_polygon,
    filter_polygons_by_area,
)

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

T = TypeVar("T")
"""
COCO Dataset Splitting Utilities for Supervision
Enhanced version based on community contribution for efficient COCO dataset splitting.
"""

import json
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict
from PIL import Image

try:
    from pycocotools.coco import COCO
except ImportError:
    raise ImportError(
        "pycocotools is required for COCO dataset splitting. "
        "Install it with: pip install pycocotools"
    )


def approximate_mask_with_polygons(
    mask: np.ndarray,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> list[np.ndarray]:
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
    rle: npt.NDArray[np.int_] | list[int], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """
    Converts run-length encoding (RLE) to a binary mask.

    Args:
        rle (Union[npt.NDArray[np.int_], List[int]]): The 1D RLE array, the format
            used in the COCO dataset (column-wise encoding, values of an array with
            even indices represent the number of pixels assigned as background,
            values of an array with odd indices represent the number of pixels
            assigned as foreground object).
        resolution_wh (Tuple[int, int]): The width (w) and height (h)
            of the desired binary mask.

    Returns:
        The generated 2D Boolean mask of shape `(h, w)`, where the foreground object is
            marked with `True`'s and the rest is filled with `False`'s.

    Raises:
        AssertionError: If the sum of pixels encoded in RLE differs from the
            number of pixels in the expected mask (computed based on resolution_wh).

    Examples:
        ```python
        import supervision as sv

        sv.rle_to_mask([5, 2, 2, 2, 5], (4, 4))
        # array([
        #     [False, False, False, False],
        #     [False, True,  True,  False],
        #     [False, True,  True,  False],
        #     [False, False, False, False],
        # ])
        ```
    """
    if isinstance(rle, list):
        rle = np.array(rle, dtype=int)

    width, height = resolution_wh

    assert width * height == np.sum(rle), (
        "the sum of the number of pixels in the RLE must be the same "
        "as the number of pixels in the expected mask"
    )

    zero_one_values = np.zeros(shape=(rle.size, 1), dtype=np.uint8)
    zero_one_values[1::2] = 1

    decoded_rle = np.repeat(zero_one_values, rle, axis=0)
    decoded_rle = np.append(
        decoded_rle, np.zeros(width * height - len(decoded_rle), dtype=np.uint8)
    )
    return decoded_rle.reshape((height, width), order="F")


def mask_to_rle(mask: npt.NDArray[np.bool_]) -> list[int]:
    """
    Converts a binary mask into a run-length encoding (RLE).

    Args:
        mask (npt.NDArray[np.bool_]): 2D binary mask where `True` indicates foreground
            object and `False` indicates background.

    Returns:
        The run-length encoded mask. Values of a list with even indices
            represent the number of pixels assigned as background (`False`), values
            of a list with odd indices represent the number of pixels assigned
            as foreground object (`True`).

    Raises:
        AssertionError: If input mask is not 2D or is empty.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        mask = np.array([
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ])
        sv.mask_to_rle(mask)
        # [0, 16]

        mask = np.array([
            [False, False, False, False],
            [False, True,  True,  False],
            [False, True,  True,  False],
            [False, False, False, False],
        ])
        sv.mask_to_rle(mask)
        # [5, 2, 2, 2, 5]
        ```

    ![mask_to_rle](https://media.roboflow.com/supervision-docs/mask-to-rle.png){ align=center width="800" }
    """  # noqa E501 // docs
    assert mask.ndim == 2, "Input mask must be 2D"
    assert mask.size != 0, "Input mask cannot be empty"

    on_value_change_indices = np.where(
        mask.ravel(order="F") != np.roll(mask.ravel(order="F"), 1)
    )[0]

    on_value_change_indices = np.append(on_value_change_indices, mask.size)
    # need to add 0 at the beginning when the same value is in the first and
    # last element of the flattened mask
    if on_value_change_indices[0] != 0:
        on_value_change_indices = np.insert(on_value_change_indices, 0, 0)

    rle = np.diff(on_value_change_indices)

    if mask[0][0] == 1:
        rle = np.insert(rle, 0, 0)

    return list(rle)


def split_coco_dataset(
    annotations_path: Union[str, Path],
    images_directory: Union[str, Path],
    output_directory: Union[str, Path],
    val_percentage: float = 0.15,
    test_percentage: float = 0.15,
    random_state: Optional[int] = None,
    verify_images: bool = True,
    min_annotations_per_image: int = 1
) -> Dict[str, Dict]:
    """
    Split a COCO dataset into train, validation, and test sets.
    
    This function efficiently splits COCO format datasets while preserving the
    annotation structure and optionally verifying image existence. It handles
    real-world scenarios like missing images and ensures each split contains
    only images with annotations.
    
    Args:
        annotations_path: Path to the COCO annotations JSON file
        images_directory: Directory containing the dataset images
        output_directory: Directory where split annotation files will be saved
        val_percentage: Percentage of data for validation (0.0 to 1.0)
        test_percentage: Percentage of data for testing (0.0 to 1.0)
        random_state: Random seed for reproducible splits. If None, uses numpy default.
        verify_images: If True, verifies image files exist before including in splits
        min_annotations_per_image: Minimum number of annotations required per image
        
    Returns:
        Dictionary containing statistics for each split:
        {
            'train': {'images': int, 'annotations': int},
            'val': {'images': int, 'annotations': int}, 
            'test': {'images': int, 'annotations': int},
            'total': {'images': int, 'annotations': int}
        }
        
    Raises:
        FileNotFoundError: If annotations file or images directory doesn't exist
        ValueError: If percentages are invalid or sum > 1.0
        ImportError: If pycocotools is not installed
        
    Example:
        ```python
        import supervision as sv
        
        # Split COCO dataset with verification
        stats = sv.split_coco_dataset(
            annotations_path="annotations.json",
            images_directory="images/",
            output_directory="splits/",
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            verify_images=True
        )
        
        print(f"Train: {stats['train']['images']} images, {stats['train']['annotations']} annotations")
        print(f"Val: {stats['val']['images']} images, {stats['val']['annotations']} annotations") 
        print(f"Test: {stats['test']['images']} images, {stats['test']['annotations']} annotations")
        ```
    """
    # Validate inputs
    annotations_path = Path(annotations_path)
    images_directory = Path(images_directory)
    output_directory = Path(output_directory)
    
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    if not images_directory.exists():
        raise FileNotFoundError(f"Images directory not found: {images_directory}")
    
    if not (0 <= val_percentage <= 1 and 0 <= test_percentage <= 1):
        raise ValueError("Percentages must be between 0 and 1")
    
    if val_percentage + test_percentage >= 1.0:
        raise ValueError("Sum of val_percentage and test_percentage must be < 1.0")
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Load COCO dataset
    print(f"Loading COCO dataset from {annotations_path}...")
    coco_data = COCO(str(annotations_path))
    
    # Calculate split sizes
    total_images = len(coco_data.imgs)
    val_count = int(np.floor(total_images * val_percentage))
    test_count = int(np.floor(total_images * test_percentage))
    train_count = total_images - val_count - test_count
    
    print(f"""
Dataset Split Configuration:
    Total images: {total_images}
    Train: {train_count} ({(1 - val_percentage - test_percentage):.1%})
    Validation: {val_count} ({val_percentage:.1%})
    Test: {test_count} ({test_percentage:.1%})
    """)
    
    # Initialize split dictionaries
    train_data = _initialize_split_dict(coco_data)
    val_data = _initialize_split_dict(coco_data) 
    test_data = _initialize_split_dict(coco_data)
    
    # Create random image order
    image_ids = list(coco_data.imgs.keys())
    random_indices = np.random.choice(len(image_ids), size=len(image_ids), replace=False)
    
    # Track processing statistics
    processed_count = 0
    skipped_count = 0
    
    # Process images and assign to splits
    print("Processing images and annotations...")
    
    for idx, random_idx in enumerate(random_indices):
        img_id = image_ids[random_idx]
        img_info = coco_data.imgs[img_id]
        
        try:
            # Check if image has minimum required annotations
            annotations = coco_data.imgToAnns.get(img_id, [])
            if len(annotations) < min_annotations_per_image:
                skipped_count += 1
                continue
            
            # Verify image exists if requested
            if verify_images:
                image_path = images_directory / img_info['file_name']
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                    skipped_count += 1
                    continue
                
                # Verify image can be opened
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"Warning: Cannot open image {image_path}: {e}")
                    skipped_count += 1
                    continue
            
            # Assign to appropriate split based on index
            if idx < val_count:
                target_data = val_data
            elif idx < val_count + test_count:
                target_data = test_data
            else:
                target_data = train_data
            
            # Add image and annotations to split
            target_data['images'].append(img_info)
            target_data['annotations'].extend(annotations)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            skipped_count += 1
            continue
    
    print(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Save split files
    split_files = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    statistics = {}
    
    for split_name, split_data in split_files.items():
        if split_data['images']:  # Only save non-empty splits
            output_file = output_directory / f"{split_name}_annotations.json"
            
            # Create final COCO format structure
            final_data = {
                'info': coco_data.dataset.get('info', {}),
                'licenses': coco_data.dataset.get('licenses', []),
                'categories': split_data['categories'],
                'images': split_data['images'],
                'annotations': split_data['annotations']
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            print(f"Saved {output_file.name}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")
            
            # Collect statistics
            statistics[split_name] = {
                'images': len(split_data['images']),
                'annotations': len(split_data['annotations'])
            }
        else:
            statistics[split_name] = {'images': 0, 'annotations': 0}
    
    # Add total statistics
    statistics['total'] = {
        'images': sum(stats['images'] for stats in statistics.values()),
        'annotations': sum(stats['annotations'] for stats in statistics.values())
    }
    
    print(f"\nSplit Summary:")
    for split_name, stats in statistics.items():
        if split_name != 'total':
            print(f"  {split_name.capitalize()}: {stats['images']} images, {stats['annotations']} annotations")
    
    return statistics


def _initialize_split_dict(coco_data: COCO) -> Dict[str, List]:
    """
    Initialize a dictionary structure for a dataset split.
    
    Args:
        coco_data: COCO dataset object
        
    Returns:
        Dictionary with empty lists for images, annotations, and copied categories
    """
    return {
        'categories': list(coco_data.cats.values()),
        'images': [],
        'annotations': []
    }


def verify_coco_split(
    train_annotations: Union[str, Path],
    val_annotations: Union[str, Path],
    test_annotations: Union[str, Path],
    original_annotations: Union[str, Path]
) -> Dict[str, bool]:
    """
    Verify that COCO dataset splits are valid and complete.
    
    This function checks that:
    1. All split files are valid COCO format
    2. No images are duplicated across splits
    3. Total images/annotations match original dataset
    4. All categories are preserved in each split
    
    Args:
        train_annotations: Path to training annotations file
        val_annotations: Path to validation annotations file  
        test_annotations: Path to test annotations file
        original_annotations: Path to original annotations file
        
    Returns:
        Dictionary with verification results:
        {
            'valid_format': bool,
            'no_duplicates': bool, 
            'complete_split': bool,
            'categories_preserved': bool,
            'details': Dict[str, any]
        }
        
    Example:
        ```python
        import supervision as sv
        
        # Verify split integrity
        verification = sv.verify_coco_split(
            train_annotations="splits/train_annotations.json",
            val_annotations="splits/val_annotations.json", 
            test_annotations="splits/test_annotations.json",
            original_annotations="original_annotations.json"
        )
        
        if all(verification.values()):
            print("✅ Split verification passed!")
        else:
            print("❌ Split verification failed!")
            print(verification['details'])
        ```
    """
    results = {
        'valid_format': True,
        'no_duplicates': True,
        'complete_split': True,
        'categories_preserved': True,
        'details': {}
    }
    
    try:
        # Load all datasets
        original = COCO(str(original_annotations))
        train = COCO(str(train_annotations)) if Path(train_annotations).exists() else None
        val = COCO(str(val_annotations)) if Path(val_annotations).exists() else None
        test = COCO(str(test_annotations)) if Path(test_annotations).exists() else None
        
        splits = {'train': train, 'val': val, 'test': test}
        valid_splits = {k: v for k, v in splits.items() if v is not None}
        
        # Check for duplicated images across splits
        all_image_ids = []
        for split_name, split_data in valid_splits.items():
            image_ids = list(split_data.imgs.keys())
            all_image_ids.extend(image_ids)
            results['details'][f'{split_name}_images'] = len(image_ids)
            results['details'][f'{split_name}_annotations'] = len(split_data.anns)
        
        # Check for duplicates
        if len(all_image_ids) != len(set(all_image_ids)):
            results['no_duplicates'] = False
            results['details']['duplicate_images'] = len(all_image_ids) - len(set(all_image_ids))
        
        # Check completeness
        original_image_count = len(original.imgs)
        split_image_count = len(set(all_image_ids))
        
        if original_image_count != split_image_count:
            results['complete_split'] = False
            results['details']['missing_images'] = original_image_count - split_image_count
        
        # Check category preservation
        original_categories = set(cat['id'] for cat in original.cats.values())
        for split_name, split_data in valid_splits.items():
            split_categories = set(cat['id'] for cat in split_data.cats.values())
            if original_categories != split_categories:
                results['categories_preserved'] = False
                results['details'][f'{split_name}_missing_categories'] = original_categories - split_categories
        
        results['details']['original_images'] = original_image_count
        results['details']['original_annotations'] = len(original.anns)
        results['details']['split_images_total'] = split_image_count
        
    except Exception as e:
        results['valid_format'] = False
        results['details']['error'] = str(e)
    
    return results


# Backward compatibility - alias to match your original function name
def split_data(
    root: Union[str, Path],
    imgs_path: Union[str, Path], 
    val_perc: float = 0.15,
    test_perc: float = 0.15
) -> Dict[str, Dict]:
    """
    Backward compatibility wrapper for the original split_data function.
    
    Args:
        root: Path to COCO annotations file
        imgs_path: Path to images directory
        val_perc: Validation percentage (0.0 to 1.0)
        test_perc: Test percentage (0.0 to 1.0)
        
    Returns:
        Split statistics dictionary
    """
    import warnings
    warnings.warn(
        "split_data() is deprecated. Use split_coco_dataset() instead for better functionality.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Extract directory from root path for output
    output_dir = Path(root).parent / "splits"
    
    return split_coco_dataset(
        annotations_path=root,
        images_directory=imgs_path,
        output_directory=output_dir,
        val_percentage=val_perc,
        test_percentage=test_perc,
        verify_images=True
    )