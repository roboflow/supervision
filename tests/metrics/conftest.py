from typing import Optional

import numpy as np
import pytest

from supervision.detection.core import Detections


@pytest.fixture
def detections_50_50():
    return Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


@pytest.fixture
def targets_50_50():
    return Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )


@pytest.fixture
def dummy_prediction():
    return Detections(
        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
        confidence=np.array([0.8]),
        class_id=np.array([0]),
    )


@pytest.fixture
def predictions_no_overlap():
    return Detections(
        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


@pytest.fixture
def targets_no_overlap():
    return Detections(
        xyxy=np.array([[100, 100, 110, 110]], dtype=np.float32),
        class_id=np.array([0]),
    )


@pytest.fixture
def targets_two_objects_class_0():
    return Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],
                [100, 100, 110, 110],
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 0]),
    )


@pytest.fixture
def predictions_multiple_classes():
    return Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # class 0, matches target
                [60, 60, 100, 100],  # class 1, matches target
                [120, 120, 130, 130],  # class 1, false positive
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.8, 0.7]),
        class_id=np.array([0, 1, 1]),
    )


@pytest.fixture
def targets_multiple_classes():
    return Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],  # class 0
                [60, 60, 100, 100],  # class 1
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
    )


@pytest.fixture
def predictions_iou_064():
    return Detections(
        xyxy=np.array([[15, 15, 55, 55]], dtype=np.float32),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


@pytest.fixture
def targets_iou_064():
    return Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float32),
        class_id=np.array([0]),
    )


@pytest.fixture
def predictions_confidence_ranking():
    return Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],
                [11, 11, 49, 49],
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.6, 0.9]),
        class_id=np.array([0, 0]),
    )


@pytest.fixture
def prediction_class_1():
    return Detections(
        xyxy=np.array([[60, 60, 100, 100]], dtype=np.float32),
        confidence=np.array([0.8]),
        class_id=np.array([1]),
    )


@pytest.fixture
def target_class_1():
    return Detections(
        xyxy=np.array([[60, 60, 100, 100]], dtype=np.float32),
        class_id=np.array([1]),
    )


def _yolo_dataset_factory(
    tmp_path,
    num_images: int = 20,
    classes: Optional[list[str]] = None,
    objects_per_image_range: tuple[int, int] = (1, 3),
):
    """
    Factory function to create synthetic YOLO-format datasets with custom parameters.

    Args:
        tmp_path: Pytest tmp_path fixture
        num_images: Number of images to generate
        classes: List of class names
        objects_per_image_range: Range of objects per image as (min, max)

    Returns:
        dict with dataset paths and metadata
    """
    from tests.helpers import create_yolo_dataset

    if classes is None:
        classes = ["dog", "cat", "person"]

    return create_yolo_dataset(
        dataset_dir=str(tmp_path / "yolo_dataset"),
        num_images=num_images,
        image_size=(640, 640, 3),
        classes=classes,
        objects_per_image_range=objects_per_image_range,
        seed=42,
    )


@pytest.fixture
def yolo_dataset_structure(tmp_path):
    """
    Synthetic YOLO-format dataset for testing confusion matrix and detection metrics.

    Configuration:
    - 20 images
    - 640x640 resolution
    - 3 classes: ["dog", "cat", "person"]
    - 1-3 objects per image

    Use this for tests that need multi-class scenarios (3+ classes).

    Returns:
        dict with dataset paths and metadata
    """
    return _yolo_dataset_factory(
        tmp_path,
        num_images=20,
        classes=["dog", "cat", "person"],
        objects_per_image_range=(1, 3),
    )


@pytest.fixture
def yolo_dataset_two_classes(tmp_path):
    """
    Synthetic YOLO-format dataset for testing mAR and binary classification metrics.

    Configuration:
    - 15 images
    - 640x640 resolution
    - 2 classes: ["class_0", "class_1"]
    - 2-4 objects per image

    Use this for tests that specifically need 2-class scenarios or depend on
    specific class distributions (e.g., mAR @ K per-image limiting tests).

    Returns:
        dict with dataset paths and metadata
    """
    return _yolo_dataset_factory(
        tmp_path,
        num_images=15,
        classes=["class_0", "class_1"],
        objects_per_image_range=(2, 4),
    )
