"""
Helper functions and utilities for testing the `supervision` library.

This module provides convenient factory functions for creating `Detections`
and `KeyPoints` objects from simple list-based inputs, as well as utilities
for generating synthetic test data and performing custom assertions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from supervision.detection.core import Detections
from supervision.key_points.core import KeyPoints


def _create_detections(
    xyxy: list[list[float]],
    mask: list[np.ndarray] | None = None,
    confidence: list[float] | None = None,
    class_id: list[int] | None = None,
    tracker_id: list[int] | None = None,
    data: dict[str, list[Any]] | None = None,
) -> Detections:
    """
    Create a Detections object from list-based inputs.

    This is a helper function primarily used for testing purposes to quickly
    instantiate a Detections object without manually converting lists to numpy arrays.

    Args:
        xyxy: Bounding boxes in `(x_min, y_min, x_max, y_max)`
            format.
        mask: Binary masks for each detection.
        confidence: Confidence scores for each detection.
        class_id: Class identifiers for each detection.
        tracker_id: Tracker identifiers for each detection.
        data: Additional data to be associated with
            each detection.

    Returns:
        A Detections object containing the provided data.

    Examples:
        >>> import numpy as np
        >>> detections = _create_detections(
        ...     xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]],
        ...     confidence=[0.5, 0.8],
        ...     class_id=[0, 1]
        ... )
        >>> detections.xyxy
        array([[ 0.,  0., 10., 10.],
               [20., 20., 30., 30.]], dtype=float32)
        >>> detections.confidence
        array([0.5, 0.8], dtype=float32)
        >>> detections.class_id
        array([0, 1])
    """

    def convert_data(data: dict[str, list[Any]]):
        return {k: np.array(v) for k, v in data.items()}

    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=(mask if mask is None else np.array(mask, dtype=bool)),
        confidence=(
            confidence if confidence is None else np.array(confidence, dtype=np.float32)
        ),
        class_id=(class_id if class_id is None else np.array(class_id, dtype=int)),
        tracker_id=(
            tracker_id if tracker_id is None else np.array(tracker_id, dtype=int)
        ),
        data=convert_data(data) if data else {},
    )


def _create_key_points(
    xy: list[list[list[float]]],
    confidence: list[list[float]] | None = None,
    class_id: list[int] | None = None,
    data: dict[str, list[Any]] | None = None,
) -> KeyPoints:
    """
    Create a KeyPoints object from list-based inputs.

    This is a helper function primarily used for testing purposes to quickly
    instantiate a KeyPoints object without manually converting lists to numpy arrays.

    Args:
        xy: Keypoint coordinates in `(x, y)` format for
            each detection.
        confidence: Confidence scores for each keypoint.
        class_id: Class identifiers for each keypoint set.
        data: Additional data to be associated with
            each keypoint set.

    Returns:
        A KeyPoints object containing the provided data.

    Examples:
        >>> import numpy as np
        >>> key_points = _create_key_points(
        ...     xy=[[[0, 0], [10, 10]], [[20, 20], [30, 30]]],
        ...     confidence=[[0.5, 0.8], [0.9, 0.1]],
        ...     class_id=[0, 1]
        ... )
        >>> key_points.xy
        array([[[ 0.,  0.],
                [10., 10.]],
        <BLANKLINE>
               [[20., 20.],
                [30., 30.]]], dtype=float32)
        >>> key_points.confidence
        array([[0.5, 0.8],
               [0.9, 0.1]], dtype=float32)
        >>> key_points.class_id
        array([0, 1])
    """

    def convert_data(data: dict[str, list[Any]]):
        return {k: np.array(v) for k, v in data.items()}

    return KeyPoints(
        xy=np.array(xy, dtype=np.float32),
        confidence=(
            confidence if confidence is None else np.array(confidence, dtype=np.float32)
        ),
        class_id=(class_id if class_id is None else np.array(class_id, dtype=int)),
        data=convert_data(data) if data else {},
    )


def _generate_random_boxes(
    count: int,
    image_size: tuple[int, int] = (1920, 1080),
    min_box_size: int = 20,
    max_box_size: int = 200,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate random bounding boxes within given image dimensions and size constraints.

    Creates `count` bounding boxes randomly positioned and sized, ensuring each
    stays within image bounds and has width and height in the specified range.

    Args:
        count: Number of random bounding boxes to generate.
        image_size: Image size as `(width, height)`.
        min_box_size: Minimum side length (pixels) for generated boxes.
        max_box_size: Maximum side length (pixels) for generated boxes.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape `(count, 4)` with bounding boxes as
            `(x_min, y_min, x_max, y_max)`.

    Examples:
        >>> boxes = _generate_random_boxes(
        ...     count=2, image_size=(1000, 1000),
        ...     min_box_size=10, max_box_size=20, seed=42)
        >>> boxes.shape
        (2, 4)
        >>> boxes
        array([[843.36676, 687.33374, 861.1063 , 701.72253],
               [752.81146, 770.53467, 763.75323, 790.2909 ]], dtype=float32)
    """
    rng = np.random.default_rng(seed)

    img_w, img_h = image_size
    out = np.zeros((count, 4), dtype=np.float32)

    for i in range(count):
        w = rng.uniform(min_box_size, max_box_size)
        h = rng.uniform(min_box_size, max_box_size)

        x_min = rng.uniform(0, img_w - w)
        y_min = rng.uniform(0, img_h - h)
        x_max = x_min + w
        y_max = y_min + h

        out[i] = (x_min, y_min, x_max, y_max)

    return out


def assert_almost_equal(actual, expected, tolerance=1e-5):
    """
    Assert that two values are equal within a specified tolerance.

    Args:
        actual: The value to check.
        expected: The expected value.
        tolerance: The maximum allowed difference between `actual`
            and `expected`.

    Examples:
        >>> assert_almost_equal(0.500001, 0.5)
        >>> assert_almost_equal(0.6, 0.5, tolerance=0.2)
        >>> assert_almost_equal(0.6, 0.5)
        Traceback (most recent call last):
            ...
        AssertionError: Expected 0.5, but got 0.6.
    """
    assert abs(actual - expected) < tolerance, f"Expected {expected}, but got {actual}."


def assert_image_mostly_same(
    original: np.ndarray, annotated: np.ndarray, similarity_threshold: float = 0.9
) -> None:
    """
    Assert that the annotated image is mostly the same as the original.

    Args:
        original: Original image
        annotated: Annotated image
        similarity_threshold:
          Minimum percentage of pixels that should be the same (0.0 to 1.0)
    """
    # Check that images have the same shape
    assert original.shape == annotated.shape

    # Calculate number of identical pixels
    identical_pixels = np.sum(np.all(original == annotated, axis=-1))
    total_pixels = original.shape[0] * original.shape[1]
    similarity = identical_pixels / total_pixels

    # Check that at least similarity_threshold of pixels are identical
    assert similarity >= similarity_threshold, (
        f"Images are only {similarity:.1%} similar, "
        f"which is below the {similarity_threshold:.1%} threshold"
    )

    # Check that the image is not completely identical
    assert not np.array_equal(original, annotated), "Images are completely identical"


class _FakeTensor:
    """Minimal tensor wrapper for cpu().numpy() and int()."""

    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def int(self) -> _FakeTensor:
        return _FakeTensor(self._arr.astype(int))


class _FakeYOLOv5Results:
    """YOLOv5-like results exposing pred list."""

    def __init__(self, pred0: np.ndarray):
        self.pred = [_FakeTensor(pred0)]


class _FakeUltralyticsBoxes:
    """Ultralytics-like Boxes exposing xyxy/conf/cls and optional id."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        id_: np.ndarray | None = None,
    ):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)
        self.id = _FakeTensor(id_) if id_ is not None else None


class _FakeUltralyticsResults:
    """Ultralytics-like results container used by from_ultralytics."""

    def __init__(self, boxes, names: dict[int, str], length: int = 0):
        self.boxes = boxes
        self.names = names
        self.obb = None
        self.masks = None
        self._length = length

    def __len__(self) -> int:
        return self._length


class _FakeYoloNasPrediction:
    """YOLO-NAS-like prediction struct."""

    def __init__(self, bboxes_xyxy, confidence, labels):
        self.bboxes_xyxy = bboxes_xyxy
        self.confidence = confidence
        self.labels = labels


class _FakeYoloNasResults:
    """YOLO-NAS-like results exposing prediction."""

    def __init__(self, prediction: _FakeYoloNasPrediction):
        self.prediction = prediction


def create_yolo_dataset(
    dataset_dir: str,
    num_images: int = 15,
    image_size: tuple[int, int, int] = (640, 640, 3),
    classes: list[str] | None = None,
    objects_per_image_range: tuple[int, int] = (2, 4),
    seed: int = 42,
) -> dict[str, Any]:
    """
    Create a synthetic YOLO-format dataset on disk.

    Generates dummy images with YOLO-format annotations, `data.yaml` file,
    and directory structure suitable for testing dataset loading.

    Args:
        dataset_dir: Root directory path for the dataset.
        num_images: Number of images to generate.
        image_size: Image dimensions as `(width, height, channels)`.
        classes: List of class names. Defaults to `["class_0", "class_1"]`.
        objects_per_image_range: Range of objects per image as `(min, max)`.
            Actual count will cycle through this range.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - `tmpdir`: Root dataset directory path
            - `images_dir`: Images directory path
            - `labels_dir`: Labels directory path
            - `data_yaml_path`: `data.yaml` file path
            - `num_images`: Number of images created
            - `image_size`: Image dimensions
            - `image_annotations`: List of annotations per image

    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> tmpdir = Path(tempfile.mkdtemp())
        >>> dataset_info = create_yolo_dataset(str(tmpdir), num_images=5)
        >>> dataset_info["num_images"]
        5
        >>> len(list(Path(dataset_info["images_dir"]).glob("*.jpg")))
        5
    """
    from pathlib import Path

    import cv2

    if classes is None:
        classes = ["class_0", "class_1"]

    np.random.seed(seed)

    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    min_objects, max_objects = objects_per_image_range
    num_classes = len(classes)
    image_annotations = []

    for i in range(num_images):
        # Create dummy image
        img_path = images_dir / f"image_{i:03d}.jpg"
        img = np.zeros(image_size, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        # Determine number of objects for this image
        num_objects = min_objects + (i % (max_objects - min_objects + 1))
        yolo_lines = []
        objects = []

        for j in range(num_objects):
            class_id = j % num_classes
            # Random positions with spacing to avoid overlap
            x_center = 0.15 + (j * 0.25) + np.random.uniform(-0.05, 0.05)
            y_center = 0.15 + (j * 0.2) + np.random.uniform(-0.05, 0.05)
            width = 0.12
            height = 0.12

            # Clip to valid range [0, 1]
            x_center = np.clip(x_center, width / 2, 1 - width / 2)
            y_center = np.clip(y_center, height / 2, 1 - height / 2)

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )
            objects.append((class_id, x_center, y_center, width, height))

        # Write YOLO annotation file
        label_path = labels_dir / f"image_{i:03d}.txt"
        label_path.write_text("".join(yolo_lines))
        image_annotations.append(objects)

    # Create data.yaml
    data_yaml_path = dataset_path / "data.yaml"
    yaml_content = "names:\n" + "\n".join(f"- {cls}" for cls in classes) + "\n"
    data_yaml_path.write_text(yaml_content)

    return {
        "tmpdir": dataset_path,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "data_yaml_path": str(data_yaml_path),
        "num_images": num_images,
        "image_size": image_size,
        "image_annotations": image_annotations,
    }


def create_predictions_with_class_iou_tests(
    gt_detections: Detections, num_classes: int
) -> Detections:
    """
    Create predictions that test IoU+class matching behavior.

    For each ground truth detection, creates predictions with different patterns:
    - Pattern 0 (i%3==0): Correct match (same bbox, correct class)
    - Pattern 1 (i%3==1): Wrong class with perfect IoU + correct class with offset
    - Pattern 2 (i%3==2): Correct class with slight offset

    This tests that predictions with wrong class don't match even with high IoU,
    which is the key fix in the confusion matrix calculation.

    Args:
        gt_detections: Ground truth detections to create predictions for
        num_classes: Total number of classes in the dataset

    Returns:
        Detections object with predictions designed to test IoU+class matching
    """
    if len(gt_detections) == 0:
        # No ground truth, return a single false positive
        return _create_detections(
            xyxy=[[10, 10, 50, 50]], class_id=[0], confidence=[0.9]
        )

    pred_boxes = []
    pred_classes = []
    pred_confs = []

    for i, (box, cls) in enumerate(zip(gt_detections.xyxy, gt_detections.class_id)):
        if i % 3 == 0:
            # Pattern 1: Correct match
            pred_boxes.append(box)
            pred_classes.append(cls)
            pred_confs.append(0.95)

        elif i % 3 == 1:
            # Pattern 2: Test the fix - add wrong class prediction with perfect IoU,
            # then correct class with slightly offset bbox
            wrong_cls = (cls + 1) % num_classes
            pred_boxes.append(box)  # Perfect IoU
            pred_classes.append(wrong_cls)  # Wrong class
            pred_confs.append(0.90)

            # Add correct class with slight offset
            offset_box = box + np.array([2, 2, 2, 2], dtype=np.float32)
            pred_boxes.append(offset_box)
            pred_classes.append(cls)  # Correct class
            pred_confs.append(0.85)

        else:
            # Pattern 3: Correct match with slight offset
            offset_box = box + np.array([1, 1, 1, 1], dtype=np.float32)
            pred_boxes.append(offset_box)
            pred_classes.append(cls)
            pred_confs.append(0.92)

    return _create_detections(
        xyxy=pred_boxes, class_id=pred_classes, confidence=pred_confs
    )
