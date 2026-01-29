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
        >>> from test.helpers import _create_detections
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
        >>> from test.helpers import _create_key_points
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
        >>> from test.helpers import _generate_random_boxes
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
        >>> from test.helpers import assert_almost_equal
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
