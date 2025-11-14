from __future__ import annotations

import random
from typing import Any

import numpy as np

from supervision.detection.core import Detections
from supervision.key_points.core import KeyPoints


def mock_detections(
    xyxy: list[list[float]],
    mask: list[np.ndarray] | None = None,
    confidence: list[float] | None = None,
    class_id: list[int] | None = None,
    tracker_id: list[int] | None = None,
    data: dict[str, list[Any]] | None = None,
) -> Detections:
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


def mock_key_points(
    xy: list[list[list[float]]],
    confidence: list[list[float]] | None = None,
    class_id: list[int] | None = None,
    data: dict[str, list[Any]] | None = None,
) -> KeyPoints:
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


def random_boxes(
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
        count (`int`): Number of random bounding boxes to generate.
        image_size (`tuple[int, int]`): Image size as `(width, height)`. Defaults to `(1920, 1080)`.
        min_box_size (`int`): Minimum side length (pixels) for generated boxes. Defaults to `20`.
        max_box_size (`int`): Maximum side length (pixels) for generated boxes. Defaults to `200`.
        seed (`int` or `None`): Optional random seed for reproducibility. Defaults to `None`.

    Returns:
        (`numpy.ndarray`): Array of shape `(count, 4)` with bounding boxes as
            `(x_min, y_min, x_max, y_max)`.
    """
    if seed is not None:
        random.seed(seed)

    img_w, img_h = image_size
    out = np.zeros((count, 4), dtype=np.float32)

    for i in range(count):
        w = random.uniform(min_box_size, max_box_size)
        h = random.uniform(min_box_size, max_box_size)

        x_min = random.uniform(0, img_w - w)
        y_min = random.uniform(0, img_h - h)
        x_max = x_min + w
        y_max = y_min + h

        out[i] = (x_min, y_min, x_max, y_max)

    return out


def assert_almost_equal(actual, expected, tolerance=1e-5):
    assert abs(actual - expected) < tolerance, f"Expected {expected}, but got {actual}."
