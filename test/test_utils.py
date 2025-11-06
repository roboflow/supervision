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


def mock_boxes(
    n: int,
    resolution_wh: tuple[int, int] = (1920, 1080),
    min_size: int = 20,
    max_size: int = 200,
    seed: int | None = None,
) -> list[list[float]]:
    """
    Generate N valid bounding boxes of format [x_min, y_min, x_max, y_max].

    Args:
        n: Number of boxes to generate.
        resolution_wh: Image resolution as (width, height). Defaults to (1920, 1080).
        min_size: Minimum box size (width/height). Defaults to 20.
        max_size: Maximum box size (width/height). Defaults to 200.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        List of boxes, each as [x_min, y_min, x_max, y_max].
    """
    if seed is not None:
        random.seed(seed)
    width, height = resolution_wh
    boxes = []
    for _ in range(n):
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)
        x1 = random.uniform(0, width - w)
        y1 = random.uniform(0, height - h)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append([x1, y1, x2, y2])
    return boxes


def assert_almost_equal(actual, expected, tolerance=1e-5):
    assert abs(actual - expected) < tolerance, f"Expected {expected}, but got {actual}."
