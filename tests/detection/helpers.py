"""Shared Detections fixtures and OBB builders for detection tests."""

import numpy as np

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import (
    Detections,
)
from supervision.detection.utils.boxes import xyxyxyxy_to_xyxy

# Merge test
TEST_MASK = np.zeros((1000, 1000), dtype=bool)


TEST_MASK[300:351, 200:251] = True


TEST_DET_1 = Detections(
    xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]]),
    mask=np.array([TEST_MASK, TEST_MASK, TEST_MASK]),
    confidence=np.array([0.1, 0.2, 0.3]),
    class_id=np.array([1, 2, 3]),
    tracker_id=np.array([1, 2, 3]),
    data={
        "some_key": [1, 2, 3],
        "other_key": [["1", "2"], ["3", "4"], ["5", "6"]],
    },
)


def _rotated_rect(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    rotated: np.ndarray = (corners @ rot.T + [cx, cy]).astype(np.float32)
    return rotated


def _make_obb_detections(
    quads: list[np.ndarray], scores: list[float], class_ids: list[int]
) -> Detections:
    """Build OBB Detections from a list of (4, 2) corner arrays."""
    oriented_boxes = np.stack(quads)
    xyxy = xyxyxyxy_to_xyxy(oriented_boxes)
    return Detections(
        xyxy=xyxy,
        confidence=np.array(scores, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        data={ORIENTED_BOX_COORDINATES: oriented_boxes},
    )
