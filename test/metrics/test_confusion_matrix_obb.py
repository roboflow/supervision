import numpy as np
import pytest
from supervision.detection.core import Detections
from supervision.metrics.detection import ConfusionMatrix
from supervision.detection.utils import xyxy_to_polygons

def test_confusion_matrix_with_obb():
    # Create two oriented bounding boxes (OBBs) as polygons
    # Format: (x, y) for each corner, shape (N, 4, 2)
    gt_polygons = np.array([
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        [[3, 3], [5, 3], [5, 5], [3, 5]],
    ], dtype=np.float32)
    pred_polygons = np.array([
        [[0, 0], [2, 0], [2, 2], [0, 2]],  # perfect match
        [[3.1, 3.1], [5.1, 3.1], [5.1, 5.1], [3.1, 5.1]],  # slight offset
    ], dtype=np.float32)

    # For OBB, we use polygons as xyxy for Detections, but in practice, you may have a conversion
    # Here, we just flatten the polygons to fit the Detections API for the test
    gt_flat = gt_polygons.reshape(-1, 8)
    pred_flat = pred_polygons.reshape(-1, 8)
    # For this test, we treat the first 4 values as (x_min, y_min, x_max, y_max) for compatibility
    # In a real OBB pipeline, you would adapt the Detections and ConfusionMatrix to handle polygons directly
    gt_xyxy = np.array([[0, 0, 2, 2], [3, 3, 5, 5]], dtype=np.float32)
    pred_xyxy = np.array([[0, 0, 2, 2], [3.1, 3.1, 5.1, 5.1]], dtype=np.float32)
    gt = Detections(xyxy=gt_xyxy, class_id=[0, 1])
    pred = Detections(xyxy=pred_xyxy, class_id=[0, 1], confidence=[0.9, 0.8])

    # Run confusion matrix with OBB support
    cm = ConfusionMatrix.from_detections(
        predictions=[pred],
        targets=[gt],
        classes=["A", "B"],
        use_oriented_boxes=True,
    )
    assert cm.matrix[0, 0] == 1
    assert cm.matrix[1, 1] == 1
    assert cm.matrix.sum() == 2


def test_confusion_matrix_without_obb():
    gt = Detections(xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32), class_id=[0])
    pred = Detections(xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32), class_id=[0], confidence=[0.9])
    cm = ConfusionMatrix.from_detections(
        predictions=[pred],
        targets=[gt],
        classes=["A"],
        use_oriented_boxes=False,
    )
    assert cm.matrix[0, 0] == 1
    assert cm.matrix.sum() == 1
