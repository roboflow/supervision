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
