from types import SimpleNamespace

import numpy as np
import pytest

from supervision.detection.core import Detections


def test_transform_remap_and_filter():
    # Simulate a model that predicts 'dog', 'cat', 'eagle', 'car'
    det = Detections(
        xyxy=np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]]),
        class_id=np.array([0, 1, 2, 3]),
        confidence=np.array([0.9, 0.8, 0.7, 0.6]),
        data={"class_name": np.array(["dog", "cat", "eagle", "car"])},
    )
    # Dataset expects 'animal', 'bird', 'car' (in that order)
    dataset = SimpleNamespace(classes=["animal", "bird", "car"])
    class_mapping = {"dog": "animal", "cat": "animal", "eagle": "bird"}
    det2 = det.transform(dataset, class_mapping=class_mapping)
    # Only 'dog', 'cat', 'eagle', 'car' should remain,
    # but 'dog' and 'cat' become 'animal', 'eagle' becomes 'bird'
    assert set(det2.data["class_name"]) <= set([*dataset.classes, "car"])
    assert all([name in dataset.classes for name in det2.data["class_name"]])
    # class_id should be remapped to dataset.classes indices
    for name, cid in zip(det2.data["class_name"], det2.class_id):
        assert dataset.classes[cid] == name
    # Only 'dog', 'cat', 'eagle', 'car' remain, but 'car' is already in dataset.classes
    assert len(det2) == 4


def test_transform_no_class_mapping():
    det = Detections(
        xyxy=np.array([[0, 0, 1, 1], [1, 1, 2, 2]]),
        class_id=np.array([0, 1]),
        confidence=np.array([0.9, 0.8]),
        data={"class_name": np.array(["car", "truck"])},
    )
    dataset = SimpleNamespace(classes=["car"])
    det2 = det.transform(dataset)
    assert len(det2) == 1
    assert det2.data["class_name"][0] == "car"
    assert det2.class_id[0] == 0


def test_transform_raises_without_class_name():
    det = Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
        data={},
    )
    dataset = SimpleNamespace(classes=["car"])
    with pytest.raises(ValueError):
        det.transform(dataset)
