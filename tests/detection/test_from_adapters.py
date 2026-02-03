from __future__ import annotations

import numpy as np
import pytest

import supervision.detection.core as detection_core
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from tests.helpers import (
    _FakeUltralyticsBoxes,
    _FakeUltralyticsResults,
    _FakeYoloNasPrediction,
    _FakeYoloNasResults,
    _FakeYOLOv5Results,
)


def test_from_yolov5_maps_columns_correctly() -> None:
    pred = np.array(
        [
            [10, 20, 30, 40, 0.9, 2],
            [1, 2, 3, 4, 0.1, 7],
        ],
        dtype=np.float32,
    )
    results = _FakeYOLOv5Results(pred0=pred)

    det = Detections.from_yolov5(results)

    assert isinstance(det, Detections)
    np.testing.assert_allclose(det.xyxy, pred[:, :4])
    np.testing.assert_allclose(det.confidence, pred[:, 4])
    np.testing.assert_array_equal(det.class_id, pred[:, 5].astype(int))


def test_from_ultralytics_boxes_branch_maps_fields_and_class_names() -> None:
    xyxy = np.array([[0, 0, 10, 10], [5, 6, 7, 8]], dtype=np.float32)
    conf = np.array([0.8, 0.2], dtype=np.float32)
    cls = np.array([1, 0], dtype=np.float32)
    names = {0: "cat", 1: "dog"}

    boxes = _FakeUltralyticsBoxes(xyxy=xyxy, conf=conf, cls=cls, id_=None)
    results = _FakeUltralyticsResults(boxes=boxes, names=names)

    det = Detections.from_ultralytics(results)

    np.testing.assert_allclose(det.xyxy, xyxy)
    np.testing.assert_allclose(det.confidence, conf)
    np.testing.assert_array_equal(det.class_id, cls.astype(int))
    assert det.tracker_id is None

    assert CLASS_NAME_DATA_FIELD in det.data
    expected_names = np.array([names[i] for i in cls.astype(int)])
    np.testing.assert_array_equal(det.data[CLASS_NAME_DATA_FIELD], expected_names)


def test_from_ultralytics_segmentation_only_branch_uses_masks_and_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _FakeUltralyticsResults(boxes=None, names={}, length=3)

    fake_masks = np.zeros((3, 10, 10), dtype=bool)
    fake_xyxy = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)

    monkeypatch.setattr(
        detection_core, "extract_ultralytics_masks", lambda _: fake_masks
    )
    monkeypatch.setattr(detection_core, "mask_to_xyxy", lambda masks: fake_xyxy)

    det = Detections.from_ultralytics(results)

    np.testing.assert_allclose(det.xyxy, fake_xyxy)
    np.testing.assert_array_equal(det.mask, fake_masks)
    np.testing.assert_array_equal(det.class_id, np.arange(len(results)))


@pytest.mark.parametrize(
    ("bboxes", "conf", "labels", "expected_len"),
    [
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            0,
        ),
        (
            np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32),
            np.array([0.3, 0.9], dtype=np.float32),
            np.array([5, 6], dtype=np.int64),
            2,
        ),
    ],
)
def test_from_yolo_nas_handles_empty_and_non_empty(
    bboxes: np.ndarray,
    conf: np.ndarray,
    labels: np.ndarray,
    expected_len: int,
) -> None:
    pred = _FakeYoloNasPrediction(
        bboxes_xyxy=bboxes,
        confidence=conf,
        labels=labels,
    )
    results = _FakeYoloNasResults(prediction=pred)

    det = Detections.from_yolo_nas(results)

    assert len(det) == expected_len
    if expected_len > 0:
        np.testing.assert_allclose(det.xyxy, bboxes)
        np.testing.assert_allclose(det.confidence, conf)
        np.testing.assert_array_equal(det.class_id, labels.astype(int))
