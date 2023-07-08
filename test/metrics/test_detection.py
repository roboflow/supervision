from contextlib import ExitStack as DoesNotRaise
from test.utils import dummy_detection_dataset_with_map_img_to_annotation
from typing import Optional

import numpy as np
import pytest

from supervision import Detections
from supervision.metrics.detection import ConfusionMatrix

PREDICTIONS = np.array(
    [
        [2254, 906, 2447, 1353, 0.90538, 0],
        [2049, 1133, 2226, 1371, 0.59002, 56],
        [727, 1224, 838, 1601, 0.51119, 39],
        [808, 1214, 910, 1564, 0.45287, 39],
        [6, 52, 1131, 2133, 0.45057, 72],
        [299, 1225, 512, 1663, 0.45029, 39],
        [529, 874, 645, 945, 0.31101, 39],
        [8, 47, 1935, 2135, 0.28192, 72],
        [2265, 813, 2328, 901, 0.2714, 62],
    ],
    dtype=np.float32,
)

DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=PREDICTIONS[:, 4],
    class_id=PREDICTIONS[:, 5].astype(int),
)
CERTAIN_DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=np.ones_like(PREDICTIONS[:, 4]),
    class_id=PREDICTIONS[:, 5].astype(int),
)

IDEAL_MATCHES = np.stack(
    [
        np.arange(len(PREDICTIONS)),
        np.arange(len(PREDICTIONS)),
        np.ones(len(PREDICTIONS)),
    ],
    axis=1,
)

IDEAL_CONF_MATRIX = np.zeros((81, 81))
for class_id, count in zip(*np.unique(PREDICTIONS[:, 5], return_counts=True)):
    class_id = int(class_id)
    IDEAL_CONF_MATRIX[class_id, class_id] = count

CLASSES = np.arange(80)
NUM_CLASSES = len(CLASSES)


@pytest.mark.parametrize(
    "predictions, targets, classes, conf_threshold, iou_threshold, expected_result, exception",
    [
        (
            [CERTAIN_DETECTIONS],
            [DETECTIONS],
            CLASSES,
            0.3,
            0.5,
            IDEAL_CONF_MATRIX,
            DoesNotRaise(),
        )
    ],
)
def test_from_detections(
    predictions,
    targets,
    classes,
    conf_threshold,
    iou_threshold,
    expected_result: Optional[np.ndarray],
    exception: Exception,
):
    with exception:
        result = ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        assert result.matrix.diagonal().sum() == result.matrix.sum()
        assert np.array_equal(result.matrix, expected_result)


@pytest.mark.parametrize(
    "predictions, targets, num_classes, conf_threshold, iou_threshold, expected_result, exception",
    [
        (
            CERTAIN_DETECTIONS,
            DETECTIONS,
            NUM_CLASSES,
            0.3,
            0.5,
            IDEAL_CONF_MATRIX,
            DoesNotRaise(),
        )
    ],
)
def test_evaluate_detection_batch(
    predictions,
    targets,
    num_classes,
    conf_threshold,
    iou_threshold,
    expected_result: Optional[np.ndarray],
    exception: Exception,
):
    with exception:
        result = ConfusionMatrix._evaluate_detection_batch(
            true_detections=targets,
            pred_detections=predictions,
            num_classes=num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        assert result.diagonal().sum() == result.sum()
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "matches, expected_result, exception",
    [
        (
            IDEAL_MATCHES,
            IDEAL_MATCHES,
            DoesNotRaise(),
        )
    ],
)
def test_drop_extra_matches(
    matches,
    expected_result: Optional[np.ndarray],
    exception: Exception,
):
    with exception:
        result = ConfusionMatrix._drop_extra_matches(matches)

        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "dataset, conf_threshold, iou_threshold, expected_result, exception",
    [
        (
            dummy_detection_dataset_with_map_img_to_annotation(),
            0.3,
            0.5,
            IDEAL_CONF_MATRIX,
            DoesNotRaise(),
        )
    ],
)
def test_benchmark(dataset, conf_threshold, iou_threshold, expected_result, exception):
    with exception:

        def callback(img):
            return dataset.map_img_to_annotation(img)

        result = ConfusionMatrix.benchmark(
            dataset=dataset,
            callback=callback,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        assert result.matrix.diagonal().sum() == result.matrix.sum()
        assert np.array_equal(result, expected_result)


def test_from_matrix():
    matrix = np.array(
        [
            [0, 1, 0],
            [1, 4, 0],
            [0, 1, 3],
        ]
    )
    conf_threshold = 0.3
    iou_threshold = 0.5
    classes = ["a", "b", "c"]

    result = ConfusionMatrix.from_matrix(
        matrix,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
    )

    assert (
        np.array_equal(result.matrix, matrix)
        and result.conf_threshold == conf_threshold
        and result.iou_threshold == iou_threshold
        and result.classes == classes
    )
