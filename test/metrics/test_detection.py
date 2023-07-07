from contextlib import ExitStack as DoesNotRaise
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

IDEAL_RESULT = np.zeros((81, 81))
for class_id, count in zip(*np.unique(PREDICTIONS[:, 5], return_counts=True)):
    class_id = int(class_id)
    IDEAL_RESULT[class_id, class_id] = count

classes = np.arange(80)


@pytest.mark.parametrize(
    "predictions, targets, classes, conf_threshold, iou_threshold, expected_result, exception",
    [
        (
            [CERTAIN_DETECTIONS],
            [DETECTIONS],
            classes,
            0.3,
            0.5,
            IDEAL_RESULT,
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


def test_evaluate_detection_batch():
    ...


def test_drop_extra_matches():
    ...


def test_benchmark():
    ...
