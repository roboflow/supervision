from contextlib import ExitStack as DoesNotRaise
from test.test_utils import assert_almost_equal, mock_detections
from typing import Optional, Union

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.utils import mask_iou_batch
from supervision.metrics.detection import (
    ConfusionMatrix,
    MeanAveragePrecision,
    detections_to_tensor,
)

CLASSES = np.arange(80)
NUM_CLASSES = len(CLASSES)

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

TARGET_TENSORS = [
    np.array(
        [
            [2254, 906, 2447, 1353, 0],
            [2049, 1133, 2226, 1371, 56],
            [727, 1224, 838, 1601, 39],
            [808, 1214, 910, 1564, 39],
            [6, 52, 1131, 2133, 72],
            [299, 1225, 512, 1663, 39],
            [529, 874, 645, 945, 39],
            [8, 47, 1935, 2135, 72],
            [2265, 813, 2328, 901, 62],
        ]
    )
]

DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=PREDICTIONS[:, 4],
    class_id=PREDICTIONS[:, 5].astype(int),
)
CERTAIN_DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=np.ones(len(PREDICTIONS)),
    class_id=PREDICTIONS[:, 5].astype(int),
)

DETECTION_TENSORS = [
    np.concatenate(
        [
            det.xyxy,
            np.expand_dims(det.class_id, 1),
            np.expand_dims(det.confidence, 1),
        ],
        axis=1,
    )
    for det in [DETECTIONS]
]


CERTAIN_DETECTION_TENSORS = [
    np.concatenate(
        [
            det.xyxy,
            np.expand_dims(det.class_id, 1),
            np.ones((len(det), 1)),
        ],
        axis=1,
    )
    for det in [DETECTIONS]
]

CUSTOM_DETECTION_TENSORS_1 = [
    np.array(
        [
            [0.0, 0.0, 3.0, 3.0, 0, 0.9],  # correct detection of [0]
            [
                0.1,
                0.1,
                3.0,
                3.0,
                0,
                0.9,
            ],  # additional detection of [0] - FP
            [
                6.0,
                1.0,
                8.0,
                3.0,
                1,
                0.8,
            ],  # correct detection with incorrect class
            [1.0, 6.0, 2.0, 7.0, 1, 0.8],  # incorrect detection - FP
            [
                1.0,
                2.0,
                2.0,
                4.0,
                1,
                0.8,
            ],  # incorrect detection with low IoU - FP
        ]
    )
]

CUSTOM_TARGET_TENSORS_1 = [
    np.array(
        [
            [0.0, 0.0, 3.0, 3.0, 0],  # [0] detected
            [2.0, 2.0, 5.0, 5.0, 1],  # [1] undetected - FN
            [
                6.0,
                1.0,
                8.0,
                3.0,
                2,
            ],  # [2] correct detection with incorrect class
        ]
    )
]

CUSTOM_DETECTION_TENSORS_2 = [
    np.array(
        [
            [0.0, 0.0, 3.0, 3.0, 0, 0.9],  # correct detection of [0]
            [
                0.1,
                0.1,
                3.0,
                3.0,
                0,
                0.9,
            ],  # additional detection of [0] - FP
            [
                6.0,
                1.0,
                8.0,
                3.0,
                1,
                0.8,
            ],  # correct detection with incorrect class
            [1.0, 6.0, 2.0, 7.0, 1, 0.8],  # incorrect detection - FP
            [
                1.0,
                2.0,
                2.0,
                4.0,
                1,
                0.8,
            ],  # incorrect detection with low IoU - FP
        ]
    )
]

CUSTOM_TARGET_TENSORS_2 = [
    np.array(
        [
            [0.0, 0.0, 3.0, 3.0, 0],  # [0] detected
            [2.0, 2.0, 5.0, 5.0, 1],  # [1] undetected - FN
            [
                6.0,
                1.0,
                8.0,
                3.0,
                2,
            ],  # [2] correct detection with incorrect class
        ]
    )
]


IDEAL_MATCHES = np.stack(
    [
        np.arange(len(PREDICTIONS)),
        np.arange(len(PREDICTIONS)),
        np.ones(len(PREDICTIONS)),
    ],
    axis=1,
)


def create_mask_from_boxes(boxes, max_x, max_y):
    num_boxes = len(boxes)
    masks = np.zeros((num_boxes, max_y, max_x), dtype=bool)  # Create an empty 3D array

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        masks[
            idx, y1 : y2 + 1, x1 : x2 + 1
        ] = True  # Directly fill the appropriate slice of the 3D array

    return masks


def create_masks_from_tensors(detection_tensors, target_tensors):
    max_x = int(np.max(detection_tensors[0][:, [0, 2]]))
    max_x = max(max_x, int(np.max(target_tensors[0][:, [0, 2]])))
    max_y = int(np.max(detection_tensors[0][:, [1, 3]]))
    max_y = max(max_x, int(np.max(target_tensors[0][:, [1, 3]])))

    detection_masks = create_mask_from_boxes(detection_tensors[0][:, :4], max_x, max_y)
    target_masks = create_mask_from_boxes(target_tensors[0][:, :4], max_x, max_y)

    return [detection_masks], [target_masks]


def create_empty_conf_matrix(num_classes: int, do_add_dummy_class: bool = True):
    if do_add_dummy_class:
        num_classes += 1
    return np.zeros((num_classes, num_classes))


def update_ideal_conf_matrix(conf_matrix: np.ndarray, class_ids: np.ndarray):
    for class_id, count in zip(*np.unique(class_ids, return_counts=True)):
        class_id = int(class_id)
        conf_matrix[class_id, class_id] += count
    return conf_matrix


def worsen_ideal_conf_matrix(
    conf_matrix: np.ndarray, class_ids: Union[np.ndarray, list]
):
    for class_id in class_ids:
        class_id = int(class_id)
        conf_matrix[class_id, class_id] -= 1
        conf_matrix[class_id, 80] += 1
    return conf_matrix


DETECTION_MASKS, TARGET_MASKS = create_masks_from_tensors(
    DETECTION_TENSORS, TARGET_TENSORS
)


IDEAL_CONF_MATRIX = create_empty_conf_matrix(NUM_CLASSES)
IDEAL_CONF_MATRIX = update_ideal_conf_matrix(IDEAL_CONF_MATRIX, PREDICTIONS[:, 5])

GOOD_CONF_MATRIX = worsen_ideal_conf_matrix(IDEAL_CONF_MATRIX.copy(), [62, 72])

BAD_CONF_MATRIX = worsen_ideal_conf_matrix(
    IDEAL_CONF_MATRIX.copy(), [62, 72, 72, 39, 39, 39, 39, 56]
)


@pytest.mark.parametrize(
    "detections, with_confidence, expected_result, exception",
    [
        (
            Detections.empty(),
            False,
            np.empty((0, 5), dtype=np.float32),
            DoesNotRaise(),
        ),  # empty detections; no confidence
        (
            Detections.empty(),
            True,
            np.empty((0, 6), dtype=np.float32),
            DoesNotRaise(),
        ),  # empty detections; with confidence
        (
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]),
            False,
            np.array([[0, 0, 10, 10, 0]], dtype=np.float32),
            DoesNotRaise(),
        ),  # single detection; no confidence
        (
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]),
            True,
            np.array([[0, 0, 10, 10, 0, 0.5]], dtype=np.float32),
            DoesNotRaise(),
        ),  # single detection; with confidence
        (
            mock_detections(
                xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
                class_id=[0, 1],
                confidence=[0.5, 0.2],
            ),
            False,
            np.array([[0, 0, 10, 10, 0], [0, 0, 20, 20, 1]], dtype=np.float32),
            DoesNotRaise(),
        ),  # multiple detections; no confidence
        (
            mock_detections(
                xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
                class_id=[0, 1],
                confidence=[0.5, 0.2],
            ),
            True,
            np.array(
                [[0, 0, 10, 10, 0, 0.5], [0, 0, 20, 20, 1, 0.2]], dtype=np.float32
            ),
            DoesNotRaise(),
        ),  # multiple detections; with confidence
    ],
)
def test_detections_to_tensor(
    detections: Detections,
    with_confidence: bool,
    expected_result: Optional[np.ndarray],
    exception: Exception,
):
    with exception:
        result = detections_to_tensor(
            detections=detections, with_confidence=with_confidence
        )
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "predictions, targets, prediction_masks, target_masks, classes, "
    "conf_threshold, iou_threshold,"
    "expected_box_confusion_matrix, expected_segmentation_confusion_matrix, exception",
    [
        (
            DETECTION_TENSORS,
            TARGET_TENSORS,
            None,
            None,
            CLASSES,
            0.2,
            0.5,
            IDEAL_CONF_MATRIX,
            None,
            DoesNotRaise(),
        ),
        (
            DETECTION_TENSORS,
            TARGET_TENSORS,
            DETECTION_MASKS,
            TARGET_MASKS,
            CLASSES,
            0.2,
            0.5,
            IDEAL_CONF_MATRIX,
            IDEAL_CONF_MATRIX,
            DoesNotRaise(),
        ),
        (
            [],
            [],
            [],
            [],
            CLASSES,
            0.2,
            0.5,
            create_empty_conf_matrix(NUM_CLASSES),
            create_empty_conf_matrix(NUM_CLASSES),
            DoesNotRaise(),
        ),
        (
            DETECTION_TENSORS,
            TARGET_TENSORS,
            DETECTION_MASKS,
            TARGET_MASKS,
            CLASSES,
            0.3,
            0.5,
            GOOD_CONF_MATRIX,
            GOOD_CONF_MATRIX,
            DoesNotRaise(),
        ),
        (
            DETECTION_TENSORS,
            TARGET_TENSORS,
            DETECTION_MASKS,
            TARGET_MASKS,
            CLASSES,
            0.6,
            0.5,
            BAD_CONF_MATRIX,
            BAD_CONF_MATRIX,
            DoesNotRaise(),
        ),
        (
            CUSTOM_DETECTION_TENSORS_1,
            CUSTOM_TARGET_TENSORS_1,
            *create_masks_from_tensors(
                CUSTOM_DETECTION_TENSORS_1, CUSTOM_TARGET_TENSORS_1
            ),
            CLASSES[:3],
            0.6,
            0.5,
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 2, 0, 0]]),
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 2, 0, 0]]),
            DoesNotRaise(),
        ),
        (
            CUSTOM_DETECTION_TENSORS_2,
            CUSTOM_TARGET_TENSORS_2,
            *create_masks_from_tensors(
                CUSTOM_DETECTION_TENSORS_2, CUSTOM_TARGET_TENSORS_2
            ),
            CLASSES[:3],
            0.6,
            1.0,
            np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [2, 3, 0, 0]]),
            np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [2, 3, 0, 0]]),
            DoesNotRaise(),
        ),
    ],
)
def test_from_tensors(
    predictions,
    targets,
    classes,
    prediction_masks,
    target_masks,
    conf_threshold,
    iou_threshold,
    expected_box_confusion_matrix: Optional[np.ndarray],
    expected_segmentation_confusion_matrix: Optional[np.ndarray],
    exception: Exception,
):
    with exception:
        result = ConfusionMatrix.from_tensors(
            predictions=predictions,
            targets=targets,
            classes=classes,
            prediction_masks=prediction_masks,
            target_masks=target_masks,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        assert (
            result.matrix.diagonal().sum()
            == expected_box_confusion_matrix.diagonal().sum()
        )
        assert np.array_equal(result.matrix, expected_box_confusion_matrix)
        assert np.array_equal(
            result.segmentationMatrix, expected_segmentation_confusion_matrix
        )


@pytest.mark.parametrize(
    "predictions, targets, num_classes, conf_threshold, iou_threshold, "
    "prediction_mask, target_mask, "
    "expected_box_confusion_matrix, expected_segmentation_confusion_matrix,  exception",
    [
        (
            DETECTION_TENSORS[0],
            CERTAIN_DETECTION_TENSORS[0],
            NUM_CLASSES,
            0.2,
            0.5,
            DETECTION_MASKS[0],
            TARGET_MASKS[0],
            IDEAL_CONF_MATRIX,
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
    prediction_mask,
    target_mask,
    expected_box_confusion_matrix: Optional[np.ndarray],
    expected_segmentation_confusion_matrix: Optional[np.ndarray],
    exception: Exception,
):
    with exception:
        matrix, segmentation_matrix = ConfusionMatrix.evaluate_detection_batch(
            predictions=predictions,
            targets=targets,
            num_classes=num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            prediction_mask=prediction_mask,
            target_mask=target_mask,
        )

        assert matrix.diagonal().sum() == matrix.sum()
        assert np.array_equal(matrix, expected_box_confusion_matrix)
        assert np.array_equal(
            segmentation_matrix, expected_segmentation_confusion_matrix
        )


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
    "masks_true, masks_detection, expected_iou, exception",
    [
        (
            np.array(
                [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]]
            ),
            np.array(
                [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 1], [0, 1, 1], [0, 0, 0]]]
            ),
            np.array([[1.0, 0.25], [0.25, 0.33333333]]),
            DoesNotRaise(),
        )
    ],
)
def test_mask_iou_batch(masks_true, masks_detection, expected_iou, exception):
    with exception:
        iou = mask_iou_batch(masks_true, masks_detection)
        assert np.allclose(
            iou, expected_iou, atol=1e-6
        ), f"Expected {expected_iou}, but got {iou}"


@pytest.mark.parametrize(
    "recall, precision, expected_result, exception",
    [
        (
            np.array([1.0]),
            np.array([1.0]),
            1.0,
            DoesNotRaise(),
        ),  # perfect recall and precision
        (
            np.array([0.0]),
            np.array([0.0]),
            0.0,
            DoesNotRaise(),
        ),  # no recall and precision
        (
            np.array([0.0, 0.2, 0.2, 0.8, 0.8, 1.0]),
            np.array([0.7, 0.8, 0.4, 0.5, 0.1, 0.2]),
            0.5,
            DoesNotRaise(),
        ),
        (
            np.array([0.0, 0.5, 0.5, 1.0]),
            np.array([0.75, 0.75, 0.75, 0.75]),
            0.75,
            DoesNotRaise(),
        ),
    ],
)
def test_compute_average_precision(
    recall: np.ndarray,
    precision: np.ndarray,
    expected_result: float,
    exception: Exception,
) -> None:
    with exception:
        result = MeanAveragePrecision.compute_average_precision(
            recall=recall, precision=precision
        )
        assert_almost_equal(result, expected_result, tolerance=0.01)
