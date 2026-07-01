from contextlib import ExitStack as DoesNotRaise
from typing import ClassVar

import cv2
import numpy as np
import pytest

from supervision.dataset.core import DetectionDataset
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget
from supervision.metrics.detection import (
    ConfusionMatrix,
    MeanAveragePrecision,
    _split_detections_by_outcome,
    _validate_input_tensors,
    detections_to_tensor,
)
from tests.helpers import (
    _create_detections,
    assert_almost_equal,
    create_predictions_with_class_iou_tests,
)


def _call_confusion_matrix_from_detections_masks() -> None:
    ConfusionMatrix.from_detections(
        predictions=[
            Detections(
                xyxy=np.zeros((1, 4), dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
            )
        ],
        targets=[
            Detections(
                xyxy=np.zeros((1, 4), dtype=np.float32),
                class_id=np.array([0]),
            )
        ],
        classes=["box"],
        metric_target=MetricTarget.MASKS,
    )


def _call_confusion_matrix_from_tensors_masks() -> None:
    ConfusionMatrix.from_tensors(
        predictions=[np.zeros((1, 6), dtype=np.float32)],
        targets=[np.zeros((1, 5), dtype=np.float32)],
        classes=["box"],
        metric_target=MetricTarget.MASKS,
    )


def _call_confusion_matrix_evaluate_detection_batch_masks() -> None:
    ConfusionMatrix.evaluate_detection_batch(
        predictions=np.zeros((1, 6), dtype=np.float32),
        targets=np.zeros((1, 5), dtype=np.float32),
        num_classes=1,
        conf_threshold=0.3,
        iou_threshold=0.5,
        metric_target=MetricTarget.MASKS,
    )


class TestDetectionMetrics:
    """
    Verify that detection metrics are computed accurately.

    Ensures that detection metrics (mAP, Conf. Matrix, etc.) are computed accurately.
    These metrics are the primary way users evaluate the performance of their models
    within the `supervision` ecosystem.
    """

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

    TARGET_TENSORS: ClassVar[list[np.ndarray]] = [
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

    DETECTION_TENSORS: ClassVar[list[np.ndarray]] = [
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
    CERTAIN_DETECTION_TENSORS: ClassVar[list[np.ndarray]] = [
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

    IDEAL_MATCHES = np.stack(
        [
            np.arange(len(PREDICTIONS)),
            np.arange(len(PREDICTIONS)),
            np.ones(len(PREDICTIONS)),
        ],
        axis=1,
    )

    @staticmethod
    def create_empty_conf_matrix(num_classes: int, do_add_dummy_class: bool = True):
        if do_add_dummy_class:
            num_classes += 1
        return np.zeros((num_classes, num_classes))

    @staticmethod
    def update_ideal_conf_matrix(conf_matrix: np.ndarray, class_ids: np.ndarray):
        for class_id, count in zip(*np.unique(class_ids, return_counts=True)):
            class_id = int(class_id)
            conf_matrix[class_id, class_id] += count
        return conf_matrix

    @staticmethod
    def worsen_ideal_conf_matrix(conf_matrix: np.ndarray, class_ids: np.ndarray | list):
        for class_id in class_ids:
            class_id = int(class_id)
            conf_matrix[class_id, class_id] -= 1
            conf_matrix[class_id, 80] += 1
        return conf_matrix

    IDEAL_CONF_MATRIX = create_empty_conf_matrix.__func__(NUM_CLASSES)
    IDEAL_CONF_MATRIX = update_ideal_conf_matrix.__func__(
        IDEAL_CONF_MATRIX, PREDICTIONS[:, 5]
    )

    GOOD_CONF_MATRIX = worsen_ideal_conf_matrix.__func__(
        IDEAL_CONF_MATRIX.copy(), [62, 72]
    )

    BAD_CONF_MATRIX = worsen_ideal_conf_matrix.__func__(
        IDEAL_CONF_MATRIX.copy(), [62, 72, 72, 39, 39, 39, 39, 56]
    )

    @pytest.mark.parametrize(
        (
            "detections",
            "with_confidence",
            "metric_target",
            "expected_result",
            "exception",
        ),
        [
            (
                Detections.empty(),
                False,
                MetricTarget.BOXES,
                np.empty((0, 5), dtype=np.float32),
                DoesNotRaise(),
            ),  # empty detections; no confidence
            (
                Detections.empty(),
                True,
                MetricTarget.BOXES,
                np.empty((0, 6), dtype=np.float32),
                DoesNotRaise(),
            ),  # empty detections; with confidence
            (
                Detections.empty(),
                False,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                np.empty((0, 9), dtype=np.float32),
                DoesNotRaise(),
            ),  # empty OBB detections; no confidence
            (
                Detections.empty(),
                True,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                np.empty((0, 10), dtype=np.float32),
                DoesNotRaise(),
            ),  # empty OBB detections; with confidence
            (
                _create_detections(
                    xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]
                ),
                False,
                MetricTarget.BOXES,
                np.array([[0, 0, 10, 10, 0]], dtype=np.float32),
                DoesNotRaise(),
            ),  # single detection; no confidence
            (
                _create_detections(
                    xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]
                ),
                True,
                MetricTarget.BOXES,
                np.array([[0, 0, 10, 10, 0, 0.5]], dtype=np.float32),
                DoesNotRaise(),
            ),  # single detection; with confidence
            (
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                    confidence=np.array([0.5], dtype=np.float32),
                    data={
                        "xyxyxyxy": np.array(
                            [[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32
                        )
                    },
                ),
                False,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0]], dtype=np.float32),
                DoesNotRaise(),
            ),  # single OBB detection; no confidence
            (
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                    confidence=np.array([0.5], dtype=np.float32),
                    data={
                        "xyxyxyxy": np.array(
                            [[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32
                        )
                    },
                ),
                True,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0, 0.5]], dtype=np.float32),
                DoesNotRaise(),
            ),  # single OBB detection; with confidence
            (
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                    confidence=np.array([0.5], dtype=np.float32),
                ),
                False,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                None,
                pytest.raises(ValueError, match="ORIENTED_BOUNDING_BOXES requested"),
            ),  # OBB requested but data missing
            (
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                    confidence=np.array([0.5], dtype=np.float32),
                ),
                False,
                MetricTarget.MASKS,
                None,
                pytest.raises(ValueError, match=r"MetricTarget\.MASKS"),
            ),  # MASKS requested but not supported
            (
                _create_detections(
                    xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
                    class_id=[0, 1],
                    confidence=[0.5, 0.2],
                ),
                False,
                MetricTarget.BOXES,
                np.array([[0, 0, 10, 10, 0], [0, 0, 20, 20, 1]], dtype=np.float32),
                DoesNotRaise(),
            ),  # multiple detections; no confidence
            (
                _create_detections(
                    xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
                    class_id=[0, 1],
                    confidence=[0.5, 0.2],
                ),
                True,
                MetricTarget.BOXES,
                np.array(
                    [[0, 0, 10, 10, 0, 0.5], [0, 0, 20, 20, 1, 0.2]], dtype=np.float32
                ),
                DoesNotRaise(),
            ),  # multiple detections; with confidence
            pytest.param(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                    data={"xyxyxyxy": np.zeros((1, 4), dtype=np.float32)},
                ),
                False,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                None,
                pytest.raises(ValueError, match="Expected xyxyxyxy to contain"),
                id="obb-malformed-element-count",
            ),  # OBB data present but wrong element count (4 instead of 8)
            pytest.param(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                    data={
                        "xyxyxyxy": np.array(
                            [[0, 0, 10, 0, 10, 10, 0, 10]], dtype=np.float32
                        )
                    },
                ),
                True,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                None,
                pytest.raises(ValueError, match="Detections with confidence"),
                id="obb-with-confidence-but-confidence-none",
            ),  # OBB + with_confidence=True but confidence is None
            pytest.param(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=None,
                    data={
                        "xyxyxyxy": np.array(
                            [[0, 0, 10, 0, 10, 10, 0, 10]], dtype=np.float32
                        )
                    },
                ),
                False,
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                None,
                pytest.raises(ValueError, match="class_id"),
                id="obb-class-id-none",
            ),  # OBB + class_id=None
        ],
    )
    def test_detections_to_tensor(
        self,
        detections: Detections,
        with_confidence: bool,
        metric_target: MetricTarget,
        expected_result: np.ndarray | None,
        exception: Exception,
    ) -> None:
        """
        Verify that Detections objects are correctly converted to NumPy tensors.

        Scenario: Converting Detections objects to NumPy tensors.
        Expected: Tensors are correctly formatted for consumption by metric functions,
        preserving coordinates, class IDs, and optionally confidence scores.
        """
        with exception:
            result = detections_to_tensor(
                detections=detections,
                with_confidence=with_confidence,
                metric_target=metric_target,
            )
            np.testing.assert_allclose(result, expected_result, atol=1e-5)

    @pytest.mark.parametrize(
        (
            "predictions",
            "targets",
            "classes",
            "conf_threshold",
            "iou_threshold",
            "expected_result",
            "exception",
        ),
        [
            (
                DETECTION_TENSORS,
                TARGET_TENSORS,
                CLASSES,
                0.2,
                0.5,
                IDEAL_CONF_MATRIX,
                DoesNotRaise(),
            ),
            (
                [],
                [],
                CLASSES,
                0.2,
                0.5,
                create_empty_conf_matrix.__func__(NUM_CLASSES),
                DoesNotRaise(),
            ),
            (
                DETECTION_TENSORS,
                TARGET_TENSORS,
                CLASSES,
                0.3,
                0.5,
                GOOD_CONF_MATRIX,
                DoesNotRaise(),
            ),
            (
                DETECTION_TENSORS,
                TARGET_TENSORS,
                CLASSES,
                0.6,
                0.5,
                BAD_CONF_MATRIX,
                DoesNotRaise(),
            ),
            (
                [
                    np.array(
                        [
                            # correct detection of [0]
                            [0.0, 0.0, 3.0, 3.0, 0, 0.9],
                            # additional detection of [0] - FP
                            [0.1, 0.1, 3.0, 3.0, 0, 0.9],
                            # correct detection with incorrect class
                            [6.0, 1.0, 8.0, 3.0, 1, 0.8],
                            # incorrect detection - FP
                            [1.0, 6.0, 2.0, 7.0, 1, 0.8],
                            # incorrect detection with low IoU - FP
                            [1.0, 2.0, 2.0, 4.0, 1, 0.8],
                        ]
                    )
                ],
                [
                    np.array(
                        [  # [0] detected
                            [0.0, 0.0, 3.0, 3.0, 0],
                            # [1] undetected - FN
                            [2.0, 2.0, 5.0, 5.0, 1],
                            # [2] correct detection with incorrect class
                            [6.0, 1.0, 8.0, 3.0, 2],
                        ]
                    )
                ],
                CLASSES[:3],
                0.6,
                0.5,
                np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 2, 0, 0]]),
                DoesNotRaise(),
            ),
            (
                [
                    np.array(
                        [
                            # correct detection of [0]
                            [0.0, 0.0, 3.0, 3.0, 0, 0.9],
                            # additional detection of [0] - FP
                            [0.1, 0.1, 3.0, 3.0, 0, 0.9],
                            # correct detection with incorrect class
                            [6.0, 1.0, 8.0, 3.0, 1, 0.8],
                            # incorrect detection - FP
                            [1.0, 6.0, 2.0, 7.0, 1, 0.8],
                            # incorrect detection with low IoU - FP
                            [1.0, 2.0, 2.0, 4.0, 1, 0.8],
                        ]
                    )
                ],
                [
                    np.array(
                        [
                            # [0] detected
                            [0.0, 0.0, 3.0, 3.0, 0],
                            # [1] undetected - FN
                            [2.0, 2.0, 5.0, 5.0, 1],
                            # [2] correct detection with incorrect class
                            [6.0, 1.0, 8.0, 3.0, 2],
                        ]
                    )
                ],
                CLASSES[:3],
                0.6,
                1.0,
                np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [2, 3, 0, 0]]),
                DoesNotRaise(),
            ),
        ],
    )
    def test_from_tensors(
        self,
        predictions,
        targets,
        classes,
        conf_threshold,
        iou_threshold,
        expected_result: np.ndarray | None,
        exception: Exception,
    ) -> None:
        with exception:
            result = ConfusionMatrix.from_tensors(
                predictions=predictions,
                targets=targets,
                classes=classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        assert result.matrix.diagonal().sum() == expected_result.diagonal().sum()
        assert np.array_equal(result.matrix, expected_result)

    @pytest.mark.parametrize(
        (
            "predictions",
            "targets",
            "num_classes",
            "conf_threshold",
            "iou_threshold",
            "expected_result",
            "exception",
        ),
        [
            (
                DETECTION_TENSORS[0],
                TARGET_TENSORS[0],
                NUM_CLASSES,
                0.2,
                0.5,
                IDEAL_CONF_MATRIX,
                DoesNotRaise(),
            )
        ],
    )
    def test_evaluate_detection_batch(
        self,
        predictions,
        targets,
        num_classes,
        conf_threshold,
        iou_threshold,
        expected_result: np.ndarray | None,
        exception: Exception,
    ) -> None:
        with exception:
            result = ConfusionMatrix.evaluate_detection_batch(
                predictions=predictions,
                targets=targets,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        assert result.diagonal().sum() == result.sum()
        assert np.array_equal(result, expected_result)

    @pytest.mark.parametrize(
        ("matches", "expected_result", "exception"),
        [
            (
                IDEAL_MATCHES,
                IDEAL_MATCHES,
                DoesNotRaise(),
            )
        ],
    )
    def test_drop_extra_matches(
        self,
        matches,
        expected_result: np.ndarray | None,
        exception: Exception,
    ) -> None:
        with exception:
            result = ConfusionMatrix._drop_extra_matches(matches)

            assert np.array_equal(result, expected_result)

    @pytest.mark.parametrize(
        ("recall", "precision", "expected_result", "exception"),
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
        self,
        recall: np.ndarray,
        precision: np.ndarray,
        expected_result: float,
        exception: Exception,
    ) -> None:
        """
        Verify that Average Precision is correctly calculated from PR curve points.

        Scenario: Computing Average Precision (AP) from PR curve points.
        Expected: AP is correctly calculated using the area under the curve, which is
        the standard for evaluating detection models (mAP components).
        """
        with exception:
            result = MeanAveragePrecision.compute_average_precision(
                recall=recall, precision=precision
            )
            assert_almost_equal(result, expected_result, tolerance=0.01)

    @pytest.mark.parametrize(
        (
            "predictions",
            "targets",
            "classes",
            "conf_threshold",
            "iou_threshold",
            "expected_result",
            "exception",
        ),
        [
            # Test 1: Class priority over IoU - correct class with lower IoU should win
            (
                [
                    _create_detections(  # Predicted bboxes
                        xyxy=[[0.1, 0.1, 2.1, 2.1], [0.0, 0.0, 2.0, 2.0]],
                        class_id=[0, 1],
                        confidence=[0.9, 0.95],
                    )
                ],
                [_create_detections(xyxy=[[0, 0, 2, 2]], class_id=[0])],  # GT bboxes
                [0, 1, 2],  # Class ids
                0.5,  # Confidence Threshold
                0.5,  # IOU Threshold
                np.array(  # Expected confusion matrix
                    [
                        [1.0, 0.0, 0.0, 0.0],  # 1 TP
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 1.0, 0.0, 0.0],  # 1 FP:
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 2: Multiple overlapping predictions with different classes
            (
                [
                    _create_detections(
                        xyxy=[
                            [0.1, 0.1, 2.1, 2.1],
                            [0.2, 0.2, 2.2, 2.2],
                            [0.3, 0.3, 2.3, 2.3],
                            [4.1, 4.1, 6.1, 6.1],
                        ],
                        class_id=[0, 1, 2, 1],
                        confidence=[0.9, 0.8, 0.7, 0.85],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6]], class_id=[0, 1]
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],  # 1 TP
                        [0.0, 1.0, 0.0, 0.0],  # 1 TP
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 1.0, 1.0, 0.0],  # 2 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 3: Confidence threshold filtering with edge cases
            (
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6], [8, 8, 10, 10]],
                        class_id=[0, 1, 2],
                        confidence=[0.6, 0.4, 0.8],  # middle one below threshold
                    )
                ],
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6]], class_id=[0, 1]
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],  # 1 TP
                        [0.0, 0.0, 0.0, 1.0],  # 1 FN (filtered by conf)
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 0.0, 1.0, 0.0],  # 1 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 4: IoU threshold boundary (IoU = 0.5625, slightly above threshold)
            (
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 1.5, 1.5],
                            [4, 4, 5.5, 5.5],
                        ],  # IoU = 0.5625 for both
                        class_id=[0, 1],
                        confidence=[0.9, 0.8],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6]], class_id=[0, 1]
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],  # 1 TP (IoU exceeds threshold)
                        [0.0, 1.0, 0.0, 0.0],  # 1 TP (IoU exceeds threshold)
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 0.0, 0.0, 0.0],  # none
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 5: Chain of overlapping detections
            (
                [
                    _create_detections(
                        xyxy=[[0.1, 0.1, 2.1, 2.1], [1.9, 1.9, 3.9, 3.9]],
                        class_id=[0, 2],
                        confidence=[0.9, 0.8],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]],
                        class_id=[0, 1, 2],
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],  # 1 TP
                        [0.0, 0.0, 0.0, 1.0],  # 1 FN (no matching label)
                        [0.0, 0.0, 1.0, 0.0],  # 1 TP
                        [0.0, 0.0, 0.0, 0.0],  # none
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 6: All false positives (no ground truth)
            (
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6], [8, 8, 10, 10]],
                        class_id=[0, 1, 2],
                        confidence=[0.9, 0.8, 0.7],
                    )
                ],
                [
                    _create_detections(
                        xyxy=np.empty((0, 4)), class_id=np.array([], dtype=int)
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [0.0, 0.0, 0.0, 0.0],  # none
                        [1.0, 1.0, 1.0, 0.0],  # 3 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 7: Empty predictions and empty ground truth
            (
                [
                    _create_detections(
                        xyxy=np.empty((0, 4)),
                        class_id=np.array([], dtype=int),
                        confidence=np.array([], dtype=float),
                    )
                ],
                [
                    _create_detections(
                        xyxy=np.empty((0, 4)), class_id=np.array([], dtype=int)
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.zeros((4, 4)),
                DoesNotRaise(),
            ),
            # Test 8: Multi-class misclassifications
            (
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6], [10, 10, 12, 12]],
                        class_id=[0, 2, 1],
                        confidence=[0.9, 0.8, 0.7],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [4, 4, 6, 6], [8, 8, 10, 10]],
                        class_id=[0, 1, 2],
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],  # 1 TP
                        [0.0, 0.0, 1.0, 0.0],  # 1 misclassified
                        [0.0, 0.0, 0.0, 1.0],  # 1 FN
                        [0.0, 1.0, 0.0, 0.0],  # 1 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 9: Complex multiple predictions with mixed results
            (
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [4, 4, 6, 6],
                            [8, 8, 10, 10],
                            [12, 12, 14, 14],
                            [16, 16, 18, 18],
                        ],
                        class_id=[0, 1, 1, 2, 2],
                        confidence=[0.9, 0.8, 0.7, 0.6, 0.5],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [4, 4, 6, 6],
                            [8, 8, 10, 10],
                            [12, 12, 14, 14],
                        ],
                        class_id=[0, 1, 2, 0],
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 1.0, 0.0],  # 1 TP and 1 misclassified
                        [0.0, 1.0, 0.0, 0.0],  # 1 TP
                        [0.0, 1.0, 0.0, 0.0],  # 1 misclassified
                        [0.0, 0.0, 1.0, 0.0],  # 1 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 10: Large complex example with confidence filtering
            (
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [4, 4, 6, 6],
                            [8, 8, 10, 10],
                            [12, 12, 14, 14],
                            [16, 16, 18, 18],
                            [18, 18, 20, 20],
                        ],
                        class_id=[0, 0, 1, 2, 1, 2],
                        confidence=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # last one filtered
                    )
                ],
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [4, 4, 6, 6],
                            [8, 8, 10, 10],
                            [12, 12, 14, 14],
                        ],
                        class_id=[0, 1, 2, 0],
                    )
                ],
                [0, 1, 2],
                0.5,  # conf_threshold filters out last prediction
                0.5,
                np.array(
                    [
                        [1.0, 0.0, 1.0, 0.0],  # 1 TP and 1 misclassified
                        [1.0, 0.0, 0.0, 0.0],  # 1 misclassified
                        [0.0, 1.0, 0.0, 0.0],  # 1 misclassified
                        [0.0, 1.0, 0.0, 0.0],  # 1 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 11: High counts with multiple TPs and misclassifications
            (
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [0, 3, 2, 5],
                            [0, 6, 2, 8],
                            [4, 0, 6, 2],
                            [4, 3, 6, 5],
                            [8, 0, 10, 2],
                            [12, 0, 14, 2],
                        ],
                        class_id=[0, 0, 0, 2, 2, 2, 0],
                        confidence=[0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.8],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [0, 3, 2, 5],
                            [0, 6, 2, 8],
                            [4, 0, 6, 2],
                            [4, 3, 6, 5],
                            [8, 0, 10, 2],
                            [8, 3, 10, 5],
                        ],
                        class_id=[0, 0, 0, 1, 1, 2, 2],
                    )
                ],
                [0, 1, 2],
                0.5,
                0.5,
                np.array(
                    [
                        [3.0, 0.0, 0.0, 0.0],  # 3 TP
                        [0.0, 0.0, 2.0, 0.0],  # 2 misclassified
                        [0.0, 0.0, 1.0, 1.0],  # 1 TP, 1 FN
                        [1.0, 0.0, 0.0, 0.0],  # 1 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 12: Symmetric multi-class confusions with higher counts
            (
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [0, 4, 2, 6],
                            [4, 0, 6, 2],
                            [4, 4, 6, 6],
                            [8, 0, 10, 2],
                            [8, 4, 10, 6],
                            [12, 0, 14, 2],
                            [12, 4, 14, 6],
                        ],
                        class_id=[0, 0, 1, 1, 0, 0, 1, 1],
                        confidence=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8],
                    )
                ],
                [
                    _create_detections(
                        xyxy=[
                            [0, 0, 2, 2],
                            [0, 4, 2, 6],
                            [4, 0, 6, 2],
                            [4, 4, 6, 6],
                            [8, 0, 10, 2],
                            [8, 4, 10, 6],
                        ],
                        class_id=[0, 0, 1, 1, 2, 2],
                    )
                ],
                [0, 1, 2],  # Class ids
                0.5,  # Confidence threshold
                0.5,  # IOU threshold
                np.array(
                    [
                        [2.0, 0.0, 0.0, 0.0],  # 2 TP
                        [0.0, 2.0, 0.0, 0.0],  # TP
                        [2.0, 0.0, 0.0, 0.0],  # 2 misclassified
                        [0.0, 2.0, 0.0, 0.0],  # 2 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 13: Empty Ground Truths
            (
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [0, 4, 2, 6]],
                        class_id=[0, 0],
                        confidence=[0.9, 0.9],
                    )
                ],
                [Detections.empty()],
                [0, 1, 2],  # Class ids
                0.5,  # Confidence threshold
                0.5,  # IOU threshold
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],  # 2 FP
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 14: Empty Detections
            (
                [Detections.empty()],
                [
                    _create_detections(
                        xyxy=[[0, 0, 2, 2], [0, 4, 2, 6]], class_id=[0, 0]
                    )
                ],
                [0, 1, 2],  # Class ids
                0.5,  # Confidence threshold
                0.5,  # IOU threshold
                np.array(
                    [
                        [0.0, 0.0, 0.0, 2.0],  # 2 TP
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                DoesNotRaise(),
            ),
            # Test 15: Symmetric multi-class confusions with higher counts
            (
                [Detections.empty()],
                [Detections.empty()],
                [0, 1, 2],  # Class ids
                0.5,  # Confidence threshold
                0.5,  # IOU threshold
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                DoesNotRaise(),
            ),
        ],
    )
    def test_confusion_matrix(
        self,
        predictions,
        targets,
        classes,
        conf_threshold,
        iou_threshold,
        expected_result,
        exception: Exception,
    ) -> None:
        with exception:
            confusion_matrix = ConfusionMatrix.from_detections(
                predictions=predictions,
                targets=targets,
                classes=classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        # Verify the confusion matrix matches expected
        # AssertionError if the two arrays are not equal
        np.testing.assert_array_equal(confusion_matrix.matrix, expected_result)

    def test_confusion_matrix_on_yolo_dataset(self, yolo_dataset_structure) -> None:
        """
        Test confusion matrix calculation on a YOLO-format dataset.

        This test verifies that the confusion matrix fix (considering both IoU AND
        class agreement) works correctly when applied to a dataset loaded from
        roboflow-format YOLO data. It creates a synthetic dataset with specific
        scenarios where predictions have high IoU but wrong class, ensuring only
        predictions with correct class are matched.
        """
        dataset_info = yolo_dataset_structure
        classes = ["dog", "cat", "person"]

        # Load dataset using supervision's YOLO loader
        dataset = DetectionDataset.from_yolo(
            images_directory_path=dataset_info["images_dir"],
            annotations_directory_path=dataset_info["labels_dir"],
            data_yaml_path=dataset_info["data_yaml_path"],
        )

        # Verify dataset loaded correctly
        assert len(dataset) == dataset_info["num_images"], (
            f"Dataset should have {dataset_info['num_images']} images, "
            f"but got {len(dataset)}. Dataset loading may have failed."
        )
        assert dataset.classes == classes, (
            f"Dataset classes should be {classes}, but got {dataset.classes}. "
            f"Check data.yaml parsing."
        )

        # Test confusion matrix with the dataset
        # Split the dataset to test split functionality
        train_dataset, test_dataset = dataset.split(
            split_ratio=0.5, random_state=42, shuffle=True
        )

        assert len(train_dataset) + len(test_dataset) == len(dataset), (
            f"Split datasets should sum to original dataset size ({len(dataset)}), "
            f"but got {len(train_dataset)} + {len(test_dataset)} = "
            f"{len(train_dataset) + len(test_dataset)}. Dataset split may be broken."
        )
        assert train_dataset.classes == classes, (
            "Train dataset should preserve class list after split"
        )
        assert test_dataset.classes == classes, (
            "Test dataset should preserve class list after split"
        )

        # Create predictions that test the IoU+class matching fix
        predictions = []
        targets = []

        for img_path, img, gt_detections in test_dataset:
            targets.append(gt_detections)
            predictions.append(
                create_predictions_with_class_iou_tests(gt_detections, len(classes))
            )

        # Calculate confusion matrix
        confusion_matrix = ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=list(range(len(classes))),
            conf_threshold=0.5,
            iou_threshold=0.5,
        )

        # Verify confusion matrix structure and basic properties
        n = len(classes) + 1
        assert confusion_matrix.matrix.shape == (n, n), (
            f"Expected shape ({n}, {n}), got {confusion_matrix.matrix.shape}"
        )

        # Count TPs (diagonal) and total ground truths
        total_gt = sum(len(t) for t in targets if len(t) > 0)
        total_tp = sum(confusion_matrix.matrix[i, i] for i in range(len(classes)))

        assert total_tp > 0, (
            f"No TPs found (TP={total_tp}, GT={total_gt}), matching is broken"
        )

        # Count FPs (last column) - should include wrong-class predictions
        total_fp = confusion_matrix.matrix[: len(classes), -1].sum()
        assert total_fp >= 0, f"FP count negative ({total_fp}), computation bug"

        # Verify IoU+class fix: wrong-class preds should become FPs,
        # not match GTs
        assert total_fp > 0 or total_tp == total_gt, (
            f"Expected FPs from wrong-class preds (got {total_fp}) or all GTs "
            f"matched (TP={total_tp}, GT={total_gt}). IoU+class fix may be broken: "
            f"wrong-class preds with high IoU might incorrectly match GTs."
        )

    def test_confusion_matrix_benchmark_saves_validation_visualizations(
        self,
        tmp_path,
    ):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        targets = Detections(
            xyxy=np.array([[2, 2, 12, 12], [18, 18, 28, 28]], dtype=np.float32),
            class_id=np.array([0, 1]),
        )
        predictions = Detections(
            xyxy=np.array([[2, 2, 12, 12], [4, 18, 12, 28]], dtype=np.float32),
            confidence=np.array([0.95, 0.88]),
            class_id=np.array([0, 1]),
        )

        class Dataset:
            classes: ClassVar[list[str]] = ["cat", "dog"]

            def __iter__(self):
                yield "sample.jpg", image, targets

        def callback(_: np.ndarray) -> Detections:
            return predictions

        confusion_matrix = ConfusionMatrix.benchmark(
            dataset=Dataset(),
            callback=callback,
            save_directory_path=tmp_path,
        )

        saved_image_path = tmp_path / "sample.jpg"
        assert saved_image_path.exists()

        saved_image = cv2.imread(str(saved_image_path))
        assert saved_image is not None
        assert saved_image.shape[:2] == (64, 64)
        gt_panel = saved_image[:32, :32]
        tp_panel = saved_image[:32, 32:]
        fp_panel = saved_image[32:, :32]
        fn_panel = saved_image[32:, 32:]

        # Assert boxes are rendered in the expected panels, away from borders/dividers.
        assert np.any(gt_panel[2:13, 2:13] != 0)
        assert np.any(tp_panel[2:13, 2:13] != 0)
        # Rows 25-28 are below the panel title text; cols 4-9 are at the left edge
        # of the FP box [4,18,12,28] — non-zero only if the box is rendered.
        assert np.any(fp_panel[25:29, 4:9] != 0)
        # Rows 25-28 are below the panel title text; cols 18-23 are at the left
        # edge of the FN target box [18,18,28,28] — non-zero only if box rendered.
        assert np.any(fn_panel[25:29, 18:23] != 0)

        # Basic sanity that panels differ.
        assert not np.array_equal(gt_panel, tp_panel)
        assert not np.array_equal(gt_panel, fp_panel)
        assert not np.array_equal(gt_panel, fn_panel)
        assert not np.array_equal(tp_panel, fp_panel)
        assert not np.array_equal(tp_panel, fn_panel)
        assert not np.array_equal(fp_panel, fn_panel)

        assert confusion_matrix.matrix.shape == (3, 3)

    @pytest.mark.parametrize(
        ("predictions", "targets", "metric_target", "exception"),
        [
            (
                [np.zeros((1, 10), dtype=np.float32)],
                [np.zeros((1, 9), dtype=np.float32)],
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                DoesNotRaise(),
            ),
            (
                [np.zeros((1, 6), dtype=np.float32)],
                [np.zeros((1, 5), dtype=np.float32)],
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                pytest.raises(ValueError, match="Predictions must have shape"),
            ),
            (
                [np.zeros((1, 10), dtype=np.float32)],
                [np.zeros((1, 9), dtype=np.float32)],
                MetricTarget.BOXES,
                pytest.raises(ValueError, match="Predictions must have shape"),
            ),
            pytest.param(
                [np.zeros((1, 10), dtype=np.float32)],
                [np.zeros((1, 5), dtype=np.float32)],
                MetricTarget.ORIENTED_BOUNDING_BOXES,
                pytest.raises(ValueError, match="Targets must have shape"),
                id="obb-correct-pred-cols-wrong-target-cols",
            ),
        ],
    )
    def test_validate_input_tensors_obb(
        self, predictions, targets, metric_target, exception
    ):
        with exception:
            _validate_input_tensors(predictions, targets, metric_target=metric_target)

    def test_confusion_matrix_obb(self):
        """
        Verify OBB support in ConfusionMatrix.
        Test scenarios:
        1. Perfect OBB overlap (Rotation Match)
        2. Rotation Sensitivity (Same AABB, different rotation)
        3. Regression (BOXES mode)
        """
        classes = ["box"]

        # Perfect OBB overlap
        # 45 degree rotated box
        obb_coords = np.array(
            [[[5, 0], [10, 5], [5, 10], [0, 5]]], dtype=np.float32
        )  # Diamond shape
        gt = [
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                class_id=np.array([0]),
                data={"xyxyxyxy": obb_coords},
            )
        ]
        pred = [
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                data={"xyxyxyxy": obb_coords},
            )
        ]

        cm_obb = ConfusionMatrix.from_detections(
            predictions=pred,
            targets=gt,
            classes=classes,
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES,
        )
        # Expected TP = 1
        assert cm_obb.matrix[0, 0] == 1
        assert cm_obb.matrix.sum() == 1

        # Rotation Sensitivity
        # GT is 45 deg, Pred is axis-aligned box with same AABB
        aabb_coords = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32)
        pred_aabb = [
            Detections(
                xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                data={"xyxyxyxy": aabb_coords},
            )
        ]

        # In OBB mode, IoU between diamond and square is 0.5
        cm_sensitivity_obb = ConfusionMatrix.from_detections(
            predictions=pred_aabb,
            targets=gt,
            classes=classes,
            iou_threshold=0.6,  # Threshold higher than 0.5
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES,
        )
        # Expected FN=1, FP=1 (no match because IoU=0.5 < 0.6)
        assert cm_sensitivity_obb.matrix[0, 1] == 1  # FN
        assert cm_sensitivity_obb.matrix[1, 0] == 1  # FP

        # In BOXES mode, they match perfectly
        cm_sensitivity_boxes = ConfusionMatrix.from_detections(
            predictions=pred_aabb,
            targets=gt,
            classes=classes,
            iou_threshold=0.6,
            metric_target=MetricTarget.BOXES,
        )
        # Expected TP = 1
        assert cm_sensitivity_boxes.matrix[0, 0] == 1

        # Deterministic comparison: OBB IoU should be less than AABB IoU
        # Here OBB IoU is 0.5, AABB IoU is 1.0
        # We can verify this by checking that a threshold between them
        # differentiates behavior
        assert cm_sensitivity_obb.matrix[0, 0] == 0
        assert cm_sensitivity_boxes.matrix[0, 0] == 1

    def test_confusion_matrix_obb_regression_1760(self):
        """Regression for #1760: thin OBBs with same AABB must not match.

        Two thin bars (100 px long, 10 px wide) at 45° and -45° share an identical
        AABB (AABB IoU = 1.0), but their actual OBB overlap is only at the crossing
        centre (OBB IoU ≈ 0.05).

        With iou_threshold=0.5:
        - ORIENTED_BOUNDING_BOXES mode: no match → FP + FN (bug before fix: TP)
        - BOXES mode: AABB IoU=1.0 → TP (controls: confirms AABB path unbroken)

        A regression that swaps oriented_box_iou_batch back to box_iou_batch would
        flip the OBB assertion to TP, surfacing the exact bug from issue #1760.
        """
        classes = ["bar"]
        sq2 = np.float32(1.0 / np.sqrt(2))
        cx, cy = np.float32(100.0), np.float32(100.0)
        hl, hw = np.float32(50.0), np.float32(5.0)

        # Bar at 45°: length direction (sq2, sq2), width direction (-sq2, sq2)
        bar_45 = np.array(
            [
                [cx + (hl - hw) * sq2, cy + (hl + hw) * sq2],
                [cx + (hl + hw) * sq2, cy + (hl - hw) * sq2],
                [cx - (hl - hw) * sq2, cy - (hl + hw) * sq2],
                [cx - (hl + hw) * sq2, cy - (hl - hw) * sq2],
            ],
            dtype=np.float32,
        )
        # Bar at -45°: length direction (sq2, -sq2), width direction (sq2, sq2)
        bar_neg45 = np.array(
            [
                [cx + (hl + hw) * sq2, cy - (hl - hw) * sq2],
                [cx + (hl - hw) * sq2, cy - (hl + hw) * sq2],
                [cx - (hl + hw) * sq2, cy + (hl - hw) * sq2],
                [cx - (hl - hw) * sq2, cy + (hl + hw) * sq2],
            ],
            dtype=np.float32,
        )

        # Both bars share the same axis-aligned bounding box
        half_aabb = (hl + hw) * sq2
        shared_xyxy = np.array(
            [[cx - half_aabb, cy - half_aabb, cx + half_aabb, cy + half_aabb]],
            dtype=np.float32,
        )

        gt = [
            Detections(
                xyxy=shared_xyxy,
                class_id=np.array([0]),
                data={"xyxyxyxy": bar_45[np.newaxis]},
            )
        ]
        pred = [
            Detections(
                xyxy=shared_xyxy,
                class_id=np.array([0]),
                confidence=np.array([0.9], dtype=np.float32),
                data={"xyxyxyxy": bar_neg45[np.newaxis]},
            )
        ]

        # OBB mode: orthogonal bars barely overlap → FP and FN, not TP
        cm_obb = ConfusionMatrix.from_detections(
            predictions=pred,
            targets=gt,
            classes=classes,
            iou_threshold=0.5,
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES,
        )
        assert cm_obb.matrix[0, 0] == 0  # no TP
        assert cm_obb.matrix[0, 1] == 1  # FN
        assert cm_obb.matrix[1, 0] == 1  # FP

        # BOXES mode: identical AABB → TP (controls that AABB path is intact)
        cm_boxes = ConfusionMatrix.from_detections(
            predictions=pred,
            targets=gt,
            classes=classes,
            iou_threshold=0.5,
            metric_target=MetricTarget.BOXES,
        )
        assert cm_boxes.matrix[0, 0] == 1
        assert cm_boxes.matrix.sum() == 1

    @pytest.mark.parametrize(
        ("pred_tensor", "target_tensor", "iou_threshold", "expected_matrix"),
        [
            pytest.param(
                np.array([[5, 0, 10, 5, 5, 10, 0, 5, 0, 0.9]], dtype=np.float32),
                np.array([[5, 0, 10, 5, 5, 10, 0, 5, 0]], dtype=np.float32),
                0.5,
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                id="perfect_match_tp",
            ),
            pytest.param(
                np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0, 0.9]], dtype=np.float32),
                np.array([[5, 0, 10, 5, 5, 10, 0, 5, 0]], dtype=np.float32),
                0.3,
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                id="partial_obb_match_tp_at_threshold_0_3",
            ),
            pytest.param(
                np.array([[0, 0, 10, 0, 10, 10, 0, 10, 0, 0.9]], dtype=np.float32),
                np.array([[5, 0, 10, 5, 5, 10, 0, 5, 0]], dtype=np.float32),
                0.7,
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                id="partial_obb_no_match_at_threshold_0_7",
            ),
        ],
    )
    def test_confusion_matrix_from_tensors_obb(
        self, pred_tensor, target_tensor, iou_threshold, expected_matrix
    ):
        """Direct from_tensors OBB end-to-end coverage with pre-built tensors."""
        cm = ConfusionMatrix.from_tensors(
            predictions=[pred_tensor],
            targets=[target_tensor],
            classes=["box"],
            iou_threshold=iou_threshold,
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES,
        )
        np.testing.assert_array_equal(cm.matrix, expected_matrix)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                _call_confusion_matrix_from_detections_masks,
                id="from_detections",
            ),
            pytest.param(
                _call_confusion_matrix_from_tensors_masks,
                id="from_tensors",
            ),
            pytest.param(
                _call_confusion_matrix_evaluate_detection_batch_masks,
                id="evaluate_detection_batch",
            ),
        ],
    )
    def test_confusion_matrix_masks_rejection(self, call):
        """MetricTarget.MASKS raises ValueError at every public entry point."""
        with pytest.raises(ValueError, match=r"MetricTarget\.MASKS"):
            call()

    def test_confusion_matrix_multiclass_obb(self):
        """Multi-class OBB: TP for exact match, FP+FN for rotation mismatch."""
        classes = ["box", "circle"]

        # Ground truth: class-0 diamond at [0,0,10,10], class-1 diamond at [20,20,30,30]
        gt = [
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
                class_id=np.array([0, 1]),
                data={
                    "xyxyxyxy": np.array(
                        [
                            [[5, 0], [10, 5], [5, 10], [0, 5]],
                            [[25, 20], [30, 25], [25, 30], [20, 25]],
                        ],
                        dtype=np.float32,
                    )
                },
            )
        ]
        # Predictions: class-0 same diamond (OBB IoU=1.0 → TP);
        #              class-1 axis-aligned square (OBB IoU=0.5 < threshold=0.6 → FP+FN)
        pred = [
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
                class_id=np.array([0, 1]),
                confidence=np.array([0.9, 0.8], dtype=np.float32),
                data={
                    "xyxyxyxy": np.array(
                        [
                            [[5, 0], [10, 5], [5, 10], [0, 5]],
                            [[20, 20], [30, 20], [30, 30], [20, 30]],
                        ],
                        dtype=np.float32,
                    )
                },
            )
        ]

        cm = ConfusionMatrix.from_detections(
            predictions=pred,
            targets=gt,
            classes=classes,
            iou_threshold=0.6,
            metric_target=MetricTarget.ORIENTED_BOUNDING_BOXES,
        )

        assert cm.matrix[0, 0] == 1  # TP class 0
        assert cm.matrix[1, 2] == 1  # FN class 1 (unmatched GT)
        assert cm.matrix[2, 1] == 1  # FP class 1 (unmatched pred)

    def test_confusion_matrix_metric_target_persistence_from_detections(self):
        """metric_target field reflects the value passed to from_detections."""
        xyxy = np.array([[0, 0, 10, 10]], dtype=np.float32)
        cm = ConfusionMatrix.from_detections(
            predictions=[
                Detections(
                    xyxy=xyxy, class_id=np.array([0]), confidence=np.array([0.9])
                )
            ],
            targets=[Detections(xyxy=xyxy, class_id=np.array([0]))],
            classes=["box"],
            metric_target=MetricTarget.BOXES,
        )
        assert cm.metric_target == MetricTarget.BOXES

    def test_confusion_matrix_metric_target_persistence_from_tensors(self):
        """metric_target field reflects the value passed to from_tensors."""
        cm = ConfusionMatrix.from_tensors(
            predictions=[np.array([[0, 0, 10, 10, 0, 0.9]], dtype=np.float32)],
            targets=[np.array([[0, 0, 10, 10, 0]], dtype=np.float32)],
            classes=["box"],
            metric_target=MetricTarget.BOXES,
        )
        assert cm.metric_target == MetricTarget.BOXES

    def test_greedy_matching_two_valid_pairs(self):
        """Greedy matching finds both TPs; np.unique style missed the second pair."""
        preds = Detections(
            xyxy=np.array([[40, 60, 380, 470], [108, 60, 448, 470]], dtype=np.float32),
            confidence=np.array([0.95, 0.90]),
            class_id=np.array([0, 0]),
        )
        targets = Detections(
            xyxy=np.array([[40, 60, 380, 470], [210, 60, 550, 470]], dtype=np.float32),
            class_id=np.array([0, 0]),
        )

        result = MeanAveragePrecision.from_detections(
            predictions=[preds], targets=[targets]
        )

        assert result.map50 == pytest.approx(1.0, abs=0.01)


class TestSplitDetectionsByOutcome:
    """Tests for _split_detections_by_outcome matching and filtering logic."""

    def test_confidence_none_all_survive(self):
        """Predictions with no confidence score pass threshold regardless."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=None,
            class_id=np.array([0]),
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
        )

        tp, fp, fn = _split_detections_by_outcome(predictions, targets, 0.5, 0.5)

        assert len(tp) == 1
        assert len(fp) == 0
        assert len(fn) == 0

    def test_all_below_threshold_returns_zero_tp(self):
        """Predictions below conf_threshold are excluded, leaving only FN."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.2]),
            class_id=np.array([0]),
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
        )

        tp, fp, fn = _split_detections_by_outcome(predictions, targets, 0.5, 0.5)

        assert len(tp) == 0
        assert len(fp) == 0
        assert len(fn) == 1

    def test_cross_class_spatial_match(self):
        """Spatially overlapping but class-mismatched pair → FP + FN."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([1]),
        )

        tp, fp, fn = _split_detections_by_outcome(predictions, targets, 0.5, 0.0)

        assert len(tp) == 0
        assert len(fp) == 1
        assert len(fn) == 1

    def test_empty_predictions_all_fn(self):
        """Zero predictions → every target becomes a false negative."""
        predictions = Detections.empty()
        predictions.class_id = np.array([], dtype=np.int64)
        targets = Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
            class_id=np.array([0, 1]),
        )

        tp, fp, fn = _split_detections_by_outcome(predictions, targets, 0.5, 0.5)

        assert len(tp) == 0
        assert len(fp) == 0
        assert len(fn) == 2

    def test_empty_targets_all_fp(self):
        """Zero targets → every prediction becomes a false positive."""
        predictions = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        targets = Detections.empty()
        targets.class_id = np.array([], dtype=np.int64)

        tp, fp, fn = _split_detections_by_outcome(predictions, targets, 0.5, 0.5)

        assert len(tp) == 0
        assert len(fp) == 1
        assert len(fn) == 0

    @pytest.mark.parametrize(
        ("predictions", "targets"),
        [
            pytest.param(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=None,
                ),
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="predictions-class-id-none",
            ),
            pytest.param(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    class_id=None,
                ),
                id="targets-class-id-none",
            ),
        ],
    )
    def test_class_id_none_raises_value_error(self, predictions, targets):
        """Missing class_id on either input raises ValueError."""
        with pytest.raises(ValueError, match="class_id"):
            _split_detections_by_outcome(predictions, targets, 0.5, 0.5)
