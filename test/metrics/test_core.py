from contextlib import ExitStack as DoesNotRaise
from test.test_utils import mock_detections
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import (
    InternalMetricDataStore,
    MetricTarget,
    CLASS_ID_NONE
)

# Boxes, class-agnostic 
@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Empty
        (
            Detections.empty(),
            Detections.empty(),
            [],
            DoesNotRaise()
        ),
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            [],
            DoesNotRaise()
        ),
        (
            Detections.empty(),
            np.empty((0, 4), dtype=np.float32),
            [],
            DoesNotRaise()
        ),

        # Single box + Empty
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32)
            ),
            Detections.empty(),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1]], dtype=np.float32),
                    None
                )
            ],
            DoesNotRaise()
        ),
        (
            Detections.empty(),
            Detections(
                xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32)
            ),
            [
                (
                    CLASS_ID_NONE,
                    None,
                    np.array([[0, 0, 1, 1]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),

        # Multiple boxes
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
            ),
            Detections.empty(),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                    None
                )
            ],
            DoesNotRaise()
        ),
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
            ),
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
            ),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
            ),
            np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),

        # with classes - should be ignored.
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                class_id=np.array([1, 2], dtype=int)
            ),
            np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),
    ]
)
def test_store_boxes_class_agnostic(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
    exception: Exception
) -> None:
    store = InternalMetricDataStore(MetricTarget.BOXES, class_agnostic=True)
    store.update(data_1, data_2)
    result = [result for result in store]
    assert len(result) == len(expected_result)
    with exception:
        for (class_id, content_1, content_2), (expected_class_id, expected_content_1, expected_content_2) in zip(result, expected_result):
            assert class_id == expected_class_id
            assert (content_1 is None and expected_content_1 is None) or np.array_equal(content_1, expected_content_1)
            assert (content_2 is None and expected_content_2 is None) or np.array_equal(content_2, expected_content_2)

# Boxes, by-class
@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Single box + Empty
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32),
                class_id=np.array([1], dtype=int)
            ),
            Detections.empty(),
            [
                (
                    1,
                    np.array([[0, 0, 1, 1]], dtype=np.float32),
                    None
                )
            ],
            DoesNotRaise()
        ),
        (
            Detections.empty(),
            Detections(
                xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32),
                class_id=np.array([1], dtype=int)
            ),
            [
                (
                    1,
                    None,
                    np.array([[0, 0, 1, 1]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),

        # Multiple classes
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                class_id=np.array([1, 2], dtype=int)
            ),
            Detections.empty(),
            [
                (
                    1,
                    np.array([[0, 0, 1, 1]], dtype=np.float32),
                    None
                ),
                (
                    2,
                    np.array([[0, 0, 2, 2]], dtype=np.float32),
                    None
                )
            ],
            DoesNotRaise()
        ),
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                class_id=np.array([1, 2], dtype=int)
            ),
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                class_id=np.array([2, 3], dtype=int)
            ),
            [
                (
                    1,
                    np.array([[0, 0, 1, 1]], dtype=np.float32),
                    None
                ),
                (
                    2,
                    np.array([[0, 0, 2, 2]], dtype=np.float32),
                    np.array([[0, 0, 1, 1]], dtype=np.float32)
                ),
                (
                    3,
                    None,
                    np.array([[0, 0, 2, 2]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
            ),
            np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),

        # with classes - should be ignored.
        (
            Detections(
                xyxy=np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                class_id=np.array([1, 2], dtype=int)
            ),
            np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
            [
                (
                    CLASS_ID_NONE,
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32),
                    np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.float32)
                )
            ],
            DoesNotRaise()
        ),
    ]
)
def test_store_boxes_by_class(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
    exception: Exception
) -> None:
    store = InternalMetricDataStore(MetricTarget.BOXES, class_agnostic=False)
    store.update(data_1, data_2)
    result = [result for result in store]
    assert len(result) == len(expected_result)
    with exception:
        for (class_id, content_1, content_2), (expected_class_id, expected_content_1, expected_content_2) in zip(result, expected_result):
            assert class_id == expected_class_id
            assert (content_1 is None and expected_content_1 is None) or np.array_equal(content_1, expected_content_1)
            assert (content_2 is None and expected_content_2 is None) or np.array_equal(content_2, expected_content_2)
