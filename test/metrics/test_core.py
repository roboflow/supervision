from contextlib import AbstractContextManager as PytestExceptionType
from contextlib import ExitStack as DoesNotRaise
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pytest

from supervision.detection.core import Detections
from supervision.metrics.core import (
    CLASS_ID_NONE,
    InternalMetricDataStore,
    MetricTarget,
)


def mock_xyxy(*box_index: int, box_width=10) -> npt.NDArray[np.float32]:
    """
    Quickly generate a list of boxes.
    For each index in `box_index`, a box is generated with the top-left corner at
    (i, i) and the bottom-right corner at (i + box_width, i + box_width).
    """
    box_list = []
    for i in box_index:
        x0 = y0 = i
        x1 = y1 = i + box_width
        box_list.append([x0, y0, x1, y1])
    return np.array(box_list, dtype=np.float32)


def mock_detections(*box_indices: int, class_id: Optional[List[int]] = None):
    """Mock detections with xyxy and class_ids"""
    if len(box_indices) == 0:
        if class_id is not None and len(class_id) > 0:
            raise ValueError("class_id should be None or empty if box_indices is empty")
        return Detections.empty()

    return Detections(
        xyxy=mock_xyxy(*box_indices),
        class_id=None if class_id is None else np.array(class_id, dtype=int),
    )


def helper_test_store(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    class_agnostic: bool,
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
):
    store = InternalMetricDataStore(MetricTarget.BOXES, class_agnostic=class_agnostic)
    store.update(data_1, data_2)
    result = [result for result in store]
    assert len(result) == len(expected_result)

    for (class_id, content_1, content_2), (
        expected_class_id,
        expected_content_1,
        expected_content_2,
    ) in zip(result, expected_result):
        assert class_id == expected_class_id
        assert (content_1 is None and expected_content_1 is None) or np.array_equal(
            content_1, expected_content_1
        )
        assert (content_2 is None and expected_content_2 is None) or np.array_equal(
            content_2, expected_content_2
        )


@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Empty
        (mock_detections(), mock_detections(), [], DoesNotRaise()),
        (mock_xyxy(), mock_xyxy(), [], DoesNotRaise()),
        (mock_detections(), mock_xyxy(), [], DoesNotRaise()),
        (mock_xyxy(), mock_detections(), [], DoesNotRaise()),
        # With boxes
        (
            mock_detections(1),
            mock_detections(),
            [(CLASS_ID_NONE, mock_xyxy(1), None)],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_detections(1),
            [(CLASS_ID_NONE, None, mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2),
            mock_detections(),
            [
                (
                    CLASS_ID_NONE,
                    mock_xyxy(1, 2),
                    None,
                )
            ],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2),
            mock_detections(3, 4),
            [
                (
                    CLASS_ID_NONE,
                    mock_xyxy(1, 2),
                    mock_xyxy(3, 4),
                )
            ],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2),
            mock_xyxy(3, 4),
            [(CLASS_ID_NONE, mock_xyxy(1, 2), mock_xyxy(3, 4))],
            DoesNotRaise(),
        ),
        # With classes (that are ignored)
        (
            mock_detections(1, 2, class_id=[1, 2]),
            mock_detections(),
            [(CLASS_ID_NONE, mock_xyxy(1, 2), None)],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2, class_id=[1, 2]),
            mock_xyxy(3, 4),
            [(CLASS_ID_NONE, mock_xyxy(1, 2), mock_xyxy(3, 4))],
            DoesNotRaise(),
        ),
    ],
)
def test_store_boxes_class_agnostic(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
    exception: PytestExceptionType,
) -> None:
    with exception:
        helper_test_store(
            data_1,
            data_2,
            class_agnostic=True,
            expected_result=expected_result,
        )


# Boxes, by-class
@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Single box + Empty
        (
            mock_detections(1, class_id=[1]),
            mock_detections(),
            [(1, mock_xyxy(1), None)],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_detections(1, class_id=[1]),
            [(1, None, mock_xyxy(1))],
            DoesNotRaise(),
        ),
        # Multiple classes
        (
            mock_detections(1, 2, class_id=[1, 2]),
            mock_detections(),
            [
                (1, mock_xyxy(1), None),
                (2, mock_xyxy(2), None),
            ],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2, class_id=[1, 2]),
            mock_detections(3, 4, 5, class_id=[2, 3, 3]),
            [
                (1, mock_xyxy(1), None),
                (
                    2,
                    mock_xyxy(2),
                    mock_xyxy(3),
                ),
                (3, None, mock_xyxy(4, 5)),
            ],
            DoesNotRaise(),
        ),
        # array is the same as no class
        (
            mock_detections(1, 2, class_id=None),
            mock_xyxy(3, 4),
            [(CLASS_ID_NONE, mock_xyxy(1, 2), mock_xyxy(3, 4))],
            DoesNotRaise(),
        ),
    ],
)
def test_store_boxes_by_class(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
    exception: PytestExceptionType,
) -> None:
    with exception:
        helper_test_store(
            data_1, data_2, class_agnostic=False, expected_result=expected_result
        )


# Boxes, by-class
@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Single box + Empty
        (
            [],
            mock_detections(),
            [(1, mock_xyxy(), None)],
            pytest.raises(ValueError),
        ),
        (
            mock_detections(),
            [],
            [(1, None, mock_xyxy())],
            pytest.raises(ValueError),
        ),
    ],
)
def test_store_invalid_args(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
    exception: PytestExceptionType,
) -> None:
    with exception:
        helper_test_store(
            data_1,
            data_2,
            class_agnostic=False,
            expected_result=expected_result,
        )
