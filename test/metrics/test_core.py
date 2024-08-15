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


class TestError(Exception):
    __test__ = False


def mock_xyxy(*box_index: int, box_width=10) -> npt.NDArray[np.float32]:
    """
    Quickly generate a list of boxes.
    For each index in `box_index`, a box is generated with the top-left corner at
    (i, i) and the bottom-right corner at (i + box_width, i + box_width).
    """
    if len(box_index) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    box_list = []
    for i in box_index:
        x0 = y0 = i
        x1 = y1 = i + box_width
        box_list.append([x0, y0, x1, y1])
    return np.array(box_list, dtype=np.float32)


def mock_mask(*box_index: int, box_width=10, image_size=20) -> np.ndarray:
    if len(box_index) == 0:
        return np.zeros((0, image_size, image_size), dtype=bool)

    if image_size < max(box_index) + box_width:
        raise TestError(
            f"image_size is too small. It should be at least"
            f" {max(box_index) + box_width}"
        )

    mask_list = []
    xyxy = mock_xyxy(*box_index, box_width=box_width).astype(int)
    for box in xyxy:
        mask = np.zeros((image_size, image_size), dtype=bool)
        mask[box[0] : box[2], box[1] : box[3]] = True
        mask_list.append(mask)
    return np.array(mask_list, dtype=bool)


def mock_detections(
    *box_indices: int, class_id: Optional[List[int]] = None, with_mask=False
):
    """Mock detections with xyxy and class_ids"""
    if len(box_indices) == 0:
        if class_id is not None and len(class_id) > 0:
            raise TestError("class_id should be None or empty if box_indices is empty")
        return Detections.empty()

    mask = None
    if with_mask:
        mask = mock_mask(*box_indices)

    return Detections(
        xyxy=mock_xyxy(*box_indices),
        class_id=None if class_id is None else np.array(class_id, dtype=int),
        mask=mask,
    )


def helper_test_store(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    class_agnostic: bool,
    metric_target: MetricTarget,
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
):
    store = InternalMetricDataStore(metric_target, class_agnostic=class_agnostic)
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


# Boxes


@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Empty
        (mock_detections(), mock_detections(), [], DoesNotRaise()),
        (mock_xyxy(), mock_xyxy(), [], DoesNotRaise()),
        (mock_detections(), mock_xyxy(), [], DoesNotRaise()),
        (mock_xyxy(), mock_detections(), [], DoesNotRaise()),
        # Exactly 1 box
        (
            mock_detections(1),
            mock_detections(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_detections(1),
            mock_xyxy(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(1),
            mock_detections(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(1),
            mock_xyxy(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_detections(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_xyxy(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(),
            mock_detections(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(),
            mock_xyxy(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        # More boxes
        (
            mock_detections(1, 2),
            mock_detections(),
            [
                (
                    CLASS_ID_NONE,
                    mock_xyxy(1, 2),
                    mock_xyxy(),
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
            [(CLASS_ID_NONE, mock_xyxy(1, 2), mock_xyxy())],
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
            metric_target=MetricTarget.BOXES,
            expected_result=expected_result,
        )


@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Empty
        (mock_detections(), mock_detections(), [], DoesNotRaise()),
        (mock_xyxy(), mock_xyxy(), [], DoesNotRaise()),
        (mock_detections(), mock_xyxy(), [], DoesNotRaise()),
        (mock_xyxy(), mock_detections(), [], DoesNotRaise()),
        # Exactly 1 box
        (
            mock_detections(1),
            mock_detections(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_detections(1),
            mock_xyxy(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(1),
            mock_detections(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(1),
            mock_xyxy(),
            [(CLASS_ID_NONE, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_detections(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_xyxy(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(),
            mock_detections(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        (
            mock_xyxy(),
            mock_xyxy(1),
            [(CLASS_ID_NONE, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        # More boxes
        (
            mock_detections(1, 2),
            mock_detections(),
            [
                (
                    CLASS_ID_NONE,
                    mock_xyxy(1, 2),
                    mock_xyxy(),
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
    ],
)
def test_store_boxes_by_class_regression(
    data_1: Union[npt.NDArray, Detections],
    data_2: Union[npt.NDArray, Detections],
    expected_result: List[Tuple[int, Optional[npt.NDArray], Optional[npt.NDArray]]],
    exception: PytestExceptionType,
) -> None:
    """Behaves exactly like class_agnostic if no classes are specified"""
    with exception:
        helper_test_store(
            data_1,
            data_2,
            class_agnostic=False,
            metric_target=MetricTarget.BOXES,
            expected_result=expected_result,
        )


@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Single box + Empty
        (
            mock_detections(1, class_id=[1]),
            mock_detections(),
            [(1, mock_xyxy(1), mock_xyxy())],
            DoesNotRaise(),
        ),
        (
            mock_detections(),
            mock_detections(1, class_id=[1]),
            [(1, mock_xyxy(), mock_xyxy(1))],
            DoesNotRaise(),
        ),
        # Multiple classes
        (
            mock_detections(1, 2, class_id=[1, 2]),
            mock_detections(),
            [
                (1, mock_xyxy(1), mock_xyxy()),
                (2, mock_xyxy(2), mock_xyxy()),
            ],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2, class_id=[1, 2]),
            mock_detections(3, 4, 5, class_id=[2, 3, 3]),
            [
                (1, mock_xyxy(1), mock_xyxy()),
                (
                    2,
                    mock_xyxy(2),
                    mock_xyxy(3),
                ),
                (3, mock_xyxy(), mock_xyxy(4, 5)),
            ],
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
            data_1,
            data_2,
            class_agnostic=False,
            metric_target=MetricTarget.BOXES,
            expected_result=expected_result,
        )


@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Single box + Empty
        (
            [],
            mock_detections(),
            [(1, mock_xyxy(), mock_xyxy())],
            pytest.raises(ValueError),
        ),
        (
            mock_detections(),
            [],
            [(1, mock_xyxy(), mock_xyxy())],
            pytest.raises(ValueError),
        ),
    ],
)
def test_store_boxes_invalid_args(
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
            metric_target=MetricTarget.BOXES,
            expected_result=expected_result,
        )


# Masks


@pytest.mark.parametrize(
    "data_1, data_2, expected_result, exception",
    [
        # Empty
        (
            mock_detections(with_mask=True),
            mock_detections(with_mask=True),
            [],
            DoesNotRaise(),
        ),
        (mock_mask(), mock_mask(), [], DoesNotRaise()),
        (mock_detections(with_mask=True), mock_mask(), [], DoesNotRaise()),
        (mock_mask(), mock_detections(with_mask=True), [], DoesNotRaise()),
        # Exactly 1 box
        (
            mock_detections(1, with_mask=True),
            mock_detections(with_mask=True),
            [(CLASS_ID_NONE, mock_mask(1), mock_mask())],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, with_mask=True),
            mock_mask(),
            [(CLASS_ID_NONE, mock_mask(1), mock_mask())],
            DoesNotRaise(),
        ),
        (
            mock_mask(1),
            mock_detections(with_mask=True),
            [(CLASS_ID_NONE, mock_mask(1), mock_mask())],
            DoesNotRaise(),
        ),
        (
            mock_mask(1),
            mock_mask(),
            [(CLASS_ID_NONE, mock_mask(1), mock_mask())],
            DoesNotRaise(),
        ),
        (
            mock_detections(with_mask=True),
            mock_detections(1, with_mask=True),
            [(CLASS_ID_NONE, mock_mask(), mock_mask(1))],
            DoesNotRaise(),
        ),
        (
            mock_detections(with_mask=True),
            mock_mask(1),
            [(CLASS_ID_NONE, mock_mask(), mock_mask(1))],
            DoesNotRaise(),
        ),
        (
            mock_mask(),
            mock_detections(1, with_mask=True),
            [(CLASS_ID_NONE, mock_mask(), mock_mask(1))],
            DoesNotRaise(),
        ),
        (
            mock_mask(),
            mock_mask(1),
            [(CLASS_ID_NONE, mock_mask(), mock_mask(1))],
            DoesNotRaise(),
        ),
        # More masks
        (
            mock_detections(1, 2, with_mask=True),
            mock_detections(with_mask=True),
            [
                (
                    CLASS_ID_NONE,
                    mock_mask(1, 2),
                    mock_mask(),
                )
            ],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2, with_mask=True),
            mock_detections(3, 4, with_mask=True),
            [
                (
                    CLASS_ID_NONE,
                    mock_mask(1, 2),
                    mock_mask(3, 4),
                )
            ],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2, with_mask=True),
            mock_mask(3, 4),
            [(CLASS_ID_NONE, mock_mask(1, 2), mock_mask(3, 4))],
            DoesNotRaise(),
        ),
        # With classes (that are ignored)
        (
            mock_detections(1, 2, class_id=[1, 2], with_mask=True),
            mock_detections(with_mask=True),
            [(CLASS_ID_NONE, mock_mask(1, 2), mock_mask())],
            DoesNotRaise(),
        ),
        (
            mock_detections(1, 2, class_id=[1, 2], with_mask=True),
            mock_mask(3, 4),
            [(CLASS_ID_NONE, mock_mask(1, 2), mock_mask(3, 4))],
            DoesNotRaise(),
        ),
    ],
)
def test_store_masks_class_agnostic(
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
            metric_target=MetricTarget.MASKS,
            expected_result=expected_result,
        )
