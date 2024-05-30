from contextlib import ExitStack as DoesNotRaise
from typing import List, Optional

import numpy as np
import pytest

from supervision.detection.overlap_filter import (
    box_non_max_suppression,
    group_overlapping_boxes,
    mask_non_max_suppression,
)


@pytest.mark.parametrize(
    "predictions, iou_threshold, expected_result, exception",
    [
        (
            np.empty(shape=(0, 5), dtype=float),
            0.5,
            [],
            DoesNotRaise(),
        ),
        (
            np.array([[0, 0, 10, 10, 1.0]]),
            0.5,
            [[0]],
            DoesNotRaise(),
        ),
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 9, 9, 1.0]]),
            0.5,
            [[1, 0]],
            DoesNotRaise(),
        ),  # High overlap, tie-break to second det
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 9, 9, 0.99]]),
            0.5,
            [[0, 1]],
            DoesNotRaise(),
        ),  # High overlap, merge to high confidence
        (
            np.array([[0, 0, 10, 10, 0.99], [0, 0, 9, 9, 1.0]]),
            0.5,
            [[1, 0]],
            DoesNotRaise(),
        ),  # (test symmetry) High overlap, merge to high confidence
        (
            np.array([[0, 0, 10, 10, 0.90], [0, 0, 9, 9, 1.0]]),
            0.5,
            [[1, 0]],
            DoesNotRaise(),
        ),  # (test symmetry) High overlap, merge to high confidence
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 9, 9, 1.0]]),
            1.0,
            [[1], [0]],
            DoesNotRaise(),
        ),  # High IOU required
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 9, 9, 1.0]]),
            0.0,
            [[1, 0]],
            DoesNotRaise(),
        ),  # No IOU required
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 5, 5, 0.9]]),
            0.25,
            [[0, 1]],
            DoesNotRaise(),
        ),  # Below IOU requirement
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 5, 5, 0.9]]),
            0.26,
            [[0], [1]],
            DoesNotRaise(),
        ),  # Above IOU requirement
        (
            np.array([[0, 0, 10, 10, 1.0], [0, 0, 9, 9, 1.0], [0, 0, 8, 8, 1.0]]),
            0.5,
            [[2, 1, 0]],
            DoesNotRaise(),
        ),  # 3 boxes
        (
            np.array(
                [
                    [0, 0, 10, 10, 1.0],
                    [0, 0, 9, 9, 1.0],
                    [5, 5, 10, 10, 1.0],
                    [6, 6, 10, 10, 1.0],
                    [9, 9, 10, 10, 1.0],
                ]
            ),
            0.5,
            [[4], [3, 2], [1, 0]],
            DoesNotRaise(),
        ),  # 5 boxes, 2 merges, 1 separate
        (
            np.array(
                [
                    [0, 0, 2, 1, 1.0],
                    [1, 0, 3, 1, 1.0],
                    [2, 0, 4, 1, 1.0],
                    [3, 0, 5, 1, 1.0],
                    [4, 0, 6, 1, 1.0],
                ]
            ),
            0.33,
            [[4, 3], [2, 1], [0]],
            DoesNotRaise(),
        ),  # sequential merge, half overlap
        (
            np.array(
                [
                    [0, 0, 2, 1, 0.9],
                    [1, 0, 3, 1, 0.9],
                    [2, 0, 4, 1, 1.0],
                    [3, 0, 5, 1, 0.9],
                    [4, 0, 6, 1, 0.9],
                ]
            ),
            0.33,
            [[2, 3, 1], [4], [0]],
            DoesNotRaise(),
        ),  # confidence
    ],
)
def test_group_overlapping_boxes(
    predictions: np.ndarray,
    iou_threshold: float,
    expected_result: List[List[int]],
    exception: Exception,
) -> None:
    with exception:
        result = group_overlapping_boxes(
            predictions=predictions, iou_threshold=iou_threshold
        )

        assert result == expected_result


@pytest.mark.parametrize(
    "predictions, iou_threshold, expected_result, exception",
    [
        (
            np.empty(shape=(0, 5)),
            0.5,
            np.array([]),
            DoesNotRaise(),
        ),  # single box with no category
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.8]]),
            0.5,
            np.array([True]),
            DoesNotRaise(),
        ),  # single box with no category
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.8, 0]]),
            0.5,
            np.array([True]),
            DoesNotRaise(),
        ),  # single box with category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8],
                    [15.0, 15.0, 40.0, 40.0, 0.9],
                ]
            ),
            0.5,
            np.array([False, True]),
            DoesNotRaise(),
        ),  # two boxes with no category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                    [15.0, 15.0, 40.0, 40.0, 0.9, 1],
                ]
            ),
            0.5,
            np.array([True, True]),
            DoesNotRaise(),
        ),  # two boxes with different category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                    [15.0, 15.0, 40.0, 40.0, 0.9, 0],
                ]
            ),
            0.5,
            np.array([False, True]),
            DoesNotRaise(),
        ),  # two boxes with same category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8],
                    [5.0, 5.0, 35.0, 45.0, 0.9],
                    [10.0, 10.0, 40.0, 50.0, 0.85],
                ]
            ),
            0.5,
            np.array([False, True, False]),
            DoesNotRaise(),
        ),  # three boxes with no category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                    [5.0, 5.0, 35.0, 45.0, 0.9, 1],
                    [10.0, 10.0, 40.0, 50.0, 0.85, 2],
                ]
            ),
            0.5,
            np.array([True, True, True]),
            DoesNotRaise(),
        ),  # three boxes with same category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                    [5.0, 5.0, 35.0, 45.0, 0.9, 0],
                    [10.0, 10.0, 40.0, 50.0, 0.85, 1],
                ]
            ),
            0.5,
            np.array([False, True, True]),
            DoesNotRaise(),
        ),  # three boxes with different category
    ],
)
def test_box_non_max_suppression(
    predictions: np.ndarray,
    iou_threshold: float,
    expected_result: Optional[np.ndarray],
    exception: Exception,
) -> None:
    with exception:
        result = box_non_max_suppression(
            predictions=predictions, iou_threshold=iou_threshold
        )
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "predictions, masks, iou_threshold, expected_result, exception",
    [
        (
            np.empty((0, 6)),
            np.empty((0, 5, 5)),
            0.5,
            np.array([]),
            DoesNotRaise(),
        ),  # empty predictions and masks
        (
            np.array([[0, 0, 0, 0, 0.8]]),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ]
                ]
            ),
            0.5,
            np.array([True]),
            DoesNotRaise(),
        ),  # single mask with no category
        (
            np.array([[0, 0, 0, 0, 0.8, 0]]),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ]
                ]
            ),
            0.5,
            np.array([True]),
            DoesNotRaise(),
        ),  # single mask with category
        (
            np.array([[0, 0, 0, 0, 0.8], [0, 0, 0, 0, 0.9]]),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, True, True],
                        [False, False, False, True, True],
                        [False, False, False, False, False],
                    ],
                ]
            ),
            0.5,
            np.array([True, True]),
            DoesNotRaise(),
        ),  # two masks non-overlapping with no category
        (
            np.array([[0, 0, 0, 0, 0.8], [0, 0, 0, 0, 0.9]]),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, True, True, True],
                        [False, False, True, True, True],
                        [False, False, True, True, True],
                        [False, False, False, False, False],
                    ],
                ]
            ),
            0.4,
            np.array([False, True]),
            DoesNotRaise(),
        ),  # two masks partially overlapping with no category
        (
            np.array([[0, 0, 0, 0, 0.8, 0], [0, 0, 0, 0, 0.9, 1]]),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, True, True, True],
                        [False, False, True, True, True],
                        [False, False, True, True, True],
                        [False, False, False, False, False],
                    ],
                ]
            ),
            0.5,
            np.array([True, True]),
            DoesNotRaise(),
        ),  # two masks partially overlapping with different category
        (
            np.array(
                [
                    [0, 0, 0, 0, 0.8],
                    [0, 0, 0, 0, 0.85],
                    [0, 0, 0, 0, 0.9],
                ]
            ),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, True, True],
                        [False, False, False, True, True],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                ]
            ),
            0.5,
            np.array([False, True, True]),
            DoesNotRaise(),
        ),  # three masks with no category
        (
            np.array(
                [
                    [0, 0, 0, 0, 0.8, 0],
                    [0, 0, 0, 0, 0.85, 1],
                    [0, 0, 0, 0, 0.9, 2],
                ]
            ),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, True, True, False, False],
                        [False, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                ]
            ),
            0.5,
            np.array([True, True, True]),
            DoesNotRaise(),
        ),  # three masks with different category
    ],
)
def test_mask_non_max_suppression(
    predictions: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float,
    expected_result: Optional[np.ndarray],
    exception: Exception,
) -> None:
    with exception:
        result = mask_non_max_suppression(
            predictions=predictions, masks=masks, iou_threshold=iou_threshold
        )
        assert np.array_equal(result, expected_result)
