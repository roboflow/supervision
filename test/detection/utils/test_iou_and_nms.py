from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    _group_overlapping_boxes,
    box_iou,
    box_iou_batch,
    box_non_max_suppression,
    mask_non_max_merge,
    mask_non_max_suppression,
)
from test.test_utils import random_boxes


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
    expected_result: list[list[int]],
    exception: Exception,
) -> None:
    with exception:
        result = _group_overlapping_boxes(
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
    expected_result: np.ndarray | None,
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
    expected_result: np.ndarray | None,
    exception: Exception,
) -> None:
    with exception:
        result = mask_non_max_suppression(
            predictions=predictions, masks=masks, iou_threshold=iou_threshold
        )
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "predictions, masks, iou_threshold, expected_result, exception",
    [
        (
            np.empty((0, 6)),
            np.empty((0, 5, 5)),
            0.5,
            [],
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
            [[0]],
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
            [[0]],
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
            [[0], [1]],
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
            [[0, 1]],
            DoesNotRaise(),
        ),  # two masks partially overlapping with no category, merge
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
            0.6,
            [[0, 1]],
            DoesNotRaise(),
        ),  # two masks partially overlapping with no category, no merge
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
            0.4,
            [[0], [1]],
            DoesNotRaise(),
        ),  # two masks partially overlapping with different categories
        (
            np.array([[0, 0, 0, 0, 0.8, 0], [0, 0, 0, 0, 0.9, 0]]),
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
            [[0, 1]],
            DoesNotRaise(),
        ),  # two masks partially overlapping with same category
    ],
)
def test_mask_non_max_merge(
    predictions: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float,
    expected_result: list[list[int]],
    exception: Exception,
) -> None:
    with exception:
        result = mask_non_max_merge(
            predictions=predictions, masks=masks, iou_threshold=iou_threshold
        )
        sorted_result = sorted([sorted(group) for group in result])
        sorted_expected_result = sorted([sorted(group) for group in expected_result])
        assert sorted_result == sorted_expected_result


@pytest.mark.parametrize(
    "box_true, box_detection, overlap_metric, expected_overlap, exception",
    [
        (
            [100.0, 100.0, 200.0, 200.0],
            [150.0, 150.0, 250.0, 250.0],
            OverlapMetric.IOU,
            0.14285714285714285,
            DoesNotRaise(),
        ),  # partial overlap, IOU
        (
            [100.0, 100.0, 200.0, 200.0],
            [150.0, 150.0, 250.0, 250.0],
            OverlapMetric.IOS,
            0.25,
            DoesNotRaise(),
        ),  # partial overlap, IOS
        (
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            OverlapMetric.IOU,
            1.0,
            DoesNotRaise(),
        ),  # identical boxes, both boxes are arrays, IOU
        (
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            OverlapMetric.IOS,
            1.0,
            DoesNotRaise(),
        ),  # identical boxes, both boxes are arrays, IOS
        (
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            "iou",
            1.0,
            DoesNotRaise(),
        ),  # identical boxes, both boxes are arrays, IOU as lowercase string
        (
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            "ios",
            1.0,
            DoesNotRaise(),
        ),  # identical boxes, both boxes are arrays, IOS as lowercase string
        (
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            "IOU",
            1.0,
            DoesNotRaise(),
        ),  # identical boxes, both boxes are arrays, IOU as uppercase string
        (
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            "IOU",
            1.0,
            DoesNotRaise(),
        ),  # identical boxes, both boxes are arrays, IOS as uppercase string
        (
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            OverlapMetric.IOU,
            0.0,
            DoesNotRaise(),
        ),  # no overlap, IOU
        (
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            OverlapMetric.IOS,
            0.0,
            DoesNotRaise(),
        ),  # no overlap, IOS
        (
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 0.0, 20.0, 10.0],
            OverlapMetric.IOU,
            0.0,
            DoesNotRaise(),
        ),  # boxes touch at edge, zero intersection, IOU
        (
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 0.0, 20.0, 10.0],
            OverlapMetric.IOS,
            0.0,
            DoesNotRaise(),
        ),  # boxes touch at edge, zero intersection, IOU
        (
            [0.0, 0.0, 10.0, 10.0],
            [2.0, 2.0, 8.0, 8.0],
            OverlapMetric.IOU,
            0.36,
            DoesNotRaise(),
        ),  # one box inside another, IOU
        (
            [0.0, 0.0, 10.0, 10.0],
            [2.0, 2.0, 8.0, 8.0],
            OverlapMetric.IOS,
            1.0,
            DoesNotRaise(),
        ),  # one box inside another, IOS
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 10.0],
            OverlapMetric.IOU,
            0.0,
            DoesNotRaise(),
        ),  # degenerate true box with zero area, IOU
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 10.0],
            OverlapMetric.IOS,
            0.0,
            DoesNotRaise(),
        ),  # degenerate true box with zero area, IOS
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            OverlapMetric.IOU,
            0.0,
            DoesNotRaise(),
        ),  # both boxes fully degenerate, IOU
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            OverlapMetric.IOS,
            0.0,
            DoesNotRaise(),
        ),  # both boxes fully degenerate, IOS
        (
            [-5.0, 0.0, 5.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            OverlapMetric.IOU,
            1.0 / 3.0,
            DoesNotRaise(),
        ),  # negative x_min, overlapping boxes, IOU is 1/3
        (
            [-5.0, 0.0, 5.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            OverlapMetric.IOS,
            0.5,
            DoesNotRaise(),
        ),  # negative x_min, overlapping boxes, IOS is 0.5
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5, 1.5],
            OverlapMetric.IOU,
            0.14285714285714285,
            DoesNotRaise(),
        ),  # partial overlap with fractional coordinates, IOU
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5, 1.5],
            OverlapMetric.IOS,
            0.25,
            DoesNotRaise(),
        ),  # partial overlap with fractional coordinates, IOS
    ],
)
def test_box_iou(
    box_true: list[float] | np.ndarray,
    box_detection: list[float] | np.ndarray,
    overlap_metric: str | OverlapMetric,
    expected_overlap: float,
    exception: Exception,
) -> None:
    with exception:
        result = box_iou(
            box_true=box_true,
            box_detection=box_detection,
            overlap_metric=overlap_metric,
        )
        assert result == pytest.approx(expected_overlap, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize(
    "boxes_true, boxes_detection, overlap_metric, expected_overlap, exception",
    [
        # both inputs empty
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            OverlapMetric.IOU,
            np.empty((0, 0), dtype=np.float32),
            DoesNotRaise(),
        ),
        # one true box, no detections
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            OverlapMetric.IOU,
            np.empty((1, 0), dtype=np.float32),
            DoesNotRaise(),
        ),
        # no true boxes, one detection
        (
            np.empty((0, 4), dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.empty((0, 1), dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 partial overlap, IOU
        (
            np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32),
            np.array([[150.0, 150.0, 250.0, 250.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.14285715]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 partial overlap, IOS
        (
            np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32),
            np.array([[150.0, 150.0, 250.0, 250.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.25]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 identical boxes, IOU as lowercase string
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            "iou",
            np.array([[1.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 identical boxes, IOS as lowercase string
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            "ios",
            np.array([[1.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 identical boxes, IOU as uppercase string
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            "IOU",
            np.array([[1.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 identical boxes, IOS as uppercase string
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            "IOS",
            np.array([[1.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 no overlap, IOU
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 no overlap, IOS
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 touching at edge, zero intersection, IOU
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[10.0, 0.0, 20.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 touching at edge, zero intersection, IOS
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[10.0, 0.0, 20.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 box inside another, IOU
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 8.0, 8.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.36]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 box inside another, IOS
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 8.0, 8.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[1.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 degenerate true box, IOU
        (
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 degenerate true box, IOS
        (
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 both boxes degenerate, IOU
        (
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 both boxes degenerate, IOS
        (
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 negative coordinate, partial overlap, IOU
        (
            np.array([[-5.0, 0.0, 5.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[1.0 / 3.0]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 negative coordinate, partial overlap, IOS
        (
            np.array([[-5.0, 0.0, 5.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.5]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 fractional coordinates, partial overlap, IOU
        (
            np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[0.5, 0.5, 1.5, 1.5]], dtype=np.float32),
            OverlapMetric.IOU,
            np.array([[0.14285715]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # 1x1 fractional coordinates, partial overlap, IOS
        (
            np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[0.5, 0.5, 1.5, 1.5]], dtype=np.float32),
            OverlapMetric.IOS,
            np.array([[0.25]], dtype=np.float32),
            DoesNotRaise(),
        ),
        # true batch case, 2x2, IOU
        (
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [10.0, 10.0, 20.0, 20.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [5.0, 5.0, 15.0, 15.0],
                ],
                dtype=np.float32,
            ),
            OverlapMetric.IOU,
            np.array(
                [
                    [1.0, 0.14285715],
                    [0.0, 0.14285715],
                ],
                dtype=np.float32,
            ),
            DoesNotRaise(),
        ),
        # true batch case, 2x2, IOS
        (
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [10.0, 10.0, 20.0, 20.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [5.0, 5.0, 15.0, 15.0],
                ],
                dtype=np.float32,
            ),
            OverlapMetric.IOS,
            np.array(
                [
                    [1.0, 0.25],
                    [0.0, 0.25],
                ],
                dtype=np.float32,
            ),
            DoesNotRaise(),
        ),
        # invalid overlap_metric
        (
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            "invalid",
            None,
            pytest.raises(ValueError),
        ),
    ],
)
def test_box_iou_batch(
    boxes_true: np.ndarray,
    boxes_detection: np.ndarray,
    overlap_metric: str | OverlapMetric,
    expected_overlap: np.ndarray | None,
    exception: Exception,
) -> None:
    with exception:
        result = box_iou_batch(
            boxes_true=boxes_true,
            boxes_detection=boxes_detection,
            overlap_metric=overlap_metric,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == expected_overlap.shape
        assert np.allclose(
            result,
            expected_overlap,
            rtol=1e-6,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "num_true, num_det",
    [
        (5, 5),
        (5, 10),
        (10, 5),
        (10, 10),
        (20, 30),
        (30, 20),
        (50, 50),
        (100, 100),
    ],
)
@pytest.mark.parametrize(
    "overlap_metric",
    [OverlapMetric.IOU, OverlapMetric.IOS],
)
def test_box_iou_batch_symmetric_large(
    num_true: int,
    num_det: int,
    overlap_metric: OverlapMetric,
) -> None:
    boxes_true = random_boxes(num_true)
    boxes_det = random_boxes(num_det)

    result_ab = box_iou_batch(
        boxes_true=boxes_true,
        boxes_detection=boxes_det,
        overlap_metric=overlap_metric,
    )
    result_ba = box_iou_batch(
        boxes_true=boxes_det,
        boxes_detection=boxes_true,
        overlap_metric=overlap_metric,
    )

    assert result_ab.shape == (num_true, num_det)
    assert result_ba.shape == (num_det, num_true)
    assert np.allclose(
        result_ab,
        result_ba.T,
        rtol=1e-6,
        atol=1e-12,
    )
