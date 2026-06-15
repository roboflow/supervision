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
    oriented_box_iou_batch,
    oriented_box_non_max_merge,
    oriented_box_non_max_suppression,
)
from tests.helpers import _generate_random_boxes


def _rotated_rect(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    """Return the 4 corners of a rotated rectangle as a (4, 2) float32 array."""
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


def _aabb_of(corners: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding box of a (4, 2) OBB corner array."""
    return np.array(
        [
            corners[:, 0].min(),
            corners[:, 1].min(),
            corners[:, 0].max(),
            corners[:, 1].max(),
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    ("predictions", "iou_threshold", "expected_result", "exception"),
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
    ("predictions", "iou_threshold", "expected_result", "exception"),
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
    ("predictions", "masks", "iou_threshold", "expected_result", "exception"),
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
    ("predictions", "masks", "iou_threshold", "expected_result", "exception"),
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
    ("box_true", "box_detection", "overlap_metric", "expected_overlap", "exception"),
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
    (
        "boxes_true",
        "boxes_detection",
        "overlap_metric",
        "expected_overlap",
        "exception",
    ),
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
            pytest.raises(ValueError, match="Invalid value: INVALID"),
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
    ("num_true", "num_det"),
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
    boxes_true = _generate_random_boxes(num_true)
    boxes_det = _generate_random_boxes(num_det)

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


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(np.array([[10, 1]], dtype=np.float32), id="x-dominant"),
        pytest.param(np.array([[1, 10]], dtype=np.float32), id="y-dominant"),
    ],
)
def test_oriented_box_iou_batch_is_invariant_to_non_square_scaling(
    scale: np.ndarray,
) -> None:
    """IoU is invariant under per-axis (anisotropic) scaling.

    An affine map multiplies every area — intersection and union alike — by the
    same determinant, so exact polygon IoU is unchanged. Equality is exact
    because the computation is rasterization-free; the analytic IoU of these two
    rectangles is 0.25.
    """
    boxes_true = np.array([[[1, 0], [0, 1], [3, 4], [4, 3]]], dtype=np.float32)
    boxes_detection = np.array([[[1, 1], [2, 0], [4, 2], [3, 3]]], dtype=np.float32)

    baseline_iou = oriented_box_iou_batch(boxes_true, boxes_detection)
    scaled_iou = oriented_box_iou_batch(boxes_true * scale, boxes_detection * scale)

    assert baseline_iou.shape == (1, 1)
    assert scaled_iou.shape == (1, 1)
    assert baseline_iou[0, 0] == pytest.approx(0.25)
    assert np.allclose(scaled_iou, baseline_iou, rtol=1e-5, atol=1e-7)


class TestOrientedBoxIouBatch:
    """Tests for `oriented_box_iou_batch`."""

    @pytest.mark.parametrize(
        ("scale", "offset"),
        [
            pytest.param(80.0, 0.0, id="large-scale"),
            pytest.param(1.0, 3000.0, id="far-from-origin"),
            pytest.param(80.0, 3000.0, id="large-and-far"),
        ],
    )
    def test_is_invariant_to_canvas_transforms(
        self, scale: float, offset: float
    ) -> None:
        """IoU is invariant under uniform scaling and translation.

        Both are affine maps, so exact polygon IoU matches the small-coordinate
        baseline exactly — free of the pixel-quantization error that the prior
        rasterization-based implementation exhibited."""
        boxes_true = _rotated_rect(50, 50, 40, 20, 30)[None]
        boxes_detection = _rotated_rect(52, 48, 40, 20, 35)[None]
        baseline = oriented_box_iou_batch(boxes_true, boxes_detection)

        transformed = oriented_box_iou_batch(
            boxes_true * scale + offset,
            boxes_detection * scale + offset,
        )

        assert baseline.shape == (1, 1)
        assert transformed.shape == (1, 1)
        assert baseline[0, 0] > 0.4
        assert np.allclose(transformed, baseline, rtol=1e-5, atol=1e-7)

    def test_supports_overlap_metric(self) -> None:
        """`overlap_metric=IOS` divides by the smaller area, so a small box fully
        contained in a larger one scores exactly 1.0, while IoU is the area ratio."""
        small = _rotated_rect(50, 50, 20, 20, 0)[None]
        large = _rotated_rect(50, 50, 60, 60, 0)[None]

        iou = oriented_box_iou_batch(small, large, OverlapMetric.IOU)[0, 0]
        ios = oriented_box_iou_batch(small, large, OverlapMetric.IOS)[0, 0]

        # 20x20 inside 60x60: IoU = 400 / 3600 = 1/9, IoS = 400 / 400 = 1.0.
        assert iou == pytest.approx(1 / 9, rel=1e-3)
        assert ios == pytest.approx(1.0)

    def test_half_overlap_matches_analytic_value(self) -> None:
        """Two unit-height 2x2 squares offset by 1 share half their area: IoU=1/3."""
        boxes_true = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32)
        boxes_detection = np.array([[[1, 0], [3, 0], [3, 2], [1, 2]]], dtype=np.float32)

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        # intersection = 2, union = 4 + 4 - 2 = 6, IoU = 2/6 = 1/3.
        assert iou[0, 0] == pytest.approx(1 / 3)

    def test_disjoint_boxes_score_zero(self) -> None:
        """Boxes whose envelopes do not overlap are pruned by the gate to 0.0."""
        boxes_true = _rotated_rect(10, 10, 8, 4, 20)[None]
        boxes_detection = _rotated_rect(500, 500, 8, 4, 70)[None]

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        assert iou[0, 0] == 0.0

    @pytest.mark.parametrize(
        "boxes",
        [
            pytest.param(
                np.stack([_rotated_rect(10, 10, 8, 4, 20)]),
                id="n=1",
            ),
            pytest.param(
                np.stack(
                    [_rotated_rect(10, 10, 8, 4, 20), _rotated_rect(40, 40, 8, 4, 70)]
                ),
                id="n=2",
            ),
        ],
    )
    def test_self_comparison_is_symmetric_with_unit_diagonal(
        self, boxes: np.ndarray
    ) -> None:
        """Comparing a set with itself yields a symmetric matrix and a 1.0 diagonal."""
        iou = oriented_box_iou_batch(boxes, boxes)

        assert np.allclose(iou, iou.T)
        for i in range(len(boxes)):
            assert iou[i, i] == pytest.approx(1.0)

    def test_envelope_overlap_without_polygon_overlap_scores_zero(self) -> None:
        """Parallel rotated bars share an envelope but not a body, so they score 0.

        Exercises the path where a pair passes the axis-aligned gate yet has no
        exact polygon intersection.
        """
        boxes_true = _rotated_rect(50, 50, 100, 4, 45)[None]
        boxes_detection = _rotated_rect(72, 28, 100, 4, 45)[None]

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        assert iou[0, 0] == 0.0

    def test_rejects_unsupported_overlap_metric(self) -> None:
        """An overlap metric other than IOU or IOS raises ValueError."""
        boxes = _rotated_rect(50, 50, 20, 20, 0)[None]

        with pytest.raises(ValueError, match="is not supported"):
            oriented_box_iou_batch(boxes, boxes, "invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("boxes_a", "boxes_b", "expected"),
        [
            pytest.param(
                np.array([[[5, 5], [5, 5], [5, 5], [5, 5]]], dtype=np.float32),
                np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32),
                0.0,
                id="collapsed-box-vs-normal",
            ),
            pytest.param(
                np.array([[[0, 0], [1, 0], [2, 0], [3, 0]]], dtype=np.float32),
                np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32),
                0.0,
                id="collinear-box-vs-normal",
            ),
            pytest.param(
                np.array([[[5, 5], [5, 5], [5, 5], [5, 5]]], dtype=np.float32),
                np.array([[[5, 5], [5, 5], [5, 5], [5, 5]]], dtype=np.float32),
                0.0,
                id="self-comparison-zero-area",
            ),
        ],
    )
    def test_degenerate_boxes_score_zero(
        self,
        boxes_a: np.ndarray,
        boxes_b: np.ndarray,
        expected: float,
    ) -> None:
        """Degenerate boxes (zero-area collapsed or collinear) always score 0.

        Note: self-comparison of a zero-area box returns 0.0, not 1.0 — this
        diverges from ``box_iou_batch`` which returns 1.0 for degenerate AABB
        self-comparison.
        """
        iou = oriented_box_iou_batch(boxes_a, boxes_b)
        assert iou[0, 0] == expected

    @pytest.mark.parametrize(
        ("n", "m"),
        [
            pytest.param(0, 3, id="zero-true"),
            pytest.param(3, 0, id="zero-detection"),
            pytest.param(0, 0, id="both-zero"),
        ],
    )
    def test_empty_input_returns_correct_shape(self, n: int, m: int) -> None:
        """Empty inputs trigger the early-return path; result is zero-filled."""
        boxes_true = np.empty((n, 4, 2), dtype=np.float32)
        boxes_detection = np.empty((m, 4, 2), dtype=np.float32)

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        assert iou.shape == (n, m)
        assert iou.dtype == np.float64
        assert np.all(iou == 0.0)

    @pytest.mark.parametrize(
        ("bad_shape", "match"),
        [
            pytest.param((2, 3, 2), r"expected \(N, 4, 2\)", id="3d-wrong-inner-dims"),
            pytest.param((2, 5), r"expected \(N, 8\)", id="2d-wrong-columns"),
            pytest.param((8,), r"must be 2-D", id="1d-input"),
        ],
    )
    def test_invalid_shape_raises_value_error(
        self, bad_shape: tuple, match: str
    ) -> None:
        """Inputs with wrong shape raise ValueError before any computation."""
        boxes_bad = np.zeros(bad_shape, dtype=np.float32)
        boxes_ok = np.zeros((2, 4, 2), dtype=np.float32)

        with pytest.raises(ValueError, match=match):
            oriented_box_iou_batch(boxes_bad, boxes_ok)


class TestOrientedBoxNonMaxSuppression:
    """Tests for `oriented_box_non_max_suppression`."""

    def test_keeps_x_pattern(self) -> None:
        """X-pattern: two thin rectangles crossing at +/-45° share an AABB but
        barely overlap as OBBs. AABB-NMS would suppress one; OBB-NMS must keep
        both."""
        quad_a = _rotated_rect(50, 50, 100, 10, +45)
        quad_b = _rotated_rect(50, 50, 100, 10, -45)
        oriented_boxes = np.stack([quad_a, quad_b])
        predictions = np.array(
            [
                [*_aabb_of(quad_a), 0.9, 0],
                [*_aabb_of(quad_b), 0.85, 0],
            ],
            dtype=np.float32,
        )

        assert box_iou_batch(predictions[:, :4], predictions[:, :4])[0, 1] > 0.95
        assert oriented_box_iou_batch(quad_a[None], quad_b[None])[0, 0] < 0.2

        keep = oriented_box_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes, iou_threshold=0.5
        )
        assert np.array_equal(keep, np.array([True, True]))

    @pytest.mark.parametrize(
        ("class_id_b", "expected_keep"),
        [
            pytest.param(0, [True, False], id="same-class"),
            pytest.param(1, [True, True], id="diff-class"),
        ],
    )
    def test_suppression_is_class_aware(
        self, class_id_b: int, expected_keep: list[bool]
    ) -> None:
        """Same class: lower-score OBB suppressed. Different class: both kept."""
        quad = _rotated_rect(50, 50, 100, 10, 45)
        shifted = _rotated_rect(51, 51, 100, 10, 45)
        oriented_boxes = np.stack([quad, shifted])
        predictions = np.array(
            [
                [*_aabb_of(quad), 0.9, 0],
                [*_aabb_of(shifted), 0.85, class_id_b],
            ],
            dtype=np.float32,
        )

        assert oriented_box_iou_batch(quad[None], shifted[None])[0, 0] > 0.9

        keep = oriented_box_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes, iou_threshold=0.5
        )
        assert np.array_equal(keep, np.array(expected_keep))

    def test_length_mismatch_raises(self) -> None:
        """Mismatched predictions and oriented_boxes must fail loudly, not
        silently misalign rows."""
        predictions = np.zeros((3, 5), dtype=np.float32)
        oriented_boxes = np.zeros((2, 4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="same length"):
            oriented_box_non_max_suppression(
                predictions=predictions, oriented_boxes=oriented_boxes
            )

    def test_empty_predictions(self) -> None:
        """No OBB predictions should produce an empty boolean keep mask."""
        predictions = np.empty((0, 5), dtype=np.float32)
        oriented_boxes = np.empty((0, 4, 2), dtype=np.float32)

        keep = oriented_box_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes
        )

        assert keep.shape == (0,)
        assert keep.dtype == bool

    @pytest.mark.parametrize("iou_threshold", [0.0, 0.5, 1.0])
    def test_keeps_single_prediction(self, iou_threshold: float) -> None:
        """A single OBB prediction is always kept, regardless of threshold."""
        quad = _rotated_rect(50, 50, 40, 20, 30)
        oriented_boxes = quad[None]
        predictions = np.array([[*_aabb_of(quad), 0.9, 0]], dtype=np.float32)

        keep = oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=oriented_boxes,
            iou_threshold=iou_threshold,
        )

        assert np.array_equal(keep, np.array([True]))

    @pytest.mark.parametrize(
        ("iou_threshold", "expected_keep"),
        [
            pytest.param(0.0, [True, False], id="threshold-0"),
            pytest.param(1.0, [True, True], id="threshold-1"),
        ],
    )
    def test_threshold_extremes(
        self, iou_threshold: float, expected_keep: list[bool]
    ) -> None:
        """At threshold extremes, positive-overlap non-identical OBBs suppress at
        0.0 and are both kept at 1.0."""
        quad_a = _rotated_rect(50, 50, 40, 40, 0)
        quad_b = _rotated_rect(55, 50, 40, 40, 0)
        oriented_boxes = np.stack([quad_a, quad_b])
        predictions = np.array(
            [
                [*_aabb_of(quad_a), 0.9, 0],
                [*_aabb_of(quad_b), 0.85, 0],
            ],
            dtype=np.float32,
        )

        overlap = oriented_box_iou_batch(quad_a[None], quad_b[None])[0, 0]
        assert 0.0 < overlap < 1.0

        keep = oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=oriented_boxes,
            iou_threshold=iou_threshold,
        )

        assert np.array_equal(keep, np.array(expected_keep))

    @pytest.mark.parametrize(
        ("overlap_metric", "expected_keep"),
        [
            pytest.param(OverlapMetric.IOU, [True, True], id="iou-keeps-both"),
            pytest.param(OverlapMetric.IOS, [True, False], id="ios-suppresses-small"),
        ],
    )
    def test_overlap_metric_determines_suppression(
        self, overlap_metric: OverlapMetric, expected_keep: list[bool]
    ) -> None:
        """Small box inside large: IOU keeps both; IOS suppresses small."""
        large = _rotated_rect(50, 50, 60, 60, 0)
        small = _rotated_rect(50, 50, 20, 20, 0)
        oriented_boxes = np.stack([large, small])
        predictions = np.array(
            [
                [*_aabb_of(large), 0.9, 0],
                [*_aabb_of(small), 0.85, 0],
            ],
            dtype=np.float32,
        )

        keep = oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=oriented_boxes,
            iou_threshold=0.5,
            overlap_metric=overlap_metric,
        )

        assert np.array_equal(keep, np.array(expected_keep))


class TestOrientedBoxNonMaxMerge:
    """Tests for `oriented_box_non_max_merge`."""

    def test_empty_predictions_returns_empty_groups(self) -> None:
        """No OBB predictions should produce no merge groups."""
        predictions = np.empty((0, 5), dtype=np.float32)
        oriented_boxes = np.empty((0, 4, 2), dtype=np.float32)

        groups = oriented_box_non_max_merge(
            predictions=predictions, oriented_boxes=oriented_boxes
        )

        assert groups == []

    def test_single_prediction_returns_singleton_group(self) -> None:
        """A single OBB prediction should be returned as one singleton group."""
        quad = _rotated_rect(50, 50, 40, 20, 30)
        oriented_boxes = quad[None]
        predictions = np.array([[*_aabb_of(quad), 0.9, 0]], dtype=np.float32)

        groups = oriented_box_non_max_merge(
            predictions=predictions, oriented_boxes=oriented_boxes
        )

        assert groups == [[0]]

    def test_groups_overlapping_oriented_boxes(self) -> None:
        """Two near-identical OBBs should be merged into one group; an X-pattern
        pair should produce two separate groups."""
        quad_dup_a = _rotated_rect(50, 50, 100, 10, 45)
        quad_dup_b = _rotated_rect(51, 51, 100, 10, 45)
        quad_x = _rotated_rect(50, 50, 100, 10, -45)
        oriented_boxes = np.stack([quad_dup_a, quad_dup_b, quad_x])
        predictions = np.array(
            [
                [*_aabb_of(quad_dup_a), 0.90, 0],
                [*_aabb_of(quad_dup_b), 0.85, 0],
                [*_aabb_of(quad_x), 0.80, 0],
            ],
            dtype=np.float32,
        )

        groups = oriented_box_non_max_merge(
            predictions=predictions, oriented_boxes=oriented_boxes, iou_threshold=0.5
        )

        sorted_groups = sorted(sorted(g) for g in groups)
        assert sorted_groups == [[0, 1], [2]]
