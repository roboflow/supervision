from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils._fusion import (
    _box_weighted_box_fusion,
    _box_weighted_fusion_clusters,
    _oriented_box_weighted_fusion_clusters,
)
from supervision.detection.utils.iou import OverlapMetric
from supervision.detection.utils.nms import (
    _box_non_max_suppression,
    _box_soft_non_max_suppression,
    _group_overlapping_boxes,
    _mask_non_max_suppression,
    _mask_soft_non_max_suppression,
    box_non_max_merge,
    mask_non_max_merge,
    oriented_box_non_max_merge,
)
from supervision.utils.internal import SupervisionWarnings


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
    exception: AbstractContextManager[object],
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
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = _box_non_max_suppression(
            predictions=predictions, iou_threshold=iou_threshold
        )
        assert expected_result is not None
        assert np.array_equal(result, expected_result)


def test_box_soft_non_max_suppression_preserves_agnostic_score_precision() -> None:
    """Keep a positive decayed score for five-column float32 predictions."""
    predictions = np.array(
        [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.01]], dtype=np.float32
    )

    result = _box_soft_non_max_suppression(predictions, sigma=0.01)

    assert result.dtype == np.float64
    assert result[1] > 0.0


@pytest.mark.parametrize(
    ("predictions", "sigma", "expected_result", "exception"),
    [
        (
            np.empty(shape=(0, 5)),
            0.1,
            np.array([]),
            DoesNotRaise(),
        ),  # empty predictions
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.8]]),
            0.8,
            np.array([0.8]),
            DoesNotRaise(),
        ),  # single box with no category
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.8, 0]]),
            0.9,
            np.array([0.8]),
            DoesNotRaise(),
        ),  # single box with category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8],
                    [15.0, 15.0, 40.0, 40.0, 0.9],
                ]
            ),
            0.2,
            np.array([0.07176137, 0.9]),
            DoesNotRaise(),
        ),  # two boxes with no category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                    [15.0, 15.0, 40.0, 40.0, 0.9, 1],
                ]
            ),
            0.3,
            np.array([0.8, 0.9]),
            DoesNotRaise(),
        ),  # two boxes with different category
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                    [15.0, 15.0, 40.0, 40.0, 0.9, 0],
                ]
            ),
            0.9,
            np.array([0.46814354, 0.9]),
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
            0.7,
            np.array([0.42648529, 0.9, 0.53109062]),
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
            np.array([0.8, 0.9, 0.85]),
            DoesNotRaise(),
        ),  # three boxes with different category
        (
            np.array(
                [
                    [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                    [5.0, 5.0, 35.0, 45.0, 0.9, 0],
                    [10.0, 10.0, 40.0, 50.0, 0.85, 1],
                ]
            ),
            0.9,
            np.array([0.55491779, 0.9, 0.85]),
            DoesNotRaise(),
        ),  # three boxes with different category, one class isolated
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.5]]),
            0.0,
            None,
            pytest.raises(ValueError, match="sigma"),
        ),  # sigma at the lower boundary (not > 0) must raise
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.5]]),
            -0.1,
            None,
            pytest.raises(ValueError, match="sigma"),
        ),  # negative sigma must raise
        (
            np.array([[10.0, 10.0, 40.0, 40.0, 0.5]]),
            5.0,
            np.array([0.5]),
            DoesNotRaise(),
        ),  # sigma > 1 is valid for Soft-NMS (no upper bound)
        (
            np.array(
                [
                    [10.0, 10.0, 40.0, 40.0, 0.7],
                    [10.0, 10.0, 40.0, 40.0, 0.7],
                ]
            ),
            0.5,
            np.array([0.0947347, 0.7]),
            DoesNotRaise(),
        ),  # tied confidence: argsort's tie order picks a winner, exactly one of
        # the two identical, fully-overlapping boxes is decayed
    ],
)
def test_box_soft_non_max_suppression(
    predictions: np.ndarray,
    sigma: float,
    expected_result: np.ndarray | None,
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = _box_soft_non_max_suppression(predictions=predictions, sigma=sigma)
        assert expected_result is not None
        np.testing.assert_almost_equal(result, expected_result, decimal=5)


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
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = _mask_non_max_suppression(
            predictions=predictions, masks=masks, iou_threshold=iou_threshold
        )
        assert expected_result is not None
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("predictions", "masks", "sigma", "expected_result", "exception"),
    [
        (
            np.empty((0, 6)),
            np.empty((0, 5, 5)),
            0.1,
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
            0.2,
            np.array([0.8]),
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
            0.99,
            np.array([0.8]),
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
            0.8,
            np.array([0.8, 0.9]),
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
            0.6,
            np.array([0.5273925, 0.9]),
            DoesNotRaise(),
        ),  # two masks partially overlapping with no category — exact
        # full-resolution IoU (this module no longer resizes masks before
        # computing IoU, matching _mask_non_max_suppression's convention)
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
            0.9,
            np.array([0.8, 0.9]),
            DoesNotRaise(),
        ),  # two masks partially overlapping with different category
        (
            np.array([[0, 0, 0, 0, 0.5]]),
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
            0.0,
            None,
            pytest.raises(ValueError, match="sigma"),
        ),  # sigma at the lower boundary (not > 0) must raise
    ],
)
def test_mask_soft_non_max_suppression(
    predictions: np.ndarray,
    masks: np.ndarray,
    sigma: float,
    expected_result: np.ndarray | None,
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = _mask_soft_non_max_suppression(
            predictions=predictions, masks=masks, sigma=sigma
        )
        assert expected_result is not None
        np.testing.assert_almost_equal(result, expected_result, decimal=5)


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
            [[0], [1]],
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
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = mask_non_max_merge(
            predictions=predictions, masks=masks, iou_threshold=iou_threshold
        )
        sorted_result = sorted([sorted(group) for group in result])
        sorted_expected_result = sorted([sorted(group) for group in expected_result])
        assert sorted_result == sorted_expected_result


def test_mask_non_max_merge_ignores_mask_dimension_for_exact_iou() -> None:
    """Mask NMM ignores downscale dimension and uses exact mask overlap."""
    predictions = np.array(
        [[0, 0, 4, 4, 0.9, 0], [0, 0, 4, 4, 0.8, 0]], dtype=np.float32
    )
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[0, 0, 1] = True
    masks[1, 0, 0] = True
    masks[1, 1, 0] = True

    result = mask_non_max_merge(
        predictions=predictions,
        masks=masks,
        iou_threshold=0.9,
        mask_dimension=1,
    )

    assert sorted([sorted(group) for group in result]) == [[0], [1]]


def test_mask_non_max_merge_warns_for_legacy_positional_trailing_args() -> None:
    """Mask NMM supports legacy positional trailing args with warning."""
    predictions = np.array(
        [[0, 0, 4, 4, 0.9, 0], [0, 0, 4, 4, 0.8, 0]], dtype=np.float32
    )
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 0:2, 0:2] = True

    with pytest.warns(SupervisionWarnings, match="positionally.*deprecated"):
        legacy_result = mask_non_max_merge(
            predictions,
            masks,
            0.5,
            640,
            OverlapMetric.IOU,
        )
    with pytest.warns(SupervisionWarnings, match="positionally.*deprecated"):
        reordered_result = mask_non_max_merge(
            predictions,
            masks,
            0.5,
            OverlapMetric.IOU,
            640,
        )

    assert legacy_result == [[0, 1]]
    assert reordered_result == [[0, 1]]


def test_mask_non_max_merge_compact_mask_matches_dense_chained_union() -> None:
    """CompactMask NMM matches dense masks when merge candidates expand."""
    from supervision.detection.compact_mask import CompactMask

    predictions = np.array(
        [
            [0, 0, 6, 2, 0.9, 0],
            [0, 0, 6, 2, 0.8, 0],
            [0, 0, 6, 2, 0.7, 0],
        ],
        dtype=np.float32,
    )
    masks = np.zeros((3, 2, 6), dtype=bool)
    masks[0, :, 0:2] = True
    masks[1, :, 1:4] = True
    masks[2, :, 3:5] = True
    compact_mask = CompactMask.from_dense(
        masks=masks,
        xyxy=np.array([[0, 0, 5, 1], [0, 0, 5, 1], [0, 0, 5, 1]], dtype=np.float32),
        image_shape=(2, 6),
    )

    dense_result = mask_non_max_merge(
        predictions=predictions,
        masks=masks,
        iou_threshold=0.2,
    )
    compact_result = mask_non_max_merge(
        predictions=predictions,
        masks=compact_mask,
        iou_threshold=0.2,
    )

    assert sorted([sorted(group) for group in dense_result]) == [[0, 1, 2]]
    assert sorted([sorted(group) for group in compact_result]) == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# box_non_max_merge
# ---------------------------------------------------------------------------


class TestBoxNonMaxMerge:
    """box_non_max_merge groups overlapping boxes into clusters."""

    def test_empty_input_returns_empty_list(self) -> None:
        """An empty predictions array produces no merge groups."""
        predictions = np.empty((0, 5), dtype=np.float32)
        result = box_non_max_merge(predictions, iou_threshold=0.5)
        assert result == []

    def test_non_overlapping_boxes_each_own_group(self) -> None:
        """Boxes with zero overlap remain in separate singleton groups."""
        predictions = np.array(
            [
                [0, 0, 10, 10, 0.9],
                [20, 20, 30, 30, 0.8],
                [50, 50, 60, 60, 0.7],
            ],
            dtype=np.float32,
        )
        result = box_non_max_merge(predictions, iou_threshold=0.5)
        assert len(result) == 3
        assert all(len(g) == 1 for g in result)

    def test_identical_boxes_merged_into_one_group(self) -> None:
        """Perfectly overlapping boxes (IoU=1) are merged into a single group."""
        predictions = np.array(
            [
                [0, 0, 10, 10, 0.9],
                [0, 0, 10, 10, 0.8],
                [0, 0, 10, 10, 0.7],
            ],
            dtype=np.float32,
        )
        result = box_non_max_merge(predictions, iou_threshold=0.5)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_two_distinct_clusters(self) -> None:
        """Two groups of overlapping boxes produce exactly two merge groups."""
        predictions = np.array(
            [
                [0, 0, 10, 10, 0.9],
                [1, 1, 11, 11, 0.85],
                [50, 50, 60, 60, 0.8],
                [51, 51, 61, 61, 0.75],
            ],
            dtype=np.float32,
        )
        result = box_non_max_merge(predictions, iou_threshold=0.3)
        assert len(result) == 2
        group_sizes = sorted(len(g) for g in result)
        assert group_sizes == [2, 2]

    def test_6col_different_class_ids_not_merged(self) -> None:
        """Identical boxes with different class_ids stay in separate groups."""
        predictions = np.array(
            [
                [0, 0, 10, 10, 0.9, 0],
                [0, 0, 10, 10, 0.8, 1],
            ],
            dtype=np.float32,
        )
        result = box_non_max_merge(predictions, iou_threshold=0.5)
        assert len(result) == 2
        assert all(len(g) == 1 for g in result)

    def test_6col_same_class_id_merged(self) -> None:
        """Identical boxes with the same class_id are merged into one group."""
        predictions = np.array(
            [
                [0, 0, 10, 10, 0.9, 0],
                [0, 0, 10, 10, 0.8, 0],
            ],
            dtype=np.float32,
        )
        result = box_non_max_merge(predictions, iou_threshold=0.5)
        assert len(result) == 1
        assert len(result[0]) == 2


class TestBoxWeightedBoxFusion:
    """_box_weighted_box_fusion fuses overlapping boxes into confidence-weighted
    ones."""

    def test_fuses_overlap_just_above_threshold(self) -> None:
        """Fuse boxes whose float64 overlap rounds down to the threshold in float32."""
        predictions = np.array(
            [[0, 0, 100, 100, 0.9, 0], [0, 0, 100, 50.000002, 0.8, 0]],
            dtype=np.float64,
        )

        result = _box_weighted_box_fusion(predictions)

        assert result.shape == (1, 6)

    def test_keeps_identical_boxes_with_large_coordinates_in_one_cluster(self) -> None:
        """Fuse identical boxes at a huge coordinate origin into a single cluster.

        Averaging absolute coordinates of 1e16 loses low bits, which would shift the
        fused box away from its identical members and split the cluster.
        """
        predictions = np.array(
            [
                [1e16, 0, 1e16 + 2, 2, 0.9, 0],
                [1e16, 0, 1e16 + 2, 2, 0.8, 0],
                [1e16, 0, 1e16 + 2, 2, 0.7, 0],
            ],
            dtype=np.float64,
        )

        result = _box_weighted_box_fusion(predictions)

        assert result.shape == (1, 6)
        assert np.allclose(result[0, 4], 0.8)

    @pytest.mark.parametrize(
        ("overlap_metric", "expected_result"),
        [
            pytest.param(
                OverlapMetric.IOU,
                [[0.0, 0.0, 10.0, 10.0, 0.9], [2.0, 2.0, 8.0, 8.0, 0.6]],
                id="iou-keeps-contained-box-separate",
            ),
            pytest.param(
                OverlapMetric.IOS,
                [[0.8, 0.8, 9.2, 9.2, 0.75]],
                id="ios-fuses-contained-box",
            ),
        ],
    )
    def test_overlap_metric_controls_fusion_of_contained_box(
        self, overlap_metric: OverlapMetric, expected_result: list[list[float]]
    ) -> None:
        """A contained box fuses under IOS but stays separate under IOU."""
        predictions = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], [2.0, 2.0, 8.0, 8.0, 0.6]])

        result = _box_weighted_box_fusion(
            predictions, iou_threshold=0.5, overlap_metric=overlap_metric
        )

        assert result.shape == np.shape(expected_result)
        np.testing.assert_allclose(result, expected_result)

    @pytest.mark.parametrize(
        ("iou_threshold", "second_box", "expected_result"),
        [
            pytest.param(
                0.0,
                [20.0, 0.0, 30.0, 10.0, 0.6],
                [[0.0, 0.0, 10.0, 10.0, 0.9], [20.0, 0.0, 30.0, 10.0, 0.6]],
                id="zero-threshold-keeps-disjoint-boxes",
            ),
            pytest.param(
                0.0,
                [10.0, 0.0, 20.0, 10.0, 0.6],
                [[0.0, 0.0, 10.0, 10.0, 0.9], [10.0, 0.0, 20.0, 10.0, 0.6]],
                id="zero-threshold-keeps-touching-boxes",
            ),
            pytest.param(
                0.0,
                [2.0, 2.0, 8.0, 8.0, 0.6],
                [[0.8, 0.8, 9.2, 9.2, 0.75]],
                id="zero-threshold-fuses-positive-overlap",
            ),
            pytest.param(
                1.0,
                [0.0, 0.0, 10.0, 10.0, 0.6],
                [[0.0, 0.0, 10.0, 10.0, 0.9], [0.0, 0.0, 10.0, 10.0, 0.6]],
                id="unit-threshold-keeps-identical-boxes",
            ),
        ],
    )
    def test_threshold_endpoints_require_strictly_greater_overlap(
        self,
        iou_threshold: float,
        second_box: list[float],
        expected_result: list[list[float]],
    ) -> None:
        """Endpoint thresholds accept only overlaps strictly above the threshold."""
        predictions = np.array([[0.0, 0.0, 10.0, 10.0, 0.9], second_box])

        result = _box_weighted_box_fusion(predictions, iou_threshold=iou_threshold)

        assert result.shape == np.shape(expected_result)
        np.testing.assert_allclose(result, expected_result)

    @pytest.mark.parametrize(
        ("predictions", "iou_threshold", "expected_result"),
        [
            pytest.param(
                np.empty(shape=(0, 6)),
                0.5,
                np.empty(shape=(0, 6)),
                id="empty-input-preserves-shape",
            ),
            pytest.param(
                np.array([[10.0, 10.0, 40.0, 40.0, 0.8, 0.0]]),
                0.5,
                np.array([[10.0, 10.0, 40.0, 40.0, 0.8, 0.0]]),
                id="single-box-returned-unchanged",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
                    ]
                ),
                0.5,
                np.array([[0.8 / 1.7, 0.8 / 1.7, 17.8 / 1.7, 17.8 / 1.7, 0.85, 0.0]]),
                id="two-overlapping-same-class-confidence-weighted-fuse",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 5.0, 5.0, 0.7, 0.0],
                        [100.0, 100.0, 110.0, 110.0, 0.9, 1.0],
                    ]
                ),
                0.5,
                np.array(
                    [
                        [100.0, 100.0, 110.0, 110.0, 0.9, 1.0],
                        [0.0, 0.0, 5.0, 5.0, 0.7, 0.0],
                    ]
                ),
                id="non-overlapping-kept-and-ordered-by-score",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [0.0, 0.0, 10.0, 10.0, 0.8, 1.0],
                    ]
                ),
                0.5,
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [0.0, 0.0, 10.0, 10.0, 0.8, 1.0],
                    ]
                ),
                id="same-box-different-class-not-fused",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9],
                        [1.0, 1.0, 11.0, 11.0, 0.8],
                    ]
                ),
                0.5,
                np.array([[0.8 / 1.7, 0.8 / 1.7, 17.8 / 1.7, 17.8 / 1.7, 0.85]]),
                id="no-class-column-fuses-and-keeps-five-columns",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [3.0, 0.0, 13.0, 10.0, 0.8, 0.0],
                        [4.0, 0.0, 14.0, 10.0, 0.7, 0.0],
                    ]
                ),
                0.5,
                np.array([[5.2 / 2.4, 0.0, 29.2 / 2.4, 10.0, 0.8, 0.0]]),
                id="three-box-chain-joins-drifted-fused-box",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.95, 0.0],
                        [3.0, 0.0, 13.0, 10.0, 0.9, 0.0],
                        [4.5, 0.0, 14.5, 10.0, 0.8, 0.0],
                    ]
                ),
                0.5,
                np.array([[6.3 / 2.65, 0.0, 32.8 / 2.65, 10.0, 2.65 / 3, 0.0]]),
                id="three-box-chain-missing-seed-joins-fused-box",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [6.0, 0.0, 16.0, 10.0, 0.8, 0.0],
                        [4.0, 0.0, 14.0, 10.0, 0.7, 0.0],
                    ]
                ),
                0.3,
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [7.6 / 1.5, 0.0, 22.6 / 1.5, 10.0, 0.75, 0.0],
                    ]
                ),
                id="box-joins-best-overlapping-cluster-not-first",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [0.0, 0.0, 10.0, 5.0, 0.8, 0.0],
                    ]
                ),
                0.5,
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [0.0, 0.0, 10.0, 5.0, 0.8, 0.0],
                    ]
                ),
                id="overlap-equal-to-threshold-keeps-boxes-apart",
            ),
        ],
    )
    def test_fuses_same_class_boxes_and_preserves_other_predictions(
        self,
        predictions: np.ndarray,
        iou_threshold: float,
        expected_result: np.ndarray,
    ) -> None:
        """WBF fuses overlapping same-class boxes by confidence-weighted averaging."""
        result = _box_weighted_box_fusion(
            predictions=predictions, iou_threshold=iou_threshold
        )

        assert result.shape == expected_result.shape
        assert np.allclose(result, expected_result)

    def test_zero_scores_use_uniform_average(self) -> None:
        """All-zero confidences fuse via a plain coordinate average."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.0, 0.0],
            ]
        )

        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)

        assert result.shape == (1, 6)
        assert np.allclose(result[0, :4], [0.5, 0.5, 10.5, 10.5])

    def test_pulls_toward_higher_confidence_box(self) -> None:
        """The fused box lies closer to its higher-confidence member."""
        high_box = np.array([0.0, 0.0, 10.0, 10.0])
        low_box = np.array([4.0, 4.0, 14.0, 14.0])
        predictions = np.array(
            [
                [*high_box, 0.95, 0.0],
                [*low_box, 0.55, 0.0],
            ]
        )

        fused_box = _box_weighted_box_fusion(predictions, iou_threshold=0.2)[0, :4]

        assert np.all(np.abs(fused_box - high_box) < np.abs(fused_box - low_box))

    @pytest.mark.parametrize(
        "num_models", [2, pytest.param(np.int64(2), id="numpy-integer")]
    )
    def test_num_models_demotes_box_missed_by_other_models(
        self, num_models: int
    ) -> None:
        """A box seen by one of two models is halved; a box both models saw is not."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [50.0, 50.0, 60.0, 60.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
            ]
        )
        result = _box_weighted_box_fusion(
            predictions, iou_threshold=0.5, num_models=num_models
        )
        assert np.allclose(result[:, 4], [0.85, 0.45])

    def test_cluster_larger_than_num_models_keeps_mean_score(self) -> None:
        """A cluster with more boxes than models keeps its plain mean confidence."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
                [0.0, 1.0, 10.0, 11.0, 0.7, 0.0],
            ]
        )
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5, num_models=2)
        assert np.allclose(result[:, 4], [0.8])

    @pytest.mark.parametrize("num_models", [None, 1])
    def test_single_model_count_keeps_unscaled_confidence(
        self, num_models: int | None
    ) -> None:
        """Setting num_models to 1 gives the same output as leaving it unset."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
                [50.0, 50.0, 60.0, 60.0, 0.6, 0.0],
                [0.0, 0.0, 10.0, 10.0, 0.7, 1.0],
            ]
        )
        expected = np.array(
            [
                [0.8 / 1.7, 0.8 / 1.7, 17.8 / 1.7, 17.8 / 1.7, 0.85, 0.0],
                [0.0, 0.0, 10.0, 10.0, 0.7, 1.0],
                [50.0, 50.0, 60.0, 60.0, 0.6, 0.0],
            ]
        )

        result = _box_weighted_box_fusion(
            predictions, iou_threshold=0.5, num_models=num_models
        )

        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("num_models", [0, -1, 2.0, True])
    def test_rejects_num_models_that_is_not_positive_integer(
        self, num_models: int
    ) -> None:
        """A zero, negative, float or bool model count raises even for empty input."""
        predictions = np.empty((0, 6))
        with pytest.raises(ValueError, match="num_models must be a positive integer"):
            _box_weighted_box_fusion(predictions, num_models=num_models)

    def test_tied_classes_come_back_in_ascending_class_id(self) -> None:
        """Equal-score boxes of different classes come back in ascending class id."""
        predictions = np.array(
            [
                [40.0, 0.0, 50.0, 10.0, 0.5, 2.0],
                [0.0, 0.0, 10.0, 10.0, 0.5, 0.0],
                [20.0, 0.0, 30.0, 10.0, 0.5, 1.0],
            ]
        )
        expected_result = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.5, 0.0],
                [20.0, 0.0, 30.0, 10.0, 0.5, 1.0],
                [40.0, 0.0, 50.0, 10.0, 0.5, 2.0],
            ]
        )
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert np.array_equal(result, expected_result)

    def test_tied_boxes_of_one_class_keep_input_order(self) -> None:
        """Twenty disjoint equal-score boxes of one class come back in input order."""
        x_min = np.arange(20)[::-1] * 20.0
        predictions = np.column_stack(
            [
                x_min,
                np.zeros(20),
                x_min + 10.0,
                np.full(20, 10.0),
                np.full(20, 0.5),
                np.zeros(20),
            ]
        )
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert np.array_equal(result, predictions)

    @pytest.mark.parametrize("confidence", [-0.5, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_confidence(self, confidence: float) -> None:
        """A negative, NaN or infinite confidence raises instead of fusing a bad box."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, confidence, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="must be finite and non-negative"):
            _box_weighted_box_fusion(predictions, iou_threshold=0.5)

    def test_zero_confidence_box_adds_no_weight_to_cluster(self) -> None:
        """A zero-confidence box is accepted and joins without moving the cluster."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.8, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.0, 0.0],
            ]
        )
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert np.allclose(result, [[0.0, 0.0, 10.0, 10.0, 0.4, 0.0]])

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param(np.empty((0, 6)), id="empty-input"),
            pytest.param(np.array([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]]), id="single-box"),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [0.0, 0.0, 10.0, 10.0, 0.8, 1.0],
                    ]
                ),
                id="one-box-per-class",
            ),
            pytest.param(
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                        [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
                    ]
                ),
                id="overlapping-boxes",
            ),
        ],
    )
    def test_rejects_invalid_overlap_metric(self, predictions: np.ndarray) -> None:
        """An unknown overlap metric raises even when no overlap is ever computed."""
        with pytest.raises(ValueError, match="Invalid value"):
            _box_weighted_box_fusion(
                predictions,
                overlap_metric="bogus",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param(np.zeros((2, 4)), id="four-columns"),
            pytest.param(np.zeros((2, 7)), id="seven-columns"),
            pytest.param(
                np.array([0.0, 0.0, 10.0, 10.0, 0.9, 0.0]), id="one-dimensional-row"
            ),
            pytest.param(np.zeros((1, 2, 6)), id="three-dimensional"),
            pytest.param(np.empty((0,)), id="empty-one-dimensional"),
            pytest.param(np.empty((0, 4)), id="empty-four-columns"),
        ],
    )
    def test_rejects_predictions_without_five_or_six_columns(
        self, predictions: np.ndarray
    ) -> None:
        """Predictions not shaped (n, 5) or (n, 6) raise a clear ValueError."""
        with pytest.raises(ValueError, match=r"must have shape \(n, 5\) or \(n, 6\)"):
            _box_weighted_box_fusion(predictions, iou_threshold=0.5)

    @pytest.mark.parametrize("box_count", [0, 2])
    @pytest.mark.parametrize("column_count", [5, 6])
    def test_keeps_column_count_of_valid_predictions(
        self, box_count: int, column_count: int
    ) -> None:
        """Empty and non-empty (n, 5) and (n, 6) predictions keep their column count."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
            ]
        )[:box_count, :column_count]
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert result.shape[1] == column_count

    @pytest.mark.parametrize(
        ("dtype", "expected_dtype"),
        [
            pytest.param(np.float32, np.float32, id="float32-stays-float32"),
            pytest.param(np.float64, np.float64, id="float64-stays-float64"),
            pytest.param(np.int16, np.float32, id="int16-becomes-float32"),
            pytest.param(np.int64, np.float64, id="int64-becomes-float64"),
        ],
    )
    @pytest.mark.parametrize("box_count", [0, 2])
    def test_output_dtype_is_same_for_empty_and_fused_input(
        self, dtype: type, expected_dtype: type, box_count: int
    ) -> None:
        """Float input keeps its dtype and integer input turns float, empty or not."""
        predictions: np.ndarray = np.array(
            [
                [0, 0, 10, 10, 1, 0],
                [1, 1, 11, 11, 1, 0],
            ],
            dtype=dtype,
        )[:box_count]
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert result.dtype == expected_dtype

    def test_float32_input_fuses_to_weighted_box_and_mean_score(self) -> None:
        """Float32 input fuses to the confidence-weighted box and the mean score."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        expected_result = np.array(
            [[0.8 / 1.7, 0.8 / 1.7, 17.8 / 1.7, 17.8 / 1.7, 0.85, 0.0]]
        )
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert np.allclose(result, expected_result)

    def test_float32_scores_tied_after_rounding_keep_class_order(self) -> None:
        """Scores that tie only once rounded to float32 still come back by class id."""
        score = np.float32(0.5)
        score_above = np.nextafter(score, np.float32(1.0))
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, score, 1.0],
                [0.0, 0.0, 10.0, 10.0, score_above, 1.0],
                [50.0, 50.0, 60.0, 60.0, score, 0.0],
            ],
            dtype=np.float32,
        )
        expected_result = np.array(
            [
                [50.0, 50.0, 60.0, 60.0, 0.5, 0.0],
                [0.0, 0.0, 10.0, 10.0, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)
        assert np.array_equal(result, expected_result)

    @pytest.mark.parametrize(
        ("predictions", "expected_result"),
        [
            pytest.param(
                [
                    [0.0, 0.0, 100.0, 100.0, 1e308, 0.0],
                    [5.0, 5.0, 105.0, 105.0, 1e308, 0.0],
                ],
                [[2.5, 2.5, 102.5, 102.5, 1e308, 0.0]],
                id="equal-huge-scores-average-shifted-boxes",
            ),
            pytest.param(
                [
                    [0.0, 0.0, 10.0, 10.0, 1e308, 0.0],
                    [1.0, 1.0, 11.0, 11.0, 1e308, 0.0],
                ],
                [[0.5, 0.5, 10.5, 10.5, 1e308, 0.0]],
                id="equal-huge-scores-average-unit-shift",
            ),
            pytest.param(
                [
                    [0.0, 0.0, 100.0, 100.0, 1e308, 0.0],
                    [6.0, 6.0, 106.0, 106.0, 5e307, 0.0],
                ],
                [[2.0, 2.0, 102.0, 102.0, 7.5e307, 0.0]],
                id="unequal-huge-scores-keep-weighting",
            ),
            pytest.param(
                [
                    [0.0, 0.0, 10.0, 10.0, np.finfo(np.float64).max, 0.0],
                    [0.0, 0.0, 10.0, 10.0, np.finfo(np.float64).max, 0.0],
                ],
                [[0.0, 0.0, 10.0, 10.0, np.finfo(np.float64).max, 0.0]],
                id="largest-finite-score-stays-finite",
            ),
        ],
    )
    def test_huge_finite_confidences_fuse_without_overflow(
        self, predictions: list[list[float]], expected_result: list[list[float]]
    ) -> None:
        """Scores near the float64 limit fuse to the weighted box and mean score."""
        predictions_array = np.array(predictions, dtype=np.float64)

        result = _box_weighted_box_fusion(predictions_array, iou_threshold=0.5)

        np.testing.assert_allclose(result, expected_result)

    @pytest.mark.parametrize(
        ("predictions", "threshold", "expected_x"),
        [
            pytest.param(
                [[0.0, 0.0, 1e308, 0.1, 0.125]]
                + [[8e307, 0.0, 1.6e308, 0.1, 0.125]] * 5,
                0.1,
                6.666666666666666e307,
                id="finite-mean-of-overflowing-offsets",
            ),
            pytest.param(
                [[0.0, 0.0, 1e150, 1.0, 1e308], [1e148, 0.0, 1.01e150, 1.0, 1e-24]],
                0.5,
                1e-184,
                id="representable-offset-with-underflowing-weight",
            ),
        ],
    )
    def test_retains_weighted_offsets_across_extreme_ranges(
        self, predictions: list[list[float]], threshold: float, expected_x: float
    ) -> None:
        """A representable weighted coordinate neither overflows nor rounds to zero."""
        result = _box_weighted_box_fusion(
            np.array(predictions, dtype=np.float64), iou_threshold=threshold
        )

        assert result.shape == (1, 5)
        assert result[0, 0] == pytest.approx(expected_x, rel=1e-12, abs=0)

    @pytest.mark.parametrize("class_id", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_class_ids(self, class_id: float) -> None:
        """A NaN or infinite class id raises instead of forming its own fused class."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, class_id],
                [1.0, 1.0, 11.0, 11.0, 0.8, class_id],
            ]
        )

        with pytest.raises(
            ValueError,
            match="Class ids passed to Weighted Box Fusion must be finite",
        ):
            _box_weighted_box_fusion(predictions, iou_threshold=0.5)

    @pytest.mark.parametrize(
        "invalid_box",
        [
            pytest.param([np.nan, 0.0, 10.0, 10.0], id="nan-coordinate"),
            pytest.param([0.0, 0.0, np.inf, 10.0], id="infinite-coordinate"),
            pytest.param([10.0, 0.0, 0.0, 10.0], id="reversed-x-corners"),
            pytest.param([0.0, 10.0, 10.0, 0.0], id="reversed-y-corners"),
        ],
    )
    def test_rejects_non_finite_or_reversed_boxes(
        self, invalid_box: list[float]
    ) -> None:
        """A box with a non-finite coordinate or reversed corners raises."""
        predictions = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [*invalid_box, 0.8, 0.0],
            ]
        )

        with pytest.raises(
            ValueError, match="Boxes passed to Weighted Box Fusion must have finite"
        ):
            _box_weighted_box_fusion(predictions, iou_threshold=0.5)

    def test_keeps_zero_width_and_zero_height_boxes(self) -> None:
        """Degenerate boxes with equal corners are accepted and returned unfused."""
        predictions = np.array(
            [
                [0.0, 0.0, 0.0, 10.0, 0.9, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.8, 0.0],
            ]
        )

        result = _box_weighted_box_fusion(predictions, iou_threshold=0.5)

        assert np.array_equal(result, predictions)


class TestBoxWeightedFusionClusters:
    """The private cluster kernel reports which fused row each input row joined."""

    def test_labels_point_at_rows_sorted_by_fused_confidence(self) -> None:
        """Labels index the returned rows, not creation order or input order."""
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float64
        )
        scores = np.array([0.9, 0.8, 0.95])
        class_ids = np.array([0, 0, 1])

        fused_boxes, fused_scores, labels = _box_weighted_fusion_clusters(
            boxes, scores, class_ids, 0.5, OverlapMetric.IOU
        )

        assert np.allclose(fused_scores, [0.95, 0.85])
        assert np.array_equal(labels, [1, 1, 0])
        weighted_corner = 0.8 / 1.7
        assert np.allclose(
            fused_boxes[1],
            [
                weighted_corner,
                weighted_corner,
                10 + weighted_corner,
                10 + weighted_corner,
            ],
        )

    def test_tied_scores_keep_class_then_creation_order(self) -> None:
        """Equal fused scores order by class id, then by cluster creation order."""
        boxes = np.array(
            [[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float64
        )
        scores = np.array([0.9, 0.9, 0.9])
        class_ids = np.array([0, 1, 0])

        _, _, labels = _box_weighted_fusion_clusters(
            boxes, scores, class_ids, 0.5, OverlapMetric.IOU
        )

        assert np.array_equal(labels, [0, 2, 1])

    def test_without_class_ids_fuses_across_classes(self) -> None:
        """Passing no class ids puts overlapping boxes of any class in one cluster."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float64)
        scores = np.array([0.9, 0.8])

        _, fused_scores, labels = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU
        )

        assert np.allclose(fused_scores, [0.85])
        assert np.array_equal(labels, [0, 0])

    def test_output_dtype_applies_to_boxes_and_scores(self) -> None:
        """The fused boxes and scores come back in the requested dtype."""
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        fused_boxes, fused_scores, _ = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU, output_dtype=np.float32
        )

        assert fused_boxes.dtype == np.float32
        assert fused_scores.dtype == np.float32

    def test_sources_keep_one_box_per_source_in_a_cluster(self) -> None:
        """A second box of the same source starts its own cluster."""
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float64
        )
        scores = np.array([0.9, 0.8, 0.6])
        sources = np.array([0, 0, 1])

        _, fused_scores, labels = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU, sources=sources
        )

        assert np.allclose(fused_scores, [0.8, 0.75])
        assert np.array_equal(labels, [1, 0, 1])

    def test_sources_check_eligibility_before_picking_the_best_cluster(self) -> None:
        """A full best cluster does not hide an eligible runner-up."""
        boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [1, 0, 11, 10]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        sources = np.array([0, 1, 0])

        _, fused_scores, labels = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.4, OverlapMetric.IOU, sources=sources
        )

        assert np.allclose(fused_scores, [0.9, 0.75])
        assert np.array_equal(labels, [0, 1, 1])

    def test_uniform_box_weights_average_boxes_without_changing_scores(self) -> None:
        """Uniform weights give the plain mean box and the same mean score."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float64)
        scores = np.array([0.9, 0.1])

        uniform_boxes, uniform_scores, _ = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU, box_weights="uniform"
        )
        confidence_boxes, confidence_scores, _ = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU
        )

        assert np.allclose(uniform_boxes, [[0.5, 0.5, 10.5, 10.5]])
        assert np.allclose(confidence_boxes, [[0.1, 0.1, 10.1, 10.1]])
        assert np.allclose(uniform_scores, [0.5])
        assert np.array_equal(uniform_scores, confidence_scores)

    def test_uniform_box_weights_survive_a_subnormal_seed_score(self) -> None:
        """A subnormal score with uniform weights neither overflows nor gives NaN."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float64)
        scores = np.array([1e-310, 1e-310])

        fused_boxes, fused_scores, _ = _box_weighted_fusion_clusters(
            boxes, scores, None, 0.5, OverlapMetric.IOU, box_weights="uniform"
        )

        assert np.allclose(fused_boxes, [[0.5, 0.5, 10.5, 10.5]])
        assert np.allclose(fused_scores, [1e-310], rtol=1e-6, atol=0)

    def test_unknown_box_weights_raises(self) -> None:
        """Only 'confidence' and 'uniform' are valid box weights."""
        with pytest.raises(ValueError, match="box_weights must be"):
            _box_weighted_fusion_clusters(
                np.zeros((1, 4)),
                np.ones(1),
                None,
                0.5,
                OverlapMetric.IOU,
                box_weights="median",  # type: ignore[arg-type]
            )

    def test_empty_input_returns_empty_arrays(self) -> None:
        """No boxes yield no fused rows and no labels."""
        fused_boxes, fused_scores, labels = _box_weighted_fusion_clusters(
            np.empty((0, 4)), np.empty(0), None, 0.5, OverlapMetric.IOU
        )

        assert fused_boxes.shape == (0, 4)
        assert fused_scores.shape == (0,)
        assert labels.shape == (0,)


def _nmm_groups(kind: str, rectangles: list[list[int]]) -> list[list[int]]:
    """Run the NMM grouping for `kind` on pixel rectangles at `iou_threshold=0`.

    Each `[x1, y1, x2, y2]` rectangle covers pixels `x1:x2` by `y1:y2`; masks and
    oriented boxes are derived from it so every kind sees the same geometry. Confidences
    descend with the index.
    """
    boxes = np.array(rectangles, dtype=np.float32)
    confidence = np.linspace(0.9, 0.5, len(rectangles), dtype=np.float32)
    predictions = np.column_stack([boxes, confidence, np.zeros(len(boxes))])
    if kind == "box":
        return box_non_max_merge(predictions, iou_threshold=0.0)
    if kind == "oriented-box":
        x1, y1, x2, y2 = boxes.T
        corners = np.stack(
            [
                np.stack([x1, y1], axis=1),
                np.stack([x2, y1], axis=1),
                np.stack([x2, y2], axis=1),
                np.stack([x1, y2], axis=1),
            ],
            axis=1,
        )
        return oriented_box_non_max_merge(predictions, corners, iou_threshold=0.0)

    height = width = int(boxes.max()) + 1
    masks = np.zeros((len(rectangles), height, width), dtype=bool)
    for index, (left, top, right, bottom) in enumerate(rectangles):
        masks[index, top:bottom, left:right] = True
    if kind == "dense-mask":
        return mask_non_max_merge(predictions, masks, iou_threshold=0.0)
    from supervision.detection.compact_mask import CompactMask

    compact_mask = CompactMask.from_dense(
        masks=masks,
        xyxy=np.tile([0, 0, width - 1, height - 1], (len(rectangles), 1)),
        image_shape=(height, width),
    )
    return mask_non_max_merge(predictions, compact_mask, iou_threshold=0.0)


class TestNonMaxMergeZeroThreshold:
    """Non-max merge at `iou_threshold=0` groups only detections that overlap."""

    @pytest.mark.parametrize(
        "kind", ["box", "dense-mask", "compact-mask", "oriented-box"]
    )
    @pytest.mark.parametrize(
        ("rectangles", "expected_groups"),
        [
            pytest.param(
                [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]],
                [[0], [1], [2]],
                id="disjoint-stay-separate",
            ),
            pytest.param(
                [[0, 0, 10, 10], [10, 0, 20, 10]],
                [[0], [1]],
                id="shared-edge-stays-separate",
            ),
            pytest.param(
                [[0, 0, 10, 10], [9, 9, 19, 19]],
                [[0, 1]],
                id="one-pixel-overlap-merges",
            ),
        ],
    )
    def test_groups_only_overlapping_detections(
        self, kind: str, rectangles: list[list[int]], expected_groups: list[list[int]]
    ) -> None:
        """At threshold zero only pairs with positive overlap share a group."""
        groups = _nmm_groups(kind, rectangles)

        assert sorted(sorted(group) for group in groups) == expected_groups


class TestMaskNonMaxMergeDenseInputs:
    """Dense inputs of mask NMM keep their values and shape contract on any NumPy."""

    def test_float_mask_values_are_kept_in_the_first_overlap_step(self) -> None:
        """Non-bool masks keep their raw values in the first overlap step."""
        masks = np.zeros((2, 1, 4), dtype=np.float32)
        masks[0] = 0.6
        masks[1] = 1.0
        predictions = np.array([[0, 0, 3, 0, 0.9, 0], [0, 0, 3, 0, 0.8, 0]])

        result = mask_non_max_merge(predictions, masks, iou_threshold=0.7)

        assert result == [[0], [1]]

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((3, 12), id="flattened-2d"),
            pytest.param((3, 2, 3, 2), id="four-dimensional"),
        ],
    )
    def test_rejects_masks_that_are_not_3d(self, shape: tuple[int, ...]) -> None:
        """Boolean masks that are not (N, H, W) raise, as they do on the matmul path."""
        masks = np.ones(shape, dtype=bool)
        predictions = np.array(
            [[0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.8, 0], [0, 0, 1, 1, 0.7, 0]]
        )

        with pytest.raises(ValueError, match="3D"):
            mask_non_max_merge(predictions, masks, iou_threshold=0.5)


def _corners_of(xyxy: Sequence[Sequence[float]]) -> np.ndarray:
    """Corners (top-left, top-right, bottom-right, bottom-left) of axis-aligned
    boxes."""
    boxes = np.asarray(xyxy, dtype=np.float64)
    x1, y1, x2, y2 = boxes.T
    return np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )


class TestOrientedBoxWeightedFusionClusters:
    """The oriented cluster kernel mirrors the axis-aligned one on unrotated boxes."""

    def test_labels_and_scores_match_the_axis_aligned_kernel(self) -> None:
        """Two classes with tied scores give the same rows and labels as the boxes."""
        xyxy = [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 60, 60]]
        scores = np.array([0.9, 0.8, 0.9, 0.9])
        class_ids = np.array([0, 0, 1, 0])
        _, expected_scores, expected_labels = _box_weighted_fusion_clusters(
            np.asarray(xyxy, dtype=np.float64),
            scores,
            class_ids,
            0.5,
            OverlapMetric.IOU,
        )

        fused, fused_scores, labels = _oriented_box_weighted_fusion_clusters(
            _corners_of(xyxy), scores, class_ids, 0.5, OverlapMetric.IOU
        )

        assert fused.shape == (3, 4, 2)
        assert np.allclose(fused_scores, expected_scores)
        assert np.array_equal(labels, expected_labels)

    def test_one_box_per_source_in_a_cluster(self) -> None:
        """A box never joins a cluster that already holds a box of its source."""
        corners = _corners_of([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]])
        scores = np.array([0.9, 0.8, 0.7])
        sources = np.array([0, 0, 1])

        _, _, labels = _oriented_box_weighted_fusion_clusters(
            corners, scores, None, 0.5, OverlapMetric.IOU, sources=sources
        )

        assert labels[0] != labels[1]
        assert labels[2] in (labels[0], labels[1])

    def test_uniform_weights_average_the_extents_but_keep_the_score(self) -> None:
        """Uniform box weights give the plain mean box, leaving the score unchanged."""
        corners = _corners_of([[0, 0, 10, 10], [2, 0, 12, 10]])
        scores = np.array([0.9, 0.1])

        fused, fused_scores, _ = _oriented_box_weighted_fusion_clusters(
            corners, scores, None, 0.5, OverlapMetric.IOU, box_weights="uniform"
        )

        assert np.allclose(fused[0].min(axis=0), [1, 0])
        assert np.allclose(fused[0].max(axis=0), [11, 10])
        assert np.allclose(fused_scores, [0.5])

    def test_empty_input_returns_empty_arrays(self) -> None:
        """No boxes give empty fused corners, scores and labels."""
        fused, fused_scores, labels = _oriented_box_weighted_fusion_clusters(
            np.empty((0, 4, 2)), np.empty(0), None, 0.5, OverlapMetric.IOU
        )

        assert fused.shape == (0, 4, 2)
        assert len(fused_scores) == 0
        assert len(labels) == 0

    def test_unknown_box_weights_raise(self) -> None:
        """Only `confidence` and `uniform` weightings exist."""
        with pytest.raises(ValueError, match="box_weights must be"):
            _oriented_box_weighted_fusion_clusters(
                _corners_of([[0, 0, 10, 10]]),
                np.array([0.9]),
                None,
                0.5,
                OverlapMetric.IOU,
                box_weights="median",  # type: ignore[arg-type]
            )
