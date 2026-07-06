from __future__ import annotations

import math

import numpy as np
import pytest

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.converters import mask_to_xyxy
from supervision.detection.utils.mask_metrics import (
    boundary_f_score,
    boundary_iou,
    dice_coefficient,
    mask_iou,
)


def _make_compact_mask(mask: np.ndarray) -> CompactMask:
    dense_mask = np.expand_dims(mask.astype(bool), axis=0)
    xyxy = mask_to_xyxy(dense_mask).astype(np.float32)
    return CompactMask.from_dense(dense_mask, xyxy, image_shape=mask.shape)


def test_dice_coefficient_perfect_overlap() -> None:
    mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert dice_coefficient(mask, mask) == 1.0


def test_dice_coefficient_disjoint_masks() -> None:
    prediction = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    target = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )

    assert dice_coefficient(prediction, target) == 0.0


def test_dice_coefficient_partial_overlap() -> None:
    prediction = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    target = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert dice_coefficient(prediction, target) == 0.5


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (dice_coefficient, 1.0),
        (mask_iou, 1.0),
        (boundary_iou, 1.0),
        (boundary_f_score, 1.0),
    ],
)
def test_metrics_empty_vs_empty(metric, expected: float) -> None:
    mask = np.zeros((4, 4), dtype=bool)

    assert metric(mask, mask) == expected


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (dice_coefficient, 0.0),
        (mask_iou, 0.0),
        (boundary_iou, 0.0),
        (boundary_f_score, 0.0),
    ],
)
def test_metrics_empty_vs_non_empty(metric, expected: float) -> None:
    empty_mask = np.zeros((4, 4), dtype=bool)
    filled_mask = np.zeros((4, 4), dtype=bool)
    filled_mask[1:3, 1:3] = True

    assert metric(empty_mask, filled_mask) == expected


def test_mask_iou_partial_overlap() -> None:
    prediction = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    target = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert mask_iou(prediction, target) == pytest.approx(1.0 / 3.0)


def test_boundary_metrics_perfect_overlap() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True

    assert boundary_iou(mask, mask) == 1.0
    assert boundary_f_score(mask, mask) == 1.0


def test_boundary_metrics_shift_is_more_forgiving_with_tolerance() -> None:
    prediction = np.zeros((7, 7), dtype=bool)
    prediction[2:5, 1:4] = True

    target = np.zeros((7, 7), dtype=bool)
    target[2:5, 2:5] = True

    strict_boundary_iou = boundary_iou(prediction, target, tolerance=0)
    tolerant_boundary_iou = boundary_iou(prediction, target, tolerance=1)
    strict_boundary_f_score = boundary_f_score(prediction, target, tolerance=0)
    tolerant_boundary_f_score = boundary_f_score(prediction, target, tolerance=1)

    assert strict_boundary_iou < tolerant_boundary_iou <= 1.0
    assert strict_boundary_f_score < tolerant_boundary_f_score <= 1.0


def test_boundary_metrics_disjoint_masks() -> None:
    prediction = np.zeros((7, 7), dtype=bool)
    prediction[1:3, 1:3] = True

    target = np.zeros((7, 7), dtype=bool)
    target[4:6, 4:6] = True

    assert boundary_iou(prediction, target, tolerance=0) == 0.0
    assert boundary_f_score(prediction, target, tolerance=0) == 0.0


def test_shape_mismatch_raises_error() -> None:
    prediction = np.zeros((4, 4), dtype=bool)
    target = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError, match="must have the same shape"):
        dice_coefficient(prediction, target)


@pytest.mark.parametrize("tolerance", [-1, 1.5, True])
def test_invalid_tolerance_raises_error(tolerance: object) -> None:
    mask = np.zeros((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="tolerance must be a non-negative integer"):
        boundary_iou(mask, mask, tolerance=tolerance)  # type: ignore[arg-type]


def test_bool_and_uint8_masks_produce_same_result() -> None:
    prediction_bool = np.zeros((6, 6), dtype=bool)
    prediction_bool[1:5, 1:4] = True

    target_bool = np.zeros((6, 6), dtype=bool)
    target_bool[1:5, 2:5] = True

    prediction_uint8 = prediction_bool.astype(np.uint8)
    target_uint8 = target_bool.astype(np.uint8)

    assert dice_coefficient(prediction_bool, target_bool) == dice_coefficient(
        prediction_uint8, target_uint8
    )
    assert mask_iou(prediction_bool, target_bool) == mask_iou(
        prediction_uint8, target_uint8
    )
    assert boundary_iou(prediction_bool, target_bool, tolerance=1) == boundary_iou(
        prediction_uint8, target_uint8, tolerance=1
    )
    assert boundary_f_score(
        prediction_bool, target_bool, tolerance=1
    ) == boundary_f_score(prediction_uint8, target_uint8, tolerance=1)


def test_compact_mask_and_dense_mask_produce_same_scores() -> None:
    prediction = np.zeros((6, 6), dtype=bool)
    prediction[1:5, 1:4] = True

    target = np.zeros((6, 6), dtype=bool)
    target[1:5, 2:5] = True

    compact_prediction = _make_compact_mask(prediction)
    compact_target = _make_compact_mask(target)

    assert math.isclose(
        dice_coefficient(compact_prediction, compact_target),
        dice_coefficient(prediction, target),
    )
    assert math.isclose(
        mask_iou(compact_prediction, compact_target), mask_iou(prediction, target)
    )
    assert math.isclose(
        boundary_iou(compact_prediction, compact_target, tolerance=1),
        boundary_iou(prediction, target, tolerance=1),
    )
    assert math.isclose(
        boundary_f_score(compact_prediction, compact_target, tolerance=1),
        boundary_f_score(prediction, target, tolerance=1),
    )
