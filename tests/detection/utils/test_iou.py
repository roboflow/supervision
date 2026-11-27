"""Tests for box and mask IoU computations."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils import iou
from supervision.detection.utils.iou import (
    OverlapFilter,
    OverlapMetric,
    box_iou,
    box_iou_batch,
    box_iou_batch_with_jaccard,
    mask_iou_batch,
)
from supervision.detection.utils.nms import (
    _box_non_max_suppression,
    _box_weighted_box_fusion,
    _mask_non_max_suppression,
    _mask_soft_non_max_suppression,
    _oriented_box_non_max_suppression,
    box_non_max_merge,
    mask_non_max_merge,
    oriented_box_non_max_merge,
)
from tests.helpers import _generate_random_boxes


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
    exception: AbstractContextManager[object],
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
    exception: AbstractContextManager[object],
) -> None:
    """Return pairwise overlaps with the public float32 dtype and expected values."""
    with exception:
        result = box_iou_batch(
            boxes_true=boxes_true,
            boxes_detection=boxes_detection,
            overlap_metric=overlap_metric,
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert expected_overlap is not None
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


def _boundary_box_pair(
    origin: int, side: int = 50, shift: int = 25, dtype: type = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Build two overlapping boxes sharing an origin for precision tests.

    Both boxes are `side`x`side`; the second is shifted by `shift` along x so
    the intersection is `shift`x`side` and the union is `2*side**2 - shift*side`.
    """
    box_a: np.ndarray = np.array(
        [[origin, origin, origin + side, origin + side]], dtype=dtype
    )
    box_b: np.ndarray = np.array(
        [[origin + shift, origin, origin + side + shift, origin + side]],
        dtype=dtype,
    )
    return box_a, box_b


@pytest.mark.parametrize(
    ("origin", "overlap_metric", "expected"),
    [
        pytest.param(
            2**24 + 1, OverlapMetric.IOU, 1.0 / 3.0, id="iou-float64-above-2pow24"
        ),
        pytest.param(2**24 + 1, OverlapMetric.IOS, 0.5, id="ios-float64-above-2pow24"),
        pytest.param(2**24, OverlapMetric.IOU, 1.0 / 3.0, id="iou-float64-at-2pow24"),
        pytest.param(
            2**24 - 1, OverlapMetric.IOU, 1.0 / 3.0, id="iou-float64-just-below-2pow24"
        ),
    ],
)
def test_box_iou_batch_float64_input_precision_at_2pow24_boundary(
    origin: int,
    overlap_metric: OverlapMetric,
    expected: float,
) -> None:
    """`float64`/`int64` inputs keep full precision at the `2**24` boundary.

    `float64` accumulation is exact for these integer-valued coordinates, so the result
    must match the analytic value. Two 50x50 boxes shifted by 25 in x give intersection
    25*50=1250 and union 2*2500-1250=3750, so IoU=1/3 and IoS=0.5. This does not
    exercise the `float32`-storage case, which cannot be recovered inside this function;
    it guards the `float64`/`int64` accumulation path only.
    """
    box_a, box_b = _boundary_box_pair(origin, dtype=np.float64)

    result = box_iou_batch(
        boxes_true=box_a, boxes_detection=box_b, overlap_metric=overlap_metric
    )

    assert result[0, 0] == pytest.approx(expected, rel=1e-6)


def test_box_iou_batch_int32_input_does_not_overflow() -> None:
    """`int32` coordinates with large areas must not overflow to a wrong IoU.

    A 60000x60000 box has area 3.6e9, which wraps to a negative value in `int32`. Before
    upcasting the corners to `float64`, this made the union non-positive and the
    function returned `0.0`; it must now return the analytic IoU of 1/3.
    """
    side, shift = 60000, 30000
    box_a, box_b = _boundary_box_pair(0, side=side, shift=shift, dtype=np.int32)

    result = box_iou_batch(boxes_true=box_a, boxes_detection=box_b)

    assert result[0, 0] != 0.0
    assert result[0, 0] == pytest.approx(1.0 / 3.0, rel=1e-6)


@pytest.mark.parametrize(
    ("overlap_metric", "expected"),
    [
        pytest.param(OverlapMetric.IOU, 1.0 / 3.0, id="iou"),
        pytest.param(OverlapMetric.IOS, 0.5, id="ios"),
    ],
)
def test_box_iou_int32_input_does_not_overflow(
    overlap_metric: OverlapMetric, expected: float
) -> None:
    """Large int32 boxes preserve their analytic overlap scores."""
    side, shift = 60000, 30000
    box_a, box_b = _boundary_box_pair(0, side=side, shift=shift, dtype=np.int32)

    result = box_iou(
        box_true=box_a[0],
        box_detection=box_b[0],
        overlap_metric=overlap_metric,
    )

    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("overlap_metric", "expected"),
    [
        pytest.param(OverlapMetric.IOU, 1.0 / 3.0, id="iou"),
        pytest.param(OverlapMetric.IOS, 0.5, id="ios"),
    ],
)
def test_box_iou_large_int64_origin_preserves_coordinate_differences(
    overlap_metric: OverlapMetric, expected: float
) -> None:
    """Large int64 origins do not collapse distinct box endpoints."""
    origin = 2**53
    box_true = np.array([origin, 0, origin + 2, 2], dtype=np.int64)
    box_detection = np.array([origin + 1, 0, origin + 3, 2], dtype=np.int64)

    result = box_iou(box_true, box_detection, overlap_metric)

    assert result == pytest.approx(expected, rel=1e-6)


class TestBoxIouBatchLargeIntegerCoordinates:
    """Check overlap behavior for large integer coordinates."""

    def test_dense_pairs_across_row_chunks_preserve_overlap(self) -> None:
        """Dense large-coordinate pairs retain their overlaps across row chunks."""
        origin = 2**60
        boxes = np.tile(
            np.array([[origin, origin, origin + 10, origin + 10]]), (129, 1)
        )

        result = box_iou_batch(boxes, boxes)

        np.testing.assert_array_equal(result, np.ones((129, 129), dtype=np.float32))

    @pytest.mark.parametrize(
        ("dtype", "origin", "overlap_metric", "expected"),
        [
            pytest.param(
                np.int64, 2**60, OverlapMetric.IOU, 81.0 / 119.0, id="int64-pos-iou"
            ),
            pytest.param(
                np.int64, 2**60, OverlapMetric.IOS, 81.0 / 100.0, id="int64-pos-ios"
            ),
            pytest.param(
                np.int64, -(2**63), OverlapMetric.IOU, 81.0 / 119.0, id="int64-min-iou"
            ),
            pytest.param(
                np.int64, -(2**63), OverlapMetric.IOS, 81.0 / 100.0, id="int64-min-ios"
            ),
            pytest.param(
                np.uint64, 2**60, OverlapMetric.IOU, 81.0 / 119.0, id="uint64-pos-iou"
            ),
            pytest.param(
                np.uint64, 2**60, OverlapMetric.IOS, 81.0 / 100.0, id="uint64-pos-ios"
            ),
            pytest.param(
                np.uint64,
                2**64 - 20,
                OverlapMetric.IOU,
                81.0 / 119.0,
                id="uint64-max-iou",
            ),
            pytest.param(
                np.uint64,
                2**64 - 20,
                OverlapMetric.IOS,
                81.0 / 100.0,
                id="uint64-max-ios",
            ),
        ],
    )
    def test_box_iou_batch_large_64bit_integer_origins_preserve_overlap(
        self,
        dtype: type[np.integer],
        origin: int,
        overlap_metric: OverlapMetric,
        expected: float,
    ) -> None:
        """Batch IoU/IoS preserves overlap for narrow 64-bit integer boxes."""
        boxes_true = np.array(
            [[origin, origin, origin + 10, origin + 10]],
            dtype=dtype,
        )
        boxes_detection = np.array(
            [[origin + 1, origin + 1, origin + 11, origin + 11]],
            dtype=dtype,
        )

        result = box_iou_batch(boxes_true, boxes_detection, overlap_metric)

        assert result.shape == (1, 1)
        assert result.dtype == np.float32
        assert result[0, 0] == pytest.approx(expected, rel=1e-6)
        assert result[0, 0] == pytest.approx(
            box_iou(boxes_true[0], boxes_detection[0], overlap_metric),
            rel=1e-6,
        )

    @pytest.mark.parametrize(
        ("overlap_metric", "expected"),
        [
            pytest.param(OverlapMetric.IOU, 81.0 / 119.0, id="iou"),
            pytest.param(OverlapMetric.IOS, 81.0 / 100.0, id="ios"),
        ],
    )
    def test_box_iou_batch_mixed_int64_and_uint64_at_large_origin(
        self,
        overlap_metric: OverlapMetric,
        expected: float,
    ) -> None:
        """Mixed int64 and uint64 batches avoid float64 promotion at large origins."""
        origin = 2**60
        boxes_true = np.array(
            [
                [origin, origin, origin + 10, origin + 10],
                [-(2**60), 0, -(2**60) + 10, 10],
            ],
            dtype=np.int64,
        )
        boxes_detection = np.array(
            [[origin + 1, origin + 1, origin + 11, origin + 11]],
            dtype=np.uint64,
        )

        result = box_iou_batch(boxes_true, boxes_detection, overlap_metric)

        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(expected, rel=1e-6)
        assert result[0, 0] == pytest.approx(
            box_iou(
                boxes_true[0].tolist(), boxes_detection[0].tolist(), overlap_metric
            ),
            rel=1e-6,
        )
        assert result[1, 0] == 0.0

    def test_box_iou_batch_multi_origin_int64_pairs(self) -> None:
        """Same-origin pairs match exactly while cross-origin pairs score zero."""
        neg = -(2**60)
        pos = 2**60
        boxes_true = np.array(
            [
                [neg, 0, neg + 10, 10],
                [pos, 0, pos + 10, 10],
            ],
            dtype=np.int64,
        )
        boxes_detection = np.array(
            [
                [neg + 1, 1, neg + 11, 11],
                [pos + 1, 1, pos + 11, 11],
            ],
            dtype=np.int64,
        )

        result = box_iou_batch(boxes_true, boxes_detection, OverlapMetric.IOU)

        expected = 81.0 / 119.0
        assert result[0, 0] == pytest.approx(expected, rel=1e-6)
        assert result[1, 1] == pytest.approx(expected, rel=1e-6)
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0

    @pytest.mark.parametrize(
        ("overlap_metric", "expected"),
        [
            pytest.param(OverlapMetric.IOU, 100.0 / float((2**64 - 1) * 10), id="iou"),
            pytest.param(OverlapMetric.IOS, 1.0, id="ios"),
        ],
    )
    def test_box_iou_batch_full_range_int64_span_does_not_overflow(
        self,
        overlap_metric: OverlapMetric,
        expected: float,
    ) -> None:
        """Full-range int64 spans compute exact overlap without overflow."""
        origin = 2**60
        spanning = np.array([[-(2**63), 0, 2**63 - 1, 10]], dtype=np.int64)
        small = np.array([[origin, 0, origin + 10, 10]], dtype=np.int64)

        with np.errstate(over="raise", invalid="raise"):
            result = box_iou_batch(spanning, small, overlap_metric)

        assert result[0, 0] == pytest.approx(expected, rel=1e-6, abs=0.0)

    @pytest.mark.parametrize(
        "overlap_metric",
        [
            pytest.param(OverlapMetric.IOU, id="iou"),
            pytest.param(OverlapMetric.IOS, id="ios"),
        ],
    )
    def test_box_iou_batch_large_int64_edge_touching_and_zero_area_score_zero(
        self,
        overlap_metric: OverlapMetric,
    ) -> None:
        """Edge-touching and zero-area boxes at large int64 origins score zero."""
        origin = 2**60
        reference = np.array(
            [[origin, origin, origin + 10, origin + 10]], dtype=np.int64
        )
        edge_and_zero = np.array(
            [
                [origin + 10, origin, origin + 20, origin + 10],
                [origin, origin, origin, origin + 10],
            ],
            dtype=np.int64,
        )

        scores = box_iou_batch(reference, edge_and_zero, overlap_metric)

        np.testing.assert_array_equal(scores, np.zeros((1, 2), dtype=np.float32))

    def test_small_int64_boxes_skip_python_integer_areas(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ordinary small int64 boxes stay on the vectorized float64 fast path."""

        def _fail_if_called(*args: object, **kwargs: object) -> object:
            raise AssertionError("_large_integer_box_areas should not be called")

        monkeypatch.setattr(iou, "_large_integer_box_areas", _fail_if_called)
        boxes_true = np.array([[0, 0, 10, 10]], dtype=np.int64)
        boxes_detection = np.array([[5, 5, 15, 15]], dtype=np.int64)

        result = box_iou_batch(boxes_true, boxes_detection, OverlapMetric.IOU)

        assert result[0, 0] == pytest.approx(25.0 / 175.0, rel=1e-6)

    @pytest.mark.parametrize(
        ("overlap_metric", "expected"),
        [
            pytest.param(OverlapMetric.IOU, 1.0 / 3.0, id="iou"),
            pytest.param(OverlapMetric.IOS, 0.5, id="ios"),
        ],
    )
    def test_coordinates_above_float64_exact_range_keep_unit_overlap(
        self,
        overlap_metric: OverlapMetric,
        expected: float,
    ) -> None:
        """Coordinates beyond 2**53 dispatch to the large-integer path for exactness."""
        origin = 2**53
        boxes_true = np.array([[origin, 0, origin + 2, 2]], dtype=np.int64)
        boxes_detection = np.array([[origin + 1, 0, origin + 3, 2]], dtype=np.int64)

        result = box_iou_batch(boxes_true, boxes_detection, overlap_metric)

        assert result[0, 0] == pytest.approx(expected, rel=1e-6)


def test_box_iou_rejects_complex_coordinates() -> None:
    """Complex coordinates are rejected instead of silently truncating values."""
    box = np.array([0, 0, 2, 2], dtype=np.complex128)

    with pytest.raises(TypeError, match="real-valued"):
        box_iou(box, box)


class TestIouThresholdValidation:
    """Invalid IoU thresholds must raise `ValueError` on public overlap filters."""

    @pytest.mark.parametrize(
        ("fn_function", "kwargs"),
        [
            pytest.param(
                _box_non_max_suppression,
                {"predictions": np.empty((0, 5), dtype=np.float32)},
                id="box-nms",
            ),
            pytest.param(
                box_non_max_merge,
                {"predictions": np.empty((0, 5), dtype=np.float32)},
                id="box-nmm",
            ),
            pytest.param(
                _box_weighted_box_fusion,
                {"predictions": np.empty((0, 5), dtype=np.float32)},
                id="box-weighted-box-fusion",
            ),
            pytest.param(
                _mask_non_max_suppression,
                {
                    "predictions": np.empty((0, 5), dtype=np.float32),
                    "masks": np.empty((0, 1, 1), dtype=bool),
                },
                id="mask-nms",
            ),
            pytest.param(
                mask_non_max_merge,
                {
                    "predictions": np.empty((0, 5), dtype=np.float32),
                    "masks": np.empty((0, 1, 1), dtype=bool),
                },
                id="mask-nmm",
            ),
            pytest.param(
                _oriented_box_non_max_suppression,
                {
                    "predictions": np.empty((0, 5), dtype=np.float32),
                    "oriented_boxes": np.empty((0, 4, 2), dtype=np.float32),
                },
                id="obb-nms",
            ),
            pytest.param(
                oriented_box_non_max_merge,
                {
                    "predictions": np.empty((0, 5), dtype=np.float32),
                    "oriented_boxes": np.empty((0, 4, 2), dtype=np.float32),
                },
                id="obb-nmm",
            ),
        ],
    )
    @pytest.mark.parametrize("iou_threshold", [-0.1, 1.1])
    def test_rejects_thresholds_outside_closed_unit_interval(
        self,
        fn_function: Callable[..., object],
        kwargs: dict[str, object],
        iou_threshold: float,
    ) -> None:
        """Each public overlap filter rejects thresholds outside [0, 1]."""
        with pytest.raises(ValueError, match="closed range from 0 to 1"):
            fn_function(iou_threshold=iou_threshold, **kwargs)


def _naive_mask_iou(
    masks_true: np.ndarray,
    masks_detection: np.ndarray,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> np.ndarray:
    """Reference IoU that materialises the (N, M, H, W) overlap explicitly."""
    intersection = np.logical_and(masks_true[:, None], masks_detection).sum(axis=(2, 3))
    area_true = masks_true.sum(axis=(1, 2))
    area_detection = masks_detection.sum(axis=(1, 2))
    if overlap_metric == OverlapMetric.IOU:
        denominator = area_true[:, None] + area_detection - intersection
    else:
        denominator = np.minimum(area_true[:, None], area_detection)
    iou: np.ndarray = np.divide(
        intersection,
        denominator,
        out=np.zeros((len(masks_true), len(masks_detection)), dtype=float),
        where=denominator != 0,
    )
    return iou


class TestMaskIouBatch:
    """`mask_iou_batch` (matmul intersection) matches the naive boolean reference."""

    @pytest.mark.parametrize(
        "overlap_metric",
        [
            pytest.param(OverlapMetric.IOU, id="iou"),
            pytest.param(OverlapMetric.IOS, id="ios"),
        ],
    )
    @pytest.mark.parametrize(
        ("n_true", "n_detection", "height", "width"),
        [
            pytest.param(1, 1, 20, 24, id="single"),
            pytest.param(7, 7, 32, 40, id="several-square"),
            pytest.param(5, 9, 28, 36, id="rectangular"),
        ],
    )
    def test_matches_naive_reference_on_random_masks(
        self,
        n_true: int,
        n_detection: int,
        height: int,
        width: int,
        overlap_metric: OverlapMetric,
    ) -> None:
        """Random masks give the same matrix as the explicit (N, M, H, W) reference."""
        rng = np.random.default_rng(0)
        masks_true = rng.random((n_true, height, width)) > 0.5
        masks_detection = rng.random((n_detection, height, width)) > 0.5

        result = mask_iou_batch(masks_true, masks_detection, overlap_metric)

        expected = _naive_mask_iou(masks_true, masks_detection, overlap_metric)
        np.testing.assert_allclose(result, expected)

    def test_chunking_matches_single_pass(self) -> None:
        """A tiny memory_limit forces chunking yet returns the same matrix."""
        rng = np.random.default_rng(1)
        masks_true = rng.random((6, 16, 16)) > 0.5
        masks_detection = rng.random((4, 16, 16)) > 0.5

        single_pass = mask_iou_batch(masks_true, masks_detection)
        chunked = mask_iou_batch(masks_true, masks_detection, memory_limit=0)

        np.testing.assert_allclose(single_pass, chunked, rtol=1e-6)

        # empty masks_true under a tiny limit must not hit np.vstack([])
        empty = mask_iou_batch(
            np.zeros((0, 16, 16), dtype=bool), masks_detection, memory_limit=0
        )
        assert empty.shape == (0, 4)

    def test_mismatched_spatial_shape_raises(self) -> None:
        """Equal pixel counts but different (H, W) must raise, not silently mismatch."""
        masks_true = np.zeros((1, 4, 9), dtype=bool)
        masks_detection = np.zeros((1, 6, 6), dtype=bool)  # 36 px, different (H, W)
        with pytest.raises(ValueError, match="must share the same"):
            mask_iou_batch(masks_true, masks_detection)

    def test_compact_masks_mismatched_spatial_shape_raises(self) -> None:
        """Compact masks must reject different image canvases like dense masks."""
        from supervision.detection.compact_mask import CompactMask

        true_mask = np.zeros((1, 10, 10), dtype=bool)
        detection_mask = np.zeros((1, 20, 20), dtype=bool)
        true_compact = CompactMask.from_dense(
            true_mask, np.array([[0, 0, 9, 9]], dtype=np.float32), (10, 10)
        )
        detection_compact = CompactMask.from_dense(
            detection_mask, np.array([[0, 0, 19, 19]], dtype=np.float32), (20, 20)
        )

        with pytest.raises(ValueError, match="must share the same"):
            mask_iou_batch(true_compact, detection_compact)

    def test_compact_masks_match_dense_precision_at_nms_threshold(self) -> None:
        """Compact and dense NMS must make the same threshold decision."""
        from supervision.detection.compact_mask import CompactMask

        masks = np.zeros((2, 1, 10), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 0, :] = True
        boxes = np.array([[0, 0, 9, 0], [0, 0, 9, 0]], dtype=np.float32)
        compact_masks = CompactMask.from_dense(masks, boxes, image_shape=(1, 10))
        predictions = np.array(
            [[0, 0, 9, 0, 0.8, 0], [0, 0, 9, 0, 0.9, 0]], dtype=np.float32
        )

        dense_keep = _mask_non_max_suppression(predictions, masks, iou_threshold=0.1)
        compact_keep = _mask_non_max_suppression(
            predictions, compact_masks, iou_threshold=0.1
        )

        np.testing.assert_array_equal(compact_keep, dense_keep)

    def test_compact_masks_validate_overlap_metric_without_overlapping_boxes(
        self,
    ) -> None:
        """Invalid metrics must fail even when the bbox pre-filter finds no pairs."""
        from supervision.detection.compact_mask import CompactMask

        true_mask = np.zeros((1, 4, 4), dtype=bool)
        true_mask[0, 0, 0] = True
        detection_mask = np.zeros((1, 4, 4), dtype=bool)
        detection_mask[0, 3, 3] = True
        true_compact = CompactMask.from_dense(
            true_mask, np.array([[0, 0, 0, 0]], dtype=np.float32), image_shape=(4, 4)
        )
        detection_compact = CompactMask.from_dense(
            detection_mask,
            np.array([[3, 3, 3, 3]], dtype=np.float32),
            image_shape=(4, 4),
        )

        with pytest.raises(ValueError, match="overlap_metric"):
            mask_iou_batch(
                true_compact,
                detection_compact,
                overlap_metric="bad",  # type: ignore[arg-type]
            )

    def test_invalid_ndim_raises(self) -> None:
        """2D and 4D inputs must raise ValueError before reaching shape[1:] check."""
        mask_3d = np.zeros((1, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match="3D"):
            mask_iou_batch(np.zeros((10, 10), dtype=bool), mask_3d)
        with pytest.raises(ValueError, match="3D"):
            mask_iou_batch(mask_3d, np.zeros((10, 10), dtype=bool))
        with pytest.raises(ValueError, match="3D"):
            mask_iou_batch(np.zeros((1, 10, 10, 1), dtype=bool), mask_3d)

    def test_identical_disjoint_and_empty(self) -> None:
        """Identical/all-True → 1.0; all-False/disjoint → 0.0; empty keeps shape."""
        mask = np.zeros((1, 10, 10), dtype=bool)
        mask[0, 2:6, 2:6] = True
        far = np.zeros((1, 10, 10), dtype=bool)
        far[0, 7:9, 7:9] = True

        assert mask_iou_batch(mask, mask)[0, 0] == pytest.approx(1.0)
        assert mask_iou_batch(mask, far)[0, 0] == 0.0
        assert mask_iou_batch(np.zeros((0, 10, 10), dtype=bool), mask).shape == (0, 1)

        # all-True: denominator is non-zero, result must be exactly 1.0
        all_true = np.ones((2, 5, 5), dtype=bool)
        np.testing.assert_allclose(mask_iou_batch(all_true, all_true), np.ones((2, 2)))

        # all-False: denominator is zero; result must be 0.0 (not NaN)
        all_false = np.zeros((2, 5, 5), dtype=bool)
        np.testing.assert_array_equal(
            mask_iou_batch(all_false, all_false), np.zeros((2, 2))
        )

    def test_matmul_emits_no_runtime_warnings(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """mask_iou_batch on bool input must not emit any RuntimeWarning."""
        rng = np.random.default_rng(2)
        masks_true = rng.random((7, 32, 40)) > 0.5
        masks_detection = rng.random((5, 32, 40)) > 0.5
        mask_iou_batch(masks_true, masks_detection)
        runtime = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
        assert not runtime, (
            f"Unexpected RuntimeWarnings: {[str(w.message) for w in runtime]}"
        )

    def test_float64_promotion_for_large_float32_masks(self) -> None:
        """Float32 masks with pixel count > 2**24 must give exact IoU via float64."""
        # 1x(2**24+1) pixels: float32 area sum rounds, float64 area sum is exact.
        # Without CRIT-2 fix, IoU would exceed 1.0 due to dtype mismatch.
        mask = np.ones((1, 1, 2**24 + 1), dtype=np.float32)
        result = mask_iou_batch(mask, mask)
        assert result[0, 0] == pytest.approx(1.0)

    def test_compact_mask_mixed_input_gives_same_result(self) -> None:
        """CompactMask + dense mixed dispatch must match dense + dense."""
        from supervision.detection.compact_mask import CompactMask
        from supervision.detection.utils.converters import mask_to_xyxy

        rng = np.random.default_rng(3)
        masks = (rng.random((4, 20, 24)) > 0.3).astype(bool)
        xyxy = mask_to_xyxy(masks)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(20, 24))

        dense_vs_dense = mask_iou_batch(masks, masks)
        np.testing.assert_allclose(mask_iou_batch(cm, masks), dense_vs_dense, rtol=1e-5)
        np.testing.assert_allclose(mask_iou_batch(masks, cm), dense_vs_dense, rtol=1e-5)

    def test_detection_exceeds_limit_warns_and_completes(self) -> None:
        """Detection > memory_limit emits UserWarning; result shape is still correct."""
        rng = np.random.default_rng(4)
        masks_true = rng.random((3, 32, 32)) > 0.5
        # 500 x 32x32 float32 masks are ~2 MB > memory_limit=1 MB.
        masks_detection = (rng.random((500, 32, 32)) > 0.5).astype(np.float32)
        with pytest.warns(UserWarning, match="exceed"):
            result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)
        assert result.shape == (3, 500)


def _reference_jaccard_loop(
    boxes_true: list[list[float]],
    boxes_detection: list[list[float]],
    is_crowd: list[bool],
) -> np.ndarray:
    """Independent per-pair COCO Jaccard reference, shape (len(dt), len(gt))."""
    eps = np.spacing(1)
    out = np.zeros((len(boxes_detection), len(boxes_true)), dtype=np.float64)
    for gt_idx, (gx, gy, gw, gh) in enumerate(boxes_true):
        for dt_idx, (dx, dy, dw, dh) in enumerate(boxes_detection):
            inter_w = max(min(dx + dw, gx + gw) - max(dx, gx), 0.0)
            inter_h = max(min(dy + dh, gy + gh) - max(dy, gy), 0.0)
            area_inter = inter_w * inter_h
            area_det = max(dw, 0.0) * max(dh, 0.0)
            area_gt = max(gw, 0.0) * max(gh, 0.0)
            if is_crowd[gt_idx]:
                out[dt_idx, gt_idx] = area_inter / (area_det + eps)
            else:
                out[dt_idx, gt_idx] = area_inter / (
                    area_det + area_gt - area_inter + eps
                )
    return out


class TestBoxIouBatchWithJaccard:
    """Verify the vectorized COCO-style Jaccard IoU batch."""

    # ~1080 pairs: (1x1) + (3x5=15) + (20x50=1000) + (8x8=64) + (3x4=12)
    @pytest.mark.parametrize(
        ("n_gt", "n_dt", "seed"),
        [
            pytest.param(1, 1, 1, id="single-pair"),
            pytest.param(3, 5, 2, id="small-batch"),
            pytest.param(20, 50, 3, id="busy-batch"),
            pytest.param(8, 8, 4, id="degenerate-and-crowd"),
            pytest.param(3, 4, 5, id="all-crowd-multi-gt"),
        ],
    )
    def test_matches_per_pair_reference(self, n_gt: int, n_dt: int, seed: int) -> None:
        """Vectorized output equals an independent per-pair Jaccard loop."""
        rng = np.random.default_rng(seed)

        def _boxes(n: int) -> list[list[float]]:
            xy = rng.uniform(0, 100, (n, 2))
            wh = rng.uniform(0, 40, (n, 2))
            if seed == 4 and n:  # inject zero/negative-width degenerate boxes
                wh[rng.integers(0, n), 0] = rng.choice([0.0, -5.0])
            boxes: list[list[float]] = np.hstack([xy, wh]).tolist()
            return boxes

        boxes_true, boxes_detection = _boxes(n_gt), _boxes(n_dt)
        is_crowd = (rng.random(n_gt) < 0.3).tolist()

        result = box_iou_batch_with_jaccard(boxes_true, boxes_detection, is_crowd)
        expected = _reference_jaccard_loop(boxes_true, boxes_detection, is_crowd)

        assert result.shape == (n_dt, n_gt)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_crowd_uses_detection_area_as_union(self) -> None:
        """A crowd gt enclosing the detection scores 1.0 (union == detection area)."""
        boxes_true = [[0.0, 0.0, 100.0, 100.0]]
        boxes_detection = [[10.0, 10.0, 20.0, 20.0]]  # fully inside the gt
        crowd = box_iou_batch_with_jaccard(boxes_true, boxes_detection, [True])
        non_crowd = box_iou_batch_with_jaccard(boxes_true, boxes_detection, [False])
        assert crowd[0, 0] == pytest.approx(1.0)
        assert non_crowd[0, 0] == pytest.approx(0.04)  # 400 / 10000

    @pytest.mark.parametrize(
        ("boxes_true", "boxes_detection", "is_crowd", "expected_iou"),
        [
            pytest.param(
                [[0.0, 0.0, 10.0, 10.0]],
                [[20.0, 20.0, 10.0, 10.0]],
                [False],
                0.0,
                id="non-overlapping-non-crowd",
            ),
            pytest.param(
                [[0.0, 0.0, 10.0, 10.0]],
                [[0.0, 0.0, 10.0, 10.0]],
                [False],
                1.0,
                id="identical-non-crowd",
            ),
        ],
    )
    def test_non_crowd_uses_standard_iou(
        self,
        boxes_true: list[list[float]],
        boxes_detection: list[list[float]],
        is_crowd: list[bool],
        expected_iou: float,
    ) -> None:
        """Non-crowd GTs use standard IoU (intersection / union)."""
        result = box_iou_batch_with_jaccard(boxes_true, boxes_detection, is_crowd)

        assert result[0, 0] == pytest.approx(expected_iou)

    def test_all_crowd_multi_gt_matches_reference(self) -> None:
        """All-crowd 3-GT x 4-det batch matches per-pair reference loop."""
        rng = np.random.default_rng(5)
        n_gt, n_dt = 3, 4

        def _boxes(n: int) -> list[list[float]]:
            xy = rng.uniform(0, 100, (n, 2))
            wh = rng.uniform(1, 40, (n, 2))
            boxes: list[list[float]] = np.hstack([xy, wh]).tolist()
            return boxes

        boxes_true = _boxes(n_gt)
        boxes_detection = _boxes(n_dt)
        is_crowd = [True] * n_gt

        result = box_iou_batch_with_jaccard(boxes_true, boxes_detection, is_crowd)
        expected = _reference_jaccard_loop(boxes_true, boxes_detection, is_crowd)

        assert result.shape == (n_dt, n_gt)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize(
        ("n_gt", "n_dt"),
        [pytest.param(0, 3, id="empty-gt"), pytest.param(3, 0, id="empty-dt")],
    )
    def test_empty_input_returns_empty_array(self, n_gt: int, n_dt: int) -> None:
        """Empty gt or detections yields an empty array, preserving the contract."""
        boxes_true = [[0.0, 0.0, 10.0, 10.0]] * n_gt
        boxes_detection = [[0.0, 0.0, 10.0, 10.0]] * n_dt
        result = box_iou_batch_with_jaccard(boxes_true, boxes_detection, [False] * n_gt)
        assert result.size == 0

    def test_mismatched_is_crowd_length_raises(self) -> None:
        """`is_crowd` must align with `boxes_true`."""
        with pytest.raises(ValueError, match="`is_crowd` length"):
            box_iou_batch_with_jaccard(
                [[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]], []
            )


def _record_packed_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    """Wrap the packed IoU driver and record the mask counts of every call."""
    calls: list[tuple[int, int]] = []
    original = iou._mask_iou_batch_packed

    def recording(
        masks_true: np.ndarray, masks_detection: np.ndarray, *args: object
    ) -> np.ndarray:
        """Record one call, then run the real packed driver."""
        calls.append((len(masks_true), len(masks_detection)))
        return original(masks_true, masks_detection, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(iou, "_mask_iou_batch_packed", recording)
    return calls


def _naive_overlap(
    masks_true: np.ndarray,
    masks_detection: np.ndarray,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> np.ndarray:
    """Pairwise IoU or IoS of two mask stacks computed pixel by pixel in float64."""
    result = np.zeros((len(masks_true), len(masks_detection)))
    for true_index, mask_true in enumerate(masks_true.astype(bool)):
        for detection_index, mask_detection in enumerate(masks_detection.astype(bool)):
            inter = np.logical_and(mask_true, mask_detection).sum()
            if overlap_metric == OverlapMetric.IOU:
                denom = np.logical_or(mask_true, mask_detection).sum()
            else:
                denom = min(mask_true.sum(), mask_detection.sum())
            result[true_index, detection_index] = inter / denom if denom else 0.0
    return result


@pytest.mark.skipif(
    not hasattr(np, "bitwise_count"), reason="bit-packed path needs NumPy >= 2.0"
)
class TestBitPackedMaskOverlap:
    """Bit-packed dense mask overlap matches the float matmul path exactly."""

    @pytest.mark.parametrize(
        ("memory_limit", "expected_calls"),
        [
            pytest.param(1024 * 5, 0, id="float-copies-fit-keep-matmul"),
            pytest.param(0, 1, id="float-copies-do-not-fit-use-packed"),
        ],
    )
    def test_packed_path_runs_only_when_float_copies_exceed_limit(
        self,
        memory_limit: int,
        expected_calls: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bool masks take the packed path only when the float matmul would not fit."""
        rng = np.random.default_rng(1)
        masks = rng.random((4, 16, 16)) > 0.5
        calls = _record_packed_calls(monkeypatch)

        mask_iou_batch(masks, masks.copy(), memory_limit=memory_limit)

        assert len(calls) == expected_calls

    @pytest.mark.parametrize(
        ("dtype_true", "dtype_detection"),
        [
            pytest.param(bool, np.uint8, id="bool-uint8"),
            pytest.param(bool, np.float32, id="bool-float32"),
            pytest.param(np.uint8, np.uint8, id="uint8-uint8"),
            pytest.param(np.float32, bool, id="float32-bool"),
        ],
    )
    def test_non_bool_pairs_keep_the_matmul_path(
        self,
        dtype_true: type,
        dtype_detection: type,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only a bool-by-bool pair is packed; any other dtype pair uses the matmul."""
        rng = np.random.default_rng(2)
        masks_true: np.ndarray = (rng.random((5, 12, 14)) > 0.5).astype(dtype_true)
        masks_detection: np.ndarray = (rng.random((6, 12, 14)) > 0.5).astype(
            dtype_detection
        )
        calls = _record_packed_calls(monkeypatch)

        result = mask_iou_batch(masks_true, masks_detection, memory_limit=0)

        assert not calls
        np.testing.assert_allclose(
            result, _naive_overlap(masks_true, masks_detection), rtol=1e-6
        )

    def test_packed_detection_below_limit_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """A packed detection batch below the limit must not report a float floor."""
        masks_true = np.ones((3, 32, 32), dtype=bool)
        masks_detection = np.ones((500, 32, 32), dtype=bool)

        result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)

        assert not [
            warning for warning in recwarn if issubclass(warning.category, UserWarning)
        ]
        np.testing.assert_array_equal(result, np.ones((3, 500)))

    def test_packed_detection_above_limit_warns(self) -> None:
        """A packed detection batch exceeding the limit still reports its floor."""
        masks_true = np.zeros((1, 128, 128), dtype=bool)
        masks_detection = np.zeros((600, 128, 128), dtype=bool)

        with pytest.warns(UserWarning, match="exceed"):
            result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)

        np.testing.assert_array_equal(result, np.zeros((1, 600)))

    def test_floor_counts_pack_and_reduced_copies_of_the_detections(self) -> None:
        """The warning floor covers the detections' pack and reduced copies, not one."""
        masks_true = np.ones((1, 1024, 1024), dtype=bool)
        # 5 packed masks take 640 KiB, below the 1 MiB limit, but packing them and
        # keeping the reduced copy beside them needs about 1.3 MiB.
        masks_detection = np.ones((5, 1024, 1024), dtype=bool)

        with pytest.warns(UserWarning, match="exceed"):
            result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)

        np.testing.assert_array_equal(result, np.ones((1, 5)))

    def test_strided_detection_view_below_limit_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """A strided batch with view-compatible masks does not warn."""
        masks_true = np.ones((1, 1024, 1024), dtype=bool)
        masks_detection = np.ones((10, 1024, 1024), dtype=bool)[::2]

        result = mask_iou_batch(masks_true, masks_detection, memory_limit=4)

        assert not [
            warning for warning in recwarn if issubclass(warning.category, UserWarning)
        ]
        np.testing.assert_array_equal(result, np.ones((1, 5)))

    def test_noncontiguous_detection_copy_above_limit_warns(self) -> None:
        """A required detection reshape copy counts toward the warning floor."""
        masks_true = np.ones((1, 1024, 1024), dtype=bool)
        masks_detection = np.ones((5, 1024, 1024), dtype=bool).transpose(0, 2, 1)

        with pytest.warns(UserWarning, match="exceed"):
            result = mask_iou_batch(masks_true, masks_detection, memory_limit=4)

        np.testing.assert_array_equal(result, np.ones((1, 5)))

    def test_chunked_masks_pack_detections_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunking preserves IoU while packing the detection batch only once."""
        masks_true = np.ones((100, 32, 32), dtype=bool)
        masks_detection = np.ones((500, 32, 32), dtype=bool)
        original_pack = iou._pack_masks
        detection_pack_count = 0
        true_pack_count = 0

        def count_pack(masks: np.ndarray) -> np.ndarray:
            """Count packing of the original detection batch."""
            nonlocal detection_pack_count, true_pack_count
            if masks is masks_detection:
                detection_pack_count += 1
            else:
                true_pack_count += 1
            return original_pack(masks)

        monkeypatch.setattr(iou, "_pack_masks", count_pack)
        result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)

        assert detection_pack_count == 1
        assert true_pack_count > 1
        np.testing.assert_array_equal(result, np.ones((100, 500)))

    def test_self_overlap_packs_the_stack_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Comparing a stack with itself slices its packed words for every chunk."""
        rng = np.random.default_rng(9)
        masks = rng.random((6, 10, 12)) > 0.5
        original_pack = iou._pack_masks
        pack_count = 0

        def count_pack(masks_to_pack: np.ndarray) -> np.ndarray:
            """Count every packing call."""
            nonlocal pack_count
            pack_count += 1
            return original_pack(masks_to_pack)

        monkeypatch.setattr(iou, "_pack_masks", count_pack)
        result = mask_iou_batch(masks, masks, memory_limit=0)

        assert pack_count == 1
        np.testing.assert_array_equal(
            result, mask_iou_batch(masks, masks.copy(), memory_limit=1024 * 5)
        )

    @pytest.mark.parametrize(
        "overlap_metric",
        [
            pytest.param(OverlapMetric.IOU, id="iou"),
            pytest.param(OverlapMetric.IOS, id="ios"),
        ],
    )
    @pytest.mark.parametrize(
        ("n_true", "n_detection", "height", "width", "density"),
        [
            pytest.param(0, 3, 5, 7, 0.5, id="empty-true"),
            pytest.param(3, 0, 5, 7, 0.5, id="empty-detection"),
            pytest.param(4, 5, 5, 7, 0.0, id="all-false"),
            pytest.param(4, 5, 5, 7, 1.0, id="all-true"),
            pytest.param(6, 4, 3, 3, 0.5, id="pixels-not-multiple-of-8"),
            pytest.param(6, 4, 9, 9, 0.3, id="pixels-not-multiple-of-64"),
            pytest.param(8, 8, 32, 40, 0.1, id="sparse"),
        ],
    )
    def test_mask_iou_batch_matches_float_path(
        self,
        n_true: int,
        n_detection: int,
        height: int,
        width: int,
        density: float,
        overlap_metric: OverlapMetric,
    ) -> None:
        """Packed and matmul intersections give identical IoU and IoS matrices."""
        rng = np.random.default_rng(5)
        masks_true = rng.random((n_true, height, width)) < density
        masks_detection = rng.random((n_detection, height, width)) < density
        masks_detection[: min(n_true, n_detection)] = masks_true[:n_detection]

        result = mask_iou_batch(
            masks_true, masks_detection, overlap_metric, memory_limit=0
        )
        expected = mask_iou_batch(masks_true, masks_detection, overlap_metric)

        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)

    def test_chunks_with_a_remainder_match_the_float_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Several chunks of random masks and a short last chunk keep exact IoU."""
        rng = np.random.default_rng(10)
        masks_true = rng.random((700, 40, 50)) > 0.7
        masks_detection = rng.random((60, 40, 50)) > 0.7
        original = iou._packed_mask_iou_batch
        chunk_sizes: list[int] = []

        def recording(true_words: np.ndarray, *args: object) -> np.ndarray:
            """Record the number of true masks of every chunk."""
            chunk_sizes.append(len(true_words))
            return original(true_words, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(iou, "_packed_mask_iou_batch", recording)
        result = mask_iou_batch(masks_true, masks_detection, memory_limit=3)

        assert len(chunk_sizes) > 1
        assert 1 < chunk_sizes[-1] < chunk_sizes[0]
        np.testing.assert_array_equal(
            result, mask_iou_batch(masks_true, masks_detection)
        )

    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param("strided", id="strided-batch"),
            pytest.param("fortran", id="fortran-order"),
            pytest.param("transposed", id="transposed-spatial-axes"),
            pytest.param("read-only", id="read-only"),
        ],
    )
    def test_memory_layouts_of_random_masks_match_the_float_path(
        self, layout: str
    ) -> None:
        """Views and unusual memory layouts give the same overlaps as a plain copy."""
        rng = np.random.default_rng(11)
        dense = rng.random((12, 20, 30)) > 0.6
        views = {
            "strided": dense[::2],
            "fortran": np.asfortranarray(dense),
            "transposed": np.ascontiguousarray(dense.transpose(0, 2, 1)).transpose(
                0, 2, 1
            ),
            "read-only": dense,
        }
        masks_detection = views[layout]
        if layout == "read-only":
            masks_detection = dense.copy()
            masks_detection.flags.writeable = False
        masks_true = dense[3:9]

        result = mask_iou_batch(masks_true, masks_detection, memory_limit=0)

        expected = mask_iou_batch(masks_true, np.array(masks_detection, copy=True))
        np.testing.assert_array_equal(result, expected)

    def test_noncontiguous_true_masks_match_float_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Budget reshape copies while preserving packed overlap for true masks."""
        masks_true = np.ones((100, 128, 128), dtype=bool).transpose(0, 2, 1)
        masks_detection = np.ones((100, 128, 128), dtype=bool)
        original = iou._packed_mask_iou_batch
        chunk_sizes: list[int] = []

        def recording(true_words: np.ndarray, *args: object) -> np.ndarray:
            """Record true-mask chunk sizes before computing overlaps."""
            chunk_sizes.append(len(true_words))
            return original(true_words, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(iou, "_packed_mask_iou_batch", recording)
        result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)
        expected = mask_iou_batch(masks_true.copy(), masks_detection)

        assert chunk_sizes == [13] * 7 + [9]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "overlap_metric",
        [
            pytest.param(OverlapMetric.IOU, id="iou"),
            pytest.param(OverlapMetric.IOS, id="ios"),
        ],
    )
    @pytest.mark.parametrize("iou_threshold", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("seed", [6, 16, 26])
    def test_mask_non_max_merge_matches_fallback(
        self,
        seed: int,
        iou_threshold: float,
        overlap_metric: OverlapMetric,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Packed and original merging give identical groups, tied scores included."""
        rng = np.random.default_rng(seed)
        masks = rng.random((12, 9, 11)) < 0.5
        masks[6:] = masks[0]
        predictions = np.zeros((12, 6))
        predictions[:, 4] = rng.integers(0, 3, 12) / 3

        result = mask_non_max_merge(
            predictions, masks, iou_threshold, overlap_metric=overlap_metric
        )
        monkeypatch.setattr(iou, "_HAS_BITWISE_COUNT", False)
        expected = mask_non_max_merge(
            predictions, masks, iou_threshold, overlap_metric=overlap_metric
        )

        assert result == expected

    def test_mask_non_max_merge_grows_the_candidate_with_each_merge(self) -> None:
        """A mask that overlaps only the union of two merged masks joins their group."""
        masks = np.zeros((3, 1, 16), dtype=bool)
        masks[0, 0, 0:6] = True
        masks[1, 0, 4:10] = True
        masks[2, 0, [1, 8, 12, 13, 14, 15]] = True
        predictions = np.array(
            [[0, 0, 15, 0, 0.9, 0], [0, 0, 15, 0, 0.8, 0], [0, 0, 15, 0, 0.7, 0]]
        )

        result = mask_non_max_merge(
            predictions, masks, iou_threshold=0.3, overlap_metric=OverlapMetric.IOS
        )

        assert result == [[0, 1, 2]]

    def test_mask_non_max_merge_keeps_disjoint_masks_apart_at_zero_threshold(
        self,
    ) -> None:
        """Disjoint boolean masks never share a group, even at an overlap of zero."""
        masks = np.zeros((3, 4, 4), dtype=bool)
        masks[0, :2, :2] = True
        masks[1, 2:, 2:] = True
        masks[2, :2, :2] = True
        predictions = np.array(
            [[0, 0, 3, 3, 0.9, 0], [0, 0, 3, 3, 0.8, 0], [0, 0, 3, 3, 0.7, 0]]
        )

        result = mask_non_max_merge(predictions, masks, iou_threshold=0.0)

        assert result == [[0, 2], [1]]

    @pytest.mark.parametrize(
        "overlap_metric",
        [
            pytest.param(OverlapMetric.IOU, id="iou"),
            pytest.param(OverlapMetric.IOS, id="ios"),
        ],
    )
    def test_mask_non_max_suppression_matches_fallback(
        self,
        overlap_metric: OverlapMetric,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Packed and matmul overlaps give identical NMS keeps and Soft-NMS scores."""
        rng = np.random.default_rng(7)
        masks = rng.random((12, 9, 11)) < 0.5
        masks[6:] = masks[0]
        predictions = np.zeros((12, 6))
        predictions[:, 4] = rng.random(12)
        predictions[:, 5] = rng.integers(0, 2, 12)
        expected_keep = _mask_non_max_suppression(
            predictions, masks, 0.3, overlap_metric=overlap_metric
        )
        expected_scores = _mask_soft_non_max_suppression(
            predictions, masks, overlap_metric=overlap_metric
        )

        original = iou.mask_iou_batch

        def packed_overlap(
            masks_true: np.ndarray,
            masks_detection: np.ndarray,
            overlap_metric: OverlapMetric = OverlapMetric.IOU,
        ) -> np.ndarray:
            """Run the overlap with a limit that no float copy fits."""
            return original(masks_true, masks_detection, overlap_metric, memory_limit=0)

        monkeypatch.setattr(iou, "mask_iou_batch", packed_overlap)
        keep = _mask_non_max_suppression(
            predictions, masks, 0.3, overlap_metric=overlap_metric
        )
        scores = _mask_soft_non_max_suppression(
            predictions, masks, overlap_metric=overlap_metric
        )

        np.testing.assert_array_equal(keep, expected_keep)
        np.testing.assert_array_equal(scores, expected_scores)

    def test_mask_iou_batch_budgets_packed_masks(self) -> None:
        """Bool masks whose float copies exceed memory_limit run without a warning."""
        rng = np.random.default_rng(8)
        masks_true = rng.random((3, 32, 32)) > 0.5
        masks_detection = rng.random((500, 32, 32)) > 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = mask_iou_batch(masks_true, masks_detection, memory_limit=1)

        np.testing.assert_array_equal(
            result, mask_iou_batch(masks_true, masks_detection)
        )

    @pytest.mark.parametrize("memory_limit", [1024 * 5, 0])
    def test_mask_iou_batch_counts_exactly_above_2pow24_pixels(
        self, memory_limit: int
    ) -> None:
        """Masks over 2**24 pixels count in float64, so a one-pixel gap is kept."""
        masks_true = np.ones((1, 1, 2**24 + 8), dtype=bool)
        masks_detection = masks_true.copy()
        masks_detection[0, 0, 0] = False

        result = mask_iou_batch(masks_true, masks_detection, memory_limit=memory_limit)

        assert result[0, 0] == (2**24 + 7) / (2**24 + 8)


class TestOverlapFilterKeepAll:
    """`KEEP_ALL` replaces `NONE`, which stays an alias of the same member."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(OverlapFilter.KEEP_ALL, id="member"),
            pytest.param(OverlapFilter.NONE, id="legacy-member"),
            pytest.param("keep_all", id="new-string"),
            pytest.param("KEEP_ALL", id="new-string-uppercase"),
            pytest.param("none", id="legacy-string"),
            pytest.param("NONE", id="legacy-string-uppercase"),
        ],
    )
    def test_from_value_resolves_to_keep_all(self, value: OverlapFilter | str) -> None:
        """Every spelling of the no-filter strategy resolves to `KEEP_ALL`."""
        result = OverlapFilter.from_value(value)

        assert result is OverlapFilter.KEEP_ALL

    def test_list_names_each_strategy_once(self) -> None:
        """The alias adds no entry to the listed values."""
        result = OverlapFilter.list()

        assert result == ["keep_all", "non_max_suppression", "non_max_merge"]
