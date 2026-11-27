from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    _box_non_max_suppression,
    _box_soft_non_max_suppression,
    _box_weighted_box_fusion,
    _mask_non_max_suppression,
    _mask_soft_non_max_suppression,
    _oriented_box_non_max_suppression,
    _oriented_box_soft_non_max_suppression,
    box_non_max_suppression,
    box_soft_non_max_suppression,
    box_weighted_box_fusion,
    mask_non_max_suppression,
    mask_soft_non_max_suppression,
    oriented_box_non_max_suppression,
    oriented_box_soft_non_max_suppression,
)

#: A large box and a small box nested in its corner: IoU is 0.16 and IoS is 1.0, so
#: a low threshold or the IoS metric changes what survives.
_PREDICTIONS = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 4, 4, 0.8, 0]], dtype=float)
_MASKS = np.zeros((2, 10, 10), dtype=bool)
_MASKS[0, :10, :10] = True
_MASKS[1, :4, :4] = True
_CORNERS = np.array(
    [
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        [[0, 0], [4, 0], [4, 4], [0, 4]],
    ],
    dtype=float,
)

_NMS_CASES = [
    pytest.param(
        box_non_max_suppression,
        _box_non_max_suppression,
        (_PREDICTIONS,),
        (0.1, OverlapMetric.IOS),
        id="box-nms",
    ),
    pytest.param(
        mask_non_max_suppression,
        _mask_non_max_suppression,
        (_PREDICTIONS, _MASKS),
        (0.1, OverlapMetric.IOS),
        id="mask-nms",
    ),
    pytest.param(
        oriented_box_non_max_suppression,
        _oriented_box_non_max_suppression,
        (_PREDICTIONS, _CORNERS),
        (0.1, OverlapMetric.IOS),
        id="oriented-nms",
    ),
    pytest.param(
        box_soft_non_max_suppression,
        _box_soft_non_max_suppression,
        (_PREDICTIONS,),
        (0.1, OverlapMetric.IOS),
        id="box-soft-nms",
    ),
    pytest.param(
        mask_soft_non_max_suppression,
        _mask_soft_non_max_suppression,
        (_PREDICTIONS, _MASKS),
        (0.1, OverlapMetric.IOS),
        id="mask-soft-nms",
    ),
    pytest.param(
        oriented_box_soft_non_max_suppression,
        _oriented_box_soft_non_max_suppression,
        (_PREDICTIONS, _CORNERS),
        (0.1, OverlapMetric.IOS),
        id="oriented-soft-nms",
    ),
]

_THRESHOLD_NAMES = {"nms": "iou_threshold", "soft": "sigma"}


def _threshold_name(public: Callable[..., Any]) -> str:
    """Return the name of the public kernel's second tunable argument."""
    return _THRESHOLD_NAMES["soft" if "soft" in public.__name__ else "nms"]


@pytest.mark.filterwarnings("ignore::supervision.utils.internal.SupervisionWarnings")
class TestDeprecatedKernelsDelegate:
    """Deprecated public kernels forward every argument to their private kernel."""

    @pytest.mark.parametrize(("public", "private", "required", "tuning"), _NMS_CASES)
    def test_positional_arguments_reach_the_kernel(
        self,
        public: Callable[..., Any],
        private: Callable[..., Any],
        required: tuple[Any, ...],
        tuning: tuple[Any, ...],
    ) -> None:
        """Non-default positional values give the private kernel's result."""
        expected = private(*required, *tuning)

        result = public(*required, *tuning)

        assert np.array_equal(result, expected)
        assert not np.array_equal(result, private(*required))

    @pytest.mark.parametrize(("public", "private", "required", "tuning"), _NMS_CASES)
    def test_keyword_arguments_reach_the_kernel(
        self,
        public: Callable[..., Any],
        private: Callable[..., Any],
        required: tuple[Any, ...],
        tuning: tuple[Any, ...],
    ) -> None:
        """Non-default keyword values give the private kernel's result."""
        threshold, metric = tuning
        keywords = {_threshold_name(public): threshold, "overlap_metric": metric}
        expected = private(*required, **keywords)

        result = public(*required, **keywords)

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        ("public", "private", "required"),
        [
            pytest.param(
                mask_non_max_suppression,
                _mask_non_max_suppression,
                (_PREDICTIONS, _MASKS),
                id="mask-nms",
            ),
            pytest.param(
                mask_soft_non_max_suppression,
                _mask_soft_non_max_suppression,
                (_PREDICTIONS, _MASKS),
                id="mask-soft-nms",
            ),
        ],
    )
    def test_mask_kernels_still_accept_mask_dimension(
        self,
        public: Callable[..., Any],
        private: Callable[..., Any],
        required: tuple[Any, ...],
    ) -> None:
        """The unused `mask_dimension` is accepted positionally and by keyword."""
        expected = private(*required)

        by_position = public(*required, 0.5, OverlapMetric.IOU, 32)
        by_keyword = public(*required, mask_dimension=32)

        assert np.array_equal(by_position, expected)
        assert np.array_equal(by_keyword, expected)

    def test_weighted_box_fusion_forwards_every_argument(self) -> None:
        """`iou_threshold`, `overlap_metric` and `num_models` reach the kernel."""
        predictions = np.array(
            [[0, 0, 10, 10, 0.9, 0], [0, 0, 4, 4, 0.8, 0], [50, 50, 60, 60, 0.7, 0]],
            dtype=float,
        )

        by_position = box_weighted_box_fusion(predictions, 0.1, OverlapMetric.IOS, 2)
        by_keyword = box_weighted_box_fusion(
            predictions,
            iou_threshold=0.1,
            overlap_metric=OverlapMetric.IOS,
            num_models=2,
        )

        expected = _box_weighted_box_fusion(predictions, 0.1, OverlapMetric.IOS, 2)
        assert np.array_equal(by_position, expected)
        assert np.array_equal(by_keyword, expected)
        assert not np.array_equal(expected, _box_weighted_box_fusion(predictions))
