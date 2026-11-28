"""Verify the one-against-many overlap used by Weighted Box Fusion."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from supervision.detection.utils._fusion import (
    _box_overlap_one_to_many,
    _box_weighted_fusion_clusters,
)
from supervision.detection.utils.iou import OverlapMetric, _box_overlap_float64


def _random_boxes(count: int, seed: int) -> np.ndarray:
    """Random float64 `[x1, y1, x2, y2]` boxes with positive size."""
    rng = np.random.default_rng(seed)
    corners = rng.uniform(0, 500, (count, 2))
    sizes = rng.uniform(1, 150, (count, 2))
    return np.concatenate([corners, corners + sizes], axis=1)


class TestBoxOverlapOneToMany:
    """The fast path returns exactly what the general overlap function returns."""

    @pytest.mark.parametrize("overlap_metric", [OverlapMetric.IOU, OverlapMetric.IOS])
    @pytest.mark.parametrize("count", [1, 7, 120])
    def test_equals_general_function_bit_for_bit(
        self, overlap_metric: OverlapMetric, count: int
    ) -> None:
        """Random boxes give identical float64 overlaps, with no tolerance."""
        boxes = _random_boxes(count, seed=count)
        box = _random_boxes(1, seed=100 + count)[0]
        expected = _box_overlap_float64(boxes, box[np.newaxis, :], overlap_metric)[:, 0]

        result = _box_overlap_one_to_many(boxes, box, overlap_metric)

        assert np.array_equal(result, expected)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("overlap_metric", [OverlapMetric.IOU, OverlapMetric.IOS])
    @pytest.mark.parametrize(
        "box",
        [
            pytest.param((0.0, 0.0, 10.0, 10.0), id="identical-to-a-row"),
            pytest.param((10.0, 0.0, 20.0, 10.0), id="touching-edge"),
            pytest.param((100.0, 100.0, 110.0, 110.0), id="disjoint"),
            pytest.param((2.0, 2.0, 4.0, 4.0), id="contained"),
            pytest.param((5.0, 5.0, 5.0, 5.0), id="zero-area-point"),
            pytest.param((5.0, 0.0, 5.0, 10.0), id="zero-width-line"),
            pytest.param((-3.5, -3.5, 6.25, 6.25), id="negative-and-fractional"),
        ],
    )
    def test_equals_general_function_on_edge_cases(
        self, box: tuple[float, float, float, float], overlap_metric: OverlapMetric
    ) -> None:
        """Ties, touching edges, containment and zero-area boxes match exactly."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 5.0, 5.0],
                [5.0, 0.0, 5.0, 10.0],
                [1e9, 1e9, 1e9 + 3.0, 1e9 + 7.0],
            ]
        )
        candidate = np.array(box, dtype=np.float64)
        expected = _box_overlap_float64(
            boxes, candidate[np.newaxis, :], overlap_metric
        )[:, 0]

        result = _box_overlap_one_to_many(boxes, candidate, overlap_metric)

        assert np.array_equal(result, expected)

    def test_zero_area_pair_has_zero_overlap(self) -> None:
        """A normalising area of zero gives overlap zero, not NaN."""
        boxes = np.array([[5.0, 5.0, 5.0, 5.0]])
        box = np.array([5.0, 5.0, 5.0, 5.0])

        result = _box_overlap_one_to_many(boxes, box, OverlapMetric.IOU)

        assert result.tolist() == [0.0]

    def test_empty_boxes_give_an_empty_result(self) -> None:
        """No boxes to compare with gives an empty overlap array."""
        boxes = np.empty((0, 4), dtype=np.float64)
        box = np.array([0.0, 0.0, 1.0, 1.0])

        result = _box_overlap_one_to_many(boxes, box, OverlapMetric.IOU)

        assert result.shape == (0,)

    def test_does_not_modify_its_inputs(self) -> None:
        """The in-place arithmetic works on temporaries, never on the caller's boxes."""
        boxes = _random_boxes(10, seed=3)
        box = _random_boxes(1, seed=4)[0]
        boxes_before, box_before = boxes.copy(), box.copy()

        _box_overlap_one_to_many(boxes, box, OverlapMetric.IOU)

        assert np.array_equal(boxes, boxes_before)
        assert np.array_equal(box, box_before)

    def test_rejects_an_unsupported_metric(self) -> None:
        """Any metric other than IOU or IOS raises, as the general function does."""
        boxes = _random_boxes(2, seed=5)
        box = _random_boxes(1, seed=6)[0]

        with pytest.raises(ValueError, match="not supported"):
            _box_overlap_one_to_many(boxes, box, "giou")  # type: ignore[arg-type]


def _exact_mean(values: list[float]) -> float:
    """Mean of floats computed in exact arithmetic, rounded once."""
    return float(sum(Fraction(value) for value in values) / len(values))


class TestExactArithmeticPath:
    """Coordinates near the float64 limit take the exact path and stay finite."""

    @staticmethod
    def _huge_pair() -> np.ndarray:
        """Two overlapping boxes near 6e307 whose areas are still finite."""
        base = 6e307
        step = float(np.spacing(base))
        return np.array(
            [
                [base, 0.0, base + 8 * step, 100.0],
                [base + step, 0.0, base + 9 * step, 100.0],
            ]
        )

    def test_uniform_weights_give_the_exact_mean_box(self) -> None:
        """Uniform members average with equal weight, without overflowing."""
        boxes = self._huge_pair()

        fused, _, labels = _box_weighted_fusion_clusters(
            boxes,
            np.array([0.9, 0.8]),
            None,
            0.5,
            OverlapMetric.IOU,
            box_weights="uniform",
        )

        assert labels.tolist() == [0, 0]
        expected = [_exact_mean(boxes[:, column].tolist()) for column in range(4)]
        assert fused[0].tolist() == expected

    def test_an_all_zero_cluster_averages_its_boxes_equally(self) -> None:
        """Confidence weights of zero fall back to the plain exact mean."""
        boxes = self._huge_pair()

        fused, scores, labels = _box_weighted_fusion_clusters(
            boxes, np.array([0.0, 0.0]), None, 0.5, OverlapMetric.IOU
        )

        assert labels.tolist() == [0, 0]
        assert scores.tolist() == [0.0]
        expected = [_exact_mean(boxes[:, column].tolist()) for column in range(4)]
        assert fused[0].tolist() == expected


class TestSourceWeightsArguments:
    """The private clustering kernels check source weights as the public methods do."""

    def test_source_weights_without_num_models_raises(self) -> None:
        """One weight per source needs the number of sources."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])

        with pytest.raises(ValueError, match="requires num_models"):
            _box_weighted_fusion_clusters(
                boxes,
                np.array([0.9, 0.8]),
                None,
                0.5,
                OverlapMetric.IOU,
                sources=np.array([0, 1]),
                source_weights=[1.0, 2.0],
            )
