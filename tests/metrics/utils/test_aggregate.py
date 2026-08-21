"""Tests for supervision.metrics.utils.aggregate — result aggregation helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from supervision.metrics.core import AveragingMethod, MetricResult, MetricTarget
from supervision.metrics.f1_score import F1ScoreResult
from supervision.metrics.mean_average_precision import MeanAveragePrecisionResult
from supervision.metrics.precision import PrecisionResult
from supervision.metrics.utils.aggregate import (
    aggregate_metric_results,
    plot_aggregate_metric_results,
)


def _make_f1_result(f1_50: float = 0.8, f1_75: float = 0.6) -> F1ScoreResult:
    """Build a minimal F1ScoreResult for testing."""
    scores = np.array([f1_50, 0.0, 0.0, 0.0, 0.0, f1_75, 0.0, 0.0, 0.0, 0.0])
    return F1ScoreResult(
        metric_target=MetricTarget.BOXES,
        averaging_method=AveragingMethod.WEIGHTED,
        f1_scores=scores,
        f1_per_class=np.zeros((1, 10)),
        iou_thresholds=np.linspace(0.5, 0.95, 10, dtype=np.float32),
        matched_classes=np.array([0], dtype=np.int32),
        small_objects=None,
        medium_objects=None,
        large_objects=None,
    )


def _make_precision_result(p50: float = 0.9, p75: float = 0.7) -> PrecisionResult:
    """Build a minimal PrecisionResult for testing."""
    scores = np.array([p50, 0.0, 0.0, 0.0, 0.0, p75, 0.0, 0.0, 0.0, 0.0])
    return PrecisionResult(
        metric_target=MetricTarget.BOXES,
        averaging_method=AveragingMethod.WEIGHTED,
        precision_scores=scores,
        precision_per_class=np.zeros((1, 10)),
        iou_thresholds=np.linspace(0.5, 0.95, 10, dtype=np.float32),
        matched_classes=np.array([0], dtype=np.int32),
        small_objects=None,
        medium_objects=None,
        large_objects=None,
    )


def _make_map_result(
    map50: float = 0.85, map75: float = 0.65
) -> MeanAveragePrecisionResult:
    """Build a minimal MeanAveragePrecisionResult for testing."""
    scores = np.array([map50, 0.0, 0.0, 0.0, 0.0, map75, 0.0, 0.0, 0.0, 0.0])
    return MeanAveragePrecisionResult(
        metric_target=MetricTarget.BOXES,
        is_class_agnostic=False,
        mAP_scores=scores,
        ap_per_class=np.zeros((1, 10)),
        iou_thresholds=np.linspace(0.5, 0.95, 10, dtype=np.float64),
        matched_classes=np.array([0], dtype=np.int32),
        small_objects=None,
        medium_objects=None,
        large_objects=None,
    )


class TestAggregateMetricResults:
    """Tests for the aggregate_metric_results function."""

    def test_empty_list_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            aggregate_metric_results([])

    def test_mixed_types_raises(self) -> None:
        """Mixing different result types raises TypeError."""
        f1 = _make_f1_result()
        precision = _make_precision_result()
        with pytest.raises(TypeError, match="same type"):
            aggregate_metric_results([f1, precision])

    def test_model_names_length_mismatch_raises(self) -> None:
        """model_names with wrong length raises ValueError."""
        f1 = _make_f1_result()
        with pytest.raises(ValueError, match="model_names length"):
            aggregate_metric_results([f1], model_names=["a", "b"])

    def test_basic_aggregation(self) -> None:
        """Two F1 results produce a DataFrame with two rows."""
        r1 = _make_f1_result(f1_50=0.8, f1_75=0.6)
        r2 = _make_f1_result(f1_50=0.9, f1_75=0.7)
        df = aggregate_metric_results([r1, r2])
        assert len(df) == 2
        assert "F1@50" in df.columns
        assert "F1@75" in df.columns

    def test_model_names_as_index(self) -> None:
        """model_names become the DataFrame index."""
        r1 = _make_f1_result()
        r2 = _make_f1_result()
        df = aggregate_metric_results([r1, r2], model_names=["YOLO", "DETR"])
        assert list(df.index) == ["YOLO", "DETR"]

    def test_exclude_object_sizes(self) -> None:
        """Object-size columns are dropped when include_object_sizes=False."""
        small = _make_f1_result(f1_50=0.5, f1_75=0.3)
        r1 = F1ScoreResult(
            metric_target=MetricTarget.BOXES,
            averaging_method=AveragingMethod.WEIGHTED,
            f1_scores=np.array([0.8, 0, 0, 0, 0, 0.6, 0, 0, 0, 0]),
            f1_per_class=np.zeros((1, 10)),
            iou_thresholds=np.linspace(0.5, 0.95, 10, dtype=np.float32),
            matched_classes=np.array([0], dtype=np.int32),
            small_objects=small,
            medium_objects=None,
            large_objects=None,
        )
        df_no_sizes = aggregate_metric_results([r1], include_object_sizes=False)
        size_cols = [
            c
            for c in df_no_sizes.columns
            if c.startswith(("small_objects_", "medium_objects_", "large_objects_"))
        ]
        assert size_cols == []

    def test_include_object_sizes(self) -> None:
        """Object-size columns are kept when include_object_sizes=True."""
        small = _make_f1_result(f1_50=0.5, f1_75=0.3)
        r1 = F1ScoreResult(
            metric_target=MetricTarget.BOXES,
            averaging_method=AveragingMethod.WEIGHTED,
            f1_scores=np.array([0.8, 0, 0, 0, 0, 0.6, 0, 0, 0, 0]),
            f1_per_class=np.zeros((1, 10)),
            iou_thresholds=np.linspace(0.5, 0.95, 10, dtype=np.float32),
            matched_classes=np.array([0], dtype=np.int32),
            small_objects=small,
            medium_objects=None,
            large_objects=None,
        )
        df = aggregate_metric_results([r1], include_object_sizes=True)
        size_cols = [c for c in df.columns if c.startswith("small_objects_")]
        assert len(size_cols) > 0

    def test_map_results_aggregation(self) -> None:
        """MeanAveragePrecisionResult results can be aggregated."""
        r1 = _make_map_result(map50=0.85, map75=0.65)
        r2 = _make_map_result(map50=0.90, map75=0.70)
        df = aggregate_metric_results([r1, r2])
        assert len(df) == 2
        assert "mAP@50" in df.columns


class TestPlotAggregateMetricResults:
    """Tests for the plot_aggregate_metric_results function."""

    def test_empty_list_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_aggregate_metric_results([])

    def test_mixed_types_raises(self) -> None:
        """Mixing different result types raises TypeError."""
        f1 = _make_f1_result()
        precision = _make_precision_result()
        with pytest.raises(TypeError, match="same type"):
            plot_aggregate_metric_results([f1, precision])

    def test_model_names_length_mismatch_raises(self) -> None:
        """model_names with wrong length raises ValueError."""
        f1 = _make_f1_result()
        with pytest.raises(ValueError, match="model_names length"):
            plot_aggregate_metric_results([f1], model_names=["a", "b"])

    @patch("matplotlib.pyplot.show")
    def test_plot_is_called(self, mock_show: object) -> None:
        """Plotting runs without error and calls plt.show()."""
        r1 = _make_f1_result(f1_50=0.8, f1_75=0.6)
        r2 = _make_f1_result(f1_50=0.9, f1_75=0.7)
        plot_aggregate_metric_results([r1, r2], model_names=["YOLO", "DETR"])


class TestGetPlotDetails:
    """Tests for _get_plot_details across result types."""

    def test_f1_without_object_sizes(self) -> None:
        """F1 plot details without object sizes has only 2 labels."""
        r = _make_f1_result()
        details = r._get_plot_details(include_object_sizes=False)
        assert details.labels == ["F1@50", "F1@75"]
        assert len(details.values) == 2
        assert len(details.colors) == 2

    def test_f1_with_object_sizes(self) -> None:
        """F1 plot details with object sizes includes size category bars."""
        small = _make_f1_result(f1_50=0.5, f1_75=0.3)
        r = F1ScoreResult(
            metric_target=MetricTarget.BOXES,
            averaging_method=AveragingMethod.WEIGHTED,
            f1_scores=np.array([0.8, 0, 0, 0, 0, 0.6, 0, 0, 0, 0]),
            f1_per_class=np.zeros((1, 10)),
            iou_thresholds=np.linspace(0.5, 0.95, 10, dtype=np.float32),
            matched_classes=np.array([0], dtype=np.int32),
            small_objects=small,
            medium_objects=None,
            large_objects=None,
        )
        details = r._get_plot_details(include_object_sizes=True)
        assert len(details.labels) == 4
        assert "Small: F1@50" in details.labels

    def test_map_plot_details(self) -> None:
        """MeanAveragePrecisionResult returns 3 labels without sizes."""
        r = _make_map_result()
        details = r._get_plot_details(include_object_sizes=False)
        assert details.labels == ["mAP@50:95", "mAP@50", "mAP@75"]

    def test_metric_result_is_abstract(self) -> None:
        """MetricResult cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricResult()  # type: ignore[abstract]
