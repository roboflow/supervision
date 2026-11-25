"""Tests for supervision.metrics.core — Metric ABC and shared enums."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from matplotlib import pyplot as plt

from supervision.detection.core import Detections
from supervision.metrics import (
    F1Score,
    MeanAveragePrecision,
    MeanAverageRecall,
    Precision,
    Recall,
)
from supervision.metrics.core import AveragingMethod, Metric, MetricTarget


class TestMetricAbc:
    """Verify the abstract base class cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_metric(self) -> None:
        """Metric is abstract and raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_abstract_methods(self) -> None:
        """A subclass missing an abstract method still cannot be instantiated."""

        class _PartialMetric(Metric[int]):
            """Concrete class missing `reset` and `compute`."""

            def update(self, *args: object, **kwargs: object) -> Metric[int]:
                """Return self without implementing the rest of the contract."""
                return self

        with pytest.raises(TypeError):
            _PartialMetric()  # type: ignore[abstract]

    def test_fully_implemented_subclass_is_instantiable(self) -> None:
        """A subclass implementing update/reset/compute can be instantiated."""

        class _FullMetric(Metric[int]):
            """Minimal concrete metric implementing the full contract."""

            def update(self, *args: object, **kwargs: object) -> Metric[int]:
                """Return self, allowing method chaining."""
                return self

            def reset(self) -> None:
                """No internal state to reset."""

            def compute(self, *args: object, **kwargs: object) -> int:
                """Return a constant result."""
                return 0

        metric = _FullMetric()
        assert metric.update() is metric
        assert metric.compute() == 0


class TestMetricTarget:
    """Verify the MetricTarget enum members and values."""

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            pytest.param(MetricTarget.BOXES, "boxes", id="boxes"),
            pytest.param(MetricTarget.MASKS, "masks", id="masks"),
            pytest.param(
                MetricTarget.ORIENTED_BOUNDING_BOXES, "obb", id="oriented-boxes"
            ),
        ],
    )
    def test_member_value(self, member: MetricTarget, value: str) -> None:
        """Each MetricTarget member has the documented string value."""
        assert member.value == value


class TestAveragingMethod:
    """Verify the AveragingMethod enum members and values."""

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            pytest.param(AveragingMethod.MACRO, "macro", id="macro"),
            pytest.param(AveragingMethod.MICRO, "micro", id="micro"),
            pytest.param(AveragingMethod.WEIGHTED, "weighted", id="weighted"),
        ],
    )
    def test_member_value(self, member: AveragingMethod, value: str) -> None:
        """Each AveragingMethod member has the documented string value."""
        assert member.value == value


class TestMetricResultPlot:
    """Verify that plotting a metric result leaves global Matplotlib state alone."""

    @pytest.mark.parametrize(
        "metric",
        [Precision, Recall, F1Score, MeanAveragePrecision, MeanAverageRecall],
    )
    def test_keeps_user_font_family(
        self, metric: type[Metric[Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plot leaves the font family the user configured in rcParams."""
        monkeypatch.setattr(plt, "show", lambda: None)
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )
        result = metric().update(detections, detections).compute()

        with plt.rc_context({"font.family": "serif"}):
            result.plot()
            font_family = plt.rcParams["font.family"]
        plt.close("all")

        assert font_family == ["serif"]
