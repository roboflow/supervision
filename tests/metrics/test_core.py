"""Tests for supervision.metrics.core — Metric ABC and shared enums."""

from __future__ import annotations

import pytest

from supervision.metrics.core import AveragingMethod, Metric, MetricTarget


class TestMetricAbc:
    """Verify the abstract base class cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_metric(self) -> None:
        """Metric is abstract and raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_abstract_methods(self) -> None:
        """A subclass missing an abstract method still cannot be instantiated."""

        class _PartialMetric(Metric):
            """Concrete class missing `reset` and `compute`."""

            def update(self, *args: object, **kwargs: object) -> Metric:
                """Return self without implementing the rest of the contract."""
                return self

        with pytest.raises(TypeError):
            _PartialMetric()  # type: ignore[abstract]

    def test_fully_implemented_subclass_is_instantiable(self) -> None:
        """A subclass implementing update/reset/compute can be instantiated."""

        class _FullMetric(Metric):
            """Minimal concrete metric implementing the full contract."""

            def update(self, *args: object, **kwargs: object) -> Metric:
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
