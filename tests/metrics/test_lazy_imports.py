"""Regression tests for lazy metric plotting imports."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

from supervision.detection.core import Detections


def _clear_metric_modules() -> None:
    """Remove cached metric modules so import-time side effects stay visible."""
    for module_name in [
        "supervision.metrics",
        "supervision.metrics.detection",
        "supervision.metrics.f1_score",
        "supervision.metrics.mean_average_precision",
        "supervision.metrics.mean_average_recall",
        "supervision.metrics.precision",
        "supervision.metrics.recall",
    ]:
        sys.modules.pop(module_name, None)


def _make_pyplot_stub() -> ModuleType:
    """Build a minimal pyplot stub for plot smoke tests."""
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.rcParams = {}

    figure = MagicMock(name="figure")
    axis = MagicMock(name="axis")
    bar = MagicMock(name="bar")
    bar.get_height.return_value = 1.0
    bar.get_x.return_value = 0.0
    bar.get_width.return_value = 1.0
    axis.bar.return_value = [bar]

    pyplot.subplots = MagicMock(return_value=(figure, axis))
    pyplot.tight_layout = MagicMock()
    pyplot.show = MagicMock()
    pyplot.setp = MagicMock()

    return pyplot


def test_metrics_package_import_keeps_pyplot_lazy() -> None:
    """Importing supervision.metrics must not pull in matplotlib.pyplot."""
    sys.modules.pop("matplotlib.pyplot", None)
    _clear_metric_modules()

    importlib.import_module("supervision.metrics")

    assert "matplotlib.pyplot" not in sys.modules


def test_precision_plot_imports_pyplot_on_demand(monkeypatch) -> None:
    """Precision.plot should import pyplot only when plotting is requested."""
    sys.modules.pop("matplotlib.pyplot", None)
    _clear_metric_modules()

    precision_module = importlib.import_module("supervision.metrics.precision")
    assert "matplotlib.pyplot" not in sys.modules

    pyplot = _make_pyplot_stub()
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    import matplotlib

    monkeypatch.setattr(matplotlib, "pyplot", pyplot, raising=False)

    predictions = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    targets = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        class_id=np.array([0]),
    )

    result = precision_module.Precision().update(predictions, targets).compute()
    result.plot()

    pyplot.subplots.assert_called_once()
    pyplot.show.assert_called_once()
