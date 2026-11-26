import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    import matplotlib.pyplot  # old releases warn here on import

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

matplotlib.use("Agg")

import supervision as sv  # noqa: E402
from tests.helpers import _create_key_points  # noqa: E402


@pytest.fixture
def scene() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_key_points() -> sv.KeyPoints:
    return _create_key_points(
        xy=[
            [
                [10, 10],
                [20, 20],
                [30, 30],
                [40, 40],
                [50, 50],
                [60, 60],
                [70, 70],
                [80, 80],
                [90, 90],
                [10, 20],
                [20, 30],
                [30, 40],
                [40, 50],
                [50, 60],
                [60, 70],
                [70, 80],
                [80, 90],
            ],
            [
                [10, 40],
                [20, 50],
                [30, 60],
                [40, 70],
                [50, 80],
                [60, 90],
                [70, 10],
                [80, 20],
                [90, 30],
                [10, 50],
                [20, 60],
                [30, 70],
                [40, 80],
                [50, 90],
                [60, 10],
                [70, 20],
                [80, 30],
            ],
        ],
        confidence=[
            [0.8] * 17,
            [0.6] * 17,
        ],
        class_id=[0, 1],
    )


@pytest.fixture
def empty_key_points() -> sv.KeyPoints:
    return sv.KeyPoints.empty()


def _is_metrics_path(path: Path) -> bool:
    """Tell whether a path belongs to the `supervision.metrics` code or its tests."""
    posix_path = f"{path.as_posix()}/"
    return "/tests/metrics/" in posix_path or "/supervision/metrics/" in posix_path


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip importing metrics modules when `-m` deselects them.

    A deselected module is still imported during collection, and the reference packages
    some metrics tests import at module level may not import in a minimal environment.
    """
    if "not metrics" in (config.option.markexpr or "") and _is_metrics_path(
        collection_path
    ):
        return True
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test and doctest of `supervision.metrics` with `metrics`.

    The metrics package needs the optional `metrics` extra and reference packages that
    some CI jobs do not install, so they deselect it with `-m "not metrics"`.
    """
    for item in items:
        if _is_metrics_path(Path(str(item.path))):
            item.add_marker(pytest.mark.metrics)
