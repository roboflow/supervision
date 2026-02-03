import numpy as np
import pytest

import supervision as sv
from tests.helpers import _create_key_points


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
