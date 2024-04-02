import numpy as np
import pytest

from supervision.geometry.core import Point
from supervision.geometry.utils import get_polygon_center


@pytest.mark.parametrize(
    "polygon, expected_result",
    [
        (np.array([[0, 0], [0, 2], [2, 2], [2, 0]]), Point(x=1, y=1)),
        (np.array([[0, 0], [3, 4], [6, 0]]), Point(x=3, y=1)),
        (np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [5, 2]]), Point(x=2, y=2)),
        (
            np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [4, 4], [4, 0]]),
            Point(x=2, y=2),
        ),
        (np.array([[0, 2], [2, 4], [4, 2], [2, 0]]), Point(x=2, y=2)),
        (
            np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 1000]]),
            Point(x=0, y=500),
        ),
        (np.array([[0, 0], [13, 200], [0, 150]]), Point(x=4, y=100)),
        (
            np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 0]]),
            Point(x=2, y=1),
        ),
    ],
)
def test_get_polygon_center(polygon: np.ndarray, expected_result: Point) -> None:
    result = get_polygon_center(polygon)
    assert result == expected_result
