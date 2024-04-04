import numpy as np
import pytest

from supervision.geometry.core import Point
from supervision.geometry.utils import get_polygon_center


def generate_test_polygon(n: int) -> np.ndarray:
    """
     Generate a semicircle with a given number of points.

     Parameters:
         n (int): amount of points in polygon

     Returns:
         Polygon: test polygon in the form of a semicircle.

    Examples:
         ```python
         from supervision.geometry.utils import get_polygon_center
         import numpy as np

         test_polygon = generate_test_data(1000)

         get_polygon_center(test_polygon)
         Point(x=500, y=1212)
         ```
    """
    r: int = n // 2
    x_axis = np.linspace(0, 2 * r, n)
    y_axis = (r ** 2 - (x_axis - r) ** 2) ** 0.5 + 2 * r
    polygon = np.array([x_axis, y_axis]).T

    return polygon


@pytest.mark.parametrize(
    "polygon, expected_result",
    [
        (generate_test_polygon(10), Point(x=4, y=12)),
        (generate_test_polygon(50), Point(x=24, y=60)),
        (generate_test_polygon(100), Point(x=49, y=121)),
        (generate_test_polygon(1000), Point(x=500, y=1212)),
        (generate_test_polygon(3000), Point(x=1499, y=3636)),
        (generate_test_polygon(10000), Point(x=5000, y=12122)),
        (generate_test_polygon(20000), Point(x=10000, y=24244)),
        (generate_test_polygon(50000), Point(x=24999, y=60610)),
    ],
)
def test_get_polygon_center(polygon: np.ndarray, expected_result: Point) -> None:
    result = get_polygon_center(polygon)
    assert result == expected_result
