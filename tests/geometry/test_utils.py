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
    y_axis = (r**2 - (x_axis - r) ** 2) ** 0.5 + 2 * r
    polygon = np.array([x_axis, y_axis]).T

    return polygon


@pytest.mark.parametrize(
    ("polygon", "expected_result"),
    [
        (generate_test_polygon(10), Point(x=5.0, y=12.0)),
        (generate_test_polygon(50), Point(x=25.0, y=61.0)),
        (generate_test_polygon(100), Point(x=50.0, y=121.0)),
        (generate_test_polygon(1000), Point(x=500.0, y=1212.0)),
        (generate_test_polygon(3000), Point(x=1500.0, y=3637.0)),
        (generate_test_polygon(10000), Point(x=5000.0, y=12122.0)),
        (generate_test_polygon(20000), Point(x=10000.0, y=24244.0)),
        (generate_test_polygon(50000), Point(x=25000.0, y=60610.0)),
    ],
)
def test_get_polygon_center(polygon: np.ndarray, expected_result: Point) -> None:
    """
    Verify that get_polygon_center correctly calculates the centroid of a polygon.

    Scenario: Calculating the center point (centroid) of various polygons.
    Expected: The returned `Point` correctly represents the average position of all
    polygon vertices, which is used for placing labels or markers at the center
    of detected objects.
    """
    result = get_polygon_center(polygon)
    assert result == expected_result


@pytest.mark.parametrize(
    ("polygon", "expected_result"),
    [
        pytest.param(
            np.array(
                [
                    [1_000_000, 1_000_000],
                    [1_050_000, 1_000_000],
                    [1_050_000, 1_050_000],
                    [1_000_000, 1_050_000],
                ],
                dtype=np.int32,
            ),
            Point(x=1_025_000, y=1_025_000),
            id="int32-overflow",
        ),
        pytest.param(
            np.array(
                [
                    [1_000_000_000_000, 1_000_000_000_000],
                    [1_000_000_050_000, 1_000_000_000_000],
                    [1_000_000_050_000, 1_000_000_050_000],
                    [1_000_000_000_000, 1_000_000_050_000],
                ],
                dtype=np.int64,
            ),
            Point(x=1_000_000_025_000, y=1_000_000_025_000),
            id="large-offset-cancellation",
        ),
    ],
)
def test_get_polygon_center_with_large_coordinates(
    polygon: np.ndarray, expected_result: Point
) -> None:
    """Calculate centroids without integer overflow or precision loss."""
    result = get_polygon_center(polygon)

    assert result == expected_result


def test_get_polygon_center_no_deprecation_warning() -> None:
    """Regression for #2384: get_polygon_center must not fire DeprecationWarning."""
    import warnings

    polygon = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        get_polygon_center(polygon=polygon)


def test_get_polygon_center_does_not_call_np_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against reintroducing np.cross, deprecated for 2-D input in NumPy 2.0."""

    def _raise(*args, **kwargs):
        raise AssertionError("np.cross must not be called on 2-D vectors")

    monkeypatch.setattr(np, "cross", _raise)

    result = get_polygon_center(generate_test_polygon(100))

    assert result == Point(x=50.0, y=121.0)
