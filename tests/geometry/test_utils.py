from fractions import Fraction

import numpy as np
import pytest

from supervision.geometry.core import Point
from supervision.geometry.utils import (
    _clip_polygon_to_box,
    _clip_segment_to_box,
    get_polygon_center,
)


def generate_test_polygon(n: int) -> np.ndarray:
    """Generate a semicircle with a given number of points.

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
    """Verify that get_polygon_center correctly calculates the centroid of a polygon.

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
        pytest.param(
            np.array(
                [
                    [1_000_000_000_000, 1_000_000_000_000],
                    [1_000_000_000_003, 1_000_000_000_005],
                    [1_000_000_000_012, 1_000_000_000_020],
                ],
                dtype=np.int64,
            ),
            Point(x=1_000_000_000_005, y=1_000_000_000_008),
            id="int64-collinear-zero-area",
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


class TestClipSegmentToBox:
    @pytest.mark.parametrize(
        "segment",
        [
            pytest.param([[-3.25, 4.0], [7.5, -9.0]], id="inside"),
            pytest.param([[10.0, -10.0], [-10.0, 10.0]], id="corner-to-corner"),
            pytest.param([[10.0, 3.0], [0.0, 0.0]], id="end-point-on-boundary"),
            pytest.param([[2.0, 3.0], [2.0, 3.0]], id="zero-length-inside"),
            pytest.param([[10.0, 3.0], [10.0, 3.0]], id="zero-length-on-boundary"),
        ],
    )
    def test_segment_inside_is_unchanged(self, segment: list[list[float]]) -> None:
        """A segment inside the box, boundary included, is returned as given."""
        clipped = _clip_segment_to_box(np.array(segment), limit=10)

        assert clipped is not None
        np.testing.assert_array_equal(clipped, segment)

    @pytest.mark.parametrize(
        ("segment", "limit", "expected"),
        [
            pytest.param(
                [[0.0, 0.0], [30.0, 15.0]], 10, [[0.0, 0.0], [10.0, 5.0]], id="one-out"
            ),
            pytest.param(
                [[-30.0, -15.0], [30.0, 15.0]], 10, [[-10.0, -5.0], [10.0, 5.0]]
            ),
            pytest.param(
                [[-1e20, 50.0], [1e20, 50.0]],
                400,
                [[-400.0, 50.0], [400.0, 50.0]],
                id="both-far",
            ),
            pytest.param(
                [[50.0, -1e20], [50.0, 1e20]],
                400,
                [[50.0, -400.0], [50.0, 400.0]],
                id="both-far-y",
            ),
        ],
    )
    def test_segment_crossing_the_box_is_cut_at_the_boundary(
        self, segment: list[list[float]], limit: float, expected: list[list[float]]
    ) -> None:
        """Outside end points move along the segment exactly onto the boundary."""
        clipped = _clip_segment_to_box(np.array(segment), limit=limit)

        assert clipped is not None
        np.testing.assert_array_equal(clipped, expected)

    @pytest.mark.parametrize("far_first", [False, True])
    @pytest.mark.parametrize(
        "far",
        [
            pytest.param((3.0e11, 1.5e11), id="3e11"),
            pytest.param((3.0e16, 1.5e16 + 7.0), id="3e16"),
            pytest.param((2.9e16, 1.3e16), id="2.9e16"),
        ],
    )
    def test_near_horizon_end_point_stays_on_the_exact_line(
        self, far: tuple[float, float], far_first: bool
    ) -> None:
        """A huge end point is pulled back onto the exact line, whichever end it is."""
        near = (150.0, 225.0)
        segment = np.array([far, near] if far_first else [near, far])
        (x0, y0), (x1, y1) = map(Fraction, near), map(Fraction, far)
        expected_y = float(y0 + (1600 - x0) * (y1 - y0) / (x1 - x0))
        far_index = 0 if far_first else 1

        clipped = _clip_segment_to_box(segment, limit=1600)

        assert clipped is not None
        np.testing.assert_array_equal(clipped[1 - far_index], near)
        np.testing.assert_allclose(
            clipped[far_index], [1600.0, expected_y], rtol=0, atol=1e-6
        )

    @pytest.mark.parametrize(
        "segment",
        [
            pytest.param([[20.0, 0.0], [30.0, 5.0]], id="right-of-box"),
            pytest.param([[-20.0, 15.0], [20.0, 15.0]], id="parallel-above"),
            pytest.param([[5.0, 30.0], [30.0, 5.0]], id="past-corner"),
            pytest.param([[1e12, 0.0], [1e12 + 300, 0.0]], id="far-translation"),
            pytest.param([[20.0, 3.0], [20.0, 3.0]], id="zero-length-outside"),
        ],
    )
    def test_segment_missing_the_box_returns_none(
        self, segment: list[list[float]]
    ) -> None:
        """A segment with no point in the box has nothing to draw."""
        assert _clip_segment_to_box(np.array(segment), limit=10) is None


class TestClipPolygonToBox:
    @pytest.mark.parametrize(
        "polygon",
        [
            pytest.param([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]], id="inside"),
            pytest.param([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], id="vertex-at-limit"),
            pytest.param(
                [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
                id="the-box-itself",
            ),
        ],
    )
    def test_polygon_inside_is_unchanged(self, polygon: list[list[float]]) -> None:
        """A polygon inside the box, boundary included, keeps vertices and order."""
        clipped = _clip_polygon_to_box(np.array(polygon), limit=10)

        np.testing.assert_array_equal(clipped, polygon)

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize(
        ("polygon", "expected"),
        [
            pytest.param(
                [[-5.0, -5.0], [30.0, 0.0], [20.0, 30.0], [0.0, 20.0]],
                [(-5.0, -5.0), (10.0, -20 / 7), (10.0, 10.0), (-2.0, 10.0)],
                id="one-vertex-inside",
            ),
            pytest.param(
                [[-1e20, -1e20], [1e20, -1e20], [1e20, 1e20], [-1e20, 1e20]],
                [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)],
                id="far-square-around-the-box",
            ),
            pytest.param(
                [[-1e20, 5.0], [1e20, 5.0], [0.0, 1e20]],
                [(-10.0, 5.0), (10.0, 5.0), (10.0, 10.0), (-10.0, 10.0)],
                id="far-triangle-crossing-the-box",
            ),
        ],
    )
    def test_polygon_crossing_the_box_keeps_vertex_order(
        self,
        polygon: list[list[float]],
        expected: list[tuple[float, float]],
        reverse: bool,
    ) -> None:
        """The overlap is cut at the boundary with the vertices' cyclic order kept."""
        if reverse:
            polygon, expected = polygon[::-1], expected[::-1]

        clipped = _clip_polygon_to_box(np.array(polygon), limit=10)

        vertices = [tuple(vertex) for vertex in np.round(clipped, 9).tolist()]
        expected = [tuple(np.round(vertex, 9)) for vertex in expected]
        assert len(vertices) == len(expected)
        start = vertices.index(expected[0])
        assert vertices[start:] + vertices[:start] == expected

    def test_near_horizon_vertex_keeps_edge_directions(self) -> None:
        """Edges to a huge vertex are cut where they leave the box."""
        polygon = np.array([[0.0, 0.0], [2.0e11, 1.0e11], [0.0, 1.0e11]])

        clipped = _clip_polygon_to_box(polygon, limit=100)

        assert {tuple(np.round(vertex, 9)) for vertex in clipped.tolist()} == {
            (0.0, 0.0),
            (100.0, 50.0),
            (100.0, 100.0),
            (0.0, 100.0),
        }

    @pytest.mark.parametrize(
        "polygon",
        [
            pytest.param([[20.0, 20.0], [30.0, 20.0], [30.0, 30.0]], id="outside"),
            pytest.param(
                [[2.0**32, 0.0], [2.0**32 + 100, 0.0], [2.0**32 + 100, 100.0]],
                id="far-translation",
            ),
        ],
    )
    def test_polygon_outside_the_box_is_empty(self, polygon: list[list[float]]) -> None:
        """A polygon with no area in the box clips to no vertices."""
        clipped = _clip_polygon_to_box(np.array(polygon), limit=10)

        assert clipped.shape == (0, 2)
