import numpy as np
import pytest

from supervision.geometry.core import MatrixTransform, Point, Position, Vector


def test_position_list_returns_enum_values_in_definition_order() -> None:
    """Position.list returns stable string values for public option lists."""
    assert Position.list() == [position.value for position in Position]


@pytest.mark.parametrize(
    ("vector", "point", "expected_result"),
    [
        (Vector(start=Point(x=0, y=0), end=Point(x=5, y=5)), Point(x=-1, y=1), 10.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=5, y=5)), Point(x=6, y=6), 0.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=5, y=5)), Point(x=3, y=6), 15.0),
        (Vector(start=Point(x=5, y=5), end=Point(x=0, y=0)), Point(x=-1, y=1), -10.0),
        (Vector(start=Point(x=5, y=5), end=Point(x=0, y=0)), Point(x=6, y=6), 0.0),
        (Vector(start=Point(x=5, y=5), end=Point(x=0, y=0)), Point(x=3, y=6), -15.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), Point(x=0, y=0), 0.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), Point(x=0, y=-1), -1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), Point(x=0, y=1), 1.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), Point(x=0, y=0), 0.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), Point(x=0, y=-1), 1.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), Point(x=0, y=1), -1.0),
        (Vector(start=Point(x=1, y=1), end=Point(x=1, y=3)), Point(x=0, y=0), 2.0),
        (Vector(start=Point(x=1, y=1), end=Point(x=1, y=3)), Point(x=1, y=4), 0.0),
        (Vector(start=Point(x=1, y=1), end=Point(x=1, y=3)), Point(x=2, y=4), -2.0),
        (Vector(start=Point(x=1, y=3), end=Point(x=1, y=1)), Point(x=0, y=0), -2.0),
        (Vector(start=Point(x=1, y=3), end=Point(x=1, y=1)), Point(x=1, y=4), 0.0),
        (Vector(start=Point(x=1, y=3), end=Point(x=1, y=1)), Point(x=2, y=4), 2.0),
    ],
)
def test_vector_cross_product(
    vector: Vector, point: Point, expected_result: float
) -> None:
    """Verify that Vector.cross_product correctly calculates the scalar value.

    Scenario: Computing the cross product between a vector and a point.
    Expected: Correct scalar value is returned, which is used to determine which side
    of a line a point lies on (essential for line crossing counting).
    """
    result = vector.cross_product(point=point)
    assert result == expected_result


@pytest.mark.parametrize(
    ("vector", "expected_result"),
    [
        (Vector(start=Point(x=0, y=0), end=Point(x=0, y=0)), 0.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=1), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=0, y=1)), 1.0),
        (Vector(start=Point(x=-1, y=0), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=-1), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=-1, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=0, y=-1)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=3, y=4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=-3, y=4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=3, y=-4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=-3, y=-4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=4, y=3)), 5.0),
        (Vector(start=Point(x=3, y=4), end=Point(x=0, y=0)), 5.0),
        (Vector(start=Point(x=4, y=3), end=Point(x=0, y=0)), 5.0),
    ],
)
def test_vector_magnitude(vector: Vector, expected_result: float) -> None:
    """Verify that Vector.magnitude correctly calculates Euclidean distance.

    Scenario: Calculating the magnitude (length) of a vector.
    Expected: Correct Euclidean distance between start and end points is returned,
    fundamental for various spatial calculations.
    """
    result = vector.magnitude
    assert result == expected_result


def _shift_rotate_affine(
    dx: float, dy: float, degrees: float, center: tuple[float, float]
) -> np.ndarray:
    """Return a 2x3 matrix rotating by `degrees` about `center`, then shifting."""
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    cx, cy = center
    return np.array(
        [
            [cos, -sin, cx - cos * cx + sin * cy + dx],
            [sin, cos, cy - sin * cx - cos * cy + dy],
        ]
    )


SHIFT_ROTATE_AFFINE = _shift_rotate_affine(dx=30, dy=-20, degrees=5, center=(150, 150))
POINTS = np.array([[0.0, 0.0], [150.0, 150.0], [320.5, 41.25], [-12.0, 999.0]])


class TestMatrixTransform:
    @pytest.mark.parametrize("method", ["abs_to_rel", "rel_to_abs"])
    def test_affine_and_homography_forms_agree(self, method: str) -> None:
        """A 2x3 matrix and its 3x3 form with a [0, 0, 1] row map points alike."""
        affine = MatrixTransform(SHIFT_ROTATE_AFFINE)
        homography = MatrixTransform(np.vstack([SHIFT_ROTATE_AFFINE, [0, 0, 1]]))

        affine_points = getattr(affine, method)(POINTS)
        homography_points = getattr(homography, method)(POINTS)

        np.testing.assert_allclose(affine_points, homography_points, atol=1e-9)

    def test_abs_to_rel_applies_matrix(self) -> None:
        """abs_to_rel maps reference points with the matrix itself."""
        transform = MatrixTransform(np.array([[0, -1, 10], [1, 0, 20]]))

        mapped = transform.abs_to_rel(np.array([[1.0, 0.0], [0.0, 2.0]]))

        np.testing.assert_allclose(mapped, [[10.0, 21.0], [8.0, 20.0]])

    @pytest.mark.parametrize(
        "matrix",
        [
            pytest.param(SHIFT_ROTATE_AFFINE, id="shift-rotate"),
            pytest.param(
                np.array([[1.1, 0.02, 5.0], [-0.03, 0.95, 7.0], [1e-4, 2e-4, 1.0]]),
                id="homography",
            ),
        ],
    )
    def test_rel_to_abs_inverts_abs_to_rel(self, matrix: np.ndarray) -> None:
        """Mapping to the current frame and back returns the original points."""
        transform = MatrixTransform(matrix)

        round_trip = transform.rel_to_abs(transform.abs_to_rel(POINTS))

        np.testing.assert_allclose(round_trip, POINTS, atol=1e-9)

    def test_point_beyond_horizon_maps_to_nan(self) -> None:
        """A point whose homogeneous scale is not positive maps to NaN."""
        transform = MatrixTransform(np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]]))

        mapped = transform.abs_to_rel(np.array([[50.0, 0.0], [100.0, 0.0], [200.0, 0]]))

        assert np.isfinite(mapped[0]).all()
        assert np.isnan(mapped[1:]).all()

    def test_keeps_its_own_copy_of_matrix(self) -> None:
        """Mutating the caller's array afterwards does not change the transform."""
        matrix = np.eye(3)
        transform = MatrixTransform(matrix)

        matrix[0, 2] = 100.0

        np.testing.assert_allclose(transform.abs_to_rel(np.zeros((1, 2))), [[0, 0]])

    @pytest.mark.parametrize(
        ("matrix", "match"),
        [
            pytest.param(np.eye(2), "shape", id="wrong-shape"),
            pytest.param(np.zeros((3, 3)), "invertible", id="singular"),
            pytest.param(
                np.array([[1, 0, np.nan], [0, 1, 0]]), "finite", id="non-finite"
            ),
        ],
    )
    def test_rejects_invalid_matrix(self, matrix: np.ndarray, match: str) -> None:
        """Matrices that cannot map points both ways are rejected."""
        with pytest.raises(ValueError, match=match):
            MatrixTransform(matrix)

    def test_rejects_points_not_shaped_n_by_2(self) -> None:
        """Points must be an (N, 2) array."""
        transform = MatrixTransform(np.eye(3))

        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            transform.abs_to_rel(np.zeros((2, 3)))
