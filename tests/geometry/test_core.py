import pytest

from supervision.geometry.core import Point, Vector


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
    """
    Verify that Vector.cross_product correctly calculates the scalar value.

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
    """
    Verify that Vector.magnitude correctly calculates Euclidean distance.

    Scenario: Calculating the magnitude (length) of a vector.
    Expected: Correct Euclidean distance between start and end points is returned,
    fundamental for various spatial calculations.
    """
    result = vector.magnitude
    assert result == expected_result
