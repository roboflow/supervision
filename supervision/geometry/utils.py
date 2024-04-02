import numpy as np

from supervision.geometry.core import Point


def get_polygon_center(polygon: np.ndarray) -> Point:
    """
    Calculate the center of a polygon.

    This function takes in a polygon as a 2-dimensional numpy ndarray and
    returns the center of the polygon as a Point object.

    The center is calculated as center of frame.
    polygon -> polygon, where p[i] = p[i + 1] + p[i] / 2,
     with mass = length of vector p[i + 1] - p[i]

    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.

    Returns:
        Point: The center of the polygon, represented as a
            Point object with x and y attributes.

    Examples:
        ```python
        from supervision.geometry.utils import get_polygon_center
        import numpy as np

        vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        get_polygon_center(vertices)
        Point(x=1, y=1)
        ```
    """
    shifted_polygon = np.roll(polygon, 1, axis=0)
    points = (shifted_polygon + polygon) / 2
    vectors = shifted_polygon - polygon
    mass = (vectors[:, 0] ** 2 + vectors[:, 1] ** 2) ** 0.5
    mass = np.array([mass, mass]).T
    center = (np.sum(points * mass, axis=0) / np.sum(mass) * 2).round()
    return Point(x=center[0], y=center[1])
