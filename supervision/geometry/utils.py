import numpy as np

from supervision.geometry.core import Point


class PolygonCenterGetter:
    
    @staticmethod
    def mean_algo(polygon: np.ndarray) -> Point:
        """
        Calculate the center of the polygon as the center of mass
        of the points.
        """
        center = np.mean(polygon, axis=0).astype(int)
        
        return Point(x=center[0], y=center[1])
    
    @staticmethod
    def frame_algo(polygon: np.ndarray) -> Point:
        """
        Calculate the center of the polygon as the center of mass
        of the frame formed by the points of this polygon.
        """
        polygon = polygon.astype(np.float32)
        shifted_polygon = np.roll(polygon, 1, axis=0)
        points = (shifted_polygon + polygon) / 2
        vectors = shifted_polygon - polygon
        mass = np.sum(vectors ** 2, axis=1) ** 0.5
        center = np.dot(mass, points) / np.sum(mass)
        center = center.astype(int)
        
        return Point(x=center[0], y=center[1])
    
    @staticmethod
    def solid_algo(polygon: np.ndarray) -> Point:
        """
        Calculate the center of the polygon as the center of mass
        of a solid homogeneous figure
        """
        shift_polygon = np.roll(polygon, -1, axis=0)
        signed_areas = np.cross(polygon, shift_polygon) / 2
        if signed_areas.sum() == 0:
            center = np.mean(polygon, axis=0).astype(int)
            return Point(x=center[0], y=center[1])
        centroids = (polygon + shift_polygon) / 3.0
        center = np.average(centroids, axis=0, weights=signed_areas).astype(int)
        return Point(x=center[0], y=center[1])


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
    
    n: int = polygon.shape[0]  # amount of points
    
    if n <= 10 ** 2:
        return PolygonCenterGetter.mean_algo(polygon)
    
    if n <= 10 ** 4:
        return PolygonCenterGetter.solid_algo(polygon)
    
    return PolygonCenterGetter.frame_algo(polygon)
