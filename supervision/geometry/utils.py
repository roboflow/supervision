import numpy as np
from typing import Union, Optional, Tuple

from supervision.geometry.core import Point


def get_polygon_center(polygon: np.ndarray) -> Point:
    """
    Calculate the center of a polygon.

    This function takes in a polygon as a 2-dimensional numpy ndarray and
    returns the center of the polygon as a Point object.

    The center is calculated as the center
    of the solid figure formed by the points of the polygon

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
    
    # This is one of the 3 candidate algorithms considered for centroid calculation.
    # For a more detailed discussion, see PR #1084 and commit eb33176
    
    shift_polygon = np.roll(polygon, -1, axis=0)
    signed_areas = np.cross(polygon, shift_polygon) / 2
    if signed_areas.sum() == 0:
        center = np.mean(polygon, axis=0).round()
        return Point(x=center[0], y=center[1])
    centroids = (polygon + shift_polygon) / 3.0
    center = np.average(centroids, axis=0, weights=signed_areas).round()

    return Point(x=center[0], y=center[1])
    
    
def _get_rotation_matrix(angle: Union[float, Tuple[float, float, float]]) -> np.ndarray:
    """
    Calculate rotation_matrix for 2D rotation and 3D.
    """
    if isinstance(angle, tuple):
        assert len(angle) == 3
        rm_x = np.array([  # rotation matrix along the X axis
            [1, 0, 0],
            [0, np.cos(angle[0]), -np.sin(angle[0])],
            [0, np.sin(angle[0]), np.cos(angle[0])],
        ])
        rm_y = np.array([  # rotation matrix along the Y axis
            [np.cos(angle[1]), 0, np.sin(angle[1])],
            [0, 1, 0],
            [-np.sin(angle[1]), 0, np.cos(angle[1])],
        ])
        rm_z = np.array([  # rotation matrix along the Z axis
            [np.cos(angle[2]), -np.sin(angle[2]), 0],
            [np.sin(angle[2]), np.cos(angle[2]), 0],
            [0, 0, 1],
        ])
        rotation_matrix = rm_x @ rm_y @ rm_z
    else:
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
    
    return rotation_matrix


def rotate_polygon_2d(polygon: np.ndarray, angle: float,
                      rotation_center: Optional[Point] = None) -> np.ndarray:
    """
    Rotate a polygon in its plane.
    
    This function takes in a polygon as a 2-dimensional numpy ndarray, angle
    and can take point around which to rotate
    and return the rotated polygon
    
    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.
            
        angle (float): Angle in radiance, angle to be rotated.
        
        rotation_center (None or Point): Point around which to rotate.
         Calculate center of polygon if None.

    Returns:
        Polygon: Polygon rotated on angle in its plane around rotation_center point.
        
    Examples:
        ```python
        from supervision.geometry.utils import rotate_polygon_2d
        import numpy as np

        vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        rotate_polygon_2d(vertices, np.pi / 2)
        np.array([[2, 0], [0, 0], [0, 2], [2, 2]])
        ```

    """
    polygon = polygon.astype(np.float64)
    if rotation_center is None:
        rotation_center = get_polygon_center(polygon)
    rotation_center = np.array([rotation_center.x, rotation_center.y])
    rotation_center = rotation_center.astype(np.float64)
    
    if len(polygon.shape) != 2 or len(rotation_center) != 2:
        raise ValueError('Bad shape of polygon or center')
    
    rotation_matrix = _get_rotation_matrix(angle)
    polygon -= rotation_center
    polygon = (rotation_matrix @ polygon.T).T
    polygon += rotation_center
    
    return polygon


def rotate_polygon_3d(polygon: np.ndarray, angles: Tuple[float, float, float],
                      rotation_center: Optional[np.array] = None) -> np.ndarray:
    """
    Rotate a polygon of three angles in different planes.
    
    This function rotate polygon in 3 dimensions at once.
    You can feed both a two-dimensional polygon
    and a three-dimensional one into the function,
    but it will always return a three-dimensional one.
    
    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.
            
        angles (tuple(float, float, float)): Angles in radiance. angle_x, angle_y and angle_z,
         that is, the first angle specifies the fracturing
        of the polygon along the axis X and so on.
         
        rotation_center (None or np.ndarray): Point around which to rotate.
        Calculate center of polygon if None.

    Returns:
        Polygon: Polygon rotated on angle_x, angle_y, angle_z
        in its plane around rotation_center point.
        
    Examples:
        ```python
        from supervision.geometry.utils import rotate_polygon_2d
        import numpy as np

        vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        rotate_polygon_2d(vertices, (0, np.pi / 2, 0))
        np.array([[1, 0, 1], [1, 2, 1], [1, 2, -1], [1, 0, -1]])
        ```

    """
    polygon = polygon.astype(np.float64)
    if rotation_center is None:
        rotation_center = get_polygon_center(polygon[:, :2])
        rotation_center = np.array([rotation_center.x, rotation_center.y, 0.0])
    rotation_center = rotation_center.astype(np.float64)
    
    if len(rotation_center) < 3:
        rotation_center = np.append(rotation_center, [0])
    if len(polygon.shape) < 3:
        polygon = np.append(polygon, np.zeros((len(polygon), 1)), axis=1)
    
    rotation_matrix = _get_rotation_matrix(angles)
    polygon -= rotation_center
    polygon = (rotation_matrix @ polygon.T).T
    polygon += rotation_center
    
    return polygon
