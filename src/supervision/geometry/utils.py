from fractions import Fraction

import numpy as np
import numpy.typing as npt

from supervision.geometry.core import Point


def get_polygon_center(polygon: npt.NDArray[np.number]) -> Point:
    """Calculate the center of a polygon. The center is calculated as the center of the
    solid figure formed by the points of the polygon.

    Args:
        polygon: A 2-dimensional numpy ndarray representing the vertices of the
            polygon.

    Returns:
        The center of the polygon, represented as a Point object with x and y
            attributes.

    Raises:
        ValueError: If the polygon has no vertices.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> polygon = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        >>> center = sv.get_polygon_center(polygon=polygon)
        >>> float(center.x)
        1.0
        >>> float(center.y)
        1.0

        ```
    """

    # This is one of the 3 candidate algorithms considered for centroid calculation.
    # For a more detailed discussion, see PR #1084 and commit eb33176

    if len(polygon) == 0:
        raise ValueError("Polygon must have at least one vertex.")

    origin = polygon[0].astype(np.float64, copy=False)
    polygon = polygon.astype(np.float64, copy=False) - origin
    shift_polygon = np.roll(polygon, -1, axis=0)
    signed_areas = (
        polygon[..., 0] * shift_polygon[..., 1]
        - polygon[..., 1] * shift_polygon[..., 0]
    ) / 2
    if signed_areas.sum() == 0:
        center = (np.mean(polygon, axis=0) + origin).round()
        return Point(x=center[0], y=center[1])
    centroids = (polygon + shift_polygon) / 3.0
    center = (np.average(centroids, axis=0, weights=signed_areas) + origin).round()

    return Point(x=center[0], y=center[1])


def _clip_segment_to_box(
    segment: npt.NDArray[np.float64], limit: float
) -> npt.NDArray[np.float64] | None:
    """Clip a `(2, 2)` segment to the square `[-limit, limit]²` with Liang-Barsky.

    Outside end points move along the segment onto the boundary; inside ones are
    kept. Segments leaving the box are clipped in exact fractions, since float
    interpolation between end points as far apart as `1e20` loses the in-box part.
    Returns `None` if no part lies in the box or an end point is not finite.
    """
    points = np.array(segment, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        return None
    # Segments already inside, the common case, skip the slower exact arithmetic.
    if np.all(np.abs(points) <= limit):
        return points
    start, end = _to_fractions(points)
    bound = Fraction(limit)
    t_enter, t_exit = Fraction(0), Fraction(1)
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        # Each boundary gives a constraint p * t <= q on the segment parameter t.
        for p, q in ((-delta, start[axis] + bound), (delta, bound - start[axis])):
            if p == 0:
                if q < 0:
                    return None
            elif p < 0:
                t_enter = max(t_enter, q / p)
            else:
                t_exit = min(t_exit, q / p)
    if t_enter > t_exit:
        return None
    return np.array(
        [
            [float(s + t * (e - s)) for s, e in zip(start, end)]
            for t in (t_enter, t_exit)
        ],
        dtype=np.float64,
    )


def _clip_polygon_to_box(
    polygon: npt.NDArray[np.float64], limit: float
) -> npt.NDArray[np.float64]:
    """Clip an `(N, 2)` polygon to the square `[-limit, limit]²` (Sutherland-Hodgman).

    Edges are cut where they cross the boundary, keeping their directions and
    cyclic order; a polygon inside the box is returned unchanged. Polygons leaving
    the box are clipped in exact fractions, as in `_clip_segment_to_box`. Returns
    an empty `(0, 2)` array if no part lies in the box or a vertex is not finite.
    """
    points = np.array(polygon, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        return np.empty((0, 2), dtype=np.float64)
    # Polygons already inside, the common case, skip the slower exact arithmetic.
    if np.all(np.abs(points) <= limit):
        return points
    vertices = _to_fractions(points)
    bound = Fraction(limit)
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        clipped = []
        for index, vertex in enumerate(vertices):
            previous = vertices[index - 1]
            # Inside this boundary means sign * coordinate <= limit.
            inside = sign * vertex[axis] <= bound
            if inside != (sign * previous[axis] <= bound):
                ratio = (sign * bound - previous[axis]) / (
                    vertex[axis] - previous[axis]
                )
                clipped.append([p + ratio * (v - p) for p, v in zip(previous, vertex)])
            if inside:
                clipped.append(vertex)
        vertices = clipped
    return np.array(
        [[float(value) for value in vertex] for vertex in vertices], dtype=np.float64
    ).reshape(-1, 2)


def _to_fractions(points: npt.NDArray[np.float64]) -> list[list[Fraction]]:
    """Convert finite `(N, 2)` points to exact fractions; floats convert exactly."""
    return [[Fraction(value) for value in point] for point in points.tolist()]
