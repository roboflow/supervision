from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Position(Enum):
    """Enum representing the position of an anchor point."""

    CENTER = "CENTER"
    CENTER_LEFT = "CENTER_LEFT"
    CENTER_RIGHT = "CENTER_RIGHT"
    TOP_CENTER = "TOP_CENTER"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    CENTER_OF_MASS = "CENTER_OF_MASS"

    @classmethod
    def list(cls) -> list[str]:
        """Return all position values in their definition order."""
        return [position.value for position in cls]


@dataclass
class Point:
    """Represents a point in 2D space.

    Attributes:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.

    Example:
        ```pycon
        >>> from supervision.geometry.core import Point
        >>> point = Point(x=10.0, y=20.0)
        >>> point.as_xy_int_tuple()
        (10, 20)
        >>> point.as_xy_float_tuple()
        (10.0, 20.0)

        ```
    """

    x: float
    y: float

    def as_xy_int_tuple(self) -> tuple[int, int]:
        """Returns the point as a tuple of integers.

        Returns:
            The point as (x, y) integers.
        """
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> tuple[float, float]:
        """Returns the point as a tuple of floats.

        Returns:
            The point as (x, y) floats.
        """
        return self.x, self.y


@dataclass
class Vector:
    """Represents a vector in 2D space, defined by a start and an end point.

    Attributes:
        start: The starting point of the vector.
        end: The end point of the vector.

    Example:
        ```pycon
        >>> from supervision.geometry.core import Point, Vector
        >>> start_point = Point(x=0.0, y=0.0)
        >>> end_point = Point(x=3.0, y=4.0)
        >>> vector = Vector(start=start_point, end=end_point)
        >>> vector.magnitude
        5.0
        >>> vector.center
        Point(x=1.5, y=2.0)

        ```
    """

    start: Point
    end: Point

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude (length) of the vector.

        Returns:
            The magnitude of the vector.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return sqrt(dx**2 + dy**2)

    @property
    def center(self) -> Point:
        """Calculate the center point of the vector.

        Returns:
            The center point of the vector.
        """
        return Point(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def cross_product(self, point: Point) -> float:
        """Calculate the 2D cross product (also known as the vector product or outer
        product) of the vector and a point, treated as vectors in 2D space.

        Args:
            point: The point to be evaluated, treated as the endpoint of a
                vector originating from the 'start' of the main vector.

        Returns:
            The scalar value of the cross product. It is positive if 'point'
                lies to the left of the vector (when moving from 'start' to 'end'),
                negative if it lies to the right, and 0 if it is collinear with the
                vector.
        """
        dx_vector = self.end.x - self.start.x
        dy_vector = self.end.y - self.start.y
        dx_point = point.x - self.start.x
        dy_point = point.y - self.start.y
        return (dx_vector * dy_point) - (dy_vector * dx_point)


@dataclass
class Rect:
    """Represents a rectangle in 2D space.

    Attributes:
        x: The x-coordinate of the top-left corner of the rectangle.
        y: The y-coordinate of the top-left corner of the rectangle.
        width: The width of the rectangle.
        height: The height of the rectangle.

    Example:
        ```pycon
        >>> from supervision.geometry.core import Rect
        >>> rect = Rect(x=10.0, y=20.0, width=30.0, height=40.0)
        >>> rect.top_left
        Point(x=10.0, y=20.0)
        >>> rect.bottom_right
        Point(x=40.0, y=60.0)
        >>> rect.as_xyxy_int_tuple()
        (10, 20, 40, 60)

        ```
    """

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_xyxy(cls, xyxy: tuple[float, float, float, float]) -> Rect:
        """Create a rectangle from `(x_min, y_min, x_max, y_max)` coordinates."""
        x1, y1, x2, y2 = xyxy
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @property
    def top_left(self) -> Point:
        """Return the top-left corner as a `Point`."""
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        """Return the bottom-right corner as a `Point`."""
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding: int) -> Rect:
        """Return a rectangle expanded by `padding` pixels on every side."""
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )

    def as_xyxy_int_tuple(self) -> tuple[int, int, int, int]:
        """Return `(x_min, y_min, x_max, y_max)` coordinates as integers."""
        return (
            int(self.x),
            int(self.y),
            int(self.x + self.width),
            int(self.y + self.height),
        )


class CoordinatesTransform(Protocol):
    """Map points between a reference frame and the current frame.

    Zones and lines are defined in the coordinates of a reference frame
    ("absolute"). When the camera moves, a transform tells where those coordinates
    lie in the current frame ("relative"). Pass one as `coord_transform` to
    `PolygonZone.trigger`, `LineZone.trigger` and their annotators to keep a zone
    attached to the scene while the camera moves.

    Any object with these two methods satisfies the protocol, including
    `HomographyTransformation` and `IdentityTransformation` from the `trackers`
    package, which `trackers.MotionEstimator.update` returns. `MatrixTransform`
    wraps a plain 2x3 or 3x3 matrix.

    Example:
        ```pycon
        >>> import numpy as np
        >>> class Shift:
        ...     def abs_to_rel(self, points): return points + [100, 0]
        ...     def rel_to_abs(self, points): return points - [100, 0]
        >>> Shift().rel_to_abs(np.array([[150.0, 50.0]]))
        array([[50., 50.]])

        ```
    """

    def abs_to_rel(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map points from reference-frame to current-frame coordinates.

        Args:
            points: Points of shape `(N, 2)` in reference-frame coordinates.

        Returns:
            Points of shape `(N, 2)` in current-frame coordinates.
        """
        ...

    def rel_to_abs(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map points from current-frame to reference-frame coordinates.

        Args:
            points: Points of shape `(N, 2)` in current-frame coordinates.

        Returns:
            Points of shape `(N, 2)` in reference-frame coordinates.
        """
        ...


class MatrixTransform:
    """A `CoordinatesTransform` given by an affine or homography matrix.

    The matrix maps reference-frame points to current-frame points, the same
    direction as `trackers.HomographyTransformation`: `abs_to_rel` applies the
    matrix and `rel_to_abs` applies its inverse. A `(2, 3)` affine matrix, as
    returned by `cv2.estimateAffinePartial2D`, is treated as a `(3, 3)` matrix
    whose last row is `[0, 0, 1]`.

    A homography is defined only up to scale, so `matrix` and any non-zero
    multiple of it, such as `-matrix`, give the same transform: points are
    mapped with the sign that makes the homogeneous scale `w` positive at
    `(0, 0)`, or at `(1, 1)` and then `(1, 0)` if `(0, 0)` lies on the horizon.
    A point whose `w` is then not positive lies on or beyond the horizon and is
    returned as `NaN`; zones treat such points as outside.

    Attributes:
        matrix: The read-only `(3, 3)` reference-to-current matrix as given.
        inverse_matrix: The read-only `(3, 3)` inverse of `matrix`.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> transform = sv.MatrixTransform(np.array([[1, 0, 10], [0, 1, 20]]))
        >>> transform.abs_to_rel(np.array([[0.0, 0.0], [5.0, 5.0]]))
        array([[10., 20.],
               [15., 25.]])
        >>> transform.rel_to_abs(np.array([[10.0, 20.0]]))
        array([[0., 0.]])

        ```
    """

    def __init__(self, matrix: npt.ArrayLike) -> None:
        """Build a transform from a reference-to-current matrix.

        Args:
            matrix: A `(2, 3)` affine or `(3, 3)` homography matrix that maps
                reference-frame coordinates to current-frame coordinates.

        Raises:
            ValueError: If `matrix` is not of shape `(2, 3)` or `(3, 3)`,
                holds non-finite values, or is not invertible.
        """
        matrix_array = np.array(matrix, dtype=np.float64)
        if matrix_array.shape == (2, 3):
            matrix_array = np.vstack([matrix_array, [0.0, 0.0, 1.0]])
        elif matrix_array.shape != (3, 3):
            raise ValueError(
                f"Matrix must have shape (2, 3) or (3, 3); got {matrix_array.shape}."
            )
        if not np.all(np.isfinite(matrix_array)):
            raise ValueError("Matrix must contain only finite values.")
        try:
            inverse_matrix = np.linalg.inv(matrix_array).astype(np.float64)
        except np.linalg.LinAlgError as error:
            raise ValueError("Matrix must be invertible.") from error

        # Points are mapped with sign-normalised copies; negating a matrix
        # negates its inverse.
        sign = _homography_sign(matrix_array)
        self._forward_matrix: npt.NDArray[np.float64] = sign * matrix_array
        self._inverse_matrix: npt.NDArray[np.float64] = sign * inverse_matrix
        # Read-only, so editing them cannot silently diverge from the copies above.
        matrix_array.setflags(write=False)
        inverse_matrix.setflags(write=False)
        self.matrix: npt.NDArray[np.float64] = matrix_array
        self.inverse_matrix: npt.NDArray[np.float64] = inverse_matrix

    def abs_to_rel(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map points from reference-frame to current-frame coordinates.

        Args:
            points: Points of shape `(N, 2)` in reference-frame coordinates.

        Returns:
            Points of shape `(N, 2)` in current-frame coordinates, with `NaN`
                where a point maps through infinity.

        Raises:
            ValueError: If `points` is not of shape `(N, 2)`.
        """
        return _apply_homogeneous_matrix(self._forward_matrix, points)

    def rel_to_abs(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map points from current-frame to reference-frame coordinates.

        Args:
            points: Points of shape `(N, 2)` in current-frame coordinates.

        Returns:
            Points of shape `(N, 2)` in reference-frame coordinates, with `NaN`
                where a point maps through infinity.

        Raises:
            ValueError: If `points` is not of shape `(N, 2)`.
        """
        return _apply_homogeneous_matrix(self._inverse_matrix, points)


def _homography_sign(matrix: npt.NDArray[np.float64]) -> float:
    """Return the sign making `w` positive at `(0, 0)`, else `(1, 1)`, else `(1, 0)`."""
    h20, h21, h22 = matrix[2]
    for w in (h22, h20 + h21 + h22, h20 + h22):
        if w != 0:
            return 1.0 if w > 0 else -1.0
    return 1.0


def _apply_homogeneous_matrix(
    matrix: npt.NDArray[np.float64], points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Apply a `(3, 3)` matrix to `(N, 2)` points; non-positive `w` gives `NaN`."""
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"Points must have shape (N, 2); got {points_array.shape}.")
    homogeneous = points_array @ matrix[:, :2].T + matrix[:, 2]
    scale = homogeneous[:, 2:3]
    # Points with w <= 0 lie on or beyond the line at infinity; dividing would
    # mirror them back into the image, so they are reported as NaN instead.
    with np.errstate(divide="ignore", invalid="ignore"):
        mapped = homogeneous[:, :2] / scale
    return np.where(scale > 0, mapped, np.nan)


def _transform_points(
    points: npt.NDArray[np.number],
    transform: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Apply a transform method to `(..., 2)` points via `(N, 2)`, checking shape."""
    flat_points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mapped = np.asarray(transform(flat_points), dtype=np.float64)
    if mapped.shape != flat_points.shape:
        raise ValueError(
            f"coord_transform must return points of shape {flat_points.shape}; "
            f"got {mapped.shape}."
        )
    return mapped.reshape(np.shape(points))
