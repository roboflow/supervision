from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt


class Position(Enum):
    """
    Enum representing the position of an anchor point.
    """

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
        return list(map(lambda c: c.value, cls))


@dataclass
class Point:
    """
    Represents a point in 2D space.

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
        """
        Returns the point as a tuple of integers.

        Returns:
            The point as (x, y) integers.
        """
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> tuple[float, float]:
        """
        Returns the point as a tuple of floats.

        Returns:
            The point as (x, y) floats.
        """
        return self.x, self.y


@dataclass
class Vector:
    """
    Represents a vector in 2D space, defined by a start and an end point.

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
        """
        Calculate the magnitude (length) of the vector.

        Returns:
            The magnitude of the vector.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return sqrt(dx**2 + dy**2)

    @property
    def center(self) -> Point:
        """
        Calculate the center point of the vector.

        Returns:
            The center point of the vector.
        """
        return Point(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def cross_product(self, point: Point) -> float:
        """
        Calculate the 2D cross product (also known as the vector product or outer
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
    """
    Represents a rectangle in 2D space.

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
        x1, y1, x2, y2 = xyxy
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding: int) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )

    def as_xyxy_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(self.x),
            int(self.y),
            int(self.x + self.width),
            int(self.y + self.height),
        )
