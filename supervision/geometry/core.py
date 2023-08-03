from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Position(Enum):
    CENTER = "CENTER"
    BOTTOM_CENTER = "BOTTOM_CENTER"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @staticmethod
    def get_direction(p1: Point, p2: Point) -> bool:
        if p1.x < p2.x and p1.y < p2.y:
            return False
        else:
            return True

    @staticmethod
    def ccw(A: Point, B: Point, C: Point):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    @staticmethod
    def intersect(A: Point, B: Point, C: Point, D: Point):
        return Point.ccw(A, C, D) != Point.ccw(B, C, D) and Point.ccw(A, B, C) != Point.ccw(A, B, D)


@dataclass
class Vector:
    start: Point
    end: Point

    def is_in(self, point: Point) -> bool:
        v1 = Vector(self.start, self.end)
        v2 = Vector(self.start, point)
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (
            v1.end.y - v1.start.y
        ) * (v2.end.x - v2.start.x)
        return cross_product < 0


    @staticmethod
    def is_in_vectors(v1: Vector, v2: Vector):
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (
                v1.end.y - v1.start.y
        ) * (v2.end.x - v2.start.x)
        return cross_product < 0



@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )
