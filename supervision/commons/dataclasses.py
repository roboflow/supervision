from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


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


@dataclass
class Detection:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    class_id: int
    class_name: Optional[str]
    confidence: Optional[float]
    mask: Optional[np.ndarray]
    contour: Optional[np.ndarray]

    @property
    def rect(self) -> Rect:
        return Rect(x=self.x_min, y=self.y_min, width=self.x_max - self.x_min, height=self.y_max - self.y_min)
