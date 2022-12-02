from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple, List


@dataclass
class Color:
    r: int
    g: int
    b: int

    @classmethod
    def from_hex_string(cls, hex_string: str) -> Color:
        pass

    def as_bgr_tuple(self) -> Tuple[int, int, int]:
        return self.r, self.g, self.b


@dataclass
class ColorPalette:
    colors: List[Color]
