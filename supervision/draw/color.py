from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_COLOR_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")


@dataclass
class Color:
    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str):
        """
        Creates a Color instance from a color hex string

        :param color_hex: str : The color hex string in the format of "fff", "ffffff", "#fff", or "#ffffff"
        :return: Color : A Color instance representing the color

        Example:
        color = Color.from_hex('#ff00ff')
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    def as_rgb(self) -> Tuple[int, int, int]:
        """
        Returns the color as a tuple of integers in the RGB format

        :return: Tuple[int, int, int] : The color in the RGB format
        """
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """
        Returns the color as a tuple of integers in the BGR format

        :return: Tuple[int, int, int] : The color in the BGR format
        """
        return self.b, self.g, self.r

    @classmethod
    def white(cls) -> Color:
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    def black(cls) -> Color:
        return Color.from_hex(color_hex="#000000")

    @classmethod
    def red(cls) -> Color:
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    def green(cls) -> Color:
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    def blue(cls) -> Color:
        return Color.from_hex(color_hex="#0000ff")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classmethod
    def default(cls) -> ColorPalette:
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]):
        """
        Creates a ColorPalette instance from a list of color hex strings

        :param color_hex_list: List[str] : A list of color hex strings in the format of "fff", "ffffff", "#fff", or "#ffffff"
        :return: ColorPalette : A ColorPalette instance representing the color palette

        Example:
        color_palette = ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
        """
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        """
        Returns the color at a given index in the color palette.

        :param idx: int : The index of the color in the color palette
        :return: Color : The color at the given index

        Example:
        color_palette = ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
        color = color_palette.by_idx(1)
        """
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]
