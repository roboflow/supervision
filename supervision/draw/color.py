from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from supervision.utils.internal import deprecated

DEFAULT_COLOR_PALETTE = [
    "A351FB",
    "FF4040",
    "FFA1A0",
    "FF7633",
    "FFB633",
    "D1D435",
    "4CFB12",
    "94CF1A",
    "40DE8A",
    "1B9640",
    "00D6C1",
    "2E9CAA",
    "00C4FF",
    "364797",
    "6675FF",
    "0019EF",
    "863AFF",
    "530087",
    "CD3AFF",
    "FF97CA",
    "FF39C9",
]

ROBOFLOW_COLOR_PALETTE = ["C28DFC", "A351FB", "8315F9", "6706CE", "5905B3", "4D049A"]


def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")


@dataclass
class Color:
    """
    Represents a color in RGB format.

    Attributes:
        r (int): Red channel.
        g (int): Green channel.
        b (int): Blue channel.
    """

    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str) -> Color:
        """
        Create a Color instance from a hex string.

        Args:
            color_hex (str): Hex string of the color.

        Returns:
            Color: Instance representing the color.

        Example:
            ```
            >>> Color.from_hex('#ff00ff')
            Color(r=255, g=0, b=255)
            ```
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    def as_hex(self) -> str:
        """
        Converts the Color instance to a hex string.

        Returns:
            str: The hexadecimal color string.

        Example:
            ```
            >>> Color(r=255, g=0, b=255).as_hex()
            '#ff00ff'
            ```
        """
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def as_rgb(self) -> Tuple[int, int, int]:
        """
        Returns the color as an RGB tuple.

        Returns:
            Tuple[int, int, int]: RGB tuple.

        Example:
            ```
            >>> color.as_rgb()
            (255, 0, 255)
            ```
        """
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """
        Returns the color as a BGR tuple.

        Returns:
            Tuple[int, int, int]: BGR tuple.

        Example:
            ```
            >>> color.as_bgr()
            (255, 0, 255)
            ```
        """
        return self.b, self.g, self.r

    @classmethod
    @property
    def WHITE(cls):
        return Color.from_hex("#FFFFFF")

    @classmethod
    @property
    def BLACK(cls):
        return Color.from_hex("#000000")

    @classmethod
    @property
    def RED(cls):
        return Color.from_hex("#FF0000")

    @classmethod
    @property
    def GREEN(cls):
        return Color.from_hex("#00FF00")

    @classmethod
    @property
    def BLUE(cls):
        return Color.from_hex("#0000FF")

    @classmethod
    @property
    def YELLOW(cls):
        return Color.from_hex("#FFFF00")

    @classmethod
    @property
    def ROBOFLOW(cls):
        return Color.from_hex("#A351FB")

    @classmethod
    @deprecated(
        "`Color.white()` is deprecated and will be removed in "
        "`supervision-0.20.0`. Use `Color.WHITE` instead."
    )
    def white(cls) -> Color:
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    @deprecated(
        "`Color.black()` is deprecated and will be removed in "
        "`supervision-0.20.0`. Use `Color.BLACK` instead."
    )
    def black(cls) -> Color:
        return Color.from_hex(color_hex="#000000")

    @classmethod
    @deprecated(
        "`Color.red()` is deprecated and will be removed in "
        "`supervision-0.20.0`. Use `Color.RED` instead."
    )
    def red(cls) -> Color:
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    @deprecated(
        "`Color.green()` is deprecated and will be removed in "
        "`supervision-0.20.0`. Use `Color.GREEN` instead."
    )
    def green(cls) -> Color:
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    @deprecated(
        "`Color.blue()` is deprecated and will be removed in "
        "`supervision-0.20.0`. Use `Color.BLUE` instead."
    )
    def blue(cls) -> Color:
        return Color.from_hex(color_hex="#0000ff")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classmethod
    @property
    def DEFAULT(cls):
        """
        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```
            ColorPalette.DEFAULT
            # ColorPalette(colors=[Color(r=255, g=64, b=64), Color(r=255, g=161, b=160), ...])
            ```
        """  # noqa: E501 // docs
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    @property
    def ROBOFLOW(cls):
        """
        Returns a Roboflow color palette.

        Returns:
            ColorPalette: A ColorPalette instance with Roboflow colors.

        Example:
            ```
            ColorPalette.ROBOFLOW
            # ColorPalette(colors=[Color(r=194, g=141, b=252), Color(r=163, g=81, b=251), ...])
            ```
        """  # noqa: E501 // docs
        return ColorPalette.from_hex(color_hex_list=ROBOFLOW_COLOR_PALETTE)

    @classmethod
    @deprecated(
        "`ColorPalette.default()` is deprecated and will be removed in "
        "`supervision-0.20.0`. Use `Color.DEFAULT` instead."
    )
    def default(cls) -> ColorPalette:
        """
        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```
            ColorPalette.default()
            # ColorPalette(colors=[Color(r=255, g=64, b=64), Color(r=255, g=161, b=160), ...])
            ```
        """  # noqa: E501 // docs
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]) -> ColorPalette:
        """
        Create a ColorPalette instance from a list of hex strings.

        Args:
            color_hex_list (List[str]): List of color hex strings.

        Returns:
            ColorPalette: A ColorPalette instance.

        Example:
            ```
            >>> ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        """
        Return the color at a given index in the palette.

        Args:
            idx (int): Index of the color in the palette.

        Returns:
            Color: Color at the given index.

        Example:
            ```
            >>> color_palette.by_idx(1)
            Color(r=0, g=255, b=0)
            ```
        """
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]
