from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt

from supervision.utils.internal import classproperty, deprecated

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

LEGACY_COLOR_PALETTE = [
    "#A351FB",
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#0082C8",
    "#F58231",
    "#911EB4",
    "#46F0F0",
    "#F032E6",
    "#D2F53C",
    "#FABEBE",
    "#008080",
    "#E6BEFF",
    "#AA6E28",
    "#FFFAC8",
    "#800000",
    "#AAFFC3",
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

    This class provides methods to work with colors, including creating colors from hex
    codes, converting colors to hex strings, RGB tuples, and BGR tuples.

    Attributes:
        r (int): Red channel value (0-255).
        g (int): Green channel value (0-255).
        b (int): Blue channel value (0-255).

    Example:
        ```python
        import supervision as sv

        sv.Color.WHITE
        # Color(r=255, g=255, b=255)
        ```

    | Constant   | Hex Code   | RGB              |
    |------------|------------|------------------|
    | `WHITE`    | `#FFFFFF`  | `(255, 255, 255)`|
    | `BLACK`    | `#000000`  | `(0, 0, 0)`      |
    | `RED`      | `#FF0000`  | `(255, 0, 0)`    |
    | `GREEN`    | `#00FF00`  | `(0, 255, 0)`    |
    | `BLUE`     | `#0000FF`  | `(0, 0, 255)`    |
    | `YELLOW`   | `#FFFF00`  | `(255, 255, 0)`  |
    | `ROBOFLOW` | `#A351FB`  | `(163, 81, 251)` |
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
            ```python
            import supervision as sv

            sv.Color.from_hex('#ff00ff')
            # Color(r=255, g=0, b=255)
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
            ```python
            import supervision as sv

            sv.Color(r=255, g=255, b=0).as_hex()
            # '#ffff00'
            ```
        """
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def as_rgb(self) -> Tuple[int, int, int]:
        """
        Returns the color as an RGB tuple.

        Returns:
            Tuple[int, int, int]: RGB tuple.

        Example:
            ```python
            import supervision as sv

            sv.Color(r=255, g=255, b=0).as_rgb()
            # (255, 255, 0)
            ```
        """
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """
        Returns the color as a BGR tuple.

        Returns:
            Tuple[int, int, int]: BGR tuple.

        Example:
            ```python
            import supervision as sv

            sv.Color(r=255, g=255, b=0).as_bgr()
            # (0, 255, 255)
            ```
        """
        return self.b, self.g, self.r

    @classproperty
    def WHITE(cls):
        return Color.from_hex("#FFFFFF")

    @classproperty
    def BLACK(cls):
        return Color.from_hex("#000000")

    @classproperty
    def RED(cls):
        return Color.from_hex("#FF0000")

    @classproperty
    def GREEN(cls):
        return Color.from_hex("#00FF00")

    @classproperty
    def BLUE(cls):
        return Color.from_hex("#0000FF")

    @classproperty
    def YELLOW(cls):
        return Color.from_hex("#FFFF00")

    @classproperty
    def ROBOFLOW(cls):
        return Color.from_hex("#A351FB")

    @classmethod
    @deprecated(
        "`Color.white()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.WHITE` instead."
    )
    def white(cls) -> Color:
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    @deprecated(
        "`Color.black()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.BLACK` instead."
    )
    def black(cls) -> Color:
        return Color.from_hex(color_hex="#000000")

    @classmethod
    @deprecated(
        "`Color.red()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.RED` instead."
    )
    def red(cls) -> Color:
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    @deprecated(
        "`Color.green()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.GREEN` instead."
    )
    def green(cls) -> Color:
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    @deprecated(
        "`Color.blue()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.BLUE` instead."
    )
    def blue(cls) -> Color:
        return Color.from_hex(color_hex="#0000ff")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classproperty
    def DEFAULT(cls) -> ColorPalette:
        """
        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```python
            import supervision as sv

            sv.ColorPalette.DEFAULT
            # ColorPalette(colors=[Color(r=255, g=64, b=64), Color(r=255, g=161, b=160), ...])
            ```

        ![default-color-palette](https://media.roboflow.com/
        supervision-annotator-examples/default-color-palette.png)
        """  # noqa: E501 // docs
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classproperty
    def ROBOFLOW(cls) -> ColorPalette:
        """
        Returns a Roboflow color palette.

        Returns:
            ColorPalette: A ColorPalette instance with Roboflow colors.

        Example:
            ```python
            import supervision as sv

            sv.ColorPalette.ROBOFLOW
            # ColorPalette(colors=[Color(r=194, g=141, b=252), Color(r=163, g=81, b=251), ...])
            ```

        ![roboflow-color-palette](https://media.roboflow.com/
        supervision-annotator-examples/roboflow-color-palette.png)
        """  # noqa: E501 // docs
        return ColorPalette.from_hex(color_hex_list=ROBOFLOW_COLOR_PALETTE)

    @classproperty
    def LEGACY(cls) -> ColorPalette:
        return ColorPalette.from_hex(color_hex_list=LEGACY_COLOR_PALETTE)

    @classmethod
    @deprecated(
        "`ColorPalette.default()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.DEFAULT` instead."
    )
    def default(cls) -> ColorPalette:
        """
        !!! failure "Deprecated"

            `ColorPalette.default()` is deprecated and will be removed in
            `supervision-0.22.0`. Use `Color.DEFAULT` instead.

        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```python
            import supervision as sv

            sv.ColorPalette.default()
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
            ```python
            import supervision as sv

            sv.ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
            # ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    @classmethod
    def from_matplotlib(cls, palette_name: str, color_count: int) -> ColorPalette:
        """
        Create a ColorPalette instance from a Matplotlib color palette.

        Args:
            palette_name (str): Name of the Matplotlib palette.
            color_count (int): Number of colors to sample from the palette.

        Returns:
            ColorPalette: A ColorPalette instance.

        Example:
            ```python
            import supervision as sv

            sv.ColorPalette.from_matplotlib('viridis', 5)
            # ColorPalette(colors=[Color(r=68, g=1, b=84), Color(r=59, g=82, b=139), ...])
            ```

        ![visualized_color_palette](https://media.roboflow.com/
        supervision-annotator-examples/visualized_color_palette.png)
        """  # noqa: E501 // docs
        mpl_palette = plt.get_cmap(palette_name, color_count)
        colors = [
            Color(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in mpl_palette.colors
        ]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        """
        Return the color at a given index in the palette.

        Args:
            idx (int): Index of the color in the palette.

        Returns:
            Color: Color at the given index.

        Example:
            ```python
            import supervision as sv

            color_palette = sv.ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
            color_palette.by_idx(1)
            # Color(r=0, g=255, b=0)
            ```
        """
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]
