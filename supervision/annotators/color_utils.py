from typing import Tuple


def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    """Convert a HEX color string to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) not in (6, 8):
        raise ValueError("Invalid HEX color format")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    a = int(opacity * 255) if len(hex_color) == 6 else int(hex_color[6:8], 16)
    return (r, g, b, a)


def rgba_to_hex(rgba: tuple[int, int, int, int]) -> str:
    """Convert an RGBA tuple to HEX color string."""
    r, g, b, a = rgba
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


def validate_color(value: str) -> bool:
    """Check if a given string is a valid HEX color."""
    if not value.startswith("#"):
        return False
    hex_digits = value.lstrip("#")
    return len(hex_digits) in (6, 8) and all(
        c in "0123456789ABCDEFabcdef" for c in hex_digits
    )
