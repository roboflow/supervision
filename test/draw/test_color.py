from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import pytest

from supervision.draw.color import Color


@pytest.mark.parametrize(
    "color_hex, expected_result, exception",
    [
        # RGB 3-digit
        ("fff", Color.WHITE, DoesNotRaise()),
        ("#fff", Color.WHITE, DoesNotRaise()),
        ("f00", Color.RED, DoesNotRaise()),
        ("0f0", Color.GREEN, DoesNotRaise()),
        ("00f", Color.BLUE, DoesNotRaise()),
        # RGB 6-digit
        ("ffffff", Color.WHITE, DoesNotRaise()),
        ("#ffffff", Color.WHITE, DoesNotRaise()),
        ("#808000", Color(r=128, g=128, b=0), DoesNotRaise()),
        # RGBA 4-digit
        ("ffff", Color(r=255, g=255, b=255, a=255), DoesNotRaise()),
        ("#ffff", Color(r=255, g=255, b=255, a=255), DoesNotRaise()),
        ("f0f8", Color(r=255, g=0, b=255, a=136), DoesNotRaise()),
        ("#f0f8", Color(r=255, g=0, b=255, a=136), DoesNotRaise()),
        ("0000", Color(r=0, g=0, b=0, a=0), DoesNotRaise()),
        # RGBA 8-digit
        ("ffffffff", Color(r=255, g=255, b=255, a=255), DoesNotRaise()),
        ("#ffffffff", Color(r=255, g=255, b=255, a=255), DoesNotRaise()),
        ("ff00ff80", Color(r=255, g=0, b=255, a=128), DoesNotRaise()),
        ("#ff00ff80", Color(r=255, g=0, b=255, a=128), DoesNotRaise()),
        ("00000000", Color(r=0, g=0, b=0, a=0), DoesNotRaise()),
        ("#80808080", Color(r=128, g=128, b=128, a=128), DoesNotRaise()),
        # Invalid inputs
        ("", None, pytest.raises(ValueError)),
        ("00", None, pytest.raises(ValueError)),
        ("00000", None, pytest.raises(ValueError)),
        ("0000000", None, pytest.raises(ValueError)),
        ("000000000", None, pytest.raises(ValueError)),
        ("ffg", None, pytest.raises(ValueError)),
        ("ffgh", None, pytest.raises(ValueError)),
        ("ffgghhii", None, pytest.raises(ValueError)),
    ],
)
def test_color_from_hex(
    color_hex, expected_result: Color | None, exception: Exception
) -> None:
    with exception:
        result = Color.from_hex(color_hex=color_hex)
        assert result == expected_result


@pytest.mark.parametrize(
    "color, expected_result, exception",
    [
        # RGB colors (alpha=255 by default)
        (Color.WHITE, "#ffffff", DoesNotRaise()),
        (Color.BLACK, "#000000", DoesNotRaise()),
        (Color.RED, "#ff0000", DoesNotRaise()),
        (Color.GREEN, "#00ff00", DoesNotRaise()),
        (Color.BLUE, "#0000ff", DoesNotRaise()),
        (Color(r=128, g=128, b=0), "#808000", DoesNotRaise()),
        # RGBA colors with full opacity
        (Color(r=255, g=255, b=255, a=255), "#ffffff", DoesNotRaise()),
        (Color(r=0, g=0, b=0, a=255), "#000000", DoesNotRaise()),
        # RGBA colors with transparency
        (Color(r=255, g=0, b=255, a=128), "#ff00ff80", DoesNotRaise()),
        (Color(r=0, g=0, b=0, a=0), "#00000000", DoesNotRaise()),
        (Color(r=128, g=128, b=128, a=128), "#80808080", DoesNotRaise()),
        (Color(r=255, g=255, b=255, a=254), "#fffffffe", DoesNotRaise()),
    ],
)
def test_color_as_hex(
    color: Color, expected_result: str | None, exception: Exception
) -> None:
    with exception:
        result = color.as_hex()
        assert result == expected_result


@pytest.mark.parametrize(
    "color, expected_result",
    [
        # Test as_rgba method
        (Color(r=255, g=128, b=64), (255, 128, 64, 255)),
        (Color(r=255, g=128, b=64, a=128), (255, 128, 64, 128)),
        (Color(r=0, g=0, b=0, a=0), (0, 0, 0, 0)),
        (Color.WHITE, (255, 255, 255, 255)),
        (Color.BLACK, (0, 0, 0, 255)),
    ],
)
def test_color_as_rgba(color: Color, expected_result: tuple[int, int, int, int]) -> None:
    result = color.as_rgba()
    assert result == expected_result


@pytest.mark.parametrize(
    "color1, color2, expected_equal",
    [
        # Test equality with alpha channel
        (Color(r=255, g=128, b=64), Color(r=255, g=128, b=64), True),
        (Color(r=255, g=128, b=64, a=255), Color(r=255, g=128, b=64), True),
        (Color(r=255, g=128, b=64, a=128), Color(r=255, g=128, b=64, a=128), True),
        (Color(r=255, g=128, b=64, a=128), Color(r=255, g=128, b=64, a=127), False),
        (Color(r=255, g=128, b=64), Color(r=255, g=128, b=65), False),
    ],
)
def test_color_equality(color1: Color, color2: Color, expected_equal: bool) -> None:
    assert (color1 == color2) == expected_equal


@pytest.mark.parametrize(
    "color",
    [
        Color(r=255, g=128, b=64),
        Color(r=255, g=128, b=64, a=128),
        Color(r=0, g=0, b=0, a=0),
        Color.WHITE,
        Color.BLACK,
    ],
)
def test_color_hash(color: Color) -> None:
    # Test that colors can be hashed and used in sets/dicts
    color_set = {color}
    assert color in color_set
    
    color_dict = {color: "test"}
    assert color_dict[color] == "test"
