from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import pytest

from supervision.draw.color import Color


@pytest.mark.parametrize(
    "color_hex, expected_result, exception",
    [
        ("fff", Color.WHITE, DoesNotRaise()),
        ("#fff", Color.WHITE, DoesNotRaise()),
        ("ffffff", Color.WHITE, DoesNotRaise()),
        ("#ffffff", Color.WHITE, DoesNotRaise()),
        ("f00", Color.RED, DoesNotRaise()),
        ("0f0", Color.GREEN, DoesNotRaise()),
        ("00f", Color.BLUE, DoesNotRaise()),
        ("#808000", Color(r=128, g=128, b=0), DoesNotRaise()),
        # RGBA hex codes (4 digits)
        ("f0f8", Color(r=255, g=0, b=255, a=136), DoesNotRaise()),
        ("#f0f8", Color(r=255, g=0, b=255, a=136), DoesNotRaise()),
        ("ffff", Color(r=255, g=255, b=255, a=255), DoesNotRaise()),
        ("f008", Color(r=255, g=0, b=0, a=136), DoesNotRaise()),
        # RGBA hex codes (8 digits)
        ("ff00ff80", Color(r=255, g=0, b=255, a=128), DoesNotRaise()),
        ("#ff00ff80", Color(r=255, g=0, b=255, a=128), DoesNotRaise()),
        ("ffffff00", Color(r=255, g=255, b=255, a=0), DoesNotRaise()),
        ("00ff00ff", Color(r=0, g=255, b=0, a=255), DoesNotRaise()),
        # Invalid hex codes
        ("", None, pytest.raises(ValueError)),
        ("00", None, pytest.raises(ValueError)),
        ("00000", None, pytest.raises(ValueError)),
        ("0000000", None, pytest.raises(ValueError)),
        ("000000000", None, pytest.raises(ValueError)),
        ("ffg", None, pytest.raises(ValueError)),
        ("fffg", None, pytest.raises(ValueError)),
        ("ff00ffgg", None, pytest.raises(ValueError)),
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
        (Color.WHITE, "#ffffff", DoesNotRaise()),
        (Color.BLACK, "#000000", DoesNotRaise()),
        (Color.RED, "#ff0000", DoesNotRaise()),
        (Color.GREEN, "#00ff00", DoesNotRaise()),
        (Color.BLUE, "#0000ff", DoesNotRaise()),
        (Color(r=128, g=128, b=0), "#808000", DoesNotRaise()),
        # With alpha channel
        (Color(r=255, g=0, b=255, a=128), "#ff00ff80", DoesNotRaise()),
        (Color(r=255, g=255, b=255, a=255), "#ffffff", DoesNotRaise()),
        (Color(r=0, g=255, b=0, a=0), "#00ff0000", DoesNotRaise()),
        (Color(r=128, g=128, b=0, a=200), "#808000c8", DoesNotRaise()),
    ],
)
def test_color_as_hex(
    color: Color, expected_result: str | None, exception: Exception
) -> None:
    with exception:
        result = color.as_hex()
        assert result == expected_result


@pytest.mark.parametrize(
    "color_tuple, expected_result, exception",
    [
        ((255, 255, 0, 128), Color(r=255, g=255, b=0, a=128), DoesNotRaise()),
        ((0, 255, 255, 255), Color(r=0, g=255, b=255, a=255), DoesNotRaise()),
        ((128, 0, 128, 0), Color(r=128, g=0, b=128, a=0), DoesNotRaise()),
    ],
)
def test_color_from_rgba_tuple(
    color_tuple: tuple[int, int, int, int],
    expected_result: Color | None,
    exception: Exception,
) -> None:
    with exception:
        result = Color.from_rgba_tuple(color_tuple=color_tuple)
        assert result == expected_result


@pytest.mark.parametrize(
    "color_tuple, expected_result, exception",
    [
        ((0, 255, 255, 128), Color(r=255, g=255, b=0, a=128), DoesNotRaise()),
        ((255, 255, 0, 255), Color(r=0, g=255, b=255, a=255), DoesNotRaise()),
        ((128, 0, 128, 0), Color(r=128, g=0, b=128, a=0), DoesNotRaise()),
    ],
)
def test_color_from_bgra_tuple(
    color_tuple: tuple[int, int, int, int],
    expected_result: Color | None,
    exception: Exception,
) -> None:
    with exception:
        result = Color.from_bgra_tuple(color_tuple=color_tuple)
        assert result == expected_result


@pytest.mark.parametrize(
    "color, expected_result, exception",
    [
        (Color(r=255, g=255, b=0, a=128), (255, 255, 0, 128), DoesNotRaise()),
        (Color(r=0, g=255, b=255, a=255), (0, 255, 255, 255), DoesNotRaise()),
        (Color(r=128, g=0, b=128, a=0), (128, 0, 128, 0), DoesNotRaise()),
    ],
)
def test_color_as_rgba(
    color: Color,
    expected_result: tuple[int, int, int, int] | None,
    exception: Exception,
) -> None:
    with exception:
        result = color.as_rgba()
        assert result == expected_result


@pytest.mark.parametrize(
    "color, expected_result, exception",
    [
        (Color(r=255, g=255, b=0, a=128), (0, 255, 255, 128), DoesNotRaise()),
        (Color(r=0, g=255, b=255, a=255), (255, 255, 0, 255), DoesNotRaise()),
        (Color(r=128, g=0, b=128, a=0), (128, 0, 128, 0), DoesNotRaise()),
    ],
)
def test_color_as_bgra(
    color: Color,
    expected_result: tuple[int, int, int, int] | None,
    exception: Exception,
) -> None:
    with exception:
        result = color.as_bgra()
        assert result == expected_result
