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
        ("f0f8", Color(r=255, g=0, b=255, a=136), DoesNotRaise()),
        ("#f0f8", Color(r=255, g=0, b=255, a=136), DoesNotRaise()),
        ("ff00ff80", Color(r=255, g=0, b=255, a=128), DoesNotRaise()),
        ("#ff00ff80", Color(r=255, g=0, b=255, a=128), DoesNotRaise()),
        ("", None, pytest.raises(ValueError)),
        ("00", None, pytest.raises(ValueError)),
        ("0000", None, pytest.raises(ValueError)),
        ("0000000", None, pytest.raises(ValueError)),
        ("ffg", None, pytest.raises(ValueError)),
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
        (Color(r=255, g=0, b=255, a=128), "#ff00ff80", DoesNotRaise()),
        (Color(r=255, g=0, b=255, a=255), "#ff00ff", DoesNotRaise()),
    ],
)
def test_color_as_hex(
    color: Color, expected_result: str | None, exception: Exception
) -> None:
    with exception:
        result = color.as_hex()
        assert result == expected_result


def test_color_as_rgba():
    color = Color(r=255, g=0, b=255, a=128)
    assert color.as_rgba() == (255, 0, 255, 128)
    color2 = Color(r=255, g=255, b=0)
    assert color2.as_rgba() == (255, 255, 0, 255)
