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
        ((255, 255, 255), Color.WHITE, DoesNotRaise()),
        ((0, 0, 0), Color.BLACK, DoesNotRaise()),
        ((255, 0, 0), Color.RED, DoesNotRaise()),
        ((0, 255, 0), Color.GREEN, DoesNotRaise()),
        ((0, 0, 255), Color.BLUE, DoesNotRaise()),
        ((128, 128, 0), Color(r=128, g=128, b=0), DoesNotRaise()),
        ((300, 0, 0), None, pytest.raises(ValueError)),  # R out of range
        ((0, -10, 0), None, pytest.raises(ValueError)),  # G out of range
        ((0, 0, 500), None, pytest.raises(ValueError)),  # B out of range
        ((300, -10, 500), None, pytest.raises(ValueError)),  # All out of range
    ],
)
def test_color_from_rgb_tuple(
    color_tuple: tuple[int, int, int], expected_result: Color | None, exception: Exception
) -> None:
    with exception:
        result = Color.from_rgb_tuple(color_tuple=color_tuple)
        assert result == expected_result


@pytest.mark.parametrize(
    "color_tuple, expected_result, exception",
    [
        ((255, 255, 255), Color.WHITE, DoesNotRaise()),
        ((0, 0, 0), Color.BLACK, DoesNotRaise()),
        ((0, 0, 255), Color.RED, DoesNotRaise()),  # BGR format
        ((0, 255, 0), Color.GREEN, DoesNotRaise()),  # BGR format
        ((255, 0, 0), Color.BLUE, DoesNotRaise()),  # BGR format
        ((0, 128, 128), Color(r=128, g=128, b=0), DoesNotRaise()),  # BGR format
        ((300, 0, 0), None, pytest.raises(ValueError)),  # B out of range
        ((0, -10, 0), None, pytest.raises(ValueError)),  # G out of range
        ((0, 0, 500), None, pytest.raises(ValueError)),  # R out of range
        ((300, -10, 500), None, pytest.raises(ValueError)),  # All out of range
    ],
)
def test_color_from_bgr_tuple(
    color_tuple: tuple[int, int, int], expected_result: Color | None, exception: Exception
) -> None:
    with exception:
        result = Color.from_bgr_tuple(color_tuple=color_tuple)
        assert result == expected_result
