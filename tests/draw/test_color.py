from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import pytest

from supervision.draw.color import Color


@pytest.mark.parametrize(
    ("color_hex", "expected_result", "exception"),
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
        ("", None, pytest.raises(ValueError, match="Invalid length of color hash")),
        ("00", None, pytest.raises(ValueError, match="Invalid length of color hash")),
        (
            "00000",
            None,
            pytest.raises(ValueError, match="Invalid length of color hash"),
        ),
        (
            "0000000",
            None,
            pytest.raises(ValueError, match="Invalid length of color hash"),
        ),
        (
            "000000000",
            None,
            pytest.raises(ValueError, match="Invalid length of color hash"),
        ),
        ("ffg", None, pytest.raises(ValueError, match="Invalid characters in color")),
        ("fffg", None, pytest.raises(ValueError, match="Invalid characters in color")),
        (
            "ff00ffgg",
            None,
            pytest.raises(ValueError, match="Invalid characters in color"),
        ),
    ],
)
def test_color_from_hex(
    color_hex, expected_result: Color | None, exception: Exception
) -> None:
    """
    Verify that Color.from_hex correctly parses various hex string formats.

    Scenario: Creating a `Color` object from various hex string formats (3-, 4-, 6-,
    and 8-digit, with/without # prefix).
    Expected: Correct RGBA values are parsed, and invalid hex strings raise
        `ValueError`. This allows users to define colors using familiar web formats
        including alpha.
    """
    with exception:
        result = Color.from_hex(color_hex=color_hex)
        assert result == expected_result


@pytest.mark.parametrize(
    ("color", "expected_result", "exception"),
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
    """
    Verify that Color.as_hex correctly converts Color objects to hex strings.

    Scenario: Converting a `Color` object back to a hex string.
    Expected: 6-digit hex (#RRGGBB) when alpha is 255, 8-digit hex (#RRGGBBAA)
    otherwise, ensuring round-trip consistency for color definitions.
    """
    with exception:
        result = color.as_hex()
        assert result == expected_result


@pytest.mark.parametrize(
    ("color_tuple", "expected_result", "exception"),
    [
        ((255, 255, 255), Color.WHITE, DoesNotRaise()),
        ((0, 0, 0), Color.BLACK, DoesNotRaise()),
        ((255, 0, 0), Color.RED, DoesNotRaise()),
        ((0, 255, 0), Color.GREEN, DoesNotRaise()),
        ((0, 0, 255), Color.BLUE, DoesNotRaise()),
        ((128, 128, 0), Color(r=128, g=128, b=0), DoesNotRaise()),
        (
            (300, 0, 0),
            None,
            pytest.raises(ValueError, match=r"RGB values must be in range.*300, 0, 0"),
        ),  # R out of range
        (
            (0, -10, 0),
            None,
            pytest.raises(ValueError, match=r"RGB values must be in range.*0, -10, 0"),
        ),  # G out of range
        (
            (0, 0, 500),
            None,
            pytest.raises(ValueError, match=r"RGB values must be in range.*0, 0, 500"),
        ),  # B out of range
        (
            (300, -10, 500),
            None,
            pytest.raises(
                ValueError, match=r"RGB values must be in range.*300, -10, 500"
            ),
        ),  # All out of range
    ],
)
def test_color_from_rgb_tuple(
    color_tuple: tuple[int, int, int],
    expected_result: Color | None,
    exception: Exception,
) -> None:
    with exception:
        result = Color.from_rgb_tuple(color_tuple=color_tuple)
        assert result == expected_result


@pytest.mark.parametrize(
    ("color_tuple", "expected_result", "exception"),
    [
        ((255, 255, 255), Color.WHITE, DoesNotRaise()),
        ((0, 0, 0), Color.BLACK, DoesNotRaise()),
        ((0, 0, 255), Color.RED, DoesNotRaise()),  # BGR format
        ((0, 255, 0), Color.GREEN, DoesNotRaise()),  # BGR format
        ((255, 0, 0), Color.BLUE, DoesNotRaise()),  # BGR format
        ((0, 128, 128), Color(r=128, g=128, b=0), DoesNotRaise()),  # BGR format
        (
            (300, 0, 0),
            None,
            pytest.raises(ValueError, match=r"BGR values must be in range.*300, 0, 0"),
        ),  # B out of range
        (
            (0, -10, 0),
            None,
            pytest.raises(ValueError, match=r"BGR values must be in range.*0, -10, 0"),
        ),  # G out of range
        (
            (0, 0, 500),
            None,
            pytest.raises(ValueError, match=r"BGR values must be in range.*0, 0, 500"),
        ),  # R out of range
        (
            (300, -10, 500),
            None,
            pytest.raises(
                ValueError, match=r"BGR values must be in range.*300, -10, 500"
            ),
        ),  # All out of range
    ],
)
def test_color_from_bgr_tuple(
    color_tuple: tuple[int, int, int],
    expected_result: Color | None,
    exception: Exception,
) -> None:
    with exception:
        result = Color.from_bgr_tuple(color_tuple=color_tuple)
        assert result == expected_result


@pytest.mark.parametrize(
    ("color_tuple", "expected_result", "exception"),
    [
        ((255, 255, 0, 128), Color(r=255, g=255, b=0, a=128), DoesNotRaise()),
        ((0, 255, 255, 255), Color(r=0, g=255, b=255, a=255), DoesNotRaise()),
        ((128, 0, 128, 0), Color(r=128, g=0, b=128, a=0), DoesNotRaise()),
        (
            (300, 0, 0, 128),
            None,
            pytest.raises(
                ValueError, match=r"RGBA values must be in range.*300, 0, 0, 128"
            ),
        ),
        (
            (0, 0, 0, 300),
            None,
            pytest.raises(
                ValueError, match=r"RGBA values must be in range.*0, 0, 0, 300"
            ),
        ),
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
    ("color_tuple", "expected_result", "exception"),
    [
        ((0, 255, 255, 128), Color(r=255, g=255, b=0, a=128), DoesNotRaise()),
        ((255, 255, 0, 255), Color(r=0, g=255, b=255, a=255), DoesNotRaise()),
        ((128, 0, 128, 0), Color(r=128, g=0, b=128, a=0), DoesNotRaise()),
        (
            (300, 0, 0, 128),
            None,
            pytest.raises(
                ValueError, match=r"BGRA values must be in range.*300, 0, 0, 128"
            ),
        ),
        (
            (0, 0, 0, 300),
            None,
            pytest.raises(
                ValueError, match=r"BGRA values must be in range.*0, 0, 0, 300"
            ),
        ),
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
    ("color", "expected_result", "exception"),
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
    ("color", "expected_result", "exception"),
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


@pytest.mark.parametrize(
    ("color", "expected_repr"),
    [
        (Color(r=255, g=255, b=255), "Color(r=255, g=255, b=255)"),
        (Color(r=0, g=0, b=0), "Color(r=0, g=0, b=0)"),
        (Color(r=255, g=0, b=255, a=255), "Color(r=255, g=0, b=255)"),
        (Color(r=255, g=0, b=255, a=128), "Color(r=255, g=0, b=255, a=128)"),
        (Color(r=255, g=0, b=255, a=0), "Color(r=255, g=0, b=255, a=0)"),
    ],
)
def test_color_repr(color: Color, expected_repr: str) -> None:
    assert repr(color) == expected_repr


@pytest.mark.parametrize(
    ("color_a", "color_b", "expect_equal_hash"),
    [
        (Color(r=255, g=0, b=0), Color(r=255, g=0, b=0), True),
        (Color(r=255, g=0, b=0, a=255), Color(r=255, g=0, b=0), True),
        (Color(r=255, g=0, b=0, a=128), Color(r=255, g=0, b=0, a=64), False),
        (Color(r=255, g=0, b=0, a=0), Color(r=255, g=0, b=0, a=255), False),
    ],
)
def test_color_hash(color_a: Color, color_b: Color, expect_equal_hash: bool) -> None:
    assert (hash(color_a) == hash(color_b)) == expect_equal_hash
    assert (color_a == color_b) == expect_equal_hash
