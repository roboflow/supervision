from contextlib import ExitStack as DoesNotRaise
from typing import cast

import numpy as np
import pytest

from supervision.annotators.utils import (
    ColorLookup,
    Trace,
    hex_to_rgba,
    is_valid_hex,
    resolve_color,
    resolve_color_idx,
    resolve_text_background_xyxy,
    rgba_to_hex,
    wrap_text,
)
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position
from tests.helpers import _create_detections


@pytest.mark.parametrize(
    ("detections", "detection_idx", "color_lookup", "expected_result", "exception"),
    [
        (
            _create_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorLookup.INDEX,
            0,
            DoesNotRaise(),
        ),  # multiple detections; index lookup
        (
            _create_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorLookup.CLASS,
            5,
            DoesNotRaise(),
        ),  # multiple detections; class lookup
        (
            _create_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorLookup.TRACK,
            2,
            DoesNotRaise(),
        ),  # multiple detections; track lookup
        (
            Detections.empty(),
            0,
            ColorLookup.INDEX,
            None,
            pytest.raises(ValueError, match="out of bounds for detections of length 0"),
        ),  # no detections; index lookup; out of bounds
        (
            _create_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            2,
            ColorLookup.INDEX,
            None,
            pytest.raises(ValueError, match="Detection index 2"),
        ),  # multiple detections; index lookup; out of bounds
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorLookup.CLASS,
            None,
            pytest.raises(ValueError, match="resolve color by class"),
        ),  # multiple detections; class lookup; no class_id
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorLookup.TRACK,
            None,
            pytest.raises(ValueError, match="resolve color by track"),
        ),  # multiple detections; class lookup; no track_id
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            np.array([1, 0]),
            1,
            DoesNotRaise(),
        ),  # multiple detections; custom lookup; correct length
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            np.array([1]),
            None,
            pytest.raises(ValueError, match="Length of color lookup 1"),
        ),  # multiple detections; custom lookup; wrong length
    ],
)
def test_resolve_color_idx(
    detections: Detections,
    detection_idx: int,
    color_lookup: ColorLookup | np.ndarray,
    expected_result: int | None,
    exception: Exception,
) -> None:
    with exception:
        result = resolve_color_idx(
            detections=detections,
            detection_idx=detection_idx,
            color_lookup=color_lookup,
        )
        assert result == expected_result


def test_resolve_color_accepts_negative_class_id() -> None:
    """Negative class ids map deterministically instead of aborting annotation."""
    detections = _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[-1])
    palette = ColorPalette.from_hex(["#ff0000", "#00ff00", "#0000ff"])

    result = resolve_color(
        color=palette,
        detections=detections,
        detection_idx=0,
        color_lookup=ColorLookup.CLASS,
    )

    assert result == Color.from_hex("#0000ff")


def test_resolve_color_accepts_palette_from_src_namespace() -> None:
    """Palette instances imported through src.supervision still resolve by index."""
    from src.supervision.draw.color import ColorPalette as SrcColorPalette

    detections = _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0])

    result = resolve_color(
        color=SrcColorPalette.DEFAULT,
        detections=detections,
        detection_idx=0,
        color_lookup=ColorLookup.CLASS,
    )

    assert result.as_bgr() == (251, 81, 163)


@pytest.mark.parametrize(
    ("text", "max_line_length", "expected_result", "exception"),
    [
        (None, None, [""], DoesNotRaise()),  # text is None
        ("", None, [""], DoesNotRaise()),  # empty string
        ("   \t  ", 3, [""], DoesNotRaise()),  # whitespace-only (spaces + tab)
        (12345, None, ["12345"], DoesNotRaise()),  # plain integer
        (-6789, None, ["-6789"], DoesNotRaise()),  # negative integer
        (np.int64(1000), None, ["1000"], DoesNotRaise()),  # NumPy int64
        ([1, 2, 3], None, ["[1, 2, 3]"], DoesNotRaise()),  # list to string
        (
            "When you play the game of thrones, you win or you die.\nFear cuts deeper than swords.\nA mind needs books as a sword needs a whetstone.",  # noqa: E501
            None,
            [
                "When you play the game of thrones, you win or you die.",
                "Fear cuts deeper than swords.",
                "A mind needs books as a sword needs a whetstone.",
            ],
            DoesNotRaise(),
        ),  # Game-of-Thrones quotes, multiline
        ("\n", None, [""], DoesNotRaise()),  # single newline
        (
            "valarmorghulisvalardoharis",
            6,
            ["valarm", "orghul", "isvala", "rdohar", "is"],
            DoesNotRaise(),
        ),  # long Valyrian phrase, wrapped
        (
            "Winter is coming\nFire and blood",
            10,
            [
                "Winter is",
                "coming",
                "Fire and",
                "blood",
            ],
            DoesNotRaise(),
        ),  # mix of short/long with newline
        (
            "What is dead may never die",
            0,
            None,
            pytest.raises(ValueError, match="max_line_length must be"),
        ),  # width 0 - invalid
        (
            "A Lannister always pays his debts",
            -1,
            None,
            pytest.raises(ValueError, match="positive integer"),
        ),  # width -1 - invalid
        (None, 10, [""], DoesNotRaise()),  # text None, width set
    ],
)
def test_wrap_text(
    text: object,
    max_line_length: int | None,
    expected_result: list[str],
    exception: Exception,
) -> None:
    with exception:
        result = wrap_text(text=text, max_line_length=max_line_length)
        assert result == expected_result


def test_resolve_text_background_xyxy_rejects_unknown_position() -> None:
    """Unsupported positions must raise instead of returning an implicit None."""
    with pytest.raises(ValueError, match="Unsupported position"):
        resolve_text_background_xyxy(
            center_coordinates=(10, 10),
            text_wh=(20, 10),
            position=cast(Position, "invalid"),
        )


def test_trace_put_requires_tracker_id_before_mutation() -> None:
    """Trace.put must not mutate internal history before validating tracker ids."""
    trace = Trace()
    detections = _create_detections(xyxy=[[0, 0, 1, 1]], class_id=[0])

    before_frame_id = trace.current_frame_id
    before_history = trace.frame_id.copy()
    before_xy = trace.xy.copy()
    before_tracker_id = trace.tracker_id.copy()

    with pytest.raises(ValueError, match="tracker_id"):
        trace.put(detections)

    assert trace.current_frame_id == before_frame_id
    assert np.array_equal(trace.frame_id, before_history)
    assert np.array_equal(trace.xy, before_xy)
    assert np.array_equal(trace.tracker_id, before_tracker_id)


@pytest.mark.parametrize(
    ("hex_color", "expected_rgba"),
    [
        ("#FF00FF", (255, 0, 255, 255)),
        ("FF00FF", (255, 0, 255, 255)),
        ("#FF00FF80", (255, 0, 255, 128)),
        ("00FF0080", (0, 255, 0, 128)),
        ("  #ff00ff80  ", (255, 0, 255, 128)),
        ("abcdef", (171, 205, 239, 255)),
    ],
)
def test_hex_to_rgba_valid(
    hex_color: str, expected_rgba: tuple[int, int, int, int]
) -> None:
    assert hex_to_rgba(hex_color) == expected_rgba


@pytest.mark.parametrize("hex_color", ["#FF00F", "#GGHHII", "#FFF", "1234567"])
def test_hex_to_rgba_invalid(hex_color: str) -> None:
    with pytest.raises(ValueError, match="Invalid hex"):
        hex_to_rgba(hex_color)


@pytest.mark.parametrize(
    ("rgba", "expected_hex"),
    [
        ((0, 0, 0, 0), "#00000000"),
        ((255, 0, 255, 255), "#FF00FFFF"),
        ((0, 255, 0, 128), "#00FF0080"),
        ((255, 255, 255, 255), "#FFFFFFFF"),
    ],
)
def test_rgba_to_hex(rgba: tuple[int, int, int, int], expected_hex: str) -> None:
    assert rgba_to_hex(rgba) == expected_hex


@pytest.mark.parametrize(
    "rgba",
    [
        (255, 0, 0),
        (256, 0, 0, 255),
        (-1, 0, 0, 255),
        (255, 0, 0, -1),
    ],
)
def test_rgba_to_hex_invalid(rgba: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="RGBA must be a 4-tuple"):
        rgba_to_hex(rgba)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("hex_color", "expected_result"),
    [
        ("#FF00FF", True),
        ("ff00ff", True),
        ("00FF0080", True),
        (" 00ff0080 ", True),
        ("#XYZ123", False),
        ("FF00F", False),
        ("#FFF", False),
        ("#1234567", False),
    ],
)
def test_is_valid_hex(hex_color: str, expected_result: bool) -> None:
    assert is_valid_hex(hex_color) is expected_result
