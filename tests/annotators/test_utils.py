from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.annotators.utils import (
    ColorLookup,
    _resolve_color_idx,
    hex_to_rgba,
    is_valid_hex,
    resolve_color_idx,
    rgba_to_hex,
    wrap_text,
)
from supervision.detection.core import Detections
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


_CLASS_IDS = np.array([5, 3, 7])
_TRACKER_IDS = np.array([2, 6, 4])


@pytest.mark.parametrize(
    (
        "instance_idx",
        "color_lookup",
        "count",
        "class_id",
        "tracker_id",
        "keypoint_idx",
        "expected_result",
        "exception",
    ),
    [
        pytest.param(
            0,
            ColorLookup.INDEX,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            0,
            DoesNotRaise(),
            id="index-first",
        ),
        pytest.param(
            1,
            ColorLookup.INDEX,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            1,
            DoesNotRaise(),
            id="index-second",
        ),
        pytest.param(
            0,
            ColorLookup.CLASS,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            5,
            DoesNotRaise(),
            id="class-first",
        ),
        pytest.param(
            1,
            ColorLookup.CLASS,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            3,
            DoesNotRaise(),
            id="class-second",
        ),
        pytest.param(
            0,
            ColorLookup.TRACK,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            2,
            DoesNotRaise(),
            id="track-first",
        ),
        pytest.param(
            1,
            ColorLookup.TRACK,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            6,
            DoesNotRaise(),
            id="track-second",
        ),
        pytest.param(
            0,
            ColorLookup.KEYPOINT,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            4,
            4,
            DoesNotRaise(),
            id="keypoint",
        ),
        pytest.param(
            3,
            ColorLookup.INDEX,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            None,
            pytest.raises(ValueError, match="out of bounds"),
            id="out-of-bounds",
        ),
        pytest.param(
            0,
            ColorLookup.CLASS,
            3,
            None,
            _TRACKER_IDS,
            None,
            None,
            pytest.raises(ValueError, match="class_id"),
            id="class-no-class-id",
        ),
        pytest.param(
            0,
            ColorLookup.TRACK,
            3,
            _CLASS_IDS,
            None,
            None,
            None,
            pytest.raises(ValueError, match="tracker_id"),
            id="track-no-tracker-id",
        ),
        pytest.param(
            0,
            ColorLookup.KEYPOINT,
            3,
            _CLASS_IDS,
            _TRACKER_IDS,
            None,
            None,
            pytest.raises(ValueError, match="KEYPOINT"),
            id="keypoint-no-keypoint-idx",
        ),
    ],
)
def test_resolve_color_idx_from_fields(
    instance_idx: int,
    color_lookup: ColorLookup,
    count: int,
    class_id: np.ndarray | None,
    tracker_id: np.ndarray | None,
    keypoint_idx: int | None,
    expected_result: int | None,
    exception: Exception,
) -> None:
    with exception:
        result = _resolve_color_idx(
            instance_idx=instance_idx,
            color_lookup=color_lookup,
            count=count,
            class_id=class_id,
            tracker_id=tracker_id,
            keypoint_idx=keypoint_idx,
        )
        assert result == expected_result


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
