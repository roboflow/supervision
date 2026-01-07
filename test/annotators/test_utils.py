from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.annotators.utils import ColorLookup, resolve_color_idx, wrap_text
from supervision.detection.core import Detections
from test.test_utils import mock_detections


@pytest.mark.parametrize(
    "detections, detection_idx, color_lookup, expected_result, exception",
    [
        (
            mock_detections(
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
            mock_detections(
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
            mock_detections(
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
            pytest.raises(ValueError),
        ),  # no detections; index lookup; out of bounds
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            2,
            ColorLookup.INDEX,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; index lookup; out of bounds
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorLookup.CLASS,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; class lookup; no class_id
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorLookup.TRACK,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; class lookup; no track_id
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            np.array([1, 0]),
            1,
            DoesNotRaise(),
        ),  # multiple detections; custom lookup; correct length
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            np.array([1]),
            None,
            pytest.raises(ValueError),
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


@pytest.mark.parametrize(
    "text, max_line_length, expected_result, exception",
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
            pytest.raises(ValueError),
        ),  # width 0 - invalid
        (
            "A Lannister always pays his debts",
            -1,
            None,
            pytest.raises(ValueError),
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
