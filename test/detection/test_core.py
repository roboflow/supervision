from contextlib import ExitStack as DoesNotRaise

import pytest

from supervision import Detections

from typing import Optional, Union

import numpy as np


PREDICTIONS = np.array([
        [       2254,         906,        2447,        1353,     0.90538,           0],
        [       2049,        1133,        2226,        1371,     0.59002,          56],
        [        727,        1224,         838,        1601,     0.51119,          39],
        [        808,        1214,         910,        1564,     0.45287,          39],
        [          6,          52,        1131,        2133,     0.45057,          72],
        [        299,        1225,         512,        1663,     0.45029,          39],
        [        529,         874,         645,         945,     0.31101,          39],
        [          8,          47,        1935,        2135,     0.28192,          72],
        [       2265,         813,        2328,         901,      0.2714,          62]
    ], dtype=np.float32)

DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=PREDICTIONS[:, 4],
    class_id=PREDICTIONS[:, 5].astype(int)
)


PREDICTIONS = np.array([
        [       2254,         906,        2447,        1353,     0.90538,           0],
        [       2049,        1133,        2226,        1371,     0.59002,          56],
        [        727,        1224,         838,        1601,     0.51119,          39],
    ], dtype=np.float32)


@pytest.mark.parametrize(
    'detections, index, expected_result, exception',
    [
        (
            DETECTIONS,
            DETECTIONS.class_id == 0,
            Detections(
                xyxy=np.array([
                    [       2254,         906,        2447,        1353]
                ], dtype=np.float32),
                confidence=np.array([
                    0.90538
                ], dtype=np.float32),
                class_id=np.array([
                    0
                ], dtype=int)
            ),
            DoesNotRaise()
        ),  # take only detections with class_id = 0
        (
            DETECTIONS,
            DETECTIONS.confidence > 0.5,
            Detections(
                xyxy=np.array([
                    [       2254,         906,        2447,        1353],
                    [       2049,        1133,        2226,        1371],
                    [        727,        1224,         838,        1601]
                ], dtype=np.float32),
                confidence=np.array([
                    0.90538,
                    0.59002,
                    0.51119
                ], dtype=np.float32),
                class_id=np.array([
                    0,
                    56,
                    39
                ], dtype=int)
            ),
            DoesNotRaise()
        ),  # take only detections with confidence > 0.5
        (
            DETECTIONS,
            np.array([True, True, True, True, True, True, True, True, True], dtype=bool),
            DETECTIONS,
            DoesNotRaise()
        ),  # take all detections
        (
            DETECTIONS,
            np.array([False, False, False, False, False, False, False, False, False], dtype=bool),
            Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=int)
            ),
            DoesNotRaise()
        ),  # take no detections
    ]
)
def test_getitem(
        detections: Detections,
        index: Union[int, slice, np.ndarray],
        expected_result: Optional[Detections],
        exception: Exception
) -> None:
    with exception:
        result = detections[index]
        assert result == expected_result
