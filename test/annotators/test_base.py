from contextlib import ExitStack as DoesNotRaise
from test.utils import mock_detections
from typing import Optional

import pytest

from supervision.annotators.core import ColorMap, resolve_color_idx
from supervision.detection.core import Detections


@pytest.mark.parametrize(
    "detections, detection_idx, color_map, expected_result, exception",
    [
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorMap.INDEX,
            0,
            DoesNotRaise(),
        ),  # multiple detections; index mapping
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorMap.CLASS,
            5,
            DoesNotRaise(),
        ),  # multiple detections; class mapping
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            0,
            ColorMap.TRACK,
            2,
            DoesNotRaise(),
        ),  # multiple detections; track mapping
        (
            Detections.empty(),
            0,
            ColorMap.INDEX,
            None,
            pytest.raises(ValueError),
        ),  # no detections; index mapping; out of bounds
        (
            mock_detections(
                xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]],
                class_id=[5, 3],
                tracker_id=[2, 6],
            ),
            2,
            ColorMap.INDEX,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; index mapping; out of bounds
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorMap.CLASS,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; class mapping; no class_id
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            0,
            ColorMap.TRACK,
            None,
            pytest.raises(ValueError),
        ),  # multiple detections; class mapping; no track_id
    ],
)
def test_resolve_color_idx(
    detections: Detections,
    detection_idx: int,
    color_map: ColorMap,
    expected_result: Optional[int],
    exception: Exception,
) -> None:
    with exception:
        result = resolve_color_idx(
            detections=detections,
            detection_idx=detection_idx,
            color_map=color_map,
        )
        assert result == expected_result
