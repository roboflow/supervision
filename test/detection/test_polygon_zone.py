from test.test_utils import mock_detections
from typing import Dict

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.tools.polygon_zone import PolygonZone


@pytest.mark.parametrize(
    "polygon, detections, expected_result, current_count, class_in_current_count,\
        class_out_current_count, class_in_total_count, class_out_total_count",
    [
        (
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            mock_detections(xyxy=np.empty((0, 4)), class_id=[]),
            np.array([]),
            0,
            {},
            {},
            {},
            {},
        ),  # empty detections
        (
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            mock_detections(xyxy=[[5, 5, 10, 10]], class_id=[2], confidence=[0.9]),
            np.array([True]),
            1,
            {2: 1},
            {},
            {2: 1},
            {},
        ),  # single detection in zone
        (
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            mock_detections(xyxy=[[25, 25, 55, 55]], class_id=[1], confidence=[0.9]),
            np.array([False]),
            0,
            {},
            {1: 1},
            {},
            {1: 1},
        ),  # single detection outside zone
    ],
)
def test_trigger_detections_single_frame(
    polygon: np.ndarray,
    detections: Detections,
    expected_result: np.ndarray,
    current_count: int,
    class_in_current_count: Dict[int, int],
    class_out_current_count: Dict[int, int],
    class_in_total_count: Dict[int, int],
    class_out_total_count: Dict[int, int],
) -> None:
    zone = PolygonZone(polygon, (100, 100))
    result = zone.trigger(detections)

    assert np.array_equal(result, expected_result)
    assert zone.current_count == current_count
    assert zone.class_in_current_count == class_in_current_count
    assert zone.class_out_current_count == class_out_current_count
    assert zone.class_in_total_count == class_in_total_count
    assert zone.class_out_total_count == class_out_total_count
