from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.utils.internal import get_instance_variables


@pytest.mark.parametrize(
    "input_obj, include_properties, expected, exception",
    [
        (
            Detections,
            False,
            {"class_id", "confidence", "mask", "tracker_id"},
            DoesNotRaise(),
        ),
        (
            Detections.empty(),
            False,
            {"xyxy", "class_id", "confidence", "mask", "tracker_id", "data"},
            DoesNotRaise(),
        ),
        (
            Detections,
            True,
            {"class_id", "confidence", "mask", "tracker_id", "area", "box_area"},
            DoesNotRaise(),
        ),
        (
            Detections.empty(),
            True,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
                "area",
                "box_area",
            },
            DoesNotRaise(),
        ),
        (
            Detections(xyxy=np.array([[1, 2, 3, 4]])),
            False,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
            },
            DoesNotRaise(),
        ),
        (
            Detections(
                xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
                class_id=np.array([1, 2]),
                confidence=np.array([0.1, 0.2]),
                mask=np.array([[[1]], [[2]]]),
                tracker_id=np.array([1, 2]),
                data={"key_1": [1, 2], "key_2": [3, 4]},
            ),
            False,
            {
                "xyxy",
                "class_id",
                "confidence",
                "mask",
                "tracker_id",
                "data",
            },
            DoesNotRaise(),
        ),
    ],
)
def test_get_instance_variables(
    input_obj, include_properties, expected, exception
) -> None:
    result = get_instance_variables(input_obj, include_properties=include_properties)
    assert result == expected
