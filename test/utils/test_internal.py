import pytest
from contextlib import ExitStack as DoesNotRaise
from supervision.detection.core import Detections
from supervision.utils.internal import get_instance_variables


@pytest.mark.parametrize(
    "input_obj, include_properties, expected, exception",
    [
        (
            Detections,
            False,
            {"class_id", "confidence", "mask", "tracker_id"},
            DoesNotRaise()
        ),
        (
            Detections.empty(),
            False,
            {"xyxy", "class_id", "confidence", "mask", "tracker_id", "data"},
            DoesNotRaise()
        ),
        (
            Detections,
            True,
            {"class_id", "confidence", "mask", "tracker_id", "area", "box_area"},
            DoesNotRaise()
        ),
        (
            Detections.empty(),
            True,
            {"xyxy", "class_id", "confidence", "mask", "tracker_id", "data", "area", "box_area"},
            DoesNotRaise()
        ),
    ],
)
def test_get_instance_variables(input_obj, include_properties, expected, exception) -> None:
    result = get_instance_variables(input_obj, include_properties=include_properties)
    assert result == expected
