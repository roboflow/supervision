import json
import os
from typing import Any, Dict, List

import pytest

import supervision as sv
from test.test_utils import mock_detections


@pytest.mark.parametrize(
    "detections, custom_data, "
    "second_detections, second_custom_data, "
    "file_name, expected_result",
    [
        (
            mock_detections(
                xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
                confidence=[0.7, 0.8],
                class_id=[0, 0],
                tracker_id=[0, 1],
                data={"class_name": ["person", "person"]},
            ),
            {"frame_number": 42},
            mock_detections(
                xyxy=[[15, 25, 35, 45], [55, 65, 75, 85]],
                confidence=[0.6, 0.9],
                class_id=[1, 1],
                tracker_id=[2, 3],
                data={"class_name": ["car", "car"]},
            ),
            {"frame_number": 43},
            "test_detections.json",
            [
                {
                    "x_min": 10,
                    "y_min": 20,
                    "x_max": 30,
                    "y_max": 40,
                    "class_id": 0,
                    "confidence": 0.699999988079071,
                    "tracker_id": 0,
                    "class_name": "person",
                    "frame_number": 42,
                },
                {
                    "x_min": 50,
                    "y_min": 60,
                    "x_max": 70,
                    "y_max": 80,
                    "class_id": 0,
                    "confidence": 0.800000011920929,
                    "tracker_id": 1,
                    "class_name": "person",
                    "frame_number": 42,
                },
                {
                    "x_min": 15,
                    "y_min": 25,
                    "x_max": 35,
                    "y_max": 45,
                    "class_id": 1,
                    "confidence": 0.6000000238418579,
                    "tracker_id": 2,
                    "class_name": "car",
                    "frame_number": 43,
                },
                {
                    "x_min": 55,
                    "y_min": 65,
                    "x_max": 75,
                    "y_max": 85,
                    "class_id": 1,
                    "confidence": 0.8999999761581421,
                    "tracker_id": 3,
                    "class_name": "car",
                    "frame_number": 43,
                },
            ],
        ),  # Multiple detections
        (
            mock_detections(
                xyxy=[[60, 70, 80, 90], [100, 110, 120, 130]],
                tracker_id=[4, 5],
                data={"class_name": ["bike", "dog"]},
            ),
            {"frame_number": 44},
            mock_detections(
                xyxy=[[65, 75, 85, 95], [105, 115, 125, 135]],
                confidence=[0.5, 0.4],
                data={"class_name": ["tree", "cat"]},
            ),
            {"frame_number": 45},
            "test_detections_missing_fields.json",
            [
                {
                    "x_min": 60,
                    "y_min": 70,
                    "x_max": 80,
                    "y_max": 90,
                    "class_id": "",
                    "confidence": "",
                    "tracker_id": 4,
                    "class_name": "bike",
                    "frame_number": 44,
                },
                {
                    "x_min": 100,
                    "y_min": 110,
                    "x_max": 120,
                    "y_max": 130,
                    "class_id": "",
                    "confidence": "",
                    "tracker_id": 5,
                    "class_name": "dog",
                    "frame_number": 44,
                },
                {
                    "x_min": 65,
                    "y_min": 75,
                    "x_max": 85,
                    "y_max": 95,
                    "class_id": "",
                    "confidence": 0.5,
                    "tracker_id": "",
                    "class_name": "tree",
                    "frame_number": 45,
                },
                {
                    "x_min": 105,
                    "y_min": 115,
                    "x_max": 125,
                    "y_max": 135,
                    "class_id": "",
                    "confidence": 0.4000000059604645,
                    "tracker_id": "",
                    "class_name": "cat",
                    "frame_number": 45,
                },
            ],
        ),  # Missing fields
        (
            mock_detections(
                xyxy=[[10, 11, 12, 13]],
                confidence=[0.95],
                data={"class_name": "unknown", "is_detected": True, "score": 1},
            ),
            {"frame_number": 46},
            mock_detections(
                xyxy=[[14, 15, 16, 17]],
                data={"class_name": "artifact", "is_detected": False, "score": 0.85},
            ),
            {"frame_number": 47},
            "test_detections_varied_data.json",
            [
                {
                    "x_min": 10,
                    "y_min": 11,
                    "x_max": 12,
                    "y_max": 13,
                    "class_id": "",
                    "confidence": 0.949999988079071,
                    "tracker_id": "",
                    "class_name": "unknown",
                    "is_detected": "True",
                    "score": "1",
                    "frame_number": 46,
                },
                {
                    "x_min": 14,
                    "y_min": 15,
                    "x_max": 16,
                    "y_max": 17,
                    "class_id": "",
                    "confidence": "",
                    "tracker_id": "",
                    "class_name": "artifact",
                    "is_detected": "False",
                    "score": "0.85",
                    "frame_number": 47,
                },
            ],
        ),  # Inconsistent Data Types
        (
            mock_detections(
                xyxy=[[20, 21, 22, 23]],
            ),
            {
                "metadata": {"sensor_id": 101, "location": "north"},
                "tags": ["urgent", "review"],
            },
            mock_detections(
                xyxy=[[14, 15, 16, 17]],
            ),
            {
                "metadata": {"sensor_id": 104, "location": "west"},
                "tags": ["not-urgent", "done"],
            },
            "test_detections_complex_data.json",
            [
                {
                    "x_min": 20,
                    "y_min": 21,
                    "x_max": 22,
                    "y_max": 23,
                    "class_id": "",
                    "confidence": "",
                    "tracker_id": "",
                    "metadata": {"sensor_id": 101, "location": "north"},
                    "tags": ["urgent", "review"],
                },
                {
                    "x_min": 14,
                    "y_min": 15,
                    "x_max": 16,
                    "y_max": 17,
                    "class_id": "",
                    "confidence": "",
                    "tracker_id": "",
                    "metadata": {"sensor_id": 104, "location": "west"},
                    "tags": ["not-urgent", "done"],
                },
            ],
        ),  # Complex Data
    ],
)
def test_json_sink(
    detections: mock_detections,
    custom_data: Dict[str, Any],
    second_detections: mock_detections,
    second_custom_data: Dict[str, Any],
    file_name: str,
    expected_result: List[List[Any]],
) -> None:
    with sv.JSONSink(file_name) as sink:
        sink.append(detections, custom_data)
        sink.append(second_detections, second_custom_data)

    assert_json_equal(file_name, expected_result)


def assert_json_equal(file_name, expected_rows):
    with open(file_name, "r") as file:
        data = json.load(file)
        assert (
            data == expected_rows
        ), f"Data in JSON file didn't match expected output: {data} != {expected_rows}"

    os.remove(file_name)
