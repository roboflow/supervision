from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.detection.tools.csv_sink import CSVSink
from tests.helpers import _create_detections


@pytest.mark.parametrize(
    (
        "detections",
        "custom_data",
        "second_detections",
        "second_custom_data",
        "file_name",
        "expected_result",
    ),
    [
        (
            _create_detections(
                xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
                confidence=[0.7, 0.8],
                class_id=[0, 0],
                tracker_id=[0, 1],
                data={"class_name": ["person", "person"]},
            ),
            {"frame_number": 42},
            _create_detections(
                xyxy=[[15, 25, 35, 45], [55, 65, 75, 85]],
                confidence=[0.6, 0.9],
                class_id=[1, 1],
                tracker_id=[2, 3],
                data={"class_name": ["car", "car"]},
            ),
            {"frame_number": 43},
            "test_detections.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "class_name",
                    "frame_number",
                ],
                ["10.0", "20.0", "30.0", "40.0", "0", "0.7", "0", "person", "42"],
                ["50.0", "60.0", "70.0", "80.0", "0", "0.8", "1", "person", "42"],
                ["15.0", "25.0", "35.0", "45.0", "1", "0.6", "2", "car", "43"],
                ["55.0", "65.0", "75.0", "85.0", "1", "0.9", "3", "car", "43"],
            ],
        ),  # multiple detections
        (
            _create_detections(
                xyxy=[[60, 70, 80, 90], [100, 110, 120, 130]],
                tracker_id=[4, 5],
                data={"class_name": ["bike", "dog"]},
            ),
            {"frame_number": 44},
            _create_detections(
                xyxy=[[65, 75, 85, 95], [105, 115, 125, 135]],
                confidence=[0.5, 0.4],
                data={"class_name": ["tree", "cat"]},
            ),
            {"frame_number": 45},
            "test_detections_missing_fields.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "class_name",
                    "frame_number",
                ],
                ["60.0", "70.0", "80.0", "90.0", "", "", "4", "bike", "44"],
                ["100.0", "110.0", "120.0", "130.0", "", "", "5", "dog", "44"],
                ["65.0", "75.0", "85.0", "95.0", "", "0.5", "", "tree", "45"],
                ["105.0", "115.0", "125.0", "135.0", "", "0.4", "", "cat", "45"],
            ],
        ),  # missing fields
        (
            _create_detections(
                xyxy=[[10, 11, 12, 13]],
                confidence=[0.95],
                data={"class_name": "unknown", "is_detected": True, "score": 1},
            ),
            {"frame_number": 46},
            _create_detections(
                xyxy=[[14, 15, 16, 17]],
                data={"class_name": "artifact", "is_detected": False, "score": 0.85},
            ),
            {"frame_number": 47},
            "test_detections_varied_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "class_name",
                    "frame_number",
                    "is_detected",
                    "score",
                ],
                [
                    "10.0",
                    "11.0",
                    "12.0",
                    "13.0",
                    "",
                    "0.95",
                    "",
                    "unknown",
                    "46",
                    "True",
                    "1",
                ],
                [
                    "14.0",
                    "15.0",
                    "16.0",
                    "17.0",
                    "",
                    "",
                    "",
                    "artifact",
                    "47",
                    "False",
                    "0.85",
                ],
            ],
        ),  # Inconsistent Data Types
        (
            _create_detections(
                xyxy=[[20, 21, 22, 23]],
            ),
            {
                "metadata": {"sensor_id": 101, "location": "north"},
                "tags": ["urgent", "review"],
            },
            _create_detections(
                xyxy=[[14, 15, 16, 17]],
            ),
            {
                "metadata": {"sensor_id": 104, "location": "west"},
                "tags": ["not-urgent", "done"],
            },
            "test_detections_complex_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "metadata",
                    "tags",
                ],
                [
                    "20.0",
                    "21.0",
                    "22.0",
                    "23.0",
                    "",
                    "",
                    "",
                    "{'sensor_id': 101, 'location': 'north'}",
                    "['urgent', 'review']",
                ],
                [
                    "14.0",
                    "15.0",
                    "16.0",
                    "17.0",
                    "",
                    "",
                    "",
                    "{'sensor_id': 104, 'location': 'west'}",
                    "['not-urgent', 'done']",
                ],
            ],
        ),  # Complex Data
        (
            _create_detections(
                xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
                confidence=[0.9, 0.8],
                class_id=[0, 1],
            ),
            {"area": np.array([400.0, 400.0]), "frame": 5},
            _create_detections(
                xyxy=[[15, 25, 35, 45]],
                confidence=[0.7],
                class_id=[2],
            ),
            {"area": np.array([100.0]), "frame": 6},
            "test_detections_mixed_custom_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "area",
                    "frame",
                ],
                ["10.0", "20.0", "30.0", "40.0", "0", "0.9", "", "400.0", "5"],
                ["50.0", "60.0", "70.0", "80.0", "1", "0.8", "", "400.0", "5"],
                ["15.0", "25.0", "35.0", "45.0", "2", "0.7", "", "100.0", "6"],
            ],
        ),  # mixed custom_data: ndarray sliced per row, scalar broadcast to all rows
        (
            _create_detections(
                xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
                confidence=[0.9, 0.8],
                class_id=[0, 1],
            ),
            {"ids": ["a", "b"], "tags": ("x", "y"), "frame": 7},
            _create_detections(
                xyxy=[[15, 25, 35, 45]],
                confidence=[0.7],
                class_id=[2],
            ),
            {"ids": ["c"], "tags": ("z",), "frame": 8},
            "test_detections_list_custom_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "frame",
                    "ids",
                    "tags",
                ],
                ["10.0", "20.0", "30.0", "40.0", "0", "0.9", "", "7", "a", "x"],
                ["50.0", "60.0", "70.0", "80.0", "1", "0.8", "", "7", "b", "y"],
                ["15.0", "25.0", "35.0", "45.0", "2", "0.7", "", "8", "c", "z"],
            ],
        ),  # list/tuple custom_data matching detection count is sliced per row
        (
            sv.Detections(
                xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
                data={"labels": ["person", "car"]},
            ),
            None,
            sv.Detections(
                xyxy=np.array([[15, 25, 35, 45]]),
                data={"labels": ["bus"]},
            ),
            None,
            "test_detections_plain_list_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "labels",
                ],
                ["10", "20", "30", "40", "", "", "", "person"],
                ["50", "60", "70", "80", "", "", "", "car"],
                ["15", "25", "35", "45", "", "", "", "bus"],
            ],
        ),  # plain Python list in detections.data is sliced per row without custom_data
    ],
)
def test_csv_sink(
    detections: sv.Detections,
    custom_data: dict[str, Any] | None,
    second_detections: sv.Detections,
    second_custom_data: dict[str, Any] | None,
    file_name: str,
    expected_result: list[list[Any]],
) -> None:
    with sv.CSVSink(file_name) as sink:
        if custom_data is None:
            sink.append(detections)
        else:
            sink.append(detections, custom_data)
        if second_custom_data is None:
            sink.append(second_detections)
        else:
            sink.append(second_detections, second_custom_data)

    assert_csv_equal(file_name, expected_result)


@pytest.mark.parametrize(
    (
        "detections",
        "custom_data",
        "second_detections",
        "second_custom_data",
        "file_name",
        "expected_result",
    ),
    [
        (
            _create_detections(
                xyxy=[[10, 20, 30, 40], [50, 60, 70, 80]],
                confidence=[0.7, 0.8],
                class_id=[0, 0],
                tracker_id=[0, 1],
                data={"class_name": ["person", "person"]},
            ),
            {"frame_number": 42},
            _create_detections(
                xyxy=[[15, 25, 35, 45], [55, 65, 75, 85]],
                confidence=[0.6, 0.9],
                class_id=[1, 1],
                tracker_id=[2, 3],
                data={"class_name": ["car", "car"]},
            ),
            {"frame_number": 43},
            "test_detections.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "class_name",
                    "frame_number",
                ],
                ["10.0", "20.0", "30.0", "40.0", "0", "0.7", "0", "person", "42"],
                ["50.0", "60.0", "70.0", "80.0", "0", "0.8", "1", "person", "42"],
                ["15.0", "25.0", "35.0", "45.0", "1", "0.6", "2", "car", "43"],
                ["55.0", "65.0", "75.0", "85.0", "1", "0.9", "3", "car", "43"],
            ],
        ),  # multiple detections
        (
            _create_detections(
                xyxy=[[60, 70, 80, 90], [100, 110, 120, 130]],
                tracker_id=[4, 5],
                data={"class_name": ["bike", "dog"]},
            ),
            {"frame_number": 44},
            _create_detections(
                xyxy=[[65, 75, 85, 95], [105, 115, 125, 135]],
                confidence=[0.5, 0.4],
                data={"class_name": ["tree", "cat"]},
            ),
            {"frame_number": 45},
            "test_detections_missing_fields.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "class_name",
                    "frame_number",
                ],
                ["60.0", "70.0", "80.0", "90.0", "", "", "4", "bike", "44"],
                ["100.0", "110.0", "120.0", "130.0", "", "", "5", "dog", "44"],
                ["65.0", "75.0", "85.0", "95.0", "", "0.5", "", "tree", "45"],
                ["105.0", "115.0", "125.0", "135.0", "", "0.4", "", "cat", "45"],
            ],
        ),  # missing fields
        (
            _create_detections(
                xyxy=[[10, 11, 12, 13]],
                confidence=[0.95],
                data={"class_name": "unknown", "is_detected": True, "score": 1},
            ),
            {"frame_number": 46},
            _create_detections(
                xyxy=[[14, 15, 16, 17]],
                data={"class_name": "artifact", "is_detected": False, "score": 0.85},
            ),
            {"frame_number": 47},
            "test_detections_varied_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "class_name",
                    "frame_number",
                    "is_detected",
                    "score",
                ],
                [
                    "10.0",
                    "11.0",
                    "12.0",
                    "13.0",
                    "",
                    "0.95",
                    "",
                    "unknown",
                    "46",
                    "True",
                    "1",
                ],
                [
                    "14.0",
                    "15.0",
                    "16.0",
                    "17.0",
                    "",
                    "",
                    "",
                    "artifact",
                    "47",
                    "False",
                    "0.85",
                ],
            ],
        ),  # Inconsistent Data Types
        (
            _create_detections(
                xyxy=[[20, 21, 22, 23]],
            ),
            {
                "metadata": {"sensor_id": 101, "location": "north"},
                "tags": ["urgent", "review"],
            },
            _create_detections(
                xyxy=[[14, 15, 16, 17]],
            ),
            {
                "metadata": {"sensor_id": 104, "location": "west"},
                "tags": ["not-urgent", "done"],
            },
            "test_detections_complex_data.csv",
            [
                [
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "class_id",
                    "confidence",
                    "tracker_id",
                    "metadata",
                    "tags",
                ],
                [
                    "20.0",
                    "21.0",
                    "22.0",
                    "23.0",
                    "",
                    "",
                    "",
                    "{'sensor_id': 101, 'location': 'north'}",
                    "['urgent', 'review']",
                ],
                [
                    "14.0",
                    "15.0",
                    "16.0",
                    "17.0",
                    "",
                    "",
                    "",
                    "{'sensor_id': 104, 'location': 'west'}",
                    "['not-urgent', 'done']",
                ],
            ],
        ),  # Complex Data
    ],
)
def test_csv_sink_manual(
    detections: sv.Detections,
    custom_data: dict[str, Any],
    second_detections: sv.Detections,
    second_custom_data: dict[str, Any],
    file_name: str,
    expected_result: list[list[Any]],
) -> None:
    sink = sv.CSVSink(file_name)
    sink.open()
    sink.append(detections, custom_data)
    sink.append(second_detections, second_custom_data)
    sink.close()

    assert_csv_equal(file_name, expected_result)


def assert_csv_equal(file_name, expected_rows):
    with open(file_name, newline="") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert [str(item) for item in expected_rows[i]] == row, (
                f"Row in CSV didn't match expected output: {row} != {expected_rows[i]}"
            )

    os.remove(file_name)


@pytest.mark.parametrize(
    ("value", "i", "n", "expected"),
    [
        (["x"], 0, 1, "x"),
        (["a", "b", "c"], 0, 2, ["a", "b", "c"]),
        ([42, "hello", None], 1, 3, "hello"),
        ([["a", "b"], ["c", "d"]], 0, 2, ["a", "b"]),
        ("ab", 0, 2, "ab"),
        (("z",), 0, 1, "z"),
    ],
)
def test_csv_sink_slice_value(value: Any, i: int, n: int, expected: Any) -> None:
    assert CSVSink._slice_value(value, i, n) == expected
