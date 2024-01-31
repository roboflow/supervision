import csv
import os

import numpy as np
import pytest

import supervision as sv
from supervision.detection.core import Detections


@pytest.fixture(scope="module")
def detection_instances():
    # Setup detection instances as per the provided example
    detections = Detections(
        xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        confidence=np.array([0.7, 0.8]),
        class_id=np.array([0, 0]),
        tracker_id=np.array([0, 1]),
        data={"class_name": np.array(["person", "person"])},
    )

    second_detections = Detections(
        xyxy=np.array([[15, 25, 35, 45], [55, 65, 75, 85]]),
        confidence=np.array([0.6, 0.9]),
        class_id=np.array([1, 1]),
        tracker_id=np.array([2, 3]),
        data={"class_name": np.array(["car", "car"])},
    )

    custom_data = {"frame_number": 42}
    second_custom_data = {"frame_number": 43}

    return detections, custom_data, second_detections, second_custom_data


def test_csv_sink(detection_instances):
    detections, custom_data, second_detections, second_custom_data = detection_instances
    csv_filename = "test_detections.csv"
    expected_rows = [
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
        [10, 20, 30, 40, 0, 0.7, 0, "person", 42],
        [50, 60, 70, 80, 0, 0.8, 1, "person", 42],
        [15, 25, 35, 45, 1, 0.6, 2, "car", 43],
        [55, 65, 75, 85, 1, 0.9, 3, "car", 43],
    ]

    # Using the CSVSink class to write the detection data to a CSV file
    with sv.CSVSink(filename=csv_filename) as sink:
        sink.append(detections, custom_data)
        sink.append(second_detections, second_custom_data)

    # Read back the CSV file and verify its contents
    with open(csv_filename, mode="r", newline="") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert (
                [str(item) for item in expected_rows[i]] == row
            ), f"Row in CSV didn't match expected output: {row} != {expected_rows[i]}"

    # Clean up by removing the test CSV file
    os.remove(csv_filename)


def test_csv_sink_manual(detection_instances):
    detections, custom_data, second_detections, second_custom_data = detection_instances
    csv_filename = "test_detections.csv"
    expected_rows = [
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
        [10, 20, 30, 40, 0, 0.7, 0, "person", 42],
        [50, 60, 70, 80, 0, 0.8, 1, "person", 42],
        [15, 25, 35, 45, 1, 0.6, 2, "car", 43],
        [55, 65, 75, 85, 1, 0.9, 3, "car", 43],
    ]

    # Using the CSVSink class to write the detection data to a CSV file
    sink = sv.CSVSink(filename=csv_filename)
    sink.open()
    sink.append(detections, custom_data)
    sink.append(second_detections, second_custom_data)
    sink.close()

    # Read back the CSV file and verify its contents
    with open(csv_filename, mode="r", newline="") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert (
                [str(item) for item in expected_rows[i]] == row
            ), f"Row in CSV didn't match expected output: {row} != {expected_rows[i]}"

    # Clean up by removing the test CSV file
    os.remove(csv_filename)
