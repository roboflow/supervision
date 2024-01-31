import os
import pytest
import numpy as np
import json
from supervision.utils.file import JSONSink
from supervision.detection.core import Detections

@pytest.fixture(scope="module")
def detection_instances():
    # Setup detection instances as per the provided example
    detections = Detections(
        xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        confidence=np.array([0.7, 0.8]),
        class_id=np.array([0, 0]),
        tracker_id=np.array([0, 1]),
        data={'class_name': ['person', 'person']}
    )
    
    second_detections = Detections(
        xyxy=np.array([[15, 25, 35, 45], [55, 65, 75, 85]]),
        confidence=np.array([0.6, 0.9]),
        class_id=np.array([1, 1]),
        tracker_id=np.array([2, 3]),
        data={'class_name': ['car', 'car']}
    )
    
    custom_data = {'frame_number': 42}
    second_custom_data = {'frame_number': 43}
    
    return detections, custom_data, second_detections, second_custom_data

def test_json_sink(detection_instances):
    detections, custom_data, second_detections, second_custom_data = detection_instances
    json_filename = "test_detections.json"
    expected_data = [
        {
            "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40,
            "class_id": 0, "confidence": 0.7, "tracker_id": 0, "class_name": "person",
            "frame_number": 42
        },
        {
            "x_min": 50, "y_min": 60, "x_max": 70, "y_max": 80,
            "class_id": 0, "confidence": 0.8, "tracker_id": 1, "class_name": "person",
            "frame_number": 42
        },
        {
            "x_min": 15, "y_min": 25, "x_max": 35, "y_max": 45,
            "class_id": 1, "confidence": 0.6, "tracker_id": 2, "class_name": "car",
            "frame_number": 43
        },
        {
            "x_min": 55, "y_min": 65, "x_max": 75, "y_max": 85,
            "class_id": 1, "confidence": 0.9, "tracker_id": 3, "class_name": "car",
            "frame_number": 43
        }
    ]

    # Using the JSONSink class to write the detection data to a JSON file
    with JSONSink(filename=json_filename) as sink:
        sink.append(detections, custom_data)
        sink.append(second_detections, second_custom_data)

    # Read back the JSON file and verify its contents
    with open(json_filename, 'r') as file:
        data = json.load(file)
        assert data == expected_data, f"Data in JSON file did not match expected output: {data} != {expected_data}"

    # Clean up by removing the test JSON file
    os.remove(json_filename)

def test_json_sink_manual(detection_instances):
    detections, custom_data, second_detections, second_custom_data = detection_instances
    json_filename = "test_detections.json"
    expected_data = [
        {
            "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40,
            "class_id": 0, "confidence": 0.7, "tracker_id": 0, "class_name": "person",
            "frame_number": 42
        },
        {
            "x_min": 50, "y_min": 60, "x_max": 70, "y_max": 80,
            "class_id": 0, "confidence": 0.8, "tracker_id": 1, "class_name": "person",
            "frame_number": 42
        },
        {
            "x_min": 15, "y_min": 25, "x_max": 35, "y_max": 45,
            "class_id": 1, "confidence": 0.6, "tracker_id": 2, "class_name": "car",
            "frame_number": 43
        },
        {
            "x_min": 55, "y_min": 65, "x_max": 75, "y_max": 85,
            "class_id": 1, "confidence": 0.9, "tracker_id": 3, "class_name": "car",
            "frame_number": 43
        }
    ]

    sink = JSONSink(filename=json_filename)
    sink.open()
    sink.append(detections, custom_data)
    sink.append(second_detections, second_custom_data)
    sink.write_and_close()

    # Read back the JSON file and verify its contents
    with open(json_filename, 'r') as file:
        data = json.load(file)
        assert data == expected_data, f"Data in JSON file did not match expected output: {data} != {expected_data}"

    # Clean up by removing the test JSON file
    os.remove(json_filename)