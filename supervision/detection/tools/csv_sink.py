from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from supervision.detection.core import Detections

BASE_HEADER = [
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "class_id",
    "confidence",
    "tracker_id",
]

class CSVSink:
    """
    A utility class for saving detection data to a CSV file. This class is designed to
    efficiently serialize detection objects into a CSV format, allowing for the inclusion of
    bounding box coordinates and additional attributes like confidence, class ID, and tracker ID.

    The class supports the capability to include custom data alongside the detection fields,
    providing flexibility for logging various types of information.

    Args:
        filename (str): The name of the CSV file where the detections will be stored.
            Defaults to 'output.csv'.

    Example:
        ```python
        import numpy as np
        import supervision as sv
        from ultralytics import YOLO
        import time

        model = YOLO("yolov8n.pt")
        tracker = sv.ByteTrack()
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        csv_sink = sv.CSVSink(...)

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            start_time = time.time()
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {results.names[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]
            time_frame = (time.time() - start_time)

            csv_sink.append(detections, custom_data={"processing_time": time_frame})

            annotated_frame = box_annotator.annotate(
                frame.copy(), detections=detections)
            return label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels)

        csv_sink.open()
        sv.process_video(
            source_path="people-walking.mp4",
            target_path="result.mp4",
            callback=callback
        )
        csv_sink.close()
        ``` 
    """ # noqa: E501 // docs

    def __init__(self, filename: str = "output.csv"):
        self.filename = filename
        self.file: Optional[open] = None
        self.writer: Optional[csv.writer] = None
        self.header_written = False
        self.fieldnames = []  # To keep track of header names

    def __enter__(self) -> CSVSink:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    def open(self) -> None:
        self.file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.file)

    def close(self) -> None:
        if self.file:
            self.file.close()

    @staticmethod
    def parse_detection_data(detections: Detections, custom_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        parsed_rows = []
        for i in range(len(detections.xyxy)):
            row = {
                "x_min": detections.xyxy[i][0],
                "y_min": detections.xyxy[i][1],
                "x_max": detections.xyxy[i][2],
                "y_max": detections.xyxy[i][3],
                "class_id": detections.class_id[i],
                "confidence": detections.confidence[i],
                "tracker_id": detections.tracker_id[i],
            }
            if hasattr(detections, "data"):
                for key, value in detections.data.items():
                    row[key] = value[i]
            if custom_data:
                row.update(custom_data)
            parsed_rows.append(row)
        return parsed_rows

    def append(self, detections: Detections, custom_data: Dict[str, Any] = None) -> None:
        if not self.writer:
            raise Exception(f"Cannot append to CSV: The file '{self.filename}' is not open.")
        if not self.header_written:
            self.write_header(detections, custom_data)

        parsed_rows = CSVSink.parse_detection_data(detections, custom_data)
        for row in parsed_rows:
            self.writer.writerow([row.get(fieldname, "") for fieldname in self.fieldnames])

    def write_header(
        self, detections: Detections, custom_data: Dict[str, Any]
    ) -> None:
        dynamic_header = sorted(set(custom_data.keys()) | set(getattr(detections, "data", {}).keys()))
        self.fieldnames = BASE_HEADER + dynamic_header
        self.writer.writerow(self.fieldnames)
        self.header_written = True
