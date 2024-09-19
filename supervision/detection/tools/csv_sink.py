from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

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
    efficiently serialize detection objects into a CSV format, allowing for the
    inclusion of bounding box coordinates and additional attributes like `confidence`,
    `class_id`, and `tracker_id`.

    !!! tip

        CSVSink allow to pass custom data alongside the detection fields, providing
        flexibility for logging various types of information.

    Args:
        file_name (str): The name of the CSV file where the detections will be stored.
            Defaults to 'output.csv'.

    Example:
        ```python
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO(<SOURCE_MODEL_PATH>)
        csv_sink = sv.CSVSink(<RESULT_CSV_FILE_PATH>)
        frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

        with csv_sink as sink:
            for frame in frames_generator:
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                sink.append(detections, custom_data={'<CUSTOM_LABEL>':'<CUSTOM_DATA>'})
        ```
    """

    def __init__(self, file_name: str = "output.csv") -> None:
        """
        Initialize the CSVSink instance.

        Args:
            file_name (str): The name of the CSV file.

        Returns:
            None
        """
        self.file_name = file_name
        self.file: Optional[open] = None
        self.writer: Optional[csv.writer] = None
        self.header_written = False
        self.field_names = []

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
        """
        Open the CSV file for writing.

        Returns:
            None
        """
        parent_directory = os.path.dirname(self.file_name)
        if parent_directory and not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        self.file = open(self.file_name, "w", newline="")
        self.writer = csv.writer(self.file)

    def close(self) -> None:
        """
        Close the CSV file.

        Returns:
            None
        """
        if self.file:
            self.file.close()

    @staticmethod
    def parse_detection_data(
        detections: Detections, custom_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        parsed_rows = []
        for i in range(len(detections.xyxy)):
            row = {
                "x_min": detections.xyxy[i][0],
                "y_min": detections.xyxy[i][1],
                "x_max": detections.xyxy[i][2],
                "y_max": detections.xyxy[i][3],
                "class_id": ""
                if detections.class_id is None
                else str(detections.class_id[i]),
                "confidence": ""
                if detections.confidence is None
                else str(detections.confidence[i]),
                "tracker_id": ""
                if detections.tracker_id is None
                else str(detections.tracker_id[i]),
            }

            if hasattr(detections, "data"):
                for key, value in detections.data.items():
                    if value.ndim == 0:
                        row[key] = value
                    else:
                        row[key] = value[i]

            if custom_data:
                row.update(custom_data)
            parsed_rows.append(row)
        return parsed_rows

    def append(
        self, detections: Detections, custom_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Append detection data to the CSV file.

        Args:
            detections (Detections): The detection data.
            custom_data (Dict[str, Any]): Custom data to include.

        Returns:
            None
        """
        if not self.writer:
            raise Exception(
                f"Cannot append to CSV: The file '{self.file_name}' is not open."
            )
        field_names = CSVSink.parse_field_names(detections, custom_data)
        if not self.header_written:
            self.field_names = field_names
            self.writer.writerow(field_names)
            self.header_written = True

        if field_names != self.field_names:
            print(
                f"Field names do not match the header. "
                f"Expected: {self.field_names}, given: {field_names}"
            )

        parsed_rows = CSVSink.parse_detection_data(detections, custom_data)
        for row in parsed_rows:
            self.writer.writerow(
                [row.get(field_name, "") for field_name in self.field_names]
            )

    @staticmethod
    def parse_field_names(
        detections: Detections, custom_data: Dict[str, Any]
    ) -> List[str]:
        dynamic_header = sorted(
            set(custom_data.keys()) | set(getattr(detections, "data", {}).keys())
        )
        return BASE_HEADER + dynamic_header
