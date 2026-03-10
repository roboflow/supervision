from __future__ import annotations

import csv
import io
import os
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

from supervision.detection.core import Detections
from supervision.utils.logger import _get_logger

logger = _get_logger(__name__)

BASE_HEADER = [
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "class_id",
    "confidence",
    "tracker_id",
]


class WriterProtocol(Protocol):
    def writerow(self, row: Iterable[Any]) -> Any: ...


class CSVSink:
    """
    A utility class for saving detection data to a CSV file. This class is designed to
    efficiently serialize detection objects into a CSV format, allowing for the
    inclusion of bounding box coordinates and additional attributes like `confidence`,
    `class_id`, and `tracker_id`.

    !!! tip

        CSVSink allows passing custom data alongside detection fields, providing
        flexibility for logging various types of information.

    Args:
        file_name: The name of the CSV file where the detections will be stored.
            Defaults to 'output.csv'.

    Example:
        ```pycon
        >>> import supervision as sv
        >>> import numpy as np
        >>> import tempfile
        >>> import os
        >>> # Create synthetic detections
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        ...     confidence=np.array([0.9, 0.8]),
        ...     class_id=np.array([0, 1])
        ... )
        >>> # Use temporary file
        >>> temp_file = tempfile.NamedTemporaryFile(
        ...     mode='w', suffix='.csv', delete=False
        ... )
        >>> temp_file.close()
        >>> csv_sink = sv.CSVSink(temp_file.name)
        >>> with csv_sink as sink:
        ...     sink.append(detections, custom_data={'frame': 0})
        >>> os.unlink(temp_file.name)  # Clean up

        ```
    """

    def __init__(self, file_name: str = "output.csv") -> None:
        """
        Initialize the CSVSink instance.

        Args:
            file_name: The name of the CSV file.
        """
        self.file_name = file_name
        self.file: io.TextIOWrapper | None = None
        self.writer: WriterProtocol | None = None
        self.header_written = False
        self.field_names: list[str] = []

    def __enter__(self) -> CSVSink:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        """
        Open the CSV file for writing.
        """
        parent_directory = os.path.dirname(self.file_name)
        if parent_directory and not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        self.file = open(self.file_name, "w", newline="")
        self.writer = csv.writer(self.file)

    def close(self) -> None:
        """
        Close the CSV file.
        """
        if self.file:
            self.file.close()

    @staticmethod
    def parse_detection_data(
        detections: Detections, custom_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        row[key] = value
                    elif isinstance(value, np.ndarray):
                        row[key] = value[i]
                    else:
                        row[key] = value[i] if hasattr(value, "__getitem__") else value

            if custom_data:
                row.update(custom_data)
            parsed_rows.append(row)
        return parsed_rows

    def append(
        self, detections: Detections, custom_data: dict[str, Any] | None = None
    ) -> None:
        """
        Append detection data to the CSV file.

        Args:
            detections: The detection data.
            custom_data: Custom data to include.
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
            logger.warning(
                "Field names do not match the header. Expected: %s, given: %s",
                self.field_names,
                field_names,
            )

        parsed_rows = CSVSink.parse_detection_data(detections, custom_data)
        for row in parsed_rows:
            self.writer.writerow(
                [row.get(field_name, "") for field_name in self.field_names]
            )

    @staticmethod
    def parse_field_names(
        detections: Detections, custom_data: dict[str, Any] | None = None
    ) -> list[str]:
        custom_keys = set(custom_data.keys()) if custom_data else set()
        dynamic_header = sorted(
            custom_keys | set(getattr(detections, "data", {}).keys())
        )
        return BASE_HEADER + dynamic_header
