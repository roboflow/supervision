from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np

from supervision.detection.core import Detections


class JSONSink:
    """
    A utility class for saving detection data to a JSON file. This class is designed to
    efficiently serialize detection objects into a JSON format, allowing for the
    inclusion of bounding box coordinates and additional attributes like `confidence`,
    `class_id`, and `tracker_id`.

    !!! tip

        JSONSink allows passing custom data alongside detection fields, providing
        flexibility for logging various types of information.
        When a list or tuple value in custom_data (or detections.data) has the
        same length as the detection count, each element is written to the
        corresponding detection row; any other value is broadcast to all rows.

    Args:
        file_name: The name of the JSON file where the detections will be stored.
            Defaults to 'output.json'.

    Example:
        ```python
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO("<SOURCE_MODEL_PATH>")
        json_sink = sv.JSONSink(<RESULT_JSON_FILE_PATH>)
        frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

        with json_sink as sink:
            for frame in frames_generator:
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                sink.append(detections, custom_data={"<CUSTOM_LABEL>":"<CUSTOM_DATA>"})
        ```
    """

    def __init__(self, file_name: str = "output.json") -> None:
        """
        Initialize the JSONSink instance.

        Args:
            file_name: The name of the JSON file.
        """
        self.file_name = file_name
        self.file: io.TextIOWrapper | None = None
        self.data: list[dict[str, Any]] = []

    def __enter__(self) -> JSONSink:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        self.write_and_close()

    def open(self) -> None:
        """
        Open the JSON file for writing.
        """
        parent_directory = os.path.dirname(self.file_name)
        if parent_directory and not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        self.file = open(self.file_name, "w")

    def write_and_close(self) -> None:
        """
        Write and close the JSON file.
        """
        if self.file:
            json.dump(self.data, self.file, indent=4)
            self.file.close()

    @staticmethod
    def _slice_value(value: Any, i: int, n: int) -> Any:
        """
        Return the i-th element when the value stores per-detection data.

        Dispatch rules:
            - np.ndarray with ndim == 0: return as-is for broadcasting
            - np.ndarray with ndim >= 1: return value[i]
            - list or tuple with len equal to n: return value[i]
            - any other type: return as-is for broadcasting

        Args:
            value: Custom-data field value.
            i: Zero-based detection index.
            n: Total number of detections.

        Returns:
            Element at position i if value is a per-detection sequence,
            otherwise value unchanged.
        """
        if isinstance(value, np.ndarray):
            return value if value.ndim == 0 else value[i]
        if isinstance(value, (list, tuple)) and len(value) == n:
            return value[i]
        return value

    @staticmethod
    def parse_detection_data(
        detections: Detections, custom_data: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Convert detections and optional custom data into per-detection rows.

        Builds one dictionary per detection containing bounding box coordinates,
        detection attributes, and any values from ``detections.data`` or
        ``custom_data``. List and tuple values in ``custom_data`` with length
        equal to ``len(detections.xyxy)`` are sliced one element per row; all
        other values are broadcast to every row.

        Args:
            detections: Detection data to serialize into row dictionaries.
            custom_data: Optional extra fields to include in each row.

        Returns:
            A list of dictionaries, one per detection, containing ``xyxy``
            coordinates, ``class_id``, ``confidence``, ``tracker_id``, and any
            values from ``detections.data`` or ``custom_data``.
        """
        parsed_rows = []
        n = len(detections.xyxy)
        for i in range(n):
            row = {
                "x_min": float(detections.xyxy[i][0]),
                "y_min": float(detections.xyxy[i][1]),
                "x_max": float(detections.xyxy[i][2]),
                "y_max": float(detections.xyxy[i][3]),
                "class_id": ""
                if detections.class_id is None
                else int(detections.class_id[i]),
                "confidence": ""
                if detections.confidence is None
                else float(detections.confidence[i]),
                "tracker_id": ""
                if detections.tracker_id is None
                else int(detections.tracker_id[i]),
            }

            if hasattr(detections, "data"):
                for key, value in detections.data.items():
                    row[key] = str(JSONSink._slice_value(value, i, n))

            if custom_data:
                for key, value in custom_data.items():
                    v = JSONSink._slice_value(value, i, n)
                    row[key] = str(v) if isinstance(value, np.ndarray) else v

            parsed_rows.append(row)
        return parsed_rows

    def append(
        self, detections: Detections, custom_data: dict[str, Any] | None = None
    ) -> None:
        """
        Append detection data to the JSON file.

        Args:
            detections: The detection data.
            custom_data: Custom data to include. Scalars, dictionaries, and
                other non-sequence values are broadcast to every detection in
                this batch. NumPy arrays, lists, and tuples with length equal
                to ``len(detections)`` are sliced per detection; other lists
                and tuples are broadcast unchanged.
        """
        parsed_rows = JSONSink.parse_detection_data(detections, custom_data)
        self.data.extend(parsed_rows)
