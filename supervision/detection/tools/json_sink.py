from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from supervision.detection.core import Detections


class JSONSink:
    """
    A utility class for saving detection data to a JSON file. This class is designed to
    efficiently serialize detection objects into a JSON format, allowing for the
    inclusion of bounding box coordinates and additional attributes like `confidence`,
    `class_id`, and `tracker_id`.

    !!! tip

        JSONsink allow to pass custom data alongside the detection fields, providing
        flexibility for logging various types of information.

    Args:
        file_name (str): The name of the JSON file where the detections will be stored.
            Defaults to 'output.json'.

    Example:
        ```python
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO(<SOURCE_MODEL_PATH>)
        json_sink = sv.JSONSink(<RESULT_JSON_FILE_PATH>)
        frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

        with json_sink as sink:
            for frame in frames_generator:
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                sink.append(detections, custom_data={'<CUSTOM_LABEL>':'<CUSTOM_DATA>'})
        ```
    """  # noqa: E501 // docs

    def __init__(self, file_name: str = "output.json") -> None:
        """
        Initialize the JSONSink instance.

        Args:
            file_name (str): The name of the JSON file.

        Returns:
            None
        """
        self.file_name = file_name
        self.file: Optional[open] = None
        self.data: List[Dict[str, Any]] = []

    def __enter__(self) -> JSONSink:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        self.write_and_close()

    def open(self) -> None:
        """
        Open the JSON file for writing.

        Returns:
            None
        """
        parent_directory = os.path.dirname(self.file_name)
        if parent_directory and not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        self.file = open(self.file_name, "w")

    def write_and_close(self) -> None:
        """
        Write and close the JSON file.

        Returns:
            None
        """
        if self.file:
            json.dump(self.data, self.file, indent=4)
            self.file.close()

    @staticmethod
    def parse_detection_data(
        detections: Detections, custom_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        parsed_rows = []
        for i in range(len(detections.xyxy)):
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
                    row[key] = (
                        str(value[i])
                        if hasattr(value, "__getitem__") and value.ndim != 0
                        else str(value)
                    )

            if custom_data:
                row.update(custom_data)
            parsed_rows.append(row)
        return parsed_rows

    def append(
        self, detections: Detections, custom_data: Dict[str, Any] = None
    ) -> None:
        """
        Append detection data to the JSON file.

        Args:
            detections (Detections): The detection data.
            custom_data (Dict[str, Any]): Custom data to include.

        Returns:
            None
        """
        parsed_rows = JSONSink.parse_detection_data(detections, custom_data)
        self.data.extend(parsed_rows)
