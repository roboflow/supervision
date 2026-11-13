import json
from pathlib import Path

import numpy as np
import pytest

import supervision as sv


class TestJSONSinkSnapshot:
    """Appended rows retain their values when callers reuse mutable inputs."""

    @pytest.mark.parametrize("source", ["custom_data", "detections_data"])
    def test_preserves_array_values_across_buffer_reuse(
        self, tmp_path: Path, source: str
    ) -> None:
        """Reusing an embedding buffer must not rewrite an earlier frame's row."""
        path = tmp_path / "embeddings.json"
        embedding = np.array([[1.0, 2.0]])
        detections = sv.Detections(xyxy=np.array([[0, 0, 10, 10]]))
        custom_data = None
        if source == "detections_data":
            detections.data["embedding"] = embedding
        else:
            custom_data = {"embedding": embedding}

        with sv.JSONSink(str(path)) as sink:
            sink.append(detections, custom_data=custom_data)
            embedding[:] = [9.0, 9.0]
            sink.append(detections, custom_data=custom_data)
        rows = json.loads(path.read_text())

        assert [row["embedding"] for row in rows] == [[1.0, 2.0], [9.0, 9.0]]

    def test_preserves_nested_broadcast_values(self, tmp_path: Path) -> None:
        """Nested mutable metadata is captured before callers update it."""
        path = tmp_path / "metadata.json"
        detections = sv.Detections(xyxy=np.array([[0, 0, 10, 10]]))
        labels = ["first"]
        coordinates = np.array([1, 2])
        metadata = {"labels": labels, "coordinates": coordinates}

        with sv.JSONSink(str(path)) as sink:
            sink.append(detections, custom_data={"metadata": metadata})
            labels[0] = "second"
            coordinates[:] = 9
        rows = json.loads(path.read_text())

        assert rows[0]["metadata"] == {"labels": ["first"], "coordinates": [1, 2]}

    @pytest.mark.parametrize("per_detection", [False, True])
    def test_preserves_mutable_sequence_values(
        self, tmp_path: Path, per_detection: bool
    ) -> None:
        """Both sliced and broadcast lists retain their appended contents."""
        path = tmp_path / "sequences.json"
        detections = sv.Detections(xyxy=np.array([[0, 0, 10, 10]]))
        values = [[1, 2]] if per_detection else [[1, 2], [3, 4]]

        with sv.JSONSink(str(path)) as sink:
            sink.append(detections, custom_data={"values": values})
            values[0][0] = 99
        rows = json.loads(path.read_text())

        expected = [1, 2] if per_detection else [[1, 2], [3, 4]]
        assert rows[0]["values"] == expected
