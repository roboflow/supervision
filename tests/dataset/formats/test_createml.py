"""Tests for CreateML object-detection annotation load/save and conversion helpers."""

from __future__ import annotations

import json
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path

import numpy as np
import pytest

from supervision.dataset.core import DetectionDataset
from supervision.dataset.formats.createml import (
    createml_annotations_to_detections,
    detections_to_createml_annotations,
    load_createml_annotations,
    save_createml_annotations,
)
from supervision.detection.core import Detections


class TestCreatemlAnnotationsToDetections:
    @pytest.mark.parametrize(
        ("image_annotations", "class_to_index", "expected_result", "exception"),
        [
            pytest.param(
                [],
                {},
                Detections.empty(),
                DoesNotRaise(),
                id="empty-annotations",
            ),
            pytest.param(
                [
                    {
                        "label": "dog",
                        "coordinates": {"x": 50, "y": 50, "width": 20, "height": 20},
                    }
                ],
                {"dog": 0},
                Detections(
                    xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
                    class_id=np.array([0], dtype=int),
                ),
                DoesNotRaise(),
                id="single-centre-box-to-xyxy",
            ),
            pytest.param(
                [
                    {
                        "label": "cat",
                        "coordinates": {"x": 10, "y": 10, "width": 4, "height": 4},
                    },
                    {
                        "label": "dog",
                        "coordinates": {"x": 30, "y": 20, "width": 10, "height": 8},
                    },
                ],
                {"cat": 0, "dog": 1},
                Detections(
                    xyxy=np.array([[8, 8, 12, 12], [25, 16, 35, 24]], dtype=np.float32),
                    class_id=np.array([0, 1], dtype=int),
                ),
                DoesNotRaise(),
                id="multi-class-distinct-ids",
            ),
            pytest.param(
                [
                    {
                        "label": "dog",
                        "coordinates": {"x": 10, "y": 10, "width": 4, "height": 4},
                    },
                    {
                        "label": "dog",
                        "coordinates": {"x": 30, "y": 30, "width": 4, "height": 4},
                    },
                ],
                {"dog": 0},
                Detections(
                    xyxy=np.array([[8, 8, 12, 12], [28, 28, 32, 32]], dtype=np.float32),
                    class_id=np.array([0, 0], dtype=int),
                ),
                DoesNotRaise(),
                id="duplicate-labels-two-detections-same-id",
            ),
        ],
    )
    def test_converts_annotations(
        self,
        image_annotations: list[dict],
        class_to_index: dict[str, int],
        expected_result: Detections,
        exception: Exception,
    ) -> None:
        """Converts CreateML annotation list to Detections with correct xyxy and ids."""
        with exception:
            result = createml_annotations_to_detections(
                image_annotations=image_annotations, class_to_index=class_to_index
            )
            np.testing.assert_array_almost_equal(result.xyxy, expected_result.xyxy)
            assert (result.class_id is None) == (expected_result.class_id is None)
            if expected_result.class_id is not None:
                np.testing.assert_array_equal(result.class_id, expected_result.class_id)

    def test_raises_on_missing_coordinates_key(self) -> None:
        """Raises ValueError when an annotation entry lacks the 'coordinates' key."""
        with pytest.raises(ValueError, match="Malformed"):
            createml_annotations_to_detections(
                image_annotations=[{"label": "dog"}],
                class_to_index={"dog": 0},
            )

    def test_raises_on_missing_label_key(self) -> None:
        """Raises ValueError when an annotation entry lacks the 'label' key."""
        with pytest.raises(ValueError, match="Malformed"):
            createml_annotations_to_detections(
                image_annotations=[
                    {"coordinates": {"x": 10, "y": 10, "width": 4, "height": 4}}
                ],
                class_to_index={"dog": 0},
            )

    def test_raises_on_missing_coordinate_subkey(self) -> None:
        """Raises ValueError when a coordinates dict is missing a required sub-key."""
        with pytest.raises(ValueError, match="Malformed"):
            createml_annotations_to_detections(
                image_annotations=[
                    {"label": "dog", "coordinates": {"x": 10, "y": 10, "width": 4}}
                ],
                class_to_index={"dog": 0},
            )

    def test_raises_when_coordinates_is_none(self) -> None:
        """Raises ValueError when the coordinates value is None."""
        with pytest.raises(ValueError, match="Malformed"):
            createml_annotations_to_detections(
                image_annotations=[{"label": "dog", "coordinates": None}],
                class_to_index={"dog": 0},
            )


class TestDetectionsToCreatemlAnnotations:
    def test_round_trips_coordinates(self) -> None:
        """Round-trip: xyxy corners convert to CreateML centre+wh and back correctly."""
        detections = Detections(
            xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
            class_id=np.array([1], dtype=int),
        )

        result = detections_to_createml_annotations(
            detections=detections, classes=["cat", "dog"]
        )

        assert result == [
            {
                "label": "dog",
                "coordinates": {"x": 50.0, "y": 50.0, "width": 20.0, "height": 20.0},
            }
        ]

    def test_raises_when_class_id_is_none(self) -> None:
        """Raises ValueError when Detections.class_id is None."""
        detections = Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32))

        with pytest.raises(ValueError, match="class_id"):
            detections_to_createml_annotations(detections=detections, classes=["dog"])


class TestLoadCreatemlAnnotations:
    def test_loads_basic_annotations(self, tmp_path: Path) -> None:
        """Loads classes, image_paths, and Detections from a valid CreateML file."""
        annotations_path = tmp_path / "annotations.json"
        payload = [
            {
                "image": "a.jpg",
                "annotations": [
                    {
                        "label": "dog",
                        "coordinates": {"x": 50, "y": 50, "width": 20, "height": 20},
                    }
                ],
            },
            {"image": "b.jpg", "annotations": []},
        ]
        annotations_path.write_text(json.dumps(payload))

        classes, image_paths, annotations = load_createml_annotations(
            images_directory_path=str(tmp_path),
            annotations_path=str(annotations_path),
        )

        assert classes == ["dog"]
        assert image_paths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
        detections = annotations[str(tmp_path / "a.jpg")]
        np.testing.assert_array_almost_equal(
            detections.xyxy, np.array([[40, 40, 60, 60]], dtype=np.float32)
        )
        np.testing.assert_array_equal(detections.class_id, np.array([0], dtype=int))
        assert len(annotations[str(tmp_path / "b.jpg")]) == 0

    def test_assigns_global_sorted_class_ids(self, tmp_path: Path) -> None:
        """Class ids are globally sorted regardless of per-image label order."""
        annotations_path = tmp_path / "annotations.json"
        payload = [
            {
                "image": "a.jpg",
                "annotations": [
                    {
                        "label": "zebra",
                        "coordinates": {"x": 10, "y": 10, "width": 4, "height": 4},
                    }
                ],
            },
            {
                "image": "b.jpg",
                "annotations": [
                    {
                        "label": "ant",
                        "coordinates": {"x": 20, "y": 20, "width": 6, "height": 6},
                    }
                ],
            },
        ]
        annotations_path.write_text(json.dumps(payload))

        classes, image_paths, annotations = load_createml_annotations(
            images_directory_path=str(tmp_path),
            annotations_path=str(annotations_path),
        )

        assert classes == ["ant", "zebra"]
        assert image_paths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
        np.testing.assert_array_equal(
            annotations[str(tmp_path / "a.jpg")].class_id, np.array([1], dtype=int)
        )
        np.testing.assert_array_equal(
            annotations[str(tmp_path / "b.jpg")].class_id, np.array([0], dtype=int)
        )

    def test_raises_on_path_traversal(self, tmp_path: Path) -> None:
        """Raises ValueError when 'image' field attempts directory traversal."""
        annotations_path = tmp_path / "annotations.json"
        payload = [{"image": "../evil.jpg", "annotations": []}]
        annotations_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="outside"):
            load_createml_annotations(
                images_directory_path=str(tmp_path / "images"),
                annotations_path=str(annotations_path),
            )

    def test_raises_on_absolute_path(self, tmp_path: Path) -> None:
        """Raises ValueError when 'image' is an absolute path outside images dir."""
        annotations_path = tmp_path / "annotations.json"
        outside = tmp_path.parent / "evil.jpg"
        payload = [{"image": str(outside), "annotations": []}]
        annotations_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="outside"):
            load_createml_annotations(
                images_directory_path=str(tmp_path),
                annotations_path=str(annotations_path),
            )

    def test_raises_when_image_is_the_directory_itself(self, tmp_path: Path) -> None:
        """Raises ValueError when 'image' resolves to the images directory itself."""
        annotations_path = tmp_path / "annotations.json"
        payload = [{"image": ".", "annotations": []}]
        annotations_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="directory"):
            load_createml_annotations(
                images_directory_path=str(tmp_path),
                annotations_path=str(annotations_path),
            )

    def test_raises_when_json_root_is_dict(self, tmp_path: Path) -> None:
        """Raises ValueError when the JSON root is a dict instead of a list."""
        annotations_path = tmp_path / "annotations.json"
        annotations_path.write_text(json.dumps({"image": "a.jpg", "annotations": []}))

        with pytest.raises(ValueError, match="JSON list"):
            load_createml_annotations(
                images_directory_path=str(tmp_path),
                annotations_path=str(annotations_path),
            )

    def test_raises_on_missing_image_key(self, tmp_path: Path) -> None:
        """Raises ValueError when an entry lacks the required 'image' key."""
        annotations_path = tmp_path / "annotations.json"
        annotations_path.write_text(json.dumps([{"annotations": []}]))

        with pytest.raises(ValueError, match="'image'"):
            load_createml_annotations(
                images_directory_path=str(tmp_path),
                annotations_path=str(annotations_path),
            )

    def test_raises_on_duplicate_image_entry(self, tmp_path: Path) -> None:
        """Raises ValueError when the same image filename appears more than once."""
        annotations_path = tmp_path / "annotations.json"
        payload = [
            {"image": "a.jpg", "annotations": []},
            {"image": "a.jpg", "annotations": []},
        ]
        annotations_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="duplicate"):
            load_createml_annotations(
                images_directory_path=str(tmp_path),
                annotations_path=str(annotations_path),
            )


class TestSaveCreatemlAnnotations:
    def test_empty_dataset_writes_empty_list(self, tmp_path: Path) -> None:
        """Empty dataset serialises to an empty JSON array."""
        annotations_path = tmp_path / "nested" / "annotations.json"
        dataset = DetectionDataset(classes=[], images=[], annotations={})

        save_createml_annotations(
            dataset=dataset, annotations_path=str(annotations_path)
        )

        assert json.loads(annotations_path.read_text()) == []

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Save then load preserves class names, image paths, and bounding boxes."""
        images_directory_path = tmp_path / "images"
        annotations_path = tmp_path / "annotations.json"
        classes = ["cat", "dog"]
        image_paths = [str(images_directory_path / "a.jpg")]
        annotations = {
            image_paths[0]: Detections(
                xyxy=np.array([[8, 8, 12, 12], [25, 16, 35, 24]], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int),
            )
        }
        dataset = DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )

        save_createml_annotations(
            dataset=dataset, annotations_path=str(annotations_path)
        )
        loaded_classes, _, loaded_annotations = load_createml_annotations(
            images_directory_path=str(images_directory_path),
            annotations_path=str(annotations_path),
        )

        assert loaded_classes == classes
        loaded = loaded_annotations[str(images_directory_path / "a.jpg")]
        np.testing.assert_array_almost_equal(
            loaded.xyxy, annotations[image_paths[0]].xyxy
        )
        np.testing.assert_array_equal(
            loaded.class_id, annotations[image_paths[0]].class_id
        )

    def test_save_load_round_trip_float_coordinates(self, tmp_path: Path) -> None:
        """Float32 coordinates survive a save/load cycle within float32 precision."""
        images_directory_path = tmp_path / "images"
        annotations_path = tmp_path / "annotations.json"
        xyxy = np.array([[10.3, 7.9, 44.1, 88.6]], dtype=np.float32)
        image_paths = [str(images_directory_path / "a.jpg")]
        annotations = {
            image_paths[0]: Detections(xyxy=xyxy, class_id=np.array([0], dtype=int))
        }
        dataset = DetectionDataset(
            classes=["dog"], images=image_paths, annotations=annotations
        )

        save_createml_annotations(
            dataset=dataset, annotations_path=str(annotations_path)
        )
        _, _, loaded_annotations = load_createml_annotations(
            images_directory_path=str(images_directory_path),
            annotations_path=str(annotations_path),
        )

        loaded = loaded_annotations[str(images_directory_path / "a.jpg")]
        np.testing.assert_array_almost_equal(loaded.xyxy, xyxy, decimal=4)
