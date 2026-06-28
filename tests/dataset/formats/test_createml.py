"""Tests for CreateML object-detection annotation load/save and conversion helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
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
        ("image_annotations", "class_to_index", "expected_result"),
        [
            pytest.param(
                [],
                {},
                Detections.empty(),
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
                id="duplicate-labels-two-detections-same-id",
            ),
        ],
    )
    def test_converts_annotations(
        self,
        image_annotations: list[dict],
        class_to_index: dict[str, int],
        expected_result: Detections,
    ) -> None:
        """Converts CreateML annotation list to Detections with correct xyxy and ids."""
        result = createml_annotations_to_detections(
            image_annotations=image_annotations, class_to_index=class_to_index
        )
        np.testing.assert_array_almost_equal(result.xyxy, expected_result.xyxy)
        assert (result.class_id is None) == (expected_result.class_id is None)
        if expected_result.class_id is not None:
            np.testing.assert_array_equal(result.class_id, expected_result.class_id)

    @pytest.mark.parametrize(
        ("image_annotations", "class_to_index"),
        [
            pytest.param(
                [{"label": "dog"}],
                {"dog": 0},
                id="missing-coordinates-key",
            ),
            pytest.param(
                [{"coordinates": {"x": 10, "y": 10, "width": 4, "height": 4}}],
                {"dog": 0},
                id="missing-label-key",
            ),
            pytest.param(
                [{"label": "dog", "coordinates": {"x": 10, "y": 10, "width": 4}}],
                {"dog": 0},
                id="missing-coordinate-subkey",
            ),
            pytest.param(
                [{"label": "dog", "coordinates": None}],
                {"dog": 0},
                id="coordinates-is-none",
            ),
        ],
    )
    def test_raises_on_malformed_annotation(
        self,
        image_annotations: list[dict],
        class_to_index: dict[str, int],
    ) -> None:
        """Raises ValueError with 'Malformed' for any missing required field."""
        with pytest.raises(ValueError, match="Malformed"):
            createml_annotations_to_detections(
                image_annotations=image_annotations,
                class_to_index=class_to_index,
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

    @pytest.mark.parametrize(
        ("setup_fn", "match"),
        [
            pytest.param(
                lambda p: ("../evil.jpg", str(p / "images")),
                "outside",
                id="path-traversal",
            ),
            pytest.param(
                lambda p: (str(p.parent / "evil.jpg"), str(p)),
                "outside",
                id="absolute-outside",
            ),
            pytest.param(
                lambda p: (".", str(p)),
                "directory",
                id="resolves-to-images-dir",
            ),
        ],
    )
    def test_raises_on_unsafe_image_path(
        self,
        tmp_path: Path,
        setup_fn: Callable[[Path], tuple[str, str]],
        match: str,
    ) -> None:
        """Raises ValueError for unsafe image path: traversal, absolute, directory."""
        image, images_dir = setup_fn(tmp_path)
        annotations_path = tmp_path / "annotations.json"
        annotations_path.write_text(json.dumps([{"image": image, "annotations": []}]))

        with pytest.raises(ValueError, match=match):
            load_createml_annotations(
                images_directory_path=images_dir,
                annotations_path=str(annotations_path),
            )

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            pytest.param(
                {"image": "a.jpg", "annotations": []},
                "JSON list",
                id="root-is-dict",
            ),
            pytest.param(
                [{"annotations": []}],
                "'image'",
                id="missing-image-key",
            ),
            pytest.param(
                [
                    {"image": "a.jpg", "annotations": []},
                    {"image": "a.jpg", "annotations": []},
                ],
                "duplicate",
                id="duplicate-image-entry",
            ),
        ],
    )
    def test_raises_on_malformed_json(
        self,
        tmp_path: Path,
        payload: dict | list,
        match: str,
    ) -> None:
        """Raises ValueError for malformed JSON: bad root, missing key, duplicate."""
        annotations_path = tmp_path / "annotations.json"
        annotations_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match=match):
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

    @pytest.mark.parametrize(
        ("classes", "xyxy", "class_id", "decimal"),
        [
            pytest.param(
                ["cat", "dog"],
                np.array([[8, 8, 12, 12], [25, 16, 35, 24]], dtype=np.float32),
                np.array([0, 1], dtype=int),
                6,
                id="integer-coords-multi-class",
            ),
            pytest.param(
                ["dog"],
                np.array([[10.3, 7.9, 44.1, 88.6]], dtype=np.float32),
                np.array([0], dtype=int),
                4,
                id="float-coords",
            ),
        ],
    )
    def test_save_load_round_trip(
        self,
        tmp_path: Path,
        classes: list[str],
        xyxy: np.ndarray,
        class_id: np.ndarray,
        decimal: int,
    ) -> None:
        """Save then load preserves class names, image paths, and bounding boxes."""
        images_directory_path = tmp_path / "images"
        annotations_path = tmp_path / "annotations.json"
        image_paths = [str(images_directory_path / "a.jpg")]
        annotations = {image_paths[0]: Detections(xyxy=xyxy, class_id=class_id)}
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
        loaded = loaded_annotations[image_paths[0]]
        np.testing.assert_array_almost_equal(loaded.xyxy, xyxy, decimal=decimal)
        np.testing.assert_array_equal(loaded.class_id, class_id)
