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


@pytest.mark.parametrize(
    ("image_annotations", "class_to_index", "expected_result", "exception"),
    [
        ([], {}, Detections.empty(), DoesNotRaise()),  # empty annotations
        (
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
        ),  # single centre-based box -> xyxy corners
        (
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
        ),  # multi-class -> distinct class ids
        (
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
        ),  # duplicate labels -> two detections, same id, order preserved
    ],
)
def test_createml_annotations_to_detections(
    image_annotations: list[dict],
    class_to_index: dict[str, int],
    expected_result: Detections,
    exception: Exception,
) -> None:
    with exception:
        result = createml_annotations_to_detections(
            image_annotations=image_annotations, class_to_index=class_to_index
        )
        np.testing.assert_array_almost_equal(result.xyxy, expected_result.xyxy)
        assert (result.class_id is None) == (expected_result.class_id is None)
        if expected_result.class_id is not None:
            np.testing.assert_array_equal(result.class_id, expected_result.class_id)


def test_detections_to_createml_annotations_round_trips_coordinates() -> None:
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


def test_detections_to_createml_annotations_requires_class_id() -> None:
    detections = Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32))

    with pytest.raises(ValueError, match="class_id"):
        detections_to_createml_annotations(detections=detections, classes=["dog"])


def test_load_createml_annotations(tmp_path: Path) -> None:
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


def test_load_createml_annotations_assigns_global_sorted_class_ids(
    tmp_path: Path,
) -> None:
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

    # Classes are globally sorted; ids are consistent across images even though
    # "zebra" appears before "ant" in file order.
    assert classes == ["ant", "zebra"]
    assert image_paths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    np.testing.assert_array_equal(
        annotations[str(tmp_path / "a.jpg")].class_id, np.array([1], dtype=int)
    )
    np.testing.assert_array_equal(
        annotations[str(tmp_path / "b.jpg")].class_id, np.array([0], dtype=int)
    )


def test_load_createml_annotations_rejects_path_traversal(tmp_path: Path) -> None:
    annotations_path = tmp_path / "annotations.json"
    payload = [{"image": "../evil.jpg", "annotations": []}]
    annotations_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="outside"):
        load_createml_annotations(
            images_directory_path=str(tmp_path / "images"),
            annotations_path=str(annotations_path),
        )


def test_load_createml_annotations_rejects_absolute_path(tmp_path: Path) -> None:
    annotations_path = tmp_path / "annotations.json"
    outside = tmp_path.parent / "evil.jpg"
    payload = [{"image": str(outside), "annotations": []}]
    annotations_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="outside"):
        load_createml_annotations(
            images_directory_path=str(tmp_path),
            annotations_path=str(annotations_path),
        )


def test_load_createml_annotations_rejects_images_directory_itself(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "annotations.json"
    payload = [{"image": ".", "annotations": []}]
    annotations_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="directory"):
        load_createml_annotations(
            images_directory_path=str(tmp_path),
            annotations_path=str(annotations_path),
        )


def test_save_createml_annotations_empty_dataset_writes_empty_list(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "nested" / "annotations.json"
    dataset = DetectionDataset(classes=[], images=[], annotations={})

    save_createml_annotations(dataset=dataset, annotations_path=str(annotations_path))

    assert json.loads(annotations_path.read_text()) == []


def test_save_load_round_trip(tmp_path: Path) -> None:
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

    save_createml_annotations(dataset=dataset, annotations_path=str(annotations_path))
    loaded_classes, _, loaded_annotations = load_createml_annotations(
        images_directory_path=str(images_directory_path),
        annotations_path=str(annotations_path),
    )

    assert loaded_classes == classes
    loaded = loaded_annotations[str(images_directory_path / "a.jpg")]
    np.testing.assert_array_almost_equal(loaded.xyxy, annotations[image_paths[0]].xyxy)
    np.testing.assert_array_equal(loaded.class_id, annotations[image_paths[0]].class_id)


def test_save_load_round_trip_float_coordinates(tmp_path: Path) -> None:
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

    save_createml_annotations(dataset=dataset, annotations_path=str(annotations_path))
    _, _, loaded_annotations = load_createml_annotations(
        images_directory_path=str(images_directory_path),
        annotations_path=str(annotations_path),
    )

    loaded = loaded_annotations[str(images_directory_path / "a.jpg")]
    np.testing.assert_array_almost_equal(loaded.xyxy, xyxy, decimal=4)
