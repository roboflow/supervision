from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from supervision.dataset.core import DetectionDataset
from supervision.dataset.formats.labelme import (
    detections_to_labelme_shapes,
    labelme_shapes_to_detections,
    load_labelme_annotations,
)
from supervision.detection.core import Detections


def _rectangle(label: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "shape_type": "rectangle",
    }


def _polygon(label: str, points: list[list[float]]) -> dict:
    return {"label": label, "points": points, "shape_type": "polygon"}


def _write_labelme(
    path: Path, image_name: str, shapes: list[dict], wh=(64, 48)
) -> None:
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": wh[1],
        "imageWidth": wh[0],
    }
    path.write_text(json.dumps(payload))


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height)).save(path)


def test_labelme_shapes_to_detections_rectangle_box_only() -> None:
    shapes = [_rectangle("dog", 10, 20, 30, 40)]

    result = labelme_shapes_to_detections(
        shapes=shapes,
        class_to_index={"dog": 0},
        resolution_wh=(64, 48),
        with_masks=False,
    )

    np.testing.assert_array_almost_equal(
        result.xyxy, np.array([[10, 20, 30, 40]], dtype=np.float32)
    )
    np.testing.assert_array_equal(result.class_id, np.array([0], dtype=int))
    assert result.mask is None


def test_labelme_shapes_to_detections_rectangle_normalizes_corner_order() -> None:
    # bottom-right given before top-left
    shapes = [_rectangle("dog", 30, 40, 10, 20)]

    result = labelme_shapes_to_detections(
        shapes=shapes,
        class_to_index={"dog": 0},
        resolution_wh=(64, 48),
        with_masks=False,
    )

    np.testing.assert_array_almost_equal(
        result.xyxy, np.array([[10, 20, 30, 40]], dtype=np.float32)
    )


def test_labelme_shapes_to_detections_polygon_builds_mask() -> None:
    shapes = [_polygon("cat", [[10, 10], [30, 10], [30, 30], [10, 30]])]

    result = labelme_shapes_to_detections(
        shapes=shapes,
        class_to_index={"cat": 0},
        resolution_wh=(64, 48),
        with_masks=True,
    )

    np.testing.assert_array_almost_equal(
        result.xyxy, np.array([[10, 10, 30, 30]], dtype=np.float32)
    )
    assert result.mask is not None
    assert result.mask.shape == (1, 48, 64)
    # the square region is filled
    assert result.mask[0, 15:25, 15:25].all()


def test_labelme_shapes_to_detections_empty() -> None:
    result = labelme_shapes_to_detections(
        shapes=[], class_to_index={}, resolution_wh=(64, 48), with_masks=False
    )
    assert len(result) == 0


def test_labelme_shapes_to_detections_skips_unsupported_shape_with_warning() -> None:
    shapes = [
        _rectangle("dog", 10, 20, 30, 40),
        {"label": "x", "points": [[5, 5], [2, 2]], "shape_type": "circle"},
    ]

    with pytest.warns(UserWarning, match="unsupported LabelMe shape"):
        result = labelme_shapes_to_detections(
            shapes=shapes,
            class_to_index={"dog": 0},
            resolution_wh=(64, 48),
            with_masks=False,
        )

    assert len(result) == 1
    np.testing.assert_array_equal(result.class_id, np.array([0], dtype=int))
    np.testing.assert_array_almost_equal(
        result.xyxy, np.array([[10, 20, 30, 40]], dtype=np.float32)
    )


def test_load_labelme_annotations_rectangles(tmp_path: Path) -> None:
    _write_labelme(tmp_path / "a.json", "a.jpg", [_rectangle("dog", 10, 20, 30, 40)])
    _write_labelme(tmp_path / "b.json", "b.jpg", [])

    classes, image_paths, annotations = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    assert classes == ["dog"]
    assert image_paths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    np.testing.assert_array_almost_equal(
        annotations[str(tmp_path / "a.jpg")].xyxy,
        np.array([[10, 20, 30, 40]], dtype=np.float32),
    )
    assert annotations[str(tmp_path / "a.jpg")].mask is None
    assert len(annotations[str(tmp_path / "b.jpg")]) == 0


def test_load_labelme_annotations_polygons_have_masks(tmp_path: Path) -> None:
    _write_labelme(
        tmp_path / "a.json",
        "a.jpg",
        [_polygon("cat", [[10, 10], [30, 10], [30, 30], [10, 30]])],
    )

    _, _, annotations = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    detections = annotations[str(tmp_path / "a.jpg")]
    assert detections.mask is not None
    assert detections.mask.shape == (1, 48, 64)
    assert detections.mask[0, 15:25, 15:25].all()


def test_load_labelme_assigns_global_sorted_class_ids(tmp_path: Path) -> None:
    _write_labelme(tmp_path / "a.json", "a.jpg", [_rectangle("zebra", 1, 1, 5, 5)])
    _write_labelme(tmp_path / "b.json", "b.jpg", [_rectangle("ant", 2, 2, 6, 6)])

    classes, _, annotations = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    assert classes == ["ant", "zebra"]
    np.testing.assert_array_equal(
        annotations[str(tmp_path / "a.jpg")].class_id, np.array([1], dtype=int)
    )
    np.testing.assert_array_equal(
        annotations[str(tmp_path / "b.jpg")].class_id, np.array([0], dtype=int)
    )


def test_load_labelme_resolves_image_by_basename(tmp_path: Path) -> None:
    # The directory portion of imagePath (stored relative to the JSON) is
    # ignored; only the basename is joined to images_directory_path, which
    # neutralizes any annotation-supplied path traversal.
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_labelme(
        tmp_path / "a.json", "../somewhere/a.jpg", [_rectangle("dog", 1, 1, 5, 5)]
    )

    _, image_paths, _ = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(tmp_path),
    )

    assert image_paths == [str(images_dir / "a.jpg")]


def test_load_labelme_rejects_empty_image_path(tmp_path: Path) -> None:
    _write_labelme(tmp_path / "a.json", ".", [])

    with pytest.raises(ValueError, match="imagePath"):
        load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
        )


def test_detections_to_labelme_shapes_rectangle() -> None:
    detections = Detections(
        xyxy=np.array([[10, 20, 30, 40]], dtype=np.float32),
        class_id=np.array([1], dtype=int),
    )

    shapes = detections_to_labelme_shapes(detections=detections, classes=["cat", "dog"])

    assert shapes == [
        {
            "label": "dog",
            "points": [[10.0, 20.0], [30.0, 40.0]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
        }
    ]


def test_detections_to_labelme_shapes_requires_class_id() -> None:
    detections = Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32))

    with pytest.raises(ValueError, match="class_id"):
        detections_to_labelme_shapes(detections=detections, classes=["dog"])


def test_save_load_round_trip_boxes(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    _write_image(images_dir / "a.jpg", 64, 48)
    image_paths = [str(images_dir / "a.jpg")]
    annotations = {
        image_paths[0]: Detections(
            xyxy=np.array([[10, 20, 30, 40]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )
    }
    dataset = DetectionDataset(
        classes=["dog"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))
    classes, _, loaded = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(annotations_dir),
    )

    assert classes == ["dog"]
    loaded_detections = loaded[str(images_dir / "a.jpg")]
    np.testing.assert_array_almost_equal(
        loaded_detections.xyxy, annotations[image_paths[0]].xyxy
    )
    assert loaded_detections.mask is None


def test_save_load_round_trip_masks(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    _write_image(images_dir / "a.jpg", 64, 48)
    mask = np.zeros((1, 48, 64), dtype=bool)
    mask[0, 10:30, 10:30] = True
    image_paths = [str(images_dir / "a.jpg")]
    annotations = {
        image_paths[0]: Detections(
            xyxy=np.array([[10, 10, 30, 30]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )
    }
    dataset = DetectionDataset(
        classes=["cat"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))
    _, _, loaded = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(annotations_dir),
    )

    loaded_detections = loaded[str(images_dir / "a.jpg")]
    assert loaded_detections.mask is not None
    np.testing.assert_array_almost_equal(
        loaded_detections.xyxy,
        np.array([[10, 10, 30, 30]], dtype=np.float32),
        decimal=0,
    )


def test_load_labelme_rejects_dotdot_image_path(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_labelme(tmp_path / "a.json", "..", [])

    with pytest.raises(ValueError, match="imagePath"):
        load_labelme_annotations(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(tmp_path),
        )


def test_load_labelme_force_masks_on_rectangles(tmp_path: Path) -> None:
    _write_labelme(tmp_path / "a.json", "a.jpg", [_rectangle("dog", 10, 10, 30, 30)])

    _, _, annotations = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
        force_masks=True,
    )

    detections = annotations[str(tmp_path / "a.jpg")]
    assert detections.mask is not None
    assert detections.mask.shape == (1, 48, 64)
    assert detections.mask[0, 15:25, 15:25].all()


def test_load_labelme_requires_image_dims_for_masks(tmp_path: Path) -> None:
    payload = {
        "shapes": [_polygon("cat", [[1, 1], [5, 1], [5, 5], [1, 5]])],
        "imagePath": "a.jpg",
    }
    (tmp_path / "a.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="imageWidth"):
        load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
        )


def test_load_labelme_mixed_rectangle_and_polygon_with_masks(tmp_path: Path) -> None:
    _write_labelme(
        tmp_path / "a.json",
        "a.jpg",
        [
            _rectangle("dog", 5, 5, 15, 15),
            _polygon("cat", [[20, 20], [40, 20], [40, 40], [20, 40]]),
        ],
    )

    classes, _, annotations = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    detections = annotations[str(tmp_path / "a.jpg")]
    assert classes == ["cat", "dog"]
    assert len(detections) == 2
    assert detections.mask is not None
    assert detections.mask.shape == (2, 48, 64)
    np.testing.assert_array_almost_equal(
        detections.xyxy,
        np.array([[5, 5, 15, 15], [20, 20, 40, 40]], dtype=np.float32),
    )


def test_load_labelme_duplicate_labels(tmp_path: Path) -> None:
    _write_labelme(
        tmp_path / "a.json",
        "a.jpg",
        [_rectangle("dog", 1, 1, 5, 5), _rectangle("dog", 10, 10, 15, 15)],
    )

    classes, _, annotations = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    assert classes == ["dog"]
    detections = annotations[str(tmp_path / "a.jpg")]
    assert len(detections) == 2
    np.testing.assert_array_equal(detections.class_id, np.array([0, 0], dtype=int))


def test_detections_to_labelme_shapes_multi_component_mask() -> None:
    mask = np.zeros((1, 48, 64), dtype=bool)
    mask[0, 5:15, 5:15] = True
    mask[0, 30:40, 30:40] = True  # second, disconnected blob
    detections = Detections(
        xyxy=np.array([[5, 5, 40, 40]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
    )

    shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

    assert len(shapes) == 2
    assert all(shape["shape_type"] == "polygon" for shape in shapes)
    assert all(shape["label"] == "dog" for shape in shapes)


def test_detections_to_labelme_shapes_empty_mask_falls_back_to_rectangle() -> None:
    mask = np.zeros((1, 48, 64), dtype=bool)  # no usable contour
    detections = Detections(
        xyxy=np.array([[10, 20, 30, 40]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
    )

    shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

    # the detection is preserved as a rectangle, not silently dropped
    assert len(shapes) == 1
    assert shapes[0]["shape_type"] == "rectangle"
    assert shapes[0]["points"] == [[10.0, 20.0], [30.0, 40.0]]


def test_save_load_round_trip_multi_image(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    _write_image(images_dir / "a.jpg", 64, 48)
    _write_image(images_dir / "b.jpg", 64, 48)
    image_paths = [str(images_dir / "a.jpg"), str(images_dir / "b.jpg")]
    annotations = {
        image_paths[0]: Detections(
            xyxy=np.array([[1, 2, 3, 4]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        ),
        image_paths[1]: Detections(
            xyxy=np.array([[5, 6, 7, 8]], dtype=np.float32),
            class_id=np.array([1], dtype=int),
        ),
    }
    dataset = DetectionDataset(
        classes=["cat", "dog"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))
    _, loaded_paths, loaded = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(annotations_dir),
    )

    assert loaded_paths == image_paths
    np.testing.assert_array_almost_equal(
        loaded[image_paths[0]].xyxy, np.array([[1, 2, 3, 4]], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        loaded[image_paths[1]].xyxy, np.array([[5, 6, 7, 8]], dtype=np.float32)
    )


def test_as_labelme_creates_directory_and_writes_envelope(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "nested" / "annotations"  # does not exist yet
    _write_image(images_dir / "a.jpg", 64, 48)
    image_paths = [str(images_dir / "a.jpg")]
    annotations = {
        image_paths[0]: Detections(
            xyxy=np.array([[10, 20, 30, 40]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )
    }
    dataset = DetectionDataset(
        classes=["dog"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))

    output = json.loads((annotations_dir / "a.json").read_text())
    assert output["imagePath"] == "a.jpg"
    assert output["imageWidth"] == 64
    assert output["imageHeight"] == 48
    assert output["version"] == "5.5.0"
    assert output["shapes"][0]["shape_type"] == "rectangle"


def test_save_load_round_trip_float_coordinates(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    _write_image(images_dir / "a.jpg", 64, 48)
    xyxy = np.array([[10.7, 20.3, 30.1, 40.9]], dtype=np.float32)
    image_paths = [str(images_dir / "a.jpg")]
    annotations = {
        image_paths[0]: Detections(xyxy=xyxy, class_id=np.array([0], dtype=int))
    }
    dataset = DetectionDataset(
        classes=["dog"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))
    _, _, loaded = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(annotations_dir),
    )

    np.testing.assert_array_almost_equal(
        loaded[str(images_dir / "a.jpg")].xyxy, xyxy, decimal=4
    )


def test_labelme_shape_missing_label_or_points_raises(tmp_path: Path) -> None:
    _write_labelme(
        tmp_path / "a.json",
        "a.jpg",
        [{"shape_type": "rectangle", "points": [[1, 1], [5, 5]]}],  # no "label"
    )

    with pytest.raises(ValueError, match="missing the required 'label'"):
        load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
        )


def test_from_labelme_returns_detection_dataset(tmp_path: Path) -> None:
    _write_labelme(tmp_path / "a.json", "a.jpg", [_rectangle("dog", 10, 20, 30, 40)])

    dataset = DetectionDataset.from_labelme(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    assert isinstance(dataset, DetectionDataset)
    assert dataset.classes == ["dog"]
    assert len(dataset.image_paths) == 1
    detections = dataset.annotations[str(tmp_path / "a.jpg")]
    np.testing.assert_array_almost_equal(
        detections.xyxy, np.array([[10, 20, 30, 40]], dtype=np.float32)
    )


def test_load_labelme_ignores_non_json_files_in_annotations_dir(tmp_path: Path) -> None:
    _write_labelme(tmp_path / "a.json", "a.jpg", [_rectangle("dog", 1, 1, 5, 5)])
    (tmp_path / "README.txt").write_text("not an annotation")
    (tmp_path / "stray.xml").write_text("<x/>")

    classes, image_paths, _ = load_labelme_annotations(
        images_directory_path=str(tmp_path),
        annotations_directory_path=str(tmp_path),
    )

    assert classes == ["dog"]
    assert image_paths == [str(tmp_path / "a.jpg")]


def test_load_labelme_force_masks_requires_image_dims(tmp_path: Path) -> None:
    payload = {
        "shapes": [_rectangle("dog", 10, 10, 30, 30)],
        "imagePath": "a.jpg",
        # imageWidth / imageHeight intentionally omitted
    }
    (tmp_path / "a.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="imageWidth"):
        load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
            force_masks=True,
        )


def test_detections_to_labelme_shapes_single_pixel_mask_falls_back_to_rectangle() -> (
    None
):
    mask = np.zeros((1, 48, 64), dtype=bool)
    mask[0, 20, 20] = True  # single pixel yields no usable polygon contour
    detections = Detections(
        xyxy=np.array([[20, 20, 21, 21]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
    )

    shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

    assert len(shapes) == 1
    assert shapes[0]["shape_type"] == "rectangle"
    assert shapes[0]["label"] == "dog"


def test_save_load_round_trip_multi_class_id_ordering(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    for name in ["a.jpg", "b.jpg"]:
        _write_image(images_dir / name, 64, 48)
    image_paths = [str(images_dir / "a.jpg"), str(images_dir / "b.jpg")]
    annotations = {
        image_paths[0]: Detections(
            xyxy=np.array([[1, 1, 10, 10], [11, 11, 20, 20]], dtype=np.float32),
            class_id=np.array([0, 1], dtype=int),
        ),
        image_paths[1]: Detections(
            xyxy=np.array([[5, 5, 30, 30]], dtype=np.float32),
            class_id=np.array([2], dtype=int),
        ),
    }
    dataset = DetectionDataset(
        classes=["ant", "cat", "zebra"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))
    classes, _, loaded = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(annotations_dir),
    )

    assert classes == ["ant", "cat", "zebra"]
    np.testing.assert_array_equal(
        loaded[image_paths[0]].class_id, np.array([0, 1], dtype=int)
    )
    np.testing.assert_array_equal(
        loaded[image_paths[1]].class_id, np.array([2], dtype=int)
    )


def test_save_load_round_trip_mask_iou_above_threshold(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    _write_image(images_dir / "a.jpg", 64, 48)
    mask = np.zeros((1, 48, 64), dtype=bool)
    mask[0, 10:30, 10:30] = True
    image_paths = [str(images_dir / "a.jpg")]
    annotations = {
        image_paths[0]: Detections(
            xyxy=np.array([[10, 10, 30, 30]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )
    }
    dataset = DetectionDataset(
        classes=["cat"], images=image_paths, annotations=annotations
    )

    dataset.as_labelme(annotations_directory_path=str(annotations_dir))
    _, _, loaded = load_labelme_annotations(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(annotations_dir),
    )

    loaded_mask = loaded[image_paths[0]].mask
    assert loaded_mask is not None
    original, reloaded = mask[0], loaded_mask[0]
    intersection = float((original & reloaded).sum())
    union = float((original | reloaded).sum())
    iou = intersection / union
    assert iou >= 0.95, f"mask round-trip IoU {iou:.4f} below threshold"
