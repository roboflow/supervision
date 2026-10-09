"""Tests for the LabelMe dataset format loader and exporter."""

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
    save_labelme_annotations,
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


class TestLabelmeShapesToDetections:
    """Unit tests for ``labelme_shapes_to_detections``."""

    @pytest.mark.parametrize(
        ("points", "expected_xyxy"),
        [
            pytest.param([[10, 20], [30, 40]], [[10, 20, 30, 40]], id="top-left-first"),
            pytest.param(
                [[30, 40], [10, 20]], [[10, 20, 30, 40]], id="bottom-right-first"
            ),
        ],
    )
    def test_rectangle_normalizes_to_xyxy(
        self, points: list[list[float]], expected_xyxy: list[list[float]]
    ) -> None:
        """Rectangle normalises to (x_min, y_min, x_max, y_max) regardless of order."""
        shapes = [{"label": "dog", "points": points, "shape_type": "rectangle"}]

        result = labelme_shapes_to_detections(
            shapes=shapes,
            class_to_index={"dog": 0},
            resolution_wh=(64, 48),
            with_masks=False,
        )

        np.testing.assert_array_almost_equal(
            result.xyxy, np.array(expected_xyxy, dtype=np.float32)
        )
        np.testing.assert_array_equal(result.class_id, np.array([0], dtype=int))
        assert result.mask is None

    def test_polygon_builds_mask(self) -> None:
        """Polygon shapes produce a binary mask rasterised at the given resolution."""
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
        assert result.mask[0, 15:25, 15:25].all()

    @pytest.mark.parametrize("group_id", [0, 7])
    @pytest.mark.parametrize("with_masks", [False, True])
    @pytest.mark.parametrize(
        "shapes",
        [
            pytest.param(
                [
                    _polygon("dog", [[5, 5], [15, 5], [15, 15], [5, 15]]),
                    _polygon("dog", [[30, 30], [40, 30], [40, 40], [30, 40]]),
                ],
                id="polygons",
            ),
            pytest.param(
                [_rectangle("dog", 5, 5, 15, 15), _rectangle("dog", 30, 30, 40, 40)],
                id="rectangles",
            ),
            pytest.param(
                [
                    _rectangle("dog", 5, 5, 15, 15),
                    _polygon("dog", [[30, 30], [40, 30], [40, 40], [30, 40]]),
                ],
                id="mixed-shapes",
            ),
        ],
    )
    def test_explicit_group_combines_shapes(
        self, shapes: list[dict], with_masks: bool, group_id: int
    ) -> None:
        """Shapes sharing a label and group form one box and a union mask."""
        shapes = [dict(shape, group_id=group_id) for shape in shapes]
        expected_mask = np.zeros((1, 48, 64), dtype=bool)
        expected_mask[0, 5:16, 5:16] = True
        expected_mask[0, 30:41, 30:41] = True

        result = labelme_shapes_to_detections(
            shapes=shapes,
            class_to_index={"dog": 0},
            resolution_wh=(64, 48),
            with_masks=with_masks,
        )

        assert len(result) == 1
        np.testing.assert_array_equal(result.xyxy, [[5, 5, 40, 40]])
        np.testing.assert_array_equal(result.class_id, [0])
        assert result.xyxy.dtype == np.float32
        if with_masks:
            assert result.mask is not None
            assert result.mask.dtype == bool
            np.testing.assert_array_equal(result.mask, expected_mask)
        else:
            assert result.mask is None

    def test_interleaved_groups_combine_in_first_seen_order(self) -> None:
        """Interleaved grouped shapes retain first-seen output order and union masks."""
        shapes = [
            dict(
                _polygon("dog", [[5, 5], [15, 5], [15, 15], [5, 15]]),
                group_id=7,
            ),
            dict(
                _polygon("cat", [[20, 20], [25, 20], [25, 25], [20, 25]]),
                group_id=11,
            ),
            dict(
                _polygon("dog", [[30, 30], [40, 30], [40, 40], [30, 40]]),
                group_id=7,
            ),
        ]
        expected_masks = np.zeros((2, 48, 64), dtype=bool)
        expected_masks[0, 5:16, 5:16] = True
        expected_masks[0, 30:41, 30:41] = True
        expected_masks[1, 20:26, 20:26] = True

        result = labelme_shapes_to_detections(
            shapes=shapes,
            class_to_index={"dog": 0, "cat": 1},
            resolution_wh=(64, 48),
            with_masks=True,
        )

        assert len(result) == 2
        np.testing.assert_array_equal(result.xyxy, [[5, 5, 40, 40], [20, 20, 25, 25]])
        np.testing.assert_array_equal(result.class_id, [0, 1])
        np.testing.assert_array_equal(result.mask, expected_masks)

    @pytest.mark.parametrize(
        ("first_fields", "second_fields", "expected_class_ids"),
        [
            pytest.param({}, {}, [0, 0], id="missing-groups"),
            pytest.param(
                {"group_id": None}, {"group_id": None}, [0, 0], id="null-groups"
            ),
            pytest.param(
                {"group_id": 0}, {"group_id": 1}, [0, 0], id="different-groups"
            ),
            pytest.param(
                {"group_id": 0},
                {"group_id": 0, "label": "cat"},
                [0, 1],
                id="different-labels",
            ),
            pytest.param({"group_id": 0}, {}, [0, 0], id="grouped-and-ungrouped"),
        ],
    )
    def test_keeps_distinct_instances(
        self,
        first_fields: dict,
        second_fields: dict,
        expected_class_ids: list[int],
    ) -> None:
        """Ungrouped shapes, distinct groups, and distinct labels stay independent."""
        shapes = [
            dict(_polygon("dog", [[5, 5], [15, 5], [15, 15], [5, 15]]), **first_fields),
            dict(
                _polygon("dog", [[30, 30], [40, 30], [40, 40], [30, 40]]),
                **second_fields,
            ),
        ]
        expected_masks = np.zeros((2, 48, 64), dtype=bool)
        expected_masks[0, 5:16, 5:16] = True
        expected_masks[1, 30:41, 30:41] = True

        result = labelme_shapes_to_detections(
            shapes=shapes,
            class_to_index={"dog": 0, "cat": 1},
            resolution_wh=(64, 48),
            with_masks=True,
        )

        assert len(result) == 2
        np.testing.assert_array_equal(result.xyxy, [[5, 5, 15, 15], [30, 30, 40, 40]])
        np.testing.assert_array_equal(result.class_id, expected_class_ids)
        np.testing.assert_array_equal(result.mask, expected_masks)

    def test_empty_shapes(self) -> None:
        """Empty shape list returns an empty Detections instance."""
        result = labelme_shapes_to_detections(
            shapes=[], class_to_index={}, resolution_wh=(64, 48), with_masks=False
        )
        assert len(result) == 0

    def test_skips_unsupported_shape_type_with_warning(self) -> None:
        """Unsupported shape types are skipped and a UserWarning is emitted."""
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


class TestLoadLabelmeAnnotations:
    """Unit tests for ``load_labelme_annotations``."""

    def test_rectangles_loaded_as_boxes(self, tmp_path: Path) -> None:
        """Rectangle shapes load as xyxy boxes with no mask."""
        _write_labelme(
            tmp_path / "a.json", "a.jpg", [_rectangle("dog", 10, 20, 30, 40)]
        )
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

    def test_polygons_loaded_with_masks(self, tmp_path: Path) -> None:
        """Polygon shapes load with a rasterised binary mask."""
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

    def test_assigns_global_sorted_class_ids(self, tmp_path: Path) -> None:
        """Class IDs are assigned by sorted label order across all annotation files."""
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

    def test_resolves_image_by_basename(self, tmp_path: Path) -> None:
        """Directory portion of imagePath is stripped; only the basename is used."""
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

    @pytest.mark.parametrize(
        "image_path_value",
        [
            pytest.param("..\\images\\a.jpg", id="relative"),
            pytest.param("C:\\data\\images\\a.jpg", id="absolute-with-drive"),
        ],
    )
    def test_loads_image_referenced_by_windows_style_image_path(
        self, tmp_path: Path, image_path_value: str
    ) -> None:
        """A backslash-separated imagePath, as LabelMe saves on Windows, loads."""
        images_dir = tmp_path / "images"
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()
        _write_image(images_dir / "a.jpg", 64, 48)
        _write_labelme(
            annotations_dir / "a.json",
            image_path_value,
            [_rectangle("dog", 1, 1, 5, 5)],
        )

        dataset = DetectionDataset.from_labelme(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(annotations_dir),
        )
        image_path, image, _ = dataset[0]

        assert image_path == str(images_dir / "a.jpg")
        assert image.shape == (48, 64, 3)

    @pytest.mark.parametrize(
        "image_path_value",
        [
            pytest.param(".", id="dot"),
            pytest.param("..", id="dotdot"),
        ],
    )
    def test_rejects_invalid_image_path(
        self, tmp_path: Path, image_path_value: str
    ) -> None:
        """Dot and dotdot imagePath values raise ValueError."""
        _write_labelme(tmp_path / "a.json", image_path_value, [])

        with pytest.raises(ValueError, match="imagePath"):
            load_labelme_annotations(
                images_directory_path=str(tmp_path),
                annotations_directory_path=str(tmp_path),
            )

    def test_missing_image_path_key_raises_value_error(self, tmp_path: Path) -> None:
        """Annotation JSON missing imagePath key raises ValueError, not KeyError."""
        (tmp_path / "a.json").write_text('{"shapes": []}')

        with pytest.raises(ValueError, match="imagePath"):
            load_labelme_annotations(
                images_directory_path=str(tmp_path),
                annotations_directory_path=str(tmp_path),
            )

    def test_duplicate_image_basename_raises_value_error(self, tmp_path: Path) -> None:
        """Two annotation files with same image basename raise ValueError."""
        _write_labelme(
            tmp_path / "a.json", "subdir1/image.jpg", [_rectangle("dog", 1, 1, 5, 5)]
        )
        _write_labelme(
            tmp_path / "b.json", "subdir2/image.jpg", [_rectangle("cat", 2, 2, 6, 6)]
        )

        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            load_labelme_annotations(
                images_directory_path=str(tmp_path),
                annotations_directory_path=str(tmp_path),
            )

    def test_force_masks_on_rectangles(self, tmp_path: Path) -> None:
        """force_masks=True produces masks for rectangle shapes via polygon fill."""
        _write_labelme(
            tmp_path / "a.json", "a.jpg", [_rectangle("dog", 10, 10, 30, 30)]
        )

        _, _, annotations = load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
            force_masks=True,
        )

        detections = annotations[str(tmp_path / "a.jpg")]
        assert detections.mask is not None
        assert detections.mask.shape == (1, 48, 64)
        assert detections.mask[0, 15:25, 15:25].all()

    @pytest.mark.parametrize(
        ("shapes", "force_masks"),
        [
            pytest.param(
                [_polygon("cat", [[1, 1], [5, 1], [5, 5], [1, 5]])],
                False,
                id="polygon-shape",
            ),
            pytest.param(
                [_rectangle("dog", 10, 10, 30, 30)],
                True,
                id="force-masks-rectangle",
            ),
        ],
    )
    def test_requires_image_dims_when_mask_needed(
        self, tmp_path: Path, shapes: list[dict], force_masks: bool
    ) -> None:
        """ValueError raised when mask rasterisation required but image dims missing."""
        payload = {"shapes": shapes, "imagePath": "a.jpg"}
        (tmp_path / "a.json").write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="imageWidth"):
            load_labelme_annotations(
                images_directory_path=str(tmp_path),
                annotations_directory_path=str(tmp_path),
                force_masks=force_masks,
            )

    def test_mixed_rectangle_and_polygon(self, tmp_path: Path) -> None:
        """Mixed rectangle and polygon shapes in one file both load with masks."""
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

    def test_duplicate_labels_merge_to_single_class(self, tmp_path: Path) -> None:
        """Multiple shapes with the same label map to a single class entry."""
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

    def test_ignores_non_json_files_in_annotations_dir(self, tmp_path: Path) -> None:
        """Non-JSON files in the annotations directory are silently ignored."""
        _write_labelme(tmp_path / "a.json", "a.jpg", [_rectangle("dog", 1, 1, 5, 5)])
        (tmp_path / "README.txt").write_text("not an annotation")
        (tmp_path / "stray.xml").write_text("<x/>")

        classes, image_paths, _ = load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
        )

        assert classes == ["dog"]
        assert image_paths == [str(tmp_path / "a.jpg")]

    def test_missing_label_or_points_raises(self, tmp_path: Path) -> None:
        """Shape missing required label field raises ValueError during load."""
        _write_labelme(
            tmp_path / "a.json",
            "a.jpg",
            [{"shape_type": "rectangle", "points": [[1, 1], [5, 5]]}],
        )

        with pytest.raises(ValueError, match="missing the required 'label'"):
            load_labelme_annotations(
                images_directory_path=str(tmp_path),
                annotations_directory_path=str(tmp_path),
            )

    def test_annotation_path_traversal_is_stripped(self, tmp_path: Path) -> None:
        """Annotation-driven path traversal is neutralised: only basename is used."""
        _write_labelme(
            tmp_path / "evil.json",
            "../../../evil.jpg",
            [_rectangle("dog", 0, 0, 10, 10)],
        )

        classes, image_paths, _ = load_labelme_annotations(
            images_directory_path=str(tmp_path),
            annotations_directory_path=str(tmp_path),
        )

        assert classes == ["dog"]
        assert len(image_paths) == 1
        assert image_paths[0] == str(tmp_path / "evil.jpg")


class TestDetectionsToLabelmeShapes:
    """Unit tests for ``detections_to_labelme_shapes``."""

    def test_box_only_exports_rectangle(self) -> None:
        """Box-only detection is exported as a rectangle shape."""
        detections = Detections(
            xyxy=np.array([[10, 20, 30, 40]], dtype=np.float32),
            class_id=np.array([1], dtype=int),
        )

        shapes = detections_to_labelme_shapes(
            detections=detections, classes=["cat", "dog"]
        )

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

    def test_no_class_id_raises(self) -> None:
        """Detections without class_id raises ValueError."""
        detections = Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32))

        with pytest.raises(ValueError, match="class_id"):
            detections_to_labelme_shapes(detections=detections, classes=["dog"])

    @pytest.mark.parametrize(
        "class_id",
        [pytest.param(-1, id="minus-one"), pytest.param(-99, id="large-negative")],
    )
    def test_negative_class_id_raises(self, class_id: int) -> None:
        """Negative class_id must raise ValueError, not wrap via Python indexing."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([class_id], dtype=int),
        )

        with pytest.raises(ValueError, match="class_id"):
            detections_to_labelme_shapes(detections=detections, classes=["dog"])

    def test_out_of_range_class_id_raises(self) -> None:
        """class_id exceeding classes length raises ValueError."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([5], dtype=int),
        )

        with pytest.raises(ValueError, match="out of range"):
            detections_to_labelme_shapes(detections=detections, classes=["dog"])

    def test_multi_component_mask(self) -> None:
        """Disconnected mask regions export as polygons sharing one explicit group."""
        mask = np.zeros((1, 48, 64), dtype=bool)
        mask[0, 5:15, 5:15] = True
        mask[0, 30:40, 30:40] = True
        detections = Detections(
            xyxy=np.array([[5, 5, 40, 40]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )

        shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

        assert len(shapes) == 2
        assert all(shape["shape_type"] == "polygon" for shape in shapes)
        assert all(shape["label"] == "dog" for shape in shapes)
        assert shapes[0]["group_id"] is not None
        assert shapes[0]["group_id"] == shapes[1]["group_id"]

    def test_mask_with_hole_exports_single_polygon(self) -> None:
        """A hole is spliced into its outer contour instead of becoming a shape."""
        mask = np.zeros((1, 48, 64), dtype=bool)
        mask[0, 5:40, 5:40] = True
        mask[0, 15:30, 15:30] = False
        detections = Detections(
            xyxy=np.array([[5, 5, 39, 39]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )

        shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

        assert len(shapes) == 1
        assert shapes[0]["group_id"] is None

    def test_empty_mask_falls_back_to_rectangle(self) -> None:
        """All-zero mask falls back to rectangle; detection is not silently dropped."""
        mask = np.zeros((1, 48, 64), dtype=bool)
        detections = Detections(
            xyxy=np.array([[10, 20, 30, 40]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )

        shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

        assert len(shapes) == 1
        assert shapes[0]["shape_type"] == "rectangle"
        assert shapes[0]["points"] == [[10.0, 20.0], [30.0, 40.0]]

    def test_single_pixel_mask_falls_back_to_rectangle(self) -> None:
        """Single-pixel mask yields no polygon contour and falls back to rectangle."""
        mask = np.zeros((1, 48, 64), dtype=bool)
        mask[0, 20, 20] = True
        detections = Detections(
            xyxy=np.array([[20, 20, 21, 21]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )

        shapes = detections_to_labelme_shapes(detections=detections, classes=["dog"])

        assert len(shapes) == 1
        assert shapes[0]["shape_type"] == "rectangle"
        assert shapes[0]["label"] == "dog"


class TestFromLabelme:
    """Integration tests for ``DetectionDataset.from_labelme``."""

    def test_returns_detection_dataset(self, tmp_path: Path) -> None:
        """from_labelme returns DetectionDataset with classes and annotations."""
        _write_labelme(
            tmp_path / "a.json", "a.jpg", [_rectangle("dog", 10, 20, 30, 40)]
        )

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


class TestAsLabelmeRoundTrip:
    """Save-load round-trip tests for ``DetectionDataset.as_labelme``."""

    def test_boxes_round_trip(self, tmp_path: Path) -> None:
        """Box-only detections survive a save-load cycle with exact coordinates."""
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

    @pytest.mark.parametrize("instance_count", [1, 2])
    def test_disconnected_masks_round_trip(
        self, tmp_path: Path, instance_count: int
    ) -> None:
        """Disconnected same-class masks retain instance membership and geometry."""
        images_dir = tmp_path / "images"
        annotations_dir = tmp_path / "annotations"
        _write_image(images_dir / "a.jpg", 64, 48)
        image_path = str(images_dir / "a.jpg")
        masks = np.zeros((instance_count, 48, 64), dtype=bool)
        for index in range(instance_count):
            offset = index * 20
            masks[index, 5:15, 5 + offset : 15 + offset] = True
            masks[index, 30:40, 30 + offset : 40 + offset] = True
        xyxy = np.array([[5, 5, 39, 39], [25, 5, 59, 39]], dtype=np.float32)
        detections = Detections(
            xyxy=xyxy[:instance_count],
            mask=masks,
            class_id=np.zeros(instance_count, dtype=int),
        )
        dataset = DetectionDataset(
            classes=["dog"], images=[image_path], annotations={image_path: detections}
        )

        dataset.as_labelme(annotations_directory_path=str(annotations_dir))
        loaded = DetectionDataset.from_labelme(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(annotations_dir),
        )

        result = loaded.annotations[image_path]
        assert len(result) == instance_count
        np.testing.assert_array_equal(result.class_id, detections.class_id)
        np.testing.assert_array_equal(result.xyxy, detections.xyxy)
        np.testing.assert_array_equal(result.mask, masks)

    def test_mask_with_hole_round_trip(self, tmp_path: Path) -> None:
        """A ring mask reloads as one instance with its hole intact."""
        images_dir = tmp_path / "images"
        annotations_dir = tmp_path / "annotations"
        _write_image(images_dir / "a.jpg", 64, 48)
        image_path = str(images_dir / "a.jpg")
        mask = np.zeros((1, 48, 64), dtype=bool)
        mask[0, 5:40, 5:40] = True
        mask[0, 15:30, 15:30] = False
        detections = Detections(
            xyxy=np.array([[5, 5, 39, 39]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )
        dataset = DetectionDataset(
            classes=["dog"], images=[image_path], annotations={image_path: detections}
        )

        dataset.as_labelme(annotations_directory_path=str(annotations_dir))
        loaded = DetectionDataset.from_labelme(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(annotations_dir),
        )

        result = loaded.annotations[image_path]
        assert len(result) == 1
        np.testing.assert_array_equal(result.mask, mask)

    def test_masks_round_trip(self, tmp_path: Path) -> None:
        """Masked detections survive a save-load cycle with approximate coordinates."""
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

    def test_multi_image_round_trip(self, tmp_path: Path) -> None:
        """Multiple images with different detections all survive a save-load cycle."""
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

    def test_creates_directory_and_writes_envelope(self, tmp_path: Path) -> None:
        """as_labelme creates the output directory and writes correct JSON envelope."""
        images_dir = tmp_path / "images"
        annotations_dir = tmp_path / "nested" / "annotations"
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

    def test_float_coordinates_round_trip(self, tmp_path: Path) -> None:
        """Sub-pixel float coordinates are preserved across a save-load cycle."""
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

    def test_multi_class_id_ordering(self, tmp_path: Path) -> None:
        """Class IDs are preserved correctly across a multi-class save-load cycle."""
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

    def test_mask_iou_above_threshold(self, tmp_path: Path) -> None:
        """Mask round-trip preserves mask with IoU >= 0.95."""
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


class TestSaveLabelmeAnnotations:
    """LabelMe export must reject same-stem images before writing."""

    def test_raises_on_duplicate_image_stems(self, tmp_path: Path) -> None:
        """Images sharing a stem would overwrite each other's .json and are rejected."""
        image_paths = ["dir_a/img.jpg", "dir_b/img.jpg"]
        dataset = DetectionDataset(
            classes=["object"],
            images=image_paths,
            annotations={path: Detections.empty() for path in image_paths},
        )

        with pytest.raises(ValueError, match="LabelMe annotation file"):
            save_labelme_annotations(
                dataset=dataset,
                annotations_directory_path=str(tmp_path / "annotations"),
            )


class TestDetectionDatasetLabelmeImageShapes:
    """LabelMe exports preserve dimensions and geometry for each image shape."""

    @pytest.mark.parametrize(
        "image_shape",
        [
            pytest.param((16, 24), id="grayscale"),
            pytest.param((16, 24, 3), id="color"),
        ],
    )
    @pytest.mark.parametrize("with_mask", [False, True])
    def test_preserves_dimensions_geometry_and_source_image(
        self, tmp_path: Path, image_shape: tuple[int, ...], with_mask: bool
    ) -> None:
        """Grayscale and color images export boxes and masks without pixel changes."""
        image = np.arange(np.prod(image_shape), dtype=np.uint8).reshape(image_shape)
        original_image = image.copy()
        mask = np.zeros((1, 16, 24), dtype=bool)
        mask[0, 2:11, 3:16] = True
        detections = Detections(
            xyxy=np.array([[3, 2, 15, 10]], dtype=np.float32),
            class_id=np.array([0]),
            mask=mask if with_mask else None,
        )
        dataset = DetectionDataset(
            classes=["object"],
            images={"image.png": image},
            annotations={"image.png": detections},
        )

        dataset.as_labelme(annotations_directory_path=str(tmp_path))

        output = json.loads((tmp_path / "image.json").read_text())
        assert output["imageWidth"] == 24
        assert output["imageHeight"] == 16
        assert len(output["shapes"]) == 1
        shape = output["shapes"][0]
        assert shape["label"] == "object"
        if with_mask:
            assert shape["shape_type"] == "polygon"
            assert sorted(shape["points"]) == [[3, 2], [3, 10], [15, 2], [15, 10]]
        else:
            assert shape["shape_type"] == "rectangle"
            assert shape["points"] == [[3, 2], [15, 10]]
        np.testing.assert_array_equal(image, original_image)
