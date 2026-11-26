import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import yaml
from PIL import Image

from supervision.config import (
    KEYPOINT_VISIBILITY_DATA_FIELD,
    XYXY_DATA_FIELD,
)
from supervision.dataset.core import KeyPointsDataset
from supervision.dataset.formats.yolo import _load_data_yaml
from supervision.dataset.formats.yolo_pose import (
    _extract_kpt_shape,
    key_points_to_yolo_pose_annotations,
    load_yolo_pose_annotations,
    save_yolo_pose_annotations,
    yolo_pose_annotations_to_key_points,
)
from supervision.key_points.core import KeyPoints
from supervision.metrics import KeyPointMeanAveragePrecision
from tests.dataset.formats.helpers import (
    interleaved_paths,
    list_files_in_order,
    reversed_paths,
)


def _write_dataset(
    root: Path,
    label_lines: dict[str, list[str]],
    data_yaml: str,
    resolution_wh: tuple[int, int] = (200, 100),
) -> tuple[str, str, str]:
    """Write a minimal YOLO pose dataset and return its three paths."""
    images_directory = root / "images"
    annotations_directory = root / "labels"
    images_directory.mkdir(parents=True, exist_ok=True)
    annotations_directory.mkdir(parents=True, exist_ok=True)
    for stem, lines in label_lines.items():
        Image.new("RGB", resolution_wh).save(images_directory / f"{stem}.png")
        (annotations_directory / f"{stem}.txt").write_text("\n".join(lines))
    data_yaml_path = root / "data.yaml"
    data_yaml_path.write_text(data_yaml)
    return (
        str(images_directory),
        str(annotations_directory),
        str(data_yaml_path),
    )


class TestExtractKptShape:
    """Tests for reading ``kpt_shape`` out of a YOLO pose data.yaml."""

    @pytest.mark.parametrize(
        ("kpt_shape", "expected"),
        [
            pytest.param("[17, 3]", (17, 3), id="coco-person"),
            pytest.param("[4, 2]", (4, 2), id="no-visibility"),
        ],
    )
    def test_reads_shape(self, kpt_shape: str, expected: tuple[int, int]) -> None:
        """A valid ``kpt_shape`` is returned as its two parts."""
        data = yaml.safe_load(f"nc: 1\nnames: ['person']\nkpt_shape: {kpt_shape}\n")

        result = _extract_kpt_shape(data=data, file_path="data.yaml")

        assert result == expected

    @pytest.mark.parametrize(
        "kpt_shape",
        [
            pytest.param("[17]", id="missing-value-dim"),
            pytest.param("[17, 3, 1]", id="extra-value"),
            pytest.param("[17, 4]", id="unsupported-value-dim"),
            pytest.param("[0, 3]", id="zero-keypoints"),
            pytest.param("[-1, 3]", id="negative-keypoints"),
            pytest.param("[17.5, 3]", id="fractional-keypoints"),
            pytest.param("[true, 3]", id="boolean-keypoints"),
            pytest.param("17", id="scalar"),
            pytest.param("null", id="null"),
        ],
    )
    def test_rejects_an_invalid_kpt_shape(self, kpt_shape: str) -> None:
        """Only ``[K, 2]`` or ``[K, 3]`` with a positive whole ``K`` is accepted."""
        data = yaml.safe_load(f"nc: 1\nnames: ['person']\nkpt_shape: {kpt_shape}\n")

        with pytest.raises(ValueError, match="kpt_shape"):
            _extract_kpt_shape(data=data, file_path="data.yaml")

    def test_rejects_dataset_without_kpt_shape(self) -> None:
        """A data.yaml with no ``kpt_shape`` names the detection loader instead."""
        data = {"nc": 1, "names": ["person"]}

        with pytest.raises(ValueError, match=re.escape("DetectionDataset.from_yolo")):
            _extract_kpt_shape(data=data, file_path="data.yaml")


class TestYoloPoseAnnotationsToKeyPoints:
    """Tests for parsing YOLO pose label lines into key points."""

    def test_denormalizes_coordinates_against_the_image(self) -> None:
        """Normalized keypoint coordinates are scaled by the image resolution."""
        lines = ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]

        result = yolo_pose_annotations_to_key_points(
            lines=lines, resolution_wh=(200, 100), keypoint_count=2, value_dim=3
        )

        np.testing.assert_allclose(result.xy, [[[50.0, 30.0], [100.0, 50.0]]])
        np.testing.assert_array_equal(result.class_id, np.array([0]))

    def test_keeps_the_box_of_each_skeleton(self) -> None:
        """The box of a pose label is kept under the ``xyxy`` data field."""
        lines = ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 2"]

        result = yolo_pose_annotations_to_key_points(
            lines=lines, resolution_wh=(200, 100), keypoint_count=2, value_dim=3
        )

        np.testing.assert_allclose(
            np.asarray(result.data[XYXY_DATA_FIELD]), [[60.0, 20.0, 140.0, 80.0]]
        )

    def test_reads_visibility_as_visible_and_raw_flags(self) -> None:
        """``visible`` marks flags 1 and 2, while the raw ternary is kept in data."""
        lines = ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"]

        result = yolo_pose_annotations_to_key_points(
            lines=lines, resolution_wh=(200, 100), keypoint_count=3, value_dim=3
        )

        np.testing.assert_array_equal(result.visible, [[True, True, False]])
        np.testing.assert_array_equal(
            result.data[KEYPOINT_VISIBILITY_DATA_FIELD], [[2, 1, 0]]
        )

    def test_omits_visibility_for_two_value_keypoints(self) -> None:
        """A ``kpt_shape`` of ``[K, 2]`` carries no visibility at all."""
        lines = ["0 0.5 0.5 0.4 0.6 0.25 0.3 0.5 0.5"]

        result = yolo_pose_annotations_to_key_points(
            lines=lines, resolution_wh=(200, 100), keypoint_count=2, value_dim=2
        )

        assert result.visible is None
        assert KEYPOINT_VISIBILITY_DATA_FIELD not in result.data

    def test_returns_empty_for_no_lines(self) -> None:
        """An image with no annotation line yields empty key points."""
        result = yolo_pose_annotations_to_key_points(
            lines=[], resolution_wh=(200, 100), keypoint_count=2, value_dim=3
        )

        assert len(result) == 0

    @pytest.mark.parametrize(
        "line",
        [
            pytest.param("0 0.5 0.5 0.4 0.6", id="box-only"),
            pytest.param("0 0.5 0.5 0.4 0.6 0.25 0.3", id="one-keypoint-short"),
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.1", id="one-value-extra"
            ),
        ],
    )
    def test_rejects_a_row_of_the_wrong_length(self, line: str) -> None:
        """A row that does not hold ``5 + K * D`` values names the expected count."""
        with pytest.raises(ValueError, match=re.escape("expected 11")):
            yolo_pose_annotations_to_key_points(
                lines=[line], resolution_wh=(200, 100), keypoint_count=2, value_dim=3
            )


class TestKeyPointsToYoloPoseAnnotations:
    """Tests for writing key points back out as YOLO pose label lines."""

    def test_writes_the_stored_box_and_raw_visibility(self) -> None:
        """A label read from disk is written back with the same values."""
        line = "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"
        key_points = yolo_pose_annotations_to_key_points(
            lines=[line], resolution_wh=(200, 100), keypoint_count=3, value_dim=3
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=3,
            value_dim=3,
        )

        assert result == [line]

    def test_derives_a_box_from_visible_keypoints_when_none_is_stored(self) -> None:
        """Without a stored box, the hull of the visible keypoints is written."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=2,
        )

        assert result == ["0 0.375 0.5 0.25 0.4 0.25 0.3 0.5 0.7"]

    def test_maps_visible_back_to_a_flag_without_raw_values(self) -> None:
        """Key points carrying only ``visible`` are written as flags 2 and 0."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
            visible=np.array([[True, False]]),
            data={XYXY_DATA_FIELD: np.array([[0.0, 0.0, 200.0, 100.0]])},
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=3,
        )

        assert result == ["0 0.5 0.5 1 1 0.25 0.3 2 0.5 0.7 0"]

    def test_returns_no_lines_for_empty_key_points(self) -> None:
        """An image with no skeleton writes an empty label file."""
        result = key_points_to_yolo_pose_annotations(
            key_points=KeyPoints.empty(),
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=3,
        )

        assert result == []

    def test_rejects_key_points_without_class_id(self) -> None:
        """A YOLO label needs a class column, so ``class_id`` is required."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32)
        )

        with pytest.raises(ValueError, match="class_id"):
            key_points_to_yolo_pose_annotations(
                key_points=key_points,
                image_shape=(100, 200, 3),
                keypoint_count=2,
                value_dim=3,
            )

    def test_rejects_a_keypoint_count_that_contradicts_kpt_shape(self) -> None:
        """Writing stops when the skeletons do not match the declared shape."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
        )

        with pytest.raises(ValueError, match="declares 17"):
            key_points_to_yolo_pose_annotations(
                key_points=key_points,
                image_shape=(100, 200, 3),
                keypoint_count=17,
                value_dim=3,
            )


class TestKeyPointsToYoloPoseAfterEdits:
    """Tests that edits made after loading reach the written label."""

    @pytest.mark.parametrize(
        ("keypoint_index", "visible", "expected"),
        [
            pytest.param(
                1,
                False,
                "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 0 0.75 0.7 0",
                id="cleared-occluded-writes-zero",
            ),
            pytest.param(
                2,
                True,
                "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 2",
                id="newly-set-writes-two",
            ),
        ],
    )
    def test_writes_visibility_edited_after_load(
        self, keypoint_index: int, visible: bool, expected: str
    ) -> None:
        """A ``visible`` edit made after loading overrides the stored raw flag.

        The raw flags kept in ``data`` would otherwise be written back as read, silently
        discarding the edit a user made to ``visible``.
        """
        key_points = yolo_pose_annotations_to_key_points(
            lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"],
            resolution_wh=(200, 100),
            keypoint_count=3,
            value_dim=3,
        )
        assert key_points.visible is not None
        key_points.visible[0, keypoint_index] = visible

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=3,
            value_dim=3,
        )

        assert result == [expected]

    def test_stored_box_wins_over_edited_keypoints(self) -> None:
        """The stored ``data['xyxy']`` is written as is after ``xy`` is edited.

        The box is authoritative rather than re-derived, so a label loaded with a loose
        box keeps it; moving a keypoint does not silently change the box.
        """
        key_points = yolo_pose_annotations_to_key_points(
            lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 2"],
            resolution_wh=(200, 100),
            keypoint_count=2,
            value_dim=3,
        )
        key_points.xy[0, 1] = [150.0, 80.0]

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=3,
        )

        assert result == ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.75 0.8 2"]

    def test_popped_box_is_derived_from_edited_keypoints(self) -> None:
        """Popping ``data['xyxy']`` after editing ``xy`` derives the box afresh.

        This is the documented way to make the written box follow moved keypoints.
        """
        key_points = yolo_pose_annotations_to_key_points(
            lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 2"],
            resolution_wh=(200, 100),
            keypoint_count=2,
            value_dim=3,
        )
        key_points.xy[0, 1] = [150.0, 80.0]
        key_points.data.pop(XYXY_DATA_FIELD)

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=3,
        )

        assert result == ["0 0.5 0.55 0.5 0.5 0.25 0.3 2 0.75 0.8 2"]


class TestLoadYoloPoseAnnotations:
    """Tests for loading a YOLO pose dataset from disk."""

    def test_loads_classes_images_and_shape(self, tmp_path: Path) -> None:
        """A dataset directory yields its classes, images, labels and shape."""
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )

        classes, image_paths, loaded, kpt_shape, _ = load_yolo_pose_annotations(
            images_directory_path=images,
            annotations_directory_path=annotations,
            data_yaml_path=data_yaml,
        )

        assert classes == ["person"]
        assert len(image_paths) == 1
        assert kpt_shape == (2, 3)
        assert len(loaded[image_paths[0]]) == 1

    @mock.patch(
        "supervision.dataset.formats.yolo._load_data_yaml", wraps=_load_data_yaml
    )
    def test_reads_data_yaml_once(
        self, read_yaml: mock.MagicMock, tmp_path: Path
    ) -> None:
        """Shape, flip_idx and class names all come from a single data.yaml read."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]},
            data_yaml=(
                "nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n"
            ),
        )

        load_yolo_pose_annotations(*paths)

        assert read_yaml.call_count == 1

    def test_image_without_a_label_file_is_empty(self, tmp_path: Path) -> None:
        """An image with no label file is kept with empty key points."""
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )
        Image.new("RGB", (200, 100)).save(Path(images) / "b.png")

        _, image_paths, loaded, _, _ = load_yolo_pose_annotations(
            images_directory_path=images,
            annotations_directory_path=annotations,
            data_yaml_path=data_yaml,
        )

        unlabelled = next(path for path in image_paths if path.endswith("b.png"))
        assert len(loaded[unlabelled]) == 0

    @pytest.mark.parametrize(
        "argument", ["images_directory_path", "annotations_directory_path"]
    )
    def test_rejects_a_missing_directory(self, tmp_path: Path, argument: str) -> None:
        """A mistyped directory fails instead of loading unlabelled or no images."""
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )
        paths: dict[str, Any] = {
            "images_directory_path": images,
            "annotations_directory_path": annotations,
            "data_yaml_path": data_yaml,
        }
        missing = str(tmp_path / "missing")
        paths[argument] = missing

        with pytest.raises(FileNotFoundError, match=re.escape(f"'{missing}'")):
            load_yolo_pose_annotations(**paths)

    def test_rejects_a_file_given_as_a_directory(self, tmp_path: Path) -> None:
        """A labels path naming a file rather than a directory fails."""
        images, _, data_yaml = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )

        with pytest.raises(NotADirectoryError, match="annotations_directory_path"):
            load_yolo_pose_annotations(
                images_directory_path=images,
                annotations_directory_path=data_yaml,
                data_yaml_path=data_yaml,
            )

    def test_invalid_line_names_its_file(self, tmp_path: Path) -> None:
        """One bad label among many stays traceable to the file holding it."""
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )

        with pytest.raises(ValueError, match=re.escape("a.txt")):
            load_yolo_pose_annotations(
                images_directory_path=images,
                annotations_directory_path=annotations,
                data_yaml_path=data_yaml,
            )

    def test_class_id_resolves_to_its_declared_name(self, tmp_path: Path) -> None:
        """A class id indexes the name data.yaml declares for it, gaps included."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={
                "a": [
                    "2 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1",
                    "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1",
                ]
            },
            data_yaml="names:\n  0: person\n  2: dog\nkpt_shape: [2, 3]\n",
        )

        classes, image_paths, loaded, _, _ = load_yolo_pose_annotations(*paths)

        class_ids = loaded[image_paths[0]].class_id
        assert class_ids is not None
        assert [classes[class_id] for class_id in class_ids] == ["dog", "person"]

    def test_raises_for_class_id_not_in_names(self, tmp_path: Path) -> None:
        """A label whose class id data.yaml does not declare fails, naming its file."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["1 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"]},
            data_yaml="names:\n  0: person\n  2: dog\nkpt_shape: [2, 3]\n",
        )

        with pytest.raises(ValueError, match=r"a\.txt'.*class id 1 .*not declared"):
            load_yolo_pose_annotations(*paths)


class TestYoloPoseRoundTrip:
    """Tests that a dataset survives a write and read cycle unchanged."""

    def test_annotation_only_export_retains_sparse_class_ids(
        self, tmp_path: Path
    ) -> None:
        """Annotation-only export remains compatible with the source YAML."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  1: cat\n  2: dog\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)

        dataset.as_yolo(annotations_directory_path=paths[1])
        reloaded = KeyPointsDataset.from_yolo(*paths)

        class_ids = reloaded[0][2].class_id
        assert class_ids is not None
        assert reloaded.classes[class_ids[0]] == "dog"

    @pytest.mark.parametrize(
        "names_yaml",
        [
            pytest.param("names:\n  1: cat\n  '1': dog\n", id="int-and-digit-string"),
            pytest.param("names:\n  1: cat\n  1: dog\n", id="repeated-block-key"),
            pytest.param("names: {1: cat, 1: dog}\n", id="repeated-flow-key"),
        ],
    )
    def test_rejects_duplicate_numeric_names(
        self, tmp_path: Path, names_yaml: str
    ) -> None:
        """Pose datasets reject keys that resolve to the same class ID."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["1 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml=names_yaml + "kpt_shape: [1, 3]\n",
        )

        with pytest.raises(ValueError, match=r"duplicate.*class.*1"):
            KeyPointsDataset.from_yolo(*paths)

    def test_split_preserves_sparse_mapping(self, tmp_path: Path) -> None:
        """A split can export labels against the original class mapping."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  1: cat\n  2: dog\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)
        subset, _ = dataset.split(split_ratio=1, shuffle=False)

        subset.as_yolo(annotations_directory_path=paths[1])

        assert Path(paths[1], "a.txt").read_text().split()[0] == "2"

    def test_new_yaml_uses_normalized_ids(self, tmp_path: Path) -> None:
        """Exporting a new YAML writes IDs aligned with its names list."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  1: cat\n  2: dog\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)
        new_yaml = str(tmp_path / "new.yaml")

        dataset.as_yolo(annotations_directory_path=paths[1], data_yaml_path=new_yaml)
        reloaded = KeyPointsDataset.from_yolo(paths[0], paths[1], new_yaml)

        class_ids = reloaded[0][2].class_id
        assert class_ids is not None
        assert reloaded.classes[class_ids[0]] == "dog"

    def test_yaml_only_export_retains_sparse_class_ids(self, tmp_path: Path) -> None:
        """A YAML-only export stays compatible with existing sparse-ID labels."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  0: cat\n  2: dog\n  3: bird\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)

        dataset.as_yolo(data_yaml_path=paths[2])
        reloaded = KeyPointsDataset.from_yolo(*paths)

        class_ids = reloaded[0][2].class_id
        assert class_ids is not None
        assert reloaded.classes[class_ids[0]] == "dog"

    def test_direct_writer_retains_sparse_class_ids(self, tmp_path: Path) -> None:
        """The direct annotation writer remains compatible with the source YAML."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  1: cat\n  2: dog\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)

        save_yolo_pose_annotations(dataset, paths[1])
        reloaded = KeyPointsDataset.from_yolo(*paths)

        class_ids = reloaded[0][2].class_id
        assert class_ids is not None
        assert reloaded.classes[class_ids[0]] == "dog"

    def test_changed_classes_require_new_yaml(self, tmp_path: Path) -> None:
        """Changed class names cannot silently reuse the original YAML mapping."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  1: cat\n  2: dog\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)
        dataset.classes.reverse()

        with pytest.raises(ValueError, match=r"classes changed.*data\.yaml"):
            dataset.as_yolo(annotations_directory_path=paths[1])

    def test_changed_source_yaml_requires_new_yaml(self, tmp_path: Path) -> None:
        """An externally changed source mapping is not silently reused."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["2 0.5 0.5 0.4 0.6 0.25 0.3 2"]},
            data_yaml="names:\n  1: cat\n  2: dog\nkpt_shape: [1, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)
        Path(paths[2]).write_text("names: [cat, dog]\nkpt_shape: [1, 3]\n")

        with pytest.raises(ValueError, match="mapping changed"):
            dataset.as_yolo(annotations_directory_path=paths[1])

    @pytest.mark.parametrize(
        ("value_dim", "line"),
        [
            pytest.param(
                3, "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0", id="visibility"
            ),
            pytest.param(
                2, "0 0.5 0.5 0.4 0.6 0.25 0.3 0.5 0.5 0.75 0.7", id="no-visibility"
            ),
            pytest.param(
                3,
                "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0\n"
                "0 0.25 0.25 0.2 0.2 0.15 0.2 1 0.25 0.25 2 0.3 0.3 0",
                id="two-skeletons",
            ),
        ],
    )
    def test_round_trip_preserves_the_label_file(
        self, tmp_path: Path, value_dim: int, line: str
    ) -> None:
        """Reading then writing a pose dataset reproduces the original line."""
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path / "src",
            label_lines={"a": [line]},
            data_yaml=f"nc: 1\nnames: ['person']\nkpt_shape: [3, {value_dim}]\n",
        )
        dataset = KeyPointsDataset.from_yolo(
            images_directory_path=images,
            annotations_directory_path=annotations,
            data_yaml_path=data_yaml,
        )

        out = tmp_path / "out"
        dataset.as_yolo(
            images_directory_path=str(out / "images"),
            annotations_directory_path=str(out / "labels"),
            data_yaml_path=str(out / "data.yaml"),
        )

        assert next(iter(dataset.annotations.values())).xy.dtype == np.dtype(np.float64)
        assert (out / "labels" / "a.txt").read_text() == f"{line}\n"

    def test_round_trip_rounds_to_six_significant_digits(self) -> None:
        """A label carrying more precision than the writer emits is rounded."""
        line = "0 0.5 0.5 0.4 0.6 0.1234567 0.3 2 0.5 0.5 1 0.75 0.7 0"
        key_points = yolo_pose_annotations_to_key_points(
            lines=[line], resolution_wh=(200, 100), keypoint_count=3, value_dim=3
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=3,
            value_dim=3,
        )

        assert result == ["0 0.5 0.5 0.4 0.6 0.123457 0.3 2 0.5 0.5 1 0.75 0.7 0"]

    @pytest.mark.parametrize(
        ("line", "value_dim", "expected"),
        [
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.867320506 0.3",
                2,
                "0 0.5 0.5 0.4 0.6 0.867321 0.3",
                id="x-coordinate-without-visibility",
            ),
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.25 0.867320506 2",
                3,
                "0 0.5 0.5 0.4 0.6 0.25 0.867321 2",
                id="y-coordinate",
            ),
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.25 0.3 0.867320506",
                3,
                "0 0.5 0.5 0.4 0.6 0.25 0.3 0.867321",
                id="fractional-visibility",
            ),
        ],
    )
    def test_round_trip_rounds_once_when_writing(
        self, line: str, value_dim: int, expected: str
    ) -> None:
        """A keypoint value longer than six digits is rounded once, when written.

        Held as float32 in between, 0.867320506 lands below the rounding tie and is
        written as 0.86732 instead of 0.867321.
        """
        key_points = yolo_pose_annotations_to_key_points(
            lines=[line],
            resolution_wh=(640, 480),
            keypoint_count=1,
            value_dim=value_dim,
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(480, 640, 3),
            keypoint_count=1,
            value_dim=value_dim,
        )

        assert result == [expected]

    def test_round_trip_keeps_a_small_box_width(self) -> None:
        """A narrow box is written back with its six significant digits intact.

        Stored as float32, a width of 0.00478514 on a 640 pixel wide image comes back as
        0.00478516, because the corner difference loses its last digits.
        """
        line = "0 0.5 0.5 0.00478514 0.6 0.25 0.3 2"
        key_points = yolo_pose_annotations_to_key_points(
            lines=[line], resolution_wh=(640, 480), keypoint_count=1, value_dim=3
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(480, 640, 3),
            keypoint_count=1,
            value_dim=3,
        )

        assert result == [line]

    def test_round_trip_carries_flip_idx(self, tmp_path: Path) -> None:
        """`flip_idx` survives a write and read cycle."""
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path / "src",
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"]},
            data_yaml=(
                "nc: 1\nnames: ['person']\nkpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\n"
            ),
        )
        dataset = KeyPointsDataset.from_yolo(
            images_directory_path=images,
            annotations_directory_path=annotations,
            data_yaml_path=data_yaml,
        )
        out = tmp_path / "out"
        dataset.as_yolo(
            images_directory_path=str(out / "images"),
            annotations_directory_path=str(out / "labels"),
            data_yaml_path=str(out / "data.yaml"),
        )

        reloaded = KeyPointsDataset.from_yolo(
            images_directory_path=str(out / "images"),
            annotations_directory_path=str(out / "labels"),
            data_yaml_path=str(out / "data.yaml"),
        )

        assert dataset.flip_idx == [0, 2, 1]
        assert reloaded.flip_idx == [0, 2, 1]

    def test_round_trip_reloads_the_same_keypoints_and_metadata(
        self, tmp_path: Path
    ) -> None:
        """Reloading an exported dataset keeps its classes, shape and per-skeleton data.

        Equality is not asserted: it compares image paths, which differ between the
        source and the output directory.
        """
        line = "0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path / "src",
            label_lines={"a": [line]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [3, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(
            images_directory_path=images,
            annotations_directory_path=annotations,
            data_yaml_path=data_yaml,
        )

        out = tmp_path / "out"
        dataset.as_yolo(
            images_directory_path=str(out / "images"),
            annotations_directory_path=str(out / "labels"),
            data_yaml_path=str(out / "data.yaml"),
        )
        reloaded = KeyPointsDataset.from_yolo(
            images_directory_path=str(out / "images"),
            annotations_directory_path=str(out / "labels"),
            data_yaml_path=str(out / "data.yaml"),
        )

        assert reloaded.classes == dataset.classes
        assert reloaded.keypoint_count == dataset.keypoint_count
        assert reloaded.value_dim == dataset.value_dim
        original = dataset.annotations[str(Path(images) / "a.png")]
        restored = reloaded.annotations[str(out / "images" / "a.png")]
        np.testing.assert_allclose(restored.xy, original.xy)
        np.testing.assert_array_equal(
            restored.data[KEYPOINT_VISIBILITY_DATA_FIELD],
            original.data[KEYPOINT_VISIBILITY_DATA_FIELD],
        )
        np.testing.assert_allclose(
            np.asarray(restored.data[XYXY_DATA_FIELD]),
            np.asarray(original.data[XYXY_DATA_FIELD]),
            rtol=1e-5,
        )


class TestYoloPoseKeyPointValidation:
    """Tests for the checks applied to the keypoint values of a pose label."""

    @pytest.mark.parametrize(
        ("line", "value_dim"),
        [
            pytest.param(
                "0 0.5 0.5 0.4 0.6 nan 0.3 2 0.5 0.5 2", 3, id="nan-coordinate"
            ),
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.25 inf 2 0.5 0.5 2", 3, id="inf-coordinate"
            ),
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.25 0.3 nan 0.5 0.5 2", 3, id="nan-visibility"
            ),
            pytest.param(
                "0 0.5 0.5 0.4 0.6 0.25 0.3 0.5 -inf", 2, id="inf-without-visibility"
            ),
        ],
    )
    def test_rejects_non_finite_keypoint_values(
        self, line: str, value_dim: int
    ) -> None:
        """A non-finite keypoint value is refused like a non-finite box.

        Loaded silently it would surface only as a missed match in the metrics.
        """
        with pytest.raises(ValueError, match="expected finite keypoint"):
            yolo_pose_annotations_to_key_points(
                lines=[line],
                resolution_wh=(200, 100),
                keypoint_count=2,
                value_dim=value_dim,
            )

    def test_rejects_a_negative_visibility_value(self) -> None:
        """A negative third value is neither a flag nor a confidence, so it fails."""
        with pytest.raises(ValueError, match=re.escape("visibility ['-1']")):
            yolo_pose_annotations_to_key_points(
                lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 -1 0.5 0.5 2"],
                resolution_wh=(200, 100),
                keypoint_count=2,
                value_dim=3,
            )

    def test_fractional_confidence_loads_as_visible(self) -> None:
        """A keypoint confidence in place of a flag is kept and reads as visible.

        Ultralytics writes a keypoint confidence as the third value when it saves
        predictions as labels, so a positive fraction marks a labelled keypoint.
        """
        result = yolo_pose_annotations_to_key_points(
            lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 0.83 0.5 0.5 0"],
            resolution_wh=(200, 100),
            keypoint_count=2,
            value_dim=3,
        )

        np.testing.assert_array_equal(result.visible, [[True, False]])
        np.testing.assert_array_equal(
            result.data[KEYPOINT_VISIBILITY_DATA_FIELD],
            np.array([[0.83, 0.0]], dtype=np.float64),
        )

    def test_fractional_confidence_round_trips(self) -> None:
        """A label carrying a keypoint confidence is written back unchanged."""
        line = "0 0.5 0.5 0.4 0.6 0.25 0.3 0.83 0.5 0.5 2 0.75 0.7 0"
        key_points = yolo_pose_annotations_to_key_points(
            lines=[line], resolution_wh=(200, 100), keypoint_count=3, value_dim=3
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=3,
            value_dim=3,
        )

        assert result == [line]


class TestYoloPoseOccludedKeyPointScoring:
    """Tests that occluded keypoints read from YOLO pose labels are scored."""

    def test_occluded_keypoint_counts_towards_key_point_map(
        self, tmp_path: Path
    ) -> None:
        """A joint flagged 1 is a labelled target, so a matching prediction scores 1.

        The only labelled joint of the skeleton is occluded (flag 1) and the others are
        unlabelled (flag 0). Reading flag 1 as not visible would leave the target
        without a keypoint for OKS to measure, so the prediction would not count.
        """
        images, annotations, data_yaml = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 1 0.5 0.5 0 0.75 0.7 0"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [3, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(
            images_directory_path=images,
            annotations_directory_path=annotations,
            data_yaml_path=data_yaml,
        )
        targets = dataset.annotations[str(Path(images) / "a.png")]
        predictions = KeyPoints(
            xy=targets.xy.copy(),
            class_id=np.array([0]),
            detection_confidence=np.array([0.9]),
        )
        metric = KeyPointMeanAveragePrecision(sigmas=[0.25, 0.25, 0.25])

        result = metric.update(predictions, targets).compute()

        assert result.map50_95 == pytest.approx(1.0)


class TestYoloPoseBoxValidation:
    """Tests for the box checks shared with the detection loader."""

    @pytest.mark.parametrize(
        "box",
        [
            pytest.param("0.5 0.5 -0.4 0.6", id="negative-width"),
            pytest.param("0.5 0.5 0.4 -0.6", id="negative-height"),
        ],
    )
    def test_rejects_a_negative_box_extent(self, box: str) -> None:
        """A reversed box is refused rather than stored with swapped corners."""
        with pytest.raises(ValueError, match="non-negative width and height"):
            yolo_pose_annotations_to_key_points(
                lines=[f"0 {box} 0.25 0.3 2 0.5 0.5 2"],
                resolution_wh=(200, 100),
                keypoint_count=2,
                value_dim=3,
            )

    def test_rejects_a_non_finite_box(self) -> None:
        """A `nan` coordinate cannot place a box, so it is refused."""
        with pytest.raises(ValueError, match="finite center, width and height"):
            yolo_pose_annotations_to_key_points(
                lines=["0 0.5 nan 0.4 0.6 0.25 0.3 2 0.5 0.5 2"],
                resolution_wh=(200, 100),
                keypoint_count=2,
                value_dim=3,
            )

    def test_names_the_file_holding_an_invalid_box(self, tmp_path: Path) -> None:
        """A corrupt box is traced back to the label file it came from."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 -0.4 0.6 0.25 0.3 2 0.5 0.5 2"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )

        with pytest.raises(ValueError, match=r"Invalid YOLO pose annotation file"):
            load_yolo_pose_annotations(*paths)


class TestKeyPointsToYoloPoseBoxHandling:
    """Tests for how the writer derives and orders the box it writes."""

    def test_orders_the_corners_of_a_reversed_stored_box(self) -> None:
        """A reversed `data['xyxy']` is written with a positive extent."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
            data={XYXY_DATA_FIELD: np.array([[150.0, 80.0, 50.0, 20.0]])},
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=2,
        )

        assert result == ["0 0.5 0.5 0.5 0.6 0.25 0.3 0.5 0.7"]

    def test_keeps_a_representable_center_finite_at_the_float_limit(self) -> None:
        """The corner sum overflows here, but the box center does not."""
        key_points = KeyPoints(
            xy=np.zeros((1, 2, 2), dtype=np.float32),
            class_id=np.array([0]),
            data={XYXY_DATA_FIELD: np.array([[1e308, 0.0, 1.1e308, 2.0]])},
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(10, 20, 3),
            keypoint_count=2,
            value_dim=2,
        )

        assert float(result[0].split()[1]) == pytest.approx(5.25e306)

    @pytest.mark.parametrize(
        "key_points",
        [
            pytest.param(
                KeyPoints(
                    xy=np.zeros((1, 2, 2), dtype=np.float32),
                    class_id=np.array([0]),
                    data={XYXY_DATA_FIELD: np.array([[-1e308, 0.0, 1e308, 2.0]])},
                ),
                id="stored-box",
            ),
            pytest.param(
                KeyPoints(
                    xy=np.array([[[-1e308, 0.0], [1e308, 2.0]]], dtype=np.float64),
                    class_id=np.array([0]),
                ),
                id="box-derived-from-keypoints",
            ),
        ],
    )
    def test_rejects_a_box_whose_extent_overflows(self, key_points: KeyPoints) -> None:
        """A width beyond float64 would be written as `inf`, so it is refused."""
        with pytest.raises(ValueError, match="overflows float64"):
            key_points_to_yolo_pose_annotations(
                key_points=key_points,
                image_shape=(10, 20, 3),
                keypoint_count=2,
                value_dim=2,
            )

    def test_derived_box_keeps_occluded_keypoints(self) -> None:
        """A keypoint flagged 1 is labelled, so the box still bounds it."""
        key_points = yolo_pose_annotations_to_key_points(
            lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1"],
            resolution_wh=(200, 100),
            keypoint_count=2,
            value_dim=3,
        )
        del key_points.data[XYXY_DATA_FIELD]

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=3,
        )

        assert result == ["0 0.375 0.4 0.25 0.2 0.25 0.3 2 0.5 0.5 1"]

    def test_derived_box_ignores_unlabelled_keypoints_at_the_origin(self) -> None:
        """Without any flags, a joint left at the origin does not stretch the box."""
        key_points = KeyPoints(
            xy=np.array([[[0.0, 0.0], [50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=3,
            value_dim=2,
        )

        assert result[0].startswith("0 0.375 0.5 0.25 0.4 ")

    def test_derived_box_falls_back_to_every_keypoint_when_none_is_labelled(
        self,
    ) -> None:
        """A skeleton whose keypoints are all flagged 0 still yields a box."""
        key_points = KeyPoints(
            xy=np.array([[[10.0, 20.0], [50.0, 60.0]]], dtype=np.float32),
            class_id=np.array([0]),
            data={KEYPOINT_VISIBILITY_DATA_FIELD: np.array([[0, 0]])},
        )

        result = key_points_to_yolo_pose_annotations(
            key_points=key_points,
            image_shape=(100, 200, 3),
            keypoint_count=2,
            value_dim=3,
        )

        assert result == ["0 0.15 0.4 0.2 0.4 0.05 0.2 0 0.25 0.6 0"]

    def test_reordered_keypoints_write_matching_flags(self) -> None:
        """Keypoints reordered after loading are written with their own flags.

        The raw flags must follow the keypoint axis, or the reordered label would pair
        each coordinate with the flag of the keypoint that used to sit there.
        """
        key_points = yolo_pose_annotations_to_key_points(
            lines=["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"],
            resolution_wh=(200, 100),
            keypoint_count=3,
            value_dim=3,
        )
        reordered = key_points[:, [2, 1, 0]]
        assert isinstance(reordered, KeyPoints)

        result = key_points_to_yolo_pose_annotations(
            key_points=reordered,
            image_shape=(100, 200, 3),
            keypoint_count=3,
            value_dim=3,
        )

        assert result == ["0 0.5 0.5 0.4 0.6 0.75 0.7 0 0.5 0.5 1 0.25 0.3 2"]

    def test_rejects_visibility_of_the_wrong_width(self) -> None:
        """A visibility field not one value per keypoint names the field."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
            data={KEYPOINT_VISIBILITY_DATA_FIELD: np.array([[2.0, 1.0, 0.0]])},
        )

        with pytest.raises(ValueError, match=re.escape(KEYPOINT_VISIBILITY_DATA_FIELD)):
            key_points_to_yolo_pose_annotations(
                key_points=key_points,
                image_shape=(100, 200, 3),
                keypoint_count=2,
                value_dim=3,
            )

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            pytest.param(
                {KEYPOINT_VISIBILITY_DATA_FIELD: np.array([2.0])},
                r"data\['keypoint_visibility'\] of shape \(1,\)",
                id="one-dimensional-visibility",
            ),
            pytest.param(
                {KEYPOINT_VISIBILITY_DATA_FIELD: np.array([[2.0, np.nan]])},
                r"non-finite values in data\['keypoint_visibility'\]",
                id="nan-visibility",
            ),
            pytest.param(
                {KEYPOINT_VISIBILITY_DATA_FIELD: np.array([[2.0, -1.0]])},
                r"negative values in data\['keypoint_visibility'\]",
                id="negative-visibility",
            ),
            pytest.param(
                {XYXY_DATA_FIELD: np.array([10.0])},
                r"data\['xyxy'\] of shape \(1,\)",
                id="one-dimensional-box",
            ),
            pytest.param(
                {XYXY_DATA_FIELD: np.array([[np.nan, 0.0, 10.0, 10.0]])},
                r"non-finite values in data\['xyxy'\]",
                id="nan-box",
            ),
        ],
    )
    def test_rejects_metadata_the_loader_would_refuse(
        self, data: dict[str, Any], match: str
    ) -> None:
        """A malformed box or visibility field fails instead of writing a bad label."""
        key_points = KeyPoints(
            xy=np.array([[[50.0, 30.0], [100.0, 70.0]]], dtype=np.float32),
            class_id=np.array([0]),
            data=data,
        )

        with pytest.raises(ValueError, match=match):
            key_points_to_yolo_pose_annotations(
                key_points=key_points,
                image_shape=(100, 200, 3),
                keypoint_count=2,
                value_dim=3,
            )


class TestSaveYoloPoseAnnotations:
    """Tests for writing the label files of a keypoint dataset."""

    def test_writes_no_label_when_a_later_annotation_is_malformed(
        self, tmp_path: Path
    ) -> None:
        """Every annotation is checked before the first label file is written."""
        image_paths = []
        for name in ("a.png", "b.png"):
            Image.new("RGB", (200, 100)).save(tmp_path / name)
            image_paths.append(str(tmp_path / name))
        good = KeyPoints(
            xy=np.array([[[50.0, 30.0]]], dtype=np.float32), class_id=np.array([0])
        )
        malformed = KeyPoints(
            xy=np.array([[[50.0, 30.0]]], dtype=np.float32),
            class_id=np.array([0]),
            data={KEYPOINT_VISIBILITY_DATA_FIELD: np.array([2.0])},
        )
        dataset = KeyPointsDataset(
            classes=["person"],
            images=image_paths,
            annotations={image_paths[0]: good, image_paths[1]: malformed},
            keypoint_count=1,
        )
        labels = tmp_path / "labels"

        with pytest.raises(ValueError, match=re.escape("b.png")):
            save_yolo_pose_annotations(
                dataset=dataset, annotations_directory_path=str(labels)
            )

        assert not labels.exists()


class TestFlipIdx:
    """Tests for reading and writing the `flip_idx` keypoint permutation."""

    def test_absent_flip_idx_loads_as_none(self, tmp_path: Path) -> None:
        """A dataset declaring no `flip_idx` carries none."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 2"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )

        dataset = KeyPointsDataset.from_yolo(*paths)

        assert dataset.flip_idx is None

    def test_absent_flip_idx_is_not_written(self, tmp_path: Path) -> None:
        """A dataset without `flip_idx` does not gain the field on export."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 2"]},
            data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [2, 3]\n",
        )
        dataset = KeyPointsDataset.from_yolo(*paths)
        data_yaml_path = tmp_path / "out" / "data.yaml"

        dataset.as_yolo(data_yaml_path=str(data_yaml_path))

        assert "flip_idx" not in data_yaml_path.read_text()

    @pytest.mark.parametrize(
        "flip_idx",
        [
            pytest.param("[0, 1]", id="too-short"),
            pytest.param("[0, 1, 1]", id="repeated-index"),
            pytest.param("[0, 1, 3]", id="out-of-range"),
            pytest.param("[0, 2.0, 1]", id="float-index"),
            pytest.param("[false, true, 2]", id="boolean-index"),
            pytest.param("1", id="scalar-integer"),
            pytest.param("true", id="scalar-boolean"),
        ],
    )
    def test_rejects_a_flip_idx_that_is_not_a_permutation(
        self, tmp_path: Path, flip_idx: str
    ) -> None:
        """A `flip_idx` that would drop or duplicate a joint is refused."""
        paths = _write_dataset(
            root=tmp_path,
            label_lines={"a": ["0 0.5 0.5 0.4 0.6 0.25 0.3 2 0.5 0.5 1 0.75 0.7 0"]},
            data_yaml=(
                f"nc: 1\nnames: ['person']\nkpt_shape: [3, 3]\nflip_idx: {flip_idx}\n"
            ),
        )

        with pytest.raises(ValueError, match="permutation"):
            KeyPointsDataset.from_yolo(*paths)

    def test_numpy_flip_idx_round_trips_through_data_yaml(self, tmp_path: Path) -> None:
        """A NumPy `flip_idx` and keypoint count are written as plain YAML integers."""
        dataset = KeyPointsDataset(
            classes=["person"],
            images=[],
            annotations={},
            keypoint_count=np.int64(2),
            flip_idx=np.array([1, 0]),
        )
        images_directory = tmp_path / "images"
        images_directory.mkdir()
        (tmp_path / "labels").mkdir()
        data_yaml_path = tmp_path / "data.yaml"

        dataset.as_yolo(data_yaml_path=str(data_yaml_path))
        reloaded = KeyPointsDataset.from_yolo(
            images_directory_path=str(images_directory),
            annotations_directory_path=str(tmp_path / "labels"),
            data_yaml_path=str(data_yaml_path),
        )

        assert yaml.safe_load(data_yaml_path.read_text())["flip_idx"] == [1, 0]
        assert reloaded.keypoint_count == 2
        assert reloaded.flip_idx == [1, 0]


def test_save_pose_data_yaml_creates_parents_and_content(tmp_path: Path) -> None:
    """Verify save_pose_data_yaml creates directories and writes metadata."""
    from supervision.dataset.formats.yolo_pose import save_pose_data_yaml
    from supervision.utils.file import read_yaml_file

    yaml_path = tmp_path / "deeply" / "nested" / "pose_data.yaml"
    classes = ["athlete"]

    save_pose_data_yaml(
        data_yaml_path=str(yaml_path),
        classes=classes,
        keypoint_count=3,
        value_dim=3,
        flip_idx=[0, 2, 1],
    )

    assert yaml_path.is_file()
    data = read_yaml_file(str(yaml_path))
    assert data == {
        "nc": 1,
        "names": ["athlete"],
        "kpt_shape": [3, 3],
        "flip_idx": [0, 2, 1],
    }


_POSE_ORDER_STEMS = ["c", "a", "e", "b", "d"]


def _write_unordered_pose_dataset(root: Path) -> tuple[str, str, str]:
    """Write a YOLO pose dataset of several labelled images and return its paths."""
    return _write_dataset(
        root=root,
        label_lines={
            stem: ["0 0.5 0.5 0.4 0.6 0.25 0.3 2"] for stem in _POSE_ORDER_STEMS
        },
        data_yaml="nc: 1\nnames: ['person']\nkpt_shape: [1, 3]\n",
    )


class TestYoloPoseImageOrder:
    """Tests that ``KeyPointsDataset.from_yolo`` orders images independently of the
    filesystem listing order."""

    @pytest.mark.parametrize(
        "reorder",
        [
            pytest.param(reversed_paths, id="reversed-listing"),
            pytest.param(interleaved_paths, id="interleaved-listing"),
        ],
    )
    def test_image_paths_follow_sorted_file_names(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        reorder: Callable[[list[Path]], list[Path]],
    ) -> None:
        """Images load in sorted path order whatever order the directory lists."""
        paths = _write_unordered_pose_dataset(tmp_path)
        list_files_in_order(
            monkeypatch, reorder, "supervision.dataset.formats.yolo_pose"
        )

        dataset = KeyPointsDataset.from_yolo(*paths)

        assert [Path(path).stem for path in dataset.image_paths] == sorted(
            _POSE_ORDER_STEMS
        )

    def test_seeded_split_does_not_depend_on_listing_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same files and seed give the same split for two listing orders."""
        paths = _write_unordered_pose_dataset(tmp_path)
        splits = []
        for reorder in (reversed_paths, interleaved_paths):
            list_files_in_order(
                monkeypatch, reorder, "supervision.dataset.formats.yolo_pose"
            )
            dataset = KeyPointsDataset.from_yolo(*paths)

            train, test = dataset.split(split_ratio=0.6, random_state=0)
            splits.append((train.image_paths, test.image_paths))

        assert splits[0] == splits[1]
