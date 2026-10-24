from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.dataset.core import KeyPointsDataset
from supervision.key_points.core import KeyPoints


def _key_points(class_id: int = 0, keypoint_count: int = 2) -> KeyPoints:
    """Build a single-skeleton annotation with ascending coordinates."""
    xy = np.arange(keypoint_count * 2, dtype=np.float32).reshape(1, keypoint_count, 2)
    return KeyPoints(xy=xy, class_id=np.array([class_id]))


def _dataset(
    image_paths: list[str],
    classes: list[str] | None = None,
    class_ids: list[int] | None = None,
    keypoint_count: int = 2,
) -> KeyPointsDataset:
    """Build a dataset whose annotations are one skeleton per image."""
    resolved_classes = ["person"] if classes is None else classes
    resolved_ids = [0] * len(image_paths) if class_ids is None else class_ids
    return KeyPointsDataset(
        classes=resolved_classes,
        images=image_paths,
        annotations={
            path: _key_points(class_id=class_id, keypoint_count=keypoint_count)
            for path, class_id in zip(image_paths, resolved_ids)
        },
        keypoint_count=keypoint_count,
    )


class TestKeyPointsDatasetInit:
    """Tests for the checks applied when a keypoint dataset is built."""

    def test_assigns_class_names_from_class_id(self) -> None:
        """Each annotation gains the class name matching its class id."""
        dataset = _dataset(["a.png"], classes=["cat", "dog"], class_ids=[1])

        names = dataset.annotations["a.png"].data[CLASS_NAME_DATA_FIELD]

        np.testing.assert_array_equal(names, np.array(["dog"]))

    def test_does_not_mutate_the_annotations_passed_in(self) -> None:
        """Annotations are deep-copied, so the caller's objects are untouched."""
        annotation = _key_points()
        KeyPointsDataset(
            classes=["person"],
            images=["a.png"],
            annotations={"a.png": annotation},
            keypoint_count=2,
        )

        assert CLASS_NAME_DATA_FIELD not in annotation.data

    def test_drops_duplicate_image_paths(self) -> None:
        """A repeated image path is kept once, in its first position."""
        dataset = KeyPointsDataset(
            classes=["person"],
            images=["a.png", "a.png"],
            annotations={"a.png": _key_points()},
            keypoint_count=2,
        )

        assert dataset.image_paths == ["a.png"]

    def test_rejects_mismatched_images_and_annotations(self) -> None:
        """Every image needs an annotation entry and the reverse."""
        with pytest.raises(ValueError, match="must match"):
            KeyPointsDataset(
                classes=["person"],
                images=["a.png"],
                annotations={"b.png": _key_points()},
                keypoint_count=2,
            )

    @pytest.mark.parametrize(
        "keypoint_count",
        [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
    )
    def test_rejects_a_non_positive_keypoint_count(self, keypoint_count: int) -> None:
        """A skeleton holds at least one keypoint."""
        with pytest.raises(ValueError, match="positive integer"):
            KeyPointsDataset(
                classes=["person"],
                images=[],
                annotations={},
                keypoint_count=keypoint_count,
            )

    @pytest.mark.parametrize("value_dim", [1, 4])
    def test_rejects_an_unsupported_value_dim(self, value_dim: int) -> None:
        """YOLO pose stores two or three values per keypoint, nothing else."""
        with pytest.raises(ValueError, match="must be 2 or 3"):
            KeyPointsDataset(
                classes=["person"],
                images=[],
                annotations={},
                keypoint_count=2,
                value_dim=value_dim,
            )

    def test_rejects_an_annotation_with_a_foreign_keypoint_count(self) -> None:
        """An annotation must hold the keypoint count the dataset declares."""
        with pytest.raises(ValueError, match="declares 17"):
            KeyPointsDataset(
                classes=["person"],
                images=["a.png"],
                annotations={"a.png": _key_points(keypoint_count=2)},
                keypoint_count=17,
            )

    def test_rejects_a_class_id_outside_the_class_list(self) -> None:
        """A class id with no name names the offending image and range."""
        with pytest.raises(ValueError, match=r"outside the valid range \[0, 0\]"):
            _dataset(["a.png"], classes=["person"], class_ids=[4])


class TestKeyPointsDatasetSequence:
    """Tests for length, indexing and iteration."""

    def test_len_counts_images(self) -> None:
        """The dataset length is its image count."""
        dataset = _dataset(["a.png", "b.png"])

        assert len(dataset) == 2

    def test_getitem_returns_path_image_and_annotation(self, tmp_path: Path) -> None:
        """Indexing reads the image from disk and pairs it with its annotation."""
        image_path = tmp_path / "a.png"
        Image.new("RGB", (20, 10)).save(image_path)
        dataset = _dataset([str(image_path)])

        path, image, annotation = dataset[0]

        assert path == str(image_path)
        assert image.shape == (10, 20, 3)
        assert len(annotation) == 1

    def test_iter_yields_every_image(self, tmp_path: Path) -> None:
        """Iteration walks all images in order."""
        paths = []
        for stem in ("a", "b"):
            image_path = tmp_path / f"{stem}.png"
            Image.new("RGB", (20, 10)).save(image_path)
            paths.append(str(image_path))
        dataset = _dataset(paths)

        assert [path for path, _, _ in dataset] == paths

    def test_getitem_reports_an_unreadable_image(self, tmp_path: Path) -> None:
        """A path that is not an image names itself in the error."""
        missing = tmp_path / "missing.png"
        dataset = _dataset([str(missing)])

        with pytest.raises(ValueError, match="Could not read image"):
            dataset[0]


class TestKeyPointsDatasetEquality:
    """Tests for how two keypoint datasets are compared."""

    def test_equal_datasets_compare_equal(self) -> None:
        """Two datasets built from the same inputs are equal."""
        assert _dataset(["a.png"]) == _dataset(["a.png"])

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param({"classes": ["dog"]}, id="classes"),
            pytest.param({"keypoint_count": 3}, id="keypoint-count"),
        ],
    )
    def test_differing_fields_compare_unequal(self, other: dict) -> None:
        """A difference in any compared field makes the datasets unequal."""
        base = _dataset(["a.png"])
        changed = _dataset(
            ["a.png"],
            classes=other.get("classes"),
            keypoint_count=other.get("keypoint_count", 2),
        )

        assert base != changed

    def test_other_types_compare_unequal(self) -> None:
        """Comparing against a non-dataset returns False rather than raising."""
        assert _dataset(["a.png"]) != "not a dataset"


class TestKeyPointsDatasetSplit:
    """Tests for splitting a keypoint dataset in two."""

    def test_split_partitions_every_image(self) -> None:
        """The two parts together hold each image exactly once."""
        dataset = _dataset([f"{i}.png" for i in range(10)])

        train, test = dataset.split(split_ratio=0.7, random_state=0)

        assert len(train) == 7
        assert len(test) == 3
        assert set(train.image_paths) | set(test.image_paths) == set(
            dataset.image_paths
        )

    def test_split_carries_the_shape_forward(self) -> None:
        """Both parts keep the keypoint shape of the source dataset."""
        dataset = KeyPointsDataset(
            classes=["person"],
            images=["a.png", "b.png"],
            annotations={
                "a.png": _key_points(keypoint_count=3),
                "b.png": _key_points(keypoint_count=3),
            },
            keypoint_count=3,
            value_dim=2,
        )

        train, test = dataset.split(split_ratio=0.5, random_state=0)

        assert (train.keypoint_count, train.value_dim) == (3, 2)
        assert (test.keypoint_count, test.value_dim) == (3, 2)

    def test_split_does_not_mutate_the_source(self) -> None:
        """Splitting leaves the original dataset intact."""
        dataset = _dataset([f"{i}.png" for i in range(4)])

        dataset.split(split_ratio=0.5, random_state=0)

        assert len(dataset) == 4


class TestKeyPointsDatasetMerge:
    """Tests for merging keypoint datasets."""

    def test_merge_unions_classes_and_remaps_class_ids(self) -> None:
        """Class lists are merged and each class id is renumbered to match."""
        first = _dataset(["a.png"], classes=["cat"], class_ids=[0])
        second = _dataset(["b.png"], classes=["dog"], class_ids=[0])

        merged = KeyPointsDataset.merge([first, second])

        assert merged.classes == ["cat", "dog"]
        np.testing.assert_array_equal(
            merged.annotations["b.png"].class_id, np.array([1])
        )

    def test_merge_keeps_every_image(self) -> None:
        """The merged dataset holds the images of all its inputs."""
        first = _dataset(["a.png"])
        second = _dataset(["b.png"])

        merged = KeyPointsDataset.merge([first, second])

        assert merged.image_paths == ["a.png", "b.png"]

    def test_merge_rejects_an_empty_list(self) -> None:
        """There is no dataset to return for an empty input."""
        with pytest.raises(ValueError, match="empty list"):
            KeyPointsDataset.merge([])

    def test_merge_rejects_differing_keypoint_counts(self) -> None:
        """Skeletons of different sizes cannot share one dataset."""
        first = _dataset(["a.png"], keypoint_count=2)
        second = _dataset(["b.png"], keypoint_count=3)

        with pytest.raises(ValueError, match="different keypoint counts"):
            KeyPointsDataset.merge([first, second])

    def test_merge_rejects_a_repeated_image_path(self) -> None:
        """An image appearing in two datasets is ambiguous, so it is an error."""
        with pytest.raises(ValueError, match="not unique"):
            KeyPointsDataset.merge([_dataset(["a.png"]), _dataset(["a.png"])])


class TestKeyPointsDatasetExport:
    """Tests for writing a keypoint dataset back to disk."""

    def test_rejects_a_non_integer_class_id(self) -> None:
        """A float class_id names its image instead of failing inside numpy."""
        key_points = KeyPoints(
            xy=np.zeros((1, 2, 2), dtype=np.float32),
            class_id=np.array([0.0]),
        )

        with pytest.raises(ValueError, match="non-integer class_id"):
            KeyPointsDataset(
                classes=["person"],
                images=["a.png"],
                annotations={"a.png": key_points},
                keypoint_count=2,
            )

    def test_rejects_colliding_label_names_before_writing(self, tmp_path: Path) -> None:
        """Images sharing a label name are refused with nothing left on disk."""
        image_paths = []
        for name in ("a.png", "a.jpg"):
            image_path = tmp_path / name
            Image.new("RGB", (20, 10)).save(image_path)
            image_paths.append(str(image_path))
        dataset = _dataset(image_paths)
        output = tmp_path / "out"

        with pytest.raises(ValueError, match="YOLO pose annotation"):
            dataset.as_yolo(
                images_directory_path=str(output / "images"),
                annotations_directory_path=str(output / "labels"),
            )

        assert not output.exists()


class TestKeyPointsDatasetFlipIdx:
    """Tests for the `flip_idx` keypoint permutation on the dataset."""

    def test_rejects_a_flip_idx_that_is_not_a_permutation(self) -> None:
        """A `flip_idx` that would drop or duplicate a joint is refused."""
        with pytest.raises(ValueError, match="permutation"):
            KeyPointsDataset(
                classes=["person"],
                images=[],
                annotations={},
                keypoint_count=3,
                flip_idx=[0, 1, 1],
            )

    def test_split_carries_flip_idx_to_both_parts(self) -> None:
        """Both halves of a split keep the permutation of the source."""
        dataset = _dataset(["a.png", "b.png"], keypoint_count=2)
        dataset.flip_idx = [1, 0]

        train, test = dataset.split(split_ratio=0.5, random_state=0)

        assert train.flip_idx == [1, 0]
        assert test.flip_idx == [1, 0]

    def test_merge_rejects_differing_flip_idx(self) -> None:
        """Datasets that disagree on which keypoint is which cannot merge."""
        first = _dataset(["a.png"], keypoint_count=2)
        second = _dataset(["b.png"], keypoint_count=2)
        first.flip_idx = [0, 1]
        second.flip_idx = [1, 0]

        with pytest.raises(ValueError, match="different `flip_idx`"):
            KeyPointsDataset.merge([first, second])

    def test_merge_keeps_a_shared_flip_idx(self) -> None:
        """A permutation common to every input survives the merge."""
        first = _dataset(["a.png"], keypoint_count=2)
        second = _dataset(["b.png"], keypoint_count=2)
        first.flip_idx = [1, 0]
        second.flip_idx = [1, 0]

        merged = KeyPointsDataset.merge([first, second])

        assert merged.flip_idx == [1, 0]

    def test_differing_flip_idx_compares_unequal(self) -> None:
        """Equality is sensitive to the permutation."""
        first = _dataset(["a.png"], keypoint_count=2)
        second = _dataset(["a.png"], keypoint_count=2)
        second.flip_idx = [1, 0]

        assert first != second
