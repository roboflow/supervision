from contextlib import ExitStack as DoesNotRaise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from supervision import DetectionDataset, Detections
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.utils.internal import SupervisionWarnings
from tests.helpers import _create_detections, create_yolo_dataset


def _create_image(fill_value: int) -> npt.NDArray[np.uint8]:
    return np.full((4, 4, 3), fill_value, dtype=np.uint8)


@pytest.mark.parametrize(
    ("dataset_list", "expected_result", "exception"),
    [
        (
            [],
            DetectionDataset(classes=[], images=[], annotations={}),
            DoesNotRaise(),
        ),  # empty dataset list
        (
            [DetectionDataset(classes=[], images=[], annotations={})],
            DetectionDataset(classes=[], images=[], annotations={}),
            DoesNotRaise(),
        ),  # single empty dataset
        (
            [
                DetectionDataset(classes=["dog", "person"], images=[], annotations={}),
                DetectionDataset(classes=["dog", "person"], images=[], annotations={}),
            ],
            DetectionDataset(classes=["dog", "person"], images=[], annotations={}),
            DoesNotRaise(),
        ),  # two datasets; no images and annotations, the same classes
        (
            [
                DetectionDataset(classes=["dog", "person"], images=[], annotations={}),
                DetectionDataset(classes=["cat"], images=[], annotations={}),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"], images=[], annotations={}
            ),
            DoesNotRaise(),
        ),  # two datasets; no images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=["image-1.png", "image-2.png"],
                    annotations={
                        "image-1.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(classes=[], images=[], annotations={}),
            ],
            DetectionDataset(
                classes=["dog", "person"],
                images=["image-1.png", "image-2.png"],
                annotations={
                    "image-1.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[0]
                    ),
                    "image-2.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[1]
                    ),
                },
            ),
            DoesNotRaise(),
        ),  # two datasets; images and annotations, the same classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=["image-1.png", "image-2.png"],
                    annotations={
                        "image-1.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(classes=["cat"], images=[], annotations={}),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"],
                images=["image-1.png", "image-2.png"],
                annotations={
                    "image-1.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[1]
                    ),
                    "image-2.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[2]
                    ),
                },
            ),
            DoesNotRaise(),
        ),  # two datasets; images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=["image-1.png", "image-2.png"],
                    annotations={
                        "image-1.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(
                    classes=["cat"],
                    images=["image-3.png"],
                    annotations={
                        "image-3.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                    },
                ),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"],
                images=["image-1.png", "image-2.png", "image-3.png"],
                annotations={
                    "image-1.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[1]
                    ),
                    "image-2.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[2]
                    ),
                    "image-3.png": _create_detections(
                        xyxy=[[0, 0, 10, 10]], class_id=[0]
                    ),
                },
            ),
            DoesNotRaise(),
        ),  # two datasets; images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=["image-1.png", "image-2.png"],
                    annotations={
                        "image-1.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(
                    classes=["dog", "person"],
                    images=["image-2.png", "image-3.png"],
                    annotations={
                        "image-2.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-3.png": _create_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
            ],
            None,
            pytest.raises(ValueError, match="not unique across datasets"),
        ),
    ],
)
def test_dataset_merge(
    dataset_list: list[DetectionDataset],
    expected_result: DetectionDataset | None,
    exception: Exception,
) -> None:
    """
    Verify that multiple DetectionDataset objects can be successfully merged.

    Ensures that multiple `DetectionDataset` objects can be merged into single dataset.
    This is vital for users who need to combine data from different sources or
    augment their datasets with additional labeled examples.
    """
    with exception:
        result = DetectionDataset.merge(dataset_list=dataset_list)
        assert result == expected_result


class TestClassNamePopulation:
    """Verify that DetectionDataset populates CLASS_NAME_DATA_FIELD on init."""

    def test_class_name_populated_on_init(self) -> None:
        """Basic case: class_name data field is set from classes and class_id."""
        dataset = DetectionDataset(
            classes=["dog", "cat"],
            images=["img1.png"],
            annotations={
                "img1.png": _create_detections(
                    xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]],
                    class_id=[0, 1],
                ),
            },
        )
        annotation = dataset.annotations["img1.png"]
        assert CLASS_NAME_DATA_FIELD in annotation.data
        np.testing.assert_array_equal(
            annotation.data[CLASS_NAME_DATA_FIELD],
            np.array(["dog", "cat"]),
        )

    def test_class_name_with_empty_annotations(self) -> None:
        """Empty Detections should not raise an error."""
        dataset = DetectionDataset(
            classes=["dog"],
            images=["img1.png"],
            annotations={"img1.png": Detections.empty()},
        )
        annotation = dataset.annotations["img1.png"]
        assert CLASS_NAME_DATA_FIELD in annotation.data
        assert len(annotation.data[CLASS_NAME_DATA_FIELD]) == 0

    def test_class_name_with_empty_classes(self) -> None:
        """When classes is empty, class_name should not be populated."""
        dataset = DetectionDataset(
            classes=[],
            images=[],
            annotations={},
        )
        assert len(dataset.annotations) == 0

    def test_class_name_after_merge(self) -> None:
        """After merging datasets, class_name must match remapped class_id."""
        ds1 = DetectionDataset(
            classes=["dog", "person"],
            images=["img1.png"],
            annotations={
                "img1.png": _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            },
        )
        ds2 = DetectionDataset(
            classes=["cat"],
            images=["img2.png"],
            annotations={
                "img2.png": _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            },
        )
        merged = DetectionDataset.merge([ds1, ds2])

        # merged.classes is ["cat", "dog", "person"]
        # ds1's dog (0) -> dog (1), ds2's cat (0) -> cat (0)
        ann1 = merged.annotations["img1.png"]
        assert CLASS_NAME_DATA_FIELD in ann1.data
        np.testing.assert_array_equal(
            ann1.data[CLASS_NAME_DATA_FIELD], np.array(["dog"])
        )

        ann2 = merged.annotations["img2.png"]
        assert CLASS_NAME_DATA_FIELD in ann2.data
        np.testing.assert_array_equal(
            ann2.data[CLASS_NAME_DATA_FIELD], np.array(["cat"])
        )

    def test_class_name_from_yolo(self, tmp_path: Path) -> None:
        """Integration test: from_yolo should produce class_name data."""
        dataset_info = create_yolo_dataset(
            str(tmp_path), num_images=2, classes=["cat", "dog"]
        )
        dataset = DetectionDataset.from_yolo(
            images_directory_path=dataset_info["images_dir"],
            annotations_directory_path=dataset_info["labels_dir"],
            data_yaml_path=dataset_info["data_yaml_path"],
        )

        for _, annotation in dataset.annotations.items():
            if annotation.class_id is not None and len(annotation.class_id) > 0:
                assert CLASS_NAME_DATA_FIELD in annotation.data
                expected_names = np.array(dataset.classes)[annotation.class_id]
                np.testing.assert_array_equal(
                    annotation.data[CLASS_NAME_DATA_FIELD], expected_names
                )


class TestDetectionDatasetInMemoryImages:
    """Verify DetectionDataset keeps dict-provided images in memory (DAT-01)."""

    @staticmethod
    def _build_dataset(
        images: dict[str, npt.NDArray[np.uint8]],
    ) -> DetectionDataset:
        annotations = {
            path: _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0])
            for path in images
        }
        return DetectionDataset(classes=["dog"], images=images, annotations=annotations)

    def test_getitem_returns_in_memory_image(self) -> None:
        """Indexing a dict-constructed dataset returns the in-memory array."""
        image = _create_image(fill_value=7)
        dataset = self._build_dataset({"imgX.jpg": image})

        image_path, loaded_image, _ = dataset[0]

        assert image_path == "imgX.jpg"
        np.testing.assert_array_equal(loaded_image, image)

    def test_len_counts_in_memory_images(self) -> None:
        """`len` of a dict-constructed dataset equals the number of provided images."""
        images = {
            "img1.jpg": _create_image(fill_value=1),
            "img2.jpg": _create_image(fill_value=2),
        }

        dataset = self._build_dataset(images)

        assert len(dataset) == 2

    def test_merge_preserves_in_memory_pixel_access(self) -> None:
        """Merging two in-memory datasets keeps pixel access via public __getitem__."""
        image_1 = _create_image(fill_value=10)
        image_2 = _create_image(fill_value=20)
        ds_1 = self._build_dataset({"img1.jpg": image_1})
        ds_2 = self._build_dataset({"img2.jpg": image_2})

        merged = DetectionDataset.merge([ds_1, ds_2])

        assert len(merged) == 2
        _, loaded_1, _ = merged[0]
        _, loaded_2, _ = merged[1]
        np.testing.assert_array_equal(loaded_1, image_1)
        np.testing.assert_array_equal(loaded_2, image_2)

    def test_iteration_yields_in_memory_images(self) -> None:
        """Iteration yields (path, image, annotation) with correct pixels."""
        images = {
            "img1.jpg": _create_image(fill_value=1),
            "img2.jpg": _create_image(fill_value=2),
        }
        dataset = self._build_dataset(images)

        entries = list(dataset)

        assert [path for path, _, _ in entries] == ["img1.jpg", "img2.jpg"]
        for image_path, loaded_image, annotation in entries:
            np.testing.assert_array_equal(loaded_image, images[image_path])
            assert annotation is dataset.annotations[image_path]

    def test_dict_input_emits_deprecation_warning(self) -> None:
        """Passing a dict of images emits the SupervisionWarnings deprecation notice."""
        with pytest.warns(SupervisionWarnings, match="deprecated"):
            self._build_dataset({"img1.jpg": _create_image(fill_value=3)})

    def test_eq_reflexive_in_memory(self) -> None:
        """In-memory dataset equals itself (reflexive __eq__ via pixel comparison)."""
        images = {
            "img1.jpg": _create_image(fill_value=1),
            "img2.jpg": _create_image(fill_value=2),
        }
        dataset = self._build_dataset(images)

        assert dataset == dataset

    def test_eq_same_pixels_returns_true(self) -> None:
        """Two in-memory datasets with identical images and annotations are equal."""
        images = {"img1.jpg": _create_image(fill_value=5)}
        ds_a = self._build_dataset(images)
        ds_b = self._build_dataset(dict(images))

        assert ds_a == ds_b

    def test_eq_different_pixels_returns_false(self) -> None:
        """In-memory datasets with different pixel data are not equal."""
        ds_a = self._build_dataset({"img1.jpg": _create_image(fill_value=1)})
        ds_b = self._build_dataset({"img1.jpg": _create_image(fill_value=2)})

        assert ds_a != ds_b


class TestDetectionDatasetExportCollisions:
    """Regression tests for the basename-collision guard on export (DAT-04)."""

    def test_as_yolo_raises_on_same_basename_images(self, tmp_path: Path) -> None:
        """Same-basename images from different directories must not overwrite."""
        dataset = DetectionDataset(
            classes=["cat"],
            images=["dir_a/img.png", "dir_b/img.png"],
            annotations={
                "dir_a/img.png": _create_detections(
                    xyxy=[[0, 0, 10, 10]], class_id=[0]
                ),
                "dir_b/img.png": _create_detections(
                    xyxy=[[0, 0, 10, 10]], class_id=[0]
                ),
            },
        )

        with pytest.raises(ValueError, match="both map to image file"):
            dataset.as_yolo(images_directory_path=str(tmp_path / "images"))
