from __future__ import annotations

import json
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path

import cv2
import numpy as np
import pytest

from supervision import DetectionDataset, Detections
from supervision.config import CLASS_NAME_DATA_FIELD
from tests.helpers import _create_detections, create_yolo_dataset


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


def test_split_as_coco_preserves_annotation_id_continuity(tmp_path: Path) -> None:
    image_paths = []
    annotations = {}
    detection_counts = [2, 1, 3, 1]

    for index, detection_count in enumerate(detection_counts):
        image_path = tmp_path / f"image-{index}.png"
        cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
        image_paths.append(str(image_path))
        annotations[str(image_path)] = _create_detections(
            xyxy=[[i, i, i + 1, i + 1] for i in range(detection_count)],
            class_id=[0] * detection_count,
        )

    dataset = DetectionDataset(
        classes=["object"], images=image_paths, annotations=annotations
    )
    train_dataset, holdout_dataset = dataset.split(split_ratio=0.5, shuffle=False)
    valid_dataset, test_dataset = holdout_dataset.split(split_ratio=0.5, shuffle=False)

    train_annotations_path = tmp_path / "train.json"
    valid_annotations_path = tmp_path / "valid.json"
    test_annotations_path = tmp_path / "test.json"

    train_dataset.as_coco(annotations_path=str(train_annotations_path))
    valid_dataset.as_coco(annotations_path=str(valid_annotations_path))
    test_dataset.as_coco(annotations_path=str(test_annotations_path))

    train_ids = json.loads(train_annotations_path.read_text())["annotations"]
    valid_ids = json.loads(valid_annotations_path.read_text())["annotations"]
    test_ids = json.loads(test_annotations_path.read_text())["annotations"]

    train_annotation_ids = [annotation["id"] for annotation in train_ids]
    valid_annotation_ids = [annotation["id"] for annotation in valid_ids]
    test_annotation_ids = [annotation["id"] for annotation in test_ids]

    assert train_annotation_ids == [1, 2, 3]
    assert valid_annotation_ids == [4, 5, 6]
    assert test_annotation_ids == [7]
    assert len(
        train_annotation_ids + valid_annotation_ids + test_annotation_ids
    ) == len(set(train_annotation_ids + valid_annotation_ids + test_annotation_ids))
