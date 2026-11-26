import json
from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree

from supervision import DetectionDataset, Detections
from supervision import _cv2 as cv2
from supervision.dataset.formats.coco import (
    build_coco_class_index_mapping,
    classes_to_coco_categories,
    coco_annotations_to_detections,
    coco_annotations_to_masks,
    coco_categories_to_classes,
    group_coco_annotations_by_image_id,
    load_coco_annotations,
)
from tests.dataset.formats.helpers import mock_coco_annotation


class TestCocoMissingMaskArea:
    @pytest.mark.parametrize("encoding", ["polygon", "rle", "compressed-rle"])
    def test_decoded_masks_supply_missing_area(self, encoding: str) -> None:
        """Loading segmented labels without area records the decoded pixel count."""
        from supervision.detection.utils.converters import mask_to_rle

        mask = np.zeros((100, 100), dtype=bool)
        mask[:50, :50] = True
        segmentation = (
            [[0, 0, 49, 0, 49, 49, 0, 49]]
            if encoding == "polygon"
            else {
                "counts": mask_to_rle(mask, compressed=encoding == "compressed-rle"),
                "size": [100, 100],
            }
        )
        annotation = {
            "id": 1,
            "category_id": 0,
            "bbox": [0, 0, 80, 80],
            "segmentation": segmentation,
        }

        detections = coco_annotations_to_detections(
            [annotation],
            resolution_wh=(100, 100),
            with_masks=True,
        )

        np.testing.assert_array_equal(detections.data["area"], [2500.0])
        assert "area" not in annotation

    @pytest.mark.parametrize("stored_area", [0.0, 123.0])
    def test_annotated_area_takes_precedence(self, stored_area: float) -> None:
        """Explicit COCO areas remain authoritative even when masks differ."""
        annotation = {
            "category_id": 0,
            "bbox": [0, 0, 80, 80],
            "segmentation": [[0, 0, 49, 0, 49, 49, 0, 49]],
            "area": stored_area,
        }

        detections = coco_annotations_to_detections(
            [annotation],
            resolution_wh=(100, 100),
            with_masks=True,
        )

        np.testing.assert_array_equal(detections.data["area"], [stored_area])

    def test_mixed_annotations_keep_per_instance_area(self) -> None:
        """Missing mask areas and bbox-only areas stay aligned in mixed images."""
        annotations = [
            {"category_id": 0, "bbox": [0, 0, 80, 80]},
            {
                "category_id": 0,
                "bbox": [0, 0, 80, 80],
                "segmentation": [[0, 0, 49, 0, 49, 49, 0, 49]],
            },
        ]

        detections = coco_annotations_to_detections(
            annotations,
            resolution_wh=(100, 100),
            with_masks=True,
        )

        np.testing.assert_array_equal(detections.data["area"], [6400.0, 2500.0])


def _empty_raw_segs(n: int) -> np.ndarray:
    """Object-dtype array of n empty lists for coco_raw_segmentation (bbox-only)."""
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = []
    return arr


@pytest.fixture
def coco_data_with_and_without_segmentation() -> dict[str, object]:
    return {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [
            {"id": 1, "file_name": "with_segmentation.jpg", "width": 5, "height": 5},
            {
                "id": 2,
                "file_name": "with_polygon_segmentation.jpg",
                "width": 5,
                "height": 5,
            },
            {"id": 3, "file_name": "without_segmentation.jpg", "width": 5, "height": 5},
            {"id": 4, "file_name": "without_annotations.jpg", "width": 5, "height": 5},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "segmentation": [[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [3, 0, 2, 2],
                "area": 4,
                "segmentation": {"size": [5, 5], "counts": [15, 2, 3, 2, 3]},
                "iscrowd": 1,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 0,
            },
        ],
    }


@pytest.fixture
def coco_data_with_unannotated_image() -> dict[str, object]:
    return {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [
            {"id": 1, "file_name": "has_segmentation.jpg", "width": 5, "height": 5},
            {"id": 2, "file_name": "no_annotations.jpg", "width": 5, "height": 5},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "iscrowd": 0,
            }
        ],
    }


@pytest.mark.parametrize(
    ("coco_categories", "expected_result", "exception"),
    [
        ([], [], DoesNotRaise()),  # empty coco categories
        (
            [{"id": 0, "name": "fashion-assistant", "supercategory": "none"}],
            ["fashion-assistant"],
            DoesNotRaise(),
        ),  # single coco category with supercategory == "none"
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap"],
            DoesNotRaise(),
        ),  # two coco categories; one with supercategory == "none" and
        # one with supercategory != "none"
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap", "hoodie"],
            DoesNotRaise(),
        ),  # three coco categories; one with supercategory == "none" and
        # two with supercategory != "none"
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap", "hoodie"],
            DoesNotRaise(),
        ),  # three coco categories; one with supercategory == "none" and
        # two with supercategory != "none" (different order)
    ],
)
def test_coco_categories_to_classes(
    coco_categories: list[dict[str, Any]],
    expected_result: list[str],
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = coco_categories_to_classes(coco_categories=coco_categories)
        assert result == expected_result


@pytest.mark.parametrize(
    ("classes", "exception"),
    [
        ([], DoesNotRaise()),  # empty classes
        (["baseball cap"], DoesNotRaise()),  # single class
        (["baseball cap", "hoodie"], DoesNotRaise()),  # two classes
    ],
)
def test_classes_to_coco_categories_and_back_to_classes(
    classes: list[str], exception: AbstractContextManager[object]
) -> None:
    with exception:
        coco_categories = classes_to_coco_categories(classes=classes)
        result = coco_categories_to_classes(coco_categories=coco_categories)
        assert result == classes


@pytest.mark.parametrize(
    ("coco_annotations", "expected_result", "exception"),
    [
        ([], {}, DoesNotRaise()),  # empty coco annotations
        (
            [mock_coco_annotation(annotation_id=0, image_id=0, category_id=0)],
            {0: [mock_coco_annotation(annotation_id=0, image_id=0, category_id=0)]},
            DoesNotRaise(),
        ),  # single coco annotation
        (
            [
                mock_coco_annotation(annotation_id=0, image_id=0, category_id=0),
                mock_coco_annotation(annotation_id=1, image_id=1, category_id=0),
            ],
            {
                0: [mock_coco_annotation(annotation_id=0, image_id=0, category_id=0)],
                1: [mock_coco_annotation(annotation_id=1, image_id=1, category_id=0)],
            },
            DoesNotRaise(),
        ),  # two coco annotations
        (
            [
                mock_coco_annotation(annotation_id=0, image_id=0, category_id=0),
                mock_coco_annotation(annotation_id=1, image_id=1, category_id=1),
                mock_coco_annotation(annotation_id=2, image_id=1, category_id=2),
                mock_coco_annotation(annotation_id=3, image_id=2, category_id=3),
                mock_coco_annotation(annotation_id=4, image_id=3, category_id=1),
                mock_coco_annotation(annotation_id=5, image_id=3, category_id=2),
                mock_coco_annotation(annotation_id=5, image_id=3, category_id=3),
            ],
            {
                0: [
                    mock_coco_annotation(annotation_id=0, image_id=0, category_id=0),
                ],
                1: [
                    mock_coco_annotation(annotation_id=1, image_id=1, category_id=1),
                    mock_coco_annotation(annotation_id=2, image_id=1, category_id=2),
                ],
                2: [
                    mock_coco_annotation(annotation_id=3, image_id=2, category_id=3),
                ],
                3: [
                    mock_coco_annotation(annotation_id=4, image_id=3, category_id=1),
                    mock_coco_annotation(annotation_id=5, image_id=3, category_id=2),
                    mock_coco_annotation(annotation_id=5, image_id=3, category_id=3),
                ],
            },
            DoesNotRaise(),
        ),  # two coco annotations
    ],
)
def test_group_coco_annotations_by_image_id(
    coco_annotations: list[dict[str, Any]],
    expected_result: dict[int, list[dict[str, Any]]],
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = group_coco_annotations_by_image_id(coco_annotations=coco_annotations)
        assert result == expected_result


def test_coco_annotations_to_detections_defaults_missing_area_and_iscrowd() -> None:
    """COCO annotations missing optional area/iscrowd fields still load."""
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [1, 2, 3, 4],
        "segmentation": [],
    }

    detections = coco_annotations_to_detections(
        image_annotations=[annotation],
        resolution_wh=(10, 10),
        with_masks=False,
        use_iscrowd=True,
    )

    assert np.asarray(detections.data["iscrowd"]).tolist() == [0]
    assert np.asarray(detections.data["area"]).tolist() == [12.0]


def test_load_coco_annotations_can_skip_iscrowd_metadata(tmp_path: Path) -> None:
    """use_iscrowd=False omits COCO metadata from loaded Detections.data."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "object"}],
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 2, 3, 4],
                        "iscrowd": 1,
                        "area": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _, image_paths, annotations = load_coco_annotations(
        images_directory_path=str(images_dir),
        annotations_path=str(annotations_path),
        use_iscrowd=False,
    )

    assert image_paths == [str(image_path)]
    assert "iscrowd" not in annotations[str(image_path)].data
    assert "area" not in annotations[str(image_path)].data


def test_detection_dataset_from_coco_accepts_use_iscrowd_false(tmp_path: Path) -> None:
    """DetectionDataset.from_coco forwards use_iscrowd to the COCO loader."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "object"}],
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 2, 3, 4],
                        "iscrowd": 1,
                        "area": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = DetectionDataset.from_coco(
        images_directory_path=str(images_dir),
        annotations_path=str(annotations_path),
        use_iscrowd=False,
    )

    assert "iscrowd" not in dataset.annotations[str(image_path)].data
    assert "area" not in dataset.annotations[str(image_path)].data


def test_coco_box_only_annotation_survives_yolo_and_pascal_voc_export(
    tmp_path: Path,
) -> None:
    """A box-only COCO annotation survives both public dataset exporters."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    image_path = images_directory / "image.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "object"}],
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 1, 7, 7],
                        "area": 49,
                        "segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [6, 6, 3, 3],
                        "area": 9,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = DetectionDataset.from_coco(
        images_directory_path=str(images_directory),
        annotations_path=str(annotations_path),
        mask_rasterizer="supervision",
    )
    detections = dataset.annotations[str(image_path)]
    assert detections.mask is not None
    assert len(detections) == 2
    assert detections.mask[0].any()
    assert not detections.mask[1].any()

    yolo_directory = tmp_path / "yolo"
    dataset.as_yolo(annotations_directory_path=str(yolo_directory))
    yolo_lines = (yolo_directory / "image.txt").read_text().splitlines()
    assert len(yolo_lines) == 2
    assert yolo_lines[1] == "0 0.75000 0.75000 0.30000 0.30000"

    pascal_voc_directory = tmp_path / "pascal_voc"
    dataset.as_pascal_voc(annotations_directory_path=str(pascal_voc_directory))
    xml_string = (pascal_voc_directory / "image.xml").read_text()
    objects = ElementTree.fromstring(xml_string).findall("object")
    assert len(objects) == 2
    assert objects[1].findtext("name") == "object"
    box = objects[1].find("bndbox")
    assert box is not None
    actual = [box.findtext(tag) for tag in ("xmin", "ymin", "xmax", "ymax")]
    expected = ["7", "7", "9", "9"]
    assert actual == expected


def test_detection_dataset_from_coco_preserves_show_progress_positional_arg(
    tmp_path: Path,
) -> None:
    """The fourth positional from_coco argument remains show_progress."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "object"}],
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 2, 3, 4],
                        "iscrowd": 1,
                        "area": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = DetectionDataset.from_coco(
        str(images_dir),
        str(annotations_path),
        False,
        False,
    )

    detections = dataset.annotations[str(image_path)]
    assert np.asarray(detections.data["iscrowd"]).tolist() == [1]
    assert np.asarray(detections.data["area"]).tolist() == [12.0]


@pytest.mark.parametrize(
    (
        "image_annotations",
        "resolution_wh",
        "with_masks",
        "use_iscrowd",
        "expected_result",
        "exception",
    ),
    [
        (
            [],
            (1000, 1000),
            False,
            False,
            Detections.empty(),
            DoesNotRaise(),
        ),  # empty image annotations
        (
            [],
            (1000, 1000),
            False,
            True,
            Detections.empty(),
            DoesNotRaise(),
        ),  # empty image annotations
        (
            [
                mock_coco_annotation(
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                )
            ],
            (1000, 1000),
            False,
            False,
            Detections(
                xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                data={"coco_raw_segmentation": _empty_raw_segs(1)},
            ),
            DoesNotRaise(),
        ),  # single image annotations
        (
            [
                mock_coco_annotation(
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                )
            ],
            (1000, 1000),
            False,
            True,
            Detections(
                xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                data={
                    "iscrowd": np.array([0], dtype=int),
                    "area": np.array([100 * 100]),
                    "coco_raw_segmentation": _empty_raw_segs(1),
                },
            ),
            DoesNotRaise(),
        ),
        (
            [
                mock_coco_annotation(
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                ),
                mock_coco_annotation(
                    category_id=0, bbox=(100, 100, 100, 100), area=100 * 100
                ),
            ],
            (1000, 1000),
            False,
            False,
            Detections(
                xyxy=np.array(
                    [[0, 0, 100, 100], [100, 100, 200, 200]], dtype=np.float32
                ),
                class_id=np.array([0, 0], dtype=int),
                data={"coco_raw_segmentation": _empty_raw_segs(2)},
            ),
            DoesNotRaise(),
        ),  # two image annotations
        (
            [
                mock_coco_annotation(
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                ),
                mock_coco_annotation(
                    category_id=0, bbox=(100, 100, 100, 100), area=100 * 100
                ),
            ],
            (1000, 1000),
            False,
            True,
            Detections(
                xyxy=np.array(
                    [[0, 0, 100, 100], [100, 100, 200, 200]], dtype=np.float32
                ),
                class_id=np.array([0, 0], dtype=int),
                data={
                    "iscrowd": np.array([0, 0], dtype=int),
                    "area": np.array([100 * 100, 100 * 100]),
                    "coco_raw_segmentation": _empty_raw_segs(2),
                },
            ),
            DoesNotRaise(),
        ),
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                )
            ],
            (5, 5),
            True,
            False,
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # single image annotations with mask as polygon
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [3, 3, 4, 3, 4, 4, 3, 4],
                    ],
                )
            ],
            (5, 5),
            True,
            False,
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # single image annotation with disjoint polygon segments
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                )
            ],
            (5, 5),
            True,
            True,
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
                data={"iscrowd": np.array([0], dtype=int), "area": np.array([25])},
            ),
            DoesNotRaise(),
        ),
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation={
                        "size": [5, 5],
                        "counts": [0, 15, 2, 3, 2, 3],
                    },
                    iscrowd=True,
                )
            ],
            (5, 5),
            True,
            False,
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # single image annotations with mask, RLE segmentation mask
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation={
                        "size": [5, 5],
                        "counts": [0, 15, 2, 3, 2, 3],
                    },
                    iscrowd=True,
                )
            ],
            (5, 5),
            True,
            True,
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
                data={"iscrowd": np.array([1], dtype=int), "area": np.array([25])},
            ),
            DoesNotRaise(),
        ),
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                ),
                mock_coco_annotation(
                    category_id=0,
                    bbox=(3, 0, 2, 2),
                    area=2 * 2,
                    segmentation={
                        "size": [5, 5],
                        "counts": [15, 2, 3, 2, 3],
                    },
                    iscrowd=True,
                ),
            ],
            (5, 5),
            True,
            False,
            Detections(
                xyxy=np.array([[0, 0, 5, 5], [3, 0, 5, 2]], dtype=np.float32),
                class_id=np.array([0, 0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ],
                        [
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # two image annotations with mask, one mask as polygon and second as RLE
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                ),
                mock_coco_annotation(
                    category_id=0,
                    bbox=(3, 0, 2, 2),
                    area=2 * 2,
                    segmentation={
                        "size": [5, 5],
                        "counts": [15, 2, 3, 2, 3],
                    },
                    iscrowd=True,
                ),
            ],
            (5, 5),
            True,
            True,
            Detections(
                xyxy=np.array([[0, 0, 5, 5], [3, 0, 5, 2]], dtype=np.float32),
                class_id=np.array([0, 0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ],
                        [
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    ],
                    dtype=bool,
                ),
                data={
                    "iscrowd": np.array([0, 1], dtype=int),
                    "area": np.array([25, 4]),
                },
            ),
            DoesNotRaise(),
        ),  # two image annotations with mask, one mask as polygon with iscrowd,
        # and second as RLE without iscrowd
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(3, 0, 2, 2),
                    area=2 * 2,
                    segmentation={
                        "size": [5, 5],
                        "counts": [15, 2, 3, 2, 3],
                    },
                    iscrowd=True,
                ),
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                ),
            ],
            (5, 5),
            True,
            False,
            Detections(
                xyxy=np.array([[3, 0, 5, 2], [0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int),
                mask=np.array(
                    [
                        [
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ],
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # two image annotations with mask, first mask as RLE and second as polygon
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(3, 0, 2, 2),
                    area=2 * 2,
                    segmentation={
                        "size": [5, 5],
                        "counts": [15, 2, 3, 2, 3],
                    },
                    iscrowd=True,
                ),
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation=[[0, 0, 2, 0, 2, 2, 4, 2, 4, 4, 0, 4]],
                ),
            ],
            (5, 5),
            True,
            True,
            Detections(
                xyxy=np.array([[3, 0, 5, 2], [0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int),
                mask=np.array(
                    [
                        [
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ],
                    ],
                    dtype=bool,
                ),
                data={
                    "iscrowd": np.array([1, 0], dtype=int),
                    "area": np.array([4, 25]),
                },
            ),
            DoesNotRaise(),
        ),  # two image annotations with mask, first mask as RLE with is crowd,
        # and second as polygon without iscrowd
        (
            [
                mock_coco_annotation(
                    category_id=0,
                    bbox=(0, 0, 4, 4),
                    area=4 * 4,
                    segmentation={
                        "size": [4, 4],
                        "counts": "52203",
                    },
                    iscrowd=True,
                )
            ],
            (4, 4),
            True,
            False,
            Detections(
                xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [False, False, False, False],
                            [False, True, True, False],
                            [False, True, True, False],
                            [False, False, False, False],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            DoesNotRaise(),
        ),  # single iscrowd annotation with compressed COCO RLE string counts
    ],
)
def test_coco_annotations_to_detections(
    image_annotations: list[dict[str, Any]],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    use_iscrowd: bool,
    expected_result: Detections,
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=resolution_wh,
            with_masks=with_masks,
            use_iscrowd=use_iscrowd,
        )
        assert result == expected_result


@pytest.mark.parametrize(
    ("coco_categories", "target_classes", "expected_result", "exception"),
    [
        ([], [], {}, DoesNotRaise()),  # empty coco categories
        (
            [{"id": 0, "name": "fashion-assistant", "supercategory": "none"}],
            ["fashion-assistant"],
            {0: 0},
            DoesNotRaise(),
        ),  # single coco category starting from 0
        (
            [{"id": 1, "name": "fashion-assistant", "supercategory": "none"}],
            ["fashion-assistant"],
            {1: 0},
            DoesNotRaise(),
        ),  # single coco category starting from 1
        (
            [
                {"id": 0, "name": "fashion-assistant", "supercategory": "none"},
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["fashion-assistant", "baseball cap", "hoodie"],
            {0: 0, 1: 1, 2: 2},
            DoesNotRaise(),
        ),  # three coco categories
        (
            [
                {"id": 2, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["baseball cap", "hoodie"],
            {2: 1, 1: 0},
            DoesNotRaise(),
        ),  # two coco categories
        (
            [
                {"id": 3, "name": "hoodie", "supercategory": "fashion-assistant"},
                {"id": 1, "name": "baseball cap", "supercategory": "fashion-assistant"},
            ],
            ["baseball cap", "hoodie"],
            {3: 1, 1: 0},
            DoesNotRaise(),
        ),  # two coco categories with missing category
    ],
)
def test_build_coco_class_index_mapping(
    coco_categories: list[dict[str, Any]],
    target_classes: list[str],
    expected_result: dict[int, int],
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = build_coco_class_index_mapping(
            coco_categories=coco_categories, target_classes=target_classes
        )
        assert result == expected_result


def test_load_coco_annotations_infers_masks_from_segmentation_field(
    tmp_path: Path, coco_data_with_and_without_segmentation: dict[str, object]
) -> None:
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    annotations_path.write_text(
        json.dumps(coco_data_with_and_without_segmentation), encoding="utf-8"
    )

    classes, images, annotations = load_coco_annotations(
        images_directory_path=str(images_directory),
        annotations_path=str(annotations_path),
        force_masks=False,
        use_iscrowd=True,
        mask_rasterizer="supervision",
    )

    assert classes == ["object"]
    assert len(images) == 4

    with_segmentation_path = str(images_directory / "with_segmentation.jpg")
    with_segmentation = annotations[with_segmentation_path]
    assert with_segmentation.mask is not None
    assert with_segmentation.mask.shape == (2, 5, 5)
    assert np.array_equal(with_segmentation.data["iscrowd"], np.array([0, 1]))

    with_polygon_segmentation_path = str(
        images_directory / "with_polygon_segmentation.jpg"
    )
    with_polygon_segmentation = annotations[with_polygon_segmentation_path]
    assert with_polygon_segmentation.mask is not None
    assert with_polygon_segmentation.mask.shape == (1, 5, 5)
    assert with_polygon_segmentation.mask[0].any()

    without_segmentation_path = str(images_directory / "without_segmentation.jpg")
    without_segmentation = annotations[without_segmentation_path]
    assert without_segmentation.mask is None
    assert np.array_equal(
        without_segmentation.xyxy, np.array([[0, 0, 2, 2]], dtype=np.float32)
    )

    without_annotations_path = str(images_directory / "without_annotations.jpg")
    assert annotations[without_annotations_path] == Detections.empty()


def test_load_coco_annotations_force_masks_with_no_annotations(
    tmp_path: Path, coco_data_with_unannotated_image: dict[str, object]
) -> None:
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    annotations_path.write_text(
        json.dumps(coco_data_with_unannotated_image),
        encoding="utf-8",
    )

    _, _, annotations = load_coco_annotations(
        images_directory_path=str(images_directory),
        annotations_path=str(annotations_path),
        force_masks=True,
        mask_rasterizer="supervision",
    )

    has_segmentation_path = str(images_directory / "has_segmentation.jpg")
    has_segmentation = annotations[has_segmentation_path]
    assert has_segmentation.mask is not None
    assert has_segmentation.mask.shape == (1, 5, 5)

    no_annotations_path = str(images_directory / "no_annotations.jpg")
    assert annotations[no_annotations_path] == Detections.empty()


def test_coco_annotations_to_masks_handles_rle_polygon_and_invalid_dict() -> None:
    image_annotations = [
        {"id": 1, "segmentation": {"size": [5, 5], "counts": [15, 2, 3, 2, 3]}},
        {"id": 2, "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]},
        {"id": 3, "segmentation": {"size": [5, 5]}},
    ]

    with pytest.warns(
        UserWarning,
        match=(
            "Skipping annotation 3: segmentation is a dict but missing 'counts' key "
            r"\(expected RLE format\)"
        ),
    ):
        masks = coco_annotations_to_masks(
            image_annotations=image_annotations,
            resolution_wh=(5, 5),
        )

    assert masks.shape == (3, 5, 5)
    assert masks[0].any()
    assert masks[1].any()
    assert not masks[2].any()


def test_coco_annotations_to_masks_rounds_decimal_polygon_vertices() -> None:
    """Sub-pixel polygon vertices snap to the nearest pixel, as in the other loaders."""
    image_annotations = [
        {"id": 1, "segmentation": [[2.6, 2.6, 7.6, 2.6, 7.6, 7.6, 2.6, 7.6]]}
    ]

    masks = coco_annotations_to_masks(
        image_annotations=image_annotations,
        resolution_wh=(12, 12),
    )

    expected_mask = np.zeros((12, 12), dtype=bool)
    expected_mask[3:9, 3:9] = True
    np.testing.assert_array_equal(masks[0], expected_mask)


@pytest.mark.parametrize("vertex", [float("nan"), float("inf")])
def test_coco_annotations_to_masks_rejects_non_finite_polygon_vertex(
    vertex: float,
) -> None:
    """A polygon vertex that is not a finite number is rejected with a clear error."""
    image_annotations = [{"id": 7, "segmentation": [[0, 0, 4, 0, vertex, 4, 0, 4]]}]

    with pytest.raises(ValueError, match="id=7 has a vertex that is not a finite"):
        coco_annotations_to_masks(
            image_annotations=image_annotations,
            resolution_wh=(5, 5),
        )


@pytest.mark.parametrize(
    "file_name",
    [".", "", "subdir/.."],
)
def test_load_coco_annotations_rejects_file_name_resolving_to_images_directory(
    tmp_path: Path,
    file_name: str,
) -> None:
    """Reject file_name resolving to the images directory itself (equality guard)."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [{"id": 1, "file_name": file_name, "width": 5, "height": 5}],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    with pytest.raises(ValueError, match="resolves to the images directory itself"):
        load_coco_annotations(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
        )


@pytest.mark.parametrize(
    "malicious_file_name",
    [
        "../escape.txt",
        "../../escape.txt",
        "subdir/../../escape.txt",
    ],
)
def test_load_coco_annotations_rejects_file_name_outside_images_directory(
    tmp_path: Path,
    malicious_file_name: str,
) -> None:
    """Reject relative traversal file_name values that escape the images directory."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [
            {
                "id": 1,
                "file_name": malicious_file_name,
                "width": 5,
                "height": 5,
            }
        ],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the images directory"):
        load_coco_annotations(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
        )


def test_load_coco_annotations_rejects_absolute_file_name(tmp_path: Path) -> None:
    """Reject absolute file_name values that escape the images directory."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [
            {
                "id": 1,
                "file_name": "/etc/passwd",
                "width": 5,
                "height": 5,
            }
        ],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the images directory"):
        load_coco_annotations(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
        )


def test_load_coco_annotations_rejects_file_name_resolving_to_directory(
    tmp_path: Path,
) -> None:
    """Reject file_name resolving to a subdirectory inside images/ (is_dir guard)."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    (images_directory / "subdir").mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [{"id": 1, "file_name": "subdir", "width": 5, "height": 5}],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    with pytest.raises(ValueError, match="resolves to directory"):
        load_coco_annotations(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
        )


def test_load_coco_annotations_rejects_unresolvable_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject file_name values whose resolved path cannot be computed."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [{"id": 1, "file_name": "bad.jpg", "width": 5, "height": 5}],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    original_resolve = Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        """Raise OSError for the bad image path, delegate for every other path."""
        if self == images_directory / "bad.jpg":
            raise OSError("unresolvable path")
        return original_resolve(self, strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    with pytest.raises(ValueError, match="invalid path"):
        load_coco_annotations(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
        )


def test_load_coco_annotations_accepts_valid_nested_file_name(tmp_path: Path) -> None:
    """Accept a legitimate nested file_name inside images/ without raising."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    (images_directory / "train").mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [{"id": 1, "file_name": "train/image.jpg", "width": 5, "height": 5}],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    _, _, annotations = load_coco_annotations(
        images_directory_path=str(images_directory),
        annotations_path=str(annotations_path),
    )
    expected_path = str(images_directory / "train" / "image.jpg")
    assert expected_path in annotations


def test_load_coco_annotations_rejects_duplicate_resolved_file_names(
    tmp_path: Path,
) -> None:
    """Aliases for the same file resolve to one canonical COCO entry."""
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    (images_directory / "nested").mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [
            {"id": 1, "file_name": "image.jpg", "width": 5, "height": 5},
            {"id": 2, "file_name": "nested/../image.jpg", "width": 5, "height": 5},
        ],
        "annotations": [],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate entries for image"):
        load_coco_annotations(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
        )


def test_load_coco_annotations_force_masks_handles_missing_segmentation(
    tmp_path: Path,
) -> None:
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    annotations_path = tmp_path / "annotations.json"

    coco_data = {
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [{"id": 1, "file_name": "image.jpg", "width": 5, "height": 5}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
    }
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    _, _, annotations = load_coco_annotations(
        images_directory_path=str(images_directory),
        annotations_path=str(annotations_path),
        force_masks=True,
    )

    image_path = str(images_directory / "image.jpg")
    image_annotations = annotations[image_path]
    assert isinstance(image_annotations.mask, np.ndarray)
    assert image_annotations.mask.shape == (1, 5, 5)
    assert not image_annotations.mask.any()
    assert np.array_equal(image_annotations.xyxy, np.array([[0, 0, 2, 2]], dtype=float))


@pytest.fixture
def coco_data_with_multi_segment_segmentation() -> dict[str, object]:
    return {
        "categories": [
            {
                "id": 1,
                "name": "cat_eye",
                "supercategory": "animal_parts",
            }
        ],
        "images": [
            {
                "id": 1,
                "file_name": "image.jpg",
                "width": 5,
                "height": 5,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                # bbox spans both segments; area = sum of two 1x1 polygon areas
                "bbox": [0, 0, 5, 5],
                "area": 2,
                "segmentation": [
                    [0, 0, 1, 0, 1, 1, 0, 1],
                    [3, 3, 4, 3, 4, 4, 3, 4],
                ],
                "iscrowd": 0,
            }
        ],
    }


class TestFromCocoMasks:
    """Integration: DetectionDataset.from_coco loads multi-segment masks."""

    @pytest.mark.parametrize("force_masks", [False, True])
    def test_multi_segment_masks_merged(
        self,
        tmp_path: Path,
        coco_data_with_multi_segment_segmentation: dict[str, object],
        force_masks: bool,
    ) -> None:
        """Multi-segment masks merge correctly for both force_masks values."""
        images_directory = tmp_path / "images"
        images_directory.mkdir()
        annotations_path = tmp_path / "annotations.json"

        annotations_path.write_text(
            json.dumps(coco_data_with_multi_segment_segmentation), encoding="utf-8"
        )

        dataset = DetectionDataset.from_coco(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
            force_masks=force_masks,
            mask_rasterizer="supervision",
        )

        annotation = dataset.annotations[str(images_directory / "image.jpg")]
        assert annotation.mask is not None
        assert annotation.mask.shape == (1, 5, 5)
        np.testing.assert_array_equal(
            annotation.mask,
            np.array(
                [
                    [
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                    ]
                ],
                dtype=bool,
            ),
        )

    def test_multi_segment_masks_uneven_length_no_value_error(
        self, tmp_path: Path
    ) -> None:
        """Uneven-length segments load without ValueError (issue #1209 regression)."""
        images_directory = tmp_path / "images"
        images_directory.mkdir()
        annotations_path = tmp_path / "annotations.json"

        coco_data = {
            "categories": [
                {"id": 1, "name": "cat_eye", "supercategory": "animal_parts"}
            ],
            "images": [{"id": 1, "file_name": "image.jpg", "width": 5, "height": 5}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 5, 5],
                    "area": 2,
                    "segmentation": [
                        [0, 0, 1, 0, 1, 1, 0, 1],  # 4 points (8 coords)
                        [3, 3, 4, 3, 4, 4, 3, 4, 2, 4],  # 5 points (10 coords)
                    ],
                    "iscrowd": 0,
                }
            ],
        }
        annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

        dataset = DetectionDataset.from_coco(
            images_directory_path=str(images_directory),
            annotations_path=str(annotations_path),
            mask_rasterizer="supervision",
        )

        annotation = dataset.annotations[str(images_directory / "image.jpg")]
        assert annotation.mask is not None
        assert annotation.mask.shape == (1, 5, 5)


# --- Regression: legacy 0-indexed COCO files still load correctly (#1181) ---


def test_from_coco_loads_legacy_zero_indexed_category_ids(tmp_path: Path) -> None:
    """COCO files with 0-indexed category ids (written by supervision <=0.28.x) must
    still load and produce correct internal 0-indexed class_ids."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_path = images_dir / "img.jpg"
    assert cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    coco_data = {
        "categories": [
            {"id": 0, "name": "cat", "supercategory": "none"},
            {"id": 1, "name": "dog", "supercategory": "none"},
        ],
        "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 3, 3],
                "area": 9,
                "iscrowd": 0,
            },
        ],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(coco_data), encoding="utf-8")

    dataset = DetectionDataset.from_coco(
        images_directory_path=str(images_dir),
        annotations_path=str(annotations_path),
    )

    assert dataset.classes == ["cat", "dog"]
    dets = dataset.annotations[str(img_path)]
    assert dets.class_id is not None
    assert sorted(dets.class_id.tolist()) == [0, 1]
