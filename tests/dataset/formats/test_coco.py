from __future__ import annotations

import json
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path

import cv2
import numpy as np
import pytest

from supervision import DetectionDataset, Detections
from supervision.dataset.formats.coco import (
    build_coco_class_index_mapping,
    classes_to_coco_categories,
    coco_annotations_to_detections,
    coco_categories_to_classes,
    detections_to_coco_annotations,
    group_coco_annotations_by_image_id,
    load_coco_annotations,
    save_coco_annotations,
)


def mock_coco_annotation(
    annotation_id: int = 0,
    image_id: int = 0,
    category_id: int = 0,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    area: float = 0.0,
    segmentation: list[list] | dict | None = None,
    iscrowd: bool = False,
) -> dict:
    if not segmentation:
        segmentation = []
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": area,
        "segmentation": segmentation,
        "iscrowd": int(iscrowd),
    }


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
    coco_categories: list[dict], expected_result: list[str], exception: Exception
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
    classes: list[str], exception: Exception
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
    coco_annotations: list[dict], expected_result: dict, exception: Exception
) -> None:
    with exception:
        result = group_coco_annotations_by_image_id(coco_annotations=coco_annotations)
        assert result == expected_result


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
    image_annotations: list[dict],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    use_iscrowd: bool,
    expected_result: Detections,
    exception: Exception,
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
    coco_categories: list[dict],
    target_classes: list[str],
    expected_result: dict[int, int],
    exception: Exception,
) -> None:
    with exception:
        result = build_coco_class_index_mapping(
            coco_categories=coco_categories, target_classes=target_classes
        )
        assert result == expected_result


@pytest.mark.parametrize(
    ("detections", "image_id", "annotation_id", "expected_result", "exception"),
    [
        (
            Detections(
                xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1, bbox=(0, 0, 100, 100), area=100 * 100
                )
            ],
            DoesNotRaise(),
        ),  # no segmentation mask; internal class_id 0 -> COCO category_id 1
        (
            Detections(
                xyxy=np.array([[0, 0, 4, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 0],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 4, 5),
                    area=4 * 5,
                    segmentation=[[0, 0, 0, 4, 3, 4, 3, 0]],
                )
            ],
            DoesNotRaise(),
        ),  # segmentation mask in single component,no holes in mask,
        # expects polygon mask
        (
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation={
                        "size": [5, 5],
                        "counts": [0, 3, 2, 3, 2, 3, 5, 2, 3, 2],
                    },
                    iscrowd=True,
                )
            ],
            DoesNotRaise(),
        ),  # segmentation mask with 2 components, no holes in mask, expects RLE mask
        (
            Detections(
                xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array(
                    [
                        [
                            [0, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 0, 0, 1],
                            [1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ],
                    dtype=bool,
                ),
            ),
            0,
            0,
            [
                mock_coco_annotation(
                    category_id=1,
                    bbox=(0, 0, 5, 5),
                    area=5 * 5,
                    segmentation={
                        "size": [5, 5],
                        "counts": [2, 10, 2, 3, 2, 6],
                    },
                    iscrowd=True,
                )
            ],
            DoesNotRaise(),
        ),  # seg mask in single component, with holes in mask, expects RLE mask
    ],
)
def test_detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    expected_result: list[dict],
    exception: Exception,
) -> None:
    with exception:
        result, _ = detections_to_coco_annotations(
            detections=detections,
            image_id=image_id,
            annotation_id=annotation_id,
        )
        assert result == expected_result


def test_detections_to_coco_annotations_handles_empty_approximated_polygons() -> None:
    detections = Detections(
        xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=np.array(
            [
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                ]
            ],
            dtype=bool,
        ),
    )

    with pytest.warns(Warning, match="mask approximation returned no polygons"):
        annotations, _ = detections_to_coco_annotations(
            detections=detections,
            image_id=0,
            annotation_id=0,
            max_image_area_percentage=0.01,
        )

    assert len(annotations) == 1
    assert annotations[0]["segmentation"] == []
    assert annotations[0]["iscrowd"] == 0


def test_detections_to_coco_annotations_preserves_area_from_data() -> None:
    """area stored in detections.data should be used instead of bbox area."""
    detections = Detections(
        xyxy=np.array([[10.0, 20.0, 110.0, 120.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={"iscrowd": np.array([0], dtype=int), "area": np.array([5000.0])},
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 5000.0
    assert annotations[0]["iscrowd"] == 0
    assert type(annotations[0]["iscrowd"]) is int


def test_detections_to_coco_annotations_preserves_iscrowd_from_data_when_no_mask() -> (
    None
):
    """iscrowd stored in detections.data should be used when no mask is present."""
    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={"iscrowd": np.array([1], dtype=int), "area": np.array([1234.5])},
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["iscrowd"] == 1
    assert type(annotations[0]["iscrowd"]) is int
    assert annotations[0]["area"] == 1234.5


def test_detections_to_coco_annotations_iscrowd_is_int_when_mask_provided() -> None:
    """iscrowd should be stored as int (0 or 1), not as Python bool."""
    mask = np.zeros((1, 5, 5), dtype=bool)
    mask[0, 0:3, 0:3] = True  # simple single-component rectangle

    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["iscrowd"] == 0
    assert type(annotations[0]["iscrowd"]) is int


def test_detections_to_coco_annotations_data_area_overrides_bbox_with_mask() -> None:
    """data["area"] should override computed bbox area even when a mask is present."""
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 0:4, 0:4] = True  # 16-pixel polygon area

    detections = Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=mask,
        data={"area": np.array([999.0])},
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 999.0


def test_detections_to_coco_annotations_fallback_area_when_no_data() -> None:
    """When detections have no area in data, area should fall back to bbox area."""
    detections = Detections(
        xyxy=np.array([[10.0, 20.0, 110.0, 120.0]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert len(annotations) == 1
    assert annotations[0]["area"] == 100.0 * 100.0
    assert annotations[0]["iscrowd"] == 0


def test_load_coco_annotations_infers_masks_from_segmentation_field(
    tmp_path, coco_data_with_and_without_segmentation: dict[str, object]
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
    tmp_path, coco_data_with_unannotated_image: dict[str, object]
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
    )

    has_segmentation_path = str(images_directory / "has_segmentation.jpg")
    has_segmentation = annotations[has_segmentation_path]
    assert has_segmentation.mask is not None
    assert has_segmentation.mask.shape == (1, 5, 5)

    no_annotations_path = str(images_directory / "no_annotations.jpg")
    assert annotations[no_annotations_path] == Detections.empty()


@pytest.mark.parametrize(
    "file_name",
    [".", "", "subdir/.."],
)
def test_load_coco_annotations_rejects_file_name_resolving_to_images_directory(
    tmp_path,
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
    tmp_path,
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


def test_load_coco_annotations_rejects_absolute_file_name(tmp_path) -> None:
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
    tmp_path,
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


def test_load_coco_annotations_accepts_valid_nested_file_name(tmp_path) -> None:
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


def test_load_coco_annotations_force_masks_handles_missing_segmentation(
    tmp_path,
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
    assert image_annotations.mask is not None
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
        tmp_path,
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

    def test_multi_segment_masks_uneven_length_no_value_error(self, tmp_path) -> None:
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
        )

        annotation = dataset.annotations[str(images_directory / "image.jpg")]
        assert annotation.mask is not None
        assert annotation.mask.shape == (1, 5, 5)


# --- category_id 1-indexing (regression for #1181) ---


@pytest.mark.parametrize(
    ("classes", "expected_ids"),
    [
        ([], []),  # empty classes
        (["object"], [1]),  # single class starts at 1
        (["cat", "dog", "bird"], [1, 2, 3]),  # ids are sequential and 1-indexed
    ],
)
def test_classes_to_coco_categories_ids_start_at_one(
    classes: list[str], expected_ids: list[int]
) -> None:
    """COCO categories[].id must be 1-indexed (COCO spec / CVAT requirement)."""
    categories = classes_to_coco_categories(classes=classes)

    assert [category["id"] for category in categories] == expected_ids


def test_detections_to_coco_annotations_category_id_is_one_indexed() -> None:
    """Internal class_id k must serialize to COCO category_id k + 1."""
    detections = Detections(
        xyxy=np.array([[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 4, 4]], dtype=np.float32),
        class_id=np.array([0, 1, 2], dtype=int),
    )

    annotations, _ = detections_to_coco_annotations(
        detections=detections,
        image_id=1,
        annotation_id=1,
    )

    assert [annotation["category_id"] for annotation in annotations] == [1, 2, 3]


def test_coco_round_trip_preserves_class_ids_and_writes_one_indexed_categories(
    tmp_path,
) -> None:
    """as_coco -> from_coco is lossless for internal class_ids while the
    on-disk COCO category ids are 1-indexed (regression for #1181)."""
    classes = ["cat", "dog"]
    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    expected_class_ids = {}
    for index, class_id in enumerate([0, 1]):
        path = str(tmp_path / f"image_{index}.jpg")
        assert cv2.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint8))
        image_paths.append(path)
        detections = Detections(
            xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
            class_id=np.array([class_id], dtype=int),
        )
        annotations[path] = detections
        expected_class_ids[Path(path).name] = class_id
    dataset = DetectionDataset(
        classes=classes, images=image_paths, annotations=annotations
    )

    annotation_path = tmp_path / "annotations.json"
    dataset.as_coco(annotations_path=str(annotation_path))

    # On-disk COCO ids are 1-indexed.
    with open(annotation_path) as f:
        payload = json.load(f)
    assert sorted(category["id"] for category in payload["categories"]) == [1, 2]
    assert sorted(ann["category_id"] for ann in payload["annotations"]) == [1, 2]

    # Reading back preserves internal 0-indexed class_ids losslessly.
    loaded = DetectionDataset.from_coco(
        images_directory_path=str(tmp_path),
        annotations_path=str(annotation_path),
    )
    assert loaded.classes == classes
    for image_path, _, detections in loaded:
        name = Path(image_path).name
        assert detections.class_id is not None
        assert detections.class_id.tolist() == [expected_class_ids[name]]


# --- save_coco_annotations: cross-split id chaining (regression for #768) ---


def _tiny_detection_dataset(
    tmp_path, prefix: str, num_images: int, dets_per_image: int
) -> DetectionDataset:
    """Build a DetectionDataset of ``num_images`` 10x10 RGB images on disk,
    each holding ``dets_per_image`` 1x1 detections of class 0. Image content
    is irrelevant; only the per-image Detections drive the COCO write path."""
    classes = ["object"]
    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    for i in range(num_images):
        path = str(tmp_path / f"{prefix}_{i}.jpg")
        assert cv2.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint8))
        image_paths.append(path)
        xyxy = np.array(
            [[float(j), 0.0, float(j) + 1.0, 1.0] for j in range(dets_per_image)],
            dtype=float,
        ).reshape(-1, 4)
        annotations[path] = Detections(
            xyxy=xyxy,
            class_id=np.zeros(dets_per_image, dtype=int),
            confidence=np.ones(dets_per_image, dtype=float),
        )
    return DetectionDataset(
        classes=classes, images=image_paths, annotations=annotations
    )


def _read_ids(annotation_path) -> tuple[list[int], list[int]]:
    with open(annotation_path) as f:
        payload = json.load(f)
    image_ids = [img["id"] for img in payload["images"]]
    annotation_ids = [ann["id"] for ann in payload["annotations"]]
    return image_ids, annotation_ids


def test_save_coco_annotations_defaults_start_at_one(tmp_path):
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=3)
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset, annotation_path=str(annotation_path)
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == [1, 2]
    assert annotation_ids == [1, 2, 3, 4, 5, 6]
    # Returned ids are one greater than the highest written, ready to chain.
    assert next_image_id == 3
    assert next_annotation_id == 7


def test_save_coco_annotations_respects_starting_ids(tmp_path):
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=2)
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset,
        annotation_path=str(annotation_path),
        starting_image_id=100,
        starting_annotation_id=500,
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == [100, 101]
    assert annotation_ids == [500, 501, 502, 503]
    assert next_image_id == 102
    assert next_annotation_id == 504


def test_as_coco_chains_ids_across_splits_without_collision(tmp_path):
    """Regression for #768: exporting train/valid/test splits with the
    returned ids fed forward yields globally unique image and annotation ids."""
    train = _tiny_detection_dataset(tmp_path, "train", num_images=3, dets_per_image=2)
    valid = _tiny_detection_dataset(tmp_path, "valid", num_images=2, dets_per_image=4)
    test = _tiny_detection_dataset(tmp_path, "test", num_images=1, dets_per_image=5)

    train_path = tmp_path / "train.json"
    valid_path = tmp_path / "valid.json"
    test_path = tmp_path / "test.json"

    next_image_id, next_annotation_id = train.as_coco(annotations_path=str(train_path))
    next_image_id, next_annotation_id = valid.as_coco(
        annotations_path=str(valid_path),
        starting_image_id=next_image_id,
        starting_annotation_id=next_annotation_id,
    )
    test.as_coco(
        annotations_path=str(test_path),
        starting_image_id=next_image_id,
        starting_annotation_id=next_annotation_id,
    )

    all_image_ids: list[int] = []
    all_annotation_ids: list[int] = []
    for path in (train_path, valid_path, test_path):
        image_ids, annotation_ids = _read_ids(path)
        all_image_ids.extend(image_ids)
        all_annotation_ids.extend(annotation_ids)

    assert len(all_image_ids) == len(set(all_image_ids)), (
        "image ids collide across splits"
    )
    assert len(all_annotation_ids) == len(set(all_annotation_ids)), (
        "annotation ids collide across splits"
    )
    # Concrete chained values.
    assert all_image_ids == [1, 2, 3, 4, 5, 6]
    assert all_annotation_ids == list(range(1, 6 + 8 + 5 + 1))


def test_save_coco_annotations_empty_dataset_returns_starting_ids(tmp_path):
    """An empty dataset writes a valid (but empty) COCO file and returns
    the starting ids unchanged so chaining still composes around it."""
    dataset = DetectionDataset(classes=["object"], images=[], annotations={})
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset,
        annotation_path=str(annotation_path),
        starting_image_id=7,
        starting_annotation_id=42,
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == []
    assert annotation_ids == []
    assert next_image_id == 7
    assert next_annotation_id == 42


def test_as_coco_without_annotations_path_returns_starting_ids(tmp_path):
    """When only writing images, the starting ids round-trip unchanged so
    chaining still works in the images-only branch."""
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=1)
    next_image_id, next_annotation_id = dataset.as_coco(
        images_directory_path=str(tmp_path / "imgs"),
        starting_image_id=42,
        starting_annotation_id=99,
    )
    assert next_image_id == 42
    assert next_annotation_id == 99


def test_save_coco_annotations_annotation_image_id_references_correct_image(tmp_path):
    """Every annotation's image_id must reference an image id present in the
    same file, even when a non-default starting_image_id is used."""
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=3, dets_per_image=2)
    annotation_path = tmp_path / "annotations.json"

    save_coco_annotations(
        dataset=dataset,
        annotation_path=str(annotation_path),
        starting_image_id=100,
        starting_annotation_id=500,
    )

    with open(annotation_path) as f:
        coco = json.load(f)
    image_id_set = {img["id"] for img in coco["images"]}
    annotation_image_ids = {ann["image_id"] for ann in coco["annotations"]}
    assert annotation_image_ids <= image_id_set, (
        "annotation image_id values reference unknown image ids"
    )


def test_save_coco_annotations_zero_annotation_images(tmp_path):
    """Dataset with images but zero detections per image: image ids are
    assigned sequentially but annotation list stays empty."""
    dataset = _tiny_detection_dataset(tmp_path, "img", num_images=2, dets_per_image=0)
    annotation_path = tmp_path / "annotations.json"

    next_image_id, next_annotation_id = save_coco_annotations(
        dataset=dataset, annotation_path=str(annotation_path)
    )

    image_ids, annotation_ids = _read_ids(annotation_path)
    assert image_ids == [1, 2]
    assert annotation_ids == []
    assert next_image_id == 3
    assert next_annotation_id == 1


# --- Regression: legacy 0-indexed COCO files still load correctly (#1181) ---


def test_from_coco_loads_legacy_zero_indexed_category_ids(tmp_path) -> None:
    """COCO files with 0-indexed category ids (written by supervision <=0.28.x)
    must still load and produce correct internal 0-indexed class_ids."""
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


# --- save_coco_annotations ValueError guards ---


@pytest.mark.parametrize(
    ("starting_image_id", "starting_annotation_id"),
    [
        (0, 1),
        (1, 0),
        (0, 0),
    ],
)
def test_save_coco_annotations_rejects_zero_starting_ids(
    tmp_path, starting_image_id: int, starting_annotation_id: int
) -> None:
    """starting_image_id and starting_annotation_id below 1 must raise ValueError."""
    dataset = DetectionDataset(classes=["object"], images=[], annotations={})
    annotation_path = tmp_path / "annotations.json"

    with pytest.raises(ValueError, match="must be >= 1"):
        save_coco_annotations(
            dataset=dataset,
            annotation_path=str(annotation_path),
            starting_image_id=starting_image_id,
            starting_annotation_id=starting_annotation_id,
        )


# --- detections_to_coco_annotations: class_id=None guard ---


def test_detections_to_coco_annotations_raises_when_class_id_is_none() -> None:
    """Detections with no class_id must raise ValueError before +1 arithmetic."""
    detections = Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        class_id=None,
    )

    with pytest.raises(ValueError, match="class_id"):
        detections_to_coco_annotations(
            detections=detections,
            image_id=1,
            annotation_id=1,
        )


# --- Round-trip: multi-class-per-image case ---


def test_coco_round_trip_multi_class_single_image(tmp_path) -> None:
    """Single image with two detections of different classes round-trips losslessly."""
    img_path = str(tmp_path / "img.jpg")
    assert cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype=np.uint8))

    dataset = DetectionDataset(
        classes=["cat", "dog"],
        images=[img_path],
        annotations={
            img_path: Detections(
                xyxy=np.array([[0, 0, 5, 5], [1, 1, 4, 4]], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int),
            )
        },
    )

    annotation_path = tmp_path / "annotations.json"
    dataset.as_coco(annotations_path=str(annotation_path))

    with open(annotation_path) as f:
        payload = json.load(f)
    assert sorted(ann["category_id"] for ann in payload["annotations"]) == [1, 2]

    loaded = DetectionDataset.from_coco(
        images_directory_path=str(tmp_path),
        annotations_path=str(annotation_path),
    )
    dets = loaded.annotations[img_path]
    assert dets.class_id is not None
    assert sorted(dets.class_id.tolist()) == [0, 1]


# --- Regression: segmentation round-trip (#2285) ---


def _coco_annotation_with_segmentation(
    segmentation: list[list[int]],
    bbox: tuple[float, float, float, float] = (0, 0, 5, 5),
    area: float = 25,
) -> dict:
    return mock_coco_annotation(
        annotation_id=1,
        image_id=1,
        category_id=1,
        bbox=bbox,
        area=area,
        segmentation=segmentation,
    )


def _single_image_coco_data(annotation: dict) -> dict[str, object]:
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "cat", "supercategory": ""}],
        "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
        "annotations": [annotation],
    }


def test_detections_to_coco_annotations_exports_all_polygons() -> None:
    """All polygons from a multi-component mask must be exported, not just the first."""
    # Build a mask with two separate rectangles (disjoint components)
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:4, 1:4] = True  # top-left component
    mask[14:18, 14:18] = True  # bottom-right component

    detections = Detections(
        xyxy=np.array([[1, 1, 4, 4]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=np.array([mask]),
        data={"iscrowd": np.array([0], dtype=int)},
    )
    annotations, _ = detections_to_coco_annotations(
        detections=detections, image_id=1, annotation_id=1
    )
    assert len(annotations) == 1
    seg = annotations[0]["segmentation"]
    # Both components must appear as separate polygon entries (list of lists)
    assert isinstance(seg, list), "segmentation must be a list of polygons"
    assert len(seg) >= 2


@pytest.mark.parametrize(
    ("segmentation", "bbox", "area", "expected_min_polygon_count"),
    [
        pytest.param(
            [[0, 0, 4, 0, 4, 4, 0, 4]],
            (0, 0, 5, 5),
            25,
            1,
            id="single-polygon",
        ),
        pytest.param(
            [[0, 0, 4, 0, 4, 4, 0, 4], [6, 6, 9, 6, 9, 9, 6, 9]],
            (0, 0, 9, 9),
            32,
            2,
            id="multi-polygon",
        ),
    ],
)
def test_coco_polygon_segmentation_survives_roundtrip(
    tmp_path,
    segmentation: list[list[int]],
    bbox: tuple[float, float, float, float],
    area: float,
    expected_min_polygon_count: int,
) -> None:
    """COCO polygon segmentation survives the load/export sequence.

    1. Write source COCO JSON with polygon segmentation.
    2. Load it through DetectionDataset.from_coco().
    3. Export it back to COCO JSON with as_coco().
    4. Assert the exported segmentation keeps the expected polygon component count.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "img.jpg"
    assert cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    # 1. Write source COCO JSON with polygon segmentation.
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps(
            _single_image_coco_data(
                _coco_annotation_with_segmentation(
                    segmentation=segmentation, bbox=bbox, area=area
                )
            )
        ),
        encoding="utf-8",
    )

    # 2. Load it through the internal DetectionDataset representation.
    ds = DetectionDataset.from_coco(
        images_directory_path=str(images_dir),
        annotations_path=str(ann_path),
    )

    # 3. Export it back to COCO JSON.
    out_ann_path = tmp_path / "out_annotations.json"
    ds.as_coco(annotations_path=str(out_ann_path))

    with open(out_ann_path) as f:
        out = json.load(f)

    # 4. Assert polygon component count survives the load/export sequence.
    assert len(out["annotations"]) == 1
    seg = out["annotations"][0]["segmentation"]
    assert isinstance(seg, list)
    assert len(seg) >= expected_min_polygon_count


def test_coco_raw_segmentation_preserved_when_masks_not_decoded() -> None:
    """When masks are NOT decoded (with_masks=False), raw polygon data stored in
    data['segmentation'] is used as a lossless fallback so as_coco() still emits
    non-empty segmentation."""
    image_annotations = [
        _coco_annotation_with_segmentation(segmentation=[[0, 0, 4, 0, 4, 4, 0, 4]])
    ]

    # Load WITHOUT mask decoding — mask must be None
    detections = coco_annotations_to_detections(
        image_annotations=image_annotations,
        resolution_wh=(10, 10),
        with_masks=False,
    )
    assert detections.mask is None
    # Raw segmentation must be stored in data for fallback
    assert "coco_raw_segmentation" in detections.data

    # Export must still produce non-empty segmentation via fallback
    annotations, _ = detections_to_coco_annotations(
        detections=detections, image_id=1, annotation_id=1
    )
    assert len(annotations) == 1
    assert annotations[0]["segmentation"] != []


def test_coco_iscrowd_mask_exports_as_rle() -> None:
    """Multi-segment mask exports segmentation as RLE dict (iscrowd inferred as 1)."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True  # top-left component
    mask[7:9, 7:9] = True  # bottom-right component (two separate regions)

    detections = Detections(
        xyxy=np.array([[1, 1, 8, 8]], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        mask=np.array([mask]),
    )
    annotations, _ = detections_to_coco_annotations(
        detections=detections, image_id=1, annotation_id=1
    )
    assert len(annotations) == 1
    seg = annotations[0]["segmentation"]
    assert isinstance(seg, dict), "multi-segment mask must export as RLE dict, not list"
    assert "counts" in seg
    assert "size" in seg
