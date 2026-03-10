from __future__ import annotations

import json
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision import Detections
from supervision.dataset.formats.coco import (
    build_coco_class_index_mapping,
    classes_to_coco_categories,
    coco_annotations_to_detections,
    coco_categories_to_classes,
    detections_to_coco_annotations,
    group_coco_annotations_by_image_id,
    load_coco_annotations,
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
                    category_id=0, bbox=(0, 0, 100, 100), area=100 * 100
                )
            ],
            DoesNotRaise(),
        ),  # no segmentation mask
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
                    category_id=0,
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
                    category_id=0,
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
                    category_id=0,
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
