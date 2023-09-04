from contextlib import ExitStack as DoesNotRaise
from test.utils import mock_detections
from typing import List, Optional

import numpy as np
import pytest

from supervision import DetectionDataset
from supervision.dataset.utils import LazyLoadDict

from pathlib import Path
TEST_IMG_PATH = str(Path(__file__).parent.parent / "empty_image.png")



@pytest.mark.parametrize(
    "dataset_list, expected_result, exception",
    [
        (
            [],
            DetectionDataset(classes=[], images={}, annotations={}),
            DoesNotRaise(),
        ),  # empty dataset list
        (
            [DetectionDataset(classes=[], images={}, annotations={})],
            DetectionDataset(classes=[], images={}, annotations={}),
            DoesNotRaise(),
        ),  # single empty dataset
        (
            [
                DetectionDataset(classes=["dog", "person"], images={}, annotations={}),
                DetectionDataset(classes=["dog", "person"], images={}, annotations={}),
            ],
            DetectionDataset(classes=["dog", "person"], images={}, annotations={}),
            DoesNotRaise(),
        ),  # two datasets; no images and annotations, the same classes
        (
            [
                DetectionDataset(classes=["dog", "person"], images={}, annotations={}),
                DetectionDataset(classes=["cat"], images={}, annotations={}),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"], images={}, annotations={}
            ),
            DoesNotRaise(),
        ),  # two datasets; no images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=LazyLoadDict({
                        "image-1.png": TEST_IMG_PATH,
                        "image-2.png": TEST_IMG_PATH,
                    }),
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(classes=[], images={}, annotations={}),
            ],
            DetectionDataset(
                classes=["dog", "person"],
                images=LazyLoadDict({
                    "image-1.png": TEST_IMG_PATH,
                    "image-2.png": TEST_IMG_PATH,
                }),
                annotations={
                    "image-1.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
                    "image-2.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
                },
            ),
            DoesNotRaise(),
        ),  # two datasets; images and annotations, the same classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=LazyLoadDict({
                        "image-1.png": TEST_IMG_PATH,
                        "image-2.png": TEST_IMG_PATH,
                    }),
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(classes=["cat"], images={}, annotations={}),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"],
                images=LazyLoadDict({
                    "image-1.png": TEST_IMG_PATH,
                    "image-2.png": TEST_IMG_PATH,
                }),
                annotations={
                    "image-1.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
                    "image-2.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[2]),
                },
            ),
            DoesNotRaise(),
        ),  # two datasets; images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=LazyLoadDict({
                        "image-1.png": TEST_IMG_PATH,
                        "image-2.png": TEST_IMG_PATH,
                    }),
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(
                    classes=["cat"],
                    images=LazyLoadDict({
                        "image-3.png": TEST_IMG_PATH,                        
                    }),
                    annotations={
                        "image-3.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                    },
                ),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"],
                images=LazyLoadDict({
                    "image-1.png": TEST_IMG_PATH,
                    "image-2.png": TEST_IMG_PATH,
                    "image-3.png": TEST_IMG_PATH,
                }),
                annotations={
                    "image-1.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
                    "image-2.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[2]),
                    "image-3.png": mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
                },
            ),
            DoesNotRaise(),
        ),  # two datasets; images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes=["dog", "person"],
                    images=LazyLoadDict({
                        "image-1.png": TEST_IMG_PATH,
                        "image-2.png": TEST_IMG_PATH,
                    }),
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(
                    classes=["dog", "person"],
                    images=LazyLoadDict({
                        "image-2.png": TEST_IMG_PATH,
                        "image-3.png": TEST_IMG_PATH,
                    }),
                    annotations={
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-3.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
            ],
            None,
            pytest.raises(ValueError),
        ),
    ],
)
def test_dataset_merge(
    dataset_list: List[DetectionDataset],
    expected_result: Optional[DetectionDataset],
    exception: Exception,
) -> None:
    with exception:
        result = DetectionDataset.merge(dataset_list=dataset_list)
        assert result == expected_result
