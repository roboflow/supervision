from contextlib import ExitStack as DoesNotRaise
from test.test_utils import mock_detections
from typing import List, Optional

import numpy as np
import pytest

from supervision import DetectionDataset


@pytest.mark.parametrize(
    "dataset_list, expected_result, exception",
    [
        (
            [],
            DetectionDataset(classes={}, images={}, annotations={}),
            DoesNotRaise(),
        ),  # empty dataset list
        (
            [DetectionDataset(classes={}, images={}, annotations={})],
            DetectionDataset(classes={}, images={}, annotations={}),
            DoesNotRaise(),
        ),  # single empty dataset
        (
            [
                DetectionDataset(
                    classes={0: "dog", 1: "person"}, images={}, annotations={}
                ),
                DetectionDataset(
                    classes={0: "dog", 1: "person"}, images={}, annotations={}
                ),
            ],
            DetectionDataset(
                classes={0: "dog", 1: "person"}, images={}, annotations={}
            ),
            DoesNotRaise(),
        ),  # two datasets; no images and annotations, the same classes
        (
            [
                DetectionDataset(
                    classes={0: "dog", 1: "person"}, images={}, annotations={}
                ),
                DetectionDataset(classes={0: "cat"}, images={}, annotations={}),
            ],
            DetectionDataset(
                classes={0: "cat", 1: "dog", 2: "person"}, images={}, annotations={}
            ),
            DoesNotRaise(),
        ),  # two datasets; no images and annotations, different classes
        (
            [
                DetectionDataset(
                    classes={0: "dog", 1: "person"},
                    images={
                        "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                        "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    },
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(classes={}, images={}, annotations={}),
            ],
            DetectionDataset(
                classes={0: "dog", 1: "person"},
                images={
                    "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                },
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
                    classes={0: "dog", 1: "person"},
                    images={
                        "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                        "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    },
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[1]
                        ),
                    },
                ),
                DetectionDataset(classes={0: "cat"}, images={}, annotations={}),
            ],
            DetectionDataset(
                classes={0: "cat", 1: "dog", 2: "person"},
                images={
                    "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                },
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
                    classes={0: "dog", 1: "person"},
                    images={
                        "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                        "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    },
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
                    classes={0: "cat"},
                    images={
                        "image-3.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    },
                    annotations={
                        "image-3.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                    },
                ),
            ],
            DetectionDataset(
                classes={0: "cat", 1: "dog", 2: "person"},
                images={
                    "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    "image-3.png": np.zeros((100, 100, 3), dtype=np.uint8),
                },
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
                    classes={0: "dog", 1: "person"},
                    images={
                        "image-1.png": np.zeros((100, 100, 3), dtype=np.uint8),
                        "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    },
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
                    classes={0: "dog", 1: "person"},
                    images={
                        "image-2.png": np.zeros((100, 100, 3), dtype=np.uint8),
                        "image-3.png": np.zeros((100, 100, 3), dtype=np.uint8),
                    },
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
