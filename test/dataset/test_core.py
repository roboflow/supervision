from contextlib import ExitStack as DoesNotRaise
from typing import List, Optional

import pytest

from supervision import DetectionDataset
from test.test_utils import mock_detections


@pytest.mark.parametrize(
    "dataset_list, expected_result, exception",
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
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
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
                    images=["image-1.png", "image-2.png"],
                    annotations={
                        "image-1.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                        "image-2.png": mock_detections(
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
                    images=["image-1.png", "image-2.png"],
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
                    images=["image-3.png"],
                    annotations={
                        "image-3.png": mock_detections(
                            xyxy=[[0, 0, 10, 10]], class_id=[0]
                        ),
                    },
                ),
            ],
            DetectionDataset(
                classes=["cat", "dog", "person"],
                images=["image-1.png", "image-2.png", "image-3.png"],
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
                    images=["image-1.png", "image-2.png"],
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
                    images=["image-2.png", "image-3.png"],
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


def test_as_coco_annotation_ids(tmp_path):
    """Test that as_coco generates unique annotation IDs across splits."""
    # Create mock images in a temporary directory
    import numpy as np
    from PIL import Image
    
    image1_path = tmp_path / "image1.jpg"
    image2_path = tmp_path / "image2.jpg"
    
    # Create simple black images
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(image1_path)
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(image2_path)
    
    # Create a mock dataset with 2 images and 2 annotations each
    dataset = DetectionDataset(
        classes=["class1", "class2"],
        images=[str(image1_path), str(image2_path)],
        annotations={
            str(image1_path): mock_detections(xyxy=[[0,0,10,10], [10,10,20,20]], class_id=[0,1]),
            str(image2_path): mock_detections(xyxy=[[0,0,10,10], [10,10,20,20]], class_id=[1,0]),
        }
    )
    
    # Split the dataset into two parts
    train, val = dataset.split(split_ratio=0.5, random_state=42)
    
    # Create output directories
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    
    # Export both splits to COCO format with proper ID tracking
    train_ann_path = train_dir / "annotations.json"
    val_ann_path = val_dir / "annotations.json"
    
    last_train_id = train.as_coco(
        images_directory_path=str(train_dir),
        annotations_path=str(train_ann_path),
        annotation_id_offset=0
    )
    val.as_coco(
        images_directory_path=str(val_dir),
        annotations_path=str(val_ann_path),
        annotation_id_offset=last_train_id + 1
    )
    
    # Load both annotation files and check IDs
    import json
    
    with open(train_ann_path) as f:
        train_anns = json.load(f)
    train_ids = [ann["id"] for ann in train_anns["annotations"]]
    
    with open(val_ann_path) as f:
        val_anns = json.load(f)
    val_ids = [ann["id"] for ann in val_anns["annotations"]]
    
    # Check all IDs are unique across splits
    all_ids = train_ids + val_ids
    assert len(all_ids) == len(set(all_ids)), "Duplicate annotation IDs found across splits"
