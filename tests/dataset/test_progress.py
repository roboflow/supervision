"""Tests for show_progress parameter on dataset load/save operations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from tqdm.auto import tqdm as _real_tqdm

from supervision import DetectionDataset


def _create_dummy_yolo_dataset(root: str, num_images: int = 3) -> tuple[str, str, str]:
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i in range(num_images):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(images_dir, f"img_{i}.jpg"), img)
        with open(os.path.join(labels_dir, f"img_{i}.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    data_yaml = os.path.join(root, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write("names:\n  - class_0\nnc: 1\n")

    return images_dir, labels_dir, data_yaml


def _create_dummy_coco_dataset(root: str, num_images: int = 3) -> tuple[str, str]:
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "class_0", "supercategory": "none"}],
    }

    for i in range(num_images):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        fname = f"img_{i}.jpg"
        cv2.imwrite(os.path.join(images_dir, fname), img)
        coco["images"].append(
            {
                "id": i,
                "file_name": fname,
                "width": 100,
                "height": 100,
            }
        )
        coco["annotations"].append(
            {
                "id": i,
                "image_id": i,
                "category_id": 0,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "segmentation": [],
                "iscrowd": 0,
            }
        )

    annotations_path = os.path.join(root, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(coco, f)

    return images_dir, annotations_path


def _create_dummy_pascal_voc_dataset(root: str, num_images: int = 3) -> tuple[str, str]:
    images_dir = os.path.join(root, "images")
    annotations_dir = os.path.join(root, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for i in range(num_images):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(images_dir, f"img_{i}.jpg"), img)
        xml_content = f"""<?xml version="1.0" ?>
<annotation>
    <folder>images</folder>
    <filename>img_{i}.jpg</filename>
    <size>
        <width>100</width>
        <height>100</height>
        <depth>3</depth>
    </size>
    <object>
        <name>class_0</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>10</ymin>
            <xmax>30</xmax>
            <ymax>30</ymax>
        </bndbox>
    </object>
</annotation>"""
        with open(os.path.join(annotations_dir, f"img_{i}.xml"), "w") as f:
            f.write(xml_content)

    return images_dir, annotations_dir


# ---------------------------------------------------------------------------
# Fixtures — raw file trees (used by from_* tests that call the loader under patch)
# ---------------------------------------------------------------------------


@pytest.fixture
def yolo_dir(tmp_path: Path) -> tuple[str, str, str]:
    """YOLO images, labels, and data.yaml on disk."""
    return _create_dummy_yolo_dataset(str(tmp_path))


@pytest.fixture
def coco_dir(tmp_path: Path) -> tuple[str, str]:
    """COCO images directory and annotations JSON on disk."""
    return _create_dummy_coco_dataset(str(tmp_path))


@pytest.fixture
def pascal_voc_dir(tmp_path: Path) -> tuple[str, str]:
    """Pascal VOC images and XML annotations on disk."""
    return _create_dummy_pascal_voc_dataset(str(tmp_path))


# ---------------------------------------------------------------------------
# Fixtures — pre-loaded DetectionDataset (used by as_* and backward-compat tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def yolo_dataset(yolo_dir: tuple[str, str, str]) -> DetectionDataset:
    """DetectionDataset loaded from a dummy YOLO dataset."""
    images_dir, labels_dir, data_yaml = yolo_dir
    return DetectionDataset.from_yolo(
        images_directory_path=images_dir,
        annotations_directory_path=labels_dir,
        data_yaml_path=data_yaml,
    )


@pytest.fixture
def coco_dataset(coco_dir: tuple[str, str]) -> DetectionDataset:
    """DetectionDataset loaded from a dummy COCO dataset."""
    images_dir, annotations_path = coco_dir
    return DetectionDataset.from_coco(
        images_directory_path=images_dir,
        annotations_path=annotations_path,
    )


@pytest.fixture
def pascal_voc_dataset(pascal_voc_dir: tuple[str, str]) -> DetectionDataset:
    """DetectionDataset loaded from a dummy Pascal VOC dataset."""
    images_dir, annotations_dir = pascal_voc_dir
    return DetectionDataset.from_pascal_voc(
        images_directory_path=images_dir,
        annotations_directory_path=annotations_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_YOLO_TQDM = "supervision.dataset.formats.yolo.tqdm"
_COCO_TQDM = "supervision.dataset.formats.coco.tqdm"
_PASCAL_TQDM = "supervision.dataset.formats.pascal_voc.tqdm"
_CORE_TQDM = "supervision.dataset.core.tqdm"
_UTILS_TQDM = "supervision.dataset.utils.tqdm"


class TestYoloProgress:
    @patch(_YOLO_TQDM, wraps=_real_tqdm)
    def test_from_yolo_no_progress_by_default(
        self, mock_tqdm: object, yolo_dir: tuple[str, str, str]
    ):
        """YOLO load does not show progress bar by default."""
        images_dir, labels_dir, data_yaml = yolo_dir
        ds = DetectionDataset.from_yolo(
            images_directory_path=images_dir,
            annotations_directory_path=labels_dir,
            data_yaml_path=data_yaml,
        )
        assert mock_tqdm.call_args[1]["disable"] is True
        assert len(ds) == 3

    @patch(_YOLO_TQDM, wraps=_real_tqdm)
    def test_from_yolo_with_progress(
        self, mock_tqdm: object, yolo_dir: tuple[str, str, str]
    ):
        """YOLO load shows progress bar when show_progress=True."""
        images_dir, labels_dir, data_yaml = yolo_dir
        ds = DetectionDataset.from_yolo(
            images_directory_path=images_dir,
            annotations_directory_path=labels_dir,
            data_yaml_path=data_yaml,
            show_progress=True,
        )
        assert mock_tqdm.call_args[1]["disable"] is False
        assert len(ds) == 3

    @patch(_YOLO_TQDM, wraps=_real_tqdm)
    def test_as_yolo_with_progress(
        self, mock_tqdm: object, yolo_dataset: DetectionDataset, tmp_path: Path
    ):
        """YOLO save shows progress bar when show_progress=True."""
        out = tmp_path / "output"
        yolo_dataset.as_yolo(
            images_directory_path=str(out / "images"),
            annotations_directory_path=str(out / "labels"),
            data_yaml_path=str(out / "data.yaml"),
            show_progress=True,
        )
        assert mock_tqdm.call_args[1]["disable"] is False

    @patch(_YOLO_TQDM, wraps=_real_tqdm)
    def test_as_yolo_no_progress_by_default(
        self, mock_tqdm: object, yolo_dataset: DetectionDataset, tmp_path: Path
    ):
        """Saving YOLO annotations does not show progress bar by default."""
        yolo_dataset.as_yolo(
            annotations_directory_path=str(tmp_path / "output" / "labels")
        )
        assert mock_tqdm.call_args[1]["disable"] is True


class TestCocoProgress:
    @patch(_COCO_TQDM, wraps=_real_tqdm)
    def test_from_coco_no_progress_by_default(
        self, mock_tqdm: object, coco_dir: tuple[str, str]
    ):
        """COCO load does not show progress bar by default."""
        images_dir, annotations_path = coco_dir
        ds = DetectionDataset.from_coco(
            images_directory_path=images_dir,
            annotations_path=annotations_path,
        )
        assert mock_tqdm.call_args[1]["disable"] is True
        assert len(ds) == 3

    @patch(_COCO_TQDM, wraps=_real_tqdm)
    def test_from_coco_with_progress(
        self, mock_tqdm: object, coco_dir: tuple[str, str]
    ):
        """COCO load shows progress bar when show_progress=True."""
        images_dir, annotations_path = coco_dir
        ds = DetectionDataset.from_coco(
            images_directory_path=images_dir,
            annotations_path=annotations_path,
            show_progress=True,
        )
        assert mock_tqdm.call_args[1]["disable"] is False
        assert len(ds) == 3

    @patch(_COCO_TQDM, wraps=_real_tqdm)
    def test_as_coco_with_progress(
        self, mock_tqdm: object, coco_dataset: DetectionDataset, tmp_path: Path
    ):
        """COCO save shows progress bar when show_progress=True."""
        out = tmp_path / "output"
        coco_dataset.as_coco(
            images_directory_path=str(out / "images"),
            annotations_path=str(out / "annotations.json"),
            show_progress=True,
        )
        assert mock_tqdm.call_args[1]["disable"] is False

    @patch(_COCO_TQDM, wraps=_real_tqdm)
    def test_as_coco_no_progress_by_default(
        self, mock_tqdm: object, coco_dataset: DetectionDataset, tmp_path: Path
    ):
        """Saving COCO annotations does not show progress bar by default."""
        coco_dataset.as_coco(
            annotations_path=str(tmp_path / "output" / "annotations.json")
        )
        assert mock_tqdm.call_args[1]["disable"] is True


class TestPascalVocProgress:
    @patch(_PASCAL_TQDM, wraps=_real_tqdm)
    def test_from_pascal_voc_no_progress_by_default(
        self, mock_tqdm: object, pascal_voc_dir: tuple[str, str]
    ):
        """Pascal VOC load does not show progress bar by default."""
        images_dir, annotations_dir = pascal_voc_dir
        ds = DetectionDataset.from_pascal_voc(
            images_directory_path=images_dir,
            annotations_directory_path=annotations_dir,
        )
        assert mock_tqdm.call_args[1]["disable"] is True
        assert len(ds) == 3

    @patch(_PASCAL_TQDM, wraps=_real_tqdm)
    def test_from_pascal_voc_with_progress(
        self, mock_tqdm: object, pascal_voc_dir: tuple[str, str]
    ):
        """Pascal VOC load shows progress bar when show_progress=True."""
        images_dir, annotations_dir = pascal_voc_dir
        ds = DetectionDataset.from_pascal_voc(
            images_directory_path=images_dir,
            annotations_directory_path=annotations_dir,
            show_progress=True,
        )
        assert mock_tqdm.call_args[1]["disable"] is False
        assert len(ds) == 3

    def test_as_pascal_voc_with_progress(
        self, pascal_voc_dataset: DetectionDataset, tmp_path: Path
    ):
        """Pascal VOC save shows progress bar when show_progress=True."""
        out = tmp_path / "output"
        with (
            patch(_CORE_TQDM, wraps=_real_tqdm) as mock_tqdm,
            patch(_UTILS_TQDM, wraps=_real_tqdm),
        ):
            pascal_voc_dataset.as_pascal_voc(
                images_directory_path=str(out / "images"),
                annotations_directory_path=str(out / "annotations"),
                show_progress=True,
            )
            assert mock_tqdm.call_args[1]["disable"] is False

    @patch(_CORE_TQDM, wraps=_real_tqdm)
    def test_as_pascal_voc_no_progress_by_default(
        self, mock_tqdm: object, pascal_voc_dataset: DetectionDataset, tmp_path: Path
    ):
        """Saving Pascal VOC annotations does not show progress bar by default."""
        pascal_voc_dataset.as_pascal_voc(
            annotations_directory_path=str(tmp_path / "output" / "annotations")
        )
        assert mock_tqdm.call_args[1]["disable"] is True


class TestSaveImagesProgress:
    @patch(_UTILS_TQDM, wraps=_real_tqdm)
    def test_save_images_with_progress(
        self, mock_tqdm: object, yolo_dataset: DetectionDataset, tmp_path: Path
    ):
        """save_dataset_images shows progress bar when show_progress=True."""
        from supervision.dataset.utils import save_dataset_images

        out_images = str(tmp_path / "output_images")
        save_dataset_images(
            dataset=yolo_dataset,
            images_directory_path=out_images,
            show_progress=True,
        )
        assert mock_tqdm.call_args[1]["disable"] is False
        assert len(os.listdir(out_images)) == 3

    @patch(_UTILS_TQDM, wraps=_real_tqdm)
    def test_save_dataset_images_no_progress_by_default(
        self, mock_tqdm: object, yolo_dataset: DetectionDataset, tmp_path: Path
    ):
        """save_dataset_images does not show progress bar by default."""
        from supervision.dataset.utils import save_dataset_images

        save_dataset_images(
            dataset=yolo_dataset,
            images_directory_path=str(tmp_path / "output_images_default"),
        )
        assert mock_tqdm.call_args[1]["disable"] is True


class TestBackwardCompatibility:
    """Ensure show_progress=False (default) doesn't change behavior."""

    def test_from_yolo_default_works(self, yolo_dataset: DetectionDataset):
        """YOLO load with default args returns correct dataset size."""
        assert len(yolo_dataset) == 3

    def test_from_coco_default_works(self, coco_dataset: DetectionDataset):
        """COCO load with default args returns correct dataset size."""
        assert len(coco_dataset) == 3

    def test_from_pascal_voc_default_works(self, pascal_voc_dataset: DetectionDataset):
        """Pascal VOC load with default args returns correct dataset size."""
        assert len(pascal_voc_dataset) == 3
