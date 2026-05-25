"""Tests for show_progress parameter on dataset load/save operations."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

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


class TestYoloProgress:
    def test_from_yolo_no_progress_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, labels_dir, data_yaml = _create_dummy_yolo_dataset(tmpdir)
            with patch(
                "supervision.dataset.formats.yolo.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds = DetectionDataset.from_yolo(
                    images_directory_path=images_dir,
                    annotations_directory_path=labels_dir,
                    data_yaml_path=data_yaml,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is True
            assert len(ds) == 3

    def test_from_yolo_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, labels_dir, data_yaml = _create_dummy_yolo_dataset(tmpdir)
            with patch(
                "supervision.dataset.formats.yolo.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds = DetectionDataset.from_yolo(
                    images_directory_path=images_dir,
                    annotations_directory_path=labels_dir,
                    data_yaml_path=data_yaml,
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False
            assert len(ds) == 3

    def test_as_yolo_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, labels_dir, data_yaml = _create_dummy_yolo_dataset(tmpdir)
            ds = DetectionDataset.from_yolo(
                images_directory_path=images_dir,
                annotations_directory_path=labels_dir,
                data_yaml_path=data_yaml,
            )

            out_dir = os.path.join(tmpdir, "output")
            out_images = os.path.join(out_dir, "images")
            out_labels = os.path.join(out_dir, "labels")

            with patch(
                "supervision.dataset.formats.yolo.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds.as_yolo(
                    images_directory_path=out_images,
                    annotations_directory_path=out_labels,
                    data_yaml_path=os.path.join(out_dir, "data.yaml"),
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False


class TestCocoProgress:
    def test_from_coco_no_progress_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_path = _create_dummy_coco_dataset(tmpdir)
            with patch(
                "supervision.dataset.formats.coco.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds = DetectionDataset.from_coco(
                    images_directory_path=images_dir,
                    annotations_path=annotations_path,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is True
            assert len(ds) == 3

    def test_from_coco_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_path = _create_dummy_coco_dataset(tmpdir)
            with patch(
                "supervision.dataset.formats.coco.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds = DetectionDataset.from_coco(
                    images_directory_path=images_dir,
                    annotations_path=annotations_path,
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False
            assert len(ds) == 3

    def test_as_coco_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_path = _create_dummy_coco_dataset(tmpdir)
            ds = DetectionDataset.from_coco(
                images_directory_path=images_dir,
                annotations_path=annotations_path,
            )

            out_dir = os.path.join(tmpdir, "output")
            with patch(
                "supervision.dataset.formats.coco.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds.as_coco(
                    images_directory_path=os.path.join(out_dir, "images"),
                    annotations_path=os.path.join(out_dir, "annotations.json"),
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False


class TestPascalVocProgress:
    def test_from_pascal_voc_no_progress_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_dir = _create_dummy_pascal_voc_dataset(tmpdir)
            with patch(
                "supervision.dataset.formats.pascal_voc.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds = DetectionDataset.from_pascal_voc(
                    images_directory_path=images_dir,
                    annotations_directory_path=annotations_dir,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is True
            assert len(ds) == 3

    def test_from_pascal_voc_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_dir = _create_dummy_pascal_voc_dataset(tmpdir)
            with patch(
                "supervision.dataset.formats.pascal_voc.tqdm",
                wraps=__import__("tqdm").auto.tqdm,
            ) as mock_tqdm:
                ds = DetectionDataset.from_pascal_voc(
                    images_directory_path=images_dir,
                    annotations_directory_path=annotations_dir,
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False
            assert len(ds) == 3

    def test_as_pascal_voc_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_dir = _create_dummy_pascal_voc_dataset(tmpdir)
            ds = DetectionDataset.from_pascal_voc(
                images_directory_path=images_dir,
                annotations_directory_path=annotations_dir,
            )

            out_dir = os.path.join(tmpdir, "output")
            with patch(
                "supervision.dataset.core.tqdm", wraps=__import__("tqdm").auto.tqdm
            ) as mock_tqdm:
                ds.as_pascal_voc(
                    images_directory_path=os.path.join(out_dir, "images"),
                    annotations_directory_path=os.path.join(out_dir, "annotations"),
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False


class TestSaveImagesProgress:
    def test_save_images_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, labels_dir, data_yaml = _create_dummy_yolo_dataset(tmpdir)
            ds = DetectionDataset.from_yolo(
                images_directory_path=images_dir,
                annotations_directory_path=labels_dir,
                data_yaml_path=data_yaml,
            )

            out_images = os.path.join(tmpdir, "output_images")
            with patch(
                "supervision.dataset.utils.tqdm", wraps=__import__("tqdm").auto.tqdm
            ) as mock_tqdm:
                from supervision.dataset.utils import save_dataset_images

                save_dataset_images(
                    dataset=ds,
                    images_directory_path=out_images,
                    show_progress=True,
                )
                call_kwargs = mock_tqdm.call_args
                assert call_kwargs[1]["disable"] is False

            saved_files = os.listdir(out_images)
            assert len(saved_files) == 3


class TestBackwardCompatibility:
    """Ensure show_progress=False (default) doesn't change behavior."""

    def test_from_yolo_default_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, labels_dir, data_yaml = _create_dummy_yolo_dataset(tmpdir)
            ds = DetectionDataset.from_yolo(
                images_directory_path=images_dir,
                annotations_directory_path=labels_dir,
                data_yaml_path=data_yaml,
            )
            assert len(ds) == 3

    def test_from_coco_default_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_path = _create_dummy_coco_dataset(tmpdir)
            ds = DetectionDataset.from_coco(
                images_directory_path=images_dir,
                annotations_path=annotations_path,
            )
            assert len(ds) == 3

    def test_from_pascal_voc_default_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir, annotations_dir = _create_dummy_pascal_voc_dataset(tmpdir)
            ds = DetectionDataset.from_pascal_voc(
                images_directory_path=images_dir,
                annotations_directory_path=annotations_dir,
            )
            assert len(ds) == 3
