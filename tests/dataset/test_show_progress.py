"""
Tests that show_progress parameter works correctly for all dataset
loaders and savers, and that output is identical regardless of its value.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from supervision import DetectionDataset
from supervision.dataset.formats.coco import (
    load_coco_annotations,
    save_coco_annotations,
)
from supervision.dataset.formats.pascal_voc import load_pascal_voc_annotations
from supervision.dataset.formats.yolo import (
    load_yolo_annotations,
    save_yolo_annotations,
)
from supervision.dataset.utils import save_dataset_images
from tests.helpers import create_yolo_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> dict:
    return create_yolo_dataset(str(tmp_path / "yolo"), num_images=3)


@pytest.fixture
def coco_dataset(tmp_path: Path) -> dict:
    """Create a minimal COCO dataset on disk (JSON only; no real images needed)."""
    images_dir = tmp_path / "coco" / "images"
    images_dir.mkdir(parents=True)
    annotations_path = tmp_path / "coco" / "annotations.json"

    coco_data = {
        "categories": [
            {"id": 1, "name": "dog", "supercategory": "animal"},
            {"id": 2, "name": "cat", "supercategory": "animal"},
        ],
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 30],
                "area": 900,
                "segmentation": [],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [5, 5, 20, 20],
                "area": 400,
                "segmentation": [],
                "iscrowd": 0,
            },
        ],
    }
    annotations_path.write_text(json.dumps(coco_data))

    return {
        "images_dir": str(images_dir),
        "annotations_path": str(annotations_path),
    }


@pytest.fixture
def pascal_voc_dataset(tmp_path: Path) -> dict:
    """Create a minimal Pascal VOC dataset on disk with real images and XML files."""
    images_dir = tmp_path / "voc" / "images"
    annotations_dir = tmp_path / "voc" / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    for i in range(1, 3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img{i}.jpg"), img)

        xml = f"""<?xml version="1.0" ?>
<annotation>
  <folder>VOC</folder>
  <filename>img{i}.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object>
    <name>dog</name>
    <bndbox>
      <xmin>{10 + i}</xmin><ymin>{10 + i}</ymin>
      <xmax>{40 + i}</xmax><ymax>{40 + i}</ymax>
    </bndbox>
  </object>
</annotation>"""
        (annotations_dir / f"img{i}.xml").write_text(xml)

    return {
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yolo(info: dict, show_progress: bool) -> tuple:
    return load_yolo_annotations(
        images_directory_path=info["images_dir"],
        annotations_directory_path=info["labels_dir"],
        data_yaml_path=info["data_yaml_path"],
        show_progress=show_progress,
    )


# ---------------------------------------------------------------------------
# Load: YOLO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_load_yolo_show_progress(yolo_dataset: dict, show_progress: bool) -> None:
    classes, image_paths, annotations = _load_yolo(yolo_dataset, show_progress)
    assert len(image_paths) == yolo_dataset["num_images"]
    assert len(annotations) == yolo_dataset["num_images"]
    assert isinstance(classes, list)


def test_load_yolo_show_progress_consistent(yolo_dataset: dict) -> None:
    classes_off, paths_off, ann_off = _load_yolo(yolo_dataset, show_progress=False)
    classes_on, paths_on, ann_on = _load_yolo(yolo_dataset, show_progress=True)
    assert classes_off == classes_on
    assert paths_off == paths_on
    assert set(ann_off.keys()) == set(ann_on.keys())


# ---------------------------------------------------------------------------
# Load: COCO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_load_coco_show_progress(coco_dataset: dict, show_progress: bool) -> None:
    classes, image_paths, annotations = load_coco_annotations(
        images_directory_path=coco_dataset["images_dir"],
        annotations_path=coco_dataset["annotations_path"],
        show_progress=show_progress,
    )
    assert classes == ["dog", "cat"]
    assert len(image_paths) == 2
    assert len(annotations) == 2


def test_load_coco_show_progress_consistent(coco_dataset: dict) -> None:
    classes_off, paths_off, _ann_off = load_coco_annotations(
        images_directory_path=coco_dataset["images_dir"],
        annotations_path=coco_dataset["annotations_path"],
        show_progress=False,
    )
    classes_on, paths_on, _ann_on = load_coco_annotations(
        images_directory_path=coco_dataset["images_dir"],
        annotations_path=coco_dataset["annotations_path"],
        show_progress=True,
    )
    assert classes_off == classes_on
    assert paths_off == paths_on


# ---------------------------------------------------------------------------
# Load: Pascal VOC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_load_pascal_voc_show_progress(
    pascal_voc_dataset: dict, show_progress: bool
) -> None:
    classes, image_paths, annotations = load_pascal_voc_annotations(
        images_directory_path=pascal_voc_dataset["images_dir"],
        annotations_directory_path=pascal_voc_dataset["annotations_dir"],
        show_progress=show_progress,
    )
    assert "dog" in classes
    assert len(image_paths) == 2
    assert len(annotations) == 2


def test_load_pascal_voc_show_progress_consistent(
    pascal_voc_dataset: dict,
) -> None:
    classes_off, paths_off, _ = load_pascal_voc_annotations(
        images_directory_path=pascal_voc_dataset["images_dir"],
        annotations_directory_path=pascal_voc_dataset["annotations_dir"],
        show_progress=False,
    )
    classes_on, paths_on, _ = load_pascal_voc_annotations(
        images_directory_path=pascal_voc_dataset["images_dir"],
        annotations_directory_path=pascal_voc_dataset["annotations_dir"],
        show_progress=True,
    )
    assert classes_off == classes_on
    assert set(paths_off) == set(paths_on)


# ---------------------------------------------------------------------------
# Save: YOLO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_save_yolo_annotations_show_progress(
    yolo_dataset: dict, tmp_path: Path, show_progress: bool
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_dir = tmp_path / f"yolo_out_{show_progress}"
    save_yolo_annotations(
        dataset=ds,
        annotations_directory_path=str(out_dir),
        show_progress=show_progress,
    )
    written = list(out_dir.glob("*.txt"))
    assert len(written) == yolo_dataset["num_images"]


def test_save_yolo_annotations_show_progress_consistent(
    yolo_dataset: dict, tmp_path: Path
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_off = tmp_path / "yolo_off"
    out_on = tmp_path / "yolo_on"
    save_yolo_annotations(dataset=ds, annotations_directory_path=str(out_off))
    save_yolo_annotations(
        dataset=ds, annotations_directory_path=str(out_on), show_progress=True
    )
    files_off = sorted(f.name for f in out_off.glob("*.txt"))
    files_on = sorted(f.name for f in out_on.glob("*.txt"))
    assert files_off == files_on


# ---------------------------------------------------------------------------
# Save: COCO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_save_coco_annotations_show_progress(
    yolo_dataset: dict, tmp_path: Path, show_progress: bool
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_path = tmp_path / f"coco_{show_progress}" / "annotations.json"
    save_coco_annotations(
        dataset=ds,
        annotation_path=str(out_path),
        show_progress=show_progress,
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert len(data["images"]) == yolo_dataset["num_images"]


def test_save_coco_annotations_show_progress_consistent(
    yolo_dataset: dict, tmp_path: Path
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_off = tmp_path / "coco_off" / "annotations.json"
    out_on = tmp_path / "coco_on" / "annotations.json"
    save_coco_annotations(dataset=ds, annotation_path=str(out_off))
    save_coco_annotations(dataset=ds, annotation_path=str(out_on), show_progress=True)
    data_off = json.loads(out_off.read_text())
    data_on = json.loads(out_on.read_text())
    assert len(data_off["images"]) == len(data_on["images"])
    assert len(data_off["annotations"]) == len(data_on["annotations"])
    assert data_off["categories"] == data_on["categories"]


# ---------------------------------------------------------------------------
# Save: Pascal VOC (via DetectionDataset.as_pascal_voc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_as_pascal_voc_show_progress(
    yolo_dataset: dict, tmp_path: Path, show_progress: bool
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_dir = tmp_path / f"voc_{show_progress}"
    ds.as_pascal_voc(
        annotations_directory_path=str(out_dir),
        show_progress=show_progress,
    )
    written = list(out_dir.glob("*.xml"))
    assert len(written) == yolo_dataset["num_images"]


def test_as_pascal_voc_show_progress_consistent(
    yolo_dataset: dict, tmp_path: Path
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_off = tmp_path / "voc_off"
    out_on = tmp_path / "voc_on"
    ds.as_pascal_voc(annotations_directory_path=str(out_off))
    ds.as_pascal_voc(annotations_directory_path=str(out_on), show_progress=True)
    files_off = sorted(f.name for f in out_off.glob("*.xml"))
    files_on = sorted(f.name for f in out_on.glob("*.xml"))
    assert files_off == files_on


# ---------------------------------------------------------------------------
# Save: images (save_dataset_images)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("show_progress", [False, True])
def test_save_dataset_images_show_progress(
    yolo_dataset: dict, tmp_path: Path, show_progress: bool
) -> None:
    ds = DetectionDataset.from_yolo(
        images_directory_path=yolo_dataset["images_dir"],
        annotations_directory_path=yolo_dataset["labels_dir"],
        data_yaml_path=yolo_dataset["data_yaml_path"],
    )
    out_dir = tmp_path / f"images_{show_progress}"
    save_dataset_images(
        dataset=ds,
        images_directory_path=str(out_dir),
        show_progress=show_progress,
    )
    written = list(out_dir.glob("*.jpg"))
    assert len(written) == yolo_dataset["num_images"]
