"""Shared builders for dataset format tests."""

from typing import Any


def mock_coco_annotation(
    annotation_id: int = 0,
    image_id: int = 0,
    category_id: int = 0,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    area: float = 0.0,
    segmentation: list[list[float]] | dict[str, Any] | None = None,
    iscrowd: bool = False,
) -> dict[str, Any]:
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
