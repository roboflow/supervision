"""Shared builders for dataset format tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from supervision.utils.file import list_files_with_extensions


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


def list_files_in_order(
    monkeypatch: pytest.MonkeyPatch,
    reorder: Callable[[list[Path]], list[Path]],
    module: str,
) -> None:
    """Make a loader's directory listing return files in a filesystem-like order.

    The real listing is sorted first, so `reorder` sees the same input on every
    filesystem and the test does not depend on the machine running it.
    """

    def listing(**kwargs: Any) -> list[Path]:
        """Return the real listing rearranged by `reorder`."""
        return reorder(sorted(list_files_with_extensions(**kwargs)))

    monkeypatch.setattr(f"{module}.list_files_with_extensions", listing)


def reversed_paths(paths: list[Path]) -> list[Path]:
    """Return the paths in descending name order."""
    return paths[::-1]


def interleaved_paths(paths: list[Path]) -> list[Path]:
    """Return the paths with every other one moved to the end."""
    return paths[1::2] + paths[::2]
