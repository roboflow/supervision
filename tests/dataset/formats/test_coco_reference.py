"""COCO reference rasterization must preserve continuous polygon geometry."""

import builtins
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from supervision import DetectionDataset, polygon_to_mask
from supervision.dataset.formats.coco import coco_annotations_to_masks


def _write_dataset(tmp_path: Path, segmentation: Any) -> Path:
    """Write a synthetic image and one-object COCO dataset."""
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "image.png", "width": 8, "height": 8}
                ],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 1, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn.t accept a copy keyword:DeprecationWarning"
)
class TestCocoReferenceRasterization:
    """Opt-in masks agree with the COCO reference without changing defaults."""

    @pytest.mark.parametrize(
        "polygons",
        [
            pytest.param([[1, 1, 5, 1, 5, 5, 1, 5]], id="integer-square"),
            pytest.param([[1.6, 1.6, 5.6, 1.6, 5.6, 5.6, 1.6, 5.6]], id="subpixel"),
            pytest.param([[2.5, 2.5, 6.5, 2.5, 6.5, 6.5, 2.5, 6.5]], id="half-ties"),
            pytest.param([[1, 1, 2, 1, 2, 6, 1, 6]], id="thin-rectangle"),
            pytest.param([[1.2, 1, 1.4, 1, 1.4, 6, 1.2, 6]], id="subpixel-sliver"),
            pytest.param([[1, 1, 6, 1, 1, 6]], id="diagonal"),
            pytest.param([[1, 1, 6, 1, 6, 3, 3, 3, 3, 6, 1, 6]], id="concave"),
            pytest.param([[-1, -1, 9, -1, 9, 9, -1, 9]], id="crosses-canvas"),
            pytest.param(
                [[1, 1, 3, 1, 3, 3, 1, 3], [4, 4, 6, 4, 6, 6, 4, 6]],
                id="disjoint-parts",
            ),
            pytest.param(
                [[1, 1, 4, 1, 4, 4, 1, 4], [2, 2, 5, 2, 5, 5, 2, 5]],
                id="overlapping-parts-union",
            ),
        ],
    )
    def test_loader_matches_reference(
        self, tmp_path: Path, polygons: list[list[float]]
    ) -> None:
        """The public loader preserves fractional vertices and unions all parts."""
        coco_mask = pytest.importorskip("pycocotools.mask")
        path = _write_dataset(tmp_path, polygons)
        reference = coco_mask.decode(
            coco_mask.merge(coco_mask.frPyObjects(polygons, 8, 8))
        )

        dataset = DetectionDataset.from_coco(
            str(tmp_path), str(path), mask_rasterizer="pycocotools"
        )

        detections = next(iter(dataset.annotations.values()))
        np.testing.assert_array_equal(detections.mask[0], reference)
        assert detections.mask.dtype == bool
        np.testing.assert_array_equal(detections.class_id, [0])
        np.testing.assert_array_equal(detections.xyxy, [[1, 1, 5, 5]])

    def test_preserves_default_inclusive_mask(self, tmp_path: Path) -> None:
        """Existing callers still receive the documented inclusive square."""
        path = _write_dataset(tmp_path, [[1, 1, 5, 1, 5, 5, 1, 5]])
        expected = np.zeros((8, 8), dtype=bool)
        expected[1:6, 1:6] = True

        dataset = DetectionDataset.from_coco(str(tmp_path), str(path))

        np.testing.assert_array_equal(
            next(iter(dataset.annotations.values())).mask[0], expected
        )

    @pytest.mark.parametrize("rasterizer", ["supervision", "pycocotools"])
    def test_preserves_rle(self, tmp_path: Path, rasterizer: str) -> None:
        """The selector affects polygon conversion only, preserving RLE pixels."""
        coco_mask = pytest.importorskip("pycocotools.mask")
        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[1:5, 1:5] = 1
        rle = coco_mask.encode(np.asfortranarray(expected))
        rle["counts"] = rle["counts"].decode("ascii")
        path = _write_dataset(tmp_path, rle)

        dataset = DetectionDataset.from_coco(
            str(tmp_path), str(path), mask_rasterizer=rasterizer
        )

        np.testing.assert_array_equal(
            next(iter(dataset.annotations.values())).mask[0], expected
        )

    def test_reports_missing_optional_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reference mode reports the installation command instead of falling back."""
        path = _write_dataset(tmp_path, [[1, 1, 5, 1, 5, 5, 1, 5]])
        original_import = builtins.__import__

        def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            """Simulate an installation without the optional COCO dependency."""
            if name == "pycocotools" or name.startswith("pycocotools."):
                raise ImportError("pycocotools is unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(ImportError, match=r"supervision\[coco\]"):
            DetectionDataset.from_coco(
                str(tmp_path), str(path), mask_rasterizer="pycocotools"
            )

    @pytest.mark.parametrize("vertex", [float("nan"), float("inf")])
    def test_rejects_non_finite_geometry(self, vertex: float) -> None:
        """Bad coordinates are rejected before invoking the native rasterizer."""
        annotation = {"id": 7, "segmentation": [[1, 1, 5, 1, vertex, 5, 1, 5]]}

        with pytest.raises(ValueError, match="id=7 has a vertex that is not a finite"):
            coco_annotations_to_masks(
                [annotation], (8, 8), mask_rasterizer="pycocotools"
            )

    @pytest.mark.parametrize(
        "segmentation",
        [
            pytest.param([], id="missing"),
            pytest.param(None, id="null"),
            pytest.param([[]], id="empty-part"),
            pytest.param([[1, 1, 2, 2]], id="two-vertices"),
        ],
    )
    def test_handles_missing_or_degenerate_polygons(self, segmentation: Any) -> None:
        """Missing or insufficient polygon geometry produces an aligned empty mask."""
        annotation = {"id": 7, "segmentation": segmentation}

        if segmentation == [[]]:
            with pytest.warns(UserWarning, match="Skipping empty polygon"):
                masks = coco_annotations_to_masks(
                    [annotation], (8, 8), mask_rasterizer="pycocotools"
                )
        else:
            masks = coco_annotations_to_masks(
                [annotation], (8, 8), mask_rasterizer="pycocotools"
            )

        assert masks.shape == (1, 8, 8)
        assert not masks.any()

    def test_rejects_unknown_rasterizer(self, tmp_path: Path) -> None:
        """A misspelled selector fails even when an image has no polygons."""
        path = _write_dataset(tmp_path, [])

        with pytest.raises(ValueError, match="mask_rasterizer"):
            DetectionDataset.from_coco(str(tmp_path), str(path), mask_rasterizer="typo")

    def test_generic_rasterizer_keeps_its_convention(self) -> None:
        """COCO compatibility does not alter generic inclusive polygon drawing."""
        polygon = np.array([[1, 1], [5, 1], [5, 5], [1, 5]])

        mask = polygon_to_mask(polygon, (8, 8))

        assert mask.sum() == 25
