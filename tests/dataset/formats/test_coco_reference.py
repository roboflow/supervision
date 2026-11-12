"""COCO reference rasterization must preserve continuous polygon geometry."""

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from PIL import Image

from supervision import DetectionDataset
from supervision.dataset.formats import coco as coco_module
from supervision.dataset.formats.coco import (
    _IS_PYCOCOTOOLS_AVAILABLE,
    coco_annotations_to_masks,
)
from supervision.utils.internal import SupervisionWarnings

if _IS_PYCOCOTOOLS_AVAILABLE:
    from pycocotools import mask as coco_mask

#: Silences the NumPy 2 DeprecationWarning that ``pycocotools.mask.decode`` emits.
#: Only the tests that build their expected mask with ``decode`` carry it; every
#: other test runs under the project's error-on-deprecation filter.
_IGNORE_REFERENCE_DECODE_WARNING = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn.t accept a copy keyword:DeprecationWarning"
)
#: A well-formed polygon part with fractional vertices, used next to degenerate parts.
_SUBPIXEL_SQUARE = [1.6, 1.6, 5.6, 1.6, 5.6, 5.6, 1.6, 5.6]


def _write_dataset(
    tmp_path: Path,
    segmentation: Any,
    resolution_wh: tuple[int, int] = (8, 8),
) -> Path:
    """Write a synthetic image and one-object COCO dataset at the requested size."""
    width, height = resolution_wh
    Image.new("RGB", resolution_wh).save(tmp_path / "image.png")
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "image.png",
                        "width": width,
                        "height": height,
                    }
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


# The missing-extra tests need no ``pycocotools`` and must also run where it is
# absent; only tests that call or compare against it are gated.
_requires_pycocotools = pytest.mark.skipif(
    not _IS_PYCOCOTOOLS_AVAILABLE,
    reason="pycocotools is required as the reference implementation",
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
    @_IGNORE_REFERENCE_DECODE_WARNING
    @_requires_pycocotools
    def test_loader_matches_reference(
        self, tmp_path: Path, polygons: list[list[float]]
    ) -> None:
        """The public loader preserves fractional vertices and unions all parts."""
        path = _write_dataset(tmp_path, polygons)
        reference = coco_mask.decode(
            coco_mask.merge(coco_mask.frPyObjects(polygons, 8, 8))
        )

        dataset = DetectionDataset.from_coco(
            str(tmp_path), str(path), mask_rasterizer="pycocotools"
        )

        detections = next(iter(dataset.annotations.values()))
        assert detections.mask is not None
        np.testing.assert_array_equal(detections.mask[0], reference)
        assert detections.mask.dtype == bool

    @_requires_pycocotools
    def test_keeps_box_and_class_id(self, tmp_path: Path) -> None:
        """Reference rasterization leaves the annotation's box and class unchanged."""
        path = _write_dataset(tmp_path, [[1.6, 1.6, 5.6, 1.6, 5.6, 5.6, 1.6, 5.6]])

        dataset = DetectionDataset.from_coco(
            str(tmp_path), str(path), mask_rasterizer="pycocotools"
        )

        detections = next(iter(dataset.annotations.values()))
        np.testing.assert_array_equal(detections.class_id, [0])
        np.testing.assert_array_equal(detections.xyxy, [[1, 1, 5, 5]])

    @_requires_pycocotools
    def test_loads_polygons_without_deprecation_warning(self, tmp_path: Path) -> None:
        """Reference polygon loading emits no NumPy deprecation warning."""
        path = _write_dataset(tmp_path, [[1, 1, 5, 1, 5, 5, 1, 5]])
        # pycocotools' decode warns under NumPy 2, so the documented 16-pixel square
        # stands in for the reference mask and keeps this test free of the ignore mark.
        expected = np.zeros((8, 8), dtype=bool)
        expected[1:5, 1:5] = True

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dataset = DetectionDataset.from_coco(
                str(tmp_path), str(path), mask_rasterizer="pycocotools"
            )

        detections = next(iter(dataset.annotations.values()))
        assert detections.mask is not None
        np.testing.assert_array_equal(detections.mask[0], expected)

    @_IGNORE_REFERENCE_DECODE_WARNING
    @_requires_pycocotools
    def test_loader_matches_reference_on_rectangular_image(
        self, tmp_path: Path
    ) -> None:
        """The public loader uses width and height in the COCO order."""
        polygons = [[1.6, 1.6, 8.6, 1.6, 8.6, 5.6, 1.6, 5.6]]
        width, height = 11, 7
        path = _write_dataset(tmp_path, polygons, (width, height))
        reference = coco_mask.decode(
            coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
        )

        dataset = DetectionDataset.from_coco(
            str(tmp_path), str(path), mask_rasterizer="pycocotools"
        )

        detections = next(iter(dataset.annotations.values()))
        assert detections.mask is not None
        np.testing.assert_array_equal(detections.mask[0], reference)
        assert detections.mask[0].shape == (height, width)

    @pytest.mark.parametrize(
        "polygons",
        [
            pytest.param([_SUBPIXEL_SQUARE, [1, 1, 2, 2]], id="two-vertices-last"),
            pytest.param([[1, 1, 2, 2], _SUBPIXEL_SQUARE], id="two-vertices-first"),
            pytest.param([[3, 3], _SUBPIXEL_SQUARE], id="one-vertex-first"),
        ],
    )
    @_IGNORE_REFERENCE_DECODE_WARNING
    @_requires_pycocotools
    def test_skips_degenerate_parts_like_reference(
        self, polygons: list[list[float]]
    ) -> None:
        """Parts under three vertices are dropped and the rest match the reference."""
        annotation = {"id": 7, "segmentation": polygons}
        # pycocotools raises TypeError when a degenerate part comes first, so the
        # reference is built from the well-formed part alone.
        reference = coco_mask.decode(
            coco_mask.merge(coco_mask.frPyObjects([_SUBPIXEL_SQUARE], 8, 8))
        )

        masks = coco_annotations_to_masks(
            [annotation], (8, 8), mask_rasterizer="pycocotools"
        )

        np.testing.assert_array_equal(masks[0], reference)

    @_requires_pycocotools
    def test_aligns_masks_across_mixed_segmentations(self) -> None:
        """Polygon, RLE and empty annotations keep one aligned mask each, in order."""
        rle_pixels = np.zeros((8, 8), dtype=np.uint8)
        rle_pixels[5:7, 0:3] = 1
        rle = coco_mask.encode(np.asfortranarray(rle_pixels))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations = [
            {"id": 1, "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]},
            {"id": 2, "segmentation": rle},
            {"id": 3, "segmentation": []},
            {"id": 4, "segmentation": [[4, 4, 7, 4, 7, 7, 4, 7]]},
        ]
        expected = np.zeros((4, 8, 8), dtype=bool)
        expected[0, 1:5, 1:5] = True
        expected[1, 5:7, 0:3] = True
        expected[3, 4:7, 4:7] = True

        masks = coco_annotations_to_masks(
            annotations, (8, 8), mask_rasterizer="pycocotools"
        )

        np.testing.assert_array_equal(masks, expected)

    @pytest.mark.parametrize(
        "rasterizer_kwargs",
        [
            pytest.param({}, id="omitted"),
            pytest.param({"mask_rasterizer": None}, id="explicit-none"),
        ],
    )
    def test_preserves_default_inclusive_mask(
        self, tmp_path: Path, rasterizer_kwargs: dict[str, Any]
    ) -> None:
        """Existing callers still receive the documented inclusive square."""
        path = _write_dataset(tmp_path, [[1, 1, 5, 1, 5, 5, 1, 5]])
        expected = np.zeros((8, 8), dtype=bool)
        expected[1:6, 1:6] = True

        with pytest.warns(SupervisionWarnings, match="no earlier than `0.35.0`"):
            dataset = DetectionDataset.from_coco(
                str(tmp_path), str(path), **rasterizer_kwargs
            )

        detections = next(iter(dataset.annotations.values()))
        assert detections.mask is not None
        np.testing.assert_array_equal(detections.mask[0], expected)

    def test_default_loads_without_optional_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting the selector keeps the inclusive mask when pycocotools is absent."""
        monkeypatch.setattr(coco_module, "_IS_PYCOCOTOOLS_AVAILABLE", False)
        path = _write_dataset(tmp_path, [[1, 1, 5, 1, 5, 5, 1, 5]])
        expected = np.zeros((8, 8), dtype=bool)
        expected[1:6, 1:6] = True

        with pytest.warns(SupervisionWarnings, match="no earlier than `0.35.0`"):
            dataset = DetectionDataset.from_coco(str(tmp_path), str(path))

        detections = next(iter(dataset.annotations.values()))
        assert detections.mask is not None
        np.testing.assert_array_equal(detections.mask[0], expected)

    @pytest.mark.parametrize(
        "rasterizer_kwargs",
        [
            pytest.param({}, id="omitted"),
            pytest.param({"mask_rasterizer": None}, id="explicit-none"),
            pytest.param({"mask_rasterizer": "pycocotools"}, id="reference"),
        ],
    )
    @_requires_pycocotools
    def test_loads_box_only_file_without_masks_or_warning(
        self, tmp_path: Path, rasterizer_kwargs: dict[str, Any]
    ) -> None:
        """A file without segmentation loads no masks and emits no selector warning."""
        path = _write_dataset(tmp_path, [])

        with warnings.catch_warnings():
            warnings.simplefilter("error", SupervisionWarnings)
            dataset = DetectionDataset.from_coco(
                str(tmp_path), str(path), **rasterizer_kwargs
            )

        assert next(iter(dataset.annotations.values())).mask is None

    @pytest.mark.parametrize("rasterizer", ["supervision", "pycocotools"])
    @_requires_pycocotools
    def test_preserves_rle(
        self, tmp_path: Path, rasterizer: Literal["supervision", "pycocotools"]
    ) -> None:
        """The selector affects polygon conversion only, preserving RLE pixels."""
        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[1:5, 1:5] = 1
        rle = coco_mask.encode(np.asfortranarray(expected))
        rle["counts"] = rle["counts"].decode("ascii")
        path = _write_dataset(tmp_path, rle)

        dataset = DetectionDataset.from_coco(
            str(tmp_path), str(path), mask_rasterizer=rasterizer
        )

        detections = next(iter(dataset.annotations.values()))
        assert detections.mask is not None
        np.testing.assert_array_equal(detections.mask[0], expected)

    @pytest.mark.parametrize(
        "segmentation",
        [
            pytest.param([[1, 1, 5, 1, 5, 5, 1, 5]], id="polygon"),
            pytest.param({"size": [8, 8], "counts": [64]}, id="rle-only"),
            pytest.param([], id="box-only"),
        ],
    )
    def test_reports_missing_optional_dependency(
        self, tmp_path: Path, segmentation: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reference mode reports the install command at load start, polygons or not."""
        monkeypatch.setattr(coco_module, "_IS_PYCOCOTOOLS_AVAILABLE", False)
        path = _write_dataset(tmp_path, segmentation)

        with pytest.raises(ImportError, match=r"supervision\[coco\]"):
            DetectionDataset.from_coco(
                str(tmp_path), str(path), mask_rasterizer="pycocotools"
            )

    @pytest.mark.parametrize("vertex", [float("nan"), float("inf")])
    def test_rejects_non_finite_vertex(self, vertex: float) -> None:
        """Non-finite vertices raise a ValueError naming the annotation id."""
        annotation = {"id": 7, "segmentation": [[1, 1, 5, 1, vertex, 5, 1, 5]]}

        with pytest.raises(ValueError, match=r"id=7 has a vertex .*not a finite"):
            coco_annotations_to_masks(
                [annotation], (8, 8), mask_rasterizer="pycocotools"
            )

    @pytest.mark.parametrize(
        "segmentation",
        [
            pytest.param([], id="missing"),
            pytest.param(None, id="null"),
            pytest.param([[1, 1, 2, 2]], id="two-vertices"),
        ],
    )
    def test_handles_missing_or_degenerate_polygons(self, segmentation: Any) -> None:
        """Missing or insufficient polygon geometry produces an aligned empty mask."""
        annotation = {"id": 7, "segmentation": segmentation}

        masks = coco_annotations_to_masks(
            [annotation], (8, 8), mask_rasterizer="pycocotools"
        )

        assert masks.shape == (1, 8, 8)
        assert not masks.any()

    def test_warns_on_empty_polygon_part(self) -> None:
        """An empty polygon part is skipped with a warning, leaving an empty mask."""
        annotation = {"id": 7, "segmentation": [[]]}

        with pytest.warns(UserWarning, match="Skipping empty polygon"):
            masks = coco_annotations_to_masks(
                [annotation], (8, 8), mask_rasterizer="pycocotools"
            )

        assert masks.shape == (1, 8, 8)
        assert not masks.any()

    @pytest.mark.parametrize(
        "rasterizer", ["typo", 1, pytest.param(b"pycocotools", id="bytes-name")]
    )
    def test_rejects_unknown_rasterizer(self, tmp_path: Path, rasterizer: Any) -> None:
        """An unsupported or non-string selector fails even without polygons."""
        path = _write_dataset(tmp_path, [])

        with pytest.raises(ValueError, match="mask_rasterizer"):
            DetectionDataset.from_coco(
                str(tmp_path), str(path), mask_rasterizer=rasterizer
            )
