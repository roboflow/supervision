"""Lossless, opt-in compact masks for YOLO dataset loading."""

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch
from weakref import ReferenceType, ref

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from supervision import _cv2
from supervision._cv2._drawing import _fill_poly
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.core import DetectionDataset
from supervision.dataset.formats.yolo import yolo_annotations_to_detections
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.converters import polygon_to_mask


@pytest.fixture(
    params=[
        pytest.param(False, id="installed"),
        pytest.param(True, id="numpy-fallback"),
    ]
)
def raster_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the installed rasterizer and the NumPy fallback independently."""
    if request.param:
        monkeypatch.setattr(_cv2, "fillPoly", _fill_poly)


_with_raster_backend = pytest.mark.usefixtures("raster_backend")

_skip_unless_cpython = pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="relies on CPython reference counting to release the full frame",
)


def _random_polygon_lines(seed: int, count: int) -> list[str]:
    """Build ``count`` reproducible polygon label lines that may cross borders."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(count):
        class_id = int(rng.integers(0, 2))
        vertices = rng.uniform(-0.2, 1.2, size=(int(rng.integers(3, 7)), 2))
        coordinates = " ".join(f"{value:.4f}" for value in vertices.ravel())
        lines.append(f"{class_id} {coordinates}")
    return lines


#: Distinct polygon labels per image, so an annotation swap between images shows.
_DISTINCT_LABELS = {
    "a": "0 0.1 0.2 0.6 0.2 0.3 0.7\n",
    "b": "1 0.2 0.3 0.8 0.3 0.5 0.9\n",
}

#: Distinct ``(width, height)`` per image, so a size mix-up between images shows.
_DISTINCT_SIZES = {"a": (100, 80), "b": (60, 40)}


def _write_dataset(
    root: Path,
    label: str,
    labels: Mapping[str, str] | None = None,
    sizes: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[str, str, str]:
    """Write images ``a`` and ``b`` with a default or per-image label and size.

    Without ``labels`` and ``sizes`` both images share ``label`` and are 100x80. A
    mapping per image, keyed by ``"a"`` or ``"b"``, overrides the default for the names
    it lists, with ``sizes`` given as ``(width, height)``.
    """
    labels = labels or {}
    sizes = sizes or {}
    images = root / "images"
    annotations = root / "labels"
    images.mkdir()
    annotations.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", sizes.get(name, (100, 80))).save(images / f"{name}.png")
        (annotations / f"{name}.txt").write_text(labels.get(name, label))
    data_yaml = root / "data.yaml"
    data_yaml.write_text("names: ['cat', 'dog']\n")
    return str(images), str(annotations), str(data_yaml)


def _annotation_named(dataset: DetectionDataset, name: str) -> Detections:
    """Return the detections of the image file called ``name`` (with extension)."""
    return next(
        detections
        for image_path, detections in dataset.annotations.items()
        if Path(image_path).name == name
    )


def _read_coco_without_timestamps(path: Path) -> dict[str, Any]:
    """Load a COCO file, dropping the per-export ``date_captured`` timestamps."""
    coco: dict[str, Any] = json.loads(path.read_text())
    for image in coco["images"]:
        image.pop("date_captured")
    return coco


class TestYoloCompactMasks:
    """Compact parsing keeps the dense loader's pixels and detection metadata."""

    @_with_raster_backend
    @pytest.mark.parametrize(
        "lines",
        [
            pytest.param(["0 0.1 0.2 0.6 0.2 0.3 0.7"], id="triangle"),
            pytest.param(["1 0.105 0.20625 0.705 0.2 0.3 0.70625"], id="rounding"),
            pytest.param(["0 0.106 0.2 0.706 0.2 0.3 0.707"], id="rounds-up"),
            pytest.param(["0 -0.2 -0.1 1.2 0.2 0.6 1.2"], id="crosses-borders"),
            pytest.param(["0 -0.5 0.2 -0.1 0.2 -0.2 0.8"], id="outside-left"),
            pytest.param(["0 1.1 1.2 1.8 1.2 1.4 1.8"], id="outside-bottom-right"),
            pytest.param(["0 0 0 1 0 1 1 0 1"], id="image-boundary"),
            pytest.param(["0 0.3 0.4 0.3 0.4 0.3 0.4"], id="single-pixel"),
            pytest.param(["0 0.1 0.4 0.8 0.4 0.4 0.4"], id="collinear"),
            pytest.param(["0 0.1 0.1 0.8 0.8 0.8 0.1 0.1 0.8"], id="self-crossing"),
            pytest.param(["0 0.1 0.1 0.8 0.1 0.2 0.2 0.1 0.8"], id="concave"),
            pytest.param(["1 0.5 0.5 0.2 0.4"], id="forced-box"),
            pytest.param(["1 0.5 0.5 0 0"], id="zero-size-box"),
            pytest.param(
                ["0 0.1 0.2 0.6 0.2 0.3 0.7 0.9", "1 0.8 0.5 0.2 0.4 0.7"],
                id="mixed-with-trailing-confidence",
            ),
        ],
    )
    def test_preserves_pixels_and_metadata(self, lines: list[str]) -> None:
        """Rounding, clipping and degenerate polygons match the dense path."""
        expected = yolo_annotations_to_detections(lines, (100, 80), with_masks=True)

        actual = yolo_annotations_to_detections(
            lines, (100, 80), with_masks=True, compact_masks=True
        )

        assert isinstance(actual.mask, CompactMask)
        np.testing.assert_array_equal(actual.mask.to_dense(), expected.mask)
        np.testing.assert_array_equal(actual.xyxy, expected.xyxy)
        np.testing.assert_array_equal(actual.class_id, expected.class_id)
        assert actual.data == expected.data

    @_with_raster_backend
    @pytest.mark.parametrize(
        "resolution_wh",
        [
            pytest.param((1, 1), id="1x1"),
            pytest.param((1, 7), id="1xN"),
            pytest.param((7, 1), id="Nx1"),
            pytest.param((3, 2), id="tiny"),
        ],
    )
    @pytest.mark.parametrize(
        "lines",
        [
            pytest.param(["0 0.1 0.2 0.6 0.2 0.3 0.7"], id="triangle"),
            pytest.param(["0 -0.2 -0.1 1.2 0.2 0.6 1.2"], id="crosses-borders"),
            pytest.param(["0 1.1 1.2 1.8 1.2 1.4 1.8"], id="outside"),
        ],
    )
    def test_matches_dense_loader_on_degenerate_resolutions(
        self, lines: list[str], resolution_wh: tuple[int, int]
    ) -> None:
        """One-pixel-wide, one-pixel-tall and tiny frames match the dense path."""
        expected = yolo_annotations_to_detections(lines, resolution_wh, with_masks=True)

        actual = yolo_annotations_to_detections(
            lines, resolution_wh, with_masks=True, compact_masks=True
        )

        assert isinstance(actual.mask, CompactMask)
        assert actual.mask.image_shape == (resolution_wh[1], resolution_wh[0])
        np.testing.assert_array_equal(actual.mask.to_dense(), expected.mask)

    @_with_raster_backend
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_dense_loader_on_random_polygons(self, seed: int) -> None:
        """Seeded random polygons, rounded up or down, rasterize like the dense path.

        Random fractional vertices exercise both rounding directions and clipping, which
        a crop derived from unrounded bounds would get wrong.
        """
        lines = _random_polygon_lines(seed, count=12)
        expected = yolo_annotations_to_detections(lines, (100, 80), with_masks=True)

        actual = yolo_annotations_to_detections(
            lines, (100, 80), with_masks=True, compact_masks=True
        )

        assert isinstance(actual.mask, CompactMask)
        np.testing.assert_array_equal(actual.mask.to_dense(), expected.mask)

    def test_small_polygon_on_large_image_retains_only_its_extent(self) -> None:
        """A tiny polygon keeps crop-sized storage while reporting full-frame shape.

        The polygon spans pixels 200..206 by 200..210 of a 2000x1000 frame, so its
        retained crop must follow that rounded extent, not the image size.
        """
        lines = ["0 0.1 0.2 0.103 0.2 0.1015 0.21"]

        actual = yolo_annotations_to_detections(
            lines, (2000, 1000), with_masks=True, compact_masks=True
        )

        assert isinstance(actual.mask, CompactMask)
        assert actual.mask.image_shape == (1000, 2000)
        assert actual.mask.shape == (1, 1000, 2000)
        assert actual.mask.bbox_xyxy.tolist() == [[200, 200, 206, 210]]
        assert actual.mask.crop(0).shape == (11, 7)

    @pytest.mark.parametrize("compact_masks", [False, True])
    @pytest.mark.parametrize("coordinate", ["nan", "inf", "-inf"])
    def test_rejects_non_finite_coordinates(
        self, coordinate: str, compact_masks: bool
    ) -> None:
        """A nan or infinite polygon coordinate raises the same error in both modes."""
        lines = [f"0 {coordinate} 0.2 0.6 0.2 0.3 0.7"]

        with pytest.raises(ValueError, match="finite"):
            yolo_annotations_to_detections(
                lines, (100, 80), with_masks=True, compact_masks=compact_masks
            )

    @_with_raster_backend
    @pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
    def test_huge_finite_coordinate_matches_dense_loader(self) -> None:
        """Non-contract: a huge finite vertex yields the same masks in both modes.

        The float to int32 cast of such a vertex is platform-defined, so only agreement
        between the compact and dense loaders is asserted, not pixels.
        """
        lines = ["0 0.1 0.2 1e30 0.2 0.3 0.7"]
        expected = yolo_annotations_to_detections(lines, (100, 80), with_masks=True)

        actual = yolo_annotations_to_detections(
            lines, (100, 80), with_masks=True, compact_masks=True
        )

        assert isinstance(actual.mask, CompactMask)
        np.testing.assert_array_equal(actual.mask.to_dense(), expected.mask)

    @_with_raster_backend
    @_skip_unless_cpython
    def test_encodes_each_polygon_before_rasterizing_the_next(self) -> None:
        """CPython-only: each full frame is freed before the next polygon is drawn."""
        lines = ["0 0.1 0.2 0.2 0.2 0.2 0.3"] * 8
        original_encode = CompactMask.from_dense
        encoded: list[tuple[int, ...]] = []
        frames: list[ReferenceType[npt.NDArray[np.bool_]]] = []

        def rasterize(
            polygon: npt.NDArray[np.number], resolution_wh: tuple[int, int]
        ) -> npt.NDArray[np.uint8]:
            """Require the previous full frame to be released before drawing."""
            assert not frames or frames[-1]() is None
            return polygon_to_mask(polygon, resolution_wh)

        def encode_one(
            *,
            masks: npt.NDArray[np.bool_],
            xyxy: npt.NDArray[np.number],
            image_shape: tuple[int, int],
        ) -> CompactMask:
            """Record the dense input size without retaining its pixels."""
            encoded.append(masks.shape)
            assert isinstance(masks.base, np.ndarray)
            frames.append(ref(masks.base))
            return original_encode(masks=masks, xyxy=xyxy, image_shape=image_shape)

        with (
            patch(
                "supervision.dataset.formats.yolo._polygons_to_masks",
                side_effect=AssertionError("dense batch allocation"),
            ),
            patch("supervision.dataset.formats.yolo.polygon_to_mask", new=rasterize),
            patch.object(CompactMask, "from_dense", new=encode_one),
        ):
            actual = yolo_annotations_to_detections(
                lines, (100, 80), with_masks=True, compact_masks=True
            )

        assert isinstance(actual.mask, CompactMask)
        assert encoded == [(1, 80, 100)] * 8
        assert all(frame() is None for frame in frames)
        assert np.all(actual.mask.bbox_xyxy == [10, 16, 20, 24])

    @pytest.mark.parametrize("compact_masks", [False, True])
    def test_box_only_loading_does_not_create_masks(self, compact_masks: bool) -> None:
        """The storage option does not implicitly enable mask generation."""
        actual = yolo_annotations_to_detections(
            ["1 0.5 0.5 0.2 0.4"], (100, 80), False, compact_masks=compact_masks
        )

        assert actual.mask is None

    def test_default_remains_dense(self) -> None:
        """Existing callers continue receiving a boolean NumPy mask stack."""
        actual = yolo_annotations_to_detections(
            ["0 0.1 0.2 0.6 0.2 0.3 0.7"], (100, 80), True
        )

        assert isinstance(actual.mask, np.ndarray)
        assert actual.mask.dtype == np.bool_


class TestFromYoloCompactMasks:
    """The public dataset API propagates compact storage through normal workflows."""

    @pytest.mark.parametrize("force_masks", [False, True])
    def test_loads_segmentation_masks(self, tmp_path: Path, force_masks: bool) -> None:
        """Automatically detected and forced segmentation use compact storage."""
        paths = _write_dataset(tmp_path, "0 0.1 0.2 0.6 0.2 0.3 0.7\n")

        dataset = DetectionDataset.from_yolo(
            *paths, force_masks=force_masks, compact_masks=True
        )

        assert len(dataset) == 2
        assert all(
            isinstance(d.mask, CompactMask) for d in dataset.annotations.values()
        )

    @pytest.mark.parametrize(
        "label",
        [
            pytest.param("", id="empty-file"),
            pytest.param("1 0.5 0.5 0.2 0.4\n", id="box-only"),
        ],
    )
    def test_keeps_non_segmentation_masks_absent(
        self, tmp_path: Path, label: str
    ) -> None:
        """Empty files and ordinary boxes retain their existing mask=None result."""
        paths = _write_dataset(tmp_path, label)

        dataset = DetectionDataset.from_yolo(*paths, compact_masks=True)

        assert all(d.mask is None for d in dataset.annotations.values())

    def test_missing_labels_remain_empty(self, tmp_path: Path) -> None:
        """An unlabeled image does not acquire a mask from the storage flag."""
        paths = _write_dataset(tmp_path, "")
        (tmp_path / "labels" / "a.txt").unlink()

        dataset = DetectionDataset.from_yolo(
            *paths, compact_masks=True, force_masks=True
        )

        assert all(len(d) == 0 and d.mask is None for d in dataset.annotations.values())

    @pytest.mark.parametrize("name", ["a.png", "b.png"])
    def test_obb_keeps_corners_and_no_masks(self, tmp_path: Path, name: str) -> None:
        """Compact storage does not reinterpret oriented boxes as segmentation."""
        paths = _write_dataset(tmp_path, "0 0.1 0.2 0.6 0.2 0.6 0.7 0.1 0.7\n")

        dataset = DetectionDataset.from_yolo(*paths, is_obb=True, compact_masks=True)

        detections = _annotation_named(dataset, name)
        assert detections.mask is None
        np.testing.assert_allclose(
            np.asarray(detections.data[ORIENTED_BOX_COORDINATES], dtype=np.float32),
            [[[10, 16], [60, 16], [60, 56], [10, 56]]],
        )

    @pytest.mark.parametrize("name", ["a.png", "b.png"])
    def test_pose_force_masks_uses_boxes(self, tmp_path: Path, name: str) -> None:
        """Pose keypoints are stripped before forced box masks are encoded."""
        paths = _write_dataset(tmp_path, "0 0.5 0.5 0.2 0.4 0.1 0.2 2\n")
        (tmp_path / "data.yaml").write_text(
            "names: ['cat', 'dog']\nkpt_shape: [1, 3]\n"
        )
        expected = DetectionDataset.from_yolo(*paths, force_masks=True)

        actual = DetectionDataset.from_yolo(
            *paths, force_masks=True, compact_masks=True
        )

        actual_mask = _annotation_named(actual, name).mask
        assert isinstance(actual_mask, CompactMask)
        np.testing.assert_array_equal(
            actual_mask.to_dense(), _annotation_named(expected, name).mask
        )

    def test_split_and_merge_preserve_compact_masks(self, tmp_path: Path) -> None:
        """Dataset partitioning and class remapping do not densify annotations."""
        paths = _write_dataset(
            tmp_path, "", labels=_DISTINCT_LABELS, sizes=_DISTINCT_SIZES
        )
        dataset = DetectionDataset.from_yolo(*paths, compact_masks=True)
        train, valid = dataset.split(split_ratio=0.5, shuffle=False)

        merged = DetectionDataset.merge([train, valid])

        assert len(train) == len(valid) == 1
        assert merged == dataset
        assert all(isinstance(d.mask, CompactMask) for d in merged.annotations.values())

    def test_export_matches_dense_dataset(self, tmp_path: Path) -> None:
        """YOLO export sees the same polygon pixels in either mask representation."""
        paths = _write_dataset(
            tmp_path, "", labels=_DISTINCT_LABELS, sizes=_DISTINCT_SIZES
        )
        dense = DetectionDataset.from_yolo(*paths)
        compact = DetectionDataset.from_yolo(*paths, compact_masks=True)
        dense_dir, compact_dir = tmp_path / "dense", tmp_path / "compact"
        dense.as_yolo(annotations_directory_path=str(dense_dir))

        compact.as_yolo(annotations_directory_path=str(compact_dir))

        assert (compact_dir / "a.txt").read_text() == (dense_dir / "a.txt").read_text()
        assert (compact_dir / "b.txt").read_text() == (dense_dir / "b.txt").read_text()

    def test_coco_export_matches_dense_dataset(self, tmp_path: Path) -> None:
        """COCO export keeps each image's own polygon, class and size per image."""
        paths = _write_dataset(
            tmp_path, "", labels=_DISTINCT_LABELS, sizes=_DISTINCT_SIZES
        )
        dense = DetectionDataset.from_yolo(*paths)
        compact = DetectionDataset.from_yolo(*paths, compact_masks=True)
        dense.as_coco(annotations_path=str(tmp_path / "dense.json"))

        compact.as_coco(annotations_path=str(tmp_path / "compact.json"))

        assert _read_coco_without_timestamps(
            tmp_path / "compact.json"
        ) == _read_coco_without_timestamps(tmp_path / "dense.json")

    def test_pascal_voc_export_matches_dense_dataset(self, tmp_path: Path) -> None:
        """PASCAL VOC export writes the same per-image boxes for either mask storage."""
        paths = _write_dataset(
            tmp_path, "", labels=_DISTINCT_LABELS, sizes=_DISTINCT_SIZES
        )
        dense = DetectionDataset.from_yolo(*paths)
        compact = DetectionDataset.from_yolo(*paths, compact_masks=True)
        dense_dir, compact_dir = tmp_path / "dense", tmp_path / "compact"
        dense.as_pascal_voc(annotations_directory_path=str(dense_dir))

        compact.as_pascal_voc(annotations_directory_path=str(compact_dir))

        assert (compact_dir / "a.xml").read_text() == (dense_dir / "a.xml").read_text()
        assert (compact_dir / "b.xml").read_text() == (dense_dir / "b.xml").read_text()
