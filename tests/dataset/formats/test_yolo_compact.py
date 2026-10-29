"""Lossless, opt-in compact masks for YOLO dataset loading."""

from pathlib import Path
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
from supervision.detection.utils.converters import polygon_to_mask


@pytest.fixture(params=[False, True])
def raster_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the installed rasterizer and the NumPy fallback independently."""
    if request.param:
        monkeypatch.setattr(_cv2, "fillPoly", _fill_poly)


def _write_dataset(root: Path, label: str) -> tuple[str, str, str]:
    """Write two images with identical labels for dataset operation tests."""
    images = root / "images"
    labels = root / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (100, 80)).save(images / f"{name}.png")
        (labels / f"{name}.txt").write_text(label)
    data_yaml = root / "data.yaml"
    data_yaml.write_text("names: ['cat', 'dog']\n")
    return str(images), str(labels), str(data_yaml)


@pytest.mark.usefixtures("raster_backend")
class TestYoloCompactMasks:
    """Compact parsing keeps the dense loader's pixels and detection metadata."""

    @pytest.mark.parametrize(
        "lines",
        [
            pytest.param(["0 0.1 0.2 0.6 0.2 0.3 0.7"], id="triangle"),
            pytest.param(["1 0.105 0.20625 0.705 0.2 0.3 0.70625"], id="rounding"),
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

    def test_encodes_each_polygon_before_rasterizing_the_next(self) -> None:
        """No dense batch survives between polygons when loading compact masks."""
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

    @pytest.mark.parametrize("label", ["", "1 0.5 0.5 0.2 0.4\n"])
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

    def test_obb_keeps_corners_and_no_masks(self, tmp_path: Path) -> None:
        """Compact storage does not reinterpret oriented boxes as segmentation."""
        paths = _write_dataset(tmp_path, "0 0.1 0.2 0.6 0.2 0.6 0.7 0.1 0.7\n")

        dataset = DetectionDataset.from_yolo(*paths, is_obb=True, compact_masks=True)

        assert all(d.mask is None for d in dataset.annotations.values())
        for detections in dataset.annotations.values():
            np.testing.assert_allclose(
                np.asarray(detections.data[ORIENTED_BOX_COORDINATES], dtype=np.float32),
                [[[10, 16], [60, 16], [60, 56], [10, 56]]],
            )

    def test_pose_force_masks_uses_boxes(self, tmp_path: Path) -> None:
        """Pose keypoints are stripped before forced box masks are encoded."""
        paths = _write_dataset(tmp_path, "0 0.5 0.5 0.2 0.4 0.1 0.2 2\n")
        (tmp_path / "data.yaml").write_text(
            "names: ['cat', 'dog']\nkpt_shape: [1, 3]\n"
        )
        expected = DetectionDataset.from_yolo(*paths, force_masks=True)

        actual = DetectionDataset.from_yolo(
            *paths, force_masks=True, compact_masks=True
        )

        for path, detections in actual.annotations.items():
            assert isinstance(detections.mask, CompactMask)
            np.testing.assert_array_equal(
                detections.mask.to_dense(), expected.annotations[path].mask
            )

    def test_split_and_merge_preserve_compact_masks(self, tmp_path: Path) -> None:
        """Dataset partitioning and class remapping do not densify annotations."""
        paths = _write_dataset(tmp_path, "1 0.1 0.2 0.6 0.2 0.3 0.7\n")
        dataset = DetectionDataset.from_yolo(*paths, compact_masks=True)
        train, valid = dataset.split(split_ratio=0.5, shuffle=False)

        merged = DetectionDataset.merge([train, valid])

        assert len(train) == len(valid) == 1
        assert merged == dataset
        assert all(isinstance(d.mask, CompactMask) for d in merged.annotations.values())

    def test_export_matches_dense_dataset(self, tmp_path: Path) -> None:
        """YOLO export sees the same polygon pixels in either mask representation."""
        paths = _write_dataset(tmp_path, "0 0.1 0.2 0.6 0.2 0.3 0.7\n")
        dense = DetectionDataset.from_yolo(*paths)
        compact = DetectionDataset.from_yolo(*paths, compact_masks=True)
        dense_dir, compact_dir = tmp_path / "dense", tmp_path / "compact"
        dense.as_yolo(annotations_directory_path=str(dense_dir))

        compact.as_yolo(annotations_directory_path=str(compact_dir))

        assert (compact_dir / "a.txt").read_text() == (dense_dir / "a.txt").read_text()
        assert (compact_dir / "b.txt").read_text() == (dense_dir / "b.txt").read_text()
