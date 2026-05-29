from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import InferenceSlicer


class _FakeCRS:
    """Minimal rasterio-style CRS stub exposing only `is_projected`."""

    def __init__(self, is_projected: bool):
        self.is_projected = is_projected

    def __repr__(self) -> str:
        kind = "projected" if self.is_projected else "geographic"
        return f"_FakeCRS({kind})"


class _FakeRasterDataset:
    """Lightweight rasterio-style dataset supporting windowed reads.

    Mimics the duck-typed interface that ``InferenceSlicer`` relies on without
    requiring ``rasterio`` to be installed.
    """

    def __init__(self, image_hwc: np.ndarray, crs: object | None = None):
        self._image = image_hwc  # numpy (H, W, C)
        self.height, self.width = image_hwc.shape[:2]
        self.crs = crs  # None or object with .is_projected

    def read(self, window: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
        (row_start, row_stop), (col_start, col_stop) = window
        crop = self._image[row_start:row_stop, col_start:col_stop, :]
        return np.transpose(crop, (2, 0, 1))  # (C, H, W) like rasterio


def _fixed_detection_callback(_: np.ndarray) -> Detections:
    """Return a constant detection for every tile."""
    return Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def _sortable(detections: Detections) -> np.ndarray:
    """Sort detection boxes so two runs can be compared order-independently."""
    return np.array(
        sorted(detections.xyxy.tolist()),
        dtype=float,
    )


def test_windowed_raster_matches_in_memory_array() -> None:
    # Arrange
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    dataset = _FakeRasterDataset(image, crs=_FakeCRS(is_projected=True))
    slicer = InferenceSlicer(
        callback=_fixed_detection_callback,
        slice_wh=128,
        overlap_wh=0,
    )

    # Act
    detections_array = slicer(image)
    detections_raster = slicer(dataset)

    # Assert
    assert np.array_equal(_sortable(detections_array), _sortable(detections_raster))


def test_windowed_raster_reads_correct_window_content() -> None:
    """The windowed read must return the same pixels crop_image would, so the
    callback sees identical tile content for both input types."""
    # Arrange
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(128, 192, 3), dtype=np.uint8)
    dataset = _FakeRasterDataset(image)

    seen_array_tiles: list[np.ndarray] = []
    seen_raster_tiles: list[np.ndarray] = []

    def recording_callback(sink: list[np.ndarray]):
        def callback(tile: np.ndarray) -> Detections:
            sink.append(tile.copy())
            return Detections.empty()

        return callback

    slicer_array = InferenceSlicer(
        callback=recording_callback(seen_array_tiles),
        slice_wh=64,
        overlap_wh=0,
    )
    slicer_raster = InferenceSlicer(
        callback=recording_callback(seen_raster_tiles),
        slice_wh=64,
        overlap_wh=0,
    )

    # Act
    slicer_array(image)
    slicer_raster(dataset)

    # Assert
    assert len(seen_array_tiles) == len(seen_raster_tiles)
    for array_tile, raster_tile in zip(seen_array_tiles, seen_raster_tiles):
        assert np.array_equal(array_tile, raster_tile)


def test_windowed_raster_matches_in_memory_array_with_overlap() -> None:
    """Overlapping tiles must read identical windows for both input types."""
    # Arrange
    rng = np.random.default_rng(99)
    image = rng.integers(0, 255, size=(200, 220, 3), dtype=np.uint8)
    dataset = _FakeRasterDataset(image)

    seen_array_tiles: list[np.ndarray] = []
    seen_raster_tiles: list[np.ndarray] = []

    def recording_callback(sink: list[np.ndarray]):
        def callback(tile: np.ndarray) -> Detections:
            sink.append(tile.copy())
            return Detections.empty()

        return callback

    slicer_array = InferenceSlicer(
        callback=recording_callback(seen_array_tiles),
        slice_wh=96,
        overlap_wh=32,
    )
    slicer_raster = InferenceSlicer(
        callback=recording_callback(seen_raster_tiles),
        slice_wh=96,
        overlap_wh=32,
    )

    # Act
    slicer_array(image)
    slicer_raster(dataset)

    # Assert
    assert len(seen_array_tiles) == len(seen_raster_tiles) > 1
    for array_tile, raster_tile in zip(seen_array_tiles, seen_raster_tiles):
        assert np.array_equal(array_tile, raster_tile)


def test_windowed_raster_preserves_band_dtype() -> None:
    """Tiles read from a dataset keep the source dtype (e.g. uint16)."""
    # Arrange
    rng = np.random.default_rng(5)
    image = rng.integers(0, 4000, size=(128, 128, 3), dtype=np.uint16)
    dataset = _FakeRasterDataset(image)

    seen: list[np.ndarray] = []

    def callback(tile: np.ndarray) -> Detections:
        seen.append(tile)
        return Detections.empty()

    slicer = InferenceSlicer(callback=callback, slice_wh=64, overlap_wh=0)

    # Act
    slicer(dataset)

    # Assert
    assert seen
    assert all(tile.dtype == np.uint16 for tile in seen)


def test_windowed_raster_with_no_crs_works() -> None:
    # Arrange
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    dataset = _FakeRasterDataset(image, crs=None)
    slicer = InferenceSlicer(
        callback=_fixed_detection_callback,
        slice_wh=64,
        overlap_wh=0,
    )

    # Act
    detections = slicer(dataset)

    # Assert
    assert len(detections) == 4


def test_windowed_raster_with_geographic_crs_raises() -> None:
    # Arrange
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    dataset = _FakeRasterDataset(image, crs=_FakeCRS(is_projected=False))
    slicer = InferenceSlicer(
        callback=_fixed_detection_callback,
        slice_wh=64,
        overlap_wh=0,
    )

    # Act / Assert
    with pytest.raises(ValueError, match="projected coordinate reference"):
        slicer(dataset)


def test_windowed_raster_with_projected_crs_does_not_raise() -> None:
    # Arrange
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    dataset = _FakeRasterDataset(image, crs=_FakeCRS(is_projected=True))
    slicer = InferenceSlicer(
        callback=_fixed_detection_callback,
        slice_wh=64,
        overlap_wh=0,
    )

    # Act
    detections = slicer(dataset)

    # Assert
    assert len(detections) == 4


def test_real_rasterio_memoryfile_integration() -> None:
    """Integration check against a real rasterio dataset, skipped if rasterio
    is not installed."""
    pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile

    # Arrange
    rng = np.random.default_rng(123)
    image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    bands = np.transpose(image, (2, 0, 1))  # (C, H, W)

    slicer = InferenceSlicer(
        callback=_fixed_detection_callback,
        slice_wh=64,
        overlap_wh=0,
    )
    detections_array = slicer(image)

    profile = {
        "driver": "GTiff",
        "height": image.shape[0],
        "width": image.shape[1],
        "count": image.shape[2],
        "dtype": image.dtype,
    }

    # Act
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(bands)
        with memfile.open() as dataset:
            detections_raster = slicer(dataset)

    # Assert
    assert np.array_equal(_sortable(detections_array), _sortable(detections_raster))
