"""Tests for windowed GeoTIFF reads in InferenceSlicer."""

from __future__ import annotations

import threading
import time

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


class _ConcurrencyCheckDataset:
    """Dataset that tracks peak concurrent reads to verify read serialization."""

    def __init__(self, image_hwc: np.ndarray):
        self._image = image_hwc
        self.height, self.width = image_hwc.shape[:2]
        self.crs = None
        self._lock = threading.Lock()
        self._active = 0
        self.peak_concurrent = 0

    def read(self, window: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
        with self._lock:
            self._active += 1
            self.peak_concurrent = max(self.peak_concurrent, self._active)
        time.sleep(0.002)  # amplify race window so concurrent reads are detectable
        (row_start, row_stop), (col_start, col_stop) = window
        crop = self._image[row_start:row_stop, col_start:col_stop, :]
        result = np.transpose(crop, (2, 0, 1))
        with self._lock:
            self._active -= 1
        return result


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


class TestInferenceSlicerGeoTIFF:
    def test_windowed_raster_matches_in_memory_array(self) -> None:
        """Raster and array paths produce identical merged detections."""
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

    def test_windowed_raster_reads_correct_window_content(self) -> None:
        """The windowed read returns the same pixels crop_image would for each tile."""
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

    def test_windowed_raster_matches_in_memory_array_with_overlap(self) -> None:
        """Overlapping tiles read identical windows for both array and raster inputs."""
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

    def test_windowed_raster_preserves_band_dtype(self) -> None:
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

    @pytest.mark.parametrize(
        "crs",
        [
            pytest.param(None, id="no-crs"),
            pytest.param(_FakeCRS(is_projected=True), id="projected-crs"),
        ],
    )
    def test_crs_allows_slicing(self, crs: object | None) -> None:
        """None CRS and projected CRS both allow slicing without error."""
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        dataset = _FakeRasterDataset(image, crs=crs)
        slicer = InferenceSlicer(
            callback=_fixed_detection_callback,
            slice_wh=64,
            overlap_wh=0,
        )

        detections = slicer(dataset)

        assert len(detections) == 4

    def test_geographic_crs_raises(self) -> None:
        """Dataset with a geographic (non-projected) CRS raises ValueError."""
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        dataset = _FakeRasterDataset(image, crs=_FakeCRS(is_projected=False))
        slicer = InferenceSlicer(
            callback=_fixed_detection_callback,
            slice_wh=64,
            overlap_wh=0,
        )

        with pytest.raises(ValueError, match="projected coordinate reference"):
            slicer(dataset)

    def test_single_band_raster_produces_hwc1_tiles(self) -> None:
        """Single-band raster tiles arrive at the callback as (H, W, 1) arrays."""
        image = np.zeros((128, 128, 1), dtype=np.uint8)
        dataset = _FakeRasterDataset(image)

        seen: list[np.ndarray] = []

        def callback(tile: np.ndarray) -> Detections:
            seen.append(tile)
            return Detections.empty()

        slicer = InferenceSlicer(callback=callback, slice_wh=64, overlap_wh=0)

        slicer(dataset)

        assert seen
        assert all(tile.ndim == 3 and tile.shape[2] == 1 for tile in seen)

    def test_raster_smaller_than_slice_produces_single_tile(self) -> None:
        """Raster smaller than slice_wh is processed as exactly one tile."""
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        dataset = _FakeRasterDataset(image)

        tile_count: list[int] = [0]

        def callback(tile: np.ndarray) -> Detections:
            tile_count[0] += 1
            return Detections.empty()

        slicer = InferenceSlicer(callback=callback, slice_wh=128, overlap_wh=0)

        slicer(dataset)

        assert tile_count[0] == 1

    def test_compact_masks_with_windowed_raster(self) -> None:
        """compact_masks=True correctly moves and compresses masks from raster tiles."""
        rng = np.random.default_rng(17)
        image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
        dataset = _FakeRasterDataset(image)

        def masked_callback(tile: np.ndarray) -> Detections:
            h, w = tile.shape[:2]
            mask = np.zeros((1, h, w), dtype=bool)
            mask[0, : h // 2, : w // 2] = True
            return Detections(
                xyxy=np.array([[0, 0, w // 2, h // 2]], dtype=float),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
                mask=mask,
            )

        slicer = InferenceSlicer(
            callback=masked_callback,
            slice_wh=64,
            overlap_wh=0,
            compact_masks=True,
        )

        detections = slicer(dataset)

        assert len(detections) > 0

    def test_thread_workers_with_raster_serializes_reads(self) -> None:
        """Raster reads are serialized even when thread_workers > 1."""
        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        dataset = _ConcurrencyCheckDataset(image)

        slicer = InferenceSlicer(
            callback=_fixed_detection_callback,
            slice_wh=64,
            overlap_wh=0,
            thread_workers=4,
        )

        slicer(dataset)

        assert dataset.peak_concurrent == 1

    def test_real_rasterio_memoryfile_integration(self) -> None:
        """Real rasterio MemoryFile produces same detections as the array path."""
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
