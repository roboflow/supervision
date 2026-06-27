"""Tests for windowed GeoTIFF reads in InferenceSlicer."""

from __future__ import annotations

import threading
import time
from typing import Callable

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


def _sortable(detections: Detections) -> np.ndarray:
    """Sort detection boxes so two runs can be compared order-independently."""
    return np.array(sorted(detections.xyxy.tolist()), dtype=float)


@pytest.fixture
def make_raster_dataset() -> Callable[..., _FakeRasterDataset]:
    """Factory: create a _FakeRasterDataset from a numpy image array."""

    def factory(image_hwc: np.ndarray, crs: object | None = None) -> _FakeRasterDataset:
        return _FakeRasterDataset(image_hwc, crs=crs)

    return factory


@pytest.fixture
def fixed_detection_callback() -> Callable[[np.ndarray], Detections]:
    """Return a constant single-box detection for every tile."""

    def callback(_: np.ndarray) -> Detections:
        return Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    return callback


@pytest.fixture
def make_recording_callback() -> Callable[[list[np.ndarray]], Callable]:
    """Factory: given a sink list, return a callback that records each tile."""

    def factory(sink: list[np.ndarray]) -> Callable[[np.ndarray], Detections]:
        def callback(tile: np.ndarray) -> Detections:
            sink.append(tile.copy())
            return Detections.empty()

        return callback

    return factory


class TestInferenceSlicerGeoTIFF:
    def test_windowed_raster_matches_in_memory_array(
        self,
        make_raster_dataset: Callable,
        fixed_detection_callback: Callable,
    ) -> None:
        """Raster and array paths produce identical merged detections."""
        # Arrange
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        dataset = make_raster_dataset(image, crs=_FakeCRS(is_projected=True))
        slicer = InferenceSlicer(
            callback=fixed_detection_callback,
            slice_wh=128,
            overlap_wh=0,
        )

        # Act
        detections_array = slicer(image)
        detections_raster = slicer(dataset)

        # Assert
        assert np.array_equal(_sortable(detections_array), _sortable(detections_raster))

    @pytest.mark.parametrize(
        ("seed", "image_shape", "slice_wh", "overlap_wh"),
        [
            pytest.param(7, (128, 192, 3), 64, 0, id="no-overlap"),
            pytest.param(99, (200, 220, 3), 96, 32, id="with-overlap"),
        ],
    )
    def test_raster_tiles_match_array_tiles(
        self,
        make_raster_dataset: Callable,
        make_recording_callback: Callable,
        seed: int,
        image_shape: tuple[int, int, int],
        slice_wh: int,
        overlap_wh: int,
    ) -> None:
        """Windowed raster read returns identical pixel tiles as array path."""
        # Arrange
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 255, size=image_shape, dtype=np.uint8)
        dataset = make_raster_dataset(image)

        array_tiles: list[np.ndarray] = []
        raster_tiles: list[np.ndarray] = []

        slicer_array = InferenceSlicer(
            callback=make_recording_callback(array_tiles),
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
        )
        slicer_raster = InferenceSlicer(
            callback=make_recording_callback(raster_tiles),
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
        )

        # Act
        slicer_array(image)
        slicer_raster(dataset)

        # Assert
        assert len(array_tiles) == len(raster_tiles) > 0
        for array_tile, raster_tile in zip(array_tiles, raster_tiles):
            assert np.array_equal(array_tile, raster_tile)

    def test_windowed_raster_preserves_band_dtype(
        self, make_raster_dataset: Callable
    ) -> None:
        """Tiles read from a dataset keep the source dtype (e.g. uint16)."""
        # Arrange
        rng = np.random.default_rng(5)
        image = rng.integers(0, 4000, size=(128, 128, 3), dtype=np.uint16)
        dataset = make_raster_dataset(image)

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
    def test_crs_allows_slicing(
        self,
        make_raster_dataset: Callable,
        fixed_detection_callback: Callable,
        crs: object | None,
    ) -> None:
        """None CRS and projected CRS both allow slicing without error."""
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        dataset = make_raster_dataset(image, crs=crs)
        slicer = InferenceSlicer(
            callback=fixed_detection_callback,
            slice_wh=64,
            overlap_wh=0,
        )

        detections = slicer(dataset)

        assert len(detections) == 4

    def test_geographic_crs_raises(
        self,
        make_raster_dataset: Callable,
        fixed_detection_callback: Callable,
    ) -> None:
        """Dataset with a geographic (non-projected) CRS raises ValueError."""
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        dataset = make_raster_dataset(image, crs=_FakeCRS(is_projected=False))
        slicer = InferenceSlicer(
            callback=fixed_detection_callback,
            slice_wh=64,
            overlap_wh=0,
        )

        with pytest.raises(ValueError, match="projected coordinate reference"):
            slicer(dataset)

    def test_single_band_raster_produces_hwc1_tiles(
        self, make_raster_dataset: Callable
    ) -> None:
        """Single-band raster tiles arrive at the callback as (H, W, 1) arrays."""
        image = np.zeros((128, 128, 1), dtype=np.uint8)
        dataset = make_raster_dataset(image)

        seen: list[np.ndarray] = []

        def callback(tile: np.ndarray) -> Detections:
            seen.append(tile)
            return Detections.empty()

        slicer = InferenceSlicer(callback=callback, slice_wh=64, overlap_wh=0)

        slicer(dataset)

        assert seen
        assert all(tile.ndim == 3 and tile.shape[2] == 1 for tile in seen)

    def test_raster_smaller_than_slice_produces_single_tile(
        self, make_raster_dataset: Callable
    ) -> None:
        """Raster smaller than slice_wh is processed as exactly one tile."""
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        dataset = make_raster_dataset(image)

        tile_count: list[int] = [0]

        def callback(tile: np.ndarray) -> Detections:
            tile_count[0] += 1
            return Detections.empty()

        slicer = InferenceSlicer(callback=callback, slice_wh=128, overlap_wh=0)

        slicer(dataset)

        assert tile_count[0] == 1

    def test_compact_masks_with_windowed_raster(
        self, make_raster_dataset: Callable
    ) -> None:
        """compact_masks=True correctly moves and compresses masks from raster tiles."""
        rng = np.random.default_rng(17)
        image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
        dataset = make_raster_dataset(image)

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

    def test_thread_workers_with_raster_serializes_reads(
        self, fixed_detection_callback: Callable
    ) -> None:
        """Raster reads are serialized even when thread_workers > 1."""
        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        dataset = _ConcurrencyCheckDataset(image)

        slicer = InferenceSlicer(
            callback=fixed_detection_callback,
            slice_wh=64,
            overlap_wh=0,
            thread_workers=4,
        )

        slicer(dataset)

        assert dataset.peak_concurrent == 1

    def test_real_rasterio_memoryfile_integration(
        self, fixed_detection_callback: Callable
    ) -> None:
        """Real rasterio MemoryFile produces same detections as the array path."""
        pytest.importorskip("rasterio")
        from rasterio.io import MemoryFile

        # Arrange
        rng = np.random.default_rng(123)
        image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
        bands = np.transpose(image, (2, 0, 1))  # (C, H, W)

        slicer = InferenceSlicer(
            callback=fixed_detection_callback,
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
