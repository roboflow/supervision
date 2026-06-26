from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import move_boxes, move_oriented_boxes
from supervision.detection.utils.iou_and_nms import OverlapFilter, OverlapMetric
from supervision.detection.utils.masks import move_masks
from supervision.draw.base import ImageType
from supervision.utils.image import crop_image, get_image_resolution_wh
from supervision.utils.internal import SupervisionWarnings
from supervision.utils.iterables import create_batches


@runtime_checkable
class WindowedRasterDataset(Protocol):
    """Structural type for a rasterio-style dataset read window-by-window.

    Matched structurally by `_is_windowed_raster` rather than by import so
    `rasterio` stays an optional dependency — any object exposing these members
    works. `rasterio.io.DatasetReader` satisfies this protocol.
    """

    width: int
    height: int
    crs: object | None

    def read(self, window: Any) -> npt.NDArray[Any]: ...


def _is_windowed_raster(image: object) -> TypeGuard[WindowedRasterDataset]:
    """Duck-type check for a rasterio-style dataset that supports windowed reads.

    Avoids importing rasterio so it remains an optional dependency. numpy arrays
    and PIL images do not expose this combination of attributes.
    """
    return (
        callable(getattr(image, "read", None))
        and hasattr(image, "crs")
        and hasattr(image, "width")
        and hasattr(image, "height")
    )


def move_detections(
    detections: Detections,
    offset: npt.NDArray[Any],
    resolution_wh: tuple[int, int] | None = None,
) -> Detections:
    """Translate detections by a pixel offset, repositioning boxes and masks.

    Args:
        detections: Detections object to be moved.
        offset: An array of shape `(2,)` containing offset values in the
            format `[dx, dy]`.
        resolution_wh: The width and height of the desired mask
            resolution. Required for segmentation detections.

    Returns:
        Repositioned Detections object.
    """
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
    if ORIENTED_BOX_COORDINATES in detections.data:
        detections.data[ORIENTED_BOX_COORDINATES] = move_oriented_boxes(
            xyxyxyxy=detections.data[ORIENTED_BOX_COORDINATES], offset=offset
        )
    if detections.mask is not None:
        if resolution_wh is None:
            raise ValueError(
                "Resolution width and height are required for moving segmentation "
                "detections. This should be the same as (width, height) of image shape."
            )
        if isinstance(detections.mask, CompactMask):
            # Preserve move_masks clipping semantics without dense materialisation.
            detections.mask = detections.mask.with_offset(
                dx=int(offset[0]),
                dy=int(offset[1]),
                new_image_shape=(resolution_wh[1], resolution_wh[0]),
            )
        else:
            detections.mask = move_masks(
                masks=detections.mask, offset=offset, resolution_wh=resolution_wh
            )
    return detections


class InferenceSlicer:
    """
    Perform tiled inference on large images by slicing them into overlapping patches.

    This class divides an input image into overlapping slices of configurable size
    and overlap, runs inference on each slice through a user-provided callback, and
    merges the resulting detections. The slicing process allows efficient processing
    of large images with limited resources while preserving detection accuracy via
    configurable overlap and post-processing of overlaps. Uses multi-threading for
    parallel slice inference.

    Args:
        callback: Inference function called for each slice (or batch of slices).

            When ``batch_size=1`` (default) the function receives a single
            ``np.ndarray`` and must return a single
            :class:`~supervision.detection.core.Detections` — the original
            single-image contract, fully backward-compatible.

            When ``batch_size > 1`` the function receives
            ``list[np.ndarray]`` (one array per slice) and must return
            ``list[Detections]`` of the **same length** in the same order.
            A length mismatch or non-list return raises :exc:`ValueError`.

            The two signatures are **not interchangeable**: a callback written
            for ``batch_size=1`` will fail when ``batch_size > 1``, and vice
            versa.
        slice_wh: Size of each slice `(width, height)`. If int, both width and
            height are set to this value.
        overlap_wh: Overlap size `(width, height)` between slices. If int, both
            width and height are set to this value.
        overlap_filter: Strategy to merge overlapping detections
            (`NON_MAX_SUPPRESSION`, `NON_MAX_MERGE`, or `NONE`).
        iou_threshold: IOU threshold used in merging overlap filtering.
        overlap_metric: Metric to compute overlap (`IOU` or `IOS`).
        thread_workers: Number of threads for concurrent slice inference.
            Must be a positive integer. When the first slice returns oriented
            bounding boxes (OBB), Supervision probes additional slices until a
            non-empty result is found, then falls back to sequential processing
            for all remaining slices to avoid thread-safety issues in common OBB
            inference backends. When passing a rasterio-style dataset, tile reads
            are serialized via an internal lock regardless of this setting —
            model inference runs in parallel, but ``raster.read()`` is protected.
            Note: the first slice always runs synchronously
            regardless of this setting, so for grids with few slices
            (e.g. two-slice images) effective parallelism is reduced.
        compact_masks: If ``True``, dense ``(N, H, W)`` boolean mask
            arrays returned by the callback are immediately converted to a
            :class:`~supervision.detection.compact_mask.CompactMask`. This
            keeps masks in run-length-encoded form for the entire pipeline —
            merge, NMS, and annotation — avoiding the large ``(N, H, W)``
            allocations that cause OOM on high-resolution images with many
            objects. IoU and NMS are computed directly on the RLE crops
            without ever materialising a full ``(N, H, W)`` array.
            Defaults to ``False`` for backward compatibility.
        batch_size: Number of slices passed to the callback per call.
            Defaults to ``1``, which uses the single-image callback contract
            (``np.ndarray`` → :class:`~supervision.detection.core.Detections`).
            Set to ``> 1`` to enable the batch callback contract
            (``list[np.ndarray]`` → ``list[Detections]``).

            For GPU-backed models, prefer ``batch_size > 1`` with
            ``thread_workers=1``. A single batched forward pass is faster than
            concurrent single-image calls that compete for the same CUDA device,
            and avoids multiplying peak VRAM by ``thread_workers * batch_size``.
            Must be a positive integer.

    Raises:
        ValueError: If ``slice_wh``, ``overlap_wh``, or ``thread_workers`` are
            invalid or inconsistent.
        ValueError: If ``batch_size < 1``.
        ValueError: If the callback returns a non-list when ``batch_size > 1``.
        ValueError: If the callback returns a list whose length differs from the
            number of slices passed when ``batch_size > 1``.

    Example:
        ```python
        import cv2
        import supervision as sv
        from rfdetr import RFDETRMedium

        model = RFDETRMedium()

        def callback(tile):
            return model.predict(tile)

        slicer = sv.InferenceSlicer(callback, slice_wh=640, overlap_wh=100)

        image = cv2.imread("example.png")
        detections = slicer(image)
        ```

        ```python
        import supervision as sv
        from PIL import Image
        from ultralytics import YOLO

        model = YOLO("yolo11m.pt")

        def callback(tile):
            results = model(tile)[0]
            return sv.Detections.from_ultralytics(results)

        slicer = sv.InferenceSlicer(callback, slice_wh=640, overlap_wh=100)

        image = Image.open("example.png")
        detections = slicer(image)
        ```

        ```python
        import rasterio
        import supervision as sv

        def callback(tile):  # tile is (H, W, C); select/convert bands as needed
            ...

        slicer = sv.InferenceSlicer(callback, slice_wh=640, overlap_wh=100)

        with rasterio.open("large_orthomosaic.tif") as dataset:
            detections = slicer(dataset)
        ```

        Passing an open rasterio dataset reads each tile lazily via a windowed
        read, so multi-GB GeoTIFFs never need to be loaded into memory at once.
        `rasterio` is an optional dependency installable via
        `pip install "supervision[geotiff]"`.

        Batch inference — pass multiple slices per callback call for GPU models
        that benefit from batched forward passes:

        ```python
        import cv2
        import numpy as np
        import supervision as sv

        # Batch callback: receives list[np.ndarray], returns list[Detections]
        def batch_callback(tiles: list[np.ndarray]) -> list[sv.Detections]:
            # Run your model on the batch; return one Detections per tile.
            return [sv.Detections.empty() for _ in tiles]

        slicer = sv.InferenceSlicer(
            callback=batch_callback,
            slice_wh=640,
            overlap_wh=100,
            batch_size=8,
            thread_workers=1,  # recommended for GPU: batch not threads
        )

        image = cv2.imread("example.png")
        detections = slicer(image)
        ```
    """

    def __init__(
        self,
        callback: (
            Callable[[ImageType], Detections]
            | Callable[[list[npt.NDArray[Any]]], list[Detections]]
        ),
        slice_wh: int | tuple[int, int] = 640,
        overlap_wh: int | tuple[int, int] = 100,
        overlap_filter: OverlapFilter | str = OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold: float = 0.5,
        overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
        thread_workers: int = 1,
        compact_masks: bool = False,
        batch_size: int = 1,
    ):
        slice_wh_norm = self._normalize_slice_wh(slice_wh)
        overlap_wh_norm = self._normalize_overlap_wh(overlap_wh)

        self._validate_overlap(slice_wh=slice_wh_norm, overlap_wh=overlap_wh_norm)

        if thread_workers < 1:
            raise ValueError(
                "`thread_workers` must be a positive integer. "
                f"Received: {thread_workers}"
            )
        if batch_size < 1:
            raise ValueError(
                f"`batch_size` must be a positive integer. Received: {batch_size}"
            )

        self.slice_wh = slice_wh_norm
        self.overlap_wh = overlap_wh_norm
        self.iou_threshold = iou_threshold
        self.overlap_metric = OverlapMetric.from_value(overlap_metric)
        self.overlap_filter = OverlapFilter.from_value(overlap_filter)
        # Stored as single-image type; batch path calls with list[ndarray] via
        # _run_callback_batch which suppresses the arg-type mismatch there.
        self.callback: Callable[[npt.NDArray[Any]], Detections] = callback  # type: ignore[assignment]
        self.thread_workers = thread_workers
        self.compact_masks = compact_masks
        self.batch_size = batch_size
        self._out_of_slice_bounds_warned: bool = False
        self._out_of_slice_bounds_lock = threading.Lock()
        self._obb_thread_workers_warned: bool = False
        self._obb_thread_workers_lock = threading.Lock()
        self._raster_read_lock = threading.Lock()

    def __call__(self, image: ImageType | WindowedRasterDataset) -> Detections:
        """
        Perform tiled inference on the full image and return merged detections.

        The first slice always runs synchronously so the output type can be
        inspected before committing to a threading strategy. Detections are
        merged in a deterministic order: the first slice is always at index 0,
        followed by any probe slices, then the remaining slices in source order.
        If oriented bounding boxes are detected, all remaining slices are
        processed sequentially and a ``SupervisionWarnings`` warning is emitted
        once per slicer instance.

        Args:
            image: The full image to run inference on. In addition to in-memory
                images (NumPy arrays or PIL images), this also accepts an open
                rasterio-style dataset. When a dataset is provided, each tile is
                read lazily via a windowed read instead of loading the whole image
                into memory, enabling tiled inference on multi-GB GeoTIFFs. Tiles
                read from a dataset preserve the source dtype (e.g. ``uint16`` for
                16-bit sensors) and keep every band; convert or select bands to
                the dtype/channels your model expects inside the callback.

        Returns:
            Merged detections across all slices.

        Raises:
            ValueError: If ``image`` is a rasterio-style dataset whose CRS is
                geographic (non-projected). Reproject to a projected CRS
                (e.g. with ``gdalwarp``) before calling.
        """
        detections_list: list[Detections] = []
        resolution_wh = self._get_resolution_wh(image)

        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_wh=self.overlap_wh,
        )

        if self.batch_size > 1:
            batched = list(create_batches(offsets, self.batch_size))
            # Run first batch synchronously: fail-fast type validation + OBB probe.
            first_batch_results = self._run_callback_batch(image, batched[0])
            detections_list.extend(first_batch_results)
            obb_detected = any(
                ORIENTED_BOX_COORDINATES in det.data for det in first_batch_results
            )
            if obb_detected and self.thread_workers > 1:
                with self._obb_thread_workers_lock:
                    if not self._obb_thread_workers_warned:
                        self._obb_thread_workers_warned = True
                        warnings.warn(
                            "InferenceSlicer detected oriented bounding boxes while "
                            "`thread_workers > 1`. Remaining batches will be processed "
                            "sequentially because many OBB inference backends are not "
                            "thread-safe and can crash when shared across threads.",
                            category=SupervisionWarnings,
                            stacklevel=2,
                        )
            remaining_batches = batched[1:]
            if self.thread_workers == 1 or obb_detected:
                for offset_batch in remaining_batches:
                    detections_list.extend(
                        self._run_callback_batch(image, offset_batch)
                    )
            else:
                with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
                    batch_futures = [
                        executor.submit(self._run_callback_batch, image, ob)
                        for ob in remaining_batches
                    ]
                    for batch_future in as_completed(batch_futures):
                        detections_list.extend(batch_future.result())
            merged = Detections.merge(detections_list=detections_list)
            return self._apply_overlap_filter(merged)

        first_offset = offsets[0]
        first_detections = self._run_callback(image, first_offset)
        detections_list.append(first_detections)

        remaining_offsets = offsets[1:]
        obb_detected = ORIENTED_BOX_COORDINATES in first_detections.data
        should_run_sequentially = self.thread_workers <= 1 or obb_detected

        probe_index = 0
        if not should_run_sequentially and len(first_detections) == 0:
            while probe_index < len(remaining_offsets):
                probe_offset = remaining_offsets[probe_index]
                probe_detections = self._run_callback(image, probe_offset)
                detections_list.append(probe_detections)
                probe_index += 1

                if ORIENTED_BOX_COORDINATES in probe_detections.data:
                    obb_detected = True
                    should_run_sequentially = True
                    break

                if len(probe_detections) > 0:
                    break

        remaining_offsets = remaining_offsets[probe_index:]

        if should_run_sequentially:
            if self.thread_workers > 1 and obb_detected:
                with self._obb_thread_workers_lock:
                    if not self._obb_thread_workers_warned:
                        self._obb_thread_workers_warned = True
                        warnings.warn(
                            "InferenceSlicer detected oriented bounding boxes while "
                            "`thread_workers > 1`. Remaining slices will be processed "
                            "sequentially because many OBB inference backends are not "
                            "thread-safe and can crash when shared across threads.",
                            category=SupervisionWarnings,
                            stacklevel=2,
                        )
            for offset in remaining_offsets:
                detections_list.append(self._run_callback(image, offset))
        else:
            with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
                futures = [
                    executor.submit(self._run_callback, image, offset)
                    for offset in remaining_offsets
                ]
                for future in as_completed(futures):
                    detections_list.append(future.result())

        merged = Detections.merge(detections_list=detections_list)
        return self._apply_overlap_filter(merged)

    def _get_resolution_wh(
        self, image: ImageType | WindowedRasterDataset
    ) -> tuple[int, int]:
        """Return ``(width, height)`` for the image, validating CRS for rasters."""
        if _is_windowed_raster(image):
            crs = image.crs
            if crs is not None and not getattr(crs, "is_projected", True):
                raise ValueError(
                    "InferenceSlicer requires a projected coordinate reference "
                    "system for pixel-space tiled inference on a raster dataset. "
                    f"The provided dataset uses a geographic CRS ({crs}). Reproject "
                    "it to a projected CRS (e.g. with `gdalwarp`) before slicing."
                )
            return (image.width, image.height)
        return get_image_resolution_wh(image)

    def _apply_overlap_filter(self, merged: Detections) -> Detections:
        """Apply the configured overlap filter strategy to merged detections."""
        if self.overlap_filter == OverlapFilter.NONE:
            return merged
        if self.overlap_filter == OverlapFilter.NON_MAX_SUPPRESSION:
            return merged.with_nms(
                threshold=self.iou_threshold,
                overlap_metric=self.overlap_metric,
            )
        if self.overlap_filter == OverlapFilter.NON_MAX_MERGE:
            return merged.with_nmm(
                threshold=self.iou_threshold,
                overlap_metric=self.overlap_metric,
            )
        warnings.warn(
            f"Invalid overlap filter strategy: {self.overlap_filter}",
            category=SupervisionWarnings,
        )
        return merged

    def _run_callback(
        self, image: ImageType | WindowedRasterDataset, offset: npt.NDArray[Any]
    ) -> Detections:
        """
        Run detection callback on a sliced portion of the image and adjust coordinates.

        Args:
            image: The full image.
            offset: Coordinates `(x_min, y_min, x_max, y_max)` defining
                the slice region.

        Returns:
            Detections adjusted to the full image coordinate system.
        """
        if _is_windowed_raster(image):
            x_min, y_min, x_max, y_max = (int(v) for v in offset)
            # rasterio tuple window: ((row_start, row_stop), (col_start, col_stop))
            window = ((y_min, y_max), (x_min, x_max))
            with self._raster_read_lock:
                bands = image.read(window=window)  # shape (channels, height, width)
            image_slice = np.ascontiguousarray(
                np.transpose(bands, (1, 2, 0))
            )  # -> (H, W, C)
            resolution_wh = (image.width, image.height)
        else:
            image_slice = crop_image(image=image, xyxy=offset)
            resolution_wh = get_image_resolution_wh(image)

        detections = self.callback(image_slice)

        if (
            self.compact_masks
            and detections.mask is not None
            and isinstance(detections.mask, np.ndarray)
        ):
            slice_w, slice_h = get_image_resolution_wh(image_slice)
            detections.mask = CompactMask.from_dense(
                detections.mask,
                detections.xyxy,
                image_shape=(slice_h, slice_w),
            )

        # Fast-path: skip locking and bounds checking when the warning has already
        # been emitted or when there are no detections to inspect.
        needs_warning_check = (
            not self._out_of_slice_bounds_warned and len(detections) > 0
        )

        if needs_warning_check:
            with self._out_of_slice_bounds_lock:
                # Re-check under the lock to ensure correctness with multiple threads.
                if not self._out_of_slice_bounds_warned and len(detections) > 0:
                    slice_width = offset[2] - offset[0]
                    slice_height = offset[3] - offset[1]
                    x_exceeds = np.any(detections.xyxy[:, [0, 2]] > slice_width)
                    y_exceeds = np.any(detections.xyxy[:, [1, 3]] > slice_height)
                    x_negative = np.any(detections.xyxy[:, [0, 2]] < 0)
                    y_negative = np.any(detections.xyxy[:, [1, 3]] < 0)
                    if x_exceeds or y_exceeds or x_negative or y_negative:
                        self._out_of_slice_bounds_warned = True
                        msg = (
                            "Detections returned by the callback have coordinates "
                            "outside the slice bounds. This may be caused by the "
                            "callback running inference on the full image instead of "
                            "the provided image slice. Ensure your callback uses the "
                            "input slice for inference, not the original "
                            "full-resolution image."
                        )
                        warnings.warn(msg, category=SupervisionWarnings, stacklevel=2)
        detections = move_detections(
            detections=detections,
            offset=offset[:2],
            resolution_wh=resolution_wh,
        )
        return detections

    def _run_callback_batch(
        self,
        image: ImageType | WindowedRasterDataset,
        offsets: list[npt.NDArray[Any]],
    ) -> list[Detections]:
        """Run batch inference callback on multiple slices.

        Args:
            image: The full image or rasterio dataset.
            offsets: List of slice coordinates `(x_min, y_min, x_max, y_max)`.

        Returns:
            Detections adjusted to full-image coordinates, one per offset.
        """
        if _is_windowed_raster(image):
            slices = []
            for offset in offsets:
                x_min, y_min, x_max, y_max = (int(v) for v in offset)
                window = ((y_min, y_max), (x_min, x_max))
                with self._raster_read_lock:
                    bands = image.read(window=window)
                slices.append(np.ascontiguousarray(np.transpose(bands, (1, 2, 0))))
            resolution_wh = (image.width, image.height)
        else:
            slices = [crop_image(image=image, xyxy=offset) for offset in offsets]
            resolution_wh = get_image_resolution_wh(image)

        detections_in_slices = self.callback(slices)
        if not isinstance(detections_in_slices, list):
            raise ValueError(
                "Callback must return `list[Detections]` when `batch_size > 1`. "
                f"Got: {type(detections_in_slices)}"
            )
        if len(detections_in_slices) != len(offsets):
            raise ValueError(
                f"Callback returned {len(detections_in_slices)} Detections "
                f"for {len(offsets)} slices. Lengths must match."
            )

        if self.compact_masks:
            for det, image_slice in zip(detections_in_slices, slices):
                if det.mask is not None and isinstance(det.mask, np.ndarray):
                    slice_w, slice_h = get_image_resolution_wh(image_slice)
                    det.mask = CompactMask.from_dense(
                        det.mask,
                        det.xyxy,
                        image_shape=(slice_h, slice_w),
                    )

        if not self._out_of_slice_bounds_warned:
            for det, offset in zip(detections_in_slices, offsets):
                if self._out_of_slice_bounds_warned or len(det) == 0:
                    continue
                with self._out_of_slice_bounds_lock:
                    if not self._out_of_slice_bounds_warned and len(det) > 0:
                        slice_width = offset[2] - offset[0]
                        slice_height = offset[3] - offset[1]
                        x_exceeds = np.any(det.xyxy[:, [0, 2]] > slice_width)
                        y_exceeds = np.any(det.xyxy[:, [1, 3]] > slice_height)
                        x_negative = np.any(det.xyxy[:, [0, 2]] < 0)
                        y_negative = np.any(det.xyxy[:, [1, 3]] < 0)
                        if x_exceeds or y_exceeds or x_negative or y_negative:
                            self._out_of_slice_bounds_warned = True
                            warnings.warn(
                                "Detections returned by the callback have coordinates "
                                "outside the slice bounds. This may be caused by the "
                                "callback running inference on the full image instead "
                                "of the provided image slice. Ensure your callback "
                                "uses the input slice for inference, not the original "
                                "full-resolution image.",
                                category=SupervisionWarnings,
                                stacklevel=2,
                            )

        return [
            move_detections(
                detections=det, offset=offset[:2], resolution_wh=resolution_wh
            )
            for det, offset in zip(detections_in_slices, offsets)
        ]

    @staticmethod
    def _normalize_slice_wh(
        slice_wh: int | tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(slice_wh, int):
            if slice_wh <= 0:
                raise ValueError(
                    f"`slice_wh` must be a positive integer. Received: {slice_wh}"
                )
            return slice_wh, slice_wh

        if isinstance(slice_wh, tuple) and len(slice_wh) == 2:
            width, height = slice_wh
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"`slice_wh` values must be positive. Received: {slice_wh}"
                )
            return width, height

        raise ValueError(
            "`slice_wh` must be an int or a tuple of two positive integers "
            "(slice_w, slice_h). "
            f"Received: {slice_wh}"
        )

    @staticmethod
    def _normalize_overlap_wh(
        overlap_wh: int | tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(overlap_wh, int):
            if overlap_wh < 0:
                raise ValueError(
                    "`overlap_wh` must be a non negative integer. "
                    f"Received: {overlap_wh}"
                )
            return overlap_wh, overlap_wh

        if isinstance(overlap_wh, tuple) and len(overlap_wh) == 2:
            overlap_w, overlap_h = overlap_wh
            if overlap_w < 0 or overlap_h < 0:
                raise ValueError(
                    f"`overlap_wh` values must be non negative. Received: {overlap_wh}"
                )
            return overlap_w, overlap_h

        raise ValueError(
            "`overlap_wh` must be an int or a tuple of two non negative integers "
            "(overlap_w, overlap_h). "
            f"Received: {overlap_wh}"
        )

    @staticmethod
    def _generate_offset(
        resolution_wh: tuple[int, int],
        slice_wh: tuple[int, int],
        overlap_wh: tuple[int, int],
    ) -> npt.NDArray[Any]:
        """
        Generate bounding boxes defining the coordinates of image slices with overlap.

        Args:
            resolution_wh: Image resolution `(width, height)`.
            slice_wh: Size of each slice `(width, height)`.
            overlap_wh: Overlap size between slices `(width, height)`.

        Returns:
            Array of shape `(num_slices, 4)` with each row as
                `(x_min, y_min, x_max, y_max)` coordinates for a slice.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_width, overlap_height = overlap_wh

        stride_x = slice_width - overlap_width
        stride_y = slice_height - overlap_height

        def _compute_axis_starts(
            image_size: int,
            slice_size: int,
            stride: int,
        ) -> list[int]:
            if image_size <= slice_size:
                return [0]

            if stride == slice_size:
                return list(np.arange(0, image_size, stride).tolist())

            last_start = image_size - slice_size
            starts: list[int] = list(np.arange(0, last_start, stride).tolist())
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            return starts

        x_starts = _compute_axis_starts(
            image_size=image_width,
            slice_size=slice_width,
            stride=stride_x,
        )
        y_starts = _compute_axis_starts(
            image_size=image_height,
            slice_size=slice_height,
            stride=stride_y,
        )

        x_min, y_min = np.meshgrid(x_starts, y_starts)
        x_max = np.clip(x_min + slice_width, 0, image_width)
        y_max = np.clip(y_min + slice_height, 0, image_height)

        offsets: npt.NDArray[Any] = np.stack(
            [x_min, y_min, x_max, y_max],
            axis=-1,
        ).reshape(-1, 4)

        return offsets

    @staticmethod
    def _validate_overlap(
        slice_wh: tuple[int, int],
        overlap_wh: tuple[int, int],
    ) -> None:
        overlap_w, overlap_h = overlap_wh
        slice_w, slice_h = slice_wh

        if overlap_w < 0 or overlap_h < 0:
            raise ValueError(
                "Overlap values must be greater than or equal to 0. "
                f"Received: {overlap_wh}"
            )

        if overlap_w >= slice_w or overlap_h >= slice_h:
            raise ValueError(
                "`overlap_wh` must be smaller than `slice_wh` in both dimensions "
                f"to keep a positive stride. Received overlap_wh={overlap_wh}, "
                f"slice_wh={slice_wh}."
            )
