from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

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


def move_detections(
    detections: Detections,
    offset: npt.NDArray[Any],
    resolution_wh: tuple[int, int] | None = None,
) -> Detections:
    """
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
        callback: Inference function that takes a sliced image and returns a
            `Detections` object.
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
            inference backends. Note: the first slice always runs synchronously
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

    Raises:
        ValueError: If `slice_wh`, `overlap_wh`, or `thread_workers` are
            invalid or inconsistent.

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
    """

    def __init__(
        self,
        callback: Callable[[ImageType], Detections],
        slice_wh: int | tuple[int, int] = 640,
        overlap_wh: int | tuple[int, int] = 100,
        overlap_filter: OverlapFilter | str = OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold: float = 0.5,
        overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
        thread_workers: int = 1,
        compact_masks: bool = False,
    ):
        slice_wh_norm = self._normalize_slice_wh(slice_wh)
        overlap_wh_norm = self._normalize_overlap_wh(overlap_wh)

        self._validate_overlap(slice_wh=slice_wh_norm, overlap_wh=overlap_wh_norm)

        if thread_workers < 1:
            raise ValueError(
                "`thread_workers` must be a positive integer. "
                f"Received: {thread_workers}"
            )

        self.slice_wh = slice_wh_norm
        self.overlap_wh = overlap_wh_norm
        self.iou_threshold = iou_threshold
        self.overlap_metric = OverlapMetric.from_value(overlap_metric)
        self.overlap_filter = OverlapFilter.from_value(overlap_filter)
        self.callback: Callable[[ImageType], Detections] = callback
        self.thread_workers = thread_workers
        self.compact_masks = compact_masks
        self._out_of_slice_bounds_warned: bool = False
        self._out_of_slice_bounds_lock = threading.Lock()
        self._obb_thread_workers_warned: bool = False
        self._obb_thread_workers_lock = threading.Lock()

    def __call__(self, image: ImageType) -> Detections:
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
            image: The full image to run inference on.

        Returns:
            Merged detections across all slices.
        """
        detections_list: list[Detections] = []
        resolution_wh = get_image_resolution_wh(image)

        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_wh=self.overlap_wh,
        )

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

    def _run_callback(self, image: ImageType, offset: npt.NDArray[Any]) -> Detections:
        """
        Run detection callback on a sliced portion of the image and adjust coordinates.

        Args:
            image: The full image.
            offset: Coordinates `(x_min, y_min, x_max, y_max)` defining
                the slice region.

        Returns:
            Detections adjusted to the full image coordinate system.
        """
        image_slice = crop_image(image=image, xyxy=offset)
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

        resolution_wh = get_image_resolution_wh(image)
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
