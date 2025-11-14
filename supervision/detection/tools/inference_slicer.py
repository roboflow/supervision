from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import move_boxes, move_oriented_boxes
from supervision.detection.utils.iou_and_nms import OverlapFilter, OverlapMetric
from supervision.detection.utils.masks import move_masks
from supervision.utils.image import crop_image
from supervision.utils.internal import (
    SupervisionWarnings
)


def move_detections(
    detections: Detections,
    offset: np.ndarray,
    resolution_wh: tuple[int, int] | None = None,
) -> Detections:
    """
    Args:
        detections (sv.Detections): Detections object to be moved.
        offset (np.ndarray): An array of shape `(2,)` containing offset values in format
            is `[dx, dy]`.
        resolution_wh (Tuple[int, int]): The width and height of the desired mask
            resolution. Required for segmentation detections.

    Returns:
        (sv.Detections) repositioned Detections object.
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
        detections.mask = move_masks(
            masks=detections.mask, offset=offset, resolution_wh=resolution_wh
        )
    return detections


class InferenceSlicer:
    """
    InferenceSlicer performs slicing-based inference for small target detection. This
    method, often referred to as
    [Slicing Adaptive Inference (SAHI)](https://ieeexplore.ieee.org/document/9897990),
    involves dividing a larger image into smaller slices, performing inference on each
    slice, and then merging the detections.

    Args:
        slice_wh (Tuple[int, int]): Dimensions of each slice measured in pixels. The
            tuple should be in the format `(width, height)`.
        overlap_wh (Tuple[int, int]): A tuple representing the desired
            overlap for width and height between consecutive slices measured in pixels.
            Each value must be greater than or equal to 0.
        overlap_filter (Union[OverlapFilter, str]): Strategy for
            filtering or merging overlapping detections in slices.
        iou_threshold (float): Intersection over Union (IoU) threshold
            used when filtering by overlap.
        overlap_metric (Union[OverlapMetric, str]): Metric used for matching detections
            in slices.
        callback (Callable): A function that performs inference on a given image
            slice and returns detections.
        thread_workers (int): Number of threads for parallel execution.

    Note:
        The class ensures that slices do not exceed the boundaries of the original
        image. As a result, the final slices in the row and column dimensions might be
        smaller than the specified slice dimensions if the image's width or height is
        not a multiple of the slice's width or height minus the overlap.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], Detections],
        slice_wh: tuple[int, int] = (640, 640),
        overlap_wh: tuple[int, int] = (100, 100),
        overlap_filter: OverlapFilter | str = OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold: float = 0.5,
        overlap_metric: OverlapMetric | str = OverlapMetric.IOU,
        thread_workers: int = 1,
    ):
        self.overlap_wh = overlap_wh
        self.slice_wh = slice_wh
        self._validate_overlap(slice_wh=self.slice_wh, overlap_wh=overlap_wh)
        self.iou_threshold = iou_threshold
        self.overlap_metric = OverlapMetric.from_value(overlap_metric)
        self.overlap_filter = OverlapFilter.from_value(overlap_filter)
        self.callback = callback
        self.thread_workers = thread_workers

    def __call__(self, image: np.ndarray) -> Detections:
        """
        Performs slicing-based inference on the provided image using the specified
            callback.

        Args:
            image (np.ndarray): The input image on which inference needs to be
                performed. The image should be in the format
                `(height, width, channels)`.

        Returns:
            Detections: A collection of detections for the entire image after merging
                results from all slices and applying NMS.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(SOURCE_IMAGE_PATH)
            model = YOLO(...)

            def callback(image_slice: np.ndarray) -> sv.Detections:
                result = model(image_slice)[0]
                return sv.Detections.from_ultralytics(result)

            slicer = sv.InferenceSlicer(
                callback=callback,
                overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            )

            detections = slicer(image)
            ```
        """
        detections_list = []
        resolution_wh = (image.shape[1], image.shape[0])
        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_wh=self.overlap_wh,
        )

        with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
            futures = [
                executor.submit(self._run_callback, image, offset) for offset in offsets
            ]
            for future in as_completed(futures):
                detections_list.append(future.result())

        merged = Detections.merge(detections_list=detections_list)
        if self.overlap_filter == OverlapFilter.NONE:
            return merged
        elif self.overlap_filter == OverlapFilter.NON_MAX_SUPPRESSION:
            return merged.with_nms(
                threshold=self.iou_threshold, overlap_metric=self.overlap_metric
            )
        elif self.overlap_filter == OverlapFilter.NON_MAX_MERGE:
            return merged.with_nmm(
                threshold=self.iou_threshold, overlap_metric=self.overlap_metric
            )
        else:
            warnings.warn(
                f"Invalid overlap filter strategy: {self.overlap_filter}",
                category=SupervisionWarnings,
            )
            return merged

    def _run_callback(self, image, offset) -> Detections:
        """
        Run the provided callback on a slice of an image.

        Args:
            image (np.ndarray): The input image on which inference needs to run
            offset (np.ndarray): An array of shape `(4,)` containing coordinates
                for the slice.

        Returns:
            Detections: A collection of detections for the slice.
        """
        image_slice = crop_image(image=image, xyxy=offset)
        detections = self.callback(image_slice)
        resolution_wh = (image.shape[1], image.shape[0])
        detections = move_detections(
            detections=detections, offset=offset[:2], resolution_wh=resolution_wh
        )

        return detections

    @staticmethod
    def _generate_offset(
        resolution_wh: tuple[int, int],
        slice_wh: tuple[int, int],
        overlap_wh: tuple[int, int],
    ) -> np.ndarray:
        """
        Generate offset coordinates for slicing an image based on the given resolution,
        slice dimensions, and pixel overlap.

        Args:
            resolution_wh (Tuple[int, int]): Width and height of the image to be sliced.
            slice_wh (Tuple[int, int]): Dimensions of each slice in pixels (width, height).
            overlap_wh (Tuple[int, int]): Overlap in pixels (overlap_w, overlap_h).

        Returns:
            np.ndarray: Array of shape (n, 4) with [x_min, y_min, x_max, y_max] slices.
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

            # No overlap case, preserve original behavior, no overlapping tiles
            if stride == slice_size:
                return np.arange(0, image_size, stride).tolist()

            # Overlap case, ensure last tile touches the border without redundancy
            last_start = image_size - slice_size
            starts = np.arange(0, last_start, stride).tolist()
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

        offsets = np.stack(
            [x_min, y_min, x_max, y_max],
            axis=-1,
        ).reshape(-1, 4)

        return offsets

    @staticmethod
    def _validate_overlap(
            slice_wh: tuple[int, int],
            overlap_wh: tuple[int, int],
    ) -> None:
        if not isinstance(overlap_wh, tuple) or len(overlap_wh) != 2:
            raise ValueError(
                "`overlap_wh` must be a tuple of two non-negative values "
                "(overlap_w, overlap_h)."
            )

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
