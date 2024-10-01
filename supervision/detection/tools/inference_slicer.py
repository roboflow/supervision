import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Tuple, Union

import numpy as np

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.overlap_filter import OverlapFilter
from supervision.detection.utils import move_boxes, move_masks, move_oriented_boxes
from supervision.utils.image import crop_image
from supervision.utils.internal import (
    SupervisionWarnings,
    warn_deprecated,
)


def move_detections(
    detections: Detections,
    offset: np.ndarray,
    resolution_wh: Optional[Tuple[int, int]] = None,
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
        overlap_ratio_wh (Optional[Tuple[float, float]]): [⚠️ Deprecated: please set
                to `None` and use `overlap_wh`] A tuple representing the
            desired overlap ratio for width and height between consecutive slices.
            Each value should be in the range [0, 1), where 0 means no overlap and
            a value close to 1 means high overlap.
        overlap_wh (Optional[Tuple[int, int]]): A tuple representing the desired
            overlap for width and height between consecutive slices measured in pixels.
            Each value should be greater than or equal to 0. Takes precedence over
            `overlap_ratio_wh`.
        overlap_filter (Union[OverlapFilter, str]): Strategy for
            filtering or merging overlapping detections in slices.
        iou_threshold (float): Intersection over Union (IoU) threshold
            used when filtering by overlap.
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
        slice_wh: Tuple[int, int] = (320, 320),
        overlap_ratio_wh: Optional[Tuple[float, float]] = (0.2, 0.2),
        overlap_wh: Optional[Tuple[int, int]] = None,
        overlap_filter: Union[OverlapFilter, str] = OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold: float = 0.5,
        thread_workers: int = 1,
    ):
        if overlap_ratio_wh is not None:
            warn_deprecated(
                "`overlap_ratio_wh` in `InferenceSlicer.__init__` is deprecated and "
                "will be removed in `supervision-0.27.0`. Please manually set it to "
                "`None` and use `overlap_wh` instead."
            )

        self._validate_overlap(overlap_ratio_wh, overlap_wh)
        self.overlap_ratio_wh = overlap_ratio_wh
        self.overlap_wh = overlap_wh

        self.slice_wh = slice_wh
        self.iou_threshold = iou_threshold
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
                overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            )

            detections = slicer(image)
            ```
        """
        detections_list = []
        resolution_wh = (image.shape[1], image.shape[0])
        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_ratio_wh=self.overlap_ratio_wh,
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
            return merged.with_nms(threshold=self.iou_threshold)
        elif self.overlap_filter == OverlapFilter.NON_MAX_MERGE:
            return merged.with_nmm(threshold=self.iou_threshold)
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
        resolution_wh: Tuple[int, int],
        slice_wh: Tuple[int, int],
        overlap_ratio_wh: Optional[Tuple[float, float]],
        overlap_wh: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Generate offset coordinates for slicing an image based on the given resolution,
        slice dimensions, and overlap ratios.

        Args:
            resolution_wh (Tuple[int, int]): A tuple representing the width and height
                of the image to be sliced.
            slice_wh (Tuple[int, int]): Dimensions of each slice measured in pixels. The
            tuple should be in the format `(width, height)`.
            overlap_ratio_wh (Optional[Tuple[float, float]]): A tuple representing the
                desired overlap ratio for width and height between consecutive slices.
                Each value should be in the range [0, 1), where 0 means no overlap and
                a value close to 1 means high overlap.
            overlap_wh (Optional[Tuple[int, int]]): A tuple representing the desired
                overlap for width and height between consecutive slices measured in
                pixels. Each value should be greater than or equal to 0.

        Returns:
            np.ndarray: An array of shape `(n, 4)` containing coordinates for each
                slice in the format `[xmin, ymin, xmax, ymax]`.

        Note:
            The function ensures that slices do not exceed the boundaries of the
                original image. As a result, the final slices in the row and column
                dimensions might be smaller than the specified slice dimensions if the
                image's width or height is not a multiple of the slice's width or
                height minus the overlap.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_width = (
            overlap_wh[0]
            if overlap_wh is not None
            else int(overlap_ratio_wh[0] * slice_width)
        )
        overlap_height = (
            overlap_wh[1]
            if overlap_wh is not None
            else int(overlap_ratio_wh[1] * slice_height)
        )

        width_stride = slice_width - overlap_width
        height_stride = slice_height - overlap_height

        ws = np.arange(0, image_width, width_stride)
        hs = np.arange(0, image_height, height_stride)

        xmin, ymin = np.meshgrid(ws, hs)
        xmax = np.clip(xmin + slice_width, 0, image_width)
        ymax = np.clip(ymin + slice_height, 0, image_height)

        offsets = np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)

        return offsets

    @staticmethod
    def _validate_overlap(
        overlap_ratio_wh: Optional[Tuple[float, float]],
        overlap_wh: Optional[Tuple[int, int]],
    ) -> None:
        if overlap_ratio_wh is not None and overlap_wh is not None:
            raise ValueError(
                "Both `overlap_ratio_wh` and `overlap_wh` cannot be provided. "
                "Please provide only one of them."
            )
        if overlap_ratio_wh is None and overlap_wh is None:
            raise ValueError(
                "Either `overlap_ratio_wh` or `overlap_wh` must be provided. "
                "Please provide one of them."
            )

        if overlap_ratio_wh is not None:
            if not (0 <= overlap_ratio_wh[0] < 1 and 0 <= overlap_ratio_wh[1] < 1):
                raise ValueError(
                    "Overlap ratios must be in the range [0, 1). "
                    f"Received: {overlap_ratio_wh}"
                )
        if overlap_wh is not None:
            if not (overlap_wh[0] >= 0 and overlap_wh[1] >= 0):
                raise ValueError(
                    "Overlap values must be greater than or equal to 0. "
                    f"Received: {overlap_wh}"
                )
