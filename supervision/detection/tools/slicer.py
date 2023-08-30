from typing import Callable, Optional, Tuple

import numpy as np

from supervision import crop_image
from supervision.detection.core import Detections, validate_inference_callback
from supervision.detection.utils import move_boxes


def move_detections(detections: Detections, offset: np.array) -> Detections:
    """
    Args:
        detections (sv.Detections): Detections object to be moved.
        offset (np.array): An array of shape `(2,)` containing offset values in format is `[dx, dy]`.
    Returns:
        (sv.Detections) repositioned Detections object.
    """
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
    return detections


class InferenceSlicer:
    """
    Slicing inference(SAHI) method for small target detection.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], Detections],
        slice_wh: Tuple[int, int] = (320, 320),
        overlap_ratio_wh: Tuple[float, float] = (0.2, 0.2),
        iou_threshold: Optional[float] = 0.5,
    ):
        self.slice_wh = slice_wh
        self.overlap_ratio_wh = overlap_ratio_wh
        self.iou_threshold = iou_threshold
        self.callback = callback
        validate_inference_callback(callback=callback)

    def __call__(self, image: np.ndarray) -> Detections:
        detections_list = []
        resolution_wh = (image.shape[1], image.shape[0])
        offsets = self._generate_offset(
            resolution_wh=resolution_wh,
            slice_wh=self.slice_wh,
            overlap_ratio_wh=self.overlap_ratio_wh,
        )

        for offset in offsets:
            image_slice = crop_image(image=image, xyxy=offset)
            detections = self.callback(image_slice)
            detections = move_detections(detections=detections, offset=offset)
            detections_list.append(detections)
        return Detections.merge(detections_list=detections_list).with_nms(
            threshold=self.iou_threshold
        )

    @staticmethod
    def _generate_offset(
        resolution_wh: Tuple[int, int],
        slice_wh: Tuple[int, int],
        overlap_ratio_wh: Tuple[float, float],
    ) -> np.ndarray:
        """
        Generate offset coordinates for slicing an image based on the given resolution, slice dimensions, and overlap ratios.

        Args:
            resolution_wh (Tuple[int, int]): A tuple representing the width and height of the image to be sliced.
            slice_wh (Tuple[int, int]): A tuple representing the desired width and height of each slice.
            overlap_ratio_wh (Tuple[float, float]): A tuple representing the desired overlap ratio for width and height between consecutive slices. Each value should be in the range [0, 1), where 0 means no overlap and a value close to 1 means high overlap.

        Returns:
            np.ndarray: An array of shape `(n, 4)` containing coordinates for each slice in the format `[xmin, ymin, xmax, ymax]`.

        Note:
            The function ensures that slices do not exceed the boundaries of the original image. As a result, the final slices in the row and column dimensions might be smaller than the specified slice dimensions if the image's width or height is not a multiple of the slice's width or height minus the overlap.
        """
        slice_width, slice_height = slice_wh
        image_width, image_height = resolution_wh
        overlap_ratio_width, overlap_ratio_height = overlap_ratio_wh

        width_stride = slice_width - int(overlap_ratio_width * slice_width)
        height_stride = slice_height - int(overlap_ratio_height * slice_height)

        ws = np.arange(0, image_width, width_stride)
        hs = np.arange(0, image_height, height_stride)

        xmin, ymin = np.meshgrid(ws, hs)
        xmax = np.clip(xmin + slice_width, 0, image_width)
        ymax = np.clip(ymin + slice_height, 0, image_height)

        offsets = np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)

        return offsets
