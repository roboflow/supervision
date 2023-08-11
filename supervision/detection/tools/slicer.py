from typing import Callable, Optional, Tuple

import numpy as np

from supervision.detection.core import Detections, validate_inference_callback
from supervision.detection.utils import move_boxes


class Slicer:
    """
    Slicing inference(SAHI) method for small target detection.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], Detections],
        sliced_width: Optional[int] = 320,
        sliced_height: Optional[int] = 320,
        overlap_width_ratio: Optional[float] = 0.2,
        overlap_height_ratio: Optional[float] = 0.2,
        iou_threshold: Optional[float] = 0.5,
    ):
        """
        Args:
            callback (Callable): model callback method which returns detections as sv.Detections
            sliced_width (int): width of each slice
            sliced_height (int): height of each slice
            overlap_width_ratio (float): Fractional overlap in width of each
                                        slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                                        overlap of 20 pixels). Default 0.2.
            overlap_height_ratio (float): Fractional overlap in height of each
                                        slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                                        overlap of 20 pixels). Default 0.2.
            iou_threshold (float): non-maximum suppression iou threshold to remove overlapping detections
        """
        self.siced_width = sliced_width
        self.sliced_height = sliced_height
        self.overlap_width_ratio = overlap_width_ratio
        self.overlap_height_ratio = overlap_height_ratio
        self.iou_threshold = iou_threshold
        self.callback = callback
        validate_inference_callback(callback=callback)

    def __call__(self, image: np.ndarray) -> Detections:
        """

        Args:
            image (np.ndarray):

        Returns:
            sv.Detections

             Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> dataset = sv.DetectionDataset.from_yolo(...)

            >>> model = YOLO(...)
            >>> def callback(slice: np.ndarray) -> sv.Detections:
            ...     result = model(slice)[0]
            ...     return sv.Detections.from_ultralytics(result)

            >>> slicer = sv.Slicer(
            ...     callback = callback
            ... )

            >>> detections = slicer(image)
            ```
        """
        detections = []
        image_height, image_width, _ = image.shape
        offsets = self._offset_generation(
            image_width=image_width, image_height=image_height
        )

        for offset in offsets:
            slice = image[offset[1] : offset[3], offset[0] : offset[2]]
            det = self.callback(slice)
            det = self._reposition_detections(detection=det, offset=offset)
            detections.append(det)
        detection = Detections.merge(detections_list=detections).with_nms(
            threshold=self.iou_threshold
        )
        return detection

    def _offset_generation(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Args:
            image_width (int): width of the input image
            image_height (int): height of the input image

        Returns:
            list of slice locations according to slicer parameters
        """
        width_stride = self.siced_width - int(
            self.overlap_width_ratio * self.siced_width
        )
        height_stride = self.sliced_height - int(
            self.overlap_height_ratio * self.sliced_height
        )
        offsets = []
        for h in range(0, image_height, height_stride):
            for w in range(0, image_width, width_stride):
                xmin = w
                ymin = h
                xmax = min(image_width, w + self.siced_width)
                ymax = min(image_height, h + self.sliced_height)
                offsets.append([xmin, ymin, xmax, ymax])
        offsets = np.asarray(offsets)
        return offsets

    @staticmethod
    def _reposition_detections(detection: Detections, offset: np.array) -> Detections:
        """
        Args:
            detection (np.ndarray): result of model inference of the slice
            slice_location (Tuple[int, int, int, int]): slice location at which inference was performed
        Returns:
            (sv.Detections) repositioned detections result based on original image
        """
        if len(detection) == 0:
            return detection
        xyxy = move_boxes(boxes=detection.xyxy, offset=offset)
        detection.xyxy = xyxy
        return detection