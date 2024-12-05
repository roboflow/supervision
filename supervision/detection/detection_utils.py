from typing import Tuple

import cv2
import numpy as np

from supervision.detection.core import ORIENTED_BOX_COORDINATES, Detections


def scale_detections(
    detections: Detections,
    letterbox_wh: Tuple[int, int],
    resolution_wh: Tuple[int, int],
) -> Detections:
    """
    This function scale the coordinates of bounding boxes and optionally scales the
    masks,oriented bounding boxes to fit a new resolution, taking into account the
    letterbox padding applied during the resizing process and return Detections object.

    Args:
        detections (Detections): The Detections object to be scaled.
        letterbox_wh (Tuple[int, int]): The width and height of the letterboxed image.
        resolution_wh (Tuple[int, int]): The target width and height for scaling.

    Returns:
        Detections: A new Detections object with scaled to target resolution.
    """
    input_w, input_h = resolution_wh
    letterbox_w, letterbox_h = letterbox_wh

    target_ratio = letterbox_w / letterbox_h
    image_ratio = input_w / input_h

    if image_ratio >= target_ratio:
        width_new = letterbox_w
        height_new = int(letterbox_w / image_ratio)
    else:
        height_new = letterbox_h
        width_new = int(letterbox_h * image_ratio)

    scale = input_w / width_new
    padding_top = (letterbox_h - height_new) // 2
    padding_left = (letterbox_w - width_new) // 2

    boxes = detections.xyxy.copy()
    boxes[:, [0, 2]] -= padding_left
    boxes[:, [1, 3]] -= padding_top
    boxes[:, [0, 2]] *= scale
    boxes[:, [1, 3]] *= scale

    scaled_mask = None
    if detections.mask is not None:
        masks = []
        for mask in detections.mask:
            mask = mask[
                padding_top : padding_top + height_new,
                padding_left : padding_left + width_new,
            ]
            scaled_mask_i = cv2.resize(
                mask.astype(np.uint8),
                (input_w, input_h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(bool)
            masks.append(scaled_mask_i)
        scaled_mask = np.array(masks)

    if ORIENTED_BOX_COORDINATES in detections.data:
        obbs = np.array(detections.data[ORIENTED_BOX_COORDINATES]).copy()
        obbs[:, :, 0] -= padding_left
        obbs[:, :, 1] -= padding_top
        obbs[:, :, 0] *= scale
        obbs[:, :, 1] *= scale
        detections.data[ORIENTED_BOX_COORDINATES] = obbs

    return Detections(
        xyxy=boxes,
        mask=scaled_mask,
        confidence=detections.confidence,
        class_id=detections.class_id,
        tracker_id=detections.tracker_id,
        data=detections.data,
        metadata=detections.metadata,
    )
