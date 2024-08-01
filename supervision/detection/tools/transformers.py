from typing import Dict, Optional

import numpy as np

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.utils import mask_to_xyxy, png_to_mask


def get_data(class_ids: np.ndarray, id2label: Optional[Dict[int, str]]) -> dict:
    """Helper function to create data dictionary with class names if available."""
    data = {}
    if id2label is not None:
        class_names = np.array([id2label[class_id] for class_id in class_ids])
        data[CLASS_NAME_DATA_FIELD] = class_names
    return data


def process_tensor_result(
    segmentation_array: np.ndarray, id2label: Optional[Dict[int, str]]
) -> Detections:
    """Process segmentation array result for segmentation."""
    class_ids = np.unique(segmentation_array)
    masks = np.stack(
        [(segmentation_array == class_id).astype(bool) for class_id in class_ids],
        axis=0,
    )
    data = get_data(class_ids, id2label)

    return Detections(
        xyxy=mask_to_xyxy(masks), mask=masks, class_id=class_ids, data=data
    )


def process_detection_result(
    detection_result: dict, id2label: Optional[Dict[int, str]]
) -> Detections:
    """Process detection results containing boxes and labels."""
    class_ids = detection_result["labels"].cpu().detach().numpy().astype(int)
    data = get_data(class_ids, id2label)

    return Detections(
        xyxy=detection_result["boxes"].cpu().detach().numpy(),
        confidence=detection_result["scores"].cpu().detach().numpy(),
        class_id=class_ids,
        data=data,
    )


def process_transformers_v4_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> Detections:
    """
    Process Transformers v4 segmentation results.

    Args:
        segmentation_result (dict): Dictionary containing segmentation results with keys 'masks', 'labels', and 'scores'.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        Detections: A Detections object created from the segmentation results.
    """

    if "png_string" in segmentation_result:
        return process_png_segmentation_result(segmentation_result, id2label)
    else:
        masks = segmentation_result["masks"].cpu().detach().numpy().astype(bool)
        class_ids = segmentation_result["labels"].cpu().detach().numpy().astype(int)

        return Detections(
            xyxy=mask_to_xyxy(masks),
            mask=masks,
            confidence=segmentation_result["scores"].cpu().detach().numpy(),
            class_id=class_ids,
            data=get_data(class_ids, id2label),
        )


def process_transformers_v5_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> Detections:
    """
    Process Transformers v5 segmentation results.

    Args:
        segmentation_result (Union[dict, np.ndarray]): Either a dictionary containing segmentation results or an ndarray representing a segmentation map.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        Detections: A Detections object created from the segmentation results.
    """

    if segmentation_result.__class__.__name__ == "Tensor":
        segmentation_array = segmentation_result.cpu().detach().numpy()
        return process_tensor_result(segmentation_array, id2label)

    return process_png_segmentation_result(segmentation_result, id2label)


def process_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> Detections:
    """Process segmentation results with masks and scores."""
    segments_info = segmentation_result["segments_info"]
    scores = np.array([segment["score"] for segment in segments_info])
    class_ids = np.array([segment["label_id"] for segment in segments_info])
    segmentation_array = segmentation_result["segmentation"].cpu().detach().numpy()
    masks = np.array(
        [
            (segmentation_array == segment["id"]).astype(bool)
            for segment in segments_info
        ]
    )
    data = get_data(class_ids, id2label)

    return Detections(
        xyxy=mask_to_xyxy(masks),
        mask=masks,
        confidence=scores,
        class_id=class_ids,
        data=data,
    )


def process_png_segmentation_result(
    png_result: dict, id2label: Optional[Dict[int, str]]
) -> Detections:
    """Process segmentation results from a PNG string."""
    segments_info = png_result["segments_info"]
    class_ids = np.array([segment["category_id"] for segment in segments_info])
    segmentation_array = png_to_mask(png_result["png_string"])
    masks = np.array(
        [
            (segmentation_array == segment["id"]).astype(bool)
            for segment in segments_info
        ]
    )
    data = get_data(class_ids, id2label)

    return Detections(
        xyxy=mask_to_xyxy(masks), mask=masks, class_id=class_ids, data=data
    )
