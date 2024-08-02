from typing import Dict, Optional

import numpy as np

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.utils import mask_to_xyxy, png_to_mask


def get_data(class_ids: np.ndarray, id2label: Optional[Dict[int, str]]) -> dict:
    """
    Helper function to create data dictionary with class names if available.

    Args:
        class_ids (np.ndarray): Array of class IDs.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Dictionary containing class names if id2label is provided.
    """
    data = {}
    if id2label is not None:
        class_names = np.array([id2label[class_id] for class_id in class_ids])
        data[CLASS_NAME_DATA_FIELD] = class_names
    return data


def process_tensor_result(
    segmentation_array: np.ndarray, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Helper function to process result of transformers function
    post_process_panoptic_segmentation.

    Args:
        segmentation_array (np.ndarray): Segmentation array.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks,
              class IDs, and data.
    """
    class_ids = np.unique(segmentation_array)
    masks = np.stack(
        [(segmentation_array == class_id).astype(bool) for class_id in class_ids],
        axis=0,
    )
    data = get_data(class_ids, id2label)

    return dict(
        xyxy=mask_to_xyxy(masks), mask=masks, class_id=class_ids, data=data
    )


def process_detection_result(
    detection_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Helper function to process result of transformers functions
    post_process_object_detection and post_proces.

    Args:
        detection_result (dict): Dictionary containing detection results with keys
            'boxes', 'labels', and 'scores'.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Processed detection result including bounding boxes, confidence scores,
              class IDs, and data.
    """
    class_ids = detection_result["labels"].cpu().detach().numpy().astype(int)
    data = get_data(class_ids, id2label)

    return dict(
        xyxy=detection_result["boxes"].cpu().detach().numpy(),
        confidence=detection_result["scores"].cpu().detach().numpy(),
        class_id=class_ids,
        data=data,
    )


def process_transformers_v4_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Helper function to process result of transformers functions
    post_process_panoptic, post_process_segmentation and post_process_instance.

    Args:
        segmentation_result (dict): Dictionary containing segmentation results with keys
            'masks', 'labels', and 'scores'.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks, confidence
              scores, class IDs, and data.
    """
    if "png_string" in segmentation_result:
        return process_png_segmentation_result(segmentation_result, id2label)
    else:
        boxes = None
        if "boxes" in segmentation_result:
            boxes = segmentation_result["boxes"].cpu().detach().numpy()

        masks = segmentation_result["masks"].cpu().detach().numpy().astype(bool)
        class_ids = segmentation_result["labels"].cpu().detach().numpy().astype(int)

        return dict(
            xyxy=boxes if boxes is not None else mask_to_xyxy(masks),
            mask=np.squeeze(masks, axis=1) if boxes is not None else masks,
            confidence=segmentation_result["scores"].cpu().detach().numpy(),
            class_id=class_ids,
            data=get_data(class_ids, id2label),
        )


def process_transformers_v5_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Helper function to process result of transformers functions
    post_process_semantic_segmentation, post_process_instance_segmentation and
    post_process_panoptic_segmentation.

    Args:
        segmentation_result (Union[dict, np.ndarray]): Either a dictionary containing
            segmentation results or an ndarray representing a segmentation map.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks, confidence
              scores, class IDs, and data.
    """
    if segmentation_result.__class__.__name__ == "Tensor":
        segmentation_array = segmentation_result.cpu().detach().numpy()
        return process_tensor_result(segmentation_array, id2label)

    return process_segmentation_result(segmentation_result, id2label)


def process_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Helper function to process result of transformers functions
    post_process_semantic_segmentation and post_process_instance_segmentation.

    Args:
        segmentation_result (dict): Dictionary containing segmentation results with keys
            'segments_info' and 'segmentation'.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks, confidence
              scores, class IDs, and data.
    """
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

    return dict(
        xyxy=mask_to_xyxy(masks),
        mask=masks,
        confidence=scores,
        class_id=class_ids,
        data=data,
    )


def process_png_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Helper function to process result of transformers function post_process_panoptic.

    Args:
        segmentation_result (dict): Dictionary containing PNG string and segment information.
        id2label (Optional[Dict[int, str]]): Dictionary mapping class IDs to class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks,
              class IDs, and data.
    """
    segments_info = segmentation_result["segments_info"]
    class_ids = np.array([segment["category_id"] for segment in segments_info])
    segmentation_array = png_to_mask(segmentation_result["png_string"])
    masks = np.array(
        [
            (segmentation_array == segment["id"]).astype(bool)
            for segment in segments_info
        ]
    )
    data = get_data(class_ids, id2label)

    return dict(
        xyxy=mask_to_xyxy(masks),
        mask=masks,
        class_id=class_ids,
        data=data,
    )
