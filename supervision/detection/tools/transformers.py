import io
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.utils import mask_to_xyxy


def process_transformers_detection_result(
    detection_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Process the result of Transformers object detection functions such as
    `post_process` (v4) and `post_process_detection` (v5).

    Args:
        detection_result (dict): Dictionary containing detection results with keys
            'boxes', 'labels', and 'scores'.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.

    Returns:
        dict: Processed detection result including bounding boxes, confidence scores,
              class IDs, and data.
    """
    class_ids = detection_result["labels"].cpu().detach().numpy().astype(int)
    data = append_class_names_to_data(class_ids, id2label, {})

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
    Process the result of Transformers segmentation functions such as
    `post_process_panoptic`, `post_process_segmentation`, and `post_process_instance`
    (v4).

    Args:
        segmentation_result (dict): Dictionary containing segmentation results with keys
            'masks', 'labels', and 'scores'.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks, confidence
              scores, class IDs, and data.
    """
    if "png_string" in segmentation_result:
        return process_transformers_v4_panoptic_segmentation_result(
            segmentation_result, id2label
        )
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
            data=append_class_names_to_data(class_ids, id2label, {}),
        )


def process_transformers_v5_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Process the result of Transformers segmentation functions such as
    `post_process_semantic_segmentation`, `post_process_instance_segmentation`, and
    `post_process_panoptic_segmentation` (v5).

    Args:
        segmentation_result (Union[dict, np.ndarray]): Either a dictionary containing
            segmentation results or an ndarray representing a segmentation map.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks, confidence
              scores, class IDs, and data.
    """
    if segmentation_result.__class__.__name__ == "Tensor":
        segmentation_array = segmentation_result.cpu().detach().numpy()
        return process_transformers_v5_panoptic_segmentation_result(
            segmentation_array, id2label
        )

    return process_transformers_v5_semantic_or_instance_segmentation_result(
        segmentation_result, id2label
    )


def process_transformers_v5_semantic_or_instance_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Process the result of Transformers segmentation functions such as
    `post_process_semantic_segmentation` and `post_process_instance_segmentation` (v5).

    Args:
        segmentation_result (dict): Dictionary containing segmentation results with keys
            `segments_info` and `segmentation`.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks, confidence
              scores, class IDs, and data.
    """
    segments_info = segmentation_result["segments_info"]
    scores = np.array([segment["score"] for segment in segments_info])
    class_ids = np.array([segment["label_id"] for segment in segments_info])
    segmentation_array = segmentation_result["segmentation"].cpu().detach().numpy()
    masks = np.array(
        [segmentation_array == segment["id"] for segment in segments_info]
    ).astype(bool)
    data = append_class_names_to_data(class_ids, id2label, {})

    return dict(
        xyxy=mask_to_xyxy(masks),
        mask=masks,
        confidence=scores,
        class_id=class_ids,
        data=data,
    )


def process_transformers_v4_panoptic_segmentation_result(
    segmentation_result: dict, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Process the result of the Transformers function `post_process_panoptic` (v4).

    Args:
        segmentation_result (dict): Dictionary containing segmentation results with keys
            such as 'png_string' and 'segments_info'.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks,
              class IDs, and data.
    """
    segments_info = segmentation_result["segments_info"]
    png_string = segmentation_result["png_string"]
    class_ids = np.array([segment["category_id"] for segment in segments_info])
    segmentation_array = png_string_to_segmentation_array(png_string=png_string)
    masks = np.array(
        [segmentation_array == segment["id"] for segment in segments_info]
    ).astype(bool)
    data = append_class_names_to_data(class_ids, id2label, {})

    return dict(
        xyxy=mask_to_xyxy(masks),
        mask=masks,
        class_id=class_ids,
        data=data,
    )


def process_transformers_v5_panoptic_segmentation_result(
    segmentation_array: np.ndarray, id2label: Optional[Dict[int, str]]
) -> dict:
    """
    Process the result of the Transformers function
    `post_process_panoptic_segmentation` (v5).

    Args:
        segmentation_array (np.ndarray): Segmentation array.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.

    Returns:
        dict: Processed segmentation result including bounding boxes, masks,
              class IDs, and data.
    """
    class_ids = np.unique(segmentation_array)
    masks = np.stack(
        [segmentation_array == class_id for class_id in class_ids], axis=0
    ).astype(bool)
    data = append_class_names_to_data(class_ids, id2label, {})
    return dict(xyxy=mask_to_xyxy(masks), mask=masks, class_id=class_ids, data=data)


def png_string_to_segmentation_array(png_string: bytes) -> np.ndarray:
    """
    Convert a PNG byte string to a label mask array.

    Args:
        png_string (bytes): A byte string representing the PNG image.

    Returns:
        np.ndarray: A label mask array with shape (H, W), where H and W
        are the height and width of the image. Each unique value in the array
        represents a different object or category.
    """
    image = Image.open(io.BytesIO(png_string))
    mask = np.array(image, dtype=np.uint8)
    return mask[:, :, 0]


def append_class_names_to_data(
    class_ids: np.ndarray,
    id2label: Optional[Dict[int, str]],
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Helper function to create or append to a data dictionary with class names if
    available.

    Args:
        class_ids (np.ndarray): Array of class IDs.
        id2label (Optional[Dict[int, str]]): A dictionary mapping class IDs to labels,
            typically part of the `transformers` model configuration. If provided, the
            resulting dictionary will include class names.
        data (Optional[Dict[str, Any]]): An existing data dictionary to append to.

    Returns:
        Dict[str, Any]: Dictionary containing class names if id2label is provided.
    """
    if data is None:
        data = {}

    if id2label is not None:
        class_names = np.array([id2label[class_id] for class_id in class_ids])
        data[CLASS_NAME_DATA_FIELD] = class_names

    return data
