from __future__ import annotations

from itertools import chain
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.utils.converters import polygon_to_mask
from supervision.geometry.core import Vector


def extract_ultralytics_masks(yolov8_results) -> np.ndarray | None:
    if not yolov8_results.masks:
        return None

    orig_shape = yolov8_results.orig_shape
    inference_shape = tuple(yolov8_results.masks.data.shape[1:])

    pad = (0, 0)

    if inference_shape != orig_shape:
        gain = min(
            inference_shape[0] / orig_shape[0],
            inference_shape[1] / orig_shape[1],
        )
        pad = (
            (inference_shape[1] - orig_shape[1] * gain) / 2,
            (inference_shape[0] - orig_shape[0] * gain) / 2,
        )

    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(inference_shape[0] - pad[1]), int(inference_shape[1] - pad[0])

    mask_maps = []
    masks = yolov8_results.masks.data.cpu().numpy()
    for i in range(masks.shape[0]):
        mask = masks[i]
        mask = mask[top:bottom, left:right]

        if mask.shape != orig_shape:
            mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))

        mask_maps.append(mask)

    return np.asarray(mask_maps, dtype=bool)


def process_roboflow_result(
    roboflow_result: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, list[np.ndarray] | np.ndarray],
]:
    if not roboflow_result["predictions"]:
        return (
            np.empty((0, 4)),
            np.empty(0),
            np.empty(0),
            None,
            None,
            {CLASS_NAME_DATA_FIELD: np.empty(0)},
        )

    xyxy: list[list[float]] = []
    confidence: list[float] = []
    class_id: list[int] = []
    class_name: list[str] = []
    masks: list[np.ndarray] = []
    tracker_ids: list[int] = []

    image_width = int(roboflow_result["image"]["width"])
    image_height = int(roboflow_result["image"]["height"])

    for prediction in roboflow_result["predictions"]:
        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        x_min = x - width / 2
        y_min = y - height / 2
        x_max = x_min + width
        y_max = y_min + height

        if "points" not in prediction:
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            if "tracker_id" in prediction:
                tracker_ids.append(prediction["tracker_id"])
        elif len(prediction["points"]) >= 3:
            polygon = np.array(
                [[point["x"], point["y"]] for point in prediction["points"]], dtype=int
            )
            mask = polygon_to_mask(polygon, resolution_wh=(image_width, image_height))
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            masks.append(mask)
            if "tracker_id" in prediction:
                tracker_ids.append(prediction["tracker_id"])

    xyxy_arr = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))
    confidence_arr = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id_arr = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name_arr = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    masks_arr = np.array(masks, dtype=bool) if len(masks) > 0 else None
    tracker_id_arr = np.array(tracker_ids).astype(int) if len(tracker_ids) > 0 else None
    data: dict[str, np.ndarray] = {CLASS_NAME_DATA_FIELD: class_name_arr}

    return (
        xyxy_arr,
        confidence_arr,
        class_id_arr,
        masks_arr,
        tracker_id_arr,
        data,
    )


def is_data_equal(data_a: dict[str, np.ndarray], data_b: dict[str, np.ndarray]) -> bool:
    """
    Compares the data payloads of two Detections instances.

    Args:
        data_a, data_b: The data payloads of the instances.

    Returns:
        True if the data payloads are equal, False otherwise.
    """
    return set(data_a.keys()) == set(data_b.keys()) and all(
        np.array_equal(data_a[key], data_b[key]) for key in data_a
    )


def is_metadata_equal(metadata_a: dict[str, Any], metadata_b: dict[str, Any]) -> bool:
    """
    Compares the metadata payloads of two Detections instances.

    Args:
        metadata_a, metadata_b: The metadata payloads of the instances.

    Returns:
        True if the metadata payloads are equal, False otherwise.
    """
    return set(metadata_a.keys()) == set(metadata_b.keys()) and all(
        np.array_equal(metadata_a[key], metadata_b[key])
        if (
            isinstance(metadata_a[key], np.ndarray)
            and isinstance(metadata_b[key], np.ndarray)
        )
        else metadata_a[key] == metadata_b[key]
        for key in metadata_a
    )


def merge_data(
    data_list: list[dict[str, npt.NDArray[np.generic] | list]],
) -> dict[str, npt.NDArray[np.generic] | list]:
    """
    Merges the data payloads of a list of Detections instances.

    Warning: Assumes that empty detections were filtered-out before passing data to
    this function.

    Args:
        data_list: The data payloads of the Detections instances. Each data payload
            is a dictionary with the same keys, and the values are either lists or
            npt.NDArray[np.generic].

    Returns:
        A single data payload containing the merged data, preserving the original data
            types (list or npt.NDArray[np.generic]).

    Raises:
        ValueError: If data values within a single object have different lengths or if
            dictionaries have different keys.
    """
    if not data_list:
        return {}

    all_keys_sets = [set(data.keys()) for data in data_list]
    if not all(keys_set == all_keys_sets[0] for keys_set in all_keys_sets):
        raise ValueError("All data dictionaries must have the same keys to merge.")

    for data in data_list:
        lengths = [len(value) for value in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                "All data values within a single object must have equal length."
            )

    merged_data = {key: [] for key in all_keys_sets[0]}
    for data in data_list:
        for key in data:
            merged_data[key].append(data[key])

    for key in merged_data:
        if all(isinstance(item, list) for item in merged_data[key]):
            merged_data[key] = list(chain.from_iterable(merged_data[key]))
        elif all(isinstance(item, np.ndarray) for item in merged_data[key]):
            ndim = merged_data[key][0].ndim
            if ndim == 1:
                merged_data[key] = np.hstack(merged_data[key])
            elif ndim > 1:
                merged_data[key] = np.vstack(merged_data[key])
            else:
                raise ValueError(f"Unexpected array dimension for key '{key}'.")
        else:
            raise ValueError(
                f"Inconsistent data types for key '{key}'. Only np.ndarray and list "
                f"types are allowed."
            )

    return merged_data


def merge_metadata(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge metadata from a list of metadata dictionaries.

    This function combines the metadata dictionaries. If a key appears in more than one
    dictionary, the values must be identical for the merge to succeed.

    Warning: Assumes that empty detections were filtered-out before passing metadata to
    this function.

    Args:
        metadata_list (List[Dict[str, Any]]): A list of metadata dictionaries to merge.

    Returns:
        Dict[str, Any]: A single merged metadata dictionary.

    Raises:
        ValueError: If there are conflicting values for the same key or if
        dictionaries have different keys.
    """
    if not metadata_list:
        return {}

    all_keys_sets = [set(metadata.keys()) for metadata in metadata_list]
    if not all(keys_set == all_keys_sets[0] for keys_set in all_keys_sets):
        raise ValueError("All metadata dictionaries must have the same keys to merge.")

    merged_metadata: dict[str, Any] = {}
    for metadata in metadata_list:
        for key, value in metadata.items():
            if key not in merged_metadata:
                merged_metadata[key] = value
                continue

            other_value = merged_metadata[key]
            if isinstance(value, np.ndarray) and isinstance(other_value, np.ndarray):
                if not np.array_equal(merged_metadata[key], value):
                    raise ValueError(
                        f"Conflicting metadata for key: '{key}': "
                        "{type(value)}, {type(other_value)}."
                    )
            elif isinstance(value, np.ndarray) or isinstance(other_value, np.ndarray):
                # Since [] == np.array([]).
                raise ValueError(
                    f"Conflicting metadata for key: '{key}': "
                    "{type(value)}, {type(other_value)}."
                )
            else:
                if merged_metadata[key] != value:
                    raise ValueError(f"Conflicting metadata for key: '{key}'.")

    return merged_metadata


def get_data_item(
    data: dict[str, np.ndarray | list],
    index: int | slice | list[int] | np.ndarray,
) -> dict[str, np.ndarray | list]:
    """
    Retrieve a subset of the data dictionary based on the given index.

    Args:
        data: The data dictionary of the Detections object.
        index: The index or indices specifying the subset to retrieve.

    Returns:
        A subset of the data dictionary corresponding to the specified index.
    """
    subset_data: dict[str, np.ndarray | list] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            subset_data[key] = value[index]
        elif isinstance(value, list):
            if isinstance(index, slice):
                subset_data[key] = value[index]
            elif isinstance(index, list):
                subset_data[key] = [value[i] for i in index]
            elif isinstance(index, np.ndarray):
                if index.dtype == bool:
                    subset_data[key] = [
                        value[i] for i, index_value in enumerate(index) if index_value
                    ]
                else:
                    subset_data[key] = [value[i] for i in index]
            elif isinstance(index, int):
                subset_data[key] = [value[index]]
            else:
                raise TypeError(f"Unsupported index type: {type(index)}")
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    return subset_data


def cross_product(anchors: np.ndarray, vector: Vector) -> np.ndarray:
    """
    Get array of cross products of each anchor with a vector.
    Args:
        anchors: Array of anchors of shape (number of anchors, detections, 2)
        vector: Vector to calculate cross product with

    Returns:
        Array of cross products of shape (number of anchors, detections)
    """
    vector_at_zero = np.array(
        [
            vector.end.x - vector.start.x,
            vector.end.y - vector.start.y,
        ]
    )
    vector_start = np.array([vector.start.x, vector.start.y])
    return np.cross(vector_at_zero, anchors - vector_start)
