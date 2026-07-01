import logging
from itertools import chain
from typing import Any, Literal, cast, overload

import cv2
import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils._typing import _DetectionDataType, _MetadataType
from supervision.detection.utils.converters import polygon_to_mask, rle_to_mask
from supervision.geometry.core import Vector

logger = logging.getLogger(__name__)


def _full_image_xyxy(
    count: int,
    image_height: int,
    image_width: int,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[Any]:
    """Return full-frame inclusive boxes for lossless mask compaction."""
    return np.tile(
        np.array([[0, 0, image_width - 1, image_height - 1]], dtype=dtype),
        (count, 1),
    )


def _valid_rle_payload(prediction: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first valid RLE payload from ``rle`` then ``rle_mask``."""
    for key in ("rle", "rle_mask"):
        rle_data = prediction.get(key)
        if isinstance(rle_data, dict) and {"size", "counts"}.issubset(rle_data):
            return rle_data
    return None


def extract_ultralytics_masks(yolov8_results: Any) -> npt.NDArray[np.bool_] | None:
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

    return cast(npt.NDArray[np.bool_], np.asarray(mask_maps, dtype=bool))


def _resolve_rle_mask(
    rle_data: dict[str, Any],
    image_height: int,
    image_width: int,
    compact_masks: bool,
) -> tuple[npt.NDArray[np.bool_] | None, dict[str, Any] | None]:
    """Decode one RLE prediction into (dense_mask, compact_pending).

    Returns ``(dense_mask, None)`` when ``compact_masks=False`` or when the
    RLE size does not match the image size (fall back to dense decode).
    Returns ``(None, rle_data)`` when ``compact_masks=True`` and the RLE size
    matches, deferring the item to the post-loop batch call.
    Returns ``(None, None)`` on decode failure (caller should skip mask).

    Args:
        rle_data: Validated COCO RLE dict with ``"size"`` and ``"counts"`` keys.
        image_height: Full-image height in pixels.
        image_width: Full-image width in pixels.
        compact_masks: Whether to return a deferred item for batch processing.

    Returns:
        A 2-tuple ``(dense_mask, pending)`` where at most one element is
        non-``None``.
    """
    try:
        h, w = rle_data["size"]
        if compact_masks and (h, w) == (image_height, image_width):
            # Sizes match; defer to the post-loop CompactMask.from_coco_rle call.
            return None, rle_data
        if compact_masks and (h, w) != (image_height, image_width):
            logger.debug(
                "compact_masks=True: RLE size %s does not match image "
                "size (%d, %d); falling back to dense decode.",
                (h, w),
                image_height,
                image_width,
            )
        mask: npt.NDArray[np.bool_] = rle_to_mask(rle_data["counts"], (w, h))
        if (h, w) != (image_height, image_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask, None
    except (ValueError, AssertionError, KeyError, TypeError, OverflowError) as exc:
        logger.warning(
            "Failed to decode RLE mask payload; falling back to box-only "
            "detection. Reason: %s",
            exc,
        )
        return None, None


def _all_present_or_none(
    values: list[Any],
    label: str,
    dtype: npt.DTypeLike,
) -> npt.NDArray[Any] | None:
    # Identity check (`v is None`) is required when values may contain numpy arrays:
    # `None in values` triggers element-wise comparison and raises ValueError.
    missing = sum(v is None for v in values)
    if 0 < missing < len(values):
        logger.warning(
            "Partial %s in batch; dropping all to preserve alignment with xyxy.", label
        )
    if not values or missing > 0:
        return None
    return np.array(values, dtype=dtype)


def _decode_compact_masks(
    coco_rle_pending: list[tuple[int, Any, list[float]]],
    polygon_compact_map: dict[int, CompactMask],
    image_height: int,
    image_width: int,
    n_predictions: int,
) -> CompactMask | None:
    """Decode deferred COCO-RLE entries and merge with polygon compact masks.

    Attempts a single batched ``CompactMask.from_coco_rle`` call for all pending
    items to eliminate per-prediction call overhead on the happy path.  On any
    decode failure the call is retried per-prediction so that one malformed RLE
    payload does not abort the entire batch.

    Isolation note: this function isolates the *decode* step only.  If the
    per-prediction fallback still leaves some predictions without masks, the
    mixed-modality guard drops ALL masks to preserve alignment with ``xyxy``.

    Args:
        coco_rle_pending: Deferred COCO-RLE items collected during the prediction
            loop, each as ``(xyxy_idx, rle_dict, bbox)``.
        polygon_compact_map: Already-decoded polygon masks keyed by ``xyxy_idx``.
        image_height: Frame height in pixels.
        image_width: Frame width in pixels.
        n_predictions: Total number of accepted predictions; used by the
            mixed-modality alignment guard.

    Returns:
        A merged ``CompactMask`` when all predictions that carry masks decoded
        successfully, or ``None`` when no masks are present or the
        mixed-modality guard triggers.

    Examples:
        >>> _decode_compact_masks([], {}, 1080, 1920, 5) is None
        True
    """
    coco_compact_map: dict[int, CompactMask] = {}
    if coco_rle_pending:
        pending_indices = [t[0] for t in coco_rle_pending]
        pending_rles = [t[1] for t in coco_rle_pending]
        pending_xyxy = np.array([t[2] for t in coco_rle_pending], dtype=np.float64)
        try:
            batch_cm = CompactMask.from_coco_rle(
                pending_rles, pending_xyxy, (image_height, image_width)
            )
            for local_idx, global_idx in enumerate(pending_indices):
                coco_compact_map[global_idx] = batch_cm[local_idx : local_idx + 1]
        except (ValueError, AssertionError, KeyError, TypeError, OverflowError) as exc:
            logger.warning(
                "Batch compact RLE decode failed (%s); retrying "
                "per-prediction for fault isolation.",
                exc,
            )
            for xyxy_idx, rle_dict, bbox in coco_rle_pending:
                try:
                    single_cm = CompactMask.from_coco_rle(
                        [rle_dict],
                        np.array([bbox], dtype=np.float64),
                        (image_height, image_width),
                    )
                    coco_compact_map[xyxy_idx] = single_cm[0:1]
                except (
                    ValueError,
                    AssertionError,
                    KeyError,
                    TypeError,
                    OverflowError,
                ) as per_exc:
                    logger.warning(
                        "Compact RLE decode failed for prediction at index %d; "
                        "dropping that mask. Reason: %s",
                        xyxy_idx,
                        per_exc,
                    )
    all_compact = {**coco_compact_map, **polygon_compact_map}
    compact_parts: list[CompactMask] = [
        all_compact[i] for i in sorted(all_compact.keys())
    ]
    if 0 < len(compact_parts) < n_predictions:
        logger.warning(
            "Mixed-modality compact batch: %d of %d predictions carry masks; "
            "dropping all masks to preserve alignment with xyxy.",
            len(compact_parts),
            n_predictions,
        )
        compact_parts = []
    return CompactMask.merge(compact_parts) if compact_parts else None


@overload
def process_roboflow_result(
    roboflow_result: dict[str, Any],
    *,
    compact_masks: Literal[False] = False,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    npt.NDArray[np.bool_] | None,
    npt.NDArray[np.integer] | None,
    _DetectionDataType,
]: ...


@overload
def process_roboflow_result(
    roboflow_result: dict[str, Any],
    *,
    compact_masks: Literal[True],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    CompactMask | None,
    npt.NDArray[np.integer] | None,
    _DetectionDataType,
]: ...


def process_roboflow_result(
    roboflow_result: dict[str, Any],
    *,
    compact_masks: bool = False,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
    npt.NDArray[np.bool_] | CompactMask | None,
    npt.NDArray[np.integer] | None,
    _DetectionDataType,
]:
    """Parse a Roboflow API or Inference package result into detection arrays.

    The returned ``data`` dict always contains ``CLASS_NAME_DATA_FIELD`` as a
    string-dtype NumPy array. When ``predictions`` is empty, the array has
    shape ``(0,)`` with ``dtype=str``, preserving dtype contracts for callers
    that mix empty and non-empty results.

    Args:
        roboflow_result: Raw dict from the Roboflow REST API or the Inference
            package (after ``.dict()`` serialisation).
        compact_masks: When ``True``, return segmentation masks as
            :class:`~supervision.detection.compact_mask.CompactMask` instead of
            a dense boolean array when mask data is present.

    Returns:
        A 6-tuple of ``(xyxy, confidence, class_id, masks, tracker_ids, data)``
        where each array is aligned with the others. ``masks`` is ``None``
        when no predictions include mask data, or when only a subset do
        (mixed-modality batch) — in that case all masks are dropped to preserve
        alignment with ``xyxy``. When ``compact_masks=True`` and masks are
        present, ``masks`` is a :class:`CompactMask`; otherwise it is a dense
        boolean array. ``tracker_ids`` is ``None`` when no predictions carry a
        tracker ID, or when only a subset do (mixed batch) — in that case all
        tracker IDs are dropped to preserve alignment with ``xyxy``.

    Examples:
        >>> from supervision.detection.utils.internal import process_roboflow_result
        >>> result = {"predictions": [], "image": {"width": 100, "height": 100}}
        >>> _, _, _, _, _, data = process_roboflow_result(result)
        >>> data["class_name"].dtype.kind
        'U'
    """
    if not roboflow_result["predictions"]:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            None,
            None,
            {CLASS_NAME_DATA_FIELD: np.empty(0, dtype=str)},
        )

    xyxy: list[list[float]] = []
    confidence: list[float] = []
    class_id: list[int] = []
    class_name: list[str] = []
    masks: list[npt.NDArray[np.bool_] | None] = []
    # Deferred COCO-RLE processing: collect validated pairs here, then after the
    # loop attempt a single batched from_coco_rle call (happy path). If that batch
    # call fails, fall back to decoding each pending prediction individually for
    # fault isolation.
    _coco_rle_pending: list[tuple[int, Any, list[float]]] = []  # (xyxy_idx, rle, bbox)
    _polygon_compact_map: dict[int, CompactMask] = {}  # xyxy_idx → CompactMask
    tracker_ids: list[int | None] = []

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

        rle_data = _valid_rle_payload(prediction)
        mask: npt.NDArray[np.bool_] | None = None
        compact_mask: CompactMask | None = None
        if rle_data is not None:
            _dense, _pending = _resolve_rle_mask(
                rle_data, image_height, image_width, compact_masks
            )
            if _dense is None and _pending is None:
                # Decode failed; treat as no-mask prediction.
                rle_data = None
            elif _pending is None:
                # Dense result: compact_masks=False, or size-mismatch fallback.
                mask = _dense
                if compact_masks and mask is not None:
                    compact_mask = CompactMask.from_dense(
                        masks=mask[np.newaxis, ...],
                        xyxy=_full_image_xyxy(1, image_height, image_width),
                        image_shape=(image_height, image_width),
                    )
            # else: _pending set → deferred to batch; mask and compact_mask stay None
        if rle_data is not None:
            xyxy_idx = len(xyxy)
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            if compact_masks:
                if compact_mask is not None:
                    # Fallback dense path (size mismatch): compact_mask always set.
                    _polygon_compact_map[xyxy_idx] = compact_mask
                else:
                    # Main COCO-RLE path: (h, w) == image size; defer to batch.
                    # Pass the detector bbox so _rle_split_cols only walks
                    # crop columns, not the full image width.
                    _coco_rle_pending.append(
                        (
                            xyxy_idx,
                            rle_data,
                            [x_min, y_min, x_max, y_max],
                        )
                    )
            elif mask is not None:
                masks.append(mask)
            tracker_ids.append(prediction.get("tracker_id"))
        elif "points" not in prediction:
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            masks.append(None)
            tracker_ids.append(prediction.get("tracker_id"))
        elif len(prediction["points"]) >= 3:
            polygon = np.array(
                [[point["x"], point["y"]] for point in prediction["points"]], dtype=int
            )
            mask = polygon_to_mask(
                polygon, resolution_wh=(image_width, image_height)
            ).astype(bool)
            xyxy_idx = len(xyxy)
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(prediction["class_id"])
            class_name.append(prediction["class"])
            confidence.append(prediction["confidence"])
            if compact_masks:
                _polygon_compact_map[xyxy_idx] = CompactMask.from_dense(
                    masks=mask[np.newaxis, ...],
                    xyxy=_full_image_xyxy(1, image_height, image_width),
                    image_shape=(image_height, image_width),
                )
            else:
                masks.append(mask)
            tracker_ids.append(prediction.get("tracker_id"))

    xyxy_arr: npt.NDArray[np.floating] = (
        np.array(xyxy, dtype=np.float64) if len(xyxy) > 0 else np.empty((0, 4))
    )
    confidence_arr: npt.NDArray[np.floating] = (
        np.array(confidence, dtype=np.float64) if len(confidence) > 0 else np.empty(0)
    )
    class_id_arr: npt.NDArray[np.integer] = (
        np.array(class_id, dtype=np.int64)
        if len(class_id) > 0
        else np.empty(0, dtype=np.int64)
    )
    class_name_arr: npt.NDArray[np.str_] = (
        np.array(class_name) if len(class_name) > 0 else np.empty(0, dtype=str)
    )
    masks_arr: npt.NDArray[np.bool_] | CompactMask | None
    if compact_masks:
        masks_arr = _decode_compact_masks(
            _coco_rle_pending,
            _polygon_compact_map,
            image_height,
            image_width,
            len(xyxy),
        )
    else:
        masks_arr = _all_present_or_none(masks, "mask", dtype=bool)
    tracker_id_arr: npt.NDArray[np.integer] | None = _all_present_or_none(
        tracker_ids, "tracker_id", dtype=np.int64
    )
    data: _DetectionDataType = {CLASS_NAME_DATA_FIELD: class_name_arr}

    return (
        xyxy_arr,
        confidence_arr,
        class_id_arr,
        masks_arr,
        tracker_id_arr,
        data,
    )


def is_data_equal(
    data_a: _DetectionDataType,
    data_b: _DetectionDataType,
) -> bool:
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


def is_metadata_equal(metadata_a: _MetadataType, metadata_b: _MetadataType) -> bool:
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
    data_list: list[_DetectionDataType],
) -> _DetectionDataType:
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

    merged_data: dict[str, Any] = {key: [] for key in all_keys_sets[0]}
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

    return cast(_DetectionDataType, merged_data)


def merge_metadata(metadata_list: list[_MetadataType]) -> _MetadataType:
    """
    Merge metadata from a list of metadata dictionaries.

    This function combines the metadata dictionaries. If a key appears in more than one
    dictionary, the values must be identical for the merge to succeed.

    Warning: Assumes that empty detections were filtered-out before passing metadata to
    this function.

    Args:
        metadata_list: A list of metadata dictionaries to merge.

    Returns:
        A single merged metadata dictionary.

    Raises:
        ValueError: If there are conflicting values for the same key or if
        dictionaries have different keys.
    """
    if not metadata_list:
        return {}

    all_keys_sets = [set(metadata.keys()) for metadata in metadata_list]
    if not all(keys_set == all_keys_sets[0] for keys_set in all_keys_sets):
        raise ValueError("All metadata dictionaries must have the same keys to merge.")

    merged_metadata: _MetadataType = {}
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
    data: _DetectionDataType,
    index: int | slice | list[int] | npt.NDArray[np.integer | np.bool_],
) -> _DetectionDataType:
    """
    Retrieve a subset of the data dictionary based on the given index.

    Args:
        data: The data dictionary of the Detections object.
        index: The index or indices specifying the subset to retrieve.

    Returns:
        A subset of the data dictionary corresponding to the specified index.
    """
    subset_data: _DetectionDataType = {}
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


def cross_product(
    anchors: npt.NDArray[np.number], vector: Vector
) -> npt.NDArray[np.number]:
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
    return cast(
        npt.NDArray[np.number], np.cross(vector_at_zero, anchors - vector_start)
    )
