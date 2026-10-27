"""Parse native Inference OBB tuples without altering ordered sides or front."""

from typing import Any

import numpy as np
import numpy.typing as npt

from supervision.config import (
    CLASS_HEADING_METADATA_FIELD,
    CLASS_NAME_DATA_FIELD,
    DETECTION_ID_DATA_FIELD,
    HEADING_DATA_FIELD,
    NATIVE_OBB_DATA_FIELD,
    ORIENTED_BOX_COORDINATES,
)
from supervision.detection.utils._typing import _DetectionDataType, _MetadataType


def _parse_native_obb_result(
    result: dict[str, Any],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    _DetectionDataType,
    _MetadataType,
]:
    """Validate frozen semantics and derive float corners from native image tuples."""
    if result.get("task_type") != "oriented-object-detection":
        raise ValueError(
            "Native OBB responses require oriented-object-detection task_type"
        )
    class_heading = result.get(CLASS_HEADING_METADATA_FIELD)
    if not isinstance(class_heading, dict) or any(
        type(v) is not bool for v in class_heading.values()
    ):
        raise ValueError(
            "Native OBB responses require frozen class_heading capabilities"
        )
    rows = result.get("predictions", [])
    names = np.asarray([p["class"] for p in rows], dtype=str)
    headings = [p.get(HEADING_DATA_FIELD) for p in rows]
    if any(
        type(h) is not bool or class_heading.get(n) is not h
        for n, h in zip(names, headings)
    ):
        raise ValueError(
            "Each native OBB heading must match its frozen class capability"
        )
    if any(
        not isinstance(p.get(DETECTION_ID_DATA_FIELD), str)
        or not p[DETECTION_ID_DATA_FIELD]
        for p in rows
    ):
        raise ValueError("Native OBB responses require stable detection_id values")
    try:
        obb = np.asarray([p["obb"] for p in rows], dtype=np.float64).reshape(-1, 5)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Native OBB predictions require [cx,cy,width,height,theta]"
        ) from error
    if (
        obb.shape != (len(rows), 5)
        or not np.isfinite(obb).all()
        or np.any(obb[:, 2:4] <= 0)
    ):
        raise ValueError(
            "Native OBB geometry must be finite with positive ordered sides"
        )
    local = obb[:, None, 2:4] * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / 2
    c, s = np.cos(obb[:, 4, None]), np.sin(obb[:, 4, None])
    corners = (
        np.stack(
            (
                local[:, :, 0] * c - local[:, :, 1] * s,
                local[:, :, 0] * s + local[:, :, 1] * c,
            ),
            axis=-1,
        )
        + obb[:, None, :2]
    )
    bounds = (
        np.concatenate((corners.min(1), corners.max(1)), axis=1)
        if rows
        else np.empty((0, 4), dtype=np.float64)
    )
    confidence = np.asarray([p["confidence"] for p in rows], dtype=np.float64)
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Native OBB confidence must be finite and in [0,1]")
    data: _DetectionDataType = {
        CLASS_NAME_DATA_FIELD: names,
        ORIENTED_BOX_COORDINATES: corners,
        HEADING_DATA_FIELD: np.asarray(headings, dtype=bool),
        NATIVE_OBB_DATA_FIELD: np.ones(len(rows), dtype=bool),
        DETECTION_ID_DATA_FIELD: np.asarray(
            [p[DETECTION_ID_DATA_FIELD] for p in rows], dtype=str
        ),
    }
    metadata: _MetadataType = {
        CLASS_HEADING_METADATA_FIELD: dict(class_heading),
        NATIVE_OBB_DATA_FIELD: True,
    }
    if "image" in result:
        metadata["image_dimensions"] = [
            result["image"]["height"],
            result["image"]["width"],
        ]
    return (
        bounds,
        confidence,
        np.asarray([p["class_id"] for p in rows], dtype=np.int64),
        data,
        metadata,
    )
