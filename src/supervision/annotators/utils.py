from __future__ import annotations

import re
import textwrap
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position

PENDING_TRACK_COLOR = Color.GREY
PENDING_TRACK_ID = -1


class ColorLookup(Enum):
    """
    Enumeration class to define strategies for mapping colors to annotations.

    This enum supports three different lookup strategies:
        - `INDEX`: Colors are determined by the index of the detection within the scene.
        - `CLASS`: Colors are determined by the class label of the detected object.
        - `TRACK`: Colors are determined by the tracking identifier of the object.
    """

    INDEX = "index"
    CLASS = "class"
    TRACK = "track"

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))


def resolve_color_idx(
    detections: Detections,
    detection_idx: int,
    color_lookup: ColorLookup | npt.NDArray[np.int_] = ColorLookup.CLASS,
) -> int:
    if detection_idx >= len(detections):
        raise ValueError(
            f"Detection index {detection_idx} "
            f"is out of bounds for detections of length {len(detections)}"
        )

    if isinstance(color_lookup, np.ndarray):
        if len(color_lookup) != len(detections):
            raise ValueError(
                f"Length of color lookup {len(color_lookup)} "
                f"does not match length of detections {len(detections)}"
            )
        return int(color_lookup[detection_idx])
    elif color_lookup == ColorLookup.INDEX:
        return detection_idx
    elif color_lookup == ColorLookup.CLASS:
        if detections.class_id is None:
            raise ValueError(
                "Could not resolve color by class because "
                "Detections do not have class_id. If using an annotator, "
                "try setting color_lookup to sv.ColorLookup.INDEX or "
                "sv.ColorLookup.TRACK."
            )
        return int(detections.class_id[detection_idx])
    elif color_lookup == ColorLookup.TRACK:
        if detections.tracker_id is None:
            raise ValueError(
                "Could not resolve color by track because "
                "Detections do not have tracker_id. Did you call "
                "tracker.update_with_detections(...) before annotating?"
            )
        return int(detections.tracker_id[detection_idx])
    raise ValueError(f"Unsupported color lookup strategy: {color_lookup}")


def resolve_text_background_xyxy(
    center_coordinates: tuple[int, int],
    text_wh: tuple[int, int],
    position: Position,
) -> tuple[int, int, int, int]:
    center_x, center_y = center_coordinates
    text_w, text_h = text_wh

    if position == Position.TOP_LEFT:
        return center_x, center_y - text_h, center_x + text_w, center_y
    elif position == Position.TOP_RIGHT:
        return center_x - text_w, center_y - text_h, center_x, center_y
    elif position == Position.TOP_CENTER:
        return (
            center_x - text_w // 2,
            center_y - text_h,
            center_x + text_w // 2,
            center_y,
        )
    elif position == Position.CENTER or position == Position.CENTER_OF_MASS:
        return (
            center_x - text_w // 2,
            center_y - text_h // 2,
            center_x + text_w // 2,
            center_y + text_h // 2,
        )
    elif position == Position.BOTTOM_LEFT:
        return center_x, center_y, center_x + text_w, center_y + text_h
    elif position == Position.BOTTOM_RIGHT:
        return center_x - text_w, center_y, center_x, center_y + text_h
    elif position == Position.BOTTOM_CENTER:
        return (
            center_x - text_w // 2,
            center_y,
            center_x + text_w // 2,
            center_y + text_h,
        )
    elif position == Position.CENTER_LEFT:
        return (
            center_x - text_w,
            center_y - text_h // 2,
            center_x,
            center_y + text_h // 2,
        )
    elif position == Position.CENTER_RIGHT:
        return (
            center_x,
            center_y - text_h // 2,
            center_x + text_w,
            center_y + text_h // 2,
        )


def get_color_by_index(color: Color | ColorPalette, idx: int) -> Color:
    if isinstance(color, ColorPalette):
        return color.by_idx(idx)
    return color


def resolve_color(
    color: Color | ColorPalette,
    detections: Detections,
    detection_idx: int,
    color_lookup: ColorLookup | npt.NDArray[np.int_] = ColorLookup.CLASS,
) -> Color:
    idx = resolve_color_idx(
        detections=detections,
        detection_idx=detection_idx,
        color_lookup=color_lookup,
    )
    if (
        isinstance(color_lookup, ColorLookup)
        and color_lookup == ColorLookup.TRACK
        and idx == PENDING_TRACK_ID
    ):
        return PENDING_TRACK_COLOR
    return get_color_by_index(color=color, idx=idx)


def wrap_text(text: Any, max_line_length: int | None = None) -> list[str]:
    """
    Wrap `text` to the specified maximum line length, respecting existing
    newlines. Falls back to str() if `text` is not already a string.

    Args:
        text: The text (or object) to wrap.
        max_line_length: Maximum width for each wrapped line.

    Returns:
        Wrapped lines.
    """

    if not text:
        return [""]

    if not isinstance(text, str):
        text = str(text)

    if max_line_length is None:
        return text.splitlines() or [""]

    if max_line_length <= 0:
        raise ValueError("max_line_length must be a positive integer")

    paragraphs = text.split("\n")
    all_lines: list[str] = []

    for paragraph in paragraphs:
        if paragraph == "":
            all_lines.append("")
            continue

        wrapped = textwrap.wrap(
            paragraph,
            width=max_line_length,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=True,
        )

        all_lines.extend(wrapped or [""])

    return all_lines or [""]


def validate_labels(labels: list[str] | None, detections: Detections) -> None:
    """
    Validates that the number of provided labels matches the number of detections.

    Args:
        labels: A list of labels, one for each detection. Can
            be None.
        detections: The detections to be labeled.

    Raises:
        ValueError: If `labels` is not None and its length does not match the number
            of detections.
    """
    if labels is not None and len(labels) != len(detections):
        raise ValueError(
            f"The number of labels ({len(labels)}) does not match the "
            f"number of detections ({len(detections)}). Each detection "
            f"should have exactly 1 label."
        )


def get_labels_text(
    detections: Detections, custom_labels: list[str] | None
) -> list[str]:
    """
    Retrieves the text labels for the detections.

    If `custom_labels` are provided, they are used. Otherwise, the labels are
    extracted from the `detections` object, prioritizing the 'class_name' field,
    then the `class_id`, and finally using the detection index as a string.

    Args:
        detections: The detections to get labels for.
        custom_labels: An optional list of custom labels.

    Returns:
        A list of text labels for each detection.
    """
    if custom_labels is not None:
        return custom_labels

    labels = []
    for idx in range(len(detections)):
        if CLASS_NAME_DATA_FIELD in detections.data:
            labels.append(str(detections.data[CLASS_NAME_DATA_FIELD][idx]))
        elif detections.class_id is not None:
            labels.append(str(detections.class_id[idx]))
        else:
            labels.append(str(idx))
    return labels


def snap_boxes(
    xyxy: np.ndarray[Any, np.dtype[np.float32]],
    resolution_wh: tuple[int, int],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """
    Shifts `label` bounding boxes into the frame so that they are fully contained
    within the given resolution, prioritizing the top/left edge.
    Unlike `clip_boxes`, this function does not crop boxes.
    It moves them entirely if they exceed the frame boundaries.

    Args:
        xyxy: A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in the format
            `(x_min, y_min, x_max, y_max)`.
        resolution_wh: A tuple `(width, height)`
            representing the resolution of the frame.

    Returns:
        A numpy array of shape `(N, 4)` with boxes shifted into frame.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.annotators.utils import snap_boxes
        >>> xyxy = np.array([
        ...     [-10, 10, 30, 50],     # Off left edge
        ...     [310, 200, 350, 250],  # Off right edge
        ...     [100, -20, 150, 30],   # Off top edge
        ...     [200, 220, 250, 270],  # Off bottom edge
        ...     [-20, 10, 350, 50],    # Wider than frame (370 vs 320)
        ...     [10, -20, 30, 260]     # Taller than frame (280 vs 240)
        ... ])
        >>> resolution_wh = (320, 240)
        >>> snapped_boxes = snap_boxes(xyxy=xyxy, resolution_wh=resolution_wh)
        >>> snapped_boxes
        array([[  0.,  10.,  40.,  50.],
               [280., 190., 320., 240.],
               [100.,   0., 150.,  50.],
               [200., 190., 250., 240.],
               [  0.,  10., 370.,  50.],
               [ 10.,   0.,  30., 280.]], dtype=float32)

        ```
    """
    result = np.copy(xyxy)
    width, height = resolution_wh

    # X-axis (prioritize left edge)
    left_overflow = result[:, 0] < 0
    result[left_overflow, 0:3:2] -= result[left_overflow, 0:1]

    right_overflow = (~left_overflow) & (result[:, 2] > width)
    right_shift = width - result[right_overflow, 2]
    result[right_overflow, 0:3:2] += right_shift[:, np.newaxis]

    # Y-axis (prioritize top edge)
    top_overflow = result[:, 1] < 0
    result[top_overflow, 1:4:2] -= result[top_overflow, 1:2]

    bottom_overflow = (~top_overflow) & (result[:, 3] > height)
    bottom_shift = height - result[bottom_overflow, 3]
    result[bottom_overflow, 1:4:2] += bottom_shift[:, np.newaxis]

    return result.astype(np.float32)  # type: ignore


class Trace:
    def __init__(
        self,
        max_size: int | None = None,
        start_frame_id: int = 0,
        anchor: Position = Position.CENTER,
    ) -> None:
        self.current_frame_id = start_frame_id
        self.max_size = max_size
        self.anchor = anchor

        self.frame_id = np.array([], dtype=int)
        self.xy = np.empty((0, 2), dtype=np.float32)
        self.tracker_id = np.array([], dtype=int)

    def put(self, detections: Detections) -> None:
        frame_id: npt.NDArray[np.int_] = np.full(
            len(detections), self.current_frame_id, dtype=int
        )
        self.frame_id = np.concatenate([self.frame_id, frame_id])
        self.xy = np.concatenate(
            [
                self.xy,
                detections.get_anchors_coordinates(self.anchor),
            ]
        )
        if detections.tracker_id is None:
            raise ValueError(
                "Could not put detections into Trace because "
                "Detections do not have tracker_id."
            )

        self.tracker_id = np.concatenate([self.tracker_id, detections.tracker_id])

        unique_frame_id = np.unique(self.frame_id)

        if self.max_size is not None and 0 < self.max_size < len(unique_frame_id):
            max_allowed_frame_id = self.current_frame_id - self.max_size + 1
            filtering_mask = self.frame_id >= max_allowed_frame_id
            self.frame_id = self.frame_id[filtering_mask]
            self.xy = self.xy[filtering_mask]
            self.tracker_id = self.tracker_id[filtering_mask]

        self.current_frame_id += 1

    def get(self, tracker_id: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        filtered: np.ndarray[Any, np.dtype[np.float32]] = (
            self.xy[self.tracker_id == tracker_id].copy().astype(np.float32, copy=False)
        )
        return filtered


def hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """
    Converts a hex color string (e.g. "#FF00FF" or "#FF00FF80") to an RGBA tuple.

    Args:
        hex_color: A hex color string.

    Returns:
        RGBA values in range 0-255.

    Raises:
        ValueError: If the format is invalid.
    """
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 6:
        hex_color += "FF"  # default full opacity
    if len(hex_color) != 8:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex digits in {hex_color}") from exc
    return (r, g, b, a)


def rgba_to_hex(rgba: tuple[int, int, int, int]) -> str:
    """
    Converts an RGBA tuple (0-255 each) to a hex color string.

    Args:
        rgba: RGBA values in range 0-255.

    Returns:
        Hex color string in the format "#RRGGBBAA".

    Raises:
        ValueError: If `rgba` is not a 4-tuple or contains values outside 0-255.
    """
    if len(rgba) != 4 or not all(0 <= c <= 255 for c in rgba):
        raise ValueError("RGBA must be a 4-tuple with values between 0-255.")
    return "#{:02X}{:02X}{:02X}{:02X}".format(*rgba)


def is_valid_hex(hex_color: str) -> bool:
    """
    Checks if a given string is a valid hex color.

    Args:
        hex_color: A hex color string with an optional leading "#". Supports
            6-digit (RGB) or 8-digit (RGBA) formats.

    Returns:
        True if the string is a valid 6- or 8-digit hex color, otherwise False.
    """
    return bool(re.fullmatch(r"#?[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?", hex_color.strip()))


def calculate_dynamic_kernel_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Computes a blur kernel size proportional to the shorter side of a bounding box.

    Args:
        x1: Left edge of the bounding box.
        y1: Top edge of the bounding box.
        x2: Right edge of the bounding box.
        y2: Bottom edge of the bounding box.

    Returns:
        Kernel size as one-third of the shorter dimension, minimum 1.

    Examples:
        ```pycon
        >>> calculate_dynamic_kernel_size(0, 0, 90, 60)
        20

        ```
    """
    return max(1, min(y2 - y1, x2 - x1) // 3)


def calculate_dynamic_pixel_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Computes a pixelation size proportional to the shorter side of a bounding box.

    Args:
        x1: Left edge of the bounding box.
        y1: Top edge of the bounding box.
        x2: Right edge of the bounding box.
        y2: Bottom edge of the bounding box.

    Returns:
        Pixel size as one-half of the shorter dimension, minimum 1.

    Examples:
        ```pycon
        >>> calculate_dynamic_pixel_size(0, 0, 90, 60)
        30

        ```
    """
    return max(1, min(y2 - y1, x2 - x1) // 2)
