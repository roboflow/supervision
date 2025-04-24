import textwrap
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


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
    def list(cls):
        return list(map(lambda c: c.value, cls))


def resolve_color_idx(
    detections: Detections,
    detection_idx: int,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
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
        return color_lookup[detection_idx]
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
        return detections.class_id[detection_idx]
    elif color_lookup == ColorLookup.TRACK:
        if detections.tracker_id is None:
            raise ValueError(
                "Could not resolve color by track because "
                "Detections do not have tracker_id. Did you call "
                "tracker.update_with_detections(...) before annotating?"
            )
        return detections.tracker_id[detection_idx]


def resolve_text_background_xyxy(
    center_coordinates: Tuple[int, int],
    text_wh: Tuple[int, int],
    position: Position,
) -> Tuple[int, int, int, int]:
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


def get_color_by_index(color: Union[Color, ColorPalette], idx: int) -> Color:
    if isinstance(color, ColorPalette):
        return color.by_idx(idx)
    return color


def resolve_color(
    color: Union[Color, ColorPalette],
    detections: Detections,
    detection_idx: int,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
) -> Color:
    idx = resolve_color_idx(
        detections=detections,
        detection_idx=detection_idx,
        color_lookup=color_lookup,
    )
    return get_color_by_index(color=color, idx=idx)


def wrap_text(text: str, max_line_length=None) -> list[str]:
    """
    Wraps text to the specified maximum line length, respecting existing newlines.
    Uses the textwrap library for robust text wrapping.

    Args:
        text (str): The text to wrap.

    Returns:
        List[str]: A list of text lines after wrapping.
    """

    if not text:
        return [""]

    if max_line_length is None:
        return text.splitlines() or [""]

    paragraphs = text.split("\n")
    all_lines = []

    for paragraph in paragraphs:
        if not paragraph:
            # Keep empty lines
            all_lines.append("")
            continue

        wrapped = textwrap.wrap(
            paragraph,
            width=max_line_length,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=True,
        )

        if wrapped:
            all_lines.extend(wrapped)
        else:
            all_lines.append("")

    return all_lines if all_lines else [""]


def validate_labels(labels: Optional[list[str]], detections: Detections):
    """
    Validates that the number of provided labels matches the number of detections.

    Args:
        labels (Optional[List[str]]): A list of labels, one for each detection. Can
                                        be None.
        detections (Detections): The detections to be labeled.

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
    detections: Detections, custom_labels: Optional[list[str]]
) -> list[str]:
    """
    Retrieves the text labels for the detections.

    If `custom_labels` are provided, they are used. Otherwise, the labels are
    extracted from the `detections` object, prioritizing the 'class_name' field,
    then the `class_id`, and finally using the detection index as a string.

    Args:
        detections (Detections): The detections to get labels for.
        custom_labels (Optional[List[str]]): An optional list of custom labels.

    Returns:
        List[str]: A list of text labels for each detection.
    """
    if custom_labels is not None:
        return custom_labels

    labels = []
    for idx in range(len(detections)):
        if CLASS_NAME_DATA_FIELD in detections.data:
            labels.append(detections.data[CLASS_NAME_DATA_FIELD][idx])
        elif detections.class_id is not None:
            labels.append(str(detections.class_id[idx]))
        else:
            labels.append(str(idx))
    return labels


def snap_boxes(xyxy: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """
    Shifts bounding boxes into the frame so that they are fully contained
    within the given resolution, prioritizing the top/left edge.
    Unlike `clip_boxes`, this function does not crop boxes.
    It moves them entirely if they exceed the frame boundaries.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in the format
            `(x_min, y_min, x_max, y_max)`.
        resolution_wh (Tuple[int, int]): A tuple `(width, height)`
            representing the resolution of the frame.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` with boxes shifted into frame.

    Examples:
        ```python
        import numpy as np
        # Assuming this function is part of a library like supervision (sv)
        # import supervision as sv # or just use the function directly

        xyxy = np.array([
            [-10, 10, 30, 50], # Off left
            [310, 200, 350, 250], # Off right
            [100, -20, 150, 30], # Off top
            [200, 220, 250, 270], # Off bottom
            [-20, 10, 350, 50], # Wider than frame, (width = 370 vs 320)
            [10, -20, 30, 260]  # Taller than frame, (height = 280 vs 240)
        ])

        resolution_wh = (320, 240)

        # Expected output for the new cases:
        # [-20, 10, 350, 50] (wider) -> shifted right by -(-20) = 20 -> [0, 10, 370, 50]
        # [10, -20, 30, 260] (taller) -> shifted down by -(-20) = 20 -> [10, 0, 30, 280]
        # Note: Oversized boxes still won't be fully contained without cropping
        # but this logic ensures the primary (top/left) boundary is corrected.

        snapped_boxes = snap_boxes(xyxy=xyxy, resolution_wh=resolution_wh)
        print(snapped_boxes)
        # Expected output (including original examples and new ones):
        # [[  0  10  40  50] # Original example 1 snapped
        #  [280 200 320 250] # Original example 2 snapped
        #  [100   0 150  50] # Original example 3 snapped
        #  [200 190 250 240] # Original example 4 snapped
        #  [  0  10 370  50] # New example (wider) snapped by left edge priority
        #  [ 10   0  30 280]] # New example (taller) snapped by top edge priority
        ```
    """
    result = np.copy(xyxy)
    width, height = resolution_wh

    shift_if_left_out = -result[:, 0]
    shift_if_right_out = width - result[:, 2]

    shift_x = np.where(result[:, 0] < 0, shift_if_left_out,
                       np.where(result[:, 2] > width, shift_if_right_out, 0))

    result[:, 0] += shift_x
    result[:, 2] += shift_x


    shift_if_top_out = -result[:, 1]
    shift_if_bottom_out = height - result[:, 3]

    shift_y = np.where(result[:, 1] < 0, shift_if_top_out,
                       np.where(result[:, 3] > height, shift_if_bottom_out, 0))


    result[:, 1] += shift_y
    result[:, 3] += shift_y

    return result


class Trace:
    def __init__(
        self,
        max_size: Optional[int] = None,
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
        frame_id = np.full(len(detections), self.current_frame_id, dtype=int)
        self.frame_id = np.concatenate([self.frame_id, frame_id])
        self.xy = np.concatenate([
            self.xy,
            detections.get_anchors_coordinates(self.anchor),
        ])
        self.tracker_id = np.concatenate([self.tracker_id, detections.tracker_id])

        unique_frame_id = np.unique(self.frame_id)

        if 0 < self.max_size < len(unique_frame_id):
            max_allowed_frame_id = self.current_frame_id - self.max_size + 1
            filtering_mask = self.frame_id >= max_allowed_frame_id
            self.frame_id = self.frame_id[filtering_mask]
            self.xy = self.xy[filtering_mask]
            self.tracker_id = self.tracker_id[filtering_mask]

        self.current_frame_id += 1

    def get(self, tracker_id: int) -> np.ndarray:
        return self.xy[self.tracker_id == tracker_id]
