import importlib.metadata as importlib_metadata
from typing import List

import numpy as np

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from supervision.annotators.core import (
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    ComparisonAnnotator,
    CropAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    IconAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PixelateAnnotator,
    PolygonAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.classification.core import Classifications
from supervision.dataset.core import (
    BaseDataset,
    ClassificationDataset,
    DetectionDataset,
)
from supervision.dataset.utils import mask_to_rle, rle_to_mask
from supervision.detection.core import Detections
from supervision.detection.line_zone import (
    LineZone,
    LineZoneAnnotator,
    LineZoneAnnotatorMulticlass,
)
from supervision.detection.overlap_filter import (
    OverlapFilter,
    box_non_max_merge,
    box_non_max_suppression,
    mask_non_max_suppression,
)
from supervision.detection.tools.csv_sink import CSVSink
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.tools.json_sink import JSONSink
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.detection.tools.smoother import DetectionsSmoother
from supervision.detection.utils import (
    box_iou_batch,
    calculate_masks_centroids,
    clip_boxes,
    contains_holes,
    contains_multiple_segments,
    filter_polygons_by_area,
    mask_iou_batch,
    mask_to_polygons,
    mask_to_xyxy,
    move_boxes,
    move_masks,
    oriented_box_iou_batch,
    pad_boxes,
    polygon_to_mask,
    polygon_to_xyxy,
    scale_boxes,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_polygons,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)
from supervision.detection.vlm import LMM, VLM
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import (
    calculate_optimal_line_thickness,
    calculate_optimal_text_scale,
    draw_filled_polygon,
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_text,
)
from supervision.geometry.core import Point, Position, Rect
from supervision.geometry.utils import get_polygon_center
from supervision.keypoint.annotators import (
    EdgeAnnotator,
    VertexAnnotator,
    VertexLabelAnnotator,
)
from supervision.keypoint.core import KeyPoints
from supervision.metrics.detection import ConfusionMatrix, MeanAveragePrecision
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.utils.conversion import cv2_to_pillow, pillow_to_cv2
from supervision.utils.file import list_files_with_extensions
from supervision.utils.image import (
    ImageSink,
    create_tiles,
    crop_image,
    letterbox_image,
    overlay_image,
    resize_image,
    scale_image,
)
from supervision.utils.notebook import plot_image, plot_images_grid
from supervision.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)

__all__ = [
    "LMM",
    "BackgroundOverlayAnnotator",
    "BaseDataset",
    "BlurAnnotator",
    "BoxAnnotator",
    "BoxCornerAnnotator",
    "ByteTrack",
    "CSVSink",
    "CircleAnnotator",
    "ClassificationDataset",
    "Classifications",
    "Color",
    "ColorAnnotator",
    "ColorLookup",
    "ColorPalette",
    "ComparisonAnnotator",
    "ConfusionMatrix",
    "CropAnnotator",
    "DetectionDataset",
    "Detections",
    "DetectionsSmoother",
    "DotAnnotator",
    "EdgeAnnotator",
    "EllipseAnnotator",
    "FPSMonitor",
    "HaloAnnotator",
    "HeatMapAnnotator",
    "IconAnnotator",
    "ImageSink",
    "InferenceSlicer",
    "JSONSink",
    "KeyPoints",
    "LabelAnnotator",
    "LineZone",
    "LineZoneAnnotator",
    "LineZoneAnnotatorMulticlass",
    "MaskAnnotator",
    "MeanAveragePrecision",
    "OrientedBoxAnnotator",
    "OverlapFilter",
    "PercentageBarAnnotator",
    "PixelateAnnotator",
    "Point",
    "PolygonAnnotator",
    "PolygonZone",
    "PolygonZoneAnnotator",
    "Position",
    "Rect",
    "RichLabelAnnotator",
    "RoundBoxAnnotator",
    "TraceAnnotator",
    "TriangleAnnotator",
    "VertexAnnotator",
    "VertexLabelAnnotator",
    "VideoInfo",
    "VideoSink",
    "box_iou_batch",
    "box_iou_batch_with_jaccard",
    "box_non_max_merge",
    "box_non_max_suppression",
    "calculate_masks_centroids",
    "calculate_optimal_line_thickness",
    "calculate_optimal_text_scale",
    "clip_boxes",
    "contains_holes",
    "contains_multiple_segments",
    "create_tiles",
    "crop_image",
    "cv2_to_pillow",
    "draw_filled_polygon",
    "draw_filled_rectangle",
    "draw_image",
    "draw_line",
    "draw_polygon",
    "draw_rectangle",
    "draw_text",
    "filter_polygons_by_area",
    "get_polygon_center",
    "get_video_frames_generator",
    "letterbox_image",
    "list_files_with_extensions",
    "mask_iou_batch",
    "mask_non_max_suppression",
    "mask_to_polygons",
    "mask_to_rle",
    "mask_to_xyxy",
    "move_boxes",
    "move_masks",
    "oriented_box_iou_batch",
    "overlay_image",
    "pad_boxes",
    "pillow_to_cv2",
    "plot_image",
    "plot_images_grid",
    "polygon_to_mask",
    "polygon_to_xyxy",
    "process_video",
    "resize_image",
    "rle_to_mask",
    "scale_boxes",
    "scale_image",
    "xcycwh_to_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_polygons",
    "xyxy_to_xyah",
    "xyxy_to_xywh",
]


def _jaccard(box_a: List[float], box_b: List[float], is_crowd: bool) -> float:
    """
    Calculate the Jaccard index (intersection over union) between two bounding boxes.
    If a gt object is marked as "iscrowd", a dt is allowed to match any subregion
    of the gt. Choosing gt' in the crowd gt that best matches the dt can be done using
    gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
    iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)

    Args:
        box_a (List[float]): Box coordinates in the format [x, y, width, height].
        box_b (List[float]): Box coordinates in the format [x, y, width, height].
        iscrowd (bool): Flag indicating if the second box is a crowd region or not.

    Returns:
        float: Jaccard index between the two bounding boxes.
    """
    # Smallest number to avoid division by zero
    EPS = np.spacing(1)

    xa, ya, x2a, y2a = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    xb, yb, x2b, y2b = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    # Innermost left x
    xi = max(xa, xb)
    # Innermost right x
    x2i = min(x2a, x2b)
    # Same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # Calculate areas
    Aa = max(x2a - xa, 0.0) * max(y2a - ya, 0.0)
    Ab = max(x2b - xb, 0.0) * max(y2b - yb, 0.0)
    Ai = max(x2i - xi, 0.0) * max(y2i - yi, 0.0)

    if is_crowd:
        return Ai / (Aa + EPS)

    return Ai / (Aa + Ab - Ai + EPS)


def box_iou_batch_with_jaccard(
    boxes_true: List[List[float]],
    boxes_detection: List[List[float]],
    is_crowd: List[bool],
) -> np.ndarray:
    """
    Calculate the intersection over union (IoU) between detection bounding boxes (dt)
    and ground-truth bounding boxes (gt).
    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics

    Args:
        boxes_true (List[List[float]]): List of ground-truth bounding boxes in the \
            format [x, y, width, height].
        boxes_detection (List[List[float]]): List of detection bounding boxes in the \
            format [x, y, width, height].
        is_crowd (List[bool]): List indicating if each ground-truth bounding box \
            is a crowd region or not.

    Returns:
        np.ndarray: Array of IoU values of shape (len(dt), len(gt)).

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        boxes_true = [
            [10, 20, 30, 40],  # x, y, w, h
            [15, 25, 35, 45]
        ]
        boxes_detection = [
            [12, 22, 28, 38],
            [16, 26, 36, 46]
        ]
        is_crowd = [False, False]

        ious = sv.box_iou_batch_with_jaccard(
            boxes_true=boxes_true,
            boxes_detection=boxes_detection,
            is_crowd=is_crowd
        )
        # array([
        #     [0.8866..., 0.4960...],
        #     [0.4000..., 0.8622...]
        # ])
        ```
    """
    assert len(is_crowd) == len(boxes_true), (
        "iou(iscrowd=) must have the same length as boxes_true"
    )
    if len(boxes_detection) == 0 or len(boxes_true) == 0:
        return np.array([])
    ious = np.zeros((len(boxes_detection), len(boxes_true)), dtype=np.float64)
    for g_idx, g in enumerate(boxes_true):
        for d_idx, d in enumerate(boxes_detection):
            ious[d_idx, g_idx] = _jaccard(d, g, is_crowd[g_idx])
    return ious
