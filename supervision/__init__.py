import importlib.metadata as importlib_metadata

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
    SplineAnnotator,
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
from supervision.dataset.formats.coco import get_coco_class_index_mapping
from supervision.dataset.utils import mask_to_rle, rle_to_mask
from supervision.detection.core import Detections
from supervision.detection.line_zone import (
    LineZone,
    LineZoneAnnotator,
    LineZoneAnnotatorMulticlass,
)
from supervision.detection.tools.csv_sink import CSVSink
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.tools.json_sink import JSONSink
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.detection.tools.smoother import DetectionsSmoother
from supervision.detection.utils.boxes import (
    clip_boxes,
    denormalize_boxes,
    move_boxes,
    pad_boxes,
    scale_boxes,
)
from supervision.detection.utils.converters import (
    mask_to_polygons,
    mask_to_xyxy,
    polygon_to_mask,
    polygon_to_xyxy,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_polygons,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)
from supervision.detection.utils.iou_and_nms import (
    OverlapFilter,
    OverlapMetric,
    box_iou,
    box_iou_batch,
    box_iou_batch_with_jaccard,
    box_non_max_merge,
    box_non_max_suppression,
    mask_iou_batch,
    mask_non_max_merge,
    mask_non_max_suppression,
    oriented_box_iou_batch,
)
from supervision.detection.utils.masks import (
    calculate_masks_centroids,
    contains_holes,
    contains_multiple_segments,
    move_masks,
)
from supervision.detection.utils.polygons import (
    approximate_polygon,
    filter_polygons_by_area,
)
from supervision.detection.utils.vlms import edit_distance, fuzzy_match_index
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
from supervision.key_points.annotators import (
    EdgeAnnotator,
    VertexAnnotator,
    VertexLabelAnnotator,
)
from supervision.key_points.core import KeyPoints
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
    "OverlapMetric",
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
    "approximate_polygon",
    "box_iou",
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
    "edit_distance",
    "filter_polygons_by_area",
    "fuzzy_match_index",
    "get_coco_class_index_mapping",
    "get_polygon_center",
    "get_video_frames_generator",
    "letterbox_image",
    "list_files_with_extensions",
    "mask_iou_batch",
    "mask_non_max_merge",
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
    "xyxy_to_xcycarh",
    "xyxy_to_xywh",
]
