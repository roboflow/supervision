import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from supervision.annotators.core import (
    BlurAnnotator,
    BoundingBoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    CropAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
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
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZone, LineZoneAnnotator
from supervision.detection.tools.csv_sink import CSVSink
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.tools.json_sink import JSONSink
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.detection.tools.smoother import DetectionsSmoother
from supervision.detection.utils import (
    box_iou_batch,
    box_non_max_suppression,
    calculate_masks_centroids,
    clip_boxes,
    filter_polygons_by_area,
    mask_iou_batch,
    mask_non_max_suppression,
    mask_to_polygons,
    mask_to_xyxy,
    move_boxes,
    move_masks,
    pad_boxes,
    polygon_to_mask,
    polygon_to_xyxy,
    scale_boxes,
)
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import (
    calculate_optimal_line_thickness,
    calculate_optimal_text_scale,
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
