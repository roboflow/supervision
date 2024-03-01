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
    filter_polygons_by_area,
    mask_iou_batch,
    mask_non_max_suppression,
    mask_to_polygons,
    mask_to_xyxy,
    move_boxes,
    polygon_to_mask,
    polygon_to_xyxy,
    scale_boxes,
)
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import (
    calculate_dynamic_line_thickness,
    calculate_dynamic_text_scale,
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_text,
)
from supervision.geometry.core import Point, Position, Rect
from supervision.geometry.utils import get_polygon_center
from supervision.metrics.detection import ConfusionMatrix, MeanAveragePrecision
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.utils.file import list_files_with_extensions
from supervision.utils.image import ImageSink, crop_image
from supervision.utils.notebook import plot_image, plot_images_grid
from supervision.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)
