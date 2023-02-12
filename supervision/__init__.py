__version__ = "0.2.1"

from supervision.detection.core import BoxAnnotator, Detections
from supervision.detection.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.detection.utils import generate_2d_mask
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import draw_filled_rectangle, draw_polygon, draw_text
from supervision.geometry.core import Point, Position, Rect
from supervision.geometry.utils import get_polygon_center
from supervision.notebook.utils import show_frame_in_notebook
from supervision.video import (
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)
