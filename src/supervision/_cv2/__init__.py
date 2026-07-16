"""Private OpenCV compatibility surface used by Supervision."""

from __future__ import annotations

from supervision._cv2._color import _cvt_color, _merge, _split
from supervision._cv2._common import BackendUnavailableError, _unavailable
from supervision._cv2._components import (
    _connected_components,
    _connected_components_with_stats,
)
from supervision._cv2._contours import _find_contours
from supervision._cv2._drawing import (
    _circle,
    _draw_contours,
    _ellipse,
    _fill_poly,
    _line,
    _polylines,
    _rectangle,
)
from supervision._cv2._geometry import (
    _approx_poly_dp,
    _contour_area,
    _intersect_convex_convex,
)
from supervision._cv2._image import (
    _add_weighted,
    _convert_scale_abs,
    _copy_make_border,
    _flip,
    _imread,
    _imwrite,
    _mean,
    _resize,
)
from supervision._cv2._text import _get_text_size, _put_text
from supervision._cv2._transform import (
    _blur,
    _distance_transform,
    _get_rotation_matrix_2d,
    _warp_affine,
)
from supervision._cv2.constants import (
    _BORDER_CONSTANT,
    _CAP_PROP_FPS,
    _CAP_PROP_FRAME_COUNT,
    _CAP_PROP_FRAME_HEIGHT,
    _CAP_PROP_FRAME_WIDTH,
    _CAP_PROP_POS_FRAMES,
    _CC_STAT_AREA,
    _CHAIN_APPROX_SIMPLE,
    _COLOR_BGR2GRAY,
    _COLOR_BGR2RGB,
    _COLOR_GRAY2BGR,
    _COLOR_HSV2BGR,
    _COLOR_RGB2BGR,
    _DIST_L2,
    _FONT_HERSHEY_COMPLEX,
    _FONT_HERSHEY_COMPLEX_SMALL,
    _FONT_HERSHEY_DUPLEX,
    _FONT_HERSHEY_PLAIN,
    _FONT_HERSHEY_SCRIPT_COMPLEX,
    _FONT_HERSHEY_SCRIPT_SIMPLEX,
    _FONT_HERSHEY_SIMPLEX,
    _FONT_HERSHEY_TRIPLEX,
    _FONT_ITALIC,
    _IMREAD_COLOR,
    _IMREAD_UNCHANGED,
    _INTER_LINEAR,
    _INTER_NEAREST,
    _LINE_4,
    _LINE_8,
    _LINE_AA,
    _RETR_TREE,
)

try:
    import cv2
except (ImportError, OSError):
    _IS_CV2_AVAILABLE = False
else:
    _IS_CV2_AVAILABLE = True

if _IS_CV2_AVAILABLE:
    from cv2 import (
        BORDER_CONSTANT,
        CAP_PROP_FPS,
        CAP_PROP_FRAME_COUNT,
        CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FRAME_WIDTH,
        CAP_PROP_POS_FRAMES,
        CC_STAT_AREA,
        CHAIN_APPROX_SIMPLE,
        COLOR_BGR2GRAY,
        COLOR_BGR2RGB,
        COLOR_GRAY2BGR,
        COLOR_HSV2BGR,
        COLOR_RGB2BGR,
        DIST_L2,
        FONT_HERSHEY_COMPLEX,
        FONT_HERSHEY_COMPLEX_SMALL,
        FONT_HERSHEY_DUPLEX,
        FONT_HERSHEY_PLAIN,
        FONT_HERSHEY_SCRIPT_COMPLEX,
        FONT_HERSHEY_SCRIPT_SIMPLEX,
        FONT_HERSHEY_SIMPLEX,
        FONT_HERSHEY_TRIPLEX,
        FONT_ITALIC,
        IMREAD_COLOR,
        IMREAD_UNCHANGED,
        INTER_LINEAR,
        INTER_NEAREST,
        LINE_4,
        LINE_8,
        LINE_AA,
        RETR_TREE,
        VideoCapture,
        VideoWriter,
        VideoWriter_fourcc,
        addWeighted,
        approxPolyDP,
        blur,
        circle,
        connectedComponents,
        connectedComponentsWithStats,
        contourArea,
        convertScaleAbs,
        copyMakeBorder,
        cvtColor,
        distanceTransform,
        drawContours,
        ellipse,
        fillPoly,
        findContours,
        flip,
        getRotationMatrix2D,
        getTextSize,
        imread,
        imwrite,
        intersectConvexConvex,
        line,
        mean,
        merge,
        polylines,
        putText,
        rectangle,
        resize,
        split,
        warpAffine,
    )

    BACKEND_NAME = "opencv"
else:
    BACKEND_NAME = "fallback"

    BORDER_CONSTANT = _BORDER_CONSTANT
    CAP_PROP_FPS = _CAP_PROP_FPS
    CAP_PROP_FRAME_COUNT = _CAP_PROP_FRAME_COUNT
    CAP_PROP_FRAME_HEIGHT = _CAP_PROP_FRAME_HEIGHT
    CAP_PROP_FRAME_WIDTH = _CAP_PROP_FRAME_WIDTH
    CAP_PROP_POS_FRAMES = _CAP_PROP_POS_FRAMES
    CC_STAT_AREA = _CC_STAT_AREA
    CHAIN_APPROX_SIMPLE = _CHAIN_APPROX_SIMPLE
    COLOR_BGR2GRAY = _COLOR_BGR2GRAY
    COLOR_BGR2RGB = _COLOR_BGR2RGB
    COLOR_GRAY2BGR = _COLOR_GRAY2BGR
    COLOR_HSV2BGR = _COLOR_HSV2BGR
    COLOR_RGB2BGR = _COLOR_RGB2BGR
    DIST_L2 = _DIST_L2
    FONT_HERSHEY_COMPLEX = _FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_COMPLEX_SMALL = _FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_DUPLEX = _FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_PLAIN = _FONT_HERSHEY_PLAIN
    FONT_HERSHEY_SCRIPT_COMPLEX = _FONT_HERSHEY_SCRIPT_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = _FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SIMPLEX = _FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_TRIPLEX = _FONT_HERSHEY_TRIPLEX
    FONT_ITALIC = _FONT_ITALIC
    IMREAD_COLOR = _IMREAD_COLOR
    IMREAD_UNCHANGED = _IMREAD_UNCHANGED
    INTER_LINEAR = _INTER_LINEAR
    INTER_NEAREST = _INTER_NEAREST
    LINE_4 = _LINE_4
    LINE_8 = _LINE_8
    LINE_AA = _LINE_AA
    RETR_TREE = _RETR_TREE

    VideoCapture = _unavailable
    VideoWriter = _unavailable
    VideoWriter_fourcc = _unavailable
    addWeighted = _add_weighted
    approxPolyDP = _approx_poly_dp
    blur = _blur
    circle = _circle
    connectedComponents = _connected_components
    connectedComponentsWithStats = _connected_components_with_stats
    contourArea = _contour_area
    convertScaleAbs = _convert_scale_abs
    copyMakeBorder = _copy_make_border
    cvtColor = _cvt_color
    distanceTransform = _distance_transform
    drawContours = _draw_contours
    ellipse = _ellipse
    fillPoly = _fill_poly
    findContours = _find_contours
    flip = _flip
    getRotationMatrix2D = _get_rotation_matrix_2d
    getTextSize = _get_text_size
    imread = _imread
    imwrite = _imwrite
    intersectConvexConvex = _intersect_convex_convex
    line = _line
    mean = _mean
    merge = _merge
    polylines = _polylines
    putText = _put_text
    rectangle = _rectangle
    resize = _resize
    split = _split
    warpAffine = _warp_affine


__all__ = [
    "BACKEND_NAME",
    "BORDER_CONSTANT",
    "CAP_PROP_FPS",
    "CAP_PROP_FRAME_COUNT",
    "CAP_PROP_FRAME_HEIGHT",
    "CAP_PROP_FRAME_WIDTH",
    "CAP_PROP_POS_FRAMES",
    "CC_STAT_AREA",
    "CHAIN_APPROX_SIMPLE",
    "COLOR_BGR2GRAY",
    "COLOR_BGR2RGB",
    "COLOR_GRAY2BGR",
    "COLOR_HSV2BGR",
    "COLOR_RGB2BGR",
    "DIST_L2",
    "FONT_HERSHEY_COMPLEX",
    "FONT_HERSHEY_COMPLEX_SMALL",
    "FONT_HERSHEY_DUPLEX",
    "FONT_HERSHEY_PLAIN",
    "FONT_HERSHEY_SCRIPT_COMPLEX",
    "FONT_HERSHEY_SCRIPT_SIMPLEX",
    "FONT_HERSHEY_SIMPLEX",
    "FONT_HERSHEY_TRIPLEX",
    "FONT_ITALIC",
    "IMREAD_COLOR",
    "IMREAD_UNCHANGED",
    "INTER_LINEAR",
    "INTER_NEAREST",
    "LINE_4",
    "LINE_8",
    "LINE_AA",
    "RETR_TREE",
    "BackendUnavailableError",
    "VideoCapture",
    "VideoWriter",
    "VideoWriter_fourcc",
    "addWeighted",
    "approxPolyDP",
    "blur",
    "circle",
    "connectedComponents",
    "connectedComponentsWithStats",
    "contourArea",
    "convertScaleAbs",
    "copyMakeBorder",
    "cvtColor",
    "distanceTransform",
    "drawContours",
    "ellipse",
    "fillPoly",
    "findContours",
    "flip",
    "getRotationMatrix2D",
    "getTextSize",
    "imread",
    "imwrite",
    "intersectConvexConvex",
    "line",
    "mean",
    "merge",
    "polylines",
    "putText",
    "rectangle",
    "resize",
    "split",
    "warpAffine",
]
