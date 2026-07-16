"""Private OpenCV compatibility surface used by Supervision."""

from __future__ import annotations

from supervision._cv2._color import _cvt_color, _merge, _split
from supervision._cv2._common import BackendUnavailableError
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
from supervision._cv2._video import (
    _video_writer_fourcc,
    _VideoCapture,
    _VideoWriter,
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
    from cv2 import (  # type: ignore[attr-defined]
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
        VideoWriter_fourcc,  # type: ignore[attr-defined]
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

    # Fallback implementations when cv2 is not available. Suppress type errors because
    # fallback types differ from cv2 types, but are functionally equivalent.
    VideoCapture = _VideoCapture  # type: ignore[assignment,misc]
    VideoWriter = _VideoWriter  # type: ignore[assignment,misc]
    VideoWriter_fourcc = _video_writer_fourcc  # type: ignore[assignment]
    addWeighted = _add_weighted  # type: ignore[assignment]
    approxPolyDP = _approx_poly_dp  # type: ignore[assignment]
    blur = _blur  # type: ignore[assignment]
    circle = _circle  # type: ignore[assignment]
    connectedComponents = _connected_components  # type: ignore[assignment]
    connectedComponentsWithStats = _connected_components_with_stats  # type: ignore[assignment]
    contourArea = _contour_area  # type: ignore[assignment]
    convertScaleAbs = _convert_scale_abs  # type: ignore[assignment]
    copyMakeBorder = _copy_make_border  # type: ignore[assignment]
    cvtColor = _cvt_color  # type: ignore[assignment]
    distanceTransform = _distance_transform  # type: ignore[assignment]
    drawContours = _draw_contours  # type: ignore[assignment]
    ellipse = _ellipse  # type: ignore[assignment]
    fillPoly = _fill_poly  # type: ignore[assignment]
    findContours = _find_contours  # type: ignore[assignment]
    flip = _flip  # type: ignore[assignment]
    getRotationMatrix2D = _get_rotation_matrix_2d  # type: ignore[assignment]
    getTextSize = _get_text_size  # type: ignore[assignment]
    imread = _imread  # type: ignore[assignment]
    imwrite = _imwrite  # type: ignore[assignment]
    intersectConvexConvex = _intersect_convex_convex  # type: ignore[assignment]
    line = _line  # type: ignore[assignment]
    mean = _mean  # type: ignore[assignment]
    merge = _merge  # type: ignore[assignment]
    polylines = _polylines  # type: ignore[assignment]
    putText = _put_text  # type: ignore[assignment]
    rectangle = _rectangle  # type: ignore[assignment]
    resize = _resize  # type: ignore[assignment]
    split = _split  # type: ignore[assignment]
    warpAffine = _warp_affine  # type: ignore[assignment]


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
