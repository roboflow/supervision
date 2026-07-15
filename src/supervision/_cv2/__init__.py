"""Private OpenCV compatibility surface used by Supervision."""

from __future__ import annotations

from typing import NoReturn


class BackendUnavailableError(RuntimeError):
    """Raised when an OpenCV operation is used without an available backend."""


try:
    import cv2
except (ImportError, OSError):
    _IS_CV2_AVAILABLE = False
else:
    _IS_CV2_AVAILABLE = True

_BORDER_CONSTANT = 0
_CAP_PROP_FPS = 5
_CAP_PROP_FRAME_COUNT = 7
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_POS_FRAMES = 1
_CC_STAT_AREA = 4
_CHAIN_APPROX_SIMPLE = 2
_COLOR_BGR2GRAY = 6
_COLOR_BGR2RGB = 4
_COLOR_GRAY2BGR = 8
_COLOR_HSV2BGR = 54
_COLOR_RGB2BGR = 4
_DIST_L2 = 2
_FONT_HERSHEY_SIMPLEX = 0
_IMREAD_COLOR = 1
_IMREAD_UNCHANGED = -1
_INTER_LINEAR = 1
_INTER_NEAREST = 0
_LINE_4 = 4
_LINE_AA = 16
_RETR_CCOMP = 2
_RETR_TREE = 3

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
        FONT_HERSHEY_SIMPLEX,
        IMREAD_COLOR,
        IMREAD_UNCHANGED,
        INTER_LINEAR,
        INTER_NEAREST,
        LINE_4,
        LINE_AA,
        RETR_CCOMP,
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
    FONT_HERSHEY_SIMPLEX = _FONT_HERSHEY_SIMPLEX
    IMREAD_COLOR = _IMREAD_COLOR
    IMREAD_UNCHANGED = _IMREAD_UNCHANGED
    INTER_LINEAR = _INTER_LINEAR
    INTER_NEAREST = _INTER_NEAREST
    LINE_4 = _LINE_4
    LINE_AA = _LINE_AA
    RETR_CCOMP = _RETR_CCOMP
    RETR_TREE = _RETR_TREE

    def _unavailable(*args: object, **kwargs: object) -> NoReturn:
        """Fail clearly until the corresponding fallback is implemented."""
        del args, kwargs
        raise BackendUnavailableError(
            "OpenCV is not installed and this operation has no fallback yet."
        )

    VideoCapture = _unavailable
    VideoWriter = _unavailable
    VideoWriter_fourcc = _unavailable
    addWeighted = _unavailable
    approxPolyDP = _unavailable
    blur = _unavailable
    circle = _unavailable
    connectedComponents = _unavailable
    connectedComponentsWithStats = _unavailable
    contourArea = _unavailable
    convertScaleAbs = _unavailable
    copyMakeBorder = _unavailable
    cvtColor = _unavailable
    distanceTransform = _unavailable
    drawContours = _unavailable
    ellipse = _unavailable
    fillPoly = _unavailable
    findContours = _unavailable
    flip = _unavailable
    getRotationMatrix2D = _unavailable
    getTextSize = _unavailable
    imread = _unavailable
    imwrite = _unavailable
    intersectConvexConvex = _unavailable
    line = _unavailable
    mean = _unavailable
    merge = _unavailable
    polylines = _unavailable
    putText = _unavailable
    rectangle = _unavailable
    resize = _unavailable
    split = _unavailable
    warpAffine = _unavailable


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
    "FONT_HERSHEY_SIMPLEX",
    "IMREAD_COLOR",
    "IMREAD_UNCHANGED",
    "INTER_LINEAR",
    "INTER_NEAREST",
    "LINE_4",
    "LINE_AA",
    "RETR_CCOMP",
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
