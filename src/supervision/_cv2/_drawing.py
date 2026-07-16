"""Private Pillow-based drawing fallbacks for the OpenCV facade."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

_ImageArray = npt.NDArray[Any]
_Point = tuple[int, int]


def _drawing_mask(
    image: _ImageArray, draw_operation: Callable[[Any], None]
) -> npt.NDArray[np.bool_]:
    """Render a shape into a clipped one-bit mask with Pillow."""
    height, width = image.shape[:2]
    mask = Image.new("1", (width, height))
    draw_operation(ImageDraw.Draw(mask))
    return np.asarray(mask, dtype=bool)


def _color_for_image(image: _ImageArray, color: Any) -> Any:
    """Normalize an OpenCV scalar color to the image channel count."""
    values = np.asarray(color).reshape(-1)
    if image.ndim == 2:
        return values[0] if values.size else 0
    channels = image.shape[2]
    if values.size == 0:
        return np.zeros(channels, dtype=image.dtype)
    if values.size < channels:
        values = np.pad(values, (0, channels - values.size))
    return values[:channels]


def _paint(image: _ImageArray, mask: npt.NDArray[np.bool_], color: Any) -> _ImageArray:
    """Apply a scalar or multi-channel color to a drawing mask."""
    image[mask] = _color_for_image(image, color)
    return image


def _point(point: Sequence[int | float]) -> _Point:
    """Convert an OpenCV point to integer Pillow coordinates."""
    return round(point[0]), round(point[1])


def _points(points: npt.NDArray[Any], offset: tuple[int, int] = (0, 0)) -> list[_Point]:
    """Normalize OpenCV polygon shapes to integer Pillow coordinates."""
    values = np.asarray(points)
    if values.size == 0:
        return []
    if values.ndim not in (2, 3) or values.shape[-1] != 2:
        raise ValueError("Drawing points must have shape (N, 2) or (N, 1, 2)")
    normalized = np.rint(values.reshape(-1, 2)).astype(np.int64)
    normalized += np.asarray(offset, dtype=np.int64)
    return [(int(x), int(y)) for x, y in normalized]


def _validate_shift(shift: int) -> None:
    """Reject fixed-point coordinates not supported by the fallback."""
    if shift != 0:
        raise ValueError("Only unshifted drawing coordinates are supported")


def _line(
    img: _ImageArray,
    pt1: Sequence[int | float],
    pt2: Sequence[int | float],
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw a line in place using Pillow's integer rasterization."""
    del lineType
    _validate_shift(shift)
    width = max(1, thickness)
    mask = _drawing_mask(
        img,
        lambda draw: draw.line([_point(pt1), _point(pt2)], fill=1, width=width),
    )
    return _paint(img, mask, color)


def _rectangle(
    img: _ImageArray,
    pt1: Sequence[int | float],
    pt2: Sequence[int | float],
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw or fill an inclusive-axis-aligned rectangle in place."""
    del lineType
    _validate_shift(shift)
    first, second = _point(pt1), _point(pt2)
    if thickness < 0:
        mask = _drawing_mask(img, lambda draw: draw.rectangle([first, second], fill=1))
    else:
        width = max(1, thickness)
        mask = _drawing_mask(
            img,
            lambda draw: draw.rectangle([first, second], outline=1, width=width),
        )
    return _paint(img, mask, color)


def _circle(
    img: _ImageArray,
    center: Sequence[int | float],
    radius: int,
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw or fill a circle in place."""
    del lineType
    _validate_shift(shift)
    x, y = _point(center)
    bounds = [x - radius, y - radius, x + radius, y + radius]
    if thickness < 0:
        mask = _drawing_mask(img, lambda draw: draw.ellipse(bounds, fill=1))
    else:
        width = max(1, thickness)
        mask = _drawing_mask(
            img,
            lambda draw: draw.ellipse(bounds, outline=1, width=width),
        )
    return _paint(img, mask, color)


def _ellipse_points(
    center: Sequence[int | float],
    axes: Sequence[int | float],
    angle: float,
    start_angle: float,
    end_angle: float,
) -> list[_Point]:
    """Sample an OpenCV ellipse arc as integer image coordinates."""
    center_x, center_y = center
    axis_x, axis_y = axes
    span = max(0.0, end_angle - start_angle)
    sample_count = max(2, int(np.ceil(span * max(axis_x, axis_y) * np.pi / 90)))
    angles = np.linspace(start_angle, end_angle, sample_count + 1)
    radians = np.deg2rad(angles)
    rotation = np.deg2rad(angle)
    cosine, sine = np.cos(rotation), np.sin(rotation)
    x = center_x + axis_x * np.cos(radians) * cosine - axis_y * np.sin(radians) * sine
    y = center_y + axis_x * np.cos(radians) * sine + axis_y * np.sin(radians) * cosine
    return [(round(x_value), round(y_value)) for x_value, y_value in zip(x, y)]


def _ellipse(
    img: _ImageArray,
    center: Sequence[int | float],
    axes: Sequence[int | float],
    angle: float,
    startAngle: float,
    endAngle: float,
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw or fill an optionally rotated ellipse arc in place."""
    del lineType
    _validate_shift(shift)
    ellipse_points = _ellipse_points(center, axes, angle, startAngle, endAngle)
    is_full = endAngle - startAngle >= 360

    def draw_ellipse(draw: Any) -> None:
        if thickness < 0:
            if is_full:
                draw.polygon(ellipse_points, fill=1)
            else:
                draw.polygon([_point(center), *ellipse_points], fill=1)
            return
        width = max(1, thickness)
        draw.line(ellipse_points, fill=1, width=width, joint="curve")
        if is_full and len(ellipse_points) > 1:
            draw.line([ellipse_points[-1], ellipse_points[0]], fill=1, width=width)

    return _paint(img, _drawing_mask(img, draw_ellipse), color)


def _polylines(
    img: _ImageArray,
    pts: Sequence[npt.NDArray[Any]],
    isClosed: bool,
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw one or more open or closed polylines in place."""
    del lineType
    _validate_shift(shift)
    width = max(1, thickness)

    def draw_polylines(draw: Any) -> None:
        for polygon in pts:
            points = _points(polygon)
            if len(points) < 2:
                continue
            if isClosed:
                points.append(points[0])
            draw.line(points, fill=1, width=width, joint="curve")

    return _paint(img, _drawing_mask(img, draw_polylines), color)


def _fill_poly(
    img: _ImageArray,
    pts: Sequence[npt.NDArray[Any]],
    color: Any,
    lineType: int = 8,
    shift: int = 0,
    offset: tuple[int, int] = (0, 0),
) -> _ImageArray:
    """Fill one or more polygons in place."""
    del lineType
    _validate_shift(shift)

    def draw_polygons(draw: Any) -> None:
        for polygon in pts:
            points = _points(polygon, offset=offset)
            if len(points) >= 3:
                draw.polygon(points, fill=1)

    return _paint(img, _drawing_mask(img, draw_polygons), color)


def _draw_contours(
    image: _ImageArray,
    contours: Sequence[npt.NDArray[Any]],
    contourIdx: int,
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    hierarchy: npt.NDArray[Any] | None = None,
    maxLevel: int = 2**31 - 1,
    offset: tuple[int, int] = (0, 0),
) -> _ImageArray:
    """Draw selected contours with OpenCV-compatible in-place mutation."""
    del lineType, hierarchy, maxLevel
    selected = contours if contourIdx < 0 else contours[contourIdx : contourIdx + 1]

    def draw_selected(draw: Any) -> None:
        for contour in selected:
            points = _points(contour, offset=offset)
            if len(points) < 2:
                continue
            if thickness < 0:
                draw.polygon(points, fill=1)
            else:
                width = max(1, thickness)
                draw.line([*points, points[0]], fill=1, width=width, joint="curve")

    return _paint(image, _drawing_mask(image, draw_selected), color)
