"""Private Pillow/NumPy drawing fallbacks for the OpenCV facade."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator, Sequence
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

_ImageArray = npt.NDArray[Any]
_Point = tuple[int, int]

#: Largest border thickness OpenCV accepts (its `MAX_THICKNESS`).
_MAX_THICKNESS = 32767

#: OpenCV's fixed-point shift used by its polygon rasterizer.
_XY_SHIFT = 16
_XY_ONE = 1 << _XY_SHIFT
_XY_HALF = _XY_ONE >> 1

#: `ThickLine` flags selecting which end of a segment gets a round cap.
_CAP_START = 1
_CAP_END = 2


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
        return np.pad(values, (0, channels - values.size))
    return values[:channels]


def _paint(image: _ImageArray, mask: npt.NDArray[np.bool_], color: Any) -> _ImageArray:
    """Apply a scalar or multi-channel color to a drawing mask."""
    image[mask] = _color_for_image(image, color)
    return image


def _paint_around(
    image: _ImageArray,
    mask: npt.NDArray[np.bool_],
    color: Any,
    points: Sequence[_Point],
    margin: int,
) -> _ImageArray:
    """Paint `mask`, known to be empty farther than `margin` from `points`.

    Only the window spanned by the points is indexed, so a thin shape on a large frame
    does not pay for a full-frame boolean index.
    """
    if not points:
        return image
    height, width = mask.shape
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_start, x_stop = max(min(xs) - margin, 0), min(max(xs) + margin + 1, width)
    y_start, y_stop = max(min(ys) - margin, 0), min(max(ys) + margin + 1, height)
    if x_start >= x_stop or y_start >= y_stop:
        return image
    window = (slice(y_start, y_stop), slice(x_start, x_stop))
    image[window][mask[window]] = _color_for_image(image, color)
    return image


def _point(point: Sequence[int | float]) -> _Point:
    """Convert an OpenCV point to integer pixel coordinates."""
    return round(point[0]), round(point[1])


def _trunc_div(numerator: int, denominator: int) -> int:
    """Divide integers with C++ truncation toward zero.

    Polygon clipping and fixed-point edge slopes need this rounding rule to
    reproduce OpenCV's integer arithmetic. Python's floor division rounds negative
    results down instead, which can move a clipped endpoint or scanline intersection
    by one pixel. Dividing magnitudes first also avoids floating-point precision loss.

    Args:
        numerator: Signed integer dividend.
        denominator: Nonzero signed integer divisor.

    Returns:
        Integer quotient with any fractional part discarded toward zero.

    Raises:
        ZeroDivisionError: If the denominator is zero.

    Example:
        ```pycon
        >>> _trunc_div(-7, 3)
        -2
        >>> _trunc_div(7, -3)
        -2

        ```
    """
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator >= 0) else -quotient


def _clip_polygon_line(
    width: int, height: int, first: _Point, second: _Point
) -> tuple[_Point, _Point] | None:
    """Clip a polygon boundary segment to the image's inclusive pixel bounds.

    Boundary rasterization must start from clipped endpoints: walking a long
    off-canvas segment wastes work, and merely discarding its out-of-bounds pixels
    can produce different boundary pixels after integer rounding.

    Each endpoint receives a bit code for the sides it lies beyond: left = 1,
    right = 2, top = 4, bottom = 8. A shared bit rejects a segment whose endpoints
    lie outside the same side. Otherwise, the algorithm clips top/bottom first,
    recomputes the horizontal codes, then clips left/right if needed. Intersections
    use truncation toward zero and the already updated endpoints; this ordering
    preserves the integer clipping arithmetic used by the polygon fill path.

    Args:
        width: Image width in pixels; valid x coordinates run from 0 to width - 1.
        height: Image height in pixels; valid y coordinates run from 0 to height - 1.
        first: First endpoint as integer (x, y) coordinates.
        second: Second endpoint as integer (x, y) coordinates.

    Returns:
        Clipped endpoints in their original order, or None when the segment is
        outside the image or either image dimension is nonpositive.

    Example:
        ```pycon
        >>> _clip_polygon_line(5, 4, (-2, 1), (7, 1))
        ((0, 1), (4, 1))
        >>> _clip_polygon_line(5, 4, (-2, 0), (-2, 3)) is None
        True

        ```
    """
    if width <= 0 or height <= 0:
        return None

    x1, y1 = first
    x2, y2 = second
    right = width - 1
    bottom = height - 1
    code_1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8
    code_2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8

    if (code_1 & code_2) == 0 and (code_1 | code_2) != 0:
        if code_1 & 12:
            edge = 0 if code_1 < 8 else bottom
            if y2 != y1:
                x1 += _trunc_div((edge - y1) * (x2 - x1), y2 - y1)
            y1 = edge
            code_1 = (x1 < 0) + (x1 > right) * 2
        if code_2 & 12:
            edge = 0 if code_2 < 8 else bottom
            if y2 != y1:
                x2 += _trunc_div((edge - y2) * (x2 - x1), y2 - y1)
            y2 = edge
            code_2 = (x2 < 0) + (x2 > right) * 2
        if (code_1 & code_2) == 0 and (code_1 | code_2) != 0:
            if code_1:
                edge = 0 if code_1 == 1 else right
                if x2 != x1:
                    y1 += _trunc_div((edge - x1) * (y2 - y1), x2 - x1)
                x1 = edge
                code_1 = 0
            if code_2:
                edge = 0 if code_2 == 1 else right
                if x2 != x1:
                    y2 += _trunc_div((edge - x2) * (y2 - y1), x2 - x1)
                x2 = edge
                code_2 = 0

    if (code_1 | code_2) != 0:
        return None
    return (x1, y1), (x2, y2)


def _iter_line_points(first: _Point, second: _Point) -> Iterator[_Point]:
    """Yield integer boundary pixels using OpenCV's 8-connected line stepping.

    The polygon fill path draws boundaries separately from interiors so horizontal
    edges, vertices, and degenerate contours still contribute pixels. This iterator
    uses integer error accumulation rather than interpolating and rounding each
    point, preserving the tie-breaking rule that determines diagonal steps.

    Endpoints are swapped when necessary to make x nondecreasing. For steep lines,
    the axes are exchanged so the same stepping logic advances along the dominant
    axis. Each iteration advances one dominant-axis pixel and advances the other
    axis only when the error is negative. The negative mask selects that correction
    with bitwise operations. Both endpoints are included, including one pixel for
    a zero-length segment.

    Args:
        first: First integer (x, y) endpoint, already clipped by the caller.
        second: Second integer (x, y) endpoint, already clipped by the caller.

    Yields:
        Pixel coordinates along the segment. Iteration may reverse the input
        endpoint order; this helper does not clip or check image bounds.

    Example:
        ```pycon
        >>> list(_iter_line_points((0, 0), (2, 1)))
        [(0, 0), (1, 0), (2, 1)]
        >>> list(_iter_line_points((3, 2), (3, 2)))
        [(3, 2)]

        ```
    """
    x1, y1 = first
    x2, y2 = second
    dx = x2 - x1
    dy = y2 - y1
    delta_x = 1
    delta_y = 1

    if dx < 0:
        dx = -dx
        dy = -dy
        x1, y1 = x2, y2
    if dy < 0:
        dy = -dy
        delta_y = -1

    vertical = dy > dx
    if vertical:
        dx, dy = dy, dx
        delta_x, delta_y = delta_y, delta_x

    error = dx - 2 * dy
    plus_delta = 2 * dx
    minus_delta = -2 * dy
    minus_shift = delta_x
    plus_shift = 0
    minus_step = 0
    plus_step = delta_y
    if vertical:
        plus_step, plus_shift = plus_shift, plus_step
        minus_step, minus_shift = minus_shift, minus_step

    x, y = x1, y1
    for _ in range(dx + 1):
        yield x, y
        mask = -1 if error < 0 else 0
        error += minus_delta + (plus_delta & mask)
        x += minus_shift + (plus_shift & mask)
        y += minus_step + (plus_step & mask)


def _fill_polygon_mask(
    width: int, height: int, polygons: Sequence[list[_Point]]
) -> npt.NDArray[np.bool_]:
    """Build a polygon mask with explicit boundaries and even-odd interior filling.

    This is the shared fill path for polygon and filled-contour drawing. All
    contours participate in one even-odd fill: crossing an edge toggles whether a
    pixel is inside. Nested contours therefore create holes, while overlapping
    interiors can cancel instead of being combined as a union of separate masks.

    Rasterization has two parts. First, each contour is implicitly closed and its
    clipped boundary segments are drawn with 8-connected integer line stepping.
    This retains horizontal edges and degenerate contours that do not define an
    interior. Second, nonhorizontal edges become records (y0, y1, x0, dx), where
    x0 is the x coordinate at the upper endpoint and dx is the per-row increment.
    Both x values use 16 fractional bits, with slopes truncated toward zero to
    preserve integer rasterization rules.

    On each visible row, edges are active over the half-open interval y0 <= y < y1.
    Excluding the lower endpoint prevents a shared vertex from being counted twice.
    Sorted intersections are paired, then each inclusive span runs from the ceiling
    of its left intersection to the floor of its right intersection. Only the
    resulting spans are clipped horizontally: keeping the original edge geometry
    preserves interior crossings for polygons extending beyond the canvas.

    Args:
        width: Nonnegative output width in pixels.
        height: Nonnegative output height in pixels.
        polygons: Contours of integer (x, y) points with any offsets already applied.
            A repeated closing point is unnecessary. Empty contours are ignored.

    Returns:
        A new boolean array of shape (height, width), with True for boundary or
        interior pixels. An empty contour collection produces an all-False mask.

    Example:
        ```pycon
        >>> mask = _fill_polygon_mask(4, 4, [[(1, 1), (2, 1), (2, 2), (1, 2)]])
        >>> mask.astype(int).tolist()
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]

        ```
    """
    mask = np.zeros((height, width), dtype=bool)
    if not polygons:
        return mask

    edges: list[tuple[int, int, int, int]] = []
    for points in polygons:
        if not points:
            continue
        for first, second in zip(points, [*points[1:], points[0]]):
            clipped = _clip_polygon_line(width, height, first, second)
            if clipped is None:
                continue
            clipped_first, clipped_second = clipped
            for x, y in _iter_line_points(clipped_first, clipped_second):
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = True

        previous = points[-1]
        for current in points:
            if previous[1] != current[1]:
                y0, y1 = sorted((previous[1], current[1]))
                x0 = (previous[0] if previous[1] == y0 else current[0]) << _XY_SHIFT
                x1 = (current[0] if previous[1] == y0 else previous[0]) << _XY_SHIFT
                dx = _trunc_div(x1 - x0, y1 - y0)
                edges.append((y0, y1, x0, dx))
            previous = current

    if not edges:
        return mask

    y_start = max(0, min(edge[0] for edge in edges))
    y_stop = min(height, max(edge[1] for edge in edges))
    for y in range(y_start, y_stop):
        intersections = sorted(
            x + (y - y0) * dx for y0, y1, x, dx in edges if y0 <= y < y1
        )
        for left, right in zip(intersections[::2], intersections[1::2]):
            x_start = (left + _XY_ONE - 1) >> _XY_SHIFT
            x_stop = right >> _XY_SHIFT
            if x_start < width and x_stop >= 0:
                mask[y, max(x_start, 0) : min(x_stop, width - 1) + 1] = True

    return mask


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


def _bounded_thickness(thickness: SupportsIndex, shape: str) -> int:
    """Return `thickness` as a Python int, rejecting values OpenCV rejects.

    Raises:
        TypeError: If `thickness` is not an integer, as in OpenCV.
        ValueError: If `thickness` exceeds OpenCV's maximum of 32767, which would
            otherwise make the rasterizers walk thickness-sized spans.
    """
    # A Python int, so a NumPy integer scalar cannot wrap around in derived radii.
    thickness = operator.index(thickness)
    if thickness > _MAX_THICKNESS:
        raise ValueError(
            f"{shape} thickness must be at most {_MAX_THICKNESS}, got {thickness}"
        )
    return thickness


def _validate_shift(shift: int) -> None:
    """Reject fixed-point coordinates not supported by the fallback."""
    if shift != 0:
        raise ValueError("Only unshifted drawing coordinates are supported")


def _line(
    img: _ImageArray,
    pt1: Sequence[int | float],
    pt2: Sequence[int | float],
    color: Any,
    thickness: SupportsIndex = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw a line in place, with OpenCV's round caps when `thickness` exceeds 1.

    Raises:
        TypeError: If `thickness` is not an integer, as in OpenCV.
        ValueError: If `thickness` exceeds OpenCV's maximum of 32767.
    """
    del lineType
    _validate_shift(shift)
    thickness = _bounded_thickness(thickness, "Line")
    if thickness <= 1:
        mask = _drawing_mask(
            img,
            lambda draw: draw.line([_point(pt1), _point(pt2)], fill=1, width=1),
        )
        return _paint(img, mask, color)
    first, second = _point(pt1), _point(pt2)
    mask = np.zeros(img.shape[:2], dtype=bool)
    _thick_line_mask(mask, first, second, thickness, _CAP_START | _CAP_END)
    return _paint_around(img, mask, color, [first, second], thickness)


def _truncating_divide(numerator: int, denominator: int) -> int:
    """Divide integers rounding toward zero, as C's `/` does."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator < 0) == (denominator < 0) else -quotient


def _clip_line(
    width: int, height: int, first: _Point, second: _Point
) -> tuple[bool, _Point, _Point]:
    """Clip a segment to `[0, width) x [0, height)` like OpenCV's `clipLine`.

    The intersection arithmetic copies OpenCV's, a double-precision product truncated
    toward zero, so a clipped endpoint lands on the same integer as OpenCV's.
    """
    if width <= 0 or height <= 0:
        return False, first, second
    right, bottom = width - 1, height - 1
    (x1, y1), (x2, y2) = first, second

    def outcode(x: int, y: int) -> int:
        """Return the Cohen-Sutherland region bits of a point."""
        return (x < 0) + (x > right) * 2 + (y < 0) * 4 + (y > bottom) * 8

    c1, c2 = outcode(x1, y1), outcode(x2, y2)
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if c1 & 12:
            edge = 0 if c1 < 8 else bottom
            x1 += int((edge - y1) * (x2 - x1) / (y2 - y1))
            y1 = edge
            c1 = (x1 < 0) + (x1 > right) * 2
        if c2 & 12:
            edge = 0 if c2 < 8 else bottom
            x2 += int((edge - y2) * (x2 - x1) / (y2 - y1))
            y2 = edge
            c2 = (x2 < 0) + (x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                edge = 0 if c1 == 1 else right
                y1 += int((edge - x1) * (y2 - y1) / (x2 - x1))
                x1 = edge
                c1 = 0
            if c2:
                edge = 0 if c2 == 1 else right
                y2 += int((edge - x2) * (y2 - y1) / (x2 - x1))
                x2 = edge
                c2 = 0
    return (c1 | c2) == 0, (x1, y1), (x2, y2)


def _set_pixels(
    mask: npt.NDArray[np.bool_], xs: npt.NDArray[np.int64], ys: npt.NDArray[np.int64]
) -> None:
    """Set the listed pixels of `mask`, skipping those outside it."""
    height, width = mask.shape
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask[ys[inside], xs[inside]] = True


def _fixed_point_line_pixels(
    shape: tuple[int, int], first: _Point, second: _Point
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the x and y of OpenCV's `Line2`, an 8-connected line between fixed points.

    Both points carry `_XY_SHIFT` fractional bits. The steps along the minor axis are
    the same truncated fixed-point increments OpenCV accumulates, so each pixel of an
    edge falls where OpenCV puts it.
    """
    height, width = shape
    inside, (x1, y1), (x2, y2) = _clip_line(
        width << _XY_SHIFT, height << _XY_SHIFT, first, second
    )
    if not inside:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    dx, dy = x2 - x1, y2 - y1
    # Walk from the endpoint with the smaller major coordinate, one pixel per step;
    # the other endpoint is also set on its own, rounded to the nearest pixel.
    if abs(dx) > abs(dy):
        if dx < 0:
            (x1, y1), (x2, y2), dy = (x2, y2), (x1, y1), -dy
        y_step = _truncating_divide(dy * _XY_ONE, abs(dx) | 1)
        steps = np.arange(((x2 - x1) >> _XY_SHIFT) + 1, dtype=np.int64)
        xs = ((x1 + _XY_HALF) >> _XY_SHIFT) + steps
        ys = (y1 + _XY_HALF + steps * y_step) >> _XY_SHIFT
    else:
        if dy < 0:
            (x1, y1), (x2, y2), dx = (x2, y2), (x1, y1), -dx
        x_step = _truncating_divide(dx * _XY_ONE, abs(dy) | 1)
        steps = np.arange(((y2 - y1) >> _XY_SHIFT) + 1, dtype=np.int64)
        xs = (x1 + _XY_HALF + steps * x_step) >> _XY_SHIFT
        ys = ((y1 + _XY_HALF) >> _XY_SHIFT) + steps
    end_x = np.array([(x2 + _XY_HALF) >> _XY_SHIFT], dtype=np.int64)
    end_y = np.array([(y2 + _XY_HALF) >> _XY_SHIFT], dtype=np.int64)
    return np.concatenate([xs, end_x]), np.concatenate([ys, end_y])


def _fill_convex_fixed_point(
    mask: npt.NDArray[np.bool_], vertices: Sequence[_Point]
) -> None:
    """Fill a convex polygon with `_XY_SHIFT` fixed-point vertices like OpenCV.

    A port of OpenCV's `FillConvexPoly` for 8-connected drawing: the outline is drawn
    with :func:`_fixed_point_line_pixels`, then two edges walk down from the top vertex,
    each advancing by OpenCV's rounded per-row slope, and every row between them is
    filled.
    """
    height, width = mask.shape
    count = len(vertices)
    outline = [
        _fixed_point_line_pixels(mask.shape, vertices[index - 1], vertices[index])
        for index in range(count)
    ]
    _set_pixels(
        mask,
        np.concatenate([xs for xs, _ in outline]),
        np.concatenate([ys for _, ys in outline]),
    )
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    top_index = ys.index(min(ys))
    x_min, x_max = (min(xs) + _XY_HALF) >> _XY_SHIFT, (max(xs) + _XY_HALF) >> _XY_SHIFT
    y_min, y_max = (min(ys) + _XY_HALF) >> _XY_SHIFT, (max(ys) + _XY_HALF) >> _XY_SHIFT
    if count < 3 or x_max < 0 or y_max < 0 or x_min >= width or y_min >= height:
        return
    y_max = min(y_max, height - 1)

    # Per edge: vertex index, walking direction, x, x increment per row, last row.
    edge_index = [top_index, top_index]
    edge_direction = [1, count - 1]
    edge_x = [-_XY_ONE, -_XY_ONE]
    edge_dx = [0, 0]
    edge_end = [y_min, y_min]
    edges_left = count
    y = y_min
    while True:
        for side in range(2):
            if y < edge_end[side]:
                continue
            start = edge_index[side]
            end = (start + edge_direction[side]) % count
            while edges_left > 0:
                edges_left -= 1
                end_y = (vertices[end][1] + _XY_HALF) >> _XY_SHIFT
                if end_y > y:
                    start_x, end_x = vertices[start][0], vertices[end][0]
                    rows = end_y - y
                    edge_end[side] = end_y
                    edge_dx[side] = _truncating_divide(
                        (end_x - start_x) * 2 + rows, 2 * rows
                    )
                    edge_x[side] = start_x
                    edge_index[side] = end
                    break
                start = end
                end = (end + edge_direction[side]) % count
            else:
                edges_left -= 1
        if edges_left < 0:
            return
        if y >= 0:
            left_x, right_x = sorted(edge_x)
            x1 = (left_x + _XY_HALF) >> _XY_SHIFT
            x2 = (right_x + _XY_HALF) >> _XY_SHIFT
            if x2 >= 0 and x1 < width:
                mask[y, max(x1, 0) : min(x2, width - 1) + 1] = True
        edge_x[0] += edge_dx[0]
        edge_x[1] += edge_dx[1]
        y += 1
        if y > y_max:
            return


def _thick_line_mask(
    mask: npt.NDArray[np.bool_],
    first: _Point,
    second: _Point,
    thickness: int,
    caps: int,
) -> None:
    """Set the pixels of OpenCV's `ThickLine` for an 8-connected `thickness > 1` line.

    OpenCV draws the segment as a quadrilateral `thickness` wide in fixed point and
    puts a filled disc of radius `(thickness + 1) // 2` on each end selected by `caps`
    (:data:`_CAP_START`, :data:`_CAP_END`).
    """
    height, width = mask.shape
    inside = 0 <= first[0] < width and 0 <= first[1] < height
    inside = inside and 0 <= second[0] < width and 0 <= second[1] < height
    if not inside:
        # OpenCV clips an off-canvas segment to the canvas grown by `thickness`
        # first, so the quadrilateral it draws is that of the clipped segment.
        margin = thickness
        _, first, second = _clip_line(
            width + 2 * margin,
            height + 2 * margin,
            (first[0] + margin, first[1] + margin),
            (second[0] + margin, second[1] + margin),
        )
        first = (first[0] - margin, first[1] - margin)
        second = (second[0] - margin, second[1] - margin)
    x0, y0 = first[0] << _XY_SHIFT, first[1] << _XY_SHIFT
    x1, y1 = second[0] << _XY_SHIFT, second[1] << _XY_SHIFT
    dx = (x0 - x1) / _XY_ONE
    dy = (y1 - y0) / _XY_ONE
    squared_length = dx * dx + dy * dy
    half_width = thickness << (_XY_SHIFT - 1)
    if abs(squared_length) > np.finfo(np.float64).eps:
        scale = (half_width + (thickness & 1) * _XY_ONE * 0.5) / np.sqrt(squared_length)
        # `cvRound` rounds half to even, as `np.rint` does.
        offset_x = int(np.rint(dy * scale))
        offset_y = int(np.rint(dx * scale))
        _fill_convex_fixed_point(
            mask,
            [
                (x0 + offset_x, y0 + offset_y),
                (x0 - offset_x, y0 - offset_y),
                (x1 - offset_x, y1 - offset_y),
                (x1 + offset_x, y1 + offset_y),
            ],
        )
    radius = (half_width + _XY_HALF) >> _XY_SHIFT
    for cap, (x, y) in ((_CAP_START, (x0, y0)), (_CAP_END, (x1, y1))):
        if caps & cap:
            center = ((x + _XY_HALF) >> _XY_SHIFT, (y + _XY_HALF) >> _XY_SHIFT)
            _paint_disc(mask, center, radius)


def _paint_disc(mask: npt.NDArray[np.bool_], center: _Point, radius: int) -> None:
    """Set OpenCV's filled circle of `radius`, the integer disc, in `mask`.

    The disc is evaluated only over the part of its bounding square that falls inside
    the mask, so a huge radius on a small canvas allocates canvas-sized arrays rather
    than radius-sized ones.
    """
    height, width = mask.shape
    center_x, center_y = center
    x_start, x_stop = max(center_x - radius, 0), min(center_x + radius + 1, width)
    y_start, y_stop = max(center_y - radius, 0), min(center_y + radius + 1, height)
    if x_start >= x_stop or y_start >= y_stop:
        return
    rows = np.arange(y_start, y_stop)[:, np.newaxis] - center_y
    columns = np.arange(x_start, x_stop)[np.newaxis, :] - center_x
    mask[y_start:y_stop, x_start:x_stop] |= rows**2 + columns**2 <= radius * radius


def _rectangle_mask(
    image: _ImageArray, first: _Point, second: _Point, thickness: int
) -> npt.NDArray[np.bool_]:
    """Rasterize an OpenCV rectangle, filled or stroked, into a boolean mask."""
    height, width = image.shape[:2]
    (left, top), (right, bottom) = first, second
    mask = np.zeros((height, width), dtype=bool)

    def fill(x1: int, y1: int, x2: int, y2: int) -> None:
        """Set an inclusive box, clipped to the image."""
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width - 1), min(y2, height - 1)
        if x1 <= x2 and y1 <= y2:
            mask[y1 : y2 + 1, x1 : x2 + 1] = True

    if thickness < 0:
        fill(left, top, right, bottom)
        return mask

    # OpenCV centers each edge's band on the edge and rounds the four joints
    # with a filled disc, so a thick border spreads outside the rectangle as
    # well as inside. Pillow's `width=` only ever grows inward.
    radius = 0 if thickness <= 1 else (thickness + 1) // 2
    fill(left, top - radius, right, top + radius)
    fill(left, bottom - radius, right, bottom + radius)
    fill(left - radius, top, left + radius, bottom)
    fill(right - radius, top, right + radius, bottom)
    if radius:
        for corner in ((left, top), (right, top), (left, bottom), (right, bottom)):
            _paint_disc(mask, corner, radius)
    return mask


def _rectangle(
    img: _ImageArray,
    pt1: Sequence[int | float],
    pt2: Sequence[int | float],
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw or fill an inclusive-axis-aligned rectangle in place.

    A border of `thickness` 2 or more is centered on the rectangle's edges, as in
    OpenCV, so it extends outside the rectangle as well as inside. A negative
    `thickness` fills the rectangle.

    Raises:
        TypeError: If `thickness` is not an integer, as in OpenCV.
        ValueError: If `thickness` exceeds OpenCV's maximum of 32767.
    """
    del lineType
    _validate_shift(shift)
    thickness = _bounded_thickness(thickness, "Rectangle")
    first_point, second_point = _point(pt1), _point(pt2)
    first = (
        min(first_point[0], second_point[0]),
        min(first_point[1], second_point[1]),
    )
    second = (
        max(first_point[0], second_point[0]),
        max(first_point[1], second_point[1]),
    )
    return _paint(img, _rectangle_mask(img, first, second, thickness), color)


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
    """Draw one or more open or closed polylines in place.

    With `thickness` above 1 every segment is OpenCV's thick line, capped with a disc
    at its end, and an open polyline also at its first point.

    Raises:
        TypeError: If `thickness` is not an integer, as in OpenCV.
        ValueError: If `thickness` exceeds OpenCV's maximum of 32767.
    """
    del lineType
    _validate_shift(shift)
    thickness = _bounded_thickness(thickness, "Polyline")
    if thickness > 1:
        mask = np.zeros(img.shape[:2], dtype=bool)
        polylines = [_points(polygon) for polygon in pts]
        for points in polylines:
            _thick_polyline_mask(mask, points, isClosed, thickness)
        all_points = [point for points in polylines for point in points]
        return _paint_around(img, mask, color, all_points, thickness)

    def draw_polylines(draw: Any) -> None:
        for polygon in pts:
            points = _points(polygon)
            if not points:
                continue
            points = [
                point
                for index, point in enumerate(points)
                if index == 0 or point != points[index - 1]
            ]
            if len(points) == 1:
                draw.point(points[0], fill=1)
                continue
            if isClosed:
                points.append(points[0])
            draw.line(points, fill=1, width=1, joint="curve")

    return _paint(img, _drawing_mask(img, draw_polylines), color)


def _thick_polyline_mask(
    mask: npt.NDArray[np.bool_], points: list[_Point], is_closed: bool, thickness: int
) -> None:
    """Set the pixels of OpenCV's `PolyLine` for a `thickness > 1` polyline."""
    if not points:
        return
    previous = points[-1] if is_closed else points[0]
    caps = _CAP_END if is_closed else _CAP_START | _CAP_END
    for point in points if is_closed else points[1:]:
        _thick_line_mask(mask, previous, point, thickness, caps)
        previous = point
        caps = _CAP_END


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
    height, width = img.shape[:2]
    polygons = [_points(polygon, offset=offset) for polygon in pts]
    return _paint(
        img,
        _fill_polygon_mask(width=width, height=height, polygons=polygons),
        color,
    )


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
    """Draw selected contours with OpenCV-compatible in-place mutation.

    Raises:
        TypeError: If `thickness` is not an integer, as in OpenCV.
        ValueError: If `thickness` exceeds OpenCV's maximum of 32767, or if a
            `hierarchy` is given.
    """
    del lineType, maxLevel
    thickness = _bounded_thickness(thickness, "Contour")
    # OpenCV walks the hierarchy tree to also draw a contour's nested descendants;
    # the fallback only does flat contourIdx selection, so reject a hierarchy
    # instead of silently diverging. maxLevel is a no-op without a hierarchy, which
    # matches OpenCV ignoring it whenever hierarchy is None.
    if hierarchy is not None:
        raise ValueError("Only None hierarchy is supported by the fallback")
    selected = contours if contourIdx < 0 else contours[contourIdx : contourIdx + 1]
    if thickness < 0:
        polygons = [_points(contour, offset=offset) for contour in selected]
        return _paint(
            image,
            _fill_polygon_mask(
                width=image.shape[1],
                height=image.shape[0],
                polygons=polygons,
            ),
            color,
        )
    if thickness > 1:
        mask = np.zeros(image.shape[:2], dtype=bool)
        contour_points = [_points(contour, offset=offset) for contour in selected]
        for points in contour_points:
            _thick_polyline_mask(mask, points, True, thickness)
        all_points = [point for points in contour_points for point in points]
        return _paint_around(image, mask, color, all_points, thickness)

    def draw_selected(draw: Any) -> None:
        for contour in selected:
            points = _points(contour, offset=offset)
            if len(points) < 2:
                continue
            width = max(1, thickness)
            draw.line([*points, points[0]], fill=1, width=width, joint="curve")

    return _paint(image, _drawing_mask(image, draw_selected), color)
