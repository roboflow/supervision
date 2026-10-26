"""Private NumPy drawing fallbacks for the OpenCV facade."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Iterator, Sequence
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt

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

#: OpenCV's 451-entry sine table, copied from `drawing.cpp`.
#:
#: `ellipse2Poly` indexes it with integer degrees. The values are float32, so they
#: cannot be regenerated exactly from `numpy.sin` on every platform: about half the
#: entries differ in the last bit.
_SIN_TABLE = np.array(
    """
    0.0000000 0.0174524 0.0348995 0.0523360 0.0697565 0.0871557 0.1045285
    0.1218693 0.1391731 0.1564345 0.1736482 0.1908090 0.2079117 0.2249511
    0.2419219 0.2588190 0.2756374 0.2923717 0.3090170 0.3255682 0.3420201
    0.3583679 0.3746066 0.3907311 0.4067366 0.4226183 0.4383711 0.4539905
    0.4694716 0.4848096 0.5000000 0.5150381 0.5299193 0.5446390 0.5591929
    0.5735764 0.5877853 0.6018150 0.6156615 0.6293204 0.6427876 0.6560590
    0.6691306 0.6819984 0.6946584 0.7071068 0.7193398 0.7313537 0.7431448
    0.7547096 0.7660444 0.7771460 0.7880108 0.7986355 0.8090170 0.8191520
    0.8290376 0.8386706 0.8480481 0.8571673 0.8660254 0.8746197 0.8829476
    0.8910065 0.8987940 0.9063078 0.9135455 0.9205049 0.9271839 0.9335804
    0.9396926 0.9455186 0.9510565 0.9563048 0.9612617 0.9659258 0.9702957
    0.9743701 0.9781476 0.9816272 0.9848078 0.9876883 0.9902681 0.9925462
    0.9945219 0.9961947 0.9975641 0.9986295 0.9993908 0.9998477 1.0000000
    0.9998477 0.9993908 0.9986295 0.9975641 0.9961947 0.9945219 0.9925462
    0.9902681 0.9876883 0.9848078 0.9816272 0.9781476 0.9743701 0.9702957
    0.9659258 0.9612617 0.9563048 0.9510565 0.9455186 0.9396926 0.9335804
    0.9271839 0.9205049 0.9135455 0.9063078 0.8987940 0.8910065 0.8829476
    0.8746197 0.8660254 0.8571673 0.8480481 0.8386706 0.8290376 0.8191520
    0.8090170 0.7986355 0.7880108 0.7771460 0.7660444 0.7547096 0.7431448
    0.7313537 0.7193398 0.7071068 0.6946584 0.6819984 0.6691306 0.6560590
    0.6427876 0.6293204 0.6156615 0.6018150 0.5877853 0.5735764 0.5591929
    0.5446390 0.5299193 0.5150381 0.5000000 0.4848096 0.4694716 0.4539905
    0.4383711 0.4226183 0.4067366 0.3907311 0.3746066 0.3583679 0.3420201
    0.3255682 0.3090170 0.2923717 0.2756374 0.2588190 0.2419219 0.2249511
    0.2079117 0.1908090 0.1736482 0.1564345 0.1391731 0.1218693 0.1045285
    0.0871557 0.0697565 0.0523360 0.0348995 0.0174524 0.0000000 -0.0174524
    -0.0348995 -0.0523360 -0.0697565 -0.0871557 -0.1045285 -0.1218693 -0.1391731
    -0.1564345 -0.1736482 -0.1908090 -0.2079117 -0.2249511 -0.2419219 -0.2588190
    -0.2756374 -0.2923717 -0.3090170 -0.3255682 -0.3420201 -0.3583679 -0.3746066
    -0.3907311 -0.4067366 -0.4226183 -0.4383711 -0.4539905 -0.4694716 -0.4848096
    -0.5000000 -0.5150381 -0.5299193 -0.5446390 -0.5591929 -0.5735764 -0.5877853
    -0.6018150 -0.6156615 -0.6293204 -0.6427876 -0.6560590 -0.6691306 -0.6819984
    -0.6946584 -0.7071068 -0.7193398 -0.7313537 -0.7431448 -0.7547096 -0.7660444
    -0.7771460 -0.7880108 -0.7986355 -0.8090170 -0.8191520 -0.8290376 -0.8386706
    -0.8480481 -0.8571673 -0.8660254 -0.8746197 -0.8829476 -0.8910065 -0.8987940
    -0.9063078 -0.9135455 -0.9205049 -0.9271839 -0.9335804 -0.9396926 -0.9455186
    -0.9510565 -0.9563048 -0.9612617 -0.9659258 -0.9702957 -0.9743701 -0.9781476
    -0.9816272 -0.9848078 -0.9876883 -0.9902681 -0.9925462 -0.9945219 -0.9961947
    -0.9975641 -0.9986295 -0.9993908 -0.9998477 -1.0000000 -0.9998477 -0.9993908
    -0.9986295 -0.9975641 -0.9961947 -0.9945219 -0.9925462 -0.9902681 -0.9876883
    -0.9848078 -0.9816272 -0.9781476 -0.9743701 -0.9702957 -0.9659258 -0.9612617
    -0.9563048 -0.9510565 -0.9455186 -0.9396926 -0.9335804 -0.9271839 -0.9205049
    -0.9135455 -0.9063078 -0.8987940 -0.8910065 -0.8829476 -0.8746197 -0.8660254
    -0.8571673 -0.8480481 -0.8386706 -0.8290376 -0.8191520 -0.8090170 -0.7986355
    -0.7880108 -0.7771460 -0.7660444 -0.7547096 -0.7431448 -0.7313537 -0.7193398
    -0.7071068 -0.6946584 -0.6819984 -0.6691306 -0.6560590 -0.6427876 -0.6293204
    -0.6156615 -0.6018150 -0.5877853 -0.5735764 -0.5591929 -0.5446390 -0.5299193
    -0.5150381 -0.5000000 -0.4848096 -0.4694716 -0.4539905 -0.4383711 -0.4226183
    -0.4067366 -0.3907311 -0.3746066 -0.3583679 -0.3420201 -0.3255682 -0.3090170
    -0.2923717 -0.2756374 -0.2588190 -0.2419219 -0.2249511 -0.2079117 -0.1908090
    -0.1736482 -0.1564345 -0.1391731 -0.1218693 -0.1045285 -0.0871557 -0.0697565
    -0.0523360 -0.0348995 -0.0174524 -0.0000000 0.0174524 0.0348995 0.0523360
    0.0697565 0.0871557 0.1045285 0.1218693 0.1391731 0.1564345 0.1736482
    0.1908090 0.2079117 0.2249511 0.2419219 0.2588190 0.2756374 0.2923717
    0.3090170 0.3255682 0.3420201 0.3583679 0.3746066 0.3907311 0.4067366
    0.4226183 0.4383711 0.4539905 0.4694716 0.4848096 0.5000000 0.5150381
    0.5299193 0.5446390 0.5591929 0.5735764 0.5877853 0.6018150 0.6156615
    0.6293204 0.6427876 0.6560590 0.6691306 0.6819984 0.6946584 0.7071068
    0.7193398 0.7313537 0.7431448 0.7547096 0.7660444 0.7771460 0.7880108
    0.7986355 0.8090170 0.8191520 0.8290376 0.8386706 0.8480481 0.8571673
    0.8660254 0.8746197 0.8829476 0.8910065 0.8987940 0.9063078 0.9135455
    0.9205049 0.9271839 0.9335804 0.9396926 0.9455186 0.9510565 0.9563048
    0.9612617 0.9659258 0.9702957 0.9743701 0.9781476 0.9816272 0.9848078
    0.9876883 0.9902681 0.9925462 0.9945219 0.9961947 0.9975641 0.9986295
    0.9993908 0.9998477 1.0000000
    """.split(),
    dtype=np.float32,
)


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

    _fill_mask_edges(mask, edges)
    return mask


def _fill_mask_edges(
    mask: npt.NDArray[np.bool_], edges: list[tuple[int, int, int, int]]
) -> None:
    """Set polygon interior pixels by pairing scanline intersections.

    Each edge is `(y0, y1, x, dx)`: rows are integer pixel coordinates, while
    `x` at row `y0` and its per-row increment `dx` use 16 fractional bits
    (65536 units per pixel). Edges are active over `y0 <= y < y1`, so a shared
    vertex does not count twice. Horizontal edges must be omitted by the caller.

    On each visible row, intersections are sorted and paired using the even-odd
    fill rule. The left intersection rounds up and the right rounds down to an
    inclusive pixel span, clipped to the mask width. Boundary strokes are the
    caller's responsibility; this implements OpenCV's `FillEdgeCollection` span
    rule without clearing any pixels already set in the mask.

    Args:
        mask: Boolean `(height, width)` destination, modified in place.
        edges: Non-horizontal edges with `y0 < y1`, in any order. An empty list
            leaves the mask unchanged.

    Example:
        Fill columns 1 through 3 on rows 0 and 1; row 2 is the excluded end row.

        ```pycon
        >>> mask = np.zeros((3, 5), dtype=bool)
        >>> edges = [(0, 2, 1 << 16, 0), (0, 2, 3 << 16, 0)]
        >>> _fill_mask_edges(mask, edges)
        >>> mask.astype(int).tolist()
        [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]

        ```
    """
    if not edges:
        return
    height, width = mask.shape
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


def _fill_ellipse_arc_mask(
    mask: npt.NDArray[np.bool_],
    points: list[_Point],
    center: _Point,
) -> None:
    """Set the boundary and interior pixels of an elliptic pie slice.

    The ordered arc vertices and center use 16 fractional bits (65536 units per
    pixel). Appending the center closes the arc with two radial edges, producing
    a sector rather than the segment formed by joining the arc endpoints directly.
    All coordinates are `(x, y)`; mask indexing is `[y, x]`.

    Boundary endpoints round to pixels by adding half a pixel before shifting.
    Clipped boundary lines are painted first. For the interior, each edge keeps
    its original rounded row range, but its slope and starting x incorporate
    clipped endpoints when available. Retaining that range is necessary for
    off-canvas sectors to follow OpenCV's `CollectPolyEdges` behavior. Horizontal
    edges contribute boundary pixels only; the remaining edges use the even-odd
    scanline fill in :func:`~supervision._cv2._drawing._fill_mask_edges`.

    Args:
        mask: Boolean `(height, width)` destination. Existing true pixels remain
            set; boundary and interior pixels are added in place.
        points: Ordered fixed-point arc vertices, normally produced by
            :func:`~supervision._cv2._drawing._ellipse_fixed_points`.
        center: Fixed-point ellipse center used to close the sector.

    Example:
        Two arc vertices and the center form a small right-triangle sector.

        ```pycon
        >>> mask = np.zeros((5, 5), dtype=bool)
        >>> points = [(3 << 16, 1 << 16), (1 << 16, 3 << 16)]
        >>> _fill_ellipse_arc_mask(mask, points, (1 << 16, 1 << 16))
        >>> mask[1:4, 1:4].astype(int).tolist()
        [[1, 1, 1], [1, 1, 0], [1, 0, 0]]

        ```
    """
    height, width = mask.shape
    polygon = [*points, center]
    edges: list[tuple[int, int, int, int]] = []
    previous = polygon[-1]
    for current in polygon:
        previous_y = (previous[1] + _XY_HALF) >> _XY_SHIFT
        current_y = (current[1] + _XY_HALF) >> _XY_SHIFT
        previous_x = (previous[0] + _XY_HALF) >> _XY_SHIFT
        current_x = (current[0] + _XY_HALF) >> _XY_SHIFT
        first = (previous_x, previous_y)
        second = (current_x, current_y)

        clipped = _clip_polygon_line(width, height, first, second)
        if clipped is not None:
            for x, y in _iter_line_points(*clipped):
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = True

        edge_first_x, edge_first_y = previous_x << _XY_SHIFT, previous_y
        edge_second_x, edge_second_y = current_x << _XY_SHIFT, current_y
        if not (
            0 <= previous_x < width
            and 0 <= current_x < width
            and 0 <= previous_y < height
            and 0 <= current_y < height
        ):
            inside, clipped_first, clipped_second = _clip_line(
                width, height, first, second
            )
            if inside:
                if clipped_first[1] != clipped_second[1]:
                    edge_first_y, edge_second_y = clipped_first[1], clipped_second[1]
                edge_first_x = clipped_first[0] << _XY_SHIFT
                edge_second_x = clipped_second[0] << _XY_SHIFT

        if previous_y == current_y:
            previous = current
            continue
        dx = _trunc_div(edge_second_x - edge_first_x, edge_second_y - edge_first_y)
        if previous_y < current_y:
            edges.append(
                (
                    previous_y,
                    current_y,
                    edge_first_x + (previous_y - edge_first_y) * dx,
                    dx,
                )
            )
        else:
            edges.append(
                (
                    current_y,
                    previous_y,
                    edge_second_x + (current_y - edge_second_y) * dx,
                    dx,
                )
            )
        previous = current

    _fill_mask_edges(mask, edges)


def _points(points: npt.NDArray[Any], offset: tuple[int, int] = (0, 0)) -> list[_Point]:
    """Normalize OpenCV polygon shapes to integer pixel coordinates."""
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
    first, second = _point(pt1), _point(pt2)
    if thickness <= 1:
        return _paint_thin_segments(img, [(first, second)], color)
    mask = np.zeros(img.shape[:2], dtype=bool)
    _thick_line_mask(mask, first, second, thickness, _CAP_START | _CAP_END)
    return _paint_around(img, mask, color, [first, second], thickness)


def _paint_thin_segments(
    img: _ImageArray, segments: Sequence[tuple[_Point, _Point]], color: Any
) -> _ImageArray:
    """Paint one-pixel-wide segments with OpenCV's 8-connected `Line` routine.

    OpenCV draws lines, polyline edges and contour edges of thickness 1 by clipping
    each segment to the canvas and stepping the clipped endpoints with its
    `LineIterator`, so a partly off-canvas segment follows the clipped slope rather
    than the original one. A zero-length segment paints its single pixel.

    Args:
        img: Image to modify in place.
        segments: Integer `(first, second)` endpoint pairs.
        color: Scalar or channel values in the image's channel order.

    Returns:
        The same image, for OpenCV-style chaining.
    """
    height, width = img.shape[:2]
    color_value = _color_for_image(img, color)
    for first, second in segments:
        # `_clip_line` repeats OpenCV's floating-point intersection arithmetic,
        # which moves far-off endpoints differently from exact integer division.
        inside, clipped_first, clipped_second = _clip_line(width, height, first, second)
        if not inside:
            continue
        # Paint per segment: a clipped segment spans at most the canvas diagonal,
        # so memory stays bounded however many segments a polyline has.
        xs, ys = np.array(
            list(_iter_line_points(clipped_first, clipped_second)), dtype=np.int64
        ).T
        img[ys, xs] = color_value
    return img


def _truncating_divide(numerator: int, denominator: int) -> int:
    """Divide integers rounding toward zero, as C's `/` does."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator < 0) == (denominator < 0) else -quotient


def _clip_line(
    width: int, height: int, first: _Point, second: _Point
) -> tuple[bool, _Point, _Point]:
    """Clip a segment to `[0, width) x [0, height)` like OpenCV's `clipLine`.

    The intersection arithmetic copies OpenCV's, a double-precision product truncated
    toward zero, so a clipped endpoint lands on the same integer as OpenCV's. The
    first factor is converted to a float before multiplying, as OpenCV's
    `(double)(a - y1) * (x2 - x1)` does, so huge products round the same way.
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
            x1 += int(float(edge - y1) * (x2 - x1) / (y2 - y1))
            y1 = edge
            c1 = (x1 < 0) + (x1 > right) * 2
        if c2 & 12:
            edge = 0 if c2 < 8 else bottom
            x2 += int(float(edge - y2) * (x2 - x1) / (y2 - y1))
            y2 = edge
            c2 = (x2 < 0) + (x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                edge = 0 if c1 == 1 else right
                y1 += int(float(edge - x1) * (y2 - y1) / (x2 - x1))
                x1 = edge
                c1 = 0
            if c2:
                edge = 0 if c2 == 1 else right
                y2 += int(float(edge - x2) * (y2 - y1) / (x2 - x1))
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


def _ellipse_arc_points(
    center_x: float,
    center_y: float,
    axes_width: float,
    axes_height: float,
    angle: int,
    arc_start: int,
    arc_end: int,
    delta: int,
) -> list[tuple[float, float]]:
    """Sample a rotated ellipse arc without quantizing its coordinates to pixels.

    Sampling uses the stored float32 sine table rather than evaluating trigonometric
    functions, preserving the values used by OpenCV's `ellipse2Poly`. With image
    coordinates (x right, y down), increasing arc angles and positive rotation
    move clockwise. Zero arc angle starts at the positive first semi-axis.

    Reversed arc bounds are swapped. Bounds outside one turn are shifted together
    by multiples of 360 degrees; spans greater than 360 become a full ellipse.
    Sampling starts at the normalized lower bound and includes the upper bound,
    even when the final step is shorter than `delta`. Equal bounds return two
    copies of the center, not a point on the ellipse.

    Args:
        center_x: Horizontal center in the caller's coordinate units.
        center_y: Vertical center in the same units.
        axes_width: First semi-axis length, not the full width, in the same units.
        axes_height: Second semi-axis length, not the full height, in the same units.
        angle: Integer rotation in degrees; normalized to one turn.
        arc_start: Integer starting angle in degrees before rotation.
        arc_end: Integer ending angle in degrees before rotation.
        delta: Positive integer sampling step in degrees, supplied by the caller.

    Returns:
        Ordered floating-point `(x, y)` vertices in the input units. A full turn
        repeats its initial vertex at the end; no duplicate removal is performed.

    Example:
        Sample a quarter ellipse, then the same arc with a 90-degree rotation.

        ```pycon
        >>> _ellipse_arc_points(5, 5, 3, 2, 0, 0, 90, 90)
        [(8.0, 5.0), (5.0, 7.0)]
        >>> _ellipse_arc_points(5, 5, 3, 2, 90, 0, 90, 90)
        [(5.0, 8.0), (3.0, 5.0)]
        >>> _ellipse_arc_points(5, 5, 3, 2, 0, 45, 45, 90)
        [(5, 5), (5, 5)]

        ```
    """
    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360

    if arc_start > arc_end:
        arc_start, arc_end = arc_end, arc_start
    while arc_start < 0:
        arc_start += 360
        arc_end += 360
    while arc_end > 360:
        arc_end -= 360
        arc_start -= 360
    if arc_end - arc_start > 360:
        arc_start, arc_end = 0, 360

    alpha = float(_SIN_TABLE[450 - angle])
    beta = float(_SIN_TABLE[angle])
    points: list[tuple[float, float]] = []
    for step in range(arc_start, arc_end + delta, delta):
        current = arc_end if step > arc_end else step
        if current < 0:
            current += 360
        x = axes_width * float(_SIN_TABLE[450 - current])
        y = axes_height * float(_SIN_TABLE[current])
        points.append(
            (
                center_x + x * alpha - y * beta,
                center_y + x * beta + y * alpha,
            )
        )

    if len(points) == 1:
        points = [(center_x, center_y), (center_x, center_y)]
    return points


def _ellipse_fixed_points(
    center: _Point,
    axes: _Point,
    angle: float,
    start_angle: float,
    end_angle: float,
) -> list[_Point]:
    """Convert pixel-based ellipse geometry into fixed-point arc vertices.

    Input coordinates are integer pixels; output coordinates have 16 fractional
    bits, so `(65536, 131072)` represents pixel position `(1, 2)`. Angles round
    to integer degrees with ties to even before sampling through
    :func:`~supervision._cv2._drawing._ellipse_arc_points`.

    The larger semi-axis selects an angular step of 90 degrees below 3 pixels,
    30 below 10, 18 below 15, and 5 otherwise. Each sampled coordinate is rounded
    in two stages: nearest whole pixel, then the remaining fixed-point fraction.
    This preserves OpenCV's `EllipseEx` rounding order. Consecutive duplicate
    vertices are removed, but a full ellipse retains its closing vertex. If at
    most one vertex survives, two copies of the center represent the degenerate
    path so the stroke rasterizer can still paint its caps.

    Args:
        center: Integer `(x, y)` center in pixels.
        axes: Non-negative integer semi-axis lengths in pixels; already validated
            by the caller.
        angle: Clockwise rotation in degrees in image coordinates.
        start_angle: Arc start in degrees before rotation.
        end_angle: Arc end in degrees before rotation. Reversed and out-of-range
            bounds are normalized by the arc sampler.

    Returns:
        At least two integer `(x, y)` vertices, scaled by 65536, for the fixed-point
        stroke or fill helpers.

    Example:
        A two-pixel radius selects 90-degree steps, giving four cardinal vertices
        and a closing vertex. Dividing by 65536 makes the units visible.

        ```pycon
        >>> points = _ellipse_fixed_points((3, 3), (2, 1), 0, 0, 360)
        >>> [(x // 65536, y // 65536) for x, y in points]
        [(5, 3), (3, 4), (1, 3), (3, 2), (5, 3)]
        >>> _ellipse_fixed_points((1, 2), (0, 0), 0, 0, 360)
        [(65536, 131072), (65536, 131072)]

        ```
    """
    fixed_angle = round(angle)
    arc_start = round(start_angle)
    arc_end = round(end_angle)
    center_x, center_y = center[0] << _XY_SHIFT, center[1] << _XY_SHIFT
    axes_width, axes_height = axes[0] << _XY_SHIFT, axes[1] << _XY_SHIFT
    delta = (max(axes_width, axes_height) + _XY_HALF) >> _XY_SHIFT
    delta = 90 if delta < 3 else 30 if delta < 10 else 18 if delta < 15 else 5

    points = _ellipse_arc_points(
        float(center_x),
        float(center_y),
        float(axes_width),
        float(axes_height),
        fixed_angle,
        arc_start,
        arc_end,
        delta,
    )
    fixed: list[_Point] = []
    previous: _Point | None = None
    for x, y in points:
        point_x = round(x / _XY_ONE) << _XY_SHIFT
        point_y = round(y / _XY_ONE) << _XY_SHIFT
        point_x += round(x - point_x)
        point_y += round(y - point_y)
        point = (point_x, point_y)
        if point != previous:
            fixed.append(point)
            previous = point

    if len(fixed) <= 1:
        fixed = [(center_x, center_y), (center_x, center_y)]
    return fixed


def _thick_line_mask_fixed_point(
    mask: npt.NDArray[np.bool_],
    first: _Point,
    second: _Point,
    thickness: int,
    caps: int,
) -> None:
    """Rasterize a fixed-point segment with independently selectable round caps.

    Endpoints have 16 fractional bits (65536 units per pixel). For thickness at
    most one, endpoints round to integer pixels, the segment is clipped, and an
    inclusive one-pixel line is drawn; cap flags have no effect on this path.

    Thicker segments become a filled quadrilateral perpendicular to the segment,
    plus a disc at each requested endpoint. Odd thickness adds half a pixel to
    the perpendicular half-width, matching OpenCV's `ThickLine` construction.
    The quadrilateral is formed before clipping, because clipping the centerline
    first would change off-canvas stroke coverage. Coincident endpoints have no
    quadrilateral and draw only the requested cap discs.

    Args:
        mask: Boolean `(height, width)` destination, updated in place without
            clearing existing pixels.
        first: Fixed-point `(x, y)` start, possibly outside the mask.
        second: Fixed-point `(x, y)` end, possibly outside the mask.
        thickness: Stroke width in pixels, already bounded by the caller.
        caps: Bit mask: 1 adds the start cap, 2 the end cap, 3 both, and 0 neither.

    Example:
        Coincident endpoints with thickness two produce a radius-one disc when
        a cap is requested.

        ```pycon
        >>> mask = np.zeros((5, 5), dtype=bool)
        >>> point = (2 << 16, 2 << 16)
        >>> _thick_line_mask_fixed_point(mask, point, point, 2, 1)
        >>> mask[1:4, 1:4].astype(int).tolist()
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

        ```
    """
    height, width = mask.shape
    x0, y0 = first
    x1, y1 = second
    if thickness <= 1:
        rounded_first = ((x0 + _XY_HALF) >> _XY_SHIFT, (y0 + _XY_HALF) >> _XY_SHIFT)
        rounded_second = ((x1 + _XY_HALF) >> _XY_SHIFT, (y1 + _XY_HALF) >> _XY_SHIFT)
        clipped = _clip_polygon_line(width, height, rounded_first, rounded_second)
        if clipped is not None:
            for x, y in _iter_line_points(*clipped):
                mask[y, x] = True
        return

    dx = (x0 - x1) / _XY_ONE
    dy = (y1 - y0) / _XY_ONE
    squared_length = dx * dx + dy * dy
    half_width = thickness << (_XY_SHIFT - 1)
    if abs(squared_length) > np.finfo(np.float64).eps:
        scale = (half_width + (thickness & 1) * _XY_ONE * 0.5) / np.sqrt(squared_length)
        offset_x = round(dy * scale)
        offset_y = round(dx * scale)
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


def _thick_polyline_mask_fixed_point(
    mask: npt.NDArray[np.bool_],
    points: list[_Point],
    is_closed: bool,
    thickness: int,
) -> None:
    """Join fixed-point segments with round caps while preserving their order.

    An open path paints both caps on its first segment, then only the end cap
    on each later segment. A closed path starts with the last-to-first segment
    and paints only end caps, visiting every vertex once as a segment end. This
    reproduces OpenCV's `PolyLine` cap scheduling through
    :func:`~supervision._cv2._drawing._thick_line_mask_fixed_point`.

    Args:
        mask: Boolean `(height, width)` destination, modified in place. Previously
            set pixels remain set.
        points: Ordered `(x, y)` vertices with 16 fractional bits. An empty list
            does nothing. A single vertex draws nothing for an open path, but
            becomes a zero-length capped segment for a closed path.
        is_closed: Whether to connect the last vertex back to the first. An open
            path can also repeat its initial vertex explicitly, as full ellipses do.
        thickness: Stroke width in pixels, already bounded by the caller.

    Example:
        Closing an L-shaped path adds the diagonal from the last vertex to the
        first. Coordinates are scaled by 65536 before rasterization.

        ```pycon
        >>> points = [(x << 16, y << 16) for x, y in [(1, 1), (3, 1), (3, 3)]]
        >>> mask = np.zeros((5, 5), dtype=bool)
        >>> _thick_polyline_mask_fixed_point(mask, points, False, 1)
        >>> bool(mask[2, 2])
        False
        >>> _thick_polyline_mask_fixed_point(mask, points, True, 1)
        >>> bool(mask[2, 2])
        True

        ```
    """
    if not points:
        return
    index = len(points) - 1 if is_closed else 0
    previous = points[index]
    caps = _CAP_END if is_closed else _CAP_START | _CAP_END
    segments = points if is_closed else points[1:]
    for point in segments:
        _thick_line_mask_fixed_point(mask, previous, point, thickness, caps)
        previous = point
        caps = _CAP_END


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
    radius: int | float,
    color: Any,
    thickness: int = 1,
    lineType: int = 8,
    shift: int = 0,
) -> _ImageArray:
    """Draw or fill a circle in place.

    A non-integer `radius` is rounded with ties to even, as the thick path rounds
    its ellipse axes.

    Raises:
        ValueError: If `radius` is negative, as in OpenCV.
    """
    del lineType
    _validate_shift(shift)
    if thickness > 1:
        # OpenCV strokes a thick circle centred on its radius, half the
        # thickness on either side, by drawing it as a full ellipse arc.
        # Pillow's outline grows inward from the bounding box instead, which
        # shrinks a thick ring by half its thickness.
        return _ellipse(img, center, (radius, radius), 0, 0, 360, color, thickness)
    radius = round(radius)
    if radius < 0:
        raise ValueError("Circle radius must be non-negative")
    height, width = img.shape[:2]
    center_x, center_y = _point(center)
    if (
        center_x + radius < 0
        or center_x - radius >= width
        or center_y + radius < 0
        or center_y - radius >= height
    ):
        return img
    offsets = _circle_octant_offsets(radius)
    color_value = _color_for_image(img, color)
    if thickness < 0:
        _fill_circle_rows(img, (center_x, center_y), radius, offsets, color_value)
        return img
    # Mirror every octant offset into the other seven octants.
    major, minor = offsets[:, 0], offsets[:, 1]
    xs = center_x + np.concatenate(
        [-major, major, -major, major, -minor, minor, -minor, minor]
    )
    ys = center_y + np.concatenate(
        [-minor, -minor, minor, minor, -major, -major, major, major]
    )
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    img[ys[inside], xs[inside]] = color_value
    return img


def _circle_octant_offsets(radius: int) -> npt.NDArray[np.int64]:
    """Return OpenCV's midpoint-circle offsets `(dx, dy)` for one octant.

    This ports the integer error stepping of OpenCV's 8-connected `Circle`
    routine, which `cv2.circle` uses for unshifted outlines one pixel wide and for
    filled circles. Starting at `(radius, 0)`, `dy` grows by one every step and
    `dx` shrinks by one whenever the accumulated error turns positive; stepping
    stops once `dx < dy`. The other seven octants are mirror images.

    Args:
        radius: Non-negative circle radius in pixels.

    Returns:
        Array of shape `(N, 2)` holding `(dx, dy)` pairs with `dx >= dy >= 0`.

    Example:
        ```pycon
        >>> _circle_octant_offsets(3).tolist()
        [[3, 0], [2, 1], [2, 2]]

        ```
    """
    # The octant ends once dy passes dx, after at most radius + 1 steps.
    offsets = np.empty((radius + 1, 2), dtype=np.int64)
    count = 0
    error, dx, dy, plus, minus = 0, radius, 0, 1, 2 * radius - 1
    while dx >= dy:
        offsets[count] = dx, dy
        count += 1
        dy += 1
        error += plus
        plus += 2
        # -1 when the error is positive and dx must step inward, 0 otherwise.
        step = -1 if error > 0 else 0
        error -= minus & step
        dx += step
        minus -= step & 2
    return offsets[:count]


def _fill_circle_rows(
    img: _ImageArray,
    center: _Point,
    radius: int,
    offsets: npt.NDArray[np.int64],
    color: Any,
) -> None:
    """Fill a circle row by row with the spans of OpenCV's `Circle` routine.

    Every octant offset `(dx, dy)` fills rows `y ± dy` between `x ± dx` and rows `y ±
    dx` between `x ± dy`. A row may be reached by several offsets, so its span is the
    widest one; only the canvas window covering the circle is touched.
    """
    height, width = img.shape[:2]
    center_x, center_y = center
    y_start, y_stop = max(center_y - radius, 0), min(center_y + radius + 1, height)
    x_start, x_stop = max(center_x - radius, 0), min(center_x + radius + 1, width)
    if y_start >= y_stop or x_start >= x_stop:
        return
    major, minor = offsets[:, 0], offsets[:, 1]
    rows = center_y + np.concatenate([-minor, minor, -major, major])
    spans = np.concatenate([major, major, minor, minor])
    visible = (rows >= y_start) & (rows < y_stop)
    # Only rows inside the canvas window get a span, so a large radius on a small
    # image does not allocate one entry per circle row.
    half_widths = np.full(y_stop - y_start, -1, dtype=np.int64)
    np.maximum.at(half_widths, rows[visible] - y_start, spans[visible])
    columns = np.abs(np.arange(x_start, x_stop) - center_x)
    window = (slice(y_start, y_stop), slice(x_start, x_stop))
    img[window][columns[np.newaxis, :] <= half_widths[:, np.newaxis]] = color


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
    """Draw or fill an optionally rotated ellipse arc in place.

    The fallback samples the arc through
    :func:`~supervision._cv2._drawing._ellipse_fixed_points`, then rasterizes its
    outline with fixed-point quadrilaterals and round caps. Negative thickness
    selects filling: a rounded end-minus-start span of at least 360 degrees uses
    the convex fill path; all other spans use a sector closed through the center
    and filled with the even-odd rule. This dispatch uses the rounded bounds
    before the sampler swaps or normalizes them.

    Args:
        img: Grayscale `(height, width)` or multichannel `(height, width, channels)`
            image to modify. Pixels outside the rasterized shape are preserved.
        center: `(x, y)` center in pixels, rounded to integers with ties to even.
        axes: Semi-axis lengths in pixels, not full diameters. Values round to
            integers with ties to even before checking that they are non-negative.
        angle: Clockwise rotation in degrees (image y increases downward), rounded
            to an integer with ties to even.
        startAngle: Arc start in degrees before rotation, rounded with ties to even.
        endAngle: Arc end in degrees before rotation, rounded with ties to even.
            The sampler swaps reversed bounds and limits spans to one full turn.
        color: Scalar or channel values in the image's channel order, normally BGR.
            A scalar paints only the first channel of a multichannel image;
            missing channels are zero-filled and surplus values are ignored.
        thickness: Integer outline width in pixels; negative values fill the shape.
        lineType: Accepted for API compatibility but ignored; rendering is aliased.
        shift: Coordinate fractional-bit count. Only zero is supported by this
            entry point, despite the internal rasterizer using fixed-point vertices.

    Returns:
        The same image object, painted in place and clipped to its dimensions.

    Raises:
        TypeError: If `thickness` is not an integer, as in OpenCV.
        ValueError: If `thickness` exceeds 32767, a rounded semi-axis is negative,
            or `shift` is nonzero.

    Example:
        A filled quarter ellipse includes the center and extends right and down.

        ```pycon
        >>> image = np.zeros((9, 9), dtype=np.uint8)
        >>> result = _ellipse(image, (4, 4), (3, 2), 0, 0, 90, 7, thickness=-1)
        >>> result is image
        True
        >>> image[[4, 4, 6, 2], [4, 7, 4, 4]].tolist()
        [7, 7, 7, 0]

        ```

        Fractional arc bounds round before choosing the fill path: 359.6 degrees
        fills the same full ellipse as 360 degrees.

        ```pycon
        >>> rounded = np.zeros((9, 9), dtype=np.uint8)
        >>> full = np.zeros_like(rounded)
        >>> _ = _ellipse(rounded, (4, 4), (3, 2), 0, 0, 359.6, 1, thickness=-1)
        >>> _ = _ellipse(full, (4, 4), (3, 2), 0, 0, 360, 1, thickness=-1)
        >>> np.array_equal(rounded, full)
        True

        ```
    """
    del lineType
    _validate_shift(shift)
    thickness = _bounded_thickness(thickness, "Ellipse")
    center_point = _point(center)
    axes_point = (round(axes[0]), round(axes[1]))
    if axes_point[0] < 0 or axes_point[1] < 0:
        raise ValueError("Ellipse axes must be non-negative")

    arc_start = round(startAngle)
    arc_end = round(endAngle)
    points = _ellipse_fixed_points(center_point, axes_point, angle, arc_start, arc_end)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    if thickness < 0:
        if arc_end - arc_start >= 360:
            _fill_convex_fixed_point(mask, points)
        else:
            center_fixed = (center_point[0] << _XY_SHIFT, center_point[1] << _XY_SHIFT)
            _fill_ellipse_arc_mask(mask, points, center_fixed)
        return _paint(img, mask, color)

    _thick_polyline_mask_fixed_point(mask, points, False, thickness)
    painted_points = [
        ((x + _XY_HALF) >> _XY_SHIFT, (y + _XY_HALF) >> _XY_SHIFT) for x, y in points
    ]
    return _paint_around(img, mask, color, painted_points, thickness)


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

    # OpenCV's `PolyLine` draws every edge as its own thin line; a closed
    # polyline adds the edge from the last point back to the first.
    segments: list[tuple[_Point, _Point]] = []
    for polygon in pts:
        points = _points(polygon)
        if isClosed and points:
            segments.append((points[-1], points[0]))
        segments.extend(itertools.pairwise(points))
    return _paint_thin_segments(img, segments, color)


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

    # Thickness 0 and 1 both draw each closed contour edge as a thin line, so a
    # single-point contour still paints its pixel.
    segments: list[tuple[_Point, _Point]] = []
    for contour in selected:
        points = _points(contour, offset=offset)
        segments.extend(zip(points, [*points[1:], *points[:1]]))
    return _paint_thin_segments(image, segments, color)
