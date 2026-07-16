"""Private polygon geometry fallbacks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _point_in_polygon(point: tuple[int, int], polygon: npt.NDArray[np.float64]) -> bool:
    """Return whether an integer pixel lies inside or on a polygon boundary."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x_current, y_current = current
        x_previous, y_previous = previous
        edge = current - previous
        relative = np.array([x - x_previous, y - y_previous], dtype=np.float64)
        if (
            edge[0] * relative[1] - edge[1] * relative[0] == 0
            and min(x_previous, x_current) <= x <= max(x_previous, x_current)
            and min(y_previous, y_current) <= y <= max(y_previous, y_current)
        ):
            return True
        if (y_current > y) != (y_previous > y):
            intersection = (x_previous - x_current) * (y - y_current) / (
                y_previous - y_current
            ) + x_current
            if x < intersection:
                inside = not inside
        previous = current
    return inside


def _as_points(contour: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Normalize an OpenCV contour to an ``(N, 2)`` float64 array."""
    points = np.asarray(contour)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim not in (2, 3) or points.shape[-1] != 2:
        raise ValueError("Contours must have shape (N, 2) or (N, 1, 2)")
    return points.reshape(-1, 2).astype(np.float64, copy=False)


def _contour_area(contour: npt.NDArray[Any], oriented: bool = False) -> float:
    """Compute a contour's signed or absolute shoelace area."""
    points = _as_points(contour)
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area if oriented else abs(area)


def _douglas_peucker(
    points: npt.NDArray[np.float64], epsilon: float
) -> npt.NDArray[np.float64]:
    """Simplify an open polyline with the Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points

    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    pending = [(0, len(points) - 1)]
    epsilon_squared = float(epsilon) ** 2

    while pending:
        start, end = pending.pop()
        segment = points[end] - points[start]
        segment_length_squared = float(np.dot(segment, segment))
        candidates = points[start + 1 : end]
        if not len(candidates):
            continue

        if segment_length_squared == 0:
            distances_squared = np.sum((candidates - points[start]) ** 2, axis=1)
        else:
            offsets = candidates - points[start]
            projections = np.clip(
                np.sum(offsets * segment, axis=1) / segment_length_squared,
                0.0,
                1.0,
            )
            closest = points[start] + projections[:, None] * segment
            distances_squared = np.sum((candidates - closest) ** 2, axis=1)

        relative_index = int(np.argmax(distances_squared))
        if distances_squared[relative_index] > epsilon_squared:
            split = start + 1 + relative_index
            keep[split] = True
            pending.extend(((start, split), (split, end)))

    return cast(npt.NDArray[np.float64], points[keep])  # type: ignore[redundant-cast]


def _approx_poly_dp(
    contour: npt.NDArray[Any], epsilon: float, closed: bool
) -> npt.NDArray[Any]:
    """Approximate a contour with the supported OpenCV polygon contract."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    points = _as_points(contour)
    if len(points) == 0:
        dtype = np.asarray(contour).dtype
        return np.empty((0, 1, 2), dtype=dtype)

    if closed and len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if closed and len(points) > 2:
        # Seed the two split anchors the way OpenCV's approxPolyDP does: the point
        # farthest from points[0], then the point farthest from that one. Two O(N)
        # passes replace an O(N^2) all-pairs distance matrix while landing on cv2's
        # own arc endpoints, which matters because approximate_polygon re-invokes
        # this on the full-size polygon every simplification step.
        coordinates = points.astype(np.float64)
        anchor_a = int(np.argmax(np.sum((coordinates - coordinates[0]) ** 2, axis=1)))
        anchor_b = int(
            np.argmax(np.sum((coordinates - coordinates[anchor_a]) ** 2, axis=1))
        )
        start, end = sorted((anchor_a, anchor_b))
        first_arc = points[start : end + 1]
        second_arc = np.concatenate((points[end:], points[: start + 1]))
        first_simplified = _douglas_peucker(first_arc, epsilon)
        second_simplified = _douglas_peucker(second_arc, epsilon)
        simplified = np.concatenate((first_simplified[:-1], second_simplified[:-1]))
    else:
        simplified = _douglas_peucker(points, epsilon)
    if closed and len(simplified) > 1 and np.array_equal(simplified[0], simplified[-1]):
        simplified = simplified[:-1]

    dtype = np.asarray(contour).dtype
    return simplified.astype(dtype, copy=False).reshape(-1, 1, 2)


def _cross(edge: npt.NDArray[np.float64], point: npt.NDArray[np.float64]) -> float:
    """Return the two-dimensional cross product of two vectors."""
    return float(edge[0] * point[1] - edge[1] * point[0])


def _intersect_convex_convex(
    first: npt.NDArray[Any],
    second: npt.NDArray[Any],
    handle_nested: bool = True,
) -> tuple[float, npt.NDArray[Any]]:
    """Clip two convex polygons and return their intersection area and vertices."""
    del handle_nested
    subject = _as_points(first)
    clip = _as_points(second)
    output = subject
    if len(subject) < 3 or len(clip) < 3:
        return 0.0, np.empty((0, 1, 2), dtype=np.float32)

    orientation = 1.0 if _contour_area(clip, oriented=True) >= 0 else -1.0
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        edge = clip_end - clip_start
        input_points = output
        if len(input_points) == 0:
            break
        output_points: list[npt.NDArray[np.float64]] = []
        previous = input_points[-1]
        previous_inside = orientation * _cross(edge, previous - clip_start) >= 0
        for current in input_points:
            current_inside = orientation * _cross(edge, current - clip_start) >= 0
            if current_inside != previous_inside:
                direction = current - previous
                denominator = _cross(edge, direction)
                if denominator != 0:
                    factor = _cross(edge, clip_start - previous) / denominator
                    output_points.append(previous + factor * direction)
            if current_inside:
                output_points.append(current)
            previous = current
            previous_inside = current_inside
        output = np.asarray(output_points, dtype=np.float64)

    if len(output) == 0:
        dtype = np.asarray(first).dtype
        result_dtype = dtype if np.issubdtype(dtype, np.floating) else np.float32
        return 0.0, np.empty((0, 1, 2), dtype=result_dtype)

    output = output[np.r_[True, np.any(np.diff(output, axis=0) != 0, axis=1)]]
    if len(output) > 1 and np.array_equal(output[0], output[-1]):
        output = output[:-1]
    area = _contour_area(output)
    dtype = np.asarray(first).dtype
    result_dtype = dtype if np.issubdtype(dtype, np.floating) else np.float32
    return area, output.astype(result_dtype, copy=False).reshape(-1, 1, 2)


def _fill_poly(
    image: npt.NDArray[Any],
    polygons: list[npt.NDArray[Any]],
    color: Any,
    line_type: int = 8,
    shift: int = 0,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """Fill integer polygons for the mask and polygon conversion consumers."""
    if shift != 0:
        raise ValueError("Only unshifted polygon coordinates are supported")
    if image.ndim not in (2, 3):
        raise ValueError("fillPoly expects a two- or three-dimensional image")
    del line_type

    values = np.asarray(image)
    for polygon in polygons:
        points = _as_points(polygon)
        if len(points) < 3:
            continue
        points = points + np.asarray(offset, dtype=np.float64)
        min_x = max(0, int(np.floor(points[:, 0].min())))
        max_x = min(values.shape[1] - 1, int(np.ceil(points[:, 0].max())))
        min_y = max(0, int(np.floor(points[:, 1].min())))
        max_y = min(values.shape[0] - 1, int(np.ceil(points[:, 1].max())))
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if not _point_in_polygon((x, y), points):
                    continue
                if values.ndim == 2:
                    values[y, x] = color[0] if np.ndim(color) else color
                else:
                    values[y, x] = color
