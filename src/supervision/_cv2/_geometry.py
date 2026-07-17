"""Private polygon geometry fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


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


def _simplify_slices(
    points: npt.NDArray[np.float64], epsilon_squared: float, closed: bool
) -> npt.NDArray[np.float64]:
    """Run OpenCV's stack-based Douglas-Peucker slice traversal."""
    count = len(points)
    stack: list[tuple[int, int]] = []
    output: list[npt.NDArray[np.float64]] = []

    if closed or np.array_equal(points[0], points[-1]):
        closed = True
        iterations = 3 if not np.array_equal(points[0], points[-1]) else 1
        position = 0
        right_start = 0
        start_point = points[0]
        within_epsilon = False
        for _ in range(iterations):
            position = (position + right_start) % count
            start_point = points[position]
            maximum_distance = 0.0
            right_start = 0
            for offset in range(1, count):
                point = points[(position + offset) % count]
                difference = point - start_point
                distance = float(np.dot(difference, difference))
                if distance > maximum_distance:
                    maximum_distance = distance
                    right_start = offset
            within_epsilon = maximum_distance <= epsilon_squared

        if within_epsilon:
            output.append(start_point)
        else:
            split = (position + right_start) % count
            stack.extend(((split, position), (position, split)))
    else:
        stack.append((0, count - 1))

    while stack:
        start, end = stack.pop()
        start_point = points[start]
        end_point = points[end]
        position = (start + 1) % count
        maximum_distance = 0.0
        split = start

        if position != end:
            segment = end_point - start_point
            while position != end:
                point = points[position]
                distance = abs(
                    float(
                        (point[1] - start_point[1]) * segment[0]
                        - (point[0] - start_point[0]) * segment[1]
                    )
                )
                if distance > maximum_distance:
                    maximum_distance = distance
                    split = position
                position = (position + 1) % count
            segment_length_squared = float(np.dot(segment, segment))
            within_epsilon = (
                maximum_distance * maximum_distance
                <= epsilon_squared * segment_length_squared
            )
        else:
            within_epsilon = True

        if within_epsilon:
            output.append(start_point)
        else:
            stack.extend(((split, end), (start, split)))

    if not closed:
        output.append(points[-1])
    return np.asarray(output, dtype=np.float64)


def _cleanup_approximation(
    points: npt.NDArray[np.float64], epsilon_squared: float, closed: bool
) -> npt.NDArray[np.float64]:
    """Remove OpenCV's final near-collinear points from an approximation."""
    count = len(points)
    if count <= 2:
        return points

    destination = points.copy()
    new_count = count
    position = count - 1 if closed else 0
    start_point = destination[position]
    position = (position + 1) % count
    write_position = position
    point = destination[position]
    position = (position + 1) % count
    index = 0 if closed else 1
    stop = count if closed else count - 1

    while index < stop and new_count > 2:
        end_point = destination[position]
        position = (position + 1) % count
        segment = end_point - start_point
        offset = point - start_point
        distance = abs(float(offset[0] * segment[1] - offset[1] * segment[0]))
        inner_product = float(np.dot(offset, end_point - point))
        removable = (
            distance * distance
            <= 0.5 * epsilon_squared * float(np.dot(segment, segment))
            and segment[0] != 0
            and segment[1] != 0
            and inner_product >= 0
        )
        if removable:
            new_count -= 1
            destination[write_position] = end_point
            start_point = end_point
            write_position = (write_position + 1) % count
            point = destination[position]
            position = (position + 1) % count
            index += 2
            continue

        destination[write_position] = point
        start_point = point
        write_position = (write_position + 1) % count
        point = end_point
        index += 1

    if not closed:
        destination[write_position] = point
    return np.asarray(destination[:new_count], dtype=np.float64)


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

    epsilon_squared = float(epsilon) ** 2
    simplified = _simplify_slices(points, epsilon_squared, closed)
    simplified = _cleanup_approximation(simplified, epsilon_squared, closed)

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
