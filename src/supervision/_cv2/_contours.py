"""Private Suzuki-Abe contour fallback."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from supervision._cv2.constants import _CHAIN_APPROX_SIMPLE, _RETR_TREE

_NEIGHBORS = (
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _follow_border(
    image: npt.NDArray[np.bool_],
    labels: npt.NDArray[np.int32],
    start: tuple[int, int],
    previous: tuple[int, int],
    border_number: int,
) -> list[tuple[int, int]]:
    """Trace one border using the Suzuki-Abe neighborhood walk."""
    rows, columns = image.shape

    def is_foreground(row: int, column: int) -> bool:
        """Check a pixel while treating the image boundary as background."""
        return 0 <= row < rows and 0 <= column < columns and bool(image[row, column])

    direction = _NEIGHBORS.index((previous[0] - start[0], previous[1] - start[1]))
    first_direction = -1
    for offset in range(1, 9):
        candidate_direction = (direction + offset) % 8
        delta_row, delta_column = _NEIGHBORS[candidate_direction]
        if is_foreground(start[0] + delta_row, start[1] + delta_column):
            first_direction = candidate_direction
            break
    if first_direction < 0:
        labels[start] = -border_number
        return [start]

    first_neighbor = (
        start[0] + _NEIGHBORS[first_direction][0],
        start[1] + _NEIGHBORS[first_direction][1],
    )
    contour: list[tuple[int, int]] = []
    previous_point, current = first_neighbor, start
    while True:
        direction = _NEIGHBORS.index(
            (previous_point[0] - current[0], previous_point[1] - current[1])
        )
        east_zero = False
        next_point: tuple[int, int] | None = None
        for offset in range(1, 9):
            candidate_direction = (direction - offset) % 8
            delta_row, delta_column = _NEIGHBORS[candidate_direction]
            row = current[0] + delta_row
            column = current[1] + delta_column
            if is_foreground(row, column):
                next_point = (row, column)
                break
            if candidate_direction == 0:
                east_zero = True

        if east_zero:
            labels[current] = -border_number
        elif labels[current] == 0:
            labels[current] = border_number
        contour.append(current)

        if next_point == start and current == first_neighbor and len(contour) > 1:
            return contour
        if next_point is None:
            return contour
        previous_point, current = current, next_point
        if len(contour) > 4 * image.size:
            raise RuntimeError("Contour border tracing did not converge")


def _trace_borders(mask: npt.NDArray[np.bool_]) -> list[np.ndarray]:
    """Trace all foreground and hole borders in raster candidate order."""
    image = np.ascontiguousarray(mask, dtype=bool)
    labels = np.zeros(image.shape, dtype=np.int32)
    left_zero = image & ~np.pad(image[:, :-1], ((0, 0), (1, 0)))
    right_zero = image & ~np.pad(image[:, 1:], ((0, 0), (0, 1)))
    candidates = np.argwhere(left_zero | right_zero)
    borders: list[np.ndarray] = []
    border_number = 1
    for row, column in candidates:
        row, column = int(row), int(column)
        if left_zero[row, column] and labels[row, column] == 0:
            border_number += 1
            border = _follow_border(
                image, labels, (row, column), (row, column - 1), border_number
            )
            borders.append(np.array([(column, row) for row, column in border]))
        elif right_zero[row, column] and labels[row, column] >= 0:
            border_number += 1
            border = _follow_border(
                image, labels, (row, column), (row, column + 1), border_number
            )
            borders.append(np.array([(column, row) for row, column in border]))
    return borders


def _reverse_preserving_start(contour: np.ndarray) -> np.ndarray:
    """Reverse a traced contour while retaining its Suzuki-Abe start pixel."""
    if len(contour) < 2:
        return contour
    return np.concatenate((contour[:1], contour[:0:-1]))


def _compress_contour(contour: np.ndarray) -> np.ndarray:
    """Apply OpenCV's collinear-run compression for SIMPLE contours."""
    contour = _reverse_preserving_start(contour)
    if len(contour) < 3:
        return contour
    keep: list[np.ndarray] = []
    for index, point in enumerate(contour):
        previous = point - contour[index - 1]
        following = contour[(index + 1) % len(contour)] - point
        if (
            np.any(previous)
            and np.any(following)
            and np.array_equal(np.sign(previous), np.sign(following))
        ):
            continue
        keep.append(point)
    return np.asarray(keep, dtype=np.int32)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return whether a point is inside a polygon using an even-odd walk."""
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x_current, y_current = float(current[0]), float(current[1])
        x_previous, y_previous = float(previous[0]), float(previous[1])
        crosses = (y_current > y) != (y_previous > y)
        if crosses:
            intersection = (x_previous - x_current) * (y - y_current) / (
                y_previous - y_current
            ) + x_current
            if x < intersection:
                inside = not inside
        previous = current
    return inside


def _interior_point(contour: np.ndarray) -> np.ndarray:
    """Return a point guaranteed to lie inside a simple contour.

    The polygon centroid can fall outside concave contours, which would corrupt
    the parent/child hierarchy inferred by ``_parents``. Contour vertices are
    integer pixel coordinates, so a horizontal scanline at a half-integer height
    never passes through a vertex; the midpoint of its first edge-crossing span
    is therefore an interior sample. Falls back to the centroid for degenerate
    (sub-triangle or zero-height) contours where no such span exists.
    """
    centroid = cast(np.ndarray, contour.mean(axis=0))
    if len(contour) < 3:
        return centroid
    y_scan = float(np.floor(centroid[1])) + 0.5
    crossings: list[float] = []
    previous = contour[-1]
    for current in contour:
        x_current, y_current = float(current[0]), float(current[1])
        x_previous, y_previous = float(previous[0]), float(previous[1])
        if (y_current > y_scan) != (y_previous > y_scan):
            intersection = x_previous + (x_current - x_previous) * (
                y_scan - y_previous
            ) / (y_current - y_previous)
            crossings.append(intersection)
        previous = current
    if len(crossings) < 2:
        return centroid
    crossings.sort()
    return np.array([(crossings[0] + crossings[1]) / 2.0, y_scan])


def _parents(contours: list[np.ndarray]) -> list[int]:
    """Infer the smallest containing contour for each traced border."""
    bounds = [
        (
            contour[:, 0].min(),
            contour[:, 1].min(),
            contour[:, 0].max(),
            contour[:, 1].max(),
        )
        for contour in contours
    ]
    bounding_areas = [
        (x_max - x_min + 1) * (y_max - y_min + 1)
        for x_min, y_min, x_max, y_max in bounds
    ]
    areas = [abs(_polygon_area(contour)) for contour in contours]
    parents = [-1] * len(contours)
    for index, contour in enumerate(contours):
        point = _interior_point(contour)
        candidates = [
            candidate
            for candidate, polygon in enumerate(contours)
            if candidate != index
            and (
                areas[candidate] > areas[index]
                or (
                    areas[candidate] == areas[index]
                    and bounding_areas[candidate] > bounding_areas[index]
                )
            )
            and _point_in_polygon(point, polygon)
        ]
        if candidates:
            parents[index] = min(
                candidates,
                key=lambda candidate: (areas[candidate], bounding_areas[candidate]),
            )
    return parents


def _polygon_area(contour: np.ndarray) -> float:
    """Compute a signed area for hierarchy construction."""
    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _order_contours(contours: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Order contours and construct OpenCV-shaped tree hierarchy metadata."""
    parents = _parents(contours)
    children = {
        parent: [index for index, value in enumerate(parents) if value == parent]
        for parent in range(-1, len(contours))
    }
    ordered_indices: list[int] = []

    def visit(index: int) -> None:
        """Append a contour followed by its children in scan order."""
        ordered_indices.append(index)
        for child in reversed(children.get(index, [])):
            visit(child)

    for index in reversed(children.get(-1, [])):
        visit(index)

    ordered = [contours[index] for index in ordered_indices]
    index_map = {
        original: position for position, original in enumerate(ordered_indices)
    }
    remapped_parents = [
        -1 if parents[index] < 0 else index_map[parents[index]]
        for index in ordered_indices
    ]
    hierarchy = np.full((1, len(ordered), 4), -1, dtype=np.int32)
    for parent in range(-1, len(ordered)):
        siblings = [
            index for index, value in enumerate(remapped_parents) if value == parent
        ]
        for position, index in enumerate(siblings):
            if position + 1 < len(siblings):
                hierarchy[0, index, 0] = siblings[position + 1]
            if position > 0:
                hierarchy[0, index, 1] = siblings[position - 1]
        if parent >= 0 and siblings:
            hierarchy[0, parent, 2] = siblings[0]
    for index, parent in enumerate(remapped_parents):
        hierarchy[0, index, 3] = parent
    return ordered, hierarchy


def _find_contours(
    image: npt.NDArray[Any], mode: int, method: int
) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32] | None]:
    """Find contours for the supported tree and SIMPLE modes."""
    if mode != _RETR_TREE:
        raise ValueError("Only RETR_TREE is supported by the fallback")
    if method != _CHAIN_APPROX_SIMPLE:
        raise ValueError("Only CHAIN_APPROX_SIMPLE is supported by the fallback")
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("Contour input must be a two-dimensional image")
    traced = [_compress_contour(contour) for contour in _trace_borders(values != 0)]
    if not traced:
        return [], None
    contours, hierarchy = _order_contours(traced)
    return [contour.reshape(-1, 1, 2) for contour in contours], hierarchy
