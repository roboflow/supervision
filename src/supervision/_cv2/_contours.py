"""Private Suzuki-Abe contour fallback."""

from __future__ import annotations

from typing import Any

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
    """Trace all borders as int32 points, ordered like OpenCV's RETR_TREE.

    The raster scan follows Suzuki-Abe: ``last_border`` is the most recent border
    the scan crossed on the current row, which decides the parent of each new
    border. OpenCV links each new contour as the first child of its parent, so
    its list is a depth-first pre-order walk with siblings in reverse discovery
    order; the walk below reproduces that order.
    """
    image = np.ascontiguousarray(mask, dtype=bool)
    labels = np.zeros(image.shape, dtype=np.int32)
    left_zero = image & ~np.pad(image[:, :-1], ((0, 0), (1, 0)))
    right_zero = image & ~np.pad(image[:, 1:], ((0, 0), (0, 1)))
    starts = left_zero | right_zero
    # Border 1 is the image frame: a hole border around the whole image.
    is_hole = {1: True}
    parents = {1: 0}
    children: dict[int, list[int]] = {1: []}
    points: dict[int, np.ndarray] = {}
    border_number = 1
    for row in range(image.shape[0]):
        last_border = 1
        column = 0
        while True:
            # Only border starts and already traced pixels affect the scan.
            events = np.flatnonzero(starts[row, column:] | (labels[row, column:] != 0))
            if len(events) == 0:
                break
            column += int(events[0])
            if left_zero[row, column] and labels[row, column] == 0:
                previous, hole = (row, column - 1), False
            elif right_zero[row, column] and labels[row, column] >= 0:
                previous, hole = (row, column + 1), True
                # A hole starting on a traced pixel lies just inside that border.
                if labels[row, column] > 0:
                    last_border = int(labels[row, column])
            else:
                previous = None
            if previous is not None:
                border_number += 1
                parent = last_border
                if is_hole[last_border] == hole:
                    parent = parents[last_border]
                border = _follow_border(
                    image, labels, (row, column), previous, border_number
                )
                is_hole[border_number] = hole
                parents[border_number] = parent
                children[border_number] = []
                children[parent].append(border_number)
                points[border_number] = np.array(
                    [(point_column, point_row) for point_row, point_column in border],
                    dtype=np.int32,
                )
            if labels[row, column] != 0:
                last_border = abs(int(labels[row, column]))
            column += 1
            if column >= image.shape[1]:
                break

    borders: list[np.ndarray] = []
    stack = list(children[1])
    while stack:
        number = stack.pop()
        borders.append(points[number])
        stack.extend(children[number])
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


def _find_contours(
    image: npt.NDArray[Any], mode: int, method: int
) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32] | None]:
    """Find contours for the supported tree and SIMPLE modes in OpenCV order."""
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
    return [contour.reshape(-1, 1, 2) for contour in traced], None
