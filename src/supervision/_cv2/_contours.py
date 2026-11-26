"""Private Suzuki-Abe contour fallback."""

from __future__ import annotations

from array import array
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

# Maps (dr + 1) * 3 + (dc + 1) -> direction index in `_NEIGHBORS`.
_DIR_LOOKUP = (3, 2, 1, 4, -1, 0, 5, 6, 7)


def _follow_border_fast(
    padded_bytes: bytes,
    labels: array[int],
    stride: int,
    start_r: int,
    start_c: int,
    prev_r: int,
    prev_c: int,
    border_number: int,
    max_steps: int,
) -> list[tuple[int, int]]:
    """Trace one border in 1-pixel padded coordinates using flat Python sequences.

    Operates on flat byte and integer buffers with a 1-pixel zero border so each step
    avoids NumPy scalar indexing, bounds checks, and tuple search. Returns `(x, y)`
    coordinates in unpadded crop space (shifted by `-1`).
    """
    neighbors = _NEIGHBORS
    dir_lookup = _DIR_LOOKUP
    # Flat index deltas corresponding to the 8 directions in `_NEIGHBORS`.
    deltas = (
        1,
        -stride + 1,
        -stride,
        -stride - 1,
        -1,
        stride - 1,
        stride,
        stride + 1,
    )
    start_pos = start_r * stride + start_c
    direction = dir_lookup[(prev_r - start_r + 1) * 3 + (prev_c - start_c + 1)]
    first_direction = -1
    for offset in range(1, 9):
        cand_dir = (direction + offset) & 7
        if padded_bytes[start_pos + deltas[cand_dir]]:
            first_direction = cand_dir
            break
    if first_direction < 0:
        labels[start_pos] = -border_number
        return [(start_c - 1, start_r - 1)]

    dr0, dc0 = neighbors[first_direction]
    first_r, first_c = start_r + dr0, start_c + dc0
    first_pos = start_pos + deltas[first_direction]
    contour: list[tuple[int, int]] = []
    prev_r, prev_c = first_r, first_c
    cur_r, cur_c = start_r, start_c
    cur_pos = start_pos
    neg_border = -border_number

    while True:
        direction = dir_lookup[(prev_r - cur_r + 1) * 3 + (prev_c - cur_c + 1)]
        east_zero = False
        next_dir = -1
        for offset in range(1, 9):
            cand_dir = (direction - offset) & 7
            if padded_bytes[cur_pos + deltas[cand_dir]]:
                next_dir = cand_dir
                break
            if cand_dir == 0:
                east_zero = True

        if east_zero:
            labels[cur_pos] = neg_border
        elif labels[cur_pos] == 0:
            labels[cur_pos] = border_number
        contour.append((cur_c - 1, cur_r - 1))

        if next_dir < 0:
            return contour
        ndr, ndc = neighbors[next_dir]
        next_r, next_c = cur_r + ndr, cur_c + ndc
        next_pos = cur_pos + deltas[next_dir]
        if next_pos == start_pos and cur_pos == first_pos and len(contour) > 1:
            return contour
        prev_r, prev_c = cur_r, cur_c
        cur_r, cur_c, cur_pos = next_r, next_c, next_pos
        if len(contour) > max_steps:
            raise RuntimeError("Contour border tracing did not converge")


def _trace_borders(image: npt.NDArray[np.bool_]) -> list[np.ndarray]:
    """Trace all borders as int32 points, ordered like OpenCV's RETR_TREE.

    The raster scan follows Suzuki-Abe: ``last_border`` is the most recent border
    the scan crossed on the current row, which decides the parent of each new
    border. OpenCV links each new contour as the first child of its parent, so
    its list is a depth-first pre-order walk with siblings in reverse discovery
    order; the walk below reproduces that order.
    """
    rows, cols = image.shape
    if rows == 0 or cols == 0:
        return []

    padded_bytes = np.pad(image, 1).tobytes()
    stride = cols + 2
    # Signed labels preserve border orientation without one Python reference per pixel.
    labels = array("i", [0]) * ((rows + 2) * stride)
    max_steps = 4 * image.size

    left_zero = image & ~np.pad(image[:, :-1], ((0, 0), (1, 0)))
    right_zero = image & ~np.pad(image[:, 1:], ((0, 0), (0, 1)))
    # Border 1 is the image frame: a hole border around the whole image.
    is_hole = {1: True}
    parents = {1: 0}
    children: dict[int, list[int]] = {1: []}
    points: dict[int, np.ndarray] = {}
    border_number = 1

    active_rows = np.flatnonzero(left_zero.any(axis=1) | right_zero.any(axis=1))
    for row in active_rows.tolist():
        pr = row + 1
        row_offset = pr * stride
        last_border = 1
        # Only columns with a start pixel or a label touched by a previous border
        # can trigger a new border or update `last_border`.
        row_left = left_zero[row]
        row_right = right_zero[row]
        row_starts = row_left | row_right
        for block_start in range(0, cols, 128):
            block_end = min(block_start + 128, cols)
            if not row_starts[block_start:block_end].any():
                # Even without a start pixel, update `last_border` if any pixel in
                # this block carries a border label from a previous trace.
                seg = labels[row_offset + block_start + 1 : row_offset + block_end + 1]
                if any(seg):
                    for lbl in reversed(seg):
                        if lbl != 0:
                            last_border = abs(lbl)
                            break
                continue
            for column in range(block_start, block_end):
                pos = row_offset + column + 1
                lbl = labels[pos]
                if row_left[column] and lbl == 0:
                    prev_r, prev_c, hole = pr, column, False
                elif row_right[column] and lbl >= 0:
                    prev_r, prev_c, hole = pr, column + 2, True
                    # A hole starting on a traced pixel lies just inside that border.
                    if lbl > 0:
                        last_border = lbl
                else:
                    prev_r = -1
                if prev_r >= 0:
                    border_number += 1
                    parent = last_border
                    if is_hole[last_border] == hole:
                        parent = parents[last_border]
                    border = _follow_border_fast(
                        padded_bytes,
                        labels,
                        stride,
                        pr,
                        column + 1,
                        prev_r,
                        prev_c,
                        border_number,
                        max_steps,
                    )
                    is_hole[border_number] = hole
                    parents[border_number] = parent
                    children[border_number] = []
                    children[parent].append(border_number)
                    points[border_number] = np.array(border, dtype=np.int32)
                    lbl = labels[pos]
                if lbl != 0:
                    last_border = abs(lbl)

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
    prev_diff = contour - np.roll(contour, 1, axis=0)
    next_diff = np.roll(contour, -1, axis=0) - contour
    prev_dx = prev_diff[:, 0]
    prev_dy = prev_diff[:, 1]
    next_dx = next_diff[:, 0]
    next_dy = next_diff[:, 1]
    collinear = (
        ((prev_dx != 0) | (prev_dy != 0))
        & ((next_dx != 0) | (next_dy != 0))
        & (np.sign(prev_dx) == np.sign(next_dx))
        & (np.sign(prev_dy) == np.sign(next_dy))
    )
    return np.ascontiguousarray(contour[~collinear], dtype=np.int32)


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
    binary = values != 0
    row_any = binary.any(axis=1)
    if not row_any.any():
        return [], None
    height, width = binary.shape
    y1 = int(row_any.argmax())
    y2 = height - 1 - int(row_any[::-1].argmax())
    band = binary[y1 : y2 + 1]
    col_any = band.any(axis=0)
    x1 = int(col_any.argmax())
    x2 = width - 1 - int(col_any[::-1].argmax())
    crop = band[:, x1 : x2 + 1]

    traced = [_compress_contour(contour) for contour in _trace_borders(crop)]
    if not traced:
        return [], None
    if x1 != 0 or y1 != 0:
        offset = np.array([x1, y1], dtype=np.int32)
        traced = [contour + offset for contour in traced]
    return [contour.reshape(-1, 1, 2) for contour in traced], None
