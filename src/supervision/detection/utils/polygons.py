from typing import cast

import numpy as np
import numpy.typing as npt

from supervision import _cv2 as cv2


def _to_local_cv2_polygon(
    polygon: npt.NDArray[np.number],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.number]]:
    """Translate a polygon to a local float32 coordinate system for OpenCV.

    Integer subtraction uses object arithmetic so unsigned and signed inputs do not wrap
    before conversion. Translating first preserves representable local geometry when
    absolute coordinates exceed float32 precision.
    """
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros(2, dtype=polygon.dtype)

    origin = polygon.min(axis=0)
    if np.issubdtype(polygon.dtype, np.integer):
        local_polygon = np.asarray(
            polygon.astype(object) - origin.astype(object), dtype=np.float32
        )
    else:
        local_polygon = (
            polygon.astype(np.float64, copy=False)
            - origin.astype(np.float64, copy=False)
        ).astype(np.float32)
    return local_polygon, origin


def filter_polygons_by_area(
    polygons: list[npt.NDArray[np.number]],
    min_area: float | None = None,
    max_area: float | None = None,
) -> list[npt.NDArray[np.number]]:
    """Filters a list of polygons based on their area.

    Args:
        polygons: A list of polygons, where each polygon is
            represented by a NumPy array of shape `(N, 2)`,
            containing the `x`, `y` coordinates of the points.
        min_area: The minimum area threshold.
            Only polygons with an area greater than or equal to this value
            will be included in the output. If set to None,
            no minimum area constraint will be applied.
        max_area: The maximum area threshold.
            Only polygons with an area less than or equal to this value
            will be included in the output. If set to None,
            no maximum area constraint will be applied.

    Returns:
        A new list of polygons containing only those with
            areas within the specified thresholds.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> small = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        >>> big = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> sv.filter_polygons_by_area([small, big], min_area=50)
        [array([[ 0,  0],
               [10,  0],
               [10, 10],
               [ 0, 10]])]

        ```
    """
    if min_area is None and max_area is None:
        return polygons
    areas = [cv2.contourArea(_to_local_cv2_polygon(polygon)[0]) for polygon in polygons]
    return [
        polygon
        for polygon, area in zip(polygons, areas)
        if (min_area is None or area >= min_area)
        and (max_area is None or area <= max_area)
    ]


def approximate_polygon(
    polygon: npt.NDArray[np.number], percentage: float, epsilon_step: float = 0.05
) -> npt.NDArray[np.number]:
    """Approximates a given polygon by reducing a certain percentage of points.

    This function uses the Ramer-Douglas-Peucker algorithm to simplify the input
    polygon by reducing the number of points while preserving the general shape.

    Args:
        polygon: A 2D NumPy array of shape `(N, 2)` containing
            the `x`, `y` coordinates of the input polygon's points.
        percentage: The percentage of points to be removed from the
            input polygon, in the range `[0, 1)`.
        epsilon_step: Approximation accuracy step, must be positive.
            Epsilon is the maximum distance between the original curve
            and its approximation.

    Returns:
        A new 2D NumPy array of shape `(M, 2)` containing the `x`, `y`
            coordinates of the approximated polygon's points. `M` is at most
            `floor(N * (1 - percentage))` (minimum 3). Because epsilon
            advances in discrete `epsilon_step` increments, `M` may be
            noticeably smaller than the budget when a single step crosses
            the target band. At least 3 points are always kept; when further
            simplification would collapse the polygon below 3 points the last
            valid approximation is returned and `M` may exceed the budget.

    Raises:
        ValueError: If `percentage` is outside the range `[0, 1)`.
        ValueError: If `epsilon_step` is not positive.

    Examples:
        Reduce a polygon to at most half its original point count:

        ```pycon
        >>> import numpy as np
        >>> polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10],
        ...                     [5, 10], [5, 5], [3, 7], [1, 9]])
        >>> result = approximate_polygon(polygon, percentage=0.5)
        >>> result.shape[1]
        2
        >>> len(result) <= max(int(len(polygon) * 0.5), 3)
        True

        Polygon already at or below target — returned unchanged:

        >>> tiny = np.array([[0, 0], [5, 0], [2, 4]])
        >>> approximate_polygon(tiny, percentage=0.5) is tiny
        True

        ```
    """

    if percentage < 0 or percentage >= 1:
        raise ValueError("Percentage must be in the range [0, 1).")
    if epsilon_step <= 0:
        raise ValueError("epsilon_step must be positive.")

    target_points = max(int(len(polygon) * (1 - percentage)), 3)

    if len(polygon) <= target_points:
        return polygon

    original_dtype = polygon.dtype
    cv_polygon, origin = _to_local_cv2_polygon(polygon)

    epsilon: float = 0
    approximated_points = polygon
    while len(approximated_points) > target_points:
        epsilon += epsilon_step
        candidate = np.squeeze(
            cv2.approxPolyDP(cv_polygon, epsilon, closed=True), axis=1
        )
        # Stop before the approximation collapses below a valid polygon; keep the
        # last result with at least three points.
        if len(candidate) < 3:
            break
        if np.issubdtype(original_dtype, np.integer):
            approximated_points = np.asarray(
                candidate.astype(original_dtype).astype(object) + origin.astype(object),
                dtype=original_dtype,
            )
        else:
            approximated_points = (
                candidate.astype(np.float64) + origin.astype(np.float64)
            ).astype(original_dtype)

    return approximated_points


def _find_min_distance_indices(
    polygon1: npt.NDArray[np.number], polygon2: npt.NDArray[np.number]
) -> tuple[int, int]:
    """Find the indices of the closest pair of vertices between two polygons.

    Args:
        polygon1: Vertex coordinates of the first polygon with shape ``(N, 2)``.
        polygon2: Vertex coordinates of the second polygon with shape ``(M, 2)``.

    Returns:
        A tuple of ``(idx1, idx2)`` containing the index in ``polygon1`` and
        the index in ``polygon2`` of the mutually closest vertices.
    """
    diff = polygon1[:, np.newaxis, :] - polygon2[np.newaxis, :, :]
    distances_sq = np.sum(diff**2, axis=-1)
    flat_idx = int(np.argmin(distances_sq))
    idx1, idx2 = np.unravel_index(flat_idx, distances_sq.shape)
    return int(idx1), int(idx2)


def merge_multi_segment(
    segments: list[npt.NDArray[np.number]],
) -> npt.NDArray[np.number]:
    """Merge multiple disconnected polygon segments into a single continuous polygon.

    This function connects adjacent segments at their closest pair of vertices
    with a zero-width seam, producing a single closed polygon suitable for YOLO
    instance segmentation and similar formats that require a single continuous
    contour per object.

    Args:
        segments: A list of 2D NumPy arrays, where each array has shape ``(N, 2)``
            containing the ``(x, y)`` coordinates of a polygon's vertices.

    Returns:
        A single NumPy array of shape ``(M, 2)`` representing the unified polygon.
        If ``segments`` is empty, returns an empty array of shape ``(0, 2)``. If
        ``segments`` contains only one polygon, returns that polygon unchanged.

    Raises:
        ValueError: If any segment does not have 2D shape with 2 coordinates per vertex.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> p1 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
        >>> p2 = np.array([[35, 5], [45, 5], [45, 15], [35, 15]])
        >>> merged = sv.merge_multi_segment([p1, p2])
        >>> merged.shape
        (10, 2)

        ```
    """
    if not segments:
        return np.empty((0, 2), dtype=np.float32)

    valid_segments: list[npt.NDArray[np.number]] = []
    for s in segments:
        arr = np.asarray(s)
        if arr.ndim != 2 or arr.shape[-1] != 2:
            raise ValueError(f"Each segment must have shape (N, 2); got {arr.shape}.")
        if len(arr) > 0:
            valid_segments.append(arr)

    if not valid_segments:
        return np.empty((0, 2), dtype=np.float32)
    if len(valid_segments) == 1:
        return valid_segments[0]

    orig_dtype = np.result_type(*(s.dtype for s in valid_segments))
    k_segments = len(valid_segments)

    seams: list[tuple[int, int]] = []
    for i in range(1, k_segments):
        idx1, idx2 = _find_min_distance_indices(
            valid_segments[i - 1], valid_segments[i]
        )
        seams.append((idx1, idx2))

    # Segment 0: traverse full contour starting and ending at seam to segment 1
    u0 = seams[0][0]
    s0_rolled = np.roll(valid_segments[0], -u0, axis=0)
    pieces: list[npt.NDArray[np.number]] = [np.concatenate([s0_rolled, s0_rolled[:1]])]

    # Intermediate segments on forward pass: enter at v_in, exit at u_out
    inter_rolled: list[npt.NDArray[np.number]] = []
    inter_offsets: list[int] = []
    for i in range(1, k_segments - 1):
        v_in = seams[i - 1][1]
        u_out = seams[i][0]
        n_points = len(valid_segments[i])
        rolled = np.roll(valid_segments[i], -v_in, axis=0)
        offset = (u_out - v_in) % n_points
        inter_rolled.append(rolled)
        inter_offsets.append(offset)
        pieces.append(rolled[0 : offset + 1])

    # Last segment: traverse full contour starting and ending at seam to segment K-2
    v_last = seams[-1][1]
    s_last_rolled = np.roll(valid_segments[-1], -v_last, axis=0)
    pieces.append(np.concatenate([s_last_rolled, s_last_rolled[:1]]))

    # Intermediate segments on return pass: return from u_out back to v_in
    for i in range(k_segments - 2, 0, -1):
        rolled = inter_rolled[i - 1]
        offset = inter_offsets[i - 1]
        pieces.append(rolled[offset:])
        pieces.append(rolled[:1])

    merged = np.concatenate(pieces, axis=0)
    return cast(npt.NDArray[np.number], merged.astype(orig_dtype, copy=False))
