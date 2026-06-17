from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def filter_polygons_by_area(
    polygons: list[npt.NDArray[np.number]],
    min_area: float | None = None,
    max_area: float | None = None,
) -> list[npt.NDArray[np.number]]:
    """
    Filters a list of polygons based on their area.

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
    """
    if min_area is None and max_area is None:
        return polygons
    ares = [cv2.contourArea(polygon) for polygon in polygons]
    return [
        polygon
        for polygon, area in zip(polygons, ares)
        if (min_area is None or area >= min_area)
        and (max_area is None or area <= max_area)
    ]


def approximate_polygon(
    polygon: npt.NDArray[np.number], percentage: float, epsilon_step: float = 0.05
) -> npt.NDArray[np.number]:
    """
    Approximates a given polygon by reducing a certain percentage of points.

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
    """

    if percentage < 0 or percentage >= 1:
        raise ValueError("Percentage must be in the range [0, 1).")
    if epsilon_step <= 0:
        raise ValueError("epsilon_step must be positive.")

    target_points = max(int(len(polygon) * (1 - percentage)), 3)

    if len(polygon) <= target_points:
        return polygon

    epsilon: float = 0
    approximated_points = polygon
    while len(approximated_points) > target_points:
        epsilon += epsilon_step
        candidate = np.squeeze(cv2.approxPolyDP(polygon, epsilon, closed=True), axis=1)
        # Stop before the approximation collapses below a valid polygon; keep the
        # last result with at least three points.
        if len(candidate) < 3:
            break
        approximated_points = candidate

    return approximated_points
