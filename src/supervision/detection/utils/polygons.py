from __future__ import annotations

from typing import cast

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
    ares = [
        cv2.contourArea(np.asarray(polygon, dtype=np.float32)) for polygon in polygons
    ]
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
        epsilon_step: Approximation accuracy step.
            Epsilon is the maximum distance between the original curve
            and its approximation.

    Returns:
        A new 2D NumPy array of shape `(M, 2)`,
            where `M <= N * (1 - percentage)`, containing
            the `x`, `y` coordinates of the
            approximated polygon's points.
    """

    if percentage < 0 or percentage >= 1:
        raise ValueError("Percentage must be in the range [0, 1).")

    polygon_f32 = np.asarray(polygon, dtype=np.float32)
    target_points = max(int(len(polygon_f32) * (1 - percentage)), 3)

    if len(polygon_f32) <= target_points:
        return cast(npt.NDArray[np.number], polygon_f32)

    epsilon: float = 0
    approximated_points: npt.NDArray[np.float32] = polygon_f32
    while True:
        epsilon += epsilon_step
        new_approximated_points = cast(
            npt.NDArray[np.float32],
            cv2.approxPolyDP(polygon_f32, epsilon, closed=True),
        )
        if len(new_approximated_points) > target_points:
            approximated_points = new_approximated_points
        else:
            break

    return cast(npt.NDArray[np.number], np.squeeze(approximated_points, axis=1))
