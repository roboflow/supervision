"""Seed frame of an oriented box: the local axes that its first edge defines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _offsets_from_origin(
    points: npt.NDArray[np.number], origin: npt.NDArray[np.number]
) -> npt.NDArray[np.float64]:
    """Return `points - origin` as float64 without losing integer precision.

    Integer coordinates are subtracted as Python integers, so large values (for
    example geospatial or stitched frames) reduce to exact local extents and
    unsigned dtypes cannot wrap around for points below the origin.

    Args:
        points: Points of any shape ending in `(2,)`.
        origin: The point to measure from, shape `(2,)`.

    Returns:
        The offsets from `origin`, float64, same shape as `points`.
    """
    if np.issubdtype(points.dtype, np.integer):
        return np.asarray(
            points.astype(object) - origin.astype(object), dtype=np.float64
        )
    return points.astype(np.float64, copy=False) - origin.astype(np.float64, copy=False)


def seed_frame(
    seed_corners: npt.NDArray[np.number],
) -> tuple[float, float]:
    """Return the cosine and sine of the angle of an oriented box's first edge.

    The first edge (corner 0 to corner 1) defines the box's local x axis. Using it
    avoids `cv2.minAreaRect` surprises such as a 90-degree flip for wide boxes.
    The origin of the frame is corner 0.

    Args:
        seed_corners: Corners of the seed box, shape `(4, 2)`.

    Returns:
        A tuple `(cos, sin)` of the first edge's direction angle.
    """
    edge = _offsets_from_origin(seed_corners[1], seed_corners[0])
    angle = float(np.arctan2(float(edge[1]), float(edge[0])))
    return float(np.cos(angle)), float(np.sin(angle))


def corners_to_frame(
    corners: npt.NDArray[np.number],
    origin: npt.NDArray[np.number],
    cos: float,
    sin: float,
) -> npt.NDArray[np.float64]:
    """Express points in a seed frame: translate to its origin, then de-rotate.

    Args:
        corners: Points of any shape ending in `(2,)`.
        origin: Origin of the frame, the seed's corner 0, shape `(2,)`.
        cos: Cosine of the frame angle, from :func:`seed_frame`.
        sin: Sine of the frame angle, from :func:`seed_frame`.

    Returns:
        The points in the local frame, float64, same shape as `corners`.
    """
    to_local = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    return _offsets_from_origin(corners, origin) @ to_local


def frame_to_corners(
    local: npt.NDArray[np.float64],
    origin: npt.NDArray[np.number],
    cos: float,
    sin: float,
) -> npt.NDArray[np.float64]:
    """Map points from a seed frame back to the image: rotate, then translate.

    Args:
        local: Points in the local frame, any shape ending in `(2,)`.
        origin: Origin of the frame, the seed's corner 0, shape `(2,)`.
        cos: Cosine of the frame angle, from :func:`seed_frame`.
        sin: Sine of the frame angle, from :func:`seed_frame`.

    Returns:
        The points in image coordinates, float64, same shape as `local`.
    """
    to_world = np.array([[cos, sin], [-sin, cos]], dtype=np.float64)
    world: npt.NDArray[np.float64] = local @ to_world + origin.astype(np.float64)
    return world
