from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from supervision.key_points.core import KeyPoints


@dataclass(frozen=True)
class JointAngle:
    """A named joint, given as three key point positions, whose angle is measured.

    The angle is the interior angle at vertex ``b`` between the rays ``b -> a`` and
    ``b -> c``, between 0 and 180 degrees. For example, a left knee in the COCO
    17-keypoint layout is ``JointAngle("left_knee", a=11, b=13, c=15)``: hip, knee,
    ankle.

    Note:
        ``a``, ``b`` and ``c`` are zero-based positions along the second axis of
        :attr:`~supervision.key_points.core.KeyPoints.xy`. Skeleton edges in
        :class:`~supervision.key_points.skeletons.Skeleton` use one-based indices
        instead.

    Attributes:
        name: A label for the joint. It appears in error messages and lets callers
            name the columns of the array returned by
            :func:`~supervision.key_points.angles.joint_angles`.
        a: Position of the first end point.
        b: Position of the vertex, where the angle is measured.
        c: Position of the second end point.

    Raises:
        TypeError: If a position is not an integer.
        ValueError: If a position is negative or the three positions are not
            distinct.

    Example:
        ```pycon
        >>> from supervision.key_points.angles import JointAngle
        >>> JointAngle(name="left_knee", a=11, b=13, c=15)
        JointAngle(name='left_knee', a=11, b=13, c=15)

        ```
    """

    name: str
    a: int
    b: int
    c: int

    def __post_init__(self) -> None:
        """Reject positions that cannot define an angle."""
        positions = (self.a, self.b, self.c)
        # bool is an int subclass, but True/False as a position is always a mistake.
        if any(
            isinstance(p, bool) or not isinstance(p, (int, np.integer))
            for p in positions
        ):
            raise TypeError(
                f"JointAngle {self.name!r} positions must be integers, got {positions}."
            )
        if min(positions) < 0:
            raise ValueError(
                f"JointAngle {self.name!r} positions must be non-negative, "
                f"got {positions}."
            )
        if len(set(positions)) != 3:
            raise ValueError(
                f"JointAngle {self.name!r} positions must be distinct, got {positions}."
            )


def joint_angles(
    key_points: KeyPoints, angles: Sequence[JointAngle]
) -> npt.NDArray[np.float64]:
    """Compute joint angles in degrees for every skeleton.

    Each angle is the interior angle at the vertex of a
    :class:`~supervision.key_points.angles.JointAngle`, measured in the image plane
    from the x/y coordinates; an extra coordinate such as depth is ignored. These
    are 2D image-plane angles, not anatomical 3D joint angles: when a limb is
    foreshortened, for example a forearm pointing towards the camera, the two can
    differ widely. The interior angle never exceeds 180 degrees, so a joint that
    bends through straight reads the same angle on either side.

    An angle is ``NaN`` when one of its three points cannot be used, so no value is
    invented: a point is unusable when it is hidden by
    :attr:`~supervision.key_points.core.KeyPoints.visible`, has a non-finite x/y
    coordinate, or sits at ``[0, 0]``, the marker some models use for a missing
    point. These are the rules
    :meth:`~supervision.key_points.core.KeyPoints.as_detections` applies. The angle
    is also ``NaN`` when an end point coincides with the vertex.

    Args:
        key_points: Key points of ``n`` skeletons with ``m`` points each.
        angles: The joints to measure.

    Returns:
        An array of shape ``(n, len(angles))`` with one row per skeleton and one
            column per joint, in degrees.

    Raises:
        ValueError: If a joint refers to a position outside the ``m`` key points of
            a non-empty ``key_points``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> from supervision.key_points.angles import JointAngle
        >>> key_points = sv.KeyPoints(
        ...     xy=np.array(
        ...         [
        ...             [[100, 50], [100, 100], [150, 100]],
        ...             [[0, 0], [100, 100], [150, 100]],
        ...         ],
        ...         dtype=np.float32,
        ...     )
        ... )
        >>> knee = JointAngle(name="knee", a=0, b=1, c=2)
        >>> sv.joint_angles(key_points, [knee])
        array([[90.],
               [nan]])

        ```
    """
    num_skeletons, num_points = key_points.xy.shape[:2]
    if num_skeletons == 0:
        return np.empty((0, len(angles)), dtype=np.float64)

    for angle in angles:
        if max(angle.a, angle.b, angle.c) >= num_points:
            raise ValueError(
                f"JointAngle {angle.name!r} refers to position "
                f"{max(angle.a, angle.b, angle.c)}, but each skeleton has "
                f"{num_points} key points."
            )

    result = np.full((num_skeletons, len(angles)), np.nan, dtype=np.float64)
    xy = key_points.xy[..., :2].astype(np.float64)
    # Same rule as `as_detections`: [0, 0] marks a missing point, non-finite values
    # cannot be measured, and `visible` hides points the caller does not trust.
    usable = ~np.all(xy == 0, axis=-1) & np.isfinite(xy).all(axis=-1)
    if key_points.visible is not None:
        usable &= key_points.visible.astype(bool)

    for column, angle in enumerate(angles):
        ray_a = xy[:, angle.a] - xy[:, angle.b]
        ray_c = xy[:, angle.c] - xy[:, angle.b]
        measurable = usable[:, angle.a] & usable[:, angle.b] & usable[:, angle.c]
        measurable &= np.any(ray_a != 0, axis=1) & np.any(ray_c != 0, axis=1)
        # atan2(|cross|, dot) stays accurate near 0 and 180 degrees, where arccos of
        # the normalised dot product loses precision.
        ray_a, ray_c = ray_a[measurable], ray_c[measurable]
        cross = np.abs(ray_a[:, 0] * ray_c[:, 1] - ray_a[:, 1] * ray_c[:, 0])
        dot = np.einsum("ij,ij->i", ray_a, ray_c)
        result[measurable, column] = np.degrees(np.arctan2(cross, dot))
    return result
