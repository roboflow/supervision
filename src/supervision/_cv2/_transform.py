"""Private transform and filter fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from supervision._cv2._common import _cast_array_like_opencv

_FLT_EPSILON = float(np.finfo(np.float32).eps)
_QUAD_SHAPES = ((4, 2), (4, 1, 2), (1, 4, 2))


def _blur(
    image: npt.NDArray[Any], ksize: tuple[int, int], border_type: int = 4
) -> npt.NDArray[Any]:
    """Apply a box filter with OpenCV's default reflect-101 boundary behavior."""
    if min(ksize) <= 0:
        raise ValueError("Blur kernel dimensions must be positive")
    if border_type != 4:
        raise ValueError("Only OpenCV's default blur border is supported")

    from scipy import ndimage

    size = (*ksize[::-1], 1) if image.ndim == 3 else ksize[::-1]
    values = ndimage.uniform_filter(image.astype(np.float64), size=size, mode="mirror")
    return np.ascontiguousarray(_cast_array_like_opencv(values, image.dtype))


def _as_quad(points: npt.NDArray[Any], name: str) -> npt.NDArray[np.float64]:
    """Validate an OpenCV four-point float32 vector and return it as (4, 2) float64."""
    if not isinstance(points, np.ndarray) or points.dtype != np.float32:
        raise TypeError(f"{name} must be a float32 NumPy array, as in OpenCV")
    if points.shape not in _QUAD_SHAPES:
        raise ValueError(
            f"{name} must have shape (4, 2), (4, 1, 2) or (1, 4, 2), got {points.shape}"
        )
    quad = points.reshape(4, 2).astype(np.float64)
    if not np.all(np.isfinite(quad)):
        raise ValueError(f"{name} points must be finite")
    return quad


def _get_perspective_transform(
    src: npt.NDArray[np.float32], dst: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Compute the perspective transform that maps four source points onto four targets.

    Solves OpenCV's eight-equation system in float64. OpenCV forms the product terms
    of that system (``-x*u``, ``-y*u``, ``-x*v``, ``-y*v``) from float32 point
    products, so the two backends agree to that float32 rounding rather than bit for
    bit, and this fallback is the more accurate of the two.

    The fallback does not check conditioning: a near-singular system, such as three
    nearly collinear source points, returns a finite ill-conditioned matrix. OpenCV
    instead switches to an SVD null-space solution (unit norm, ``m[2, 2] != 1``) when
    its LU solve fails or leaves a residual of at least 1e-8.

    Raises:
        TypeError: If ``src`` or ``dst`` is not a float32 NumPy array.
        ValueError: If ``src`` or ``dst`` has an unsupported shape or a non-finite
            point, or the system is exactly singular (a zero LU pivot), where OpenCV
            returns a degenerate matrix instead.
    """
    x, y = _as_quad(src, "src").T
    u, v = _as_quad(dst, "dst").T
    ones = np.ones(4)
    zeros = np.zeros(4)
    # Rows follow OpenCV: u = (m00 x + m01 y + m02) / (m20 x + m21 y + 1),
    # rearranged to be linear in the eight unknown matrix entries.
    coefficients = np.vstack(
        (
            np.column_stack((x, y, ones, zeros, zeros, zeros, -x * u, -y * u)),
            np.column_stack((zeros, zeros, zeros, x, y, ones, -x * v, -y * v)),
        )
    )
    try:
        solution = np.linalg.solve(coefficients, np.concatenate((u, v)))
    except np.linalg.LinAlgError:
        raise ValueError("the perspective system of src and dst is singular") from None
    return np.append(solution, 1.0).reshape(3, 3)


def _perspective_transform(
    src: npt.NDArray[np.floating[Any]], m: npt.NDArray[Any]
) -> npt.NDArray[np.floating[Any]] | None:
    """Map two-channel points through a 3x3 perspective matrix like OpenCV.

    Points whose ``w`` is NaN or has ``|w| <= FLT_EPSILON`` map to ``(0, 0)``; other
    non-finite values follow IEEE arithmetic, as in OpenCV. The output keeps the
    shape and dtype of ``src``; empty input returns ``None``.
    """
    if not isinstance(src, np.ndarray) or src.dtype not in (np.float32, np.float64):
        raise TypeError("src must be a float32 or float64 NumPy array, as in OpenCV")
    is_real_matrix = isinstance(m, np.ndarray) and (
        np.issubdtype(m.dtype, np.integer) or np.issubdtype(m.dtype, np.floating)
    )
    if not is_real_matrix:
        raise TypeError("m must be a real-valued NumPy array")
    if src.ndim != 3 or src.shape[-1] != 2:
        raise ValueError(
            f"src must have shape (rows, cols, 2), such as (N, 1, 2), got {src.shape}"
        )
    if m.shape != (3, 3):
        raise ValueError(f"m must have shape (3, 3) for 2-D points, got {m.shape}")
    if src.size == 0:
        return None

    points = src.reshape(-1, 2).astype(np.float64, copy=False)
    matrix = m.astype(np.float64, copy=False)
    x = points[:, 0]
    y = points[:, 1]
    # Points whose w is NaN or |w| <= FLT_EPSILON map to (0, 0) below; other
    # non-finite intermediates follow IEEE arithmetic as in OpenCV, so skip warnings.
    # Overflow, in the sums and again in the final cast to float32, gives IEEE
    # infinity as in OpenCV, so it must not raise under a caller's strict error policy.
    with np.errstate(invalid="ignore", over="ignore"):
        # Spell out OpenCV's per-point sums instead of a matmul so rounding matches.
        projected_x = x * matrix[0, 0] + y * matrix[0, 1] + matrix[0, 2]
        projected_y = x * matrix[1, 0] + y * matrix[1, 1] + matrix[1, 2]
        w = x * matrix[2, 0] + y * matrix[2, 1] + matrix[2, 2]
        # OpenCV maps points at or near infinity (|w| <= FLT_EPSILON, or NaN) to 0.
        finite_w = np.abs(w) > _FLT_EPSILON
        inverse_w = np.reciprocal(w, out=np.zeros_like(w), where=finite_w)
        mapped = np.column_stack((projected_x * inverse_w, projected_y * inverse_w))
        # Zero explicitly: an infinite coordinate times a zero inverse_w would be NaN.
        mapped[~finite_w] = 0.0
        result = mapped.astype(src.dtype, copy=False)
    return result.reshape(src.shape)
