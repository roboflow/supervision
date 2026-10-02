"""Private transform and filter fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from supervision._cv2._common import _cast_array_like_opencv

_FLT_EPSILON = float(np.finfo(np.float32).eps)
_QUAD_SHAPES = ((4, 2), (4, 1, 2), (1, 4, 2))
_TRIPLES_OF_QUAD = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


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


def _has_collinear_triple(quad: npt.NDArray[np.float64]) -> bool:
    """Report whether any three quad points are collinear at float32 resolution.

    Flags triples whose edge cross product is at most ``FLT_EPSILON`` times the edge
    lengths' product (sine of the angle below float32 epsilon); repeated points have
    a zero-length edge and are flagged too.
    """
    triples = quad[_TRIPLES_OF_QUAD]
    first = triples[:, 1] - triples[:, 0]
    second = triples[:, 2] - triples[:, 0]
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    lengths = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return bool(np.any(cross <= _FLT_EPSILON * lengths))


def _get_perspective_transform(
    src: npt.NDArray[np.float32], dst: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Solve OpenCV's eight-equation perspective system in float64.

    Degenerate quads (three collinear or repeated points, or a transform that sends
    the source origin to infinity) raise ``ValueError`` instead of returning a matrix
    that does not map them.
    """
    source = _as_quad(src, "src")
    target = _as_quad(dst, "dst")
    for name, quad in (("src", source), ("dst", target)):
        if _has_collinear_triple(quad):
            raise ValueError(
                f"{name} contains three collinear or repeated points; "
                "no perspective transform maps such quads"
            )

    x, y = source.T
    u, v = target.T
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
    if np.linalg.matrix_rank(coefficients) < 8:
        raise ValueError(
            "no perspective transform with m[2, 2] == 1 maps src onto dst; "
            "the transform sends the source origin to infinity"
        )
    solution = np.linalg.solve(coefficients, np.concatenate((u, v)))
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
    with np.errstate(invalid="ignore"):
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
    return mapped.astype(src.dtype, copy=False).reshape(src.shape)
