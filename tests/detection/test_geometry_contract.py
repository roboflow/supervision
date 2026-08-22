"""Contract tests for geometry-aware detection ops (issue #2318).

Several fixes have been the same bug in different functions: an op reaches for
`xyxy` and forgets the real geometry lives in `data["xyxyxyxy"]`. Because `xyxy`
is always present the wrong answer is the silent default, so each occurrence
looked like a fresh bug rather than a recurring one:

* `Detections.area` returned the envelope area, not the rotated body (#2306)
* `with_nms` / `with_nmm` dropped crossed oriented boxes via envelope IoU (#2303)
* `as_yolo` dropped oriented-box rotation on export (#2289)
* `DetectionsSmoother` averaged `xyxy` and carried the corners over unsmoothed
  from the oldest frame in the window (#2489)

These tests state the shared contract once: **an op that claims to be
geometry-aware must produce a different answer for an oriented box than for its
axis-aligned envelope.** A new op is covered by adding one entry to
`_GEOMETRY_AWARE_OPS`, and an op that silently falls back to `xyxy` fails here
rather than in a bug report months later.

The fixture is a square rotated 45 degrees, whose envelope has exactly twice its
area. That makes the expected values exact rather than approximate, so a failure
says which geometry was used rather than only that a number moved.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection._geometry_dispatch import detection_area, detection_iou
from supervision.detection.core import Detections

#: Half-diagonal of the rotated square used throughout.
HALF_DIAGONAL = 10.0


def _diamond(
    cx: float = 0.0, cy: float = 0.0, half_diagonal: float = HALF_DIAGONAL
) -> np.ndarray:
    """Corners of a square rotated 45 degrees about ``(cx, cy)``.

    Its side is ``half_diagonal * sqrt(2)``, so its area is ``2 * half_diagonal ** 2``,
    exactly half the area of its axis-aligned envelope.
    """
    return np.array(
        [
            [
                [cx, cy - half_diagonal],
                [cx + half_diagonal, cy],
                [cx, cy + half_diagonal],
                [cx - half_diagonal, cy],
            ]
        ],
        dtype=np.float32,
    )


def _envelope_of(corners: np.ndarray) -> np.ndarray:
    """The axis-aligned box that bounds ``corners``."""
    return np.array(
        [
            [
                corners[0, :, 0].min(),
                corners[0, :, 1].min(),
                corners[0, :, 0].max(),
                corners[0, :, 1].max(),
            ]
        ],
        dtype=np.float32,
    )


def _oriented(cx: float = 0.0, cy: float = 0.0) -> Detections:
    """A detection carrying oriented corners and the matching envelope."""
    corners = _diamond(cx, cy)
    return Detections(
        xyxy=_envelope_of(corners),
        class_id=np.array([0]),
        data={ORIENTED_BOX_COORDINATES: corners},
    )


def _envelope_only(cx: float = 0.0, cy: float = 0.0) -> Detections:
    """The same detection with the oriented corners stripped."""
    return Detections(xyxy=_envelope_of(_diamond(cx, cy)), class_id=np.array([0]))


#: Each entry runs an op against oriented input and against its envelope. The
#: contract is that the two disagree; a fallback to `xyxy` makes them equal.
_GEOMETRY_AWARE_OPS: list[
    tuple[str, Callable[[Callable[..., Detections]], np.ndarray]]
] = [
    ("detection_area", lambda build: detection_area(build())),
    (
        "detection_iou",
        lambda build: detection_iou(build(), build(HALF_DIAGONAL, HALF_DIAGONAL)),
    ),
]


@pytest.mark.parametrize(
    ("name", "run"), _GEOMETRY_AWARE_OPS, ids=[n for n, _ in _GEOMETRY_AWARE_OPS]
)
def test_op_respects_geometry(
    name: str, run: Callable[[Callable[..., Detections]], np.ndarray]
) -> None:
    """A geometry-aware op must not answer the same for a box and its envelope."""
    with_corners = np.asarray(run(_oriented), dtype=float)
    without_corners = np.asarray(run(_envelope_only), dtype=float)

    assert with_corners.shape == without_corners.shape, (
        f"{name} changed result shape depending on whether corners were present"
    )
    assert not np.allclose(with_corners, without_corners), (
        f"{name} gave the same answer for an oriented box and its envelope "
        f"({with_corners.tolist()}), so it is reading xyxy rather than the "
        "geometry the detection actually carries"
    )


class TestGeometryContractValues:
    """The exact values behind the contract, so a failure names the geometry used."""

    def test_area_uses_the_rotated_body(self) -> None:
        """A 45-degree square has exactly half its envelope's area."""
        expected_obb = 2 * HALF_DIAGONAL**2
        expected_envelope = 4 * HALF_DIAGONAL**2

        assert_allclose(detection_area(_oriented()), [expected_obb], rtol=1e-5)
        assert_allclose(
            detection_area(_envelope_only()), [expected_envelope], rtol=1e-5
        )

    def test_iou_uses_the_rotated_body(self) -> None:
        """Diamonds offset along the diagonal touch on an edge; envelopes overlap.

        Envelopes are ``[-d, -d, d, d]`` and ``[0, 0, 2d, 2d]``: they intersect over
        ``d * d`` with a union of ``7 * d * d``. The bodies meet only along the shared
        edge, so the oriented overlap is zero.
        """
        obb = detection_iou(_oriented(), _oriented(HALF_DIAGONAL, HALF_DIAGONAL))
        envelope = detection_iou(
            _envelope_only(), _envelope_only(HALF_DIAGONAL, HALF_DIAGONAL)
        )

        assert_allclose(obb, [[0.0]], atol=1e-6)
        assert_allclose(envelope, [[1 / 7]], rtol=1e-5)

    def test_envelope_is_a_faithful_bound(self) -> None:
        """Guards the fixture itself: the envelope really does bound the corners."""
        corners = _diamond()
        x0, y0, x1, y1 = _envelope_of(corners)[0]

        assert corners[0, :, 0].min() == pytest.approx(x0)
        assert corners[0, :, 1].min() == pytest.approx(y0)
        assert corners[0, :, 0].max() == pytest.approx(x1)
        assert corners[0, :, 1].max() == pytest.approx(y1)
