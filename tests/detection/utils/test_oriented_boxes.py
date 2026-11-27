"""Tests for oriented-box IoU, NMS and NMM."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import xyxyxyxy_to_xyxy
from supervision.detection.utils.iou import (
    OverlapMetric,
    box_iou_batch,
    oriented_box_iou_batch,
)
from supervision.detection.utils.nms import (
    _oriented_box_non_max_suppression,
    _oriented_box_soft_non_max_suppression,
    oriented_box_non_max_merge,
)


def _rotated_rect(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    """Return the 4 corners of a rotated rectangle as a (4, 2) float32 array."""
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    rotated: np.ndarray = (corners @ rot.T + [cx, cy]).astype(np.float32)
    return rotated


def _rotated_rect_shifted_along_axis(
    width: float, height: float, angle_deg: float, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a float64 rotated rectangle and its copy moved along its long axis.

    Their overlapping long edges are collinear up to rounding, and the true IoU is
    exactly ``(width - shift) / (width + shift)``.
    """
    angle = np.deg2rad(angle_deg)
    axis = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-axis[1], axis[0]])
    half_width, half_height = axis * width / 2, normal * height / 2
    center = np.array([200.0, 200.0])
    box = np.array(
        [
            center - half_width - half_height,
            center + half_width - half_height,
            center + half_width + half_height,
            center - half_width + half_height,
        ]
    )
    return box, box + axis * shift


def _aabb_of(corners: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding box of a (4, 2) OBB corner array."""
    return np.array(
        [
            corners[:, 0].min(),
            corners[:, 1].min(),
            corners[:, 0].max(),
            corners[:, 1].max(),
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    ("predictions", "oriented_boxes", "sigma", "expected_result", "exception"),
    [
        pytest.param(
            np.empty((0, 6)),
            np.empty((0, 4, 2)),
            0.5,
            np.array([]),
            DoesNotRaise(),
            id="empty-input",
        ),
        pytest.param(
            np.array([[-10, -10, 10, 10, 0.9, 0], [0, 0, 20, 20, 0.8, 0]]),
            np.array(
                [
                    [[0, -10], [10, 0], [0, 10], [-10, 0]],
                    [[10, 0], [20, 10], [10, 20], [0, 10]],
                ],
                dtype=np.float32,
            ),
            0.5,
            np.array([0.9, 0.8]),
            DoesNotRaise(),
            id="edge-touching-diamonds",
        ),
        pytest.param(
            np.array([[0, 0, 40, 20, 0.9, 0], [0, 0, 40, 20, 0.8, 0]]),
            np.array(
                [
                    [[0, 0], [40, 0], [40, 20], [0, 20]],
                    [[0, 0], [40, 0], [40, 20], [0, 20]],
                ],
                dtype=np.float32,
            ),
            0.5,
            np.array([0.9, 0.10826823]),
            DoesNotRaise(),
            id="identical-boxes",
        ),
        pytest.param(
            np.array([[0, 0, 40, 20, 0.8, 0], [0, 0, 40, 20, 0.9, 1]]),
            np.array(
                [
                    [[0, 0], [40, 0], [40, 20], [0, 20]],
                    [[0, 0], [40, 0], [40, 20], [0, 20]],
                ],
                dtype=np.float32,
            ),
            0.5,
            np.array([0.8, 0.9]),
            DoesNotRaise(),
            id="different-classes",
        ),
        pytest.param(
            np.array([[0, 0, 40, 20, 0.9]]),
            np.array([[[0, 0], [40, 0], [40, 20], [0, 20]]], dtype=np.float32),
            0.0,
            None,
            pytest.raises(ValueError, match="sigma"),
            id="non-positive-sigma",
        ),
        pytest.param(
            np.array([[0, 0, 40, 20, 0.9], [0, 0, 40, 20, 0.8]]),
            np.array([[[0, 0], [40, 0], [40, 20], [0, 20]]], dtype=np.float32),
            0.5,
            None,
            pytest.raises(ValueError, match="same length"),
            id="length-mismatch",
        ),
    ],
)
def test_oriented_box_soft_non_max_suppression(
    predictions: np.ndarray,
    oriented_boxes: np.ndarray,
    sigma: float,
    expected_result: np.ndarray | None,
    exception: AbstractContextManager[object],
) -> None:
    """Oriented-box soft-NMS decays by rotated-body IoU and rejects bad inputs."""
    with exception:
        result = _oriented_box_soft_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes, sigma=sigma
        )
        assert expected_result is not None
        np.testing.assert_almost_equal(result, expected_result, decimal=5)


@pytest.mark.parametrize(
    ("origin", "overlap_metric", "expected"),
    [
        pytest.param(-(2**53), OverlapMetric.IOU, 1.0 / 3.0, id="negative-2pow53-iou"),
        pytest.param(-(2**53), OverlapMetric.IOS, 0.5, id="negative-2pow53-ios"),
        pytest.param(2**53, OverlapMetric.IOU, 1.0 / 3.0, id="positive-2pow53-iou"),
        pytest.param(2**53, OverlapMetric.IOS, 0.5, id="positive-2pow53-ios"),
    ],
)
def test_oriented_box_iou_batch_signed_int64_origin_preserves_unit_overlap(
    origin: int,
    overlap_metric: OverlapMetric,
    expected: float,
) -> None:
    """Translation-before-cast preserves signed int64 OBB IoU and IoS at 2**53."""
    box_true = np.array(
        [[origin, 0], [origin + 2, 0], [origin + 2, 2], [origin, 2]],
        dtype=np.int64,
    )
    box_detection = np.array(
        [
            [origin + 1, 0],
            [origin + 3, 0],
            [origin + 3, 2],
            [origin + 1, 2],
        ],
        dtype=np.int64,
    )

    result = oriented_box_iou_batch(box_true[None], box_detection[None], overlap_metric)

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(np.array([[10, 1]], dtype=np.float32), id="x-dominant"),
        pytest.param(np.array([[1, 10]], dtype=np.float32), id="y-dominant"),
    ],
)
def test_oriented_box_iou_batch_is_invariant_to_non_square_scaling(
    scale: np.ndarray,
) -> None:
    """IoU is invariant under per-axis (anisotropic) scaling.

    An affine map multiplies every area — intersection and union alike — by the same
    determinant, so exact polygon IoU is unchanged. Equality is exact because the
    computation is rasterization-free; the analytic IoU of these two rectangles is 0.25.
    """
    boxes_true = np.array([[[1, 0], [0, 1], [3, 4], [4, 3]]], dtype=np.float32)
    boxes_detection = np.array([[[1, 1], [2, 0], [4, 2], [3, 3]]], dtype=np.float32)

    baseline_iou = oriented_box_iou_batch(boxes_true, boxes_detection)
    scaled_iou = oriented_box_iou_batch(boxes_true * scale, boxes_detection * scale)

    assert baseline_iou.shape == (1, 1)
    assert scaled_iou.shape == (1, 1)
    assert baseline_iou[0, 0] == pytest.approx(0.25)
    assert np.allclose(scaled_iou, baseline_iou, rtol=1e-5, atol=1e-7)


class TestOrientedBoxIouBatch:
    """Tests for `oriented_box_iou_batch`."""

    @pytest.mark.parametrize(
        ("scale", "offset"),
        [
            pytest.param(80.0, 0.0, id="large-scale"),
            pytest.param(1.0, 3000.0, id="far-from-origin"),
            pytest.param(80.0, 3000.0, id="large-and-far"),
        ],
    )
    def test_is_invariant_to_canvas_transforms(
        self, scale: float, offset: float
    ) -> None:
        """IoU is invariant under uniform scaling and translation.

        Both are affine maps, so exact polygon IoU matches the small-coordinate baseline
        exactly — free of the pixel-quantization error that the prior rasterization-
        based implementation exhibited.
        """
        boxes_true = _rotated_rect(50, 50, 40, 20, 30)[None]
        boxes_detection = _rotated_rect(52, 48, 40, 20, 35)[None]
        baseline = oriented_box_iou_batch(boxes_true, boxes_detection)

        transformed = oriented_box_iou_batch(
            boxes_true * scale + offset,
            boxes_detection * scale + offset,
        )

        assert baseline.shape == (1, 1)
        assert transformed.shape == (1, 1)
        assert baseline[0, 0] > 0.4
        assert np.allclose(transformed, baseline, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param(0, id="origin-0"),
            pytest.param(10**8, id="origin-1e8"),
            pytest.param(10**9, id="origin-1e9"),
            pytest.param(10**10, id="origin-1e10"),
        ],
    )
    def test_is_invariant_to_large_origins(self, origin: float) -> None:
        """IoU stays exact for boxes at large coordinate origins.

        Regression: polygon areas were computed from absolute coordinates, so
        their large shoelace products rounded once the origin grew. Identical
        boxes scored self-IoU ~0.85 at origin 1e8 and exactly 0.0 at origin 1e10.
        """
        baseline = oriented_box_iou_batch(
            _rotated_rect(50, 50, 40, 20, 30)[None],
            _rotated_rect(52, 48, 40, 20, 35)[None],
        )
        boxes_true = (_rotated_rect(50, 50, 40, 20, 30).astype(np.float64) + origin)[
            None
        ]
        boxes_detection = (
            _rotated_rect(52, 48, 40, 20, 35).astype(np.float64) + origin
        )[None]

        transformed = oriented_box_iou_batch(boxes_true, boxes_detection)
        self_iou = oriented_box_iou_batch(boxes_true, boxes_true)[0, 0]

        assert baseline.shape == (1, 1)
        assert transformed.shape == (1, 1)
        assert np.allclose(transformed, baseline, rtol=1e-5, atol=1e-7)
        # Identical boxes must score exactly 1.0 at every origin.
        assert self_iou == pytest.approx(1.0, abs=1e-7)

    def test_supports_overlap_metric(self) -> None:
        """`overlap_metric=IOS` divides by the smaller area, so a small box fully
        contained in a larger one scores exactly 1.0, while IoU is the area ratio."""
        small = _rotated_rect(50, 50, 20, 20, 0)[None]
        large = _rotated_rect(50, 50, 60, 60, 0)[None]

        iou = oriented_box_iou_batch(small, large, OverlapMetric.IOU)[0, 0]
        ios = oriented_box_iou_batch(small, large, OverlapMetric.IOS)[0, 0]

        # 20x20 inside 60x60: IoU = 400 / 3600 = 1/9, IoS = 400 / 400 = 1.0.
        assert iou == pytest.approx(1 / 9, rel=1e-3)
        assert ios == pytest.approx(1.0)

    def test_half_overlap_matches_analytic_value(self) -> None:
        """Two unit-height 2x2 squares offset by 1 share half their area: IoU=1/3."""
        boxes_true = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32)
        boxes_detection = np.array([[[1, 0], [3, 0], [3, 2], [1, 2]]], dtype=np.float32)

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        # intersection = 2, union = 4 + 4 - 2 = 6, IoU = 2/6 = 1/3.
        assert iou[0, 0] == pytest.approx(1 / 3)

    def test_disjoint_boxes_score_zero(self) -> None:
        """Boxes whose envelopes do not overlap are pruned by the gate to 0.0."""
        boxes_true = _rotated_rect(10, 10, 8, 4, 20)[None]
        boxes_detection = _rotated_rect(500, 500, 8, 4, 70)[None]

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        assert iou[0, 0] == 0.0

    @pytest.mark.parametrize(
        "boxes",
        [
            pytest.param(
                np.stack([_rotated_rect(10, 10, 8, 4, 20)]),
                id="n=1",
            ),
            pytest.param(
                np.stack(
                    [_rotated_rect(10, 10, 8, 4, 20), _rotated_rect(40, 40, 8, 4, 70)]
                ),
                id="n=2",
            ),
        ],
    )
    def test_self_comparison_is_symmetric_with_unit_diagonal(
        self, boxes: np.ndarray
    ) -> None:
        """Comparing a set with itself yields a symmetric matrix and a 1.0 diagonal."""
        iou = oriented_box_iou_batch(boxes, boxes)

        assert np.allclose(iou, iou.T)
        for i in range(len(boxes)):
            assert iou[i, i] == pytest.approx(1.0)

    def test_envelope_overlap_without_polygon_overlap_scores_zero(self) -> None:
        """Parallel rotated bars share an envelope but not a body, so they score 0.

        Exercises the path where a pair passes the axis-aligned gate yet has no exact
        polygon intersection.
        """
        boxes_true = _rotated_rect(50, 50, 100, 4, 45)[None]
        boxes_detection = _rotated_rect(72, 28, 100, 4, 45)[None]

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        assert iou[0, 0] == 0.0

    def test_rejects_unsupported_overlap_metric(self) -> None:
        """An overlap metric other than IOU or IOS raises ValueError."""
        boxes = _rotated_rect(50, 50, 20, 20, 0)[None]

        with pytest.raises(ValueError, match="is not supported"):
            oriented_box_iou_batch(boxes, boxes, "invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("boxes_a", "boxes_b", "expected"),
        [
            pytest.param(
                np.array([[[5, 5], [5, 5], [5, 5], [5, 5]]], dtype=np.float32),
                np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32),
                0.0,
                id="collapsed-box-vs-normal",
            ),
            pytest.param(
                np.array([[[0, 0], [1, 0], [2, 0], [3, 0]]], dtype=np.float32),
                np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32),
                0.0,
                id="collinear-box-vs-normal",
            ),
            pytest.param(
                np.array([[[5, 5], [5, 5], [5, 5], [5, 5]]], dtype=np.float32),
                np.array([[[5, 5], [5, 5], [5, 5], [5, 5]]], dtype=np.float32),
                0.0,
                id="self-comparison-zero-area",
            ),
        ],
    )
    def test_degenerate_boxes_score_zero(
        self,
        boxes_a: np.ndarray,
        boxes_b: np.ndarray,
        expected: float,
    ) -> None:
        """Degenerate boxes (zero-area collapsed or collinear) always score 0.

        Note: self-comparison of a zero-area box returns 0.0, not 1.0 — this
        diverges from ``box_iou_batch`` which returns 1.0 for degenerate AABB
        self-comparison.
        """
        iou = oriented_box_iou_batch(boxes_a, boxes_b)
        assert iou[0, 0] == expected

    @pytest.mark.parametrize(
        ("n", "m"),
        [
            pytest.param(0, 3, id="zero-true"),
            pytest.param(3, 0, id="zero-detection"),
            pytest.param(0, 0, id="both-zero"),
        ],
    )
    def test_empty_input_returns_correct_shape(self, n: int, m: int) -> None:
        """Empty inputs trigger the early-return path; result is zero-filled."""
        boxes_true = np.empty((n, 4, 2), dtype=np.float32)
        boxes_detection = np.empty((m, 4, 2), dtype=np.float32)

        iou = oriented_box_iou_batch(boxes_true, boxes_detection)

        assert iou.shape == (n, m)
        assert iou.dtype == np.float64
        assert np.all(iou == 0.0)

    @pytest.mark.parametrize(
        ("bad_shape", "match"),
        [
            pytest.param((2, 3, 2), r"expected \(N, 4, 2\)", id="3d-wrong-inner-dims"),
            pytest.param((2, 5), r"expected \(N, 8\)", id="2d-wrong-columns"),
            pytest.param((8,), r"must be 2-D", id="1d-input"),
        ],
    )
    def test_invalid_shape_raises_value_error(
        self, bad_shape: tuple[int, ...], match: str
    ) -> None:
        """Inputs with wrong shape raise ValueError before any computation."""
        boxes_bad = np.zeros(bad_shape, dtype=np.float32)
        boxes_ok = np.zeros((2, 4, 2), dtype=np.float32)

        with pytest.raises(ValueError, match=match):
            oriented_box_iou_batch(boxes_bad, boxes_ok)

    @pytest.mark.parametrize(
        ("angle_deg", "shift"),
        [
            pytest.param(15.0, 20.0, id="reported-as-identical-at-15-deg"),
            pytest.param(10.0, 50.0, id="reported-as-identical-at-10-deg"),
            pytest.param(25.0, 50.0, id="reported-as-disjoint-at-25-deg"),
            pytest.param(20.0, 10.0, id="underestimated-at-20-deg"),
        ],
    )
    def test_matches_analytic_iou_for_collinear_edges(
        self, angle_deg: float, shift: float
    ) -> None:
        """Rectangles shifted along their axis score their analytic IoU."""
        box_a, box_b = _rotated_rect_shifted_along_axis(100, 40, angle_deg, shift)

        iou = oriented_box_iou_batch(box_a[None], box_b[None])

        assert iou[0, 0] == pytest.approx((100 - shift) / (100 + shift), abs=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matches_analytic_iou_for_random_collinear_edges(
        self, dtype: type[np.floating]
    ) -> None:
        """Seeded rectangles shifted along their axis all score their analytic IoU."""
        rng = np.random.default_rng(0)
        widths = rng.uniform(8, 200, 1000)
        heights = rng.uniform(8, 200, 1000)
        angles = rng.uniform(0, 180, 1000)
        shifts = rng.uniform(0, widths)
        pairs = [
            _rotated_rect_shifted_along_axis(*parameters)
            for parameters in zip(widths, heights, angles, shifts)
        ]

        scores = [
            oriented_box_iou_batch(
                box_a[None].astype(dtype), box_b[None].astype(dtype)
            )[0, 0]
            for box_a, box_b in pairs
        ]

        np.testing.assert_allclose(
            scores, (widths - shifts) / (widths + shifts), atol=1e-3
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_tiny_identical_boxes_score_one(self, dtype: type[np.floating]) -> None:
        """A box far below one coordinate unit in size fully overlaps itself."""
        box = (_rotated_rect(0, 0, 0.002, 0.001, 30).astype(np.float64) + 0.5)[None]
        boxes_true = box.astype(dtype)

        iou = oriented_box_iou_batch(boxes_true, boxes_true.copy())

        assert iou[0, 0] == pytest.approx(1.0, abs=1e-3)


class TestOrientedBoxNonMaxSuppression:
    """Tests for `_oriented_box_non_max_suppression`."""

    @pytest.mark.parametrize(
        ("angle_deg", "shift", "expected_keep"),
        [
            pytest.param(10.0, 50.0, [True, True], id="keeps-low-true-overlap"),
            pytest.param(50.0, 10.0, [True, False], id="suppresses-high-true-overlap"),
        ],
    )
    def test_uses_true_overlap_for_collinear_edges(
        self, angle_deg: float, shift: float, expected_keep: list[bool]
    ) -> None:
        """Suppression of axis-shifted rectangles follows their analytic IoU."""
        box_a, box_b = _rotated_rect_shifted_along_axis(100, 40, angle_deg, shift)
        predictions = np.array(
            [[*_aabb_of(box_a), 0.9, 0], [*_aabb_of(box_b), 0.8, 0]], dtype=np.float32
        )

        keep = _oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=np.stack([box_a, box_b]),
            iou_threshold=0.5,
        )

        assert keep.tolist() == expected_keep

    def test_keeps_x_pattern(self) -> None:
        """X-pattern: two thin rectangles crossing at +/-45° share an AABB but barely
        overlap as OBBs.

        AABB-NMS would suppress one; OBB-NMS must keep both.
        """
        quad_a = _rotated_rect(50, 50, 100, 10, +45)
        quad_b = _rotated_rect(50, 50, 100, 10, -45)
        oriented_boxes = np.stack([quad_a, quad_b])
        predictions = np.array(
            [
                [*_aabb_of(quad_a), 0.9, 0],
                [*_aabb_of(quad_b), 0.85, 0],
            ],
            dtype=np.float32,
        )

        assert box_iou_batch(predictions[:, :4], predictions[:, :4])[0, 1] > 0.95
        assert oriented_box_iou_batch(quad_a[None], quad_b[None])[0, 0] < 0.2

        keep = _oriented_box_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes, iou_threshold=0.5
        )
        assert np.array_equal(keep, np.array([True, True]))

    @pytest.mark.parametrize(
        ("class_id_b", "expected_keep"),
        [
            pytest.param(0, [True, False], id="same-class"),
            pytest.param(1, [True, True], id="diff-class"),
        ],
    )
    def test_suppression_is_class_aware(
        self, class_id_b: int, expected_keep: list[bool]
    ) -> None:
        """Same class: lower-score OBB suppressed.

        Different class: both kept.
        """
        quad = _rotated_rect(50, 50, 100, 10, 45)
        shifted = _rotated_rect(51, 51, 100, 10, 45)
        oriented_boxes = np.stack([quad, shifted])
        predictions = np.array(
            [
                [*_aabb_of(quad), 0.9, 0],
                [*_aabb_of(shifted), 0.85, class_id_b],
            ],
            dtype=np.float32,
        )

        assert oriented_box_iou_batch(quad[None], shifted[None])[0, 0] > 0.9

        keep = _oriented_box_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes, iou_threshold=0.5
        )
        assert np.array_equal(keep, np.array(expected_keep))

    def test_length_mismatch_raises(self) -> None:
        """Mismatched predictions and oriented_boxes must fail loudly, not silently
        misalign rows."""
        predictions = np.zeros((3, 5), dtype=np.float32)
        oriented_boxes = np.zeros((2, 4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="same length"):
            _oriented_box_non_max_suppression(
                predictions=predictions, oriented_boxes=oriented_boxes
            )

    def test_empty_predictions(self) -> None:
        """No OBB predictions should produce an empty boolean keep mask."""
        predictions = np.empty((0, 5), dtype=np.float32)
        oriented_boxes = np.empty((0, 4, 2), dtype=np.float32)

        keep = _oriented_box_non_max_suppression(
            predictions=predictions, oriented_boxes=oriented_boxes
        )

        assert keep.shape == (0,)
        assert keep.dtype == bool

    @pytest.mark.parametrize("iou_threshold", [0.0, 0.5, 1.0])
    def test_keeps_single_prediction(self, iou_threshold: float) -> None:
        """A single OBB prediction is always kept, regardless of threshold."""
        quad = _rotated_rect(50, 50, 40, 20, 30)
        oriented_boxes = quad[None]
        predictions = np.array([[*_aabb_of(quad), 0.9, 0]], dtype=np.float32)

        keep = _oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=oriented_boxes,
            iou_threshold=iou_threshold,
        )

        assert np.array_equal(keep, np.array([True]))

    @pytest.mark.parametrize(
        ("iou_threshold", "expected_keep"),
        [
            pytest.param(0.0, [True, False], id="threshold-0"),
            pytest.param(1.0, [True, True], id="threshold-1"),
        ],
    )
    def test_threshold_extremes(
        self, iou_threshold: float, expected_keep: list[bool]
    ) -> None:
        """At threshold extremes, positive-overlap non-identical OBBs suppress at
        0.0 and are both kept at 1.0."""
        quad_a = _rotated_rect(50, 50, 40, 40, 0)
        quad_b = _rotated_rect(55, 50, 40, 40, 0)
        oriented_boxes = np.stack([quad_a, quad_b])
        predictions = np.array(
            [
                [*_aabb_of(quad_a), 0.9, 0],
                [*_aabb_of(quad_b), 0.85, 0],
            ],
            dtype=np.float32,
        )

        overlap = oriented_box_iou_batch(quad_a[None], quad_b[None])[0, 0]
        assert 0.0 < overlap < 1.0

        keep = _oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=oriented_boxes,
            iou_threshold=iou_threshold,
        )

        assert np.array_equal(keep, np.array(expected_keep))

    @pytest.mark.parametrize(
        ("overlap_metric", "expected_keep"),
        [
            pytest.param(OverlapMetric.IOU, [True, True], id="iou-keeps-both"),
            pytest.param(OverlapMetric.IOS, [True, False], id="ios-suppresses-small"),
        ],
    )
    def test_overlap_metric_determines_suppression(
        self, overlap_metric: OverlapMetric, expected_keep: list[bool]
    ) -> None:
        """Small box inside large: IOU keeps both; IOS suppresses small."""
        large = _rotated_rect(50, 50, 60, 60, 0)
        small = _rotated_rect(50, 50, 20, 20, 0)
        oriented_boxes = np.stack([large, small])
        predictions = np.array(
            [
                [*_aabb_of(large), 0.9, 0],
                [*_aabb_of(small), 0.85, 0],
            ],
            dtype=np.float32,
        )

        keep = _oriented_box_non_max_suppression(
            predictions=predictions,
            oriented_boxes=oriented_boxes,
            iou_threshold=0.5,
            overlap_metric=overlap_metric,
        )

        assert np.array_equal(keep, np.array(expected_keep))


class TestOrientedBoxNonMaxMerge:
    """Tests for `oriented_box_non_max_merge`."""

    def test_empty_predictions_returns_empty_groups(self) -> None:
        """No OBB predictions should produce no merge groups."""
        predictions = np.empty((0, 5), dtype=np.float32)
        oriented_boxes = np.empty((0, 4, 2), dtype=np.float32)

        groups = oriented_box_non_max_merge(
            predictions=predictions, oriented_boxes=oriented_boxes
        )

        assert groups == []

    def test_single_prediction_returns_singleton_group(self) -> None:
        """A single OBB prediction should be returned as one singleton group."""
        quad = _rotated_rect(50, 50, 40, 20, 30)
        oriented_boxes = quad[None]
        predictions = np.array([[*_aabb_of(quad), 0.9, 0]], dtype=np.float32)

        groups = oriented_box_non_max_merge(
            predictions=predictions, oriented_boxes=oriented_boxes
        )

        assert groups == [[0]]

    def test_groups_overlapping_oriented_boxes(self) -> None:
        """Two near-identical OBBs should be merged into one group; an X-pattern pair
        should produce two separate groups."""
        quad_dup_a = _rotated_rect(50, 50, 100, 10, 45)
        quad_dup_b = _rotated_rect(51, 51, 100, 10, 45)
        quad_x = _rotated_rect(50, 50, 100, 10, -45)
        oriented_boxes = np.stack([quad_dup_a, quad_dup_b, quad_x])
        predictions = np.array(
            [
                [*_aabb_of(quad_dup_a), 0.90, 0],
                [*_aabb_of(quad_dup_b), 0.85, 0],
                [*_aabb_of(quad_x), 0.80, 0],
            ],
            dtype=np.float32,
        )

        groups = oriented_box_non_max_merge(
            predictions=predictions, oriented_boxes=oriented_boxes, iou_threshold=0.5
        )

        sorted_groups = sorted(sorted(g) for g in groups)
        assert sorted_groups == [[0, 1], [2]]


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("with_nms", id="with-nms"),
        pytest.param("with_nmm", id="with-nmm"),
    ],
)
def test_detections_obb_overlap_filter_large_origin_duplicates_deduplicate(
    method: str,
) -> None:
    """Public NMS/NMM preserve 1e10 OBB precision to deduplicate duplicates."""
    origin = 10**10
    oriented_box = np.array(
        [
            [origin, origin],
            [origin + 100, origin],
            [origin + 100, origin + 50],
            [origin, origin + 50],
        ],
        dtype=np.float64,
    )
    oriented_boxes = np.stack([oriented_box, oriented_box])
    detections = Detections(
        xyxy=xyxyxyxy_to_xyxy(oriented_boxes),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0]),
        data={ORIENTED_BOX_COORDINATES: oriented_boxes},
    )

    result = getattr(detections, method)(threshold=0.5)

    assert len(result) == 1
