from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils.boxes import (
    _oriented_box_anchors as oriented_box_anchors,
)
from supervision.detection.utils.boxes import (
    clip_boxes,
    denormalize_boxes,
    move_boxes,
    pad_boxes,
    scale_boxes,
    xyxyxyxy_to_xyxy,
)
from supervision.geometry.core import Position

_ALL_ANCHORS = [
    Position.CENTER,
    Position.CENTER_LEFT,
    Position.CENTER_RIGHT,
    Position.TOP_CENTER,
    Position.BOTTOM_CENTER,
    Position.TOP_LEFT,
    Position.TOP_RIGHT,
    Position.BOTTOM_LEFT,
    Position.BOTTOM_RIGHT,
]


def _rotate(corners: np.ndarray, angle_deg: float, about: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (corners - about) @ rot.T + about


@pytest.mark.parametrize(
    ("xyxy", "resolution_wh", "expected_result"),
    [
        (
            np.empty(shape=(0, 4)),
            (1280, 720),
            np.empty(shape=(0, 4)),
        ),
        (
            np.array([[1.0, 1.0, 1279.0, 719.0]]),
            (1280, 720),
            np.array([[1.0, 1.0, 1279.0, 719.0]]),
        ),
        (
            np.array([[-1.0, 1.0, 1279.0, 719.0]]),
            (1280, 720),
            np.array([[0.0, 1.0, 1279.0, 719.0]]),
        ),
        (
            np.array([[1.0, -1.0, 1279.0, 719.0]]),
            (1280, 720),
            np.array([[1.0, 0.0, 1279.0, 719.0]]),
        ),
        (
            np.array([[1.0, 1.0, 1281.0, 719.0]]),
            (1280, 720),
            np.array([[1.0, 1.0, 1280.0, 719.0]]),
        ),
        (
            np.array([[1.0, 1.0, 1279.0, 721.0]]),
            (1280, 720),
            np.array([[1.0, 1.0, 1279.0, 720.0]]),
        ),
    ],
)
def test_clip_boxes(
    xyxy: np.ndarray,
    resolution_wh: tuple[int, int],
    expected_result: np.ndarray,
) -> None:
    result = clip_boxes(xyxy=xyxy, resolution_wh=resolution_wh)
    assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("xyxy", "offset", "expected_result", "exception"),
    [
        (
            np.empty(shape=(0, 4)),
            np.array([0, 0]),
            np.empty(shape=(0, 4)),
            DoesNotRaise(),
        ),  # empty xyxy array
        (
            np.array([[0, 0, 10, 10]]),
            np.array([0, 0]),
            np.array([[0, 0, 10, 10]]),
            DoesNotRaise(),
        ),  # single box with zero offset
        (
            np.array([[0, 0, 10, 10]]),
            np.array([10, 10]),
            np.array([[10, 10, 20, 20]]),
            DoesNotRaise(),
        ),  # single box with non-zero offset
        (
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
            np.array([10, 10]),
            np.array([[10, 10, 20, 20], [10, 10, 20, 20]]),
            DoesNotRaise(),
        ),  # two boxes with non-zero offset
        (
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
            np.array([-10, -10]),
            np.array([[-10, -10, 0, 0], [-10, -10, 0, 0]]),
            DoesNotRaise(),
        ),  # two boxes with negative offset
    ],
)
def test_move_boxes(
    xyxy: np.ndarray,
    offset: np.ndarray,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = move_boxes(xyxy=xyxy, offset=offset)
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("xyxy", "factor", "expected_result", "exception"),
    [
        (
            np.empty(shape=(0, 4)),
            2.0,
            np.empty(shape=(0, 4)),
            DoesNotRaise(),
        ),  # empty xyxy array
        (
            np.array([[0, 0, 10, 10]]),
            1.0,
            np.array([[0, 0, 10, 10]]),
            DoesNotRaise(),
        ),  # single box with factor equal to 1.0
        (
            np.array([[0, 0, 10, 10]]),
            2.0,
            np.array([[-5, -5, 15, 15]]),
            DoesNotRaise(),
        ),  # single box with factor equal to 2.0
        (
            np.array([[0, 0, 10, 10]]),
            0.5,
            np.array([[2.5, 2.5, 7.5, 7.5]]),
            DoesNotRaise(),
        ),  # single box with factor equal to 0.5
        (
            np.array([[0, 0, 10, 10], [10, 10, 30, 30]]),
            2.0,
            np.array([[-5, -5, 15, 15], [0, 0, 40, 40]]),
            DoesNotRaise(),
        ),  # two boxes with factor equal to 2.0
    ],
)
def test_scale_boxes(
    xyxy: np.ndarray,
    factor: float,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = scale_boxes(xyxy=xyxy, factor=factor)
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("xyxy", "resolution_wh", "normalization_factor", "expected_result", "exception"),
    [
        (
            np.empty(shape=(0, 4)),
            (1280, 720),
            1.0,
            np.empty(shape=(0, 4)),
            DoesNotRaise(),
        ),  # empty array
        (
            np.array([[0.1, 0.2, 0.5, 0.6]]),
            (1280, 720),
            1.0,
            np.array([[128.0, 144.0, 640.0, 432.0]]),
            DoesNotRaise(),
        ),  # single box with default normalization
        (
            np.array([[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]]),
            (1280, 720),
            1.0,
            np.array([[128.0, 144.0, 640.0, 432.0], [384.0, 288.0, 896.0, 576.0]]),
            DoesNotRaise(),
        ),  # two boxes with default normalization
        (
            np.array(
                [[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8], [0.2, 0.1, 0.6, 0.5]]
            ),
            (1280, 720),
            1.0,
            np.array(
                [
                    [128.0, 144.0, 640.0, 432.0],
                    [384.0, 288.0, 896.0, 576.0],
                    [256.0, 72.0, 768.0, 360.0],
                ]
            ),
            DoesNotRaise(),
        ),  # three boxes - regression test for issue #1959
        (
            np.array([[10.0, 20.0, 50.0, 60.0]]),
            (100, 200),
            100.0,
            np.array([[10.0, 40.0, 50.0, 120.0]]),
            DoesNotRaise(),
        ),  # single box with custom normalization factor
        (
            np.array([[10.0, 20.0, 50.0, 60.0], [30.0, 40.0, 70.0, 80.0]]),
            (100, 200),
            100.0,
            np.array([[10.0, 40.0, 50.0, 120.0], [30.0, 80.0, 70.0, 160.0]]),
            DoesNotRaise(),
        ),  # two boxes with custom normalization factor
        (
            np.array([[0.0, 0.0, 1.0, 1.0]]),
            (1920, 1080),
            1.0,
            np.array([[0.0, 0.0, 1920.0, 1080.0]]),
            DoesNotRaise(),
        ),  # full frame box
        (
            np.array([[0.5, 0.5, 0.5, 0.5]]),
            (640, 480),
            1.0,
            np.array([[320.0, 240.0, 320.0, 240.0]]),
            DoesNotRaise(),
        ),  # zero-area box (point)
    ],
)
def test_denormalize_boxes(
    xyxy: np.ndarray,
    resolution_wh: tuple[int, int],
    normalization_factor: float,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = denormalize_boxes(
            xyxy=xyxy,
            resolution_wh=resolution_wh,
            normalization_factor=normalization_factor,
        )
        assert np.allclose(result, expected_result)


@pytest.mark.parametrize(
    ("corners", "expected"),
    [
        pytest.param(
            np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]], dtype=np.float32),
            np.array([[0, 0, 10, 5]], dtype=np.float32),
            id="single-axis-aligned",
        ),
        pytest.param(
            np.array(
                [
                    [[0, 0], [10, 0], [10, 5], [0, 5]],
                    [[5, 5], [15, 5], [15, 10], [5, 10]],
                ],
                dtype=np.float32,
            ),
            np.array([[0, 0, 10, 5], [5, 5, 15, 10]], dtype=np.float32),
            id="batch-axis-aligned",
        ),
        pytest.param(
            np.array([[[5, 0], [10, 5], [5, 10], [0, 5]]], dtype=np.float32),
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            id="rotated-diamond",
        ),
        pytest.param(
            np.empty((0, 4, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            id="empty-input",
        ),
    ],
)
def test_xyxyxyxy_to_xyxy(corners: np.ndarray, expected: np.ndarray) -> None:
    """Converts OBB corners to axis-aligned bounding boxes."""
    result = xyxyxyxy_to_xyxy(corners)
    assert np.allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        pytest.param(Position.CENTER, [5.0, 2.0], id="center"),
        pytest.param(Position.CENTER_LEFT, [0.0, 2.0], id="center-left"),
        pytest.param(Position.CENTER_RIGHT, [10.0, 2.0], id="center-right"),
        pytest.param(Position.TOP_CENTER, [5.0, 0.0], id="top-center"),
        pytest.param(Position.BOTTOM_CENTER, [5.0, 4.0], id="bottom-center"),
        pytest.param(Position.TOP_LEFT, [0.0, 0.0], id="top-left"),
        pytest.param(Position.TOP_RIGHT, [10.0, 0.0], id="top-right"),
        pytest.param(Position.BOTTOM_LEFT, [0.0, 4.0], id="bottom-left"),
        pytest.param(Position.BOTTOM_RIGHT, [10.0, 4.0], id="bottom-right"),
    ],
)
def test_oriented_box_anchors_axis_aligned_matches_envelope(
    anchor: Position, expected: list[float]
) -> None:
    """On an axis-aligned box the anchor equals the plain envelope anchor."""
    corners = np.array([[[0, 0], [10, 0], [10, 4], [0, 4]]], dtype=np.float32)
    result = oriented_box_anchors(corners, anchor)
    assert np.allclose(result, [expected])


@pytest.mark.parametrize("anchor", _ALL_ANCHORS, ids=lambda a: a.value.lower())
def test_oriented_box_anchors_are_rotation_covariant(anchor: Position) -> None:
    """Rotating the box rotates each anchor by the same angle about the center."""
    base = np.array([[[0, 0], [10, 0], [10, 4], [0, 4]]], dtype=np.float64)
    center = np.array([5.0, 2.0])
    rotated = _rotate(base[0], 30, center)[np.newaxis]

    expected = _rotate(oriented_box_anchors(base, anchor)[0], 30, center)
    result = oriented_box_anchors(rotated, anchor)[0]

    assert np.allclose(result, expected)


def test_oriented_box_anchors_are_points_of_the_rotated_rectangle() -> None:
    """Each anchor of a rotated box is one of its corners, side midpoints or center."""
    base = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float64)
    corners = _rotate(base, 30, np.array([5.0, 2.0]))
    rectangle_points = np.vstack(
        [corners, (corners + np.roll(corners, -1, axis=0)) / 2, corners.mean(axis=0)]
    )

    anchors = np.array(
        [oriented_box_anchors(corners[np.newaxis], a)[0] for a in _ALL_ANCHORS]
    )

    distances = np.linalg.norm(rectangle_points[None] - anchors[:, None], axis=2)
    assert np.all(distances.min(axis=1) < 1e-6)


def test_oriented_box_anchors_empty_returns_expected_shape() -> None:
    """An empty batch yields an empty `(0, 2)` array."""
    result = oriented_box_anchors(np.empty((0, 4, 2)), Position.BOTTOM_CENTER)
    assert result.shape == (0, 2)


@pytest.mark.parametrize(
    "corners",
    [
        pytest.param(np.zeros((4, 2)), id="missing-batch-axis"),
        pytest.param(np.zeros((1, 4)), id="not-corner-pairs"),
        pytest.param(np.zeros((1, 3, 2)), id="wrong-corner-count"),
    ],
)
def test_oriented_box_anchors_bad_shape_raises(corners: np.ndarray) -> None:
    """A batch that is not `(N, 4, 2)` is rejected."""
    with pytest.raises(ValueError, match="must have shape"):
        oriented_box_anchors(corners, Position.CENTER)


def test_oriented_box_anchors_center_of_mass_unsupported() -> None:
    """`CENTER_OF_MASS` is a mask anchor and has no box definition."""
    with pytest.raises(ValueError, match="not supported"):
        oriented_box_anchors(np.zeros((1, 4, 2)), Position.CENTER_OF_MASS)


@pytest.mark.parametrize("anchor", _ALL_ANCHORS, ids=lambda a: a.value.lower())
def test_oriented_box_anchors_at_90_degrees_on_box(anchor: Position) -> None:
    """All anchors of a 90-deg-rotated box lie on the box (exercises is_width=False)."""
    base = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float64)
    center = np.array([5.0, 2.0])
    corners = _rotate(base, 90, center)[np.newaxis]

    result = oriented_box_anchors(corners, anchor)[0]

    rectangle_points = np.vstack(
        [corners[0], (corners[0] + np.roll(corners[0], -1, axis=0)) / 2, center]
    )
    distances = np.linalg.norm(rectangle_points - result, axis=1)
    assert distances.min() < 1e-6


# ---------------------------------------------------------------------------
# pad_boxes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("xyxy", "px", "py", "expected"),
    [
        pytest.param(
            np.array([[10, 20, 30, 40]]),
            5,
            None,
            np.array([[5, 15, 35, 45]]),
            id="single-box-uniform",
        ),
        pytest.param(
            np.array([[10, 20, 30, 40]]),
            5,
            10,
            np.array([[5, 10, 35, 50]]),
            id="single-box-asymmetric",
        ),
        pytest.param(
            np.array([[10, 20, 30, 40], [15, 25, 35, 45]]),
            5,
            10,
            np.array([[5, 10, 35, 50], [10, 15, 40, 55]]),
            id="two-boxes",
        ),
        pytest.param(
            np.empty((0, 4), dtype=np.float32),
            5,
            None,
            np.empty((0, 4), dtype=np.float32),
            id="empty",
        ),
    ],
)
def test_pad_boxes(
    xyxy: np.ndarray,
    px: int,
    py: int | None,
    expected: np.ndarray,
) -> None:
    """pad_boxes expands each box by px horizontally and py (or px) vertically."""
    result = pad_boxes(xyxy=xyxy, px=px, py=py)
    np.testing.assert_array_equal(result, expected)
