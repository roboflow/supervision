from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.utils.boxes import (
    clip_boxes,
    denormalize_boxes,
    move_boxes,
    scale_boxes,
)


@pytest.mark.parametrize(
    "xyxy, resolution_wh, expected_result",
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
    "xyxy, offset, expected_result, exception",
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
    "xyxy, factor, expected_result, exception",
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
    "xyxy, resolution_wh, normalization_factor, expected_result, exception",
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
