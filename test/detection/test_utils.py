from contextlib import ExitStack as DoesNotRaise
from typing import Optional, Tuple

import pytest

import numpy as np

from supervision.detection.utils import non_max_suppression, clip_boxes


@pytest.mark.parametrize(
    "predictions, iou_threshold, expected_result, exception",
    [
        (
            np.empty(shape=(0, 5)),
            0.5,
            np.array([]),
            DoesNotRaise()
        ),  # single box with no category
        (
            np.array([
                [10.0, 10.0, 40.0, 40.0, 0.8]
            ]),
            0.5,
            np.array([
                True
            ]),
            DoesNotRaise()
        ),  # single box with no category
        (
            np.array([
                [10.0, 10.0, 40.0, 40.0, 0.8, 0]
            ]),
            0.5,
            np.array([
                True
            ]),
            DoesNotRaise()
        ),  # single box with category
        (
            np.array([
                [10.0, 10.0, 40.0, 40.0, 0.8],
                [15.0, 15.0, 40.0, 40.0, 0.9],
            ]),
            0.5,
            np.array([
                False,
                True
            ]),
            DoesNotRaise()
        ),  # two boxes with no category
        (
            np.array([
                [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                [15.0, 15.0, 40.0, 40.0, 0.9, 1],
            ]),
            0.5,
            np.array([
                True,
                True
            ]),
            DoesNotRaise()
        ),  # two boxes with different category
        (
            np.array([
                [10.0, 10.0, 40.0, 40.0, 0.8, 0],
                [15.0, 15.0, 40.0, 40.0, 0.9, 0],
            ]),
            0.5,
            np.array([
                False,
                True
            ]),
            DoesNotRaise()
        ),  # two boxes with same category
        (
            np.array([
                [0.0, 0.0, 30.0, 40.0, 0.8],
                [5.0, 5.0, 35.0, 45.0, 0.9],
                [10.0, 10.0, 40.0, 50.0, 0.85],
            ]),
            0.5,
            np.array([
                False,
                True,
                False
            ]),
            DoesNotRaise()
        ),  # three boxes with no category
        (
            np.array([
                [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                [5.0, 5.0, 35.0, 45.0, 0.9, 1],
                [10.0, 10.0, 40.0, 50.0, 0.85, 2],
            ]),
            0.5,
            np.array([
                True,
                True,
                True
            ]),
            DoesNotRaise()
        ),  # three boxes with same category
        (
            np.array([
                [0.0, 0.0, 30.0, 40.0, 0.8, 0],
                [5.0, 5.0, 35.0, 45.0, 0.9, 0],
                [10.0, 10.0, 40.0, 50.0, 0.85, 1],
            ]),
            0.5,
            np.array([
                False,
                True,
                True
            ]),
            DoesNotRaise()
        ),  # three boxes with different category
    ]
)
def test_non_max_suppression(
        predictions: np.ndarray,
        iou_threshold: float,
        expected_result: Optional[np.ndarray],
        exception: Exception
) -> None:
    with exception:
        result = non_max_suppression(predictions=predictions, iou_threshold=iou_threshold)
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "boxes_xyxy, frame_resolution_wh, expected_result",
    [
        (
            np.empty(shape=(0, 4)),
            (1280, 720),
            np.empty(shape=(0, 4)),
        ),
        (
            np.array([
                [1.0, 1.0, 1279.0, 719.0]
            ]),
            (1280, 720),
            np.array([
                [1.0, 1.0, 1279.0, 719.0]
            ]),
        ),
        (
            np.array([
                [-1.0, 1.0, 1279.0, 719.0]
            ]),
            (1280, 720),
            np.array([
                [0.0, 1.0, 1279.0, 719.0]
            ]),
        ),
        (
            np.array([
                [1.0, -1.0, 1279.0, 719.0]
            ]),
            (1280, 720),
            np.array([
                [1.0, 0.0, 1279.0, 719.0]
            ]),
        ),
        (
            np.array([
                [1.0, 1.0, 1281.0, 719.0]
            ]),
            (1280, 720),
            np.array([
                [1.0, 1.0, 1280.0, 719.0]
            ]),
        ),
        (
            np.array([
                [1.0, 1.0, 1279.0, 721.0]
            ]),
            (1280, 720),
            np.array([
                [1.0, 1.0, 1279.0, 720.0]
            ]),
        ),
    ]
)
def test_clip_boxes(boxes_xyxy: np.ndarray, frame_resolution_wh: Tuple[int, int], expected_result: np.ndarray) -> None:
    result = clip_boxes(boxes_xyxy=boxes_xyxy, frame_resolution_wh=frame_resolution_wh)
    assert np.array_equal(result, expected_result)
