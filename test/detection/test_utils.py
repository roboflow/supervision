from contextlib import ExitStack as DoesNotRaise
from typing import Optional, Tuple, List

import pytest

import numpy as np

from supervision.detection.utils import non_max_suppression, clip_boxes, filter_polygons_by_area


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


@pytest.mark.parametrize(
    "polygons, min_area, max_area, expected_result, exception",
    [
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise()
        ),  # single polygon without area constraints
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            50,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise()
        ),  # single polygon with min_area constraint
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            50,
            [],
            DoesNotRaise()
        ),  # single polygon with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]])
            ],
            200,
            None,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise()
        ),  # two polygons with min_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]])
            ],
            None,
            200,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise()
        ),  # two polygons with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]])
            ],
            200,
            200,
            [],
            DoesNotRaise()
        ),  # two polygons with both area constraints
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]])
            ],
            100,
            100,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise()
        ),  # two polygons with min_area and max_area equal to the area of the first polygon
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]])
            ],
            400,
            400,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise()
        ),  # two polygons with min_area and max_area equal to the area of the second polygon
    ]
)
def test_filter_polygons_by_area(
        polygons: List[np.ndarray],
        min_area: Optional[float],
        max_area: Optional[float],
        expected_result: List[np.ndarray],
        exception: Exception
) -> None:
    with exception:
        result = filter_polygons_by_area(polygons=polygons, min_area=min_area, max_area=max_area)
        assert len(result) == len(expected_result)
        for result_polygon, expected_result_polygon in zip(result, expected_result):
            assert np.array_equal(result_polygon, expected_result_polygon)
