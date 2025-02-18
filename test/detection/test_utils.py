from contextlib import ExitStack as DoesNotRaise
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.utils import (
    calculate_masks_centroids,
    clip_boxes,
    contains_holes,
    contains_multiple_segments,
    filter_polygons_by_area,
    get_data_item,
    merge_data,
    merge_metadata,
    move_boxes,
    move_masks,
    process_roboflow_result,
    scale_boxes,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_xywh,
)

TEST_MASK = np.zeros((1, 1000, 1000), dtype=bool)
TEST_MASK[:, 300:351, 200:251] = True


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
    resolution_wh: Tuple[int, int],
    expected_result: np.ndarray,
) -> None:
    result = clip_boxes(xyxy=xyxy, resolution_wh=resolution_wh)
    assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "polygons, min_area, max_area, expected_result, exception",
    [
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # single polygon without area constraints
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            50,
            None,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # single polygon with min_area constraint
        (
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            None,
            50,
            [],
            DoesNotRaise(),
        ),  # single polygon with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            200,
            None,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            None,
            200,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # two polygons with max_area constraint
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            200,
            200,
            [],
            DoesNotRaise(),
        ),  # two polygons with both area constraints
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            100,
            100,
            [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area and
        # max_area equal to the area of the first polygon
        (
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 0], [0, 20], [20, 20], [20, 0]]),
            ],
            400,
            400,
            [np.array([[0, 0], [0, 20], [20, 20], [20, 0]])],
            DoesNotRaise(),
        ),  # two polygons with min_area and
        # max_area equal to the area of the second polygon
    ],
)
def test_filter_polygons_by_area(
    polygons: List[np.ndarray],
    min_area: Optional[float],
    max_area: Optional[float],
    expected_result: List[np.ndarray],
    exception: Exception,
) -> None:
    with exception:
        result = filter_polygons_by_area(
            polygons=polygons, min_area=min_area, max_area=max_area
        )
        assert len(result) == len(expected_result)
        for result_polygon, expected_result_polygon in zip(result, expected_result):
            assert np.array_equal(result_polygon, expected_result_polygon)


@pytest.mark.parametrize(
    "roboflow_result, expected_result, exception",
    [
        (
            {"predictions": [], "image": {"width": 1000, "height": 1000}},
            (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.empty(0)},
            ),
            DoesNotRaise(),
        ),  # empty result
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # single correct object detection result
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "tracker_id": 1,
                    },
                    {
                        "x": 500.0,
                        "y": 500.0,
                        "width": 100.0,
                        "height": 100.0,
                        "confidence": 0.8,
                        "class_id": 7,
                        "class": "truck",
                        "tracker_id": 2,
                    },
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0], [450.0, 450.0, 550.0, 550.0]]),
                np.array([0.9, 0.8]),
                np.array([0, 7]),
                None,
                np.array([1, 2]),
                {CLASS_NAME_DATA_FIELD: np.array(["person", "truck"])},
            ),
            DoesNotRaise(),
        ),  # two correct object detection result
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [],
                        "tracker_id": None,
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.empty(0)},
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no points
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [{"x": 200.0, "y": 300.0}, {"x": 250.0, "y": 300.0}],
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.empty(0)},
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no enough points
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [
                            {"x": 200.0, "y": 300.0},
                            {"x": 250.0, "y": 300.0},
                            {"x": 250.0, "y": 350.0},
                            {"x": 200.0, "y": 350.0},
                        ],
                    }
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                TEST_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no enough points
        (
            {
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 300.0,
                        "width": 50.0,
                        "height": 50.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class": "person",
                        "points": [
                            {"x": 200.0, "y": 300.0},
                            {"x": 250.0, "y": 300.0},
                            {"x": 250.0, "y": 350.0},
                            {"x": 200.0, "y": 350.0},
                        ],
                    },
                    {
                        "x": 500.0,
                        "y": 500.0,
                        "width": 100.0,
                        "height": 100.0,
                        "confidence": 0.8,
                        "class_id": 7,
                        "class": "truck",
                        "points": [],
                    },
                ],
                "image": {"width": 1000, "height": 1000},
            },
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                TEST_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # two instance segmentation results - one correct, one incorrect
    ],
)
def test_process_roboflow_result(
    roboflow_result: dict,
    expected_result: Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray
    ],
    exception: Exception,
) -> None:
    with exception:
        result = process_roboflow_result(roboflow_result=roboflow_result)
        assert np.array_equal(result[0], expected_result[0])
        assert np.array_equal(result[1], expected_result[1])
        assert np.array_equal(result[2], expected_result[2])
        assert (result[3] is None and expected_result[3] is None) or (
            np.array_equal(result[3], expected_result[3])
        )
        assert (result[4] is None and expected_result[4] is None) or (
            np.array_equal(result[4], expected_result[4])
        )
        for key in result[5]:
            if isinstance(result[5][key], np.ndarray):
                assert np.array_equal(
                    result[5][key], expected_result[5][key]
                ), f"Mismatch in arrays for key {key}"
            else:
                assert (
                    result[5][key] == expected_result[5][key]
                ), f"Mismatch in non-array data for key {key}"


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
    "masks, offset, resolution_wh, expected_result, exception",
    [
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([0, 0]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-1, -1]),
            (4, 4),
            np.array(
                [
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-2, -2]),
            (4, 4),
            np.array(
                [
                    [
                        [True, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-3, -3]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-2, -1]),
            (4, 4),
            np.array(
                [
                    [
                        [True, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-1, -2]),
            (4, 4),
            np.array(
                [
                    [
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-2, 2]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [True, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([3, 3]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([3, 3]),
            (6, 6),
            np.array(
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, True, True],
                        [False, False, False, False, True, True],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_move_masks(
    masks: np.ndarray,
    offset: np.ndarray,
    resolution_wh: Tuple[int, int],
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = move_masks(masks=masks, offset=offset, resolution_wh=resolution_wh)
        np.testing.assert_array_equal(result, expected_result)


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
    "masks, expected_result, exception",
    [
        (
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            np.array([[0, 0]]),
            DoesNotRaise(),
        ),  # single mask with all zeros
        (
            np.array(
                [
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ]
            ),
            np.array([[2, 2]]),
            DoesNotRaise(),
        ),  # single mask with all ones
        (
            np.array(
                [
                    [
                        [0, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 0],
                    ]
                ]
            ),
            np.array([[2, 2]]),
            DoesNotRaise(),
        ),  # single mask with symmetric ones
        (
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            np.array([[3, 2]]),
            DoesNotRaise(),
        ),  # single mask with asymmetric ones
        (
            np.array(
                [
                    [
                        [0, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
            np.array([[2, 2], [3, 2]]),
            DoesNotRaise(),
        ),  # two masks
    ],
)
def test_calculate_masks_centroids(
    masks: np.ndarray,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = calculate_masks_centroids(masks=masks)
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "data_list, expected_result, exception",
    [
        (
            [],
            {},
            DoesNotRaise(),
        ),  # empty data list
        (
            [{}],
            {},
            DoesNotRaise(),
        ),  # single empty data dict
        (
            [{}, {}],
            {},
            DoesNotRaise(),
        ),  # two empty data dicts
        (
            [
                {"test_1": []},
            ],
            {"test_1": []},
            DoesNotRaise(),
        ),  # single data dict with a single field name and empty list values
        (
            [
                {"test_1": []},
                {"test_1": []},
            ],
            {"test_1": []},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and empty list values
        (
            [
                {"test_1": np.array([])},
            ],
            {"test_1": np.array([])},
            DoesNotRaise(),
        ),  # single data dict with a single field name and empty np.array values
        (
            [
                {"test_1": np.array([])},
                {"test_1": np.array([])},
            ],
            {"test_1": np.array([])},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and empty np.array values
        (
            [
                {"test_1": [1, 2, 3]},
            ],
            {"test_1": [1, 2, 3]},
            DoesNotRaise(),
        ),  # single data dict with a single field name and list values
        (
            [
                {"test_1": []},
                {"test_1": [3, 2, 1]},
            ],
            {"test_1": [3, 2, 1]},
            DoesNotRaise(),
        ),  # two data dicts with the same field name; one of with empty list as value
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_1": [3, 2, 1]},
            ],
            {"test_1": [1, 2, 3, 3, 2, 1]},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and list values
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_1": [3, 2, 1]},
                {"test_1": [1, 2, 3]},
            ],
            {"test_1": [1, 2, 3, 3, 2, 1, 1, 2, 3]},
            DoesNotRaise(),
        ),  # three data dicts with the same field name and list values
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_2": [3, 2, 1]},
            ],
            None,
            pytest.raises(ValueError),
        ),  # two data dicts with different field names
        (
            [
                {"test_1": np.array([1, 2, 3])},
                {"test_1": np.array([3, 2, 1])},
            ],
            {"test_1": np.array([1, 2, 3, 3, 2, 1])},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and np.array values as 1D arrays
        (
            [
                {"test_1": np.array([[1, 2, 3]])},
                {"test_1": np.array([[3, 2, 1]])},
            ],
            {"test_1": np.array([[1, 2, 3], [3, 2, 1]])},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and np.array values as 2D arrays
        (
            [
                {"test_1": np.array([1, 2, 3]), "test_2": np.array(["a", "b", "c"])},
                {"test_1": np.array([3, 2, 1]), "test_2": np.array(["c", "b", "a"])},
            ],
            {
                "test_1": np.array([1, 2, 3, 3, 2, 1]),
                "test_2": np.array(["a", "b", "c", "c", "b", "a"]),
            },
            DoesNotRaise(),
        ),  # two data dicts with the same field names and np.array values
        (
            [
                {"test_1": [1, 2, 3], "test_2": np.array(["a", "b", "c"])},
                {"test_1": [3, 2, 1], "test_2": np.array(["c", "b", "a"])},
            ],
            {
                "test_1": [1, 2, 3, 3, 2, 1],
                "test_2": np.array(["a", "b", "c", "c", "b", "a"]),
            },
            DoesNotRaise(),
        ),  # two data dicts with the same field names and mixed values
        (
            [
                {"test_1": np.array([1, 2, 3])},
                {"test_1": np.array([[3, 2, 1]])},
            ],
            None,
            pytest.raises(ValueError),
        ),  # two data dicts with the same field name and 1D and 2D arrays values
        (
            [
                {"test_1": np.array([1, 2, 3]), "test_2": np.array(["a", "b"])},
                {"test_1": np.array([3, 2, 1]), "test_2": np.array(["c", "b", "a"])},
            ],
            None,
            pytest.raises(ValueError),
        ),  # two data dicts with the same field name and different length arrays values
        (
            [{}, {"test_1": [1, 2, 3]}],
            None,
            pytest.raises(ValueError),
        ),  # two data dicts; one empty and one non-empty dict
        (
            [{"test_1": [], "test_2": []}, {"test_1": [1, 2, 3], "test_2": [1, 2, 3]}],
            {"test_1": [1, 2, 3], "test_2": [1, 2, 3]},
            DoesNotRaise(),
        ),  # two data dicts; one empty and one non-empty dict; same keys
        (
            [{"test_1": []}, {"test_1": [1, 2, 3], "test_2": [4, 5, 6]}],
            None,
            pytest.raises(ValueError),
        ),  # two data dicts; one empty and one non-empty dict; different keys
        (
            [
                {
                    "test_1": [1, 2, 3],
                    "test_2": [4, 5, 6],
                    "test_3": [7, 8, 9],
                },
                {"test_1": [1, 2, 3], "test_2": [4, 5, 6]},
            ],
            None,
            pytest.raises(ValueError),
        ),  # two data dicts; one with three keys, one with two keys
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_1": [1, 2, 3], "test_2": [1, 2, 3]},
            ],
            None,
            pytest.raises(ValueError),
        ),  # some keys missing in one dict
        (
            [
                {"test_1": [1, 2, 3], "test_2": ["a", "b"]},
                {"test_1": [4, 5], "test_2": ["c", "d", "e"]},
            ],
            None,
            pytest.raises(ValueError),
        ),  # different value lengths for the same key
    ],
)
def test_merge_data(
    data_list: List[Dict[str, Any]],
    expected_result: Optional[Dict[str, Any]],
    exception: Exception,
):
    with exception:
        result = merge_data(data_list=data_list)
        if expected_result is None:
            assert False, f"Expected an error, but got result {result}"

        for key in result:
            if isinstance(result[key], np.ndarray):
                assert np.array_equal(
                    result[key], expected_result[key]
                ), f"Mismatch in arrays for key {key}"
            else:
                assert (
                    result[key] == expected_result[key]
                ), f"Mismatch in non-array data for key {key}"


@pytest.mark.parametrize(
    "data, index, expected_result, exception",
    [
        ({}, 0, {}, DoesNotRaise()),  # empty data dict
        (
            {
                "test_1": [1, 2, 3],
            },
            0,
            {
                "test_1": [1],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and integer index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            0,
            {
                "test_1": np.array([1]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and integer index
        (
            {
                "test_1": [1, 2, 3],
            },
            slice(0, 2),
            {
                "test_1": [1, 2],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and slice index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            slice(0, 2),
            {
                "test_1": np.array([1, 2]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and slice index
        (
            {
                "test_1": [1, 2, 3],
            },
            -1,
            {
                "test_1": [3],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and negative integer index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            -1,
            {
                "test_1": np.array([3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and negative integer index
        (
            {
                "test_1": [1, 2, 3],
            },
            [0, 2],
            {
                "test_1": [1, 3],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and integer list index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            [0, 2],
            {
                "test_1": np.array([1, 3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and integer list index
        (
            {
                "test_1": [1, 2, 3],
            },
            np.array([0, 2]),
            {
                "test_1": [1, 3],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and integer np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([0, 2]),
            {
                "test_1": np.array([1, 3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and integer np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([True, True, True]),
            {
                "test_1": np.array([1, 2, 3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and all-true bool np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([False, False, False]),
            {
                "test_1": np.array([]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and all-false bool np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([False, True, False]),
            {
                "test_1": np.array([2]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and mixed bool np.array index
        (
            {"test_1": np.array([1, 2, 3]), "test_2": ["a", "b", "c"]},
            0,
            {"test_1": np.array([1]), "test_2": ["a"]},
            DoesNotRaise(),
        ),  # data dict with two fields and integer index
        (
            {"test_1": np.array([1, 2, 3]), "test_2": ["a", "b", "c"]},
            -1,
            {"test_1": np.array([3]), "test_2": ["c"]},
            DoesNotRaise(),
        ),  # data dict with two fields and negative integer index
        (
            {"test_1": np.array([1, 2, 3]), "test_2": ["a", "b", "c"]},
            np.array([False, True, False]),
            {"test_1": np.array([2]), "test_2": ["b"]},
            DoesNotRaise(),
        ),  # data dict with two fields and mixed bool np.array index
    ],
)
def test_get_data_item(
    data: Dict[str, Any],
    index: Any,
    expected_result: Optional[Dict[str, Any]],
    exception: Exception,
):
    with exception:
        result = get_data_item(data=data, index=index)
        for key in result:
            if isinstance(result[key], np.ndarray):
                assert np.array_equal(
                    result[key], expected_result[key]
                ), f"Mismatch in arrays for key {key}"
            else:
                assert (
                    result[key] == expected_result[key]
                ), f"Mismatch in non-array data for key {key}"


@pytest.mark.parametrize(
    "mask, expected_result, exception",
    [
        (
            np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # foreground object in one continuous piece
        (
            np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # foreground object in 2 separate elements
        (
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # no foreground pixels in mask
        (
            np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # only foreground pixels in mask
        (
            np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]).astype(
                bool
            ),
            True,
            DoesNotRaise(),
        ),  # foreground object has 1 hole
        (
            np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]).astype(
                bool
            ),
            True,
            DoesNotRaise(),
        ),  # foreground object has 2 holes
    ],
)
def test_contains_holes(
    mask: npt.NDArray[np.bool_], expected_result: bool, exception: Exception
) -> None:
    with exception:
        result = contains_holes(mask)
        assert result == expected_result


@pytest.mark.parametrize(
    "mask, connectivity, expected_result, exception",
    [
        (
            np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # foreground object in one continuous piece
        (
            np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            4,
            True,
            DoesNotRaise(),
        ),  # foreground object in 2 separate elements
        (
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # no foreground pixels in mask
        (
            np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # only foreground pixels in mask
        (
            np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # foreground object has 2 holes, but is in single piece
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]]).astype(
                bool
            ),
            4,
            True,
            DoesNotRaise(),
        ),  # foreground object in 2 elements with respect to 4-way connectivity
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]]).astype(
                bool
            ),
            8,
            False,
            DoesNotRaise(),
        ),  # foreground object in single piece with respect to 8-way connectivity
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]]).astype(
                bool
            ),
            5,
            None,
            pytest.raises(ValueError),
        ),  # Incorrect connectivity parameter value, raises ValueError
    ],
)
def test_contains_multiple_segments(
    mask: npt.NDArray[np.bool_],
    connectivity: int,
    expected_result: bool,
    exception: Exception,
) -> None:
    with exception:
        result = contains_multiple_segments(mask=mask, connectivity=connectivity)
        assert result == expected_result


@pytest.mark.parametrize(
    "xywh, expected_result",
    [
        (np.array([[10, 20, 30, 40]]), np.array([[10, 20, 40, 60]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 100, 100]]),
            np.array([[50, 50, 150, 150]]),
        ),  # large bounding box
        (
            np.array([[-10, -20, 30, 40]]),
            np.array([[-10, -20, 20, 20]]),
        ),  # negative coordinates
        (np.array([[50, 50, 0, 30]]), np.array([[50, 50, 50, 80]])),  # zero width
        (np.array([[50, 50, 20, 0]]), np.array([[50, 50, 70, 50]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xywh_to_xyxy(xywh: np.ndarray, expected_result: np.ndarray) -> None:
    result = xywh_to_xyxy(xywh)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "xyxy, expected_result",
    [
        (np.array([[10, 20, 40, 60]]), np.array([[10, 20, 30, 40]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 150, 150]]),
            np.array([[50, 50, 100, 100]]),
        ),  # large bounding box
        (
            np.array([[-10, -20, 20, 20]]),
            np.array([[-10, -20, 30, 40]]),
        ),  # negative coordinates
        (np.array([[50, 50, 50, 80]]), np.array([[50, 50, 0, 30]])),  # zero width
        (np.array([[50, 50, 70, 50]]), np.array([[50, 50, 20, 0]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xyxy_to_xywh(xyxy: np.ndarray, expected_result: np.ndarray) -> None:
    result = xyxy_to_xywh(xyxy)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "xcycwh, expected_result",
    [
        (np.array([[50, 50, 20, 30]]), np.array([[40, 35, 60, 65]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 100, 100]]),
            np.array([[0, 0, 100, 100]]),
        ),  # large bounding box centered at (50, 50)
        (
            np.array([[-10, -10, 20, 30]]),
            np.array([[-20, -25, 0, 5]]),
        ),  # negative coordinates
        (np.array([[50, 50, 0, 30]]), np.array([[50, 35, 50, 65]])),  # zero width
        (np.array([[50, 50, 20, 0]]), np.array([[40, 50, 60, 50]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xcycwh_to_xyxy(xcycwh: np.ndarray, expected_result: np.ndarray) -> None:
    result = xcycwh_to_xyxy(xcycwh)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "metadata_list, expected_result, exception",
    [
        # Identical metadata with a single key
        ([{"key1": "value1"}, {"key1": "value1"}], {"key1": "value1"}, DoesNotRaise()),
        # Identical metadata with multiple keys
        (
            [
                {"key1": "value1", "key2": "value2"},
                {"key1": "value1", "key2": "value2"},
            ],
            {"key1": "value1", "key2": "value2"},
            DoesNotRaise(),
        ),
        # Conflicting values for the same key
        ([{"key1": "value1"}, {"key1": "value2"}], None, pytest.raises(ValueError)),
        # Different sets of keys across dictionaries
        ([{"key1": "value1"}, {"key2": "value2"}], None, pytest.raises(ValueError)),
        # Empty metadata list
        ([], {}, DoesNotRaise()),
        # Empty metadata dictionaries
        ([{}, {}], {}, DoesNotRaise()),
        # Different declaration order for keys
        (
            [
                {"key1": "value1", "key2": "value2"},
                {"key2": "value2", "key1": "value1"},
            ],
            {"key1": "value1", "key2": "value2"},
            DoesNotRaise(),
        ),
        # Nested metadata dictionaries
        (
            [{"key1": {"sub_key": "sub_value"}}, {"key1": {"sub_key": "sub_value"}}],
            {"key1": {"sub_key": "sub_value"}},
            DoesNotRaise(),
        ),
        # Large metadata dictionaries with many keys
        (
            [
                {f"key{i}": f"value{i}" for i in range(100)},
                {f"key{i}": f"value{i}" for i in range(100)},
            ],
            {f"key{i}": f"value{i}" for i in range(100)},
            DoesNotRaise(),
        ),
        # Mixed types in list metadata values
        (
            [{"key1": ["value1", 2, True]}, {"key1": ["value1", 2, True]}],
            {"key1": ["value1", 2, True]},
            DoesNotRaise(),
        ),
        # Identical lists across metadata dictionaries
        (
            [{"key1": [1, 2, 3]}, {"key1": [1, 2, 3]}],
            {"key1": [1, 2, 3]},
            DoesNotRaise(),
        ),
        # Identical numpy arrays across metadata dictionaries
        (
            [{"key1": np.array([1, 2, 3])}, {"key1": np.array([1, 2, 3])}],
            {"key1": np.array([1, 2, 3])},
            DoesNotRaise(),
        ),
        # Identical numpy arrays across metadata dictionaries, different datatype
        (
            [
                {"key1": np.array([1, 2, 3], dtype=np.int32)},
                {"key1": np.array([1, 2, 3], dtype=np.int64)},
            ],
            {"key1": np.array([1, 2, 3])},
            DoesNotRaise(),
        ),
        # Conflicting lists for the same key
        ([{"key1": [1, 2, 3]}, {"key1": [4, 5, 6]}], None, pytest.raises(ValueError)),
        # Conflicting numpy arrays for the same key
        (
            [{"key1": np.array([1, 2, 3])}, {"key1": np.array([4, 5, 6])}],
            None,
            pytest.raises(ValueError),
        ),
        # Mixed data types: list and numpy array for the same key
        (
            [{"key1": [1, 2, 3]}, {"key1": np.array([1, 2, 3])}],
            None,
            pytest.raises(ValueError),
        ),
        # Empty lists and numpy arrays for the same key
        ([{"key1": []}, {"key1": np.array([])}], None, pytest.raises(ValueError)),
        # Identical multi-dimensional lists across metadata dictionaries
        (
            [{"key1": [[1, 2], [3, 4]]}, {"key1": [[1, 2], [3, 4]]}],
            {"key1": [[1, 2], [3, 4]]},
            DoesNotRaise(),
        ),
        # Identical multi-dimensional numpy arrays across metadata dictionaries
        (
            [
                {"key1": np.arange(4).reshape(2, 2)},
                {"key1": np.arange(4).reshape(2, 2)},
            ],
            {"key1": np.arange(4).reshape(2, 2)},
            DoesNotRaise(),
        ),
        # Conflicting multi-dimensional lists for the same key
        (
            [{"key1": [[1, 2], [3, 4]]}, {"key1": [[5, 6], [7, 8]]}],
            None,
            pytest.raises(ValueError),
        ),
        # Conflicting multi-dimensional numpy arrays for the same key
        (
            [
                {"key1": np.arange(4).reshape(2, 2)},
                {"key1": np.arange(4, 8).reshape(2, 2)},
            ],
            None,
            pytest.raises(ValueError),
        ),
        # Mixed types with multi-dimensional list and array for the same key
        (
            [{"key1": [[1, 2], [3, 4]]}, {"key1": np.arange(4).reshape(2, 2)}],
            None,
            pytest.raises(ValueError),
        ),
        # Identical higher-dimensional (3D) numpy arrays across
        # metadata dictionaries
        (
            [
                {"key1": np.arange(8).reshape(2, 2, 2)},
                {"key1": np.arange(8).reshape(2, 2, 2)},
            ],
            {"key1": np.arange(8).reshape(2, 2, 2)},
            DoesNotRaise(),
        ),
        # Differently-shaped higher-dimensional (3D) numpy arrays
        # across metadata dictionaries
        (
            [
                {"key1": np.arange(8).reshape(2, 2, 2)},
                {"key1": np.arange(8).reshape(4, 1, 2)},
            ],
            None,
            pytest.raises(ValueError),
        ),
    ],
)
def test_merge_metadata(metadata_list, expected_result, exception):
    with exception:
        result = merge_metadata(metadata_list)
        if expected_result is None:
            assert result is None, f"Expected an error, but got a result {result}"
        for key, value in result.items():
            assert key in expected_result
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected_result[key])
            else:
                assert value == expected_result[key]
