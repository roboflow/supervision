from contextlib import nullcontext as DoesNotRaise

import numpy as np
import pytest

from supervision.key_points.core import KeyPoints
from test.test_utils import mock_key_points

KEY_POINTS = mock_key_points(
    xy=[
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
    ],
    confidence=[
        [0.8, 0.2, 0.6, 0.1, 0.5],
        [0.7, 0.9, 0.3, 0.4, 0.0],
        [0.1, 0.6, 0.8, 0.2, 0.7],
    ],
    class_id=[0, 1, 2],
)


@pytest.mark.parametrize(
    "key_points, index, expected_result, exception",
    [
        (
            KeyPoints.empty(),
            slice(None),
            KeyPoints.empty(),
            DoesNotRaise(),
        ),  # slice all key points when key points object empty
        (
            KEY_POINTS,
            slice(None),
            KEY_POINTS,
            DoesNotRaise(),
        ),  # slice all key points when key points object nonempty
        (
            KEY_POINTS,
            slice(0, 1),
            mock_key_points(
                xy=[[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]],
                confidence=[[0.8, 0.2, 0.6, 0.1, 0.5]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # select the first skeleton by slice
        (
            KEY_POINTS,
            slice(0, 2),
            mock_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ],
                confidence=[
                    [0.8, 0.2, 0.6, 0.1, 0.5],
                    [0.7, 0.9, 0.3, 0.4, 0.0],
                ],
                class_id=[0, 1],
            ),
            DoesNotRaise(),
        ),  # select the first skeleton by slice
        (
            KEY_POINTS,
            0,
            mock_key_points(
                xy=[[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]],
                confidence=[[0.8, 0.2, 0.6, 0.1, 0.5]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # select the first skeleton by index
        (
            KEY_POINTS,
            -1,
            mock_key_points(
                xy=[[[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]]],
                confidence=[[0.1, 0.6, 0.8, 0.2, 0.7]],
                class_id=[2],
            ),
            DoesNotRaise(),
        ),  # select the last skeleton by index
        (
            KEY_POINTS,
            [0, 1],
            mock_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ],
                confidence=[
                    [0.8, 0.2, 0.6, 0.1, 0.5],
                    [0.7, 0.9, 0.3, 0.4, 0.0],
                ],
                class_id=[0, 1],
            ),
            DoesNotRaise(),
        ),  # select the first two skeletons by index; list
        (
            KEY_POINTS,
            np.array([0, 1]),
            mock_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ],
                confidence=[
                    [0.8, 0.2, 0.6, 0.1, 0.5],
                    [0.7, 0.9, 0.3, 0.4, 0.0],
                ],
                class_id=[0, 1],
            ),
            DoesNotRaise(),
        ),  # select the first two skeletons by index; np.array
        (
            KEY_POINTS,
            [True, True, False],
            mock_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ],
                confidence=[
                    [0.8, 0.2, 0.6, 0.1, 0.5],
                    [0.7, 0.9, 0.3, 0.4, 0.0],
                ],
                class_id=[0, 1],
            ),
            DoesNotRaise(),
        ),  # select only skeletons associated with positive filter; list
        (
            KEY_POINTS,
            np.array([True, True, False]),
            mock_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ],
                confidence=[
                    [0.8, 0.2, 0.6, 0.1, 0.5],
                    [0.7, 0.9, 0.3, 0.4, 0.0],
                ],
                class_id=[0, 1],
            ),
            DoesNotRaise(),
        ),  # select only skeletons associated with positive filter; list
        (
            KEY_POINTS,
            (slice(None), slice(None)),
            KEY_POINTS,
            DoesNotRaise(),
        ),  # slice all anchors from all skeletons
        (
            KEY_POINTS,
            (slice(None), slice(0, 1)),
            mock_key_points(
                xy=[[[0, 1]], [[10, 11]], [[20, 21]]],
                confidence=[[0.8], [0.7], [0.1]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # slice the first anchor from every skeleton
        (
            KEY_POINTS,
            (slice(None), slice(0, 2)),
            mock_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # slice the first anchor two anchors from every skeleton
        (
            KEY_POINTS,
            (slice(None), 0),
            mock_key_points(
                xy=[[[0, 1]], [[10, 11]], [[20, 21]]],
                confidence=[[0.8], [0.7], [0.1]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the first anchor from every skeleton by index
        (
            KEY_POINTS,
            (slice(None), -1),
            mock_key_points(
                xy=[[[8, 9]], [[18, 19]], [[28, 29]]],
                confidence=[[0.5], [0.0], [0.7]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the last anchor from every skeleton by index
        (
            KEY_POINTS,
            (slice(None), [0, 1]),
            mock_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the first two anchors from every skeleton by index; list
        (
            KEY_POINTS,
            (slice(None), np.array([0, 1])),
            mock_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the first two anchors from every skeleton by index; np.array
        (
            KEY_POINTS,
            (slice(None), [True, True, False, False, False]),
            mock_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select only anchors associated with positive filter; list
        (
            KEY_POINTS,
            (slice(None), np.array([True, True, False, False, False])),
            mock_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select only anchors associated with positive filter; np.array
        (
            KEY_POINTS,
            (0, 0),
            mock_key_points(
                xy=[
                    [[0, 1]],
                ],
                confidence=[
                    [0.8],
                ],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # select the first anchor from the first skeleton by index
        (
            KEY_POINTS,
            (0, -1),
            mock_key_points(
                xy=[
                    [[8, 9]],
                ],
                confidence=[
                    [0.5],
                ],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # select the last anchor from the first skeleton by index
    ],
)
def test_key_points_getitem(key_points, index, expected_result, exception):
    with exception:
        result = key_points[index]
        assert result == expected_result
