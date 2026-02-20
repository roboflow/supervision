from contextlib import nullcontext as DoesNotRaise

import numpy as np
import pytest

from supervision.key_points.core import KeyPoints
from tests.helpers import _create_key_points

KEY_POINTS = _create_key_points(
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
    ("key_points", "index", "expected_result", "exception"),
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
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]],
                confidence=[[0.8, 0.2, 0.6, 0.1, 0.5]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # select the first skeleton by slice
        (
            KEY_POINTS,
            slice(0, 2),
            _create_key_points(
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
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]],
                confidence=[[0.8, 0.2, 0.6, 0.1, 0.5]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # select the first skeleton by index
        (
            KEY_POINTS,
            -1,
            _create_key_points(
                xy=[[[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]]],
                confidence=[[0.1, 0.6, 0.8, 0.2, 0.7]],
                class_id=[2],
            ),
            DoesNotRaise(),
        ),  # select the last skeleton by index
        (
            KEY_POINTS,
            [0, 1],
            _create_key_points(
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
            _create_key_points(
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
            _create_key_points(
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
            _create_key_points(
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
            _create_key_points(
                xy=[[[0, 1]], [[10, 11]], [[20, 21]]],
                confidence=[[0.8], [0.7], [0.1]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # slice the first anchor from every skeleton
        (
            KEY_POINTS,
            (slice(None), slice(0, 2)),
            _create_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # slice the first anchor two anchors from every skeleton
        (
            KEY_POINTS,
            (slice(None), 0),
            _create_key_points(
                xy=[[[0, 1]], [[10, 11]], [[20, 21]]],
                confidence=[[0.8], [0.7], [0.1]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the first anchor from every skeleton by index
        (
            KEY_POINTS,
            (slice(None), -1),
            _create_key_points(
                xy=[[[8, 9]], [[18, 19]], [[28, 29]]],
                confidence=[[0.5], [0.0], [0.7]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the last anchor from every skeleton by index
        (
            KEY_POINTS,
            (slice(None), [0, 1]),
            _create_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the first two anchors from every skeleton by index; list
        (
            KEY_POINTS,
            (slice(None), np.array([0, 1])),
            _create_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select the first two anchors from every skeleton by index; np.array
        (
            KEY_POINTS,
            (slice(None), [True, True, False, False, False]),
            _create_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select only anchors associated with positive filter; list
        (
            KEY_POINTS,
            (slice(None), np.array([True, True, False, False, False])),
            _create_key_points(
                xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                confidence=[[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # select only anchors associated with positive filter; np.array
        (
            KEY_POINTS,
            (0, 0),
            _create_key_points(xy=[[[0, 1]]], confidence=[[0.8]], class_id=[0]),
            DoesNotRaise(),
        ),  # select the first anchor from the first skeleton by index
        (
            KEY_POINTS,
            (0, -1),
            _create_key_points(xy=[[[8, 9]]], confidence=[[0.5]], class_id=[0]),
            DoesNotRaise(),
        ),  # select the last anchor from the first skeleton by index
    ],
)
def test_key_points_getitem(key_points, index, expected_result, exception):
    with exception:
        result = key_points[index]
        assert result == expected_result


def test_key_points_empty():
    """Test the creation and behavior of an empty KeyPoints object."""
    empty_key_points = KeyPoints.empty()
    assert len(empty_key_points) == 0
    assert empty_key_points.is_empty()
    assert empty_key_points.xy.shape == (0, 0, 2)


def test_key_points_is_empty():
    """Test the is_empty method for KeyPoints objects."""
    empty_key_points = KeyPoints.empty()
    assert empty_key_points.is_empty()

    non_empty_key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]]],
        confidence=[[0.8, 0.9]],
        class_id=[0],
    )
    assert not non_empty_key_points.is_empty()


def test_key_points_setitem():
    """Test the __setitem__ method for KeyPoints objects."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]]],
        confidence=[[0.8, 0.9]],
        class_id=[0],
    )

    key_points["custom_data"] = ["value1"]
    assert "custom_data" in key_points.data
    assert np.array_equal(key_points.data["custom_data"], np.array(["value1"]))

    with pytest.raises(TypeError, match=r"Value must be a np\.ndarray or a list"):
        key_points["invalid_data"] = 123


@pytest.mark.parametrize(
    ("key_points", "expected_xyxy", "expected_confidence", "expected_class_id"),
    [
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.9, 0.7]],
                class_id=[0],
            ),
            np.array([[0, 1, 4, 5]], dtype=np.float32),
            np.array([0.8], dtype=np.float32),
            np.array([0]),
        ),
        (
            _create_key_points(
                xy=[[[0, 0], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.9, 0.7]],
                class_id=[0],
            ),
            np.array([[2, 3, 4, 5]], dtype=np.float32),
            np.array([0.8], dtype=np.float32),
            np.array([0]),
        ),
    ],
)
def test_key_points_as_detections(
    key_points, expected_xyxy, expected_confidence, expected_class_id
):
    """Test the as_detections method for KeyPoints objects."""
    detections = key_points.as_detections()
    assert len(detections) == len(expected_xyxy)
    assert np.array_equal(detections.xyxy, expected_xyxy)
    assert np.allclose(detections.confidence, expected_confidence)
    assert np.array_equal(detections.class_id, expected_class_id)


def test_key_points_as_detections_empty():
    """Test the as_detections method for empty KeyPoints objects."""
    empty_key_points = KeyPoints.empty()
    empty_detections = empty_key_points.as_detections()
    assert empty_detections.is_empty()


def test_key_points_as_detections_with_data():
    """Test the as_detections method preserves data."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3], [4, 5]]],
        confidence=[[0.8, 0.9, 0.7]],
        class_id=[0],
    )
    key_points["custom_data"] = ["value1"]
    detections = key_points.as_detections()
    assert "custom_data" in detections.data
    assert np.array_equal(detections.data["custom_data"], np.array(["value1"]))


def test_key_points_iteration():
    """Test the iteration over KeyPoints objects."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        confidence=[[0.8, 0.9], [0.7, 0.6]],
        class_id=[0, 1],
    )

    iterations = 0
    for i, (xy, confidence, class_id, data) in enumerate(key_points):
        iterations += 1
        assert xy.shape == (2, 2)
        assert confidence.shape == (2,)
        assert class_id in [0, 1]
        assert isinstance(data, dict)
    assert iterations == 2


def test_key_points_iteration_no_confidence():
    """Test the iteration over KeyPoints objects without confidence."""
    key_points_no_conf = _create_key_points(
        xy=[[[0, 1], [2, 3]]],
        confidence=None,
        class_id=[0],
    )
    for xy, confidence, class_id, data in key_points_no_conf:
        assert confidence is None


@pytest.mark.parametrize(
    ("key_points1", "key_points2", "expected_equal"),
    [
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
            ),
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
            ),
            True,
        ),
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
            ),
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[1]
            ),
            False,
        ),
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
            ),
            _create_key_points(
                xy=[[[0, 1], [2, 4]]], confidence=[[0.8, 0.9]], class_id=[0]
            ),
            False,
        ),
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
            ),
            _create_key_points(
                xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.8]], class_id=[0]
            ),
            False,
        ),
    ],
)
def test_key_points_equality(key_points1, key_points2, expected_equal):
    """Test the equality comparison for KeyPoints objects."""
    status = key_points1 == key_points2
    assert status is expected_equal


def test_key_points_equality_with_data():
    """Test the equality comparison for KeyPoints objects with data."""
    key_points1 = _create_key_points(
        xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
    )
    key_points2 = _create_key_points(
        xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
    )
    key_points2["custom"] = ["value"]
    assert key_points1 != key_points2
