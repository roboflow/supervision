from contextlib import nullcontext as DoesNotRaise
from typing import Any

import numpy as np
import pytest

from supervision.config import (
    KEYPOINT_COVARIANCE_DATA_FIELD,
    KEYPOINT_VISIBILITY_DATA_FIELD,
)
from supervision.key_points.core import KeyPoints
from tests.helpers import (
    _create_key_points,
)

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
        (
            KEY_POINTS,
            np.array(
                [
                    [True, False, True, False, False],
                    [True, True, False, False, False],
                    [False, True, True, False, False],
                ]
            ),
            _create_key_points(
                xy=[
                    [[0, 1], [4, 5]],
                    [[10, 11], [12, 13]],
                    [[22, 23], [24, 25]],
                ],
                confidence=[[0.8, 0.6], [0.7, 0.9], [0.6, 0.8]],
                class_id=[0, 1, 2],
            ),
            DoesNotRaise(),
        ),  # filter keypoints by 2D boolean mask, same count per row
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
            ),
            np.array([[True, False, True]]),
            _create_key_points(
                xy=[[[0, 1], [4, 5]]],
                confidence=[[0.8, 0.6]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # filter keypoints by 2D boolean mask, single object
        (
            _create_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5]],
                    [[10, 11], [12, 13], [14, 15]],
                ],
                confidence=[
                    [0.8, 0.2, 0.6],
                    [0.1, 0.2, 0.3],
                ],
                class_id=[0, 1],
            ),
            np.array([[True, False, True], [False, False, False]]),
            None,
            pytest.raises(ValueError, match="different numbers of True values"),
        ),  # 2D boolean mask with different counts per row raises ValueError
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                class_id=[0],
            ),
            np.array([[True, False, True]]),
            _create_key_points(
                xy=[[[0, 1], [4, 5]]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # 2D boolean mask with confidence=None — no confidence array in result
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
            ),
            np.array([[True, False]]),
            None,
            pytest.raises(ValueError, match="column count"),
        ),  # 2D boolean mask column count mismatch raises ValueError
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
            ),
            np.array([[True, False, True], [True, False, True]]),
            None,
            pytest.raises(ValueError, match="row count"),
        ),  # 2D boolean mask row count mismatch raises ValueError
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                confidence=[[0.8, 0.2], [0.6, 0.9]],
                class_id=[0, 1],
            ),
            np.array([[False, False], [False, False]]),
            KeyPoints(
                xy=np.zeros((2, 0, 2), dtype=np.float32),
                keypoint_confidence=np.zeros((2, 0), dtype=np.float32),
                class_id=np.array([0, 1]),
            ),
            DoesNotRaise(),
        ),  # all-False 2D mask — all rows select 0 keypoints, equal counts → ok
        (
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
            ),
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
            ).keypoint_confidence
            > 0.5,
            _create_key_points(
                xy=[[[0, 1], [4, 5]]],
                confidence=[[0.8, 0.6]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # kp[kp.keypoint_confidence > 0.5] — single-object canonical use case
    ],
)
def test_key_points_getitem(key_points, index, expected_result, exception):
    with exception:
        result = key_points[index]
        assert result == expected_result


def test_key_points_select_returns_subset() -> None:
    """Select returns a typed KeyPoints subset for row indexes."""
    result = KEY_POINTS.select([0, 2])

    assert result == _create_key_points(
        xy=[
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
            [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
        ],
        confidence=[
            [0.8, 0.2, 0.6, 0.1, 0.5],
            [0.1, 0.6, 0.8, 0.2, 0.7],
        ],
        class_id=[0, 2],
    )


def test_key_points_get_data_returns_data_value() -> None:
    """Get data returns the stored data value or None."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]]],
        confidence=[[0.8, 0.2]],
        class_id=[0],
    )
    key_points["custom_data"] = ["value1"]

    assert key_points.get_data("custom_data") == ["value1"]
    assert key_points.get_data("missing") is None


KEY_POINTS_WITH_DET_CONF = _create_key_points(
    xy=[
        [[0, 1], [2, 3], [4, 5]],
        [[10, 11], [12, 13], [14, 15]],
        [[20, 21], [22, 23], [24, 25]],
    ],
    confidence=[
        [0.8, 0.2, 0.6],
        [0.7, 0.9, 0.3],
        [0.1, 0.6, 0.8],
    ],
    class_id=[0, 1, 0],
    detection_confidence=[0.95, 0.40, 0.85],
)


@pytest.mark.parametrize(
    ("key_points", "index", "expected_result"),
    [
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            KEY_POINTS_WITH_DET_CONF.detection_confidence > 0.5,
            _create_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5]],
                    [[20, 21], [22, 23], [24, 25]],
                ],
                confidence=[[0.8, 0.2, 0.6], [0.1, 0.6, 0.8]],
                class_id=[0, 0],
                detection_confidence=[0.95, 0.85],
            ),
            id="filter-by-detection-confidence-threshold",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            KEY_POINTS_WITH_DET_CONF.class_id == 0,
            _create_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5]],
                    [[20, 21], [22, 23], [24, 25]],
                ],
                confidence=[[0.8, 0.2, 0.6], [0.1, 0.6, 0.8]],
                class_id=[0, 0],
                detection_confidence=[0.95, 0.85],
            ),
            id="filter-by-class-id",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            KEY_POINTS_WITH_DET_CONF.class_id == 1,
            _create_key_points(
                xy=[[[10, 11], [12, 13], [14, 15]]],
                confidence=[[0.7, 0.9, 0.3]],
                class_id=[1],
                detection_confidence=[0.40],
            ),
            id="filter-by-class-id-single-result",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            (KEY_POINTS_WITH_DET_CONF.detection_confidence > 0.5)
            & (KEY_POINTS_WITH_DET_CONF.class_id == 0),
            _create_key_points(
                xy=[
                    [[0, 1], [2, 3], [4, 5]],
                    [[20, 21], [22, 23], [24, 25]],
                ],
                confidence=[[0.8, 0.2, 0.6], [0.1, 0.6, 0.8]],
                class_id=[0, 0],
                detection_confidence=[0.95, 0.85],
            ),
            id="filter-by-detection-confidence-and-class-id",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            KEY_POINTS_WITH_DET_CONF.detection_confidence > 0.99,
            KeyPoints(
                xy=np.zeros((0, 3, 2), dtype=np.float32),
                keypoint_confidence=np.zeros((0, 3), dtype=np.float32),
                detection_confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=int),
            ),
            id="filter-all-out-returns-empty",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            KEY_POINTS_WITH_DET_CONF.class_id == 99,
            KeyPoints(
                xy=np.zeros((0, 3, 2), dtype=np.float32),
                keypoint_confidence=np.zeros((0, 3), dtype=np.float32),
                detection_confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=int),
            ),
            id="filter-by-nonexistent-class-returns-empty",
        ),
        pytest.param(
            KeyPoints.empty(),
            np.array([], dtype=bool),
            KeyPoints.empty(),
            id="filter-empty-keypoints-stays-empty",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            KEY_POINTS_WITH_DET_CONF.detection_confidence > 0.0,
            KEY_POINTS_WITH_DET_CONF,
            id="filter-keeps-all-when-all-pass",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            np.int64(0),
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
                detection_confidence=[0.95],
            ),
            id="np-integer-scalar-with-det-conf",
        ),
        pytest.param(
            KEY_POINTS_WITH_DET_CONF,
            np.array(0),
            _create_key_points(
                xy=[[[0, 1], [2, 3], [4, 5]]],
                confidence=[[0.8, 0.2, 0.6]],
                class_id=[0],
                detection_confidence=[0.95],
            ),
            id="0d-ndarray-with-det-conf",
        ),
    ],
)
def test_key_points_getitem_detection_level(key_points, index, expected_result):
    """Detection-level filtering mirrors Detections API patterns."""
    result = key_points[index]
    assert result == expected_result


class TestKeyPointsSetItem:
    """Tests for assigning `data` values through `KeyPoints.__setitem__`."""

    @staticmethod
    def _two_skeletons() -> KeyPoints:
        """Build a two-object KeyPoints with one `name` entry per object."""
        xy = np.array(
            [[[10, 10], [20, 20]], [[30, 30], [40, 40]]],
            dtype=np.float32,
        )
        key_points = KeyPoints(xy=xy, class_id=np.array([0, 1]))
        key_points["name"] = ["a", "b"]
        return key_points

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            pytest.param(
                ["a", "b", "c"],
                r"Shape of np.ndarray for key 'new' must be \(2,\)",
                id="long-list",
            ),
            pytest.param(
                ["a"],
                r"Shape of np.ndarray for key 'new' must be \(2,\)",
                id="short-list",
            ),
            pytest.param(
                [],
                r"Shape of np.ndarray for key 'new' must be \(2,\)",
                id="empty-list",
            ),
            pytest.param(
                np.array(["a", "b", "c"]),
                r"Shape of np.ndarray for key 'new' must be \(2,\)",
                id="long-1d-array",
            ),
            pytest.param(
                np.array(["a"]),
                r"Shape of np.ndarray for key 'new' must be \(2,\)",
                id="short-1d-array",
            ),
            pytest.param(
                np.zeros((3, 2)),
                r"First dimension of np.ndarray for key 'new' must have size 2",
                id="wrong-first-dimension-2d-array",
            ),
        ],
    )
    def test_rejects_value_with_wrong_length(self, value: Any, message: str) -> None:
        """Assigning a value whose length differs from `len(key_points)` raises."""
        key_points = self._two_skeletons()

        with pytest.raises(ValueError, match=message):
            key_points["new"] = value

    def test_failed_assignment_leaves_data_unchanged(self) -> None:
        """A rejected assignment neither adds a new key nor overwrites an old one."""
        key_points = self._two_skeletons()

        with pytest.raises(ValueError, match=r"key 'new' must be \(2,\)"):
            key_points["new"] = ["a", "b", "c"]
        with pytest.raises(ValueError, match=r"key 'name' must be \(2,\)"):
            key_points["name"] = np.array(["x"])

        assert list(key_points.data) == ["name"]
        np.testing.assert_array_equal(key_points.data["name"], np.array(["a", "b"]))

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(["x", "y"], id="list"),
            pytest.param(np.array(["x", "y"]), id="1d-array"),
            pytest.param(np.zeros((2, 2)), id="per-keypoint-2d-array"),
            pytest.param(np.zeros((2, 2, 3)), id="per-keypoint-3d-array"),
        ],
    )
    def test_accepts_value_with_matching_length(self, value: Any) -> None:
        """A list or array with one entry per object is stored as an array."""
        key_points = self._two_skeletons()

        key_points["name"] = value

        np.testing.assert_array_equal(key_points.data["name"], np.asarray(value))

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(1, id="scalar"),
            pytest.param("ab", id="string"),
            pytest.param(("a", "b"), id="tuple"),
        ],
    )
    def test_rejects_value_that_is_not_list_or_ndarray(self, value: Any) -> None:
        """Values other than lists and arrays raise `TypeError`, as in Detections."""
        key_points = self._two_skeletons()

        with pytest.raises(TypeError, match=r"Value must be a np\.ndarray or a list"):
            key_points["new"] = value

    def test_empty_key_points_accepts_empty_list(self) -> None:
        """An empty KeyPoints accepts a zero-length value and rejects a longer one."""
        key_points = KeyPoints.empty()

        key_points["name"] = []

        assert len(key_points.data["name"]) == 0
        with pytest.raises(ValueError, match=r"must be \(0,\)"):
            key_points["other"] = ["a"]


class TestKeyPointsSelect:
    """Tests for selecting individual keypoints without losing their fields."""

    @pytest.mark.parametrize("axis", [0, 1])
    def test_rejects_mismatched_numpy_boolean_list(self, axis: int) -> None:
        """Reject NumPy boolean scalar lists that do not match the selected axis."""
        key_points = KeyPoints(xy=np.zeros((2, 3, 2)))
        mask = [np.bool_(True)]
        index = (mask, [0]) if axis == 0 else ([0], mask)

        with pytest.raises(IndexError, match="boolean index"):
            key_points.select(index)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_selects_with_numpy_boolean_list(self, axis: int) -> None:
        """Select matching entries using a list of NumPy boolean scalars."""
        key_points = KeyPoints(xy=np.arange(12).reshape(2, 3, 2))
        mask = [np.bool_(True)] + [np.bool_(False)] * (key_points.xy.shape[axis] - 1)
        index = (mask, [0]) if axis == 0 else ([0], mask)

        result = key_points.select(index)

        np.testing.assert_array_equal(result.xy, key_points.xy[:1, :1])

    @pytest.mark.parametrize("method", ["select", "__getitem__"])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("as_array", [False, True])
    @pytest.mark.parametrize("too_long", [False, True])
    @pytest.mark.parametrize("all_false", [False, True])
    def test_rejects_mismatched_boolean_mask_length(
        self,
        method: str,
        axis: int,
        as_array: bool,
        too_long: bool,
        all_false: bool,
    ) -> None:
        """Reject nonempty boolean masks that do not match their selected axis."""
        key_points = KeyPoints(xy=np.arange(12, dtype=float).reshape(2, 3, 2))
        mask_length = key_points.xy.shape[axis] + (1 if too_long else -1)
        mask = [False] * mask_length
        if not all_false:
            mask[0] = True
        selector = np.array(mask) if as_array else mask
        index = selector if axis == 0 else (slice(None), selector)

        with pytest.raises(IndexError, match="boolean index"):
            getattr(key_points, method)(index)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("empty", [False, True])
    def test_empty_boolean_mask_preserves_axis_shape(
        self, axis: int, empty: bool
    ) -> None:
        """An empty boolean mask remains a valid selection of no entries."""
        xy = np.zeros((0 if empty else 2, 3, 2), dtype=np.float32)
        key_points = KeyPoints(xy=xy)
        selector = np.array([], dtype=bool)
        index = selector if axis == 0 else (slice(None), selector)

        result = key_points.select(index)

        expected_shape = (0, 3, 2) if axis == 0 else (xy.shape[0], 0, 2)
        assert result.xy.shape == expected_shape

    @pytest.mark.parametrize("method", ["select", "__getitem__"])
    @pytest.mark.parametrize("row", [0, -1])
    @pytest.mark.parametrize(
        "anchors",
        [
            pytest.param([2, 0], id="integer-list"),
            pytest.param(np.array([2, 0]), id="integer-array"),
            pytest.param([True, False, True], id="boolean-list"),
            pytest.param(np.array([True, False, True]), id="boolean-array"),
        ],
    )
    def test_scalar_array_row_with_multiple_anchors(
        self, method: str, row: int, anchors: Any
    ) -> None:
        """Scalar array rows retain aligned fields when selecting multiple joints."""
        xy = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        keypoint_confidence = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        visible = np.array([[True, False, True], [False, True, False]])
        key_points = KeyPoints(
            xy=xy,
            keypoint_confidence=keypoint_confidence,
            visible=visible,
            class_id=np.array([2, 3]),
            detection_confidence=np.array([0.7, 0.8]),
            data={"names": ["first", "second"], "id": np.array([10, 20])},
        )

        result = getattr(key_points, method)((np.array(row), anchors))

        np.testing.assert_array_equal(result.xy, xy[row, anchors][None, ...])
        np.testing.assert_array_equal(
            result.keypoint_confidence, keypoint_confidence[row, anchors][None, ...]
        )
        np.testing.assert_array_equal(result.visible, visible[row, anchors][None, ...])
        np.testing.assert_array_equal(result.class_id, [2 if row == 0 else 3])
        np.testing.assert_array_equal(
            result.detection_confidence, [0.7 if row == 0 else 0.8]
        )
        np.testing.assert_array_equal(result.data["id"], [10 if row == 0 else 20])
        assert result.data["names"] == ["first" if row == 0 else "second"]

    @pytest.mark.parametrize(
        "anchor_index",
        [
            pytest.param([0, 1], id="list"),
            pytest.param(0, id="integer"),
            pytest.param(np.int64(0), id="numpy-integer"),
            pytest.param(np.array([0, 1]), id="integer-array"),
            pytest.param([True, False], id="boolean-list"),
            pytest.param(np.array([True, False]), id="boolean-array"),
        ],
    )
    def test_keypoint_selection_on_empty_result_stays_empty(
        self, anchor_index: Any
    ) -> None:
        """Selecting joints on a frame without poses returns an empty result."""
        key_points = KeyPoints.empty()

        result = key_points[:, anchor_index]

        assert isinstance(result, KeyPoints)
        assert result.is_empty()
        assert result.xy.shape == (0, 0, 2)

    @pytest.mark.parametrize("empty", [True, False])
    def test_zero_step_anchor_slice_raises(self, empty: bool) -> None:
        """Reject a zero-step anchor slice for empty and populated results."""
        key_points = KeyPoints.empty() if empty else KeyPoints(xy=np.zeros((1, 2, 2)))

        with pytest.raises(ValueError, match="slice step cannot be zero"):
            key_points[:, ::0]

    @pytest.mark.parametrize("method", ["select", "__getitem__"])
    @pytest.mark.parametrize("coordinate_count", [2, 3])
    @pytest.mark.parametrize("with_fields", [False, True])
    @pytest.mark.parametrize(
        "row_index",
        [
            1,
            pytest.param(np.int64(1), id="numpy-integer"),
            pytest.param(np.array(1), id="zero-dimensional-array"),
            pytest.param(np.int32(-1), id="negative-numpy-integer"),
        ],
    )
    @pytest.mark.parametrize(
        "anchor_index",
        [1, pytest.param(np.int64(1), id="numpy-integer")],
    )
    def test_preserves_coordinates_and_fields(
        self,
        method: str,
        coordinate_count: int,
        with_fields: bool,
        row_index: int | np.integer | np.ndarray,
        anchor_index: int | np.integer,
    ) -> None:
        """Single-point selection retains coordinate depth and aligned fields."""
        xy = np.arange(6 * coordinate_count, dtype=np.float32).reshape(
            2, 3, coordinate_count
        )
        key_points = KeyPoints(xy=xy)
        if with_fields:
            key_points = KeyPoints(
                xy=xy,
                keypoint_confidence=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                detection_confidence=np.array([0.7, 0.8]),
                visible=np.array([[True, False, True], [False, True, False]]),
                class_id=np.array([2, 3]),
                data={"names": ["first", "second"], "id": np.array([10, 20])},
            )
        select = getattr(key_points, method)

        result = select((row_index, anchor_index))

        np.testing.assert_array_equal(result.xy, xy[1:2, 1:2, :])
        assert result.xy.shape == (1, 1, coordinate_count)
        assert result.xy.dtype == xy.dtype
        if with_fields:
            np.testing.assert_array_equal(result.keypoint_confidence, [[0.5]])
            np.testing.assert_array_equal(result.detection_confidence, [0.8])
            np.testing.assert_array_equal(result.visible, [[True]])
            np.testing.assert_array_equal(result.class_id, [3])
            np.testing.assert_array_equal(result.data["id"], [20])
            assert result.data["names"] == ["second"]
        else:
            assert result.keypoint_confidence is None
            assert result.detection_confidence is None
            assert result.visible is None
            assert result.class_id is None
            assert result.data == {}


class TestKeyPointsSelectKeypointAlignedData:
    """Tests that keypoint-aligned ``data`` fields follow keypoint-axis selection."""

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            pytest.param(
                (slice(None), [2, 1, 0]), [[0, 1, 2], [1, 2, 0]], id="reorder"
            ),
            pytest.param((slice(None), [0, 2]), [[2, 0], [0, 1]], id="subset"),
            pytest.param(
                (slice(None), np.array([True, False, True])),
                [[2, 0], [0, 1]],
                id="boolean-mask",
            ),
            pytest.param((np.array(1), [2, 0]), [[1, 0]], id="scalar-array-row"),
            pytest.param((0, 1), [[1]], id="single-keypoint"),
            pytest.param(([1, 0], 2), [[1], [0]], id="scalar-keypoint"),
        ],
    )
    def test_indexes_visibility_along_the_keypoint_axis(
        self, index: Any, expected: list[list[int]]
    ) -> None:
        """The visibility field is selected along the keypoint axis, like ``visible``.

        Selected by skeleton only, a reordered or narrowed annotation would keep stale
        flags against moved coordinates and export them as such.
        """
        flags = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.float32)
        key_points = KeyPoints(
            xy=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
            visible=flags > 0,
            data={KEYPOINT_VISIBILITY_DATA_FIELD: flags},
        )

        result = key_points[index]

        assert isinstance(result, KeyPoints)
        np.testing.assert_array_equal(
            result.data[KEYPOINT_VISIBILITY_DATA_FIELD], expected
        )

    def test_two_dimensional_mask_selects_visibility_per_skeleton(self) -> None:
        """A 2D boolean mask picks each skeleton's own visibility values."""
        flags = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.float32)
        key_points = KeyPoints(
            xy=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
            data={KEYPOINT_VISIBILITY_DATA_FIELD: flags},
        )
        mask = np.array([[True, False, True], [False, True, True]])

        result = key_points[mask]

        assert isinstance(result, KeyPoints)
        np.testing.assert_array_equal(
            result.data[KEYPOINT_VISIBILITY_DATA_FIELD], [[2, 0], [2, 1]]
        )

    def test_keeps_other_data_fields_row_aligned(self) -> None:
        """A field not registered as keypoint-aligned is indexed by skeleton only.

        A per-skeleton box is as wide as a 4-keypoint skeleton, so the width of a value
        must not decide whether it follows the keypoint axis.
        """
        boxes = np.array([[0, 0, 4, 4], [1, 1, 5, 5]], dtype=np.float32)
        key_points = KeyPoints(
            xy=np.zeros((2, 4, 2), dtype=np.float32), data={"xyxy": boxes}
        )

        result = key_points[:, [3, 2, 1, 0]]

        assert isinstance(result, KeyPoints)
        np.testing.assert_array_equal(result.data["xyxy"], boxes)


class TestKeyPointsVisible:
    """Tests for the `visible` mask field on KeyPoints."""

    def test_visible_defaults_to_none(self):
        kp = _create_key_points(
            xy=[[[0, 1], [2, 3]]],
            confidence=[[0.9, 0.8]],
            class_id=[0],
        )
        assert kp.visible is None

    def test_visible_set_from_confidence_threshold(self):
        kp = _create_key_points(
            xy=[[[0, 1], [2, 3], [4, 5]]],
            confidence=[[0.9, 0.1, 0.6]],
            class_id=[0],
        )
        kp.visible = kp.keypoint_confidence > 0.5
        expected = np.array([[True, False, True]])
        np.testing.assert_array_equal(kp.visible, expected)

    def test_visible_preserved_on_skeleton_filter(self):
        kp = _create_key_points(
            xy=[
                [[0, 1], [2, 3]],
                [[10, 11], [12, 13]],
            ],
            confidence=[[0.9, 0.1], [0.3, 0.8]],
            class_id=[0, 1],
            detection_confidence=[0.95, 0.40],
            visible=[[True, False], [False, True]],
        )
        result = kp[kp.detection_confidence > 0.5]
        assert result.visible is not None
        np.testing.assert_array_equal(result.visible, np.array([[True, False]]))

    def test_visible_preserved_on_int_index(self):
        kp = _create_key_points(
            xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]]],
            confidence=[[0.9, 0.1], [0.3, 0.8]],
            class_id=[0, 1],
            visible=[[True, False], [False, True]],
        )
        result = kp[0]
        assert result.visible is not None
        np.testing.assert_array_equal(result.visible, np.array([[True, False]]))

    def test_visible_preserved_on_anchor_slice(self):
        kp = _create_key_points(
            xy=[[[0, 1], [2, 3], [4, 5]]],
            confidence=[[0.9, 0.1, 0.6]],
            class_id=[0],
            visible=[[True, False, True]],
        )
        result = kp[:, [0, 2]]
        assert result.visible is not None
        np.testing.assert_array_equal(result.visible, np.array([[True, True]]))

    def test_visible_preserved_on_2d_bool_mask(self):
        kp = _create_key_points(
            xy=[
                [[0, 1], [2, 3], [4, 5]],
                [[10, 11], [12, 13], [14, 15]],
            ],
            confidence=[[0.9, 0.1, 0.6], [0.7, 0.2, 0.8]],
            class_id=[0, 1],
            visible=[[True, False, True], [True, False, True]],
        )
        mask = np.array([[True, False, True], [True, False, True]])
        result = kp[mask]
        assert result.visible is not None
        np.testing.assert_array_equal(
            result.visible, np.array([[True, True], [True, True]])
        )

    def test_visible_none_stays_none_on_filter(self):
        kp = _create_key_points(
            xy=[[[0, 1], [2, 3]], [[10, 11], [12, 13]]],
            class_id=[0, 1],
        )
        result = kp[0]
        assert result.visible is None

    def test_equality_with_visible(self):
        kp1 = _create_key_points(
            xy=[[[0, 1], [2, 3]]],
            class_id=[0],
            visible=[[True, False]],
        )
        kp2 = _create_key_points(
            xy=[[[0, 1], [2, 3]]],
            class_id=[0],
            visible=[[True, False]],
        )
        kp3 = _create_key_points(
            xy=[[[0, 1], [2, 3]]],
            class_id=[0],
            visible=[[False, True]],
        )
        assert kp1 == kp2
        assert kp1 != kp3

    def test_equality_visible_none_vs_set(self):
        kp1 = _create_key_points(
            xy=[[[0, 1], [2, 3]]],
            class_id=[0],
        )
        kp2 = _create_key_points(
            xy=[[[0, 1], [2, 3]]],
            class_id=[0],
            visible=[[True, True]],
        )
        assert kp1 != kp2


def test_key_points_empty():
    """Test the creation and behavior of an empty KeyPoints object."""
    empty_key_points = KeyPoints.empty()
    assert len(empty_key_points) == 0
    assert empty_key_points.is_empty()
    assert empty_key_points.xy.shape == (0, 0, 2)


@pytest.mark.parametrize("field", ["detection_confidence", "keypoint_confidence"])
def test_key_points_empty_filters_by_confidence(field: str) -> None:
    """A frame without poses filters by confidence like a frame with poses."""
    key_points = KeyPoints.empty()

    filtered = key_points[getattr(key_points, field) > 0.5]

    assert filtered.is_empty()


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


def test_key_points_zero_length_slice_is_empty():
    """A filtered KeyPoints object with zero rows is empty."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]]],
        confidence=[[0.8, 0.9]],
        class_id=[0],
    )

    assert key_points[np.array([], dtype=int)].is_empty()


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


def test_key_points_getitem_empty_list():
    """Selecting with an empty list returns an empty KeyPoints, like Detections."""
    key_points = _create_key_points(
        xy=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        class_id=[0, 1],
    )

    result = key_points[[]]

    assert len(result) == 0
    assert result.is_empty()


@pytest.mark.parametrize(
    ("key_points_list", "expected_result", "exception"),
    [
        (
            [],
            KeyPoints.empty(),
            DoesNotRaise(),
        ),  # empty list
        (
            [KeyPoints.empty(), KeyPoints.empty()],
            KeyPoints.empty(),
            DoesNotRaise(),
        ),  # only empty KeyPoints
        (
            [
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    confidence=[[0.9, 0.8]],
                    class_id=[0],
                ),
            ],
            _create_key_points(
                xy=[[[10, 10], [20, 20]]],
                confidence=[[0.9, 0.8]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # single KeyPoints
        (
            [
                KeyPoints.empty(),
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    confidence=[[0.9, 0.8]],
                    class_id=[0],
                ),
                KeyPoints.empty(),
            ],
            _create_key_points(
                xy=[[[10, 10], [20, 20]]],
                confidence=[[0.9, 0.8]],
                class_id=[0],
            ),
            DoesNotRaise(),
        ),  # empty KeyPoints are ignored
        (
            [
                KeyPoints(xy=np.array([[[10, 10], [20, 20]]], dtype=np.float32)),
                KeyPoints(xy=np.array([[[30, 30], [40, 40]]], dtype=np.float32)),
            ],
            KeyPoints(
                xy=np.array(
                    [[[10, 10], [20, 20]], [[30, 30], [40, 40]]], dtype=np.float32
                )
            ),
            DoesNotRaise(),
        ),  # xy only; all optional fields None
        (
            [
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    confidence=[[0.9, 0.8]],
                    class_id=[0],
                    detection_confidence=[0.95],
                    visible=[[True, False]],
                    data={"class_name": ["person"]},
                ),
                _create_key_points(
                    xy=[[[30, 30], [40, 40]], [[50, 50], [60, 60]]],
                    confidence=[[0.7, 0.6], [0.5, 0.4]],
                    class_id=[1, 2],
                    detection_confidence=[0.85, 0.75],
                    visible=[[True, True], [False, True]],
                    data={"class_name": ["dog", "cat"]},
                ),
            ],
            _create_key_points(
                xy=[
                    [[10, 10], [20, 20]],
                    [[30, 30], [40, 40]],
                    [[50, 50], [60, 60]],
                ],
                confidence=[[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]],
                class_id=[0, 1, 2],
                detection_confidence=[0.95, 0.85, 0.75],
                visible=[[True, False], [True, True], [False, True]],
                data={"class_name": ["person", "dog", "cat"]},
            ),
            DoesNotRaise(),
        ),  # all fields populated on every input
        (
            [
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    confidence=[[0.9, 0.8]],
                    class_id=[0],
                ),
                KeyPoints(xy=np.array([[[30, 30], [40, 40]]], dtype=np.float32)),
            ],
            None,
            pytest.raises(ValueError, match="All or none of the 'class_id'"),
        ),  # class_id set on one input and None on the other
        (
            [
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    visible=[[True, False]],
                ),
                KeyPoints(xy=np.array([[[30, 30], [40, 40]]], dtype=np.float32)),
            ],
            None,
            pytest.raises(ValueError, match="All or none of the 'visible'"),
        ),  # visible set on one input and None on the other
        (
            [
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    detection_confidence=[0.9],
                ),
                KeyPoints(xy=np.array([[[30, 30], [40, 40]]], dtype=np.float32)),
            ],
            None,
            pytest.raises(
                ValueError, match="All or none of the 'detection_confidence'"
            ),
        ),  # detection_confidence set on one input and None on the other
        (
            [
                KeyPoints(xy=np.array([[[10, 10], [20, 20]]], dtype=np.float32)),
                KeyPoints(
                    xy=np.array([[[10, 10], [20, 20], [30, 30]]], dtype=np.float32)
                ),
            ],
            None,
            pytest.raises(ValueError, match="same number of keypoints"),
        ),  # mismatched keypoint counts per skeleton
        (
            [
                KeyPoints(xy=np.array([[[10, 10], [20, 20]]], dtype=np.float32)),
                KeyPoints(
                    xy=np.array([[[30, 30, 0.9], [40, 40, 0.8]]], dtype=np.float32)
                ),
            ],
            None,
            pytest.raises(ValueError, match=r"got depths \[2, 3\]"),
        ),  # mismatched coordinate depths per skeleton
        (
            [
                _create_key_points(
                    xy=[[[10, 10], [20, 20]]],
                    data={"class_name": ["person"]},
                ),
                _create_key_points(
                    xy=[[[30, 30], [40, 40]]],
                    data={"tracker_id": [7]},
                ),
            ],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # data dictionaries with mismatched keys
        pytest.param(
            [
                KeyPoints(xy=np.zeros((2, 0, 2), dtype=np.float32)),
                KeyPoints(xy=np.zeros((3, 0, 2), dtype=np.float32)),
            ],
            KeyPoints(xy=np.zeros((5, 0, 2), dtype=np.float32)),
            DoesNotRaise(),
            id="zero-keypoints-merge-succeeds",
        ),
        pytest.param(
            [
                KeyPoints(xy=np.zeros((2, 0, 2), dtype=np.float32)),
                KeyPoints(xy=np.zeros((1, 2, 2), dtype=np.float32)),
            ],
            None,
            pytest.raises(ValueError, match="same number of keypoints"),
            id="zero-vs-nonzero-keypoints-mismatch",
        ),
    ],
)
def test_key_points_merge(
    key_points_list: list[KeyPoints],
    expected_result: KeyPoints | None,
    exception: Exception,
) -> None:
    """Test KeyPoints.merge field merging, mismatches, and zero-keypoint cases."""
    with exception:
        result = KeyPoints.merge(key_points_list=key_points_list)
        assert result == expected_result, f"Expected: {expected_result}, Got: {result}"


def test_key_points_iteration():
    """Test the iteration over KeyPoints objects."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        confidence=[[0.8, 0.9], [0.7, 0.6]],
        class_id=[0, 1],
    )

    iterations = 0
    for i, (xy, kp_confidence, class_id, data) in enumerate(key_points):
        iterations += 1
        assert xy.shape == (2, 2)
        assert kp_confidence.shape == (2,)
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
    for xy, kp_confidence, class_id, data in key_points_no_conf:
        assert kp_confidence is None


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


class TestKeyPointsConstructor:
    """Tests for the `KeyPoints()` constructor."""

    def test_constructor_normal_keypoint_confidence_path(self):
        """Normal keypoint_confidence= path works and emits no deprecation warning."""
        import warnings

        xy = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        kp_conf = np.array([[0.9, 0.8]], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kp = KeyPoints(xy=xy, keypoint_confidence=kp_conf)

        np.testing.assert_array_equal(kp.keypoint_confidence, kp_conf)
        assert kp.data == {}

    def test_constructor_data_none_defaults_to_empty_dict(self):
        """Explicit data=None normalizes to empty dict, not None."""
        xy = np.array([[[1.0, 2.0]]], dtype=np.float32)

        assert KeyPoints(xy=xy).data == {}
        assert KeyPoints(xy=xy, data=None).data == {}

    def test_keypoints_init_covers_all_dataclass_fields(self):
        """Custom __init__ must assign every dataclass field — guards against drift."""
        import dataclasses
        import inspect

        field_names = {f.name for f in dataclasses.fields(KeyPoints)}
        init_params = set(inspect.signature(KeyPoints.__init__).parameters) - {"self"}
        assert field_names == init_params, (
            f"Field/init drift: {field_names.symmetric_difference(init_params)}"
        )


class TestCovarianceSelection:
    """Keep each joint's uncertainty attached to its selected coordinates."""

    @pytest.mark.parametrize(
        ("keypoint_count", "index", "rows"),
        [
            pytest.param(3, [1], [1], id="object-selection"),
            pytest.param(2, (slice(None), [1, 0]), [0, 1], id="joint-permutation"),
            pytest.param(
                3,
                np.array([[True, False, True], [False, True, True]]),
                [0, 1],
                id="per-row-mask",
            ),
        ],
    )
    def test_preserves_per_object_covariance(
        self, keypoint_count: int, index: Any, rows: list[int]
    ) -> None:
        """Per-object covariance remains independent of joint selection."""
        covariance = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        key_points = KeyPoints(
            xy=np.zeros((2, keypoint_count, 2)),
            data={KEYPOINT_COVARIANCE_DATA_FIELD: covariance},
        )

        selected = key_points.select(index)

        np.testing.assert_array_equal(
            selected.data[KEYPOINT_COVARIANCE_DATA_FIELD], covariance[rows]
        )

    def test_preserves_ragged_per_object_covariance_list(self) -> None:
        """Ragged per-object list values remain valid metadata."""
        covariance = [[1, 2], [3]]
        key_points = KeyPoints(
            xy=np.zeros((2, 3, 2)), data={KEYPOINT_COVARIANCE_DATA_FIELD: covariance}
        )

        selected = key_points.select([1])

        assert selected.data[KEYPOINT_COVARIANCE_DATA_FIELD] == [[3]]

    @pytest.mark.parametrize(
        ("index", "rows", "joints"),
        [
            pytest.param((slice(None), [2, 0]), [0, 1], [[2, 0], [2, 0]], id="subset"),
            pytest.param(
                (slice(None), [2, 1, 0]),
                [0, 1],
                [[2, 1, 0], [2, 1, 0]],
                id="permutation",
            ),
            pytest.param((1, 2), [1], [[2]], id="scalar"),
            pytest.param((slice(None), []), [0, 1], [[], []], id="empty-joints"),
            pytest.param(
                np.array([[True, False, True], [False, True, True]]),
                [0, 1],
                [[0, 2], [1, 2]],
                id="per-row-mask",
            ),
        ],
    )
    def test_keeps_covariance_aligned(
        self, index: Any, rows: list[int], joints: list[list[int]]
    ) -> None:
        """Joint selection retains matching covariance matrices."""
        xy = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        covariance = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
        boxes = np.arange(8, dtype=np.float32).reshape(2, 4)
        key_points = KeyPoints(
            xy=xy, data={KEYPOINT_COVARIANCE_DATA_FIELD: covariance, "xyxy": boxes}
        )

        selected = key_points.select(index)

        row_index = np.array(rows)[:, None]
        joint_index = np.array(joints, dtype=int)
        np.testing.assert_array_equal(selected.xy, xy[row_index, joint_index])
        np.testing.assert_array_equal(
            selected.data[KEYPOINT_COVARIANCE_DATA_FIELD],
            covariance[row_index, joint_index],
        )
        np.testing.assert_array_equal(selected.data["xyxy"], boxes[rows])
        np.testing.assert_array_equal(
            key_points.data[KEYPOINT_COVARIANCE_DATA_FIELD], covariance
        )


class TestKeyPointsRemovedConfidence:
    """The `confidence` name removed in `0.32.0` cannot be silently re-created."""

    @staticmethod
    def _key_points() -> KeyPoints:
        """Build one two-keypoint skeleton with keypoint confidence of `0.9`."""
        return KeyPoints(
            xy=np.zeros((1, 2, 2), dtype=np.float32),
            keypoint_confidence=np.full((1, 2), 0.9, dtype=np.float32),
        )

    def test_assignment_raises_migration_error(self) -> None:
        """Assigning `confidence` fails, pointing at `keypoint_confidence`."""
        key_points = self._key_points()

        with pytest.raises(AttributeError, match="keypoint_confidence"):
            key_points.confidence = np.full((1, 2), 0.1, dtype=np.float32)

        assert key_points.keypoint_confidence is not None
        np.testing.assert_allclose(key_points.keypoint_confidence, 0.9)
        assert "confidence" not in vars(key_points)

    def test_read_stays_missing_attribute(self) -> None:
        """Reading `confidence` still reports a missing attribute."""
        key_points = self._key_points()

        has_confidence = hasattr(key_points, "confidence")

        assert has_confidence is False

    def test_keypoint_confidence_stays_assignable(self) -> None:
        """The guard leaves assignment of the replacement field untouched."""
        key_points = self._key_points()
        updated = np.full((1, 2), 0.1, dtype=np.float32)

        key_points.keypoint_confidence = updated

        assert key_points.keypoint_confidence is updated
