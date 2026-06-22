from contextlib import nullcontext as DoesNotRaise

import numpy as np
import pytest

from supervision.key_points.core import KeyPoints
from tests.helpers import (
    _create_key_points,
    _FakeMediapipeLandmark,
    _FakeMediapipePose,
    _FakeMediapipeResults,
    _FakeYoloNasKeyPoint,
    _FakeYoloNasKeyPointResults,
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
        ),  # kp[kp.confidence > 0.5] — single-object canonical use case
    ],
)
def test_key_points_getitem(key_points, index, expected_result, exception):
    with exception:
        result = key_points[index]
        assert result == expected_result


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


def test_key_points_as_detections_ignores_missing_keypoints():
    """A [0, 0] keypoint is treated as missing and excluded from the box."""
    key_points = _create_key_points(
        xy=[[[0, 0], [10, 20], [30, 40]]],
        confidence=[[0.0, 0.8, 0.6]],
        class_id=[0],
    )

    detections = key_points.as_detections()

    assert np.array_equal(detections.xyxy, np.array([[10, 20, 30, 40]]))


def test_key_points_as_detections_uses_detection_confidence():
    """detection_confidence is preferred over the keypoint-confidence mean."""
    key_points = _create_key_points(
        xy=[[[10, 20], [30, 40]]],
        confidence=[[0.1, 0.2]],
        class_id=[0],
        detection_confidence=[0.95],
    )

    detections = key_points.as_detections()

    assert np.allclose(detections.confidence, np.array([0.95], dtype=np.float32))


def test_key_points_as_detections_selected_keypoint_indices():
    """Only the selected keypoints contribute to the bounding box."""
    key_points = _create_key_points(
        xy=[[[0, 0], [10, 20], [30, 40], [100, 100]]],
        confidence=[[0.5, 0.8, 0.6, 0.9]],
        class_id=[0],
    )

    detections = key_points.as_detections(selected_keypoint_indices=[1, 2])

    assert np.array_equal(detections.xyxy, np.array([[10, 20, 30, 40]]))


def test_key_points_as_detections_confidence_over_selected_indices():
    """Confidence mean uses only the selected keypoint columns, not all."""
    key_points = _create_key_points(
        xy=[[[0, 0], [10, 20], [30, 40], [100, 100]]],
        confidence=[[0.5, 0.8, 0.6, 0.9]],
        class_id=[0],
    )

    detections = key_points.as_detections(selected_keypoint_indices=[1, 2])

    expected_confidence = np.mean([0.8, 0.6], dtype=np.float32)
    assert np.isclose(detections.confidence[0], expected_confidence)


def test_key_points_as_detections_mixed_valid_invalid_batch():
    """Batch with one all-zero skeleton: invalid skeleton gets box zeroed."""
    key_points = _create_key_points(
        xy=[[[0, 0], [0, 0]], [[10, 20], [30, 40]]],
        confidence=[[0.0, 0.0], [0.8, 0.6]],
        class_id=[0, 1],
    )

    detections = key_points.as_detections()

    # Only the valid skeleton survives the area>0 filter
    assert len(detections) == 1
    assert np.array_equal(detections.xyxy, np.array([[10, 20, 30, 40]]))


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


@pytest.mark.parametrize(
    ("inference_results", "expected_key_points"),
    [
        (
            {
                "predictions": [
                    {
                        "class_id": 1,
                        "class": "person",
                        "keypoints": [
                            {"x": 100, "y": 150, "confidence": 0.9},
                            {"x": 120, "y": 160, "confidence": 0.85},
                        ],
                    }
                ]
            },
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=[1],
                data={"class_name": np.array(["person"])},
            ),
        ),
        ({"predictions": []}, KeyPoints.empty()),
    ],
)
def test_from_inference_input(inference_results, expected_key_points):
    """Test the from_inference method with valid input."""
    key_points = KeyPoints.from_inference(inference_results)
    assert key_points == expected_key_points


def test_from_inference_invalid_input():
    """Test the from_inference method with invalid input."""
    key_points = _create_key_points(
        xy=[[[0, 1], [2, 3]]], confidence=[[0.8, 0.9]], class_id=[0]
    )
    with pytest.raises(
        ValueError, match=r"from_inference\(\) operates on a single result at a time.*"
    ):
        KeyPoints.from_inference([key_points])


@pytest.mark.parametrize(
    ("yolo_nas_results", "expected_key_points"),
    [
        (
            _FakeYoloNasKeyPointResults(
                _FakeYoloNasKeyPoint(
                    poses=[[[100.0, 150.0, 0.9], [120.0, 160.0, 0.85]]],
                    labels=[1],
                ),
            ),
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=[1],
            ),
        ),
        (
            _FakeYoloNasKeyPointResults(
                _FakeYoloNasKeyPoint(
                    poses=[],
                ),
            ),
            KeyPoints.empty(),
        ),
    ],
)
def test_from_yolo_nas_input(yolo_nas_results, expected_key_points):
    """Test the from_yolo_nas method with valid input."""
    key_points = KeyPoints.from_yolo_nas(yolo_nas_results)
    assert key_points == expected_key_points


@pytest.mark.parametrize(
    ("mediapipe_results", "resolution_wh", "expected_key_points"),
    [
        (
            _FakeMediapipeResults(
                pose_landmarks=_FakeMediapipePose(
                    landmarks=[
                        _FakeMediapipeLandmark(0.5, 0.75, 0.9),
                        _FakeMediapipeLandmark(0.6, 0.8, 0.85),
                    ]
                )
            ),
            (200, 200),
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=None,
            ),
        ),
        (
            _FakeMediapipeResults(
                pose_landmarks=[
                    [
                        _FakeMediapipeLandmark(0.5, 0.75, 0.9),
                        _FakeMediapipeLandmark(0.6, 0.8, 0.85),
                    ]
                ]
            ),
            (200, 200),
            _create_key_points(
                xy=[[[100.0, 150.0], [120.0, 160.0]]],
                confidence=[[0.9, 0.85]],
                class_id=None,
            ),
        ),
    ],
)
def test_from_mediapipe_input(mediapipe_results, resolution_wh, expected_key_points):
    """Test the from_mediapipe method with valid input."""
    key_points = KeyPoints.from_mediapipe(
        mediapipe_results, resolution_wh=resolution_wh
    )
    assert key_points == expected_key_points


class TestDeprecatedConfidenceConstructor:
    """Tests for backward-compatible `confidence=` kwarg in KeyPoints()."""

    def test_constructor_accepts_and_warns_on_deprecated_confidence_kwarg(self):
        """Deprecated confidence= warns and maps value to keypoint_confidence."""
        from supervision.utils.internal import SupervisionWarnings

        xy = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        confidence = np.array([[0.9, 0.8]], dtype=np.float32)

        with pytest.warns(SupervisionWarnings, match="deprecated since"):
            key_points = KeyPoints(xy=xy, confidence=confidence)

        np.testing.assert_array_equal(key_points.keypoint_confidence, confidence)
        assert key_points.xy is xy

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {"confidence": None, "keypoint_confidence": None},
                id="confidence-first",
            ),
            pytest.param(
                {"keypoint_confidence": None, "confidence": None},
                id="keypoint-confidence-first",
            ),
        ],
    )
    def test_constructor_rejects_both_confidence_and_keypoint_confidence(
        self, kwargs: dict
    ):
        """ValueError raised regardless of kwarg order when both are passed."""
        xy = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        confidence_arr = np.array([[0.9, 0.8]], dtype=np.float32)
        actual_kwargs = {k: confidence_arr for k in kwargs}

        with pytest.raises(ValueError, match="Cannot pass both"):
            KeyPoints(xy=xy, **actual_kwargs)

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

    def test_constructor_confidence_none_does_not_warn(self):
        """Explicit confidence=None is silently ignored — no warning emitted."""
        import warnings

        xy = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kp = KeyPoints(xy=xy, confidence=None)

        assert kp.keypoint_confidence is None

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
        init_params = set(inspect.signature(KeyPoints.__init__).parameters) - {
            "self",
            "confidence",
        }
        assert field_names == init_params, (
            f"Field/init drift: {field_names.symmetric_difference(init_params)}"
        )


@pytest.mark.parametrize(
    ("key_points", "threshold", "class_agnostic", "expected_result"),
    [
        pytest.param(
            KeyPoints.empty(),
            0.5,
            True,
            KeyPoints.empty(),
            id="empty",
        ),
        pytest.param(
            _create_key_points(
                xy=[[[10, 20], [30, 40]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            0.5,
            False,
            _create_key_points(
                xy=[[[10, 20], [30, 40]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="single-skeleton",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[10, 10], [20, 20]],
                    [[500, 500], [600, 600]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            0.5,
            False,
            _create_key_points(
                xy=[
                    [[10, 10], [20, 20]],
                    [[500, 500], [600, 600]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            id="two-non-overlapping",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[150, 150], [250, 250]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            0.9,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[150, 150], [250, 250]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            id="two-overlapping-below-threshold",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[[[100, 100], [200, 200]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="two-overlapping-above-threshold",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                    [[500, 500], [600, 600]],
                ],
                detection_confidence=[0.9, 0.7, 0.8],
                class_id=[0, 0, 0],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[500, 500], [600, 600]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            id="three-skeletons-two-overlap",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 1],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 1],
            ),
            id="class-aware-different-classes-kept",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 1],
            ),
            0.3,
            True,
            _create_key_points(
                xy=[[[100, 100], [200, 200]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="class-agnostic-suppresses-across-classes",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[0, 0], [100, 100], [200, 200]],
                    [[0, 0], [110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[[[0, 0], [100, 100], [200, 200]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="zero-keypoints-excluded-from-bbox",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200], [0, 0], [0, 0]],
                    [[0, 0], [0, 0], [110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 1],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200], [0, 0], [0, 0]],
                    [[0, 0], [0, 0], [110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 1],
            ),
            id="multi-skeleton-schema-class-aware",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200], [0, 0], [0, 0]],
                    [[0, 0], [0, 0], [110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 1],
            ),
            0.3,
            True,
            _create_key_points(
                xy=[[[100, 100], [200, 200], [0, 0], [0, 0]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="multi-skeleton-schema-class-agnostic",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[0, 0], [0, 0]],
                    [[100, 100], [200, 200]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            0.5,
            False,
            _create_key_points(
                xy=[
                    [[0, 0], [0, 0]],
                    [[100, 100], [200, 200]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            id="all-zero-skeleton-passes-through",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
                visible=[[True, False], [False, True]],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
                visible=[[True, False], [False, True]],
            ),
            id="visible-mask-excludes-keypoints-from-bbox",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [0, 0]],
                    [[300, 300], [0, 0]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [0, 0]],
                    [[300, 300], [0, 0]],
                ],
                detection_confidence=[0.9, 0.8],
                class_id=[0, 0],
            ),
            id="single-valid-keypoint-zero-area-bbox",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            0.0,
            False,
            _create_key_points(
                xy=[[[100, 100], [200, 200]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="threshold-zero-suppresses-any-overlap",
        ),
        pytest.param(
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            1.0,
            False,
            _create_key_points(
                xy=[
                    [[100, 100], [200, 200]],
                    [[110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            id="threshold-one-keeps-all",
        ),
    ],
)
def test_with_nms(key_points, threshold, class_agnostic, expected_result):
    """NMS filters overlapping keypoint skeletons."""
    result = key_points.with_nms(threshold=threshold, class_agnostic=class_agnostic)
    assert result == expected_result


@pytest.mark.parametrize(
    ("key_points", "threshold", "class_agnostic", "match"),
    [
        pytest.param(
            _create_key_points(
                xy=[[[10, 20], [30, 40]]],
                class_id=[0],
            ),
            0.5,
            False,
            "detection_confidence",
            id="no-detection-confidence",
        ),
        pytest.param(
            _create_key_points(
                xy=[[[10, 20], [30, 40]]],
                detection_confidence=[0.9],
            ),
            0.5,
            False,
            "class_id",
            id="no-class-id-class-aware",
        ),
        pytest.param(
            _create_key_points(
                xy=[[[10, 20], [30, 40]]],
                class_id=[0],
            ),
            0.5,
            True,
            "detection_confidence",
            id="no-detection-confidence-class-agnostic",
        ),
    ],
)
def test_with_nms_raises(key_points, threshold, class_agnostic, match):
    """NMS raises when required fields are missing."""
    with pytest.raises(ValueError, match=match):
        key_points.with_nms(threshold=threshold, class_agnostic=class_agnostic)
