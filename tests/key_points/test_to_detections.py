"""Tests for `KeyPoints.as_detections` and `KeyPoints.with_nms`."""

from typing import Any

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.key_points.core import KeyPoints
from tests.helpers import (
    _create_key_points,
)


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


@pytest.mark.parametrize(
    "dtype",
    [pytest.param(np.float32, id="float32"), pytest.param(np.float64, id="float64")],
)
@pytest.mark.parametrize("depth", [1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("with_visibility", [False, True])
class TestKeyPointsImageSpaceBoxes:
    def test_conversion_ignores_extra_coordinate(
        self,
        dtype: type[np.float32] | type[np.float64],
        depth: float,
        with_visibility: bool,
    ) -> None:
        """Missing points and valid x/y bounds do not depend on depth."""
        xy = np.array([[[0, 0], [10, 10], [20, 20]]], dtype=dtype)
        extra = np.full((1, 3, 1), depth, dtype=dtype)
        extra[:, 0] = 1
        coordinates = np.concatenate((xy, extra), axis=2)
        visible = np.ones((1, 3), dtype=bool) if with_visibility else None
        key_points = KeyPoints(
            xy=coordinates,
            visible=visible,
            class_id=np.array([3]),
            data={"name": np.array(["person"])},
        )
        original = coordinates.copy()

        detections = key_points.as_detections(selected_keypoint_indices=[0, 1, 2])

        np.testing.assert_array_equal(detections.xyxy, [[10, 10, 20, 20]])
        np.testing.assert_array_equal(detections.class_id, [3])
        np.testing.assert_array_equal(detections.data["name"], ["person"])
        np.testing.assert_array_equal(coordinates, original)

    def test_suppression_ignores_extra_coordinate(
        self,
        dtype: type[np.float32] | type[np.float64],
        depth: float,
        with_visibility: bool,
    ) -> None:
        """NMS suppresses duplicate image boxes while retaining distant skeletons."""
        xy = np.array(
            [
                [[0, 0], [100, 100], [120, 120]],
                [[0, 0], [102, 102], [122, 122]],
                [[0, 0], [140, 140], [160, 160]],
            ],
            dtype=dtype,
        )
        extra = np.full((3, 3, 1), depth, dtype=dtype)
        extra[:, 0] = 1
        coordinates = np.concatenate((xy, extra), axis=2)
        visible = np.ones((3, 3), dtype=bool) if with_visibility else None
        key_points = KeyPoints(
            xy=coordinates,
            visible=visible,
            class_id=np.zeros(3, dtype=int),
            detection_confidence=np.array([0.9, 0.8, 0.7]),
            data={"name": np.array(["first", "duplicate", "distant"])},
        )
        original = coordinates.copy()

        result = key_points.with_nms(threshold=0.5)

        np.testing.assert_array_equal(result.xy, coordinates[[0, 2]])
        np.testing.assert_array_equal(result.data["name"], ["first", "distant"])
        np.testing.assert_array_equal(coordinates, original)


def test_key_points_as_detections_empty():
    """Test the as_detections method for empty KeyPoints objects."""
    empty_key_points = KeyPoints.empty()
    empty_detections = empty_key_points.as_detections()
    assert empty_detections.is_empty()


class TestKeyPointsAsDetectionsKeepsFields:
    """Inputs with nothing to box keep their fields, like all-invalid skeletons."""

    @pytest.mark.parametrize(
        ("count", "keypoint_count"),
        [
            pytest.param(0, 17, id="no-skeletons"),
            pytest.param(0, 0, id="no-skeletons-no-keypoints"),
            pytest.param(2, 0, id="skeletons-without-keypoints"),
        ],
    )
    def test_data_class_id_and_confidence_kept(
        self, count: int, keypoint_count: int
    ) -> None:
        """Data keys, class_id and confidence survive with zero rows and dtypes."""
        key_points = KeyPoints(
            xy=np.zeros((count, keypoint_count, 2), dtype=np.float32),
            class_id=np.arange(count, dtype=np.int64),
            detection_confidence=np.full(count, 0.9, dtype=np.float64),
            data={"track": np.arange(count, dtype=np.int32)},
        )

        detections = key_points.as_detections()

        assert len(detections) == 0
        assert detections.class_id is not None
        assert detections.class_id.dtype == np.int64
        assert detections.confidence is not None
        assert detections.confidence.dtype == np.float32
        assert list(detections.data) == ["track"]
        assert detections.data["track"].dtype == np.int32
        assert detections.data["track"].shape == (0,)

    def test_matches_all_invalid_skeletons(self) -> None:
        """No skeletons gives the same data keys and dtypes as all-[0, 0] skeletons."""
        fields: dict[str, Any] = {
            "class_id": np.array([3]),
            "keypoint_confidence": np.array([[0.4, 0.6]], dtype=np.float32),
            "data": {"track": np.array([7])},
        }
        all_invalid = KeyPoints(xy=np.zeros((1, 2, 2), dtype=np.float32), **fields)
        no_skeletons = all_invalid[np.array([], dtype=int)]

        expected = all_invalid.as_detections()
        result = no_skeletons.as_detections()

        assert len(expected) == 0
        assert len(result) == 0
        assert result.class_id is not None
        assert expected.class_id is not None
        assert result.class_id.dtype == expected.class_id.dtype
        assert result.confidence is not None
        assert expected.confidence is not None
        assert result.confidence.dtype == expected.confidence.dtype
        assert list(result.data) == list(expected.data)

    @pytest.mark.parametrize(
        ("count", "keypoint_count"),
        [
            pytest.param(0, 17, id="no-skeletons"),
            pytest.param(2, 0, id="skeletons-without-keypoints"),
        ],
    )
    def test_no_optional_fields_match_detections_empty(
        self, count: int, keypoint_count: int
    ) -> None:
        """Without optional fields, the result equals `Detections.empty()`."""
        key_points = KeyPoints(
            xy=np.zeros((count, keypoint_count, 2), dtype=np.float32)
        )

        detections = key_points.as_detections()

        assert detections == Detections.empty()

    def test_input_unchanged(self) -> None:
        """The result does not alias the input's data arrays."""
        track = np.arange(2, dtype=np.int32)
        key_points = KeyPoints(
            xy=np.zeros((2, 0, 2), dtype=np.float32), data={"track": track}
        )

        detections = key_points.as_detections()

        assert not np.shares_memory(detections.data["track"], track)
        np.testing.assert_array_equal(key_points.data["track"], [0, 1])

    @pytest.mark.parametrize(
        ("count", "indices"),
        [
            pytest.param(0, [17], id="no-skeletons-past-end"),
            pytest.param(0, [0, -18], id="no-skeletons-negative-past-start"),
            pytest.param(2, [17], id="skeletons-past-end"),
            pytest.param(2, [-18], id="skeletons-negative-past-start"),
        ],
    )
    def test_out_of_range_indices_raise(self, count: int, indices: list[int]) -> None:
        """Out-of-range indices raise with and without skeletons."""
        key_points = KeyPoints(xy=np.ones((count, 17, 2), dtype=np.float32))

        with pytest.raises(IndexError, match="out of range for 17 key points"):
            key_points.as_detections(selected_keypoint_indices=indices)

    @pytest.mark.parametrize("indices", [[0, 16], [-17, -1]])
    def test_in_range_indices_on_no_skeletons(self, indices: list[int]) -> None:
        """In-range indices on a frame without skeletons give zero rows."""
        key_points = KeyPoints(
            xy=np.zeros((0, 17, 2), dtype=np.float32),
            data={"row": np.array([], dtype=int)},
        )

        detections = key_points.as_detections(selected_keypoint_indices=indices)

        assert len(detections) == 0
        assert list(detections.data) == ["row"]

    def test_indices_on_empty_keypoint_axis_give_zero_rows(self) -> None:
        """Without a key point axis, any indices give zero rows instead of raising."""
        key_points = KeyPoints(
            xy=np.zeros((2, 0, 2), dtype=np.float32),
            data={"row": np.arange(2)},
        )

        detections = key_points.as_detections(selected_keypoint_indices=[5])

        assert len(detections) == 0
        assert list(detections.data) == ["row"]

    def test_list_data_kept_as_empty_list(self) -> None:
        """A list-valued data field is kept as a new empty list."""
        names = ["left", "right"]
        key_points = KeyPoints(
            xy=np.zeros((2, 0, 2), dtype=np.float32), data={"name": names}
        )

        detections = key_points.as_detections()

        assert detections.data["name"] == []
        assert detections.data["name"] is not names
        assert names == ["left", "right"]


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

    # Only the skeleton with at least one valid keypoint survives
    assert len(detections) == 1
    assert np.array_equal(detections.xyxy, np.array([[10, 20, 30, 40]]))


@pytest.mark.parametrize(
    ("selected_keypoint_indices", "expected_xyxy"),
    [
        pytest.param(None, [[10, 20, 30, 40]], id="all-keypoints"),
        pytest.param([1, 2], [[30, 40, 30, 40]], id="selected-keypoints"),
    ],
)
def test_key_points_as_detections_ignores_invisible_keypoints(
    selected_keypoint_indices: list[int] | None, expected_xyxy: list[list[int]]
) -> None:
    """Keypoints marked not visible stay out of the box, as they do in `with_nms`."""
    key_points = KeyPoints(
        xy=np.array(
            [[[10, 20], [30, 40], [300, 400]], [[50, 60], [70, 80], [90, 100]]],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
        visible=np.array([[True, True, False], [False, False, False]]),
    )

    detections = key_points.as_detections(
        selected_keypoint_indices=selected_keypoint_indices
    )

    np.testing.assert_array_equal(
        detections.xyxy, np.array(expected_xyxy, dtype=np.float32)
    )
    np.testing.assert_array_equal(detections.class_id, np.array([0]))


@pytest.mark.parametrize(
    "selected_keypoint_indices",
    [
        pytest.param(np.array([0, 1]), id="numpy-array"),
        pytest.param(iter([0, 1]), id="generator"),
        pytest.param((0, 1), id="tuple"),
    ],
)
def test_key_points_as_detections_index_container_types(selected_keypoint_indices):
    """selected_keypoint_indices accepts any Iterable[int] container type."""
    key_points = _create_key_points(
        xy=[[[10, 10], [20, 20], [30, 15]]],
        class_id=[0],
    )

    detections = key_points.as_detections(
        selected_keypoint_indices=selected_keypoint_indices,
    )

    assert np.array_equal(detections.xyxy, np.array([[10, 10, 20, 20]]))


def test_key_points_as_detections_empty_indices_selects_all():
    """An empty selected_keypoint_indices behaves like None (selects all)."""
    key_points = _create_key_points(
        xy=[[[10, 10], [20, 20], [30, 15]]],
        confidence=[[0.1, 0.3, 0.5]],
        class_id=[0],
        detection_confidence=[0.9],
    )
    key_points["custom_data"] = ["person"]

    all_selected = key_points.as_detections()
    empty_list_selected = key_points.as_detections(selected_keypoint_indices=[])

    assert np.array_equal(all_selected.xyxy, empty_list_selected.xyxy)
    assert np.array_equal(all_selected.confidence, empty_list_selected.confidence)
    assert np.array_equal(all_selected.class_id, empty_list_selected.class_id)
    assert np.array_equal(
        all_selected.data["custom_data"], empty_list_selected.data["custom_data"]
    )


@pytest.mark.parametrize(
    ("xy", "expected_xyxy"),
    [
        pytest.param(
            [[[10, 20], [30, 20]]],
            np.array([[10, 20, 30, 20]]),
            id="collinear-keypoints",
        ),
        pytest.param(
            [[[15, 25]]],
            np.array([[15, 25, 15, 25]]),
            id="single-keypoint",
        ),
    ],
)
def test_key_points_as_detections_keeps_degenerate_skeletons(xy, expected_xyxy):
    """A skeleton with valid keypoints keeps its box even when the area is zero."""
    key_points = _create_key_points(xy=xy, class_id=[0])

    detections = key_points.as_detections()

    assert len(detections) == 1
    assert np.array_equal(detections.xyxy, expected_xyxy)


def test_key_points_as_detections_filters_invalid_and_aligns_degenerate_metadata():
    """Invalid skeletons are removed while valid degenerate metadata stays aligned."""
    key_points = _create_key_points(
        xy=[
            [[0, 0], [0, 0]],
            [[np.nan, 5], [0, 0]],
            [[np.inf, 5], [0, 0]],
            [[15, 25], [np.nan, 50]],
            [[10, 20], [30, 20]],
        ],
        class_id=[0, 1, 2, 3, 4],
        detection_confidence=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    key_points["custom_data"] = ["zero", "nan", "inf", "point", "line"]

    detections = key_points.as_detections()

    assert np.array_equal(
        detections.xyxy,
        np.array([[15, 25, 15, 25], [10, 20, 30, 20]], dtype=np.float32),
    )
    assert np.array_equal(detections.class_id, np.array([3, 4]))
    assert np.array_equal(detections.confidence, np.array([0.4, 0.5], dtype=np.float32))
    assert np.array_equal(detections.data["custom_data"], np.array(["point", "line"]))


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


def test_key_points_merge_then_with_nms_deduplicates_overlapping_detections():
    """Test merge-then-NMS removes duplicated overlapping skeletons."""
    key_points_list = [
        _create_key_points(
            xy=[[[100, 100], [200, 200]], [[400, 400], [500, 500]]],
            detection_confidence=[0.9, 0.6],
            class_id=[0, 0],
        ),
        _create_key_points(
            xy=[[[100, 100], [200, 200]], [[700, 700], [800, 800]]],
            detection_confidence=[0.85, 0.7],
            class_id=[0, 0],
        ),
    ]

    merged = KeyPoints.merge(key_points_list)
    result = merged.with_nms(threshold=0.5, class_agnostic=False)

    assert len(merged) == 4
    assert len(result) == 3
    assert len(result) < len(merged)
    np.testing.assert_allclose(
        np.sort(result.detection_confidence),
        np.array([0.6, 0.7, 0.9], dtype=np.float32),
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
                    [[100, 100], [150, 150], [200, 200]],
                    [[np.nan, np.nan], [110, 110], [210, 210]],
                ],
                detection_confidence=[0.9, 0.7],
                class_id=[0, 0],
            ),
            0.3,
            False,
            _create_key_points(
                xy=[[[100, 100], [150, 150], [200, 200]]],
                detection_confidence=[0.9],
                class_id=[0],
            ),
            id="non-finite-keypoints-excluded-from-bbox",
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


def test_with_nms_keeps_skeleton_without_any_finite_keypoint():
    """A skeleton with no usable keypoint survives NMS rather than being dropped."""
    key_points = _create_key_points(
        xy=[
            [[100, 100], [200, 200]],
            [[np.nan, np.nan], [np.inf, np.inf]],
        ],
        detection_confidence=[0.9, 0.7],
        class_id=[0, 0],
    )

    result = key_points.with_nms(threshold=0.3)

    assert np.allclose(result.detection_confidence, [0.9, 0.7])


class TestKeyPointsWithNmsEmptyJoints:
    """NMS preserves skeletons with an empty keypoint axis."""

    @pytest.mark.parametrize("skeleton_count", [1, 3])
    @pytest.mark.parametrize("class_agnostic", [False, True])
    def test_keeps_skeletons_and_aligned_fields(
        self, skeleton_count: int, class_agnostic: bool
    ) -> None:
        """Filtering out every joint preserves each skeleton and its aligned fields."""
        key_points = KeyPoints(
            xy=np.tile(
                np.array([[[10, 10], [20, 20]]], dtype=np.float32),
                (skeleton_count, 1, 1),
            ),
            keypoint_confidence=np.full((skeleton_count, 2), 0.8, dtype=np.float32),
            detection_confidence=np.array([0.3, 0.9, 0.7], dtype=np.float32)[
                :skeleton_count
            ],
            visible=np.ones((skeleton_count, 2), dtype=bool),
            class_id=np.arange(skeleton_count) % 2,
            data={"custom_data": np.arange(skeleton_count)},
        )
        filtered = key_points[np.zeros((skeleton_count, 2), dtype=bool)]

        result = filtered.with_nms(threshold=0.0, class_agnostic=class_agnostic)

        assert result.xy.shape == (skeleton_count, 0, 2)
        assert result.xy.dtype == key_points.xy.dtype
        assert result == filtered
        assert result is not filtered

    @pytest.mark.parametrize("skeleton_count", [1, 3])
    def test_keeps_skeletons_without_optional_fields(self, skeleton_count: int) -> None:
        """Class-agnostic NMS needs only detection confidence when joints are absent."""
        key_points = KeyPoints(
            xy=np.empty((skeleton_count, 0, 2), dtype=np.float32),
            detection_confidence=np.full(skeleton_count, 0.9, dtype=np.float32),
        )

        result = key_points.with_nms(class_agnostic=True)

        assert result == key_points

    @pytest.mark.parametrize(
        ("missing_field", "class_agnostic"),
        [
            pytest.param("detection_confidence", False, id="class-aware-no-confidence"),
            pytest.param(
                "detection_confidence", True, id="class-agnostic-no-confidence"
            ),
            pytest.param("class_id", False, id="class-aware-no-class-id"),
        ],
    )
    def test_requires_nms_fields(
        self, missing_field: str, class_agnostic: bool
    ) -> None:
        """An empty joint axis does not bypass required NMS fields."""
        key_points = KeyPoints(
            xy=np.empty((1, 0, 2), dtype=np.float32),
            detection_confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        setattr(key_points, missing_field, None)

        with pytest.raises(ValueError, match=missing_field):
            key_points.with_nms(class_agnostic=class_agnostic)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_rejects_invalid_threshold(self, threshold: float) -> None:
        """Empty joints still require an NMS threshold between zero and one."""
        key_points = KeyPoints(
            xy=np.empty((1, 0, 2), dtype=np.float32),
            detection_confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        with pytest.raises(ValueError, match="iou_threshold"):
            key_points.with_nms(threshold=threshold)


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
