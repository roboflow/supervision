from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise
from typing import Any

import numpy as np
import pytest

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.utils.internal import (
    get_data_item,
    merge_data,
    merge_metadata,
    process_roboflow_result,
)

TEST_MASK = np.zeros((1, 1000, 1000), dtype=bool)
TEST_MASK[:, 300:351, 200:251] = True


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
    expected_result: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
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
                assert np.array_equal(result[5][key], expected_result[5][key]), (
                    f"Mismatch in arrays for key {key}"
                )
            else:
                assert result[5][key] == expected_result[5][key], (
                    f"Mismatch in non-array data for key {key}"
                )


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
    data_list: list[dict[str, Any]],
    expected_result: dict[str, Any] | None,
    exception: Exception,
):
    with exception:
        result = merge_data(data_list=data_list)
        if expected_result is None:
            assert False, f"Expected an error, but got result {result}"

        for key in result:
            if isinstance(result[key], np.ndarray):
                assert np.array_equal(result[key], expected_result[key]), (
                    f"Mismatch in arrays for key {key}"
                )
            else:
                assert result[key] == expected_result[key], (
                    f"Mismatch in non-array data for key {key}"
                )


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
    data: dict[str, Any],
    index: Any,
    expected_result: dict[str, Any] | None,
    exception: Exception,
):
    with exception:
        result = get_data_item(data=data, index=index)
        for key in result:
            if isinstance(result[key], np.ndarray):
                assert np.array_equal(result[key], expected_result[key]), (
                    f"Mismatch in arrays for key {key}"
                )
            else:
                assert result[key] == expected_result[key], (
                    f"Mismatch in non-array data for key {key}"
                )


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
