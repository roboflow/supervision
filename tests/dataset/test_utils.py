from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise
from typing import TypeVar

import pytest

from supervision import Detections
from supervision.dataset.utils import (
    build_class_index_mapping,
    map_detections_class_id,
    merge_class_lists,
    train_test_split,
)
from tests.helpers import _create_detections

T = TypeVar("T")


@pytest.mark.parametrize(
    ("data", "train_ratio", "random_state", "shuffle", "expected_result", "exception"),
    [
        ([], 0.5, None, False, ([], []), DoesNotRaise()),  # empty data
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            None,
            False,
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 50% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            1.0,
            None,
            False,
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], []),
            DoesNotRaise(),
        ),  # data with 10 numbers and 100% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.0,
            None,
            False,
            ([], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 0% train split
        (
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            0.5,
            None,
            False,
            (["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]),
            DoesNotRaise(),
        ),  # data with 10 chars and 50% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            23,
            True,
            ([7, 8, 5, 6, 3], [2, 9, 0, 1, 4]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 50% train split with 23 random seed
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            32,
            True,
            ([4, 6, 0, 8, 9], [5, 7, 2, 3, 1]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 50% train split with 23 random seed
    ],
)
def test_train_test_split(
    data: list[T],
    train_ratio: float,
    random_state: int,
    shuffle: bool,
    expected_result: tuple[list[T], list[T]] | None,
    exception: Exception,
) -> None:
    with exception:
        result = train_test_split(
            data=data,
            train_ratio=train_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )
        assert result == expected_result


@pytest.mark.parametrize(
    ("class_lists", "expected_result", "exception"),
    [
        ([], [], DoesNotRaise()),  # empty class lists
        (
            [["dog", "person"]],
            ["dog", "person"],
            DoesNotRaise(),
        ),  # single class list; already alphabetically sorted
        (
            [["person", "dog"]],
            ["dog", "person"],
            DoesNotRaise(),
        ),  # single class list; not alphabetically sorted
        (
            [["dog", "person"], ["dog", "person"]],
            ["dog", "person"],
            DoesNotRaise(),
        ),  # two class lists; the same classes; already alphabetically sorted
        (
            [["dog", "person"], ["cat"]],
            ["cat", "dog", "person"],
            DoesNotRaise(),
        ),  # two class lists; different classes; already alphabetically sorted
    ],
)
def test_merge_class_maps(
    class_lists: list[list[str]], expected_result: list[str], exception: Exception
) -> None:
    with exception:
        result = merge_class_lists(class_lists=class_lists)
        assert result == expected_result


@pytest.mark.parametrize(
    ("source_classes", "target_classes", "expected_result", "exception"),
    [
        ([], [], {}, DoesNotRaise()),  # empty class lists
        ([], ["dog", "person"], {}, DoesNotRaise()),  # empty source class list
        (
            ["dog", "person"],
            [],
            None,
            pytest.raises(ValueError, match="Class dog not found"),
        ),  # empty target class list
        (
            ["dog", "person"],
            ["dog", "person"],
            {0: 0, 1: 1},
            DoesNotRaise(),
        ),  # same class lists
        (
            ["dog", "person"],
            ["person", "dog"],
            {0: 1, 1: 0},
            DoesNotRaise(),
        ),  # same class lists but not alphabetically sorted
        (
            ["dog", "person"],
            ["cat", "dog", "person"],
            {0: 1, 1: 2},
            DoesNotRaise(),
        ),  # source class list is a subset of target class list
        (
            ["dog", "person"],
            ["cat", "dog"],
            None,
            pytest.raises(ValueError, match="Class person not found"),
        ),  # source class list is not a subset of target class list
    ],
)
def test_build_class_index_mapping(
    source_classes: list[str],
    target_classes: list[str],
    expected_result: dict[int, int] | None,
    exception: Exception,
) -> None:
    with exception:
        result = build_class_index_mapping(
            source_classes=source_classes, target_classes=target_classes
        )
        assert result == expected_result


@pytest.mark.parametrize(
    ("source_to_target_mapping", "detections", "expected_result", "exception"),
    [
        (
            {},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            None,
            pytest.raises(ValueError, match="subset of source_to_target_mapping"),
        ),  # empty mapping
        (
            {0: 1},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
            DoesNotRaise(),
        ),  # single mapping
        (
            {0: 1, 1: 2},
            Detections.empty(),
            Detections.empty(),
            DoesNotRaise(),
        ),  # empty detections
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
            DoesNotRaise(),
        ),  # multiple mappings
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[0, 1]),
            _create_detections(xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[1, 2]),
            DoesNotRaise(),
        ),  # multiple mappings
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[2]),
            None,
            pytest.raises(ValueError, match="source_to_target_mapping keys"),
        ),  # class_id not in mapping
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]),
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[1], confidence=[0.5]),
            DoesNotRaise(),
        ),  # confidence is not None
    ],
)
def test_map_detections_class_id(
    source_to_target_mapping: dict[int, int],
    detections: Detections,
    expected_result: Detections | None,
    exception: Exception,
) -> None:
    with exception:
        result = map_detections_class_id(
            source_to_target_mapping=source_to_target_mapping, detections=detections
        )
        assert result == expected_result
