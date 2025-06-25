from contextlib import ExitStack as DoesNotRaise
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from supervision import Detections
from supervision.dataset.utils import (
    build_class_index_mapping,
    map_detections_class_id,
    mask_to_rle,
    merge_class_lists,
    rle_to_mask,
    train_test_split,
)
from test.test_utils import mock_detections

T = TypeVar("T")


@pytest.mark.parametrize(
    "data, train_ratio, random_state, shuffle, expected_result, exception",
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
    data: List[T],
    train_ratio: float,
    random_state: int,
    shuffle: bool,
    expected_result: Optional[Tuple[List[T], List[T]]],
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
    "class_lists, expected_result, exception",
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
    class_lists: List[List[str]], expected_result: List[str], exception: Exception
) -> None:
    with exception:
        result = merge_class_lists(class_lists=class_lists)
        assert result == expected_result


@pytest.mark.parametrize(
    "source_classes, target_classes, expected_result, exception",
    [
        ([], [], {}, DoesNotRaise()),  # empty class lists
        ([], ["dog", "person"], {}, DoesNotRaise()),  # empty source class list
        (
            ["dog", "person"],
            [],
            None,
            pytest.raises(ValueError),
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
            pytest.raises(ValueError),
        ),  # source class list is not a subset of target class list
    ],
)
def test_build_class_index_mapping(
    source_classes: List[str],
    target_classes: List[str],
    expected_result: Optional[Dict[int, int]],
    exception: Exception,
) -> None:
    with exception:
        result = build_class_index_mapping(
            source_classes=source_classes, target_classes=target_classes
        )
        assert result == expected_result


@pytest.mark.parametrize(
    "source_to_target_mapping, detections, expected_result, exception",
    [
        (
            {},
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            None,
            pytest.raises(ValueError),
        ),  # empty mapping
        (
            {0: 1},
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
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
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
            DoesNotRaise(),
        ),  # multiple mappings
        (
            {0: 1, 1: 2},
            mock_detections(xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[0, 1]),
            mock_detections(xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[1, 2]),
            DoesNotRaise(),
        ),  # multiple mappings
        (
            {0: 1, 1: 2},
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[2]),
            None,
            pytest.raises(ValueError),
        ),  # class_id not in mapping
        (
            {0: 1, 1: 2},
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]),
            mock_detections(xyxy=[[0, 0, 10, 10]], class_id=[1], confidence=[0.5]),
            DoesNotRaise(),
        ),  # confidence is not None
    ],
)
def test_map_detections_class_id(
    source_to_target_mapping: Dict[int, int],
    detections: Detections,
    expected_result: Optional[Detections],
    exception: Exception,
) -> None:
    with exception:
        result = map_detections_class_id(
            source_to_target_mapping=source_to_target_mapping, detections=detections
        )
        assert result == expected_result


@pytest.mark.parametrize(
    "mask, expected_rle, exception",
    [
        (
            np.zeros((3, 3)).astype(bool),
            [9],
            DoesNotRaise(),
        ),  # mask with background only (mask with only False values)
        (
            np.ones((3, 3)).astype(bool),
            [0, 9],
            DoesNotRaise(),
        ),  # mask with foreground only (mask with only True values)
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            [6, 3, 2, 1, 1, 1, 2, 3, 6],
            DoesNotRaise(),
        ),  # mask where foreground object has hole
        (
            np.array(
                [
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                ]
            ).astype(bool),
            [0, 5, 5, 5, 5, 5],
            DoesNotRaise(),
        ),  # mask where foreground consists of 3 separate components
        (
            np.array([[[]]]).astype(bool),
            None,
            pytest.raises(AssertionError),
        ),  # raises AssertionError because mask dimensionality is not 2D
        (
            np.array([[]]).astype(bool),
            None,
            pytest.raises(AssertionError),
        ),  # raises AssertionError because mask is empty
    ],
)
def test_mask_to_rle(
    mask: npt.NDArray[np.bool_], expected_rle: List[int], exception: Exception
) -> None:
    with exception:
        result = mask_to_rle(mask=mask)
        assert result == expected_rle


@pytest.mark.parametrize(
    "rle, resolution_wh, expected_mask, exception",
    [
        (
            np.array([9]),
            [3, 3],
            np.zeros((3, 3)).astype(bool),
            DoesNotRaise(),
        ),  # mask with background only (mask with only False values); rle as array
        (
            [9],
            [3, 3],
            np.zeros((3, 3)).astype(bool),
            DoesNotRaise(),
        ),  # mask with background only (mask with only False values); rle as list
        (
            np.array([0, 9]),
            [3, 3],
            np.ones((3, 3)).astype(bool),
            DoesNotRaise(),
        ),  # mask with foreground only (mask with only True values)
        (
            np.array([6, 3, 2, 1, 1, 1, 2, 3, 6]),
            [5, 5],
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            DoesNotRaise(),
        ),  # mask where foreground object has hole
        (
            np.array([0, 5, 5, 5, 5, 5]),
            [5, 5],
            np.array(
                [
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                ]
            ).astype(bool),
            DoesNotRaise(),
        ),  # mask where foreground consists of 3 separate components
        (
            np.array([0, 5, 5, 5, 5, 5]),
            [2, 2],
            None,
            pytest.raises(AssertionError),
        ),  # raises AssertionError because number of pixels in RLE does not match
        # number of pixels in expected mask (width x height).
    ],
)
def test_rle_to_mask(
    rle: npt.NDArray[np.int_],
    resolution_wh: Tuple[int, int],
    expected_mask: npt.NDArray[np.bool_],
    exception: Exception,
) -> None:
    with exception:
        result = rle_to_mask(rle=rle, resolution_wh=resolution_wh)
        assert np.all(result == expected_mask)
