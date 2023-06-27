from typing import List, TypeVar, Optional, Tuple
from contextlib import ExitStack as DoesNotRaise

import pytest

from supervision.dataset.utils import train_test_split

T = TypeVar("T")


@pytest.mark.parametrize(
    'data, train_ratio, random_state, shuffle, expected_result, exception',
    [
        (
            [],
            0.5,
            None,
            False,
            ([], []),
            DoesNotRaise()
        ),  # empty data
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            None,
            False,
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
            DoesNotRaise()
        ),  # data with 10 numbers and 50% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            1.0,
            None,
            False,
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], []),
            DoesNotRaise()
        ),  # data with 10 numbers and 100% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.0,
            None,
            False,
            ([], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            DoesNotRaise()
        ),  # data with 10 numbers and 0% train split
        (
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            0.5,
            None,
            False,
            (['a', 'b', 'c', 'd', 'e'], ['f', 'g', 'h', 'i', 'j']),
            DoesNotRaise()
        ),  # data with 10 chars and 50% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            23,
            True,
            ([7, 8, 5, 6, 3], [2, 9, 0, 1, 4]),
            DoesNotRaise()
        ),  # data with 10 numbers and 50% train split with 23 random seed
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            32,
            True,
            ([4, 6, 0, 8, 9], [5, 7, 2, 3, 1]),
            DoesNotRaise()
        ),  # data with 10 numbers and 50% train split with 23 random seed
    ]
)
def test_train_test_split(
    data: List[T],
    train_ratio: float,
    random_state: int,
    shuffle: bool,
    expected_result: Optional[Tuple[List[T], List[T]]],
    exception: Exception
) -> None:
    with exception:
        result = train_test_split(data=data, train_ratio=train_ratio, random_state=random_state, shuffle=shuffle)
        assert result == expected_result
