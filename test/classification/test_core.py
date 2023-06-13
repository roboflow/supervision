from typing import List, TypeVar, Optional, Tuple
from contextlib import ExitStack as DoesNotRaise

import pytest
import numpy as np

from supervision.classification.core import Classifications

T = TypeVar("T")


@pytest.mark.parametrize(
    'class_id, confidence, expected_result, exception',
    [
        (
            [0, 1, 2, 3, 4],
            [0.1, 0.2, 0.9, 0.4, 0.5],
            (np.array([2, 4, 3, 1, 0]), np.array([0.9, 0.5, 0.4, 0.2, 0.1])),
            DoesNotRaise()
        ),  # class_id with 5 numbers and 5 confidences
        (
            [0, 1, 2, 3, 4],
            [0.1, 0.2, 0.3, 0.4],
            None,
            pytest.raises(ValueError)
        ),  # class_id with 5 numbers and 4 confidences
    ]
)
def test_top_k(
    class_id: List[T],
    confidence: Optional[List[T]],
    expected_result: Optional[Tuple[List[T], List[T]]],
    exception: Exception
) -> None:
    with exception:
        result = Classifications(class_id=np.array(class_id), confidence=np.array(confidence)).get_top_k(len(class_id))

        assert result[0].tolist() == expected_result[0].tolist()
        assert result[1].tolist() == expected_result[1].tolist()
