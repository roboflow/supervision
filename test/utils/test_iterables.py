import pytest

from supervision.utils.iterables import create_batches, fill


@pytest.mark.parametrize(
    "sequence, batch_size, expected",
    [
        # Empty sequence, non-zero batch size. Expect empty list.
        ([], 4, []),
        # Non-zero size sequence, batch size of 0. Each item is its own batch.
        ([1, 2, 3], 0, [[1], [2], [3]]),
        # Batch size larger than sequence. All items in a single batch.
        ([1, 2], 4, [[1, 2]]),
        # Batch size evenly divides the sequence. Equal size batches.
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        # Batch size doesn't evenly divide sequence. Last batch smaller.
        ([1, 2, 3, 4], 3, [[1, 2, 3], [4]]),
    ],
)
def test_create_batches(sequence, batch_size, expected) -> None:
    result = list(create_batches(sequence=sequence, batch_size=batch_size))
    assert result == expected


@pytest.mark.parametrize(
    "sequence, desired_size, content, expected",
    [
        # Empty sequence, desired size 0. Expect empty list.
        ([], 0, 1, []),
        # Empty sequence, non-zero desired size. Filled with padding.
        ([], 3, 1, [1, 1, 1]),
        # Sequence at desired size. No changes.
        ([2, 2, 2], 3, 1, [2, 2, 2]),
        # Sequence exceeds desired size. No changes.
        ([2, 2, 2, 2], 3, 1, [2, 2, 2, 2]),
        # Non-empty sequence, shorter than desired. Padding added.
        ([2], 3, 1, [2, 1, 1]),
    ],
)
def test_fill(sequence, desired_size, content, expected) -> None:
    result = fill(sequence=sequence, desired_size=desired_size, content=content)
    assert result == expected
