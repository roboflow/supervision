from collections.abc import Iterator

import pytest

from supervision.utils.iterables import create_batches, fill


@pytest.mark.parametrize(
    ("sequence", "batch_size", "expected"),
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
def test_create_batches(
    sequence: list[int], batch_size: int, expected: list[list[int]]
) -> None:
    """Batching preserves item order across empty, partial and full batches."""
    result = list(create_batches(sequence=sequence, batch_size=batch_size))
    assert result == expected


class TestCreateBatchesStreaming:
    def test_leaves_next_item_in_source(self) -> None:
        """A full batch does not consume an item from the next batch."""
        source = iter([1, 2, 3, 4])
        batches = create_batches(source, 2)

        batch = next(batches)

        assert batch == [1, 2]
        assert next(source) == 3

    def test_yields_complete_batch_before_source_error(self) -> None:
        """A later source error does not prevent a complete batch from being yielded."""

        def source() -> Iterator[int]:
            """Raise after producing a complete batch."""
            yield 1
            yield 2
            raise RuntimeError("source failed")

        batches = create_batches(source(), 2)

        assert next(batches) == [1, 2]
        with pytest.raises(RuntimeError, match="source failed"):
            next(batches)


@pytest.mark.parametrize(
    ("sequence", "desired_size", "content", "expected"),
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
def test_fill(
    sequence: list[int], desired_size: int, content: int, expected: list[int]
) -> None:
    """Padding reaches the requested size without truncating existing items."""
    result = fill(sequence=sequence, desired_size=desired_size, content=content)
    assert result == expected
