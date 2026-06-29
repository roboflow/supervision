from collections.abc import Generator, Iterable
from typing import TypeVar

V = TypeVar("V")


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[list[V], None, None]:
    """
    Provides a generator that yields chunks of the input sequence
    of the size specified by the `batch_size` parameter. The last
    chunk may be a smaller batch.

    Args:
        sequence: The sequence to be split into batches.
        batch_size: The expected size of a batch.

    Returns:
        A generator that yields chunks
            of `sequence` of size `batch_size`, up to the length of
            the input `sequence`.

    Examples:
        ```pycon
        >>> from supervision.utils.iterables import create_batches
        >>> list(create_batches([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
        >>> list(create_batches("abcde", 3))
        [['a', 'b', 'c'], ['d', 'e']]

        ```
    """
    batch_size = max(batch_size, 1)
    current_batch: list[V] = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


def fill(sequence: list[V], desired_size: int, content: V) -> list[V]:
    """
    Fill the sequence with padding elements until the sequence reaches
    the desired size.

    Args:
        sequence: The input sequence.
        desired_size: The expected size of the output list. The
            difference between this value and the actual length of `sequence`
            (if positive) dictates how many elements will be added as padding.
        content: The element to be placed at the end of the input
            `sequence` as padding.

    Returns:
        A padded version of the input `sequence` (if needed).

    Examples:
        ```pycon
        >>> from supervision.utils.iterables import fill
        >>> fill([1, 2], 4, 0)
        [1, 2, 0, 0]
        >>> fill(['a', 'b'], 3, 'c')
        ['a', 'b', 'c']

        ```
    """
    missing_size = max(0, desired_size - len(sequence))
    sequence.extend([content] * missing_size)
    return sequence


def find_duplicates(sequence: list[V]) -> list[V]:
    """
    Find all duplicate elements in the input sequence.

    Args:
        sequence: The input sequence.

    Returns:
        A list of duplicate elements found in the sequence.

    Examples:
        ```pycon
        >>> from supervision.utils.iterables import find_duplicates
        >>> sorted(find_duplicates([1, 2, 3, 2, 4, 5, 1]))
        [1, 2]
        >>> find_duplicates(['a', 'b', 'c'])
        []

        ```
    """
    seen = set()
    duplicates = set()
    for element in sequence:
        if element in seen:
            duplicates.add(element)
        else:
            seen.add(element)
    return list(duplicates)
