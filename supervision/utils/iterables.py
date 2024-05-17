from typing import Generator, Iterable, List, TypeVar

V = TypeVar("V")


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Provides a generator that yields chunks of the input sequence
    of the size specified by the `batch_size` parameter. The last
    chunk may be a smaller batch.

    Args:
        sequence (Iterable[V]): The sequence to be split into batches.
        batch_size (int): The expected size of a batch.

    Returns:
        (Generator[List[V], None, None]): A generator that yields chunks
            of `sequence` of size `batch_size`, up to the length of
            the input `sequence`.

    Examples:
        ```python
        list(create_batches([1, 2, 3, 4, 5], 2))
        # [[1, 2], [3, 4], [5]]

        list(create_batches("abcde", 3))
        # [['a', 'b', 'c'], ['d', 'e']]
        ```
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


def fill(sequence: List[V], desired_size: int, content: V) -> List[V]:
    """
    Fill the sequence with padding elements until the sequence reaches
    the desired size.

    Args:
        sequence (List[V]): The input sequence.
        desired_size (int): The expected size of the output list. The
            difference between this value and the actual length of `sequence`
            (if positive) dictates how many elements will be added as padding.
        content (V): The element to be placed at the end of the input
            `sequence` as padding.

    Returns:
        (List[V]): A padded version of the input `sequence` (if needed).

    Examples:
        ```python
        fill([1, 2], 4, 0)
        # [1, 2, 0, 0]

        fill(['a', 'b'], 3, 'c')
        # ['a', 'b', 'c']
        ```
    """
    missing_size = max(0, desired_size - len(sequence))
    sequence.extend([content] * missing_size)
    return sequence
