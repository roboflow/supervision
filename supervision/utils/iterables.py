from typing import Generator, Iterable, List, TypeVar

SequenceElement = TypeVar("SequenceElement")


def create_batches(
    sequence: Iterable[SequenceElement], batch_size: int
) -> Generator[List[SequenceElement], None, None]:
    """
    Provides a generator that yields chunks of input sequence
    of size specified by `batch_size` parameter. Last
    chunk may be smaller batch.

    Args:
        sequence (Iterable[SequenceElement]): Sequence to be
            split into batches.
        batch_size (int): Expected size of a batch

    Returns:
        Generator[List[SequenceElement], None, None]: Generator
            to yield chinks of `sequence` of size `batch_size`,
            up to the length of input `sequence`.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if len(current_batch) > 0:
        yield current_batch


def fill(
    sequence: List[SequenceElement],
    desired_size: int,
    content: SequenceElement,
) -> List[SequenceElement]:
    """
    Fill the sequence with padding elements until sequence reaches
    desired size.

    Args:
        sequence (List[SequenceElement]): Input sequence.
        desired_size (int): Expected size of output list - difference
            between this value and actual `sequence` length (if positive)
            dictates how many elements will be added as padding.
        content (SequenceElement): Element to be placed at the end of
            input `sequence` as padding.

    Returns:
        List[SequenceElement]: Padded version of input `sequence` (if needed)
    """
    missing_size = max(0, desired_size - len(sequence))
    required_padding = [content] * missing_size
    sequence.extend(required_padding)
    return sequence
