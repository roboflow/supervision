from typing import Generator, Iterable, List, TypeVar

SequenceElement = TypeVar("SequenceElement")


def create_batches(
    sequence: Iterable[SequenceElement], batch_size: int
) -> Generator[List[SequenceElement], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if len(current_batch) > 0:
        yield current_batch
