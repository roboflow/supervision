from supervision.utils.iterables import create_batches, fill


def test_create_batches_when_empty_sequence_given() -> None:
    # when
    result = list(create_batches(sequence=[], batch_size=4))

    # then
    assert result == [], "Expected empty generator"


def test_create_batches_when_not_allowed_batch_size_given() -> None:
    # when
    result = list(create_batches(sequence=[1, 2, 3], batch_size=0))

    # then
    assert result == [[1], [2], [3]], (
        "Expected min_batch_size to be established and each element of input "
        "list provided in separate batch"
    )


def test_create_batches_when_batch_size_larger_than_sequence() -> None:
    # when
    result = list(create_batches(sequence=[1, 2], batch_size=4))

    # then
    assert result == [[1, 2]], (
        "Expected whole content to be returned in single batch as input sequence "
        "is smaller than batch size"
    )


def test_create_batches_when_batch_size_multiplier_fits_sequence_length() -> None:
    # when
    result = list(create_batches(sequence=[1, 2, 3, 4], batch_size=2))

    # then
    assert result == [[1, 2], [3, 4]], (
        "Expected input sequence to be returned in two chunks as batch size "
        "is half of sequence length"
    )


def test_create_batches_when_batch_size_multiplier_does_not_fir_sequence_length() -> (
    None
):
    # when
    result = list(create_batches(sequence=[1, 2, 3, 4], batch_size=3))

    # then
    assert result == [[1, 2, 3], [4]], (
        "Expected first batch to be of size 3 and last one to be not "
        "full, with only one element"
    )


def test_fill_when_empty_sequence_given_and_padding_not_needed() -> None:
    # given
    sequence = []

    # when
    result = fill(sequence=sequence, desired_size=0, content=1)

    # then
    assert result == [], "Expected no elements to be added into sequence"


def test_fill_when_empty_sequence_given_and_padding_needed() -> None:
    # given
    sequence = []

    # when
    result = fill(sequence=sequence, desired_size=3, content=1)

    # then
    assert result == [1, 1, 1], "Expected three padding element to be added"


def test_fill_when_non_empty_sequence_given_and_sequence_equal_to_desired_size() -> (
    None
):
    # given
    sequence = [2, 2, 2]

    # when
    result = fill(sequence=sequence, desired_size=3, content=1)

    # then
    assert result == [
        2,
        2,
        2,
    ], "Expected nothing to be added to sequence, as it is already " "in desired size"


def test_fill_when_non_empty_sequence_given_and_sequence_longer_then_desired_size() -> (
    None
):
    # given
    sequence = [2, 2, 2, 2]

    # when
    result = fill(sequence=sequence, desired_size=3, content=1)

    # then
    assert result == [
        2,
        2,
        2,
        2,
    ], "Expected nothing to be added to sequence, as it already " "exceeds desired size"


def test_fill_when_non_empty_sequence_given_and_padding_needed() -> None:
    # given
    sequence = [2]

    # when
    result = fill(sequence=sequence, desired_size=3, content=1)

    # then
    assert result == [
        2,
        1,
        1,
    ], "Expected 2 padding elements to be added to fit desired size"
