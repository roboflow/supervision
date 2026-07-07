from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.classification.core import Classifications


class _MockTensor:
    """Minimal tensor double that supports the tensor chain used by adapters."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def softmax(self, dim: int) -> _MockTensor:
        """Return a tensor double with softmax applied along the requested axis."""
        if self.value.shape[dim] == 0:
            return _MockTensor(self.value)

        exp = np.exp(self.value - np.max(self.value, axis=dim, keepdims=True))
        return _MockTensor(exp / np.sum(exp, axis=dim, keepdims=True))

    def cpu(self) -> _MockTensor:
        """Return the tensor double for chained CPU conversion calls."""
        return self

    def detach(self) -> _MockTensor:
        """Return the tensor double for chained graph-detach calls."""
        return self

    def numpy(self) -> np.ndarray:
        """Return the wrapped NumPy array."""
        return self.value


@pytest.mark.parametrize(
    ("class_id", "confidence", "k", "expected_result", "exception"),
    [
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0.1, 0.2, 0.9, 0.4, 0.5]),
            5,
            (np.array([2, 4, 3, 1, 0]), np.array([0.9, 0.5, 0.4, 0.2, 0.1])),
            DoesNotRaise(),
        ),  # class_id with 5 numbers and 5 confidences
        (
            np.array([5, 1, 2, 3, 4]),
            np.array([0.1, 0.2, 0.9, 0.4, 0.5]),
            1,
            (np.array([2]), np.array([0.9])),
            DoesNotRaise(),
        ),  # class_id with 5 numbers and 5 confidences, retrieve where k = 1
        (
            np.array([4, 1, 2, 3, 6, 5]),
            np.array([0.8, 0.2, 0.9, 0.4, 0.5, 0.1]),
            2,
            (np.array([2, 4]), np.array([0.9, 0.8])),
            DoesNotRaise(),
        ),  # class_id with 5 numbers and 5 confidences, retrieve where k = 3
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([]),
            5,
            None,
            pytest.raises(ValueError, match=r"confidence must be 1d np\.ndarray"),
        ),  # class_id with 5 numbers and 0 confidences
        (
            [0, 1, 2, 3, 4],
            [0.1, 0.2, 0.3, 0.4],
            5,
            None,
            pytest.raises(ValueError, match="\\(n, \\) shape"),
        ),  # class_id with 5 numbers and 4 confidences
    ],
)
def test_top_k(
    class_id: np.ndarray,
    confidence: np.ndarray | None,
    k: int,
    expected_result: tuple[np.ndarray, np.ndarray] | None,
    exception: Exception,
) -> None:
    """Retrieves requested top-k values or raises for malformed confidence input."""
    with exception:
        result = Classifications(
            class_id=np.array(class_id), confidence=np.array(confidence)
        ).get_top_k(k)

        assert np.array_equal(result[0], expected_result[0])
        assert np.array_equal(result[1], expected_result[1])


def test_from_clip_empty_output_dtypes() -> None:
    """Empty CLIP logits produce typed empty classification arrays."""
    result = Classifications.from_clip(_MockTensor(np.empty((1, 0), dtype=np.float32)))

    assert result.class_id.dtype == np.int_
    assert result.confidence is not None
    assert result.confidence.dtype == np.float32


def test_from_timm_empty_output_dtypes() -> None:
    """Empty timm logits produce typed empty classification arrays."""
    result = Classifications.from_timm(_MockTensor(np.empty((1, 0), dtype=np.float32)))

    assert result.class_id.dtype == np.int_
    assert result.confidence is not None
    assert result.confidence.dtype == np.float32


def test_from_timm_softmaxes_logits() -> None:
    """Timm logits are converted to normalized confidence scores."""
    logits = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)

    result = Classifications.from_timm(_MockTensor(logits))

    assert result.confidence is not None
    assert np.allclose(
        result.confidence, _MockTensor(logits).softmax(dim=-1).numpy()[0]
    )
    assert np.isclose(np.sum(result.confidence), 1.0)
