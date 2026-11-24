from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.classification.core import Classifications


class _MockTensor:
    """Minimal tensor double that supports the tensor chain used by adapters."""

    def __init__(self, value: np.ndarray) -> None:
        """Store the backing values and expose their dtype for adapter conversion."""
        self.value = value
        self.dtype = value.dtype

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
    exception: AbstractContextManager[object],
) -> None:
    """Retrieves requested top-k values or raises for malformed confidence input."""
    with exception:
        result = Classifications(
            class_id=np.array(class_id), confidence=np.array(confidence)
        ).get_top_k(k)

        assert expected_result is not None
        assert np.array_equal(result[0], expected_result[0])
        assert np.array_equal(result[1], expected_result[1])


def test_top_k_rejects_negative_count() -> None:
    """Rejects negative counts instead of applying NumPy negative slicing."""
    classifications = Classifications(
        class_id=np.array([0, 1, 2]), confidence=np.array([0.1, 0.9, 0.5])
    )

    with pytest.raises(ValueError, match="k must be non-negative"):
        classifications.get_top_k(-1)


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


class TestClassificationTensorPrecision:
    """Framework adapters preserve representable probabilities across precisions."""

    @pytest.mark.parametrize("adapter", ["from_clip", "from_timm"])
    @pytest.mark.parametrize(
        "dtype_name", ["bfloat16", "float16", "float32", "float64"]
    )
    @pytest.mark.parametrize("class_count", [0, 3])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_preserves_probabilities_in_a_numpy_supported_dtype(
        self,
        adapter: str,
        dtype_name: str,
        class_count: int,
        requires_grad: bool,
    ) -> None:
        """Convert framework probabilities without losing their values or precision."""
        torch = pytest.importorskip("torch")
        logits = torch.arange(class_count, dtype=getattr(torch, dtype_name)).reshape(
            1, -1
        )
        logits.requires_grad_(requires_grad)
        expected_dtype = np.float32 if dtype_name == "bfloat16" else dtype_name
        if class_count == 0:
            expected_dtype = np.float32
        expected = np.array(
            logits.softmax(dim=-1).detach().tolist()[0], dtype=expected_dtype
        )

        result = getattr(Classifications, adapter)(logits)

        np.testing.assert_array_equal(result.class_id, np.arange(class_count))
        np.testing.assert_array_equal(result.confidence, expected)
        assert result.confidence is not None
        assert result.confidence.dtype == expected.dtype


def test_classifications_compare_numpy_fields_by_value() -> None:
    """Classifications equality handles NumPy arrays and confidence values."""
    left = Classifications(
        class_id=np.array([0, 1], dtype=np.int_),
        confidence=np.array([0.25, 0.75], dtype=np.float32),
    )
    right = Classifications(
        class_id=np.array([0, 1], dtype=np.int_),
        confidence=np.array([0.25, 0.75], dtype=np.float32),
    )
    different = Classifications(
        class_id=np.array([0, 1], dtype=np.int_),
        confidence=np.array([0.25, 0.5], dtype=np.float32),
    )

    assert left == right
    assert left != different


class TestClassificationsGetTopK:
    """Order of the classes that `Classifications.get_top_k` returns."""

    def test_returns_first_class_among_tied_top_scores(self) -> None:
        """A tie for the best score goes to the earlier class, as `np.argmax`."""
        classifications = Classifications(
            class_id=np.array([0, 1, 2]), confidence=np.array([0.4, 0.4, 0.2])
        )

        class_id, confidence = classifications.get_top_k(1)

        np.testing.assert_array_equal(class_id, [0])
        np.testing.assert_array_equal(confidence, [0.4])

    def test_keeps_input_order_among_ties_with_many_classes(self) -> None:
        """Tied scores among 1000 classes come back in input order, not scrambled."""
        confidence = np.zeros(1000, dtype=np.float16)
        confidence[[10, 20, 30, 40]] = 0.25
        classifications = Classifications(
            class_id=np.arange(1000), confidence=confidence
        )

        class_id, _ = classifications.get_top_k(6)

        np.testing.assert_array_equal(class_id, [10, 20, 30, 40, 0, 1])

    def test_ranks_nan_confidence_last(self) -> None:
        """A NaN confidence is never returned ahead of real scores."""
        classifications = Classifications(
            class_id=np.array([0, 1, 2]), confidence=np.array([0.7, np.nan, 0.3])
        )

        class_id, _ = classifications.get_top_k(2)

        np.testing.assert_array_equal(class_id, [0, 2])

    @pytest.mark.parametrize(
        "confidence",
        [
            pytest.param(np.array([2**53, 2**53 + 1], dtype=np.int64), id="int64"),
            pytest.param(
                np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64), id="uint64"
            ),
            pytest.param(
                np.array(
                    [
                        np.longdouble(0.5),
                        np.nextafter(np.longdouble(0.5), np.longdouble(1)),
                    ]
                ),
                marks=pytest.mark.skipif(
                    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
                    reason="longdouble has no additional precision on this platform",
                ),
                id="extended-float",
            ),
        ],
    )
    def test_orders_distinct_scores_without_losing_precision(
        self, confidence: np.ndarray
    ) -> None:
        """Higher scores rank first even when float64 conversion loses precision."""
        classifications = Classifications(
            class_id=np.array([0, 1]), confidence=confidence
        )

        class_id, result_confidence = classifications.get_top_k(2)

        np.testing.assert_array_equal(class_id, [1, 0])
        np.testing.assert_array_equal(result_confidence, confidence[[1, 0]])

    def test_preserves_ties_and_nan_order_with_infinite_scores(self) -> None:
        """NaNs follow infinities while ties preserve their original order."""
        classifications = Classifications(
            class_id=np.arange(6),
            confidence=np.array([np.nan, -np.inf, np.inf, np.nan, np.inf, -np.inf]),
        )

        class_id, _ = classifications.get_top_k(6)

        np.testing.assert_array_equal(class_id, [2, 4, 1, 5, 0, 3])
