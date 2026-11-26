"""Test weighted blending with the NumPy fallback regardless of the active backend."""

from __future__ import annotations

import numpy as np
import pytest

from supervision._cv2._image import _add_weighted, _uint8_add_weighted_lut


@pytest.mark.parametrize(
    ("shape", "alpha", "beta", "gamma"),
    [
        pytest.param((150, 160, 3), 0.35, 0.65, 0.0, id="bgr-multi-strip"),
        pytest.param((150, 512), 0.7, 0.3, 12.5, id="gray-multi-strip-gamma"),
        pytest.param((130, 128, 4), 1.2, -0.4, -10.0, id="bgra-negative-weights"),
    ],
)
def test_fallback_add_weighted_uint8_lut_matches_reference_across_strips(
    shape: tuple[int, ...], alpha: float, beta: float, gamma: float
) -> None:
    """Match direct float32 fused blending bit-for-bit across LUT strip boundaries."""
    rng = np.random.default_rng(20261120)
    source = rng.integers(0, 256, size=shape, dtype=np.uint8)
    other = rng.integers(0, 256, size=shape, dtype=np.uint8)
    # Force non-LUT reference by blending in float32 then rounding to uint8.
    expected_f32 = _add_weighted(
        source.astype(np.float32), alpha, other.astype(np.float32), beta, gamma
    )
    expected = np.clip(np.rint(expected_f32), 0, 255).astype(np.uint8)

    actual = _add_weighted(source, alpha, other, beta, gamma)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_fallback_add_weighted_handles_shifted_overlapping_dst(
    dtype: type[np.generic],
) -> None:
    """Shifted overlapping dst views do not corrupt later strip reads."""
    base = np.broadcast_to(np.arange(131, dtype=dtype)[:, None], (131, 512)).copy()
    expected = base[:-1].copy()
    dst_view = base[1:]

    actual = _add_weighted(base[:-1], 1.0, base[:-1], 0.0, 0.0, dst=dst_view)

    assert actual is dst_view
    np.testing.assert_array_equal(actual, expected)


def test_fallback_add_weighted_exact_inplace_dst_preserves_values() -> None:
    """Exact in-place multi-strip dst blends match out-of-place output."""
    rng = np.random.default_rng(20261121)
    src1 = rng.integers(0, 256, size=(140, 160, 3), dtype=np.uint8)
    src2 = rng.integers(0, 256, size=(140, 160, 3), dtype=np.uint8)
    expected = _add_weighted(src1.copy(), 0.4, src2, 0.6, 0.0)

    actual = _add_weighted(src1, 0.4, src2, 0.6, 0.0, dst=src1)

    assert actual is src1
    np.testing.assert_array_equal(actual, expected)


def test_exact_destination_with_shared_rows_blends_each_value_once() -> None:
    """Every strip sees the original pixels when destination rows alias each other."""
    backing = np.zeros(512, dtype=np.uint8)
    scene = np.ndarray((130, 512), dtype=np.uint8, buffer=backing, strides=(0, 1))
    other = np.zeros_like(scene)

    actual = _add_weighted(scene, 1.0, other, 0.0, 1.0, dst=scene)

    assert actual is scene
    np.testing.assert_array_equal(backing, np.ones_like(backing))


def test_broadcast_destination_receives_the_entire_blend() -> None:
    """An extra leading destination axis broadcasts the entire output."""
    source = np.full((130, 512), 5, dtype=np.uint8)
    destination = np.zeros((2, 130, 512), dtype=np.uint8)

    actual = _add_weighted(source, 1.0, source, 0.0, 1.0, dst=destination)

    assert actual is destination
    np.testing.assert_array_equal(destination, np.full_like(destination, 6))


@pytest.mark.parametrize(
    ("source_rows", "destination_rows"),
    [
        pytest.param(130, 129, id="smaller-destination"),
        pytest.param(130, 131, id="larger-destination"),
        pytest.param(0, 1, id="empty-source"),
    ],
)
def test_invalid_destination_shape_raises_before_writing(
    source_rows: int, destination_rows: int
) -> None:
    """Reject incompatible output dimensions before the destination is mutated."""
    source = np.full((source_rows, 512), 5, dtype=np.uint8)
    destination = np.full((destination_rows, 512), 99, dtype=np.uint8)

    with pytest.raises(ValueError, match="broadcast"):
        _add_weighted(source, 1.0, source, 0.0, 0.0, dst=destination)

    np.testing.assert_array_equal(destination, np.full_like(destination, 99))


def test_cold_blend_respects_numpy_underflow_policy() -> None:
    """Nonzero inputs never trigger underflow while building unrelated table entries."""
    source = np.full((130, 512), 5, dtype=np.uint8)
    other = np.full_like(source, 3)
    _uint8_add_weighted_lut.cache_clear()

    with np.errstate(under="raise"):
        actual = _add_weighted(source, 0.5, other, 0.5, 0.0)

    np.testing.assert_array_equal(actual, np.full_like(source, 4))
