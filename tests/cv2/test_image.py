"""Tests for private image-operation and I/O fallbacks."""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from supervision._cv2._image import (
    _add_weighted,
    _convert_scale_abs,
    _copy_make_border,
    _flip,
    _fused_multiply_add_float32,
    _imdecode,
    _imencode,
    _imread,
    _imwrite,
    _mean,
    _opencv_default_save_options,
    _resize,
)
from supervision._cv2.constants import (
    _BORDER_CONSTANT,
    _IMREAD_COLOR,
    _IMREAD_UNCHANGED,
)

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    ("flip_code", "expected"),
    [
        pytest.param(0, np.array([[3, 4], [1, 2]], dtype=np.uint8), id="vertical"),
        pytest.param(1, np.array([[2, 1], [4, 3]], dtype=np.uint8), id="horizontal"),
        pytest.param(-1, np.array([[4, 3], [2, 1]], dtype=np.uint8), id="both"),
        pytest.param(
            2, np.array([[2, 1], [4, 3]], dtype=np.uint8), id="positive-horizontal"
        ),
        pytest.param(
            -2, np.array([[4, 3], [2, 1]], dtype=np.uint8), id="negative-both"
        ),
    ],
)
def test_fallback_flip_matches_opencv(flip_code: int, expected: np.ndarray) -> None:
    """Match OpenCV flip direction and return a contiguous array."""
    source = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    np.testing.assert_array_equal(_flip(source, flip_code), expected)
    np.testing.assert_array_equal(_flip(source, flip_code), cv2.flip(source, flip_code))


@pytest.mark.parametrize(
    ("source", "value"),
    [
        pytest.param(
            np.array([[0, 100], [200, 255]], dtype=np.uint8),
            7,
            id="grayscale-scalar",
        ),
        pytest.param(
            np.array([[0, 100], [200, 255]], dtype=np.uint8),
            (5, 9, 20),
            id="grayscale-sequence-uses-first-element",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            (7, 8),
            id="multichannel-sequence-shorter-than-channels-pads-with-zero",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            (7, 8, 9, 10),
            id="multichannel-sequence-longer-than-channels-truncates",
        ),
        pytest.param(
            np.array(
                [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
                dtype=np.uint8,
            ),
            100,
            id="multichannel-scalar-fills-only-first-channel",
        ),
    ],
)
def test_fallback_copy_make_border_matches_opencv(
    source: np.ndarray, value: int | tuple[int, ...]
) -> None:
    """Match OpenCV constant-border padding for scalar and Sequence values."""
    np.testing.assert_array_equal(
        _copy_make_border(source, 1, 1, 2, 2, _BORDER_CONSTANT, value),
        cv2.copyMakeBorder(source, 1, 1, 2, 2, cv2.BORDER_CONSTANT, value=value),
    )


def test_fallback_add_weighted_matches_opencv() -> None:
    """Match OpenCV weighted image blending."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    np.testing.assert_array_equal(
        _add_weighted(source, 0.5, other, 0.5, 10),
        cv2.addWeighted(source, 0.5, other, 0.5, 10),
    )


def _assert_matches_opencv_add_weighted(
    actual: np.ndarray, expected: np.ndarray
) -> None:
    """Compare with OpenCV, allowing for build-specific addWeighted kernels.

    The fallback reproduces OpenCV's fused generic kernel. Other builds round a value
    on a boundary differently: x86 builds without FMA multiply and add separately, and
    arm64 builds hand 8- and 16-bit blends to the Carotene HAL, which sums in the other
    order, `fma(src2, beta, fma(src1, alpha, gamma))`. Integer results can then be one
    level away and float32 results a few units in the last place. The exact rounding is
    pinned by the fixed-expectation tests below instead.
    """
    assert actual.dtype == expected.dtype
    if np.issubdtype(actual.dtype, np.integer):
        np.testing.assert_allclose(actual, expected, atol=1, rtol=0)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-6)


@pytest.mark.parametrize(
    ("alpha", "beta", "gamma"),
    [
        pytest.param(0.3, 0.7, 0, id="opacity-0.3"),
        pytest.param(0.7, 1 - 0.7, 0, id="opacity-0.7"),
        pytest.param(0.1, 0.9, 0, id="opacity-0.1"),
        pytest.param(0.35, 0.65, 2.5, id="with-gamma"),
    ],
)
def test_fallback_add_weighted_rounds_8bit_blends_like_opencv(
    alpha: float, beta: float, gamma: float
) -> None:
    """Blend every pair of 8-bit values in float32, as OpenCV does."""
    first, second = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    source = np.stack([first, second, 255 - first], axis=-1).astype(np.uint8)
    other = np.stack([second, first, 255 - second], axis=-1).astype(np.uint8)

    _assert_matches_opencv_add_weighted(
        _add_weighted(source, alpha, other, beta, gamma),
        cv2.addWeighted(source, alpha, other, beta, gamma),
    )


@pytest.mark.parametrize(
    ("first", "second", "alpha", "beta", "expected"),
    [
        pytest.param(0, 45, 0.3, 0.7, 32, id="midpoint-rounds-up"),
        pytest.param(11, 26, 0.3, 0.7, 21, id="just-below-midpoint-rounds-down"),
    ],
)
def test_fallback_add_weighted_rounds_fused_blend_to_fixed_values(
    first: int, second: int, alpha: float, beta: float, expected: int
) -> None:
    """Round the fused float32 blend on a boundary to the pinned 8-bit value."""
    source = np.full((16, 64), first, dtype=np.uint8)
    other = np.full_like(source, second)

    actual = _add_weighted(source, alpha, other, beta, 0)

    np.testing.assert_array_equal(actual, np.full_like(source, expected))


@pytest.mark.parametrize(
    ("dtype", "first", "second", "expected"),
    [
        pytest.param(np.uint8, 97, 1, 1, id="uint8-tie-pushed-up"),
        pytest.param(np.int8, -97, -1, -1, id="int8-tie-pushed-down"),
    ],
)
def test_fallback_add_weighted_breaks_float32_midpoint_with_dropped_addend(
    dtype: type[np.generic], first: int, second: int, expected: int
) -> None:
    """Let an addend below float64 resolution decide a float32 midpoint tie."""
    alpha = 172961 / 2**25  # 97 * alpha is 0.5 + 2**-25, a float32 midpoint
    beta = 2.0**-60  # lost when the sum is rounded to float64 first
    source = np.full((16, 64), first, dtype=dtype)
    other = np.full_like(source, second)

    actual = _add_weighted(source, alpha, other, beta, 0)

    np.testing.assert_array_equal(actual, np.full_like(source, expected))


def _round_exactly_to_float32(exact: Fraction) -> np.float32:
    """Round an exact rational to the nearest float32, ties to even."""
    guess = np.float32(float(exact))
    candidates = [
        guess,
        np.nextafter(guess, np.float32(np.inf)),
        np.nextafter(guess, np.float32(-np.inf)),
    ]
    return min(
        candidates,
        key=lambda value: (
            abs(Fraction(float(value)) - exact),
            int(value.view("u4")) & 1,
        ),
    )


def test_fused_multiply_add_float32_matches_exact_rational_rounding() -> None:
    """Round each fused multiply-add once, as exact rational arithmetic does."""
    rng = np.random.default_rng(20261028)
    values = rng.integers(0, 256, 400).astype(np.float64)
    weights = np.float32(rng.uniform(0, 1, 400)).astype(np.float64)
    addends = np.float32(rng.uniform(-300, 300, 400)).astype(np.float64)
    # Pairs whose exact product sits on a float32 midpoint, with a tiny addend.
    values = np.append(values, [97.0, 97.0, 97.0])
    weights = np.append(weights, [172961 / 2**25] * 3)
    addends = np.append(addends, [2.0**-60, -(2.0**-60), 0.0])
    expected = np.array(
        [
            _round_exactly_to_float32(Fraction(v) * Fraction(w) + Fraction(a))
            for v, w, a in zip(values, weights, addends)
        ]
    )

    actual = np.array(
        [
            _fused_multiply_add_float32(np.array([v]), np.float64(w), np.float64(a))[0]
            for v, w, a in zip(values, weights, addends)
        ]
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.int16, np.float32])
def test_fallback_add_weighted_matches_opencv_for_float32_blend_depths(
    dtype: type[np.generic],
) -> None:
    """Match OpenCV on the other depths it blends in float32."""
    rng = np.random.default_rng(20261027)
    source: np.ndarray
    other: np.ndarray
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(np.dtype(dtype))
        source = rng.integers(info.min, info.max, (64, 48, 3), endpoint=True)
        other = rng.integers(info.min, info.max, (64, 48, 3), endpoint=True)
    else:
        source = rng.uniform(-100, 400, (64, 48, 3))
        other = rng.uniform(-100, 400, (64, 48, 3))
    source, other = source.astype(dtype), other.astype(dtype)

    _assert_matches_opencv_add_weighted(
        _add_weighted(source, 0.35, other, 0.65, 0),
        cv2.addWeighted(source, 0.35, other, 0.65, 0),
    )


def test_fallback_add_weighted_supports_destination() -> None:
    """Write weighted image blending results into the provided destination."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)
    destination = np.empty_like(source)

    actual = _add_weighted(source, 0.5, other, 0.5, 10, dst=destination)

    assert actual is destination
    np.testing.assert_array_equal(actual, cv2.addWeighted(source, 0.5, other, 0.5, 10))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(None, id="none"),
        pytest.param(-1, id="opencv-sentinel"),
    ],
)
def test_fallback_add_weighted_accepts_default_dtype(dtype: int | None) -> None:
    """Treat both None and OpenCV's -1 sentinel as the default output depth."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    np.testing.assert_array_equal(
        _add_weighted(source, 0.5, other, 0.5, 10, dtype=dtype),
        cv2.addWeighted(source, 0.5, other, 0.5, 10),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(0, id="cv-8u"),
        pytest.param(5, id="cv-32f"),
    ],
)
def test_fallback_add_weighted_rejects_non_default_dtype(dtype: int) -> None:
    """Fail loud when a caller requests an unsupported output depth."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    with pytest.raises(ValueError, match="output depth"):
        _add_weighted(source, 0.5, other, 0.5, 10, dtype=dtype)


def test_fallback_convert_scale_abs_matches_opencv() -> None:
    """Match OpenCV absolute scale-and-convert semantics."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(
        _convert_scale_abs(source, 1.5, -20),
        cv2.convertScaleAbs(source, alpha=1.5, beta=-20),
    )


class TestFallbackMean:
    def test_matches_opencv_for_masked_uint8(self) -> None:
        """Match OpenCV masked mean semantics for integer images."""
        source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

        assert _mean(source, mask) == cv2.mean(source, mask)

    def test_preserves_float32_precision(self) -> None:
        """Retain small values when averaging large float32 pixels."""
        source = np.array([[1e8, 1, -1e8]], dtype=np.float32)

        assert _mean(source) == cv2.mean(source)

    def test_preserves_masked_multichannel_float32_precision(self) -> None:
        """Match OpenCV precision for selected pixels in each channel."""
        source = np.array(
            [[[1e8, -1e8], [1, 5], [-1e8, 1e8], [999, 999]]],
            dtype=np.float32,
        )
        mask = np.array([[255, 255, 255, 0]], dtype=np.uint8)

        assert _mean(source, mask) == pytest.approx(
            cv2.mean(source, mask), rel=0, abs=1e-15
        )


@pytest.mark.parametrize(
    ("interpolation", "atol"),
    [
        pytest.param(cv2.INTER_NEAREST, 0, id="nearest"),
        pytest.param(cv2.INTER_LINEAR, 1, id="linear"),
    ],
)
def test_fallback_resize_matches_opencv(interpolation: int, atol: int) -> None:
    """Match OpenCV resize shape and pixel values within the interpolation budget."""
    source = np.arange(20, dtype=np.uint8).reshape(4, 5)

    actual = _resize(source, (9, 7), interpolation=interpolation)
    expected = cv2.resize(source, (9, 7), interpolation=interpolation)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("source_width", "width"),
    [
        pytest.param(6, 34, id="six-to-thirty-four"),
        pytest.param(9, 51, id="nine-to-fifty-one"),
        pytest.param(10, 74, id="ten-to-seventy-four"),
        pytest.param(12, 34, id="twelve-to-thirty-four"),
    ],
)
def test_fallback_nearest_resize_matches_opencv_at_whole_sample_positions(
    source_width: int, width: int
) -> None:
    """Pick OpenCV's source pixel where `x * source_width / width` is a whole number."""
    source = np.arange(source_width, dtype=np.uint8).reshape(1, -1)

    actual = _resize(source, (width, 1), interpolation=cv2.INTER_NEAREST)
    expected = cv2.resize(source, (width, 1), interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(actual, expected)


def test_fallback_nearest_resize_matches_opencv_on_both_axes() -> None:
    """Match OpenCV's nearest sampling grid on rows and columns of a color image."""
    rng = np.random.default_rng(20261012)
    source = rng.integers(0, 256, (9, 6, 3), dtype=np.uint8)

    actual = _resize(source, (34, 51), interpolation=cv2.INTER_NEAREST)
    expected = cv2.resize(source, (34, 51), interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("interpolation", "atol"),
    [
        pytest.param(cv2.INTER_NEAREST, 0, id="nearest"),
        pytest.param(cv2.INTER_LINEAR, 1, id="linear"),
    ],
)
@pytest.mark.parametrize(
    ("fx", "fy"),
    [
        pytest.param(0.5, 0.5, id="half"),
        pytest.param(0.6, 0.6, id="non-integer-downscale"),
        pytest.param(1.5, 1.5, id="non-integer-upscale"),
        pytest.param(0.3, 3.7, id="mixed"),
        pytest.param(1 / 7, 1 / 7, id="pixelate"),
        pytest.param(1.03, 1.03, id="rounds-to-same-size"),
    ],
)
@pytest.mark.parametrize("shape", [(13, 10, 3), (13, 10)], ids=["bgr", "gray"])
def test_fallback_resize_with_scale_factors_matches_opencv(
    interpolation: int, atol: int, fx: float, fy: float, shape: tuple[int, ...]
) -> None:
    """Map pixels with `fx`/`fy` like OpenCV when `dsize` is not given."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, shape, dtype=np.uint8)

    actual = _resize(source, None, fx=fx, fy=fy, interpolation=interpolation)
    expected = cv2.resize(source, None, fx=fx, fy=fy, interpolation=interpolation)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def test_fallback_linear_resize_preserves_random_uint8_contract() -> None:
    """Preserve dtype, contiguity, and the one-LSB visual interpolation budget."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)

    actual = _resize(source, (31, 29), interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, (31, 29), interpolation=cv2.INTER_LINEAR)

    assert actual.dtype == source.dtype
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)


def test_fallback_linear_resize_matches_opencv_when_downsampling_uint8() -> None:
    """Preserve OpenCV's half-pixel interpolation for uint8 downsampling."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)

    actual = _resize(source, (1, 18), interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, (1, 18), interpolation=cv2.INTER_LINEAR)

    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)


def test_fallback_linear_resize_preserves_rgba_channels() -> None:
    """Avoid Pillow alpha premultiplication when resizing uint8 RGBA arrays."""
    rng = np.random.default_rng(20260717)
    source = rng.integers(0, 256, (9, 7, 4), dtype=np.uint8)

    actual = _resize(source, (13, 4), interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, (13, 4), interpolation=cv2.INTER_LINEAR)

    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)


# OpenCV's arm64 builds hand some uint8 `INTER_LINEAR` resizes to the Carotene HAL
# (`TEGRA_RESIZE`): one channel with a source at least 8 wide and an output of at
# least 8x8, or four channels with an output at least 8 tall that is not an exact
# 2x reduction. Carotene uses its own fixed-point kernel, up to two levels away from
# the generic one this fallback reproduces.
_OPENCV_MAY_USE_CAROTENE = platform.machine().lower() in {"arm64", "aarch64"}


@pytest.mark.parametrize(
    ("source_shape", "dsize", "fx", "fy", "carotene"),
    [
        pytest.param((9, 9), (13, 15), 0, 0, True, id="gray-upscale"),
        pytest.param((17, 23, 3), (31, 29), 0, 0, False, id="bgr-upscale"),
        pytest.param((40, 37, 3), (11, 26), 0, 0, False, id="bgr-mixed"),
        pytest.param((9, 7, 4), (13, 4), 0, 0, False, id="rgba"),
        pytest.param((13, 10, 3), None, 0.6, 0.6, False, id="factor-downscale"),
        pytest.param((13, 10, 3), None, 0.3, 3.7, False, id="factor-mixed"),
        pytest.param((3, 3), None, 0.5, 0.5, False, id="half-gray-odd"),
        pytest.param((23, 27, 3), None, 0.5, 0.5, False, id="half-bgr-odd"),
        pytest.param((19, 38, 4), None, 0.5, 0.5, False, id="half-rgba-odd-rows"),
        pytest.param((6, 5, 2), None, 0.5, 0.5, False, id="half-two-channel"),
        pytest.param((5, 7, 3), None, 0.5, 0.5, False, id="half-drops-last-pixel"),
        pytest.param((8, 6, 3), (6, 8), 0, 0, False, id="same-size"),
    ],
)
def test_fallback_linear_resize_is_bit_exact_for_uint8(
    source_shape: tuple[int, ...],
    dsize: tuple[int, int] | None,
    fx: float,
    fy: float,
    carotene: bool,
) -> None:
    """Reproduce OpenCV's fixed-point uint8 `INTER_LINEAR` resize pixel for pixel."""
    rng = np.random.default_rng(20261027)
    source = rng.integers(0, 256, source_shape, dtype=np.uint8)

    actual = _resize(source, dsize, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    expected = cv2.resize(source, dsize, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    assert actual.dtype == expected.dtype
    assert actual.flags.c_contiguous
    if carotene and _OPENCV_MAY_USE_CAROTENE:
        np.testing.assert_allclose(actual, expected, atol=2, rtol=0)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_fallback_linear_resize_copies_uint8_image_of_unchanged_size() -> None:
    """Return a copy, not a view, when the requested size matches the source."""
    source = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

    actual = _resize(source, (4, 2), interpolation=cv2.INTER_LINEAR)

    np.testing.assert_array_equal(actual, source)
    assert not np.shares_memory(actual, source)


def test_fallback_float_resize_preserves_mask_threshold_decision() -> None:
    """Keep float-mask interpolation on the established numeric path."""
    rng = np.random.default_rng(20260717)
    source = rng.random((3836, 17, 23), dtype=np.float32)[-1]
    actual = _resize(source, (31, 29), interpolation=cv2.INTER_LINEAR)

    assert actual[8, 27] == np.float32(0.49999991059303284)
    assert not bool((actual > 0.5)[8, 27])


def test_fallback_image_io_preserves_bgr(tmp_path: Path) -> None:
    """Preserve BGR channel order when writing and reading an image."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    image_path = tmp_path / "image.png"

    assert _imwrite(str(image_path), image)
    actual = _imread(str(image_path), _IMREAD_COLOR)
    assert actual is not None
    np.testing.assert_array_equal(actual, image)


def test_fallback_image_io_returns_none_for_missing_file(tmp_path: Path) -> None:
    """Return None when reading a missing image file."""
    assert _imread(str(tmp_path / "missing.png"), _IMREAD_COLOR) is None


def test_fallback_image_io_preserves_alpha(tmp_path: Path) -> None:
    """Preserve alpha channels when reading unchanged images."""
    alpha = np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], dtype=np.uint8)
    alpha_path = tmp_path / "alpha.png"

    assert _imwrite(str(alpha_path), alpha)
    np.testing.assert_array_equal(
        _imread(str(alpha_path), _IMREAD_UNCHANGED),
        cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED),
    )


def test_fallback_image_io_preserves_sixteen_bit_unchanged(tmp_path: Path) -> None:
    """Preserve sixteen-bit pixel values when reading unchanged images."""
    sixteen_bit = np.array([[0, 12345], [54321, 65535]], dtype=np.uint16)
    sixteen_bit_path = tmp_path / "sixteen-bit.png"

    assert _imwrite(str(sixteen_bit_path), sixteen_bit)
    np.testing.assert_array_equal(
        _imread(str(sixteen_bit_path), _IMREAD_UNCHANGED),
        cv2.imread(str(sixteen_bit_path), cv2.IMREAD_UNCHANGED),
    )


@pytest.mark.parametrize(
    ("pixels", "mode", "transparency"),
    [
        pytest.param(
            np.array([[[0, 0], [90, 128]], [[180, 200], [255, 255]]], dtype=np.uint8),
            "LA",
            None,
            id="gray-alpha",
        ),
        pytest.param(
            np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
            "RGB",
            (10, 20, 30),
            id="rgb-transparent-color",
        ),
        pytest.param(
            np.array([[255, 0], [0, 255]], dtype=np.uint8), "1", None, id="one-bit"
        ),
        pytest.param(
            np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
            "P",
            None,
            id="palette",
        ),
        pytest.param(
            np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
            "P",
            0,
            id="palette-transparent-color",
        ),
    ],
)
def test_fallback_imread_unchanged_matches_opencv_channels_and_depth(
    tmp_path: Path, pixels: np.ndarray, mode: str, transparency: object
) -> None:
    """Read PNGs unchanged with the channels and bit depth OpenCV returns."""
    from PIL import Image

    image_path = tmp_path / "image.png"
    image = Image.fromarray(pixels).convert(mode)
    if transparency is None:
        image.save(image_path)
    else:
        image.save(image_path, transparency=transparency)

    actual = _imread(str(image_path), _IMREAD_UNCHANGED)
    expected = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    assert actual is not None
    assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("suffix", [".jpg", ".tif"])
def test_fallback_imread_unchanged_converts_cmyk_like_opencv(
    tmp_path: Path, suffix: str
) -> None:
    """Read CMYK images unchanged as the color pixels OpenCV returns, not as ink."""
    from PIL import Image

    image_path = tmp_path / f"image{suffix}"
    rng = np.random.default_rng(0)
    cmyk = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
    Image.fromarray(cmyk, mode="CMYK").save(image_path)

    actual = _imread(str(image_path), _IMREAD_UNCHANGED)
    expected = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    assert actual is not None
    assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
    np.testing.assert_allclose(actual.astype(np.int16), expected, atol=2)


def test_fallback_imread_unchanged_expands_palette_with_alpha_like_opencv(
    tmp_path: Path,
) -> None:
    """Read a palette image carrying alpha as color, not as index and alpha."""
    from PIL import Image

    image_path = tmp_path / "image.tif"
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = (200, 30, 90)
    rgba[:2, :, 3] = 255
    Image.fromarray(rgba, mode="RGBA").convert("PA").save(image_path)

    actual = _imread(str(image_path), _IMREAD_UNCHANGED)
    expected = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    assert actual is not None
    assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
    np.testing.assert_array_equal(actual, expected)


def test_fallback_in_memory_codec_preserves_bgr() -> None:
    """Preserve BGR channel order across an encode and decode round trip."""
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    success, encoded = _imencode(".png", image)

    assert success
    assert encoded is not None
    decoded = _imdecode(encoded, _IMREAD_COLOR)
    assert decoded is not None
    np.testing.assert_array_equal(decoded, image)


def test_fallback_imdecode_matches_opencv_for_jpeg() -> None:
    """Decode OpenCV-encoded JPEG bytes identically to cv2.imdecode."""
    image = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1]

    np.testing.assert_array_equal(
        _imdecode(encoded, _IMREAD_COLOR),
        cv2.imdecode(encoded, cv2.IMREAD_COLOR),
    )


def test_fallback_imdecode_returns_none_for_invalid_bytes() -> None:
    """Return None when decoding bytes that are not an image."""
    invalid = np.frombuffer(b"not an image", dtype=np.uint8)

    assert _imdecode(invalid, _IMREAD_COLOR) is None


def test_fallback_imencode_reports_failure_for_unknown_extension() -> None:
    """Report failure when encoding to an extension Pillow cannot handle."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    success, encoded = _imencode(".unknown", image)

    assert not success
    assert encoded is None


def _encoded_image_format(data: bytes) -> str | None:
    """Return the image format Pillow identifies in the encoded `data`."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image_format: str | None = image.format
    return image_format


@pytest.mark.parametrize(
    ("extension", "image_format"),
    [
        pytest.param(".tif", "TIFF", id="tiff"),
        pytest.param(".jp2", "JPEG2000", id="jpeg-2000"),
        pytest.param(".pgm", "PPM", id="ppm"),
    ],
)
def test_fallback_imencode_supports_every_extension_pillow_registers(
    extension: str, image_format: str
) -> None:
    """Encode to any extension Pillow registers, not only the JPEG aliases."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    success, encoded = _imencode(extension, image)

    assert success
    assert encoded is not None
    assert _encoded_image_format(encoded.tobytes()) == image_format


@pytest.mark.parametrize("image_format", ["PNG", "BMP", "TIFF"])
def test_opencv_default_save_options_are_empty_for_already_lossless_formats(
    image_format: str,
) -> None:
    """Skip writer options for formats OpenCV and Pillow both save losslessly."""
    assert _opencv_default_save_options(image_format) == {}


def _gradient_image(channels: int = 3) -> np.ndarray:
    """Return a small gradient image whose smooth gradients lossy codecs round off.

    `channels=3` returns a BGR image; `channels=1` returns a 2D grayscale image,
    exercising the single-channel path through the same encoders.
    """
    rows, columns = np.mgrid[0:48, 0:64]
    if channels == 1:
        gray: np.ndarray = ((rows + columns) * 3 % 256).astype(np.uint8)
        return gray
    channel_values = (columns * 4 % 256, rows * 5 % 256, (rows + columns) * 3 % 256)
    image: np.ndarray = np.dstack(channel_values).astype(np.uint8)
    return image


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return a BGR view, broadcasting a 2D grayscale image across 3 channels."""
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _encode_with_imwrite(directory: Path, extension: str, image: np.ndarray) -> bytes:
    """Encode `image` with the fallback `_imwrite` and return the file's bytes."""
    path = directory / f"image{extension}"
    assert _imwrite(str(path), image)
    return path.read_bytes()


def _encode_with_imencode(directory: Path, extension: str, image: np.ndarray) -> bytes:
    """Encode `image` with the fallback `_imencode` and return the encoded bytes."""
    success, encoded = _imencode(extension, image)
    assert success
    assert encoded is not None
    return encoded.tobytes()


def _jpeg_quantization_tables(data: bytes) -> dict[int, list[int]]:
    """Return the quantization tables, which a JPEG's quality setting determines."""
    import io

    from PIL import Image
    from PIL.JpegImagePlugin import JpegImageFile

    with Image.open(io.BytesIO(data)) as image:
        assert isinstance(image, JpegImageFile)
        tables: dict[int, list[int]] = dict(image.quantization)
    return tables


def _jpeg_chroma_subsampling(data: bytes) -> int:
    """Return the chroma subsampling factor, which a JPEG's quality also determines."""
    import io

    from PIL import Image, JpegImagePlugin

    with Image.open(io.BytesIO(data)) as image:
        sampling: int = JpegImagePlugin.get_sampling(image)
    return sampling


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_gradient_image(), id="color"),
        pytest.param(_gradient_image(channels=1), id="grayscale"),
    ],
)
@pytest.mark.parametrize(
    "fn_encode",
    [
        pytest.param(_encode_with_imwrite, id="imwrite"),
        pytest.param(_encode_with_imencode, id="imencode"),
    ],
)
def test_fallback_encoders_match_opencv_default_jpeg_quality(
    tmp_path: Path,
    fn_encode: Callable[[Path, str, np.ndarray], bytes],
    image: np.ndarray,
) -> None:
    """Encode JPEG at the quality OpenCV uses when no parameters are given."""
    expected = cv2.imencode(".jpg", image)[1].tobytes()

    actual = fn_encode(tmp_path, ".jpg", image)

    assert _jpeg_quantization_tables(actual) == _jpeg_quantization_tables(expected)
    assert _jpeg_chroma_subsampling(actual) == _jpeg_chroma_subsampling(expected)


def test_fallback_imencode_treats_jpe_as_jpeg() -> None:
    """Encode `.jpe` byte-for-byte like `.jpg`, the alias `_imencode` maps manually."""
    image = _gradient_image()

    success_jpe, encoded_jpe = _imencode(".jpe", image)
    success_jpg, encoded_jpg = _imencode(".jpg", image)

    assert success_jpe
    assert success_jpg
    assert encoded_jpe is not None
    np.testing.assert_array_equal(encoded_jpe, encoded_jpg)


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_gradient_image(), id="color"),
        pytest.param(_gradient_image(channels=1), id="grayscale"),
    ],
)
@pytest.mark.parametrize(
    "fn_encode",
    [
        pytest.param(_encode_with_imwrite, id="imwrite"),
        pytest.param(_encode_with_imencode, id="imencode"),
    ],
)
def test_fallback_encoders_write_webp_losslessly_like_opencv(
    tmp_path: Path,
    fn_encode: Callable[[Path, str, np.ndarray], bytes],
    image: np.ndarray,
) -> None:
    """Encode WebP losslessly, as OpenCV does when no parameters are given."""
    expected = cv2.imdecode(cv2.imencode(".webp", image)[1], cv2.IMREAD_COLOR)

    actual = fn_encode(tmp_path, ".webp", image)

    decoded = cv2.imdecode(np.frombuffer(actual, dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(decoded, _as_bgr(image))


@pytest.mark.parametrize("extension", [".JPG", ".WEBP"])
def test_fallback_imwrite_resolves_uppercase_extension_like_lowercase(
    tmp_path: Path, extension: str
) -> None:
    """Look up an uppercase file extension the same way `_imwrite` looks up lowercase.

    `_imwrite` lowercases the extension before checking Pillow's format registry; a
    caller passing an uppercase extension (e.g. `image.JPG`) must still resolve to the
    same writer options and produce identical bytes to the lowercase form.
    """
    image = _gradient_image()
    uppercase_path = tmp_path / f"image{extension}"
    lowercase_path = tmp_path / f"image{extension.lower()}"

    assert _imwrite(str(uppercase_path), image)
    assert _imwrite(str(lowercase_path), image)

    assert uppercase_path.read_bytes() == lowercase_path.read_bytes()


def test_fallback_image_io_matches_opencv_color_conversion_for_sixteen_bit(
    tmp_path: Path,
) -> None:
    """Match OpenCV color conversion when reading a sixteen-bit image."""
    sixteen_bit = np.array([[0, 12345], [54321, 65535]], dtype=np.uint16)
    sixteen_bit_path = tmp_path / "sixteen-bit.png"

    assert _imwrite(str(sixteen_bit_path), sixteen_bit)
    np.testing.assert_array_equal(
        _imread(str(sixteen_bit_path), _IMREAD_COLOR),
        cv2.imread(str(sixteen_bit_path), cv2.IMREAD_COLOR),
    )


class TestFallbackSixteenBitImageRead:
    """Decode the complete uint16 range without rounding its discarded low byte."""

    @pytest.mark.parametrize(
        ("image_format", "dtype"),
        [
            pytest.param("PNG", "<u2", id="png"),
            pytest.param("TIFF", "<u2", id="little-endian-tiff"),
            pytest.param("TIFF", ">u2", id="big-endian-tiff"),
        ],
    )
    @pytest.mark.parametrize("flags", [_IMREAD_COLOR, _IMREAD_UNCHANGED])
    @pytest.mark.parametrize("entrypoint", ["imread", "imdecode"])
    def test_matches_opencv_for_every_pixel_value(
        self,
        tmp_path: Path,
        image_format: str,
        dtype: str,
        flags: int,
        entrypoint: str,
    ) -> None:
        """Match OpenCV values while preserving unchanged storage dtype."""
        from PIL import Image

        pixels = np.arange(65536, dtype=np.uint16).reshape(256, 256).astype(dtype)
        image_path = tmp_path / f"sixteen-bit.{image_format.lower()}"
        Image.fromarray(pixels).save(image_path)
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
        with Image.open(image_path) as image:
            unchanged_dtype = np.asarray(image).dtype
        if entrypoint == "imread":
            expected = cv2.imread(str(image_path), flags)
        else:
            expected = cv2.imdecode(encoded, flags)

        if entrypoint == "imread":
            actual = _imread(str(image_path), flags)
        else:
            actual = _imdecode(encoded, flags)

        assert actual is not None
        assert expected is not None
        if flags == _IMREAD_UNCHANGED:
            assert actual.dtype == unchanged_dtype
        else:
            assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("entrypoint", ["imread", "imdecode"])
    def test_preserves_signed_integer_tiff_unchanged(
        self, tmp_path: Path, entrypoint: str
    ) -> None:
        """Leave signed 32-bit TIFF values untouched in unchanged mode."""
        from PIL import Image

        pixels = np.array([[-(2**31), -1, 0], [129, 65535, 2**31 - 1]], dtype=np.int32)
        image_path = tmp_path / "signed.tiff"
        Image.fromarray(pixels).save(image_path)
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
        if entrypoint == "imread":
            expected = cv2.imread(str(image_path), _IMREAD_UNCHANGED)
        else:
            expected = cv2.imdecode(encoded, _IMREAD_UNCHANGED)

        if entrypoint == "imread":
            actual = _imread(str(image_path), _IMREAD_UNCHANGED)
        else:
            actual = _imdecode(encoded, _IMREAD_UNCHANGED)

        assert actual is not None
        assert actual.dtype == pixels.dtype
        np.testing.assert_array_equal(actual, expected)


def _write_png_with_exif_orientation(path: Path, orientation: int) -> None:
    """Write a small asymmetric PNG whose EXIF orientation tag is `orientation`."""
    from PIL import Image

    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) * 10
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.fromarray(pixels).save(path, exif=exif.tobytes())


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(_IMREAD_COLOR, id="color"),
        pytest.param(_IMREAD_UNCHANGED, id="unchanged"),
    ],
)
def test_fallback_imread_matches_opencv_exif_orientation(
    tmp_path: Path, orientation: int, flags: int
) -> None:
    """Apply an image's EXIF orientation exactly when OpenCV applies it."""
    image_path = tmp_path / "oriented.png"
    _write_png_with_exif_orientation(image_path, orientation)

    np.testing.assert_array_equal(
        _imread(str(image_path), flags), cv2.imread(str(image_path), flags)
    )


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_fallback_imdecode_matches_opencv_exif_orientation(
    tmp_path: Path, orientation: int
) -> None:
    """Orient decoded image bytes the same way cv2.imdecode orients them."""
    image_path = tmp_path / "oriented.png"
    _write_png_with_exif_orientation(image_path, orientation)
    encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)

    np.testing.assert_array_equal(
        _imdecode(encoded, _IMREAD_COLOR), cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    )


class TestFallbackEncoderChannels:
    """Image encoders accept OpenCV's one-, three- and four-channel layouts."""

    @pytest.mark.parametrize(
        "fn_encode",
        [
            pytest.param(_encode_with_imwrite, id="imwrite"),
            pytest.param(_encode_with_imencode, id="imencode"),
        ],
    )
    @pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".jpe"])
    @pytest.mark.parametrize("alpha", [0, 255])
    def test_jpeg_discards_alpha_without_compositing(
        self,
        tmp_path: Path,
        fn_encode: Callable[[Path, str, np.ndarray], bytes],
        extension: str,
        alpha: int,
    ) -> None:
        """JPEG keeps the BGR pixels even when their alpha is fully transparent."""
        bgr = _gradient_image()
        image = np.dstack((bgr, np.full(bgr.shape[:2], alpha, dtype=np.uint8)))
        expected = cv2.imdecode(cv2.imencode(extension, image)[1], cv2.IMREAD_COLOR)

        encoded = fn_encode(tmp_path, extension, image)

        actual = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "fn_encode",
        [
            pytest.param(_encode_with_imwrite, id="imwrite"),
            pytest.param(_encode_with_imencode, id="imencode"),
        ],
    )
    @pytest.mark.parametrize("extension", [".png", ".jpg"])
    def test_single_channel_axis_encodes_as_grayscale(
        self,
        tmp_path: Path,
        fn_encode: Callable[[Path, str, np.ndarray], bytes],
        extension: str,
    ) -> None:
        """An explicit singleton channel writes the same pixels as a 2D image."""
        image = _gradient_image(channels=1)[..., np.newaxis]
        expected = cv2.imdecode(cv2.imencode(extension, image)[1], cv2.IMREAD_UNCHANGED)

        encoded = fn_encode(tmp_path, extension, image)

        actual = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        np.testing.assert_array_equal(actual, expected)
