"""Tests for private image-operation and I/O fallbacks."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from supervision._cv2._image import (
    _add_weighted,
    _convert_scale_abs,
    _copy_make_border,
    _flip,
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


def test_fallback_mean_matches_opencv() -> None:
    """Match OpenCV masked mean semantics."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    assert _mean(source, mask) == cv2.mean(source, mask)


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
        return image.format


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
        return ((rows + columns) * 3 % 256).astype(np.uint8)
    channel_values = (columns * 4 % 256, rows * 5 % 256, (rows + columns) * 3 % 256)
    return np.dstack(channel_values).astype(np.uint8)


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

    with Image.open(io.BytesIO(data)) as image:
        return dict(image.quantization)


def _jpeg_chroma_subsampling(data: bytes) -> int:
    """Return the chroma subsampling factor, which a JPEG's quality also determines."""
    import io

    from PIL import Image, JpegImagePlugin

    with Image.open(io.BytesIO(data)) as image:
        return JpegImagePlugin.get_sampling(image)


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_gradient_image(), id="color"),
        pytest.param(_gradient_image(channels=1), id="grayscale"),
    ],
)
@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(_encode_with_imwrite, id="imwrite"),
        pytest.param(_encode_with_imencode, id="imencode"),
    ],
)
def test_fallback_encoders_match_opencv_default_jpeg_quality(
    tmp_path: Path,
    encode: Callable[[Path, str, np.ndarray], bytes],
    image: np.ndarray,
) -> None:
    """Encode JPEG at the quality OpenCV uses when no parameters are given."""
    expected = cv2.imencode(".jpg", image)[1].tobytes()

    actual = encode(tmp_path, ".jpg", image)

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
    "encode",
    [
        pytest.param(_encode_with_imwrite, id="imwrite"),
        pytest.param(_encode_with_imencode, id="imencode"),
    ],
)
def test_fallback_encoders_write_webp_losslessly_like_opencv(
    tmp_path: Path,
    encode: Callable[[Path, str, np.ndarray], bytes],
    image: np.ndarray,
) -> None:
    """Encode WebP losslessly, as OpenCV does when no parameters are given."""
    expected = cv2.imdecode(cv2.imencode(".webp", image)[1], cv2.IMREAD_COLOR)

    actual = encode(tmp_path, ".webp", image)

    decoded = cv2.imdecode(np.frombuffer(actual, dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(decoded, _as_bgr(image))


@pytest.mark.parametrize("extension", [".JPG", ".WEBP"])
def test_fallback_imwrite_resolves_uppercase_extension_like_lowercase(
    tmp_path: Path, extension: str
) -> None:
    """Look up an uppercase file extension the same way `_imwrite` looks up lowercase.

    `_imwrite` lowercases the extension before checking Pillow's format registry;
    a caller passing an uppercase extension (e.g. `image.JPG`) must still resolve to
    the same writer options and produce identical bytes to the lowercase form.
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
