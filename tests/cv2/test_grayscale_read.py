"""Reference checks for grayscale image decoding through the fallback."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from supervision._cv2._image import _imdecode, _imread

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


def _grayscale_fixture(mode: str) -> Image.Image:
    """Build deterministic image modes with colors and transparency to decode."""
    rng = np.random.default_rng(41)
    rgba = rng.integers(0, 256, (12, 16, 4), dtype=np.uint8)
    rgba[0, :3] = [[255, 0, 0, 0], [0, 255, 0, 127], [0, 0, 255, 255]]
    if mode == "I;16":
        return Image.fromarray(np.arange(65536, dtype=np.uint16).reshape(256, 256))
    image = Image.fromarray(rgba).convert(mode)
    if mode == "P":
        image.info["transparency"] = 0
    return image


class TestFallbackGrayscaleRead:
    """Read grayscale PNG and JPEG pixels through both decoder entrypoints."""

    @pytest.mark.parametrize("entrypoint", ["file", "bytes"])
    @pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "P", "I;16"])
    def test_png_matches_live_opencv(
        self, tmp_path: Path, entrypoint: str, mode: str
    ) -> None:
        """PNG grayscale conversion drops alpha and matches decoder rounding."""
        path = tmp_path / "image.png"
        _grayscale_fixture(mode).save(path)
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        expected = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        if entrypoint == "file":
            actual = _imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            actual = _imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        assert actual is not None
        assert actual.flags.writeable
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("entrypoint", ["file", "bytes"])
    @pytest.mark.parametrize("mode", ["RGB", "L"])
    def test_jpeg_matches_live_opencv(
        self, tmp_path: Path, entrypoint: str, mode: str
    ) -> None:
        """JPEG reads the encoded luminance instead of reconverting RGB pixels."""
        path = tmp_path / "image.jpg"
        _grayscale_fixture(mode).save(path, quality=93)
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        expected = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        if entrypoint == "file":
            actual = _imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            actual = _imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        assert actual is not None
        assert actual.flags.writeable
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("entrypoint", ["file", "bytes"])
    @pytest.mark.parametrize("extension", [".png", ".jpg"])
    @pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_applies_exif_orientation(
        self, tmp_path: Path, entrypoint: str, extension: str, orientation: int
    ) -> None:
        """Grayscale decoding applies EXIF without decoding JPEG as RGB first."""
        path = tmp_path / f"oriented{extension}"
        exif = Image.Exif()
        exif[0x0112] = orientation
        _grayscale_fixture("RGB").save(path, exif=exif.tobytes())
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        expected = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        if entrypoint == "file":
            actual = _imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            actual = _imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        assert actual is not None
        assert actual.flags.writeable
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("entrypoint", ["file", "bytes"])
    @pytest.mark.parametrize("extension", [".jpg", ".tif"])
    def test_other_color_modes_return_uint8_single_channel(
        self, tmp_path: Path, entrypoint: str, extension: str
    ) -> None:
        """CMYK decoding preserves the grayscale shape and depth contract."""
        path = tmp_path / f"cmyk{extension}"
        _grayscale_fixture("CMYK").save(path)
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        expected = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        if entrypoint == "file":
            actual = _imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            actual = _imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        assert actual is not None
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
