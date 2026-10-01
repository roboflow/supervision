from __future__ import annotations

import copy
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.depth.manifest import (
    encode_codes,
    encode_png16,
    parse_manifest,
    power_of_two_scale,
    read_pfm,
    read_png16,
    resolve_frame_file,
)

STILL_MANIFEST: dict[str, Any] = {
    "schema": "supervision.depth-manifest",
    "version": 1,
    "kind": "disparity_px",
    "view": "left",
    "width": 4,
    "height": 2,
    "storage": {"format": "png16", "scale": 256, "no_depth": 0},
    "camera": {"fx_px": 1050.3, "baseline_m": 0.12, "doffs_px": 0},
    "display_range_px": [4.2, 118.7],
    "image": {"file": "depth.png"},
}

CLIP_MANIFEST: dict[str, Any] = {
    "schema": "supervision.depth-manifest",
    "version": 1,
    "kind": "disparity_px",
    "width": 4,
    "height": 2,
    "storage": {"format": "png16", "scale": 1024, "no_depth": 0},
    "frames": {"count": 3, "exact": "exact/{index:06}.png"},
    "preview": {
        "file": "preview.mp4",
        "levels": "tv",
        "reserved_max": 31,
        "range_px": [0, 63.0],
    },
}


def _with(manifest: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Return a copy of a manifest with one dotted field set."""
    changed = copy.deepcopy(manifest)
    *parents, leaf = path.split(".")
    node = changed
    for parent in parents:
        node = node[parent]
    node[leaf] = value
    return changed


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split PNG bytes into (type, data) chunks after checking each CRC."""
    chunks = []
    position = 8
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(
            ">I", data[position + 8 + length : position + 12 + length]
        )
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        chunks.append((kind, body))
        position += 12 + length
    return chunks


class TestEncodePng16:
    def test_writes_16_bit_grayscale_with_up_filter_on_every_row(self) -> None:
        """Header is 16-bit gray, not interlaced, and every row byte 0 is filter 2."""
        codes = np.array([[0, 1, 2], [256, 65535, 7], [9, 9, 9]], dtype=np.uint16)

        data = encode_png16(codes)

        chunks = _png_chunks(data)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert [kind for kind, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
        assert struct.unpack(">IIBBBBB", chunks[0][1]) == (3, 3, 16, 0, 0, 0, 0)
        rows = np.frombuffer(zlib.decompress(chunks[1][1]), np.uint8).reshape(3, 7)
        assert rows[:, 0].tolist() == [2, 2, 2]

    def test_decodes_identically_in_pillow(self, tmp_path: Path) -> None:
        """A standard decoder reads back the exact codes, big-endian and unfiltered."""
        codes = np.random.default_rng(0).integers(0, 65536, (17, 23), dtype=np.uint16)
        (tmp_path / "depth.png").write_bytes(encode_png16(codes))

        decoded = read_png16(tmp_path / "depth.png")

        np.testing.assert_array_equal(decoded, codes)


class TestEncodeCodes:
    def test_rounds_and_keeps_tiny_values_above_no_depth(self) -> None:
        """Valid values that would round to 0 are written as 1."""
        values = np.array([[0.0, 0.001, 1.0, 2.5]], dtype=np.float32)
        valid = np.array([[False, True, True, True]])

        codes = encode_codes(values, valid, scale=256)

        assert codes.tolist() == [[0, 1, 256, 640]]


class TestPowerOfTwoScale:
    @pytest.mark.parametrize(
        ("max_value", "expected"),
        [(1.0, 32768.0), (0.0, 1.0)],
    )
    def test_picks_largest_power_of_two_that_fits(
        self, max_value: float, expected: float
    ) -> None:
        """The scale keeps max_value * scale within 65535."""
        assert power_of_two_scale(max_value) == expected


class TestReadPfm:
    @staticmethod
    def _write(path: Path, values: np.ndarray, little_endian: bool) -> None:
        """Write a grayscale PFM, rows bottom to top, in the given byte order."""
        height, width = values.shape
        scale = -1.0 if little_endian else 1.0
        dtype = "<f4" if little_endian else ">f4"
        header = f"Pf\n{width} {height}\n{scale}\n".encode()
        path.write_bytes(header + values[::-1].astype(dtype).tobytes())

    @pytest.mark.parametrize("little_endian", [True, False])
    def test_reads_rows_top_first_in_either_byte_order(
        self, tmp_path: Path, little_endian: bool
    ) -> None:
        """Rows are flipped and the scale sign picks the byte order."""
        values = np.array([[1.0, 2.0, np.inf], [3.0, 4.0, 5.5]], dtype=np.float32)
        self._write(tmp_path / "disp0.pfm", values, little_endian)

        loaded = read_pfm(tmp_path / "disp0.pfm")

        np.testing.assert_array_equal(loaded, values)

    def test_infinite_disparity_is_no_depth(self, tmp_path: Path) -> None:
        """Middlebury's +inf for unknown disparity reads as no depth."""
        values = np.array([[np.inf, 4.0]], dtype=np.float32)
        self._write(tmp_path / "disp0.pfm", values, little_endian=True)

        depth_map = sv.DepthMap.from_pfm(tmp_path / "disp0.pfm")

        assert depth_map.valid_mask.tolist() == [[False, True]]

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            pytest.param(b"PF\n1 1\n-1.0\n" + b"\0" * 12, "colour", id="colour"),
            pytest.param(b"P6\n1 1\n255\n\0\0\0", "not a PFM", id="ppm"),
            pytest.param(b"Pf\n2 2\n-1.0\n\0\0\0\0", "truncated", id="truncated"),
        ],
    )
    def test_rejects_other_files(
        self, tmp_path: Path, content: bytes, match: str
    ) -> None:
        """Only complete grayscale PFMs are depth maps."""
        (tmp_path / "file.pfm").write_bytes(content)

        with pytest.raises(ValueError, match=match):
            read_pfm(tmp_path / "file.pfm")


class TestResolveFrameFile:
    @pytest.mark.parametrize(
        ("pattern", "index", "expected"),
        [
            ("frame-{index}.png", 1234, "frame-1234.png"),
            ("{index:03}/{index}.png", 5, "005/5.png"),
        ],
    )
    def test_expands_plain_and_padded_index(
        self, pattern: str, index: int, expected: str
    ) -> None:
        """{index} is the plain number and {index:0N} pads it to N digits."""
        assert resolve_frame_file(pattern, index) == expected


class TestParseManifest:
    def test_ignores_unknown_fields(self) -> None:
        """Fields a newer producer adds do not stop the manifest from loading."""
        parsed = parse_manifest(_with(STILL_MANIFEST, "unknown", 1))

        assert parsed.width == 4

    @pytest.mark.parametrize(
        ("manifest", "message"),
        [
            pytest.param(
                _with(STILL_MANIFEST, "schema", "other"), "schema must be", id="schema"
            ),
            pytest.param(
                _with(STILL_MANIFEST, "version", 2), "version 2 is not", id="version"
            ),
            pytest.param(
                _with(STILL_MANIFEST, "kind", "depth"), "kind must be one of", id="kind"
            ),
            pytest.param(
                _with(STILL_MANIFEST, "width", 4.5),
                "width must be a positive integer",
                id="width",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "height", True),
                "height must be a positive integer",
                id="boolean-height",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "storage.format", "png8"),
                'storage.format must be "png16"',
                id="format",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "storage.scale", 0),
                "storage.scale must be a positive number",
                id="scale",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "storage.no_depth", 1),
                "storage.no_depth must be 0",
                id="no-depth",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "camera.fx_px", -1),
                "camera.fx_px must be a positive number",
                id="camera-fx",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "camera.cx_px", "640"),
                "camera.cx_px must be a finite number",
                id="camera-cx",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "display_range_px", [5, 5]),
                "display_range_px must have low < high",
                id="range-order",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "display_range_px", [5]),
                "display_range_px must be two finite numbers",
                id="range-length",
            ),
            pytest.param(
                _with(
                    _with(STILL_MANIFEST, "kind", "depth_m"), "display_range_px", [1, 2]
                ),
                "display_range_px is only valid for kind disparity_px",
                id="range-px-for-depth",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "display_range", [1, 2]),
                "display_range and display_range_px disagree",
                id="ranges-disagree",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "frames", {"count": 1, "exact": "{index}.png"}),
                "exactly one of image or frames",
                id="image-and-frames",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "image.file", ""),
                "image.file must be a non-empty file name",
                id="image-file",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "frames.count", 0),
                "frames.count must be a positive integer",
                id="frame-count",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "frames.exact", "exact/frame.png"),
                "frames.exact must contain {index}",
                id="frame-pattern",
            ),
        ],
    )
    def test_rejects_invalid_fields_naming_them(
        self, manifest: dict[str, Any], message: str
    ) -> None:
        """Each rule supervision-js enforces fails with the field's wire name."""
        with pytest.raises(ValueError, match="^depth.json: " + _escape(message)):
            parse_manifest(manifest)


def _escape(message: str) -> str:
    """Escape a literal message for `pytest.raises(match=...)`."""
    import re

    return re.escape(message)


class TestDepthMapSaveLoad:
    def test_float_map_round_trips_within_one_step(self, tmp_path: Path) -> None:
        """A float map saves with a power-of-two scale and loads within 1/scale."""
        values = np.array([[0.0, 1.0, 62.5, np.nan]], dtype=np.float32)
        depth_map = sv.DepthMap(
            values,
            kind="disparity_px",
            camera=sv.DepthCamera(fx_px=700.0, baseline_m=0.12, cx_px=1.5),
            display_range=(1.0, 50.0),
            view="left",
        )

        depth_map.save(tmp_path / "depth.json")
        loaded = sv.DepthMap.load(tmp_path / "depth.json")

        assert loaded.scale == 1024
        assert loaded.camera == depth_map.camera
        assert (loaded.display_range, loaded.view) == ((1.0, 50.0), "left")
        np.testing.assert_allclose(
            loaded.to_float(), depth_map.to_float(), atol=1 / 2048
        )

    def test_numpy_camera_parameters_round_trip(self, tmp_path: Path) -> None:
        """A camera built from NumPy scalars, as from a calibration file, saves."""
        camera = sv.DepthCamera(fx_px=np.float32(700.0), baseline_m=np.float64(0.125))
        depth_map = sv.DepthMap(
            np.ones((2, 2), np.float32), kind="depth_m", camera=camera
        )

        depth_map.save(tmp_path / "depth.json")

        assert sv.DepthMap.load(tmp_path / "depth.json").camera == camera

    def test_uint16_map_round_trips_byte_for_byte(self, tmp_path: Path) -> None:
        """Loading and saving again writes the same PNG and an equal map."""
        codes = np.random.default_rng(1).integers(0, 65536, (9, 11), dtype=np.uint16)
        sv.DepthMap(codes, kind="depth_m", scale=1000).save(tmp_path / "a.json")

        loaded = sv.DepthMap.load(tmp_path / "a.json")
        loaded.save(tmp_path / "b.json")

        assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
        assert loaded == sv.DepthMap(codes, kind="depth_m", scale=1000)

    def test_writes_snake_case_manifest_in_shared_order(self, tmp_path: Path) -> None:
        """The manifest has the fields and order supervision-js documents."""
        depth_map = sv.DepthMap(
            np.full((2, 4), 8.0, np.float32),
            kind="depth_m",
            display_range=(1.0, 9.0),
        )

        depth_map.save(tmp_path / "depth.json", scale=1000)

        manifest = json.loads((tmp_path / "depth.json").read_text())
        assert json.dumps(manifest) == json.dumps(
            {
                "schema": "supervision.depth-manifest",
                "version": 1,
                "kind": "depth_m",
                "width": 4,
                "height": 2,
                "storage": {"format": "png16", "scale": 1000, "no_depth": 0},
                "display_range": [1.0, 9.0],
                "image": {"file": "depth.png"},
            }
        )

    def test_save_refuses_values_that_overflow_the_scale(self, tmp_path: Path) -> None:
        """An explicit scale that cannot hold the map is refused."""
        depth_map = sv.DepthMap(np.full((2, 2), 300.0, np.float32), kind="disparity_px")

        with pytest.raises(ValueError, match="does not fit"):
            depth_map.save(tmp_path / "depth.json", scale=256)

    def test_save_refuses_a_manifest_path_named_like_the_png(
        self, tmp_path: Path
    ) -> None:
        """A `.png` manifest path would make the manifest overwrite its image."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="depth_m")

        with pytest.raises(ValueError, match="pass a manifest path"):
            depth_map.save(tmp_path / "depth.png")

    @pytest.mark.parametrize(
        ("manifest", "frame_index", "match"),
        [
            pytest.param(CLIP_MANIFEST, None, "pass a frame_index", id="clip-no-index"),
            pytest.param(CLIP_MANIFEST, 3, "pass a frame_index", id="clip-past-end"),
            pytest.param(STILL_MANIFEST, 0, "has no frames", id="still-with-index"),
        ],
    )
    def test_load_checks_frame_index(
        self,
        tmp_path: Path,
        manifest: dict[str, Any],
        frame_index: int | None,
        match: str,
    ) -> None:
        """Clips need an index in range; still images take none."""
        (tmp_path / "depth.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match=match):
            sv.DepthMap.load(tmp_path / "depth.json", frame_index=frame_index)

    @pytest.mark.parametrize(
        ("manifest", "frame_index", "match"),
        [
            pytest.param(
                _with(STILL_MANIFEST, "image.file", "../depth.png"),
                None,
                "image.file '../depth.png' must name a file inside",
                id="image-traversal",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "frames.exact", "/{index:06}.png"),
                0,
                "frames.exact '/000000.png' must name a file inside",
                id="frames-absolute",
            ),
        ],
    )
    def test_load_refuses_files_outside_the_manifest_folder(
        self,
        tmp_path: Path,
        manifest: dict[str, Any],
        frame_index: int | None,
        match: str,
    ) -> None:
        """A manifest cannot point the loader at a file beyond its own folder."""
        (tmp_path / "clip").mkdir()
        (tmp_path / "clip" / "depth.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match=_escape(match)):
            sv.DepthMap.load(tmp_path / "clip" / "depth.json", frame_index=frame_index)

    def test_load_rejects_png_of_another_size(self, tmp_path: Path) -> None:
        """The PNG must have the manifest's width and height."""
        (tmp_path / "depth.json").write_text(json.dumps(STILL_MANIFEST))
        Image.fromarray(np.ones((3, 3), np.uint16)).save(tmp_path / "depth.png")

        with pytest.raises(ValueError, match="manifest says 4x2"):
            sv.DepthMap.load(tmp_path / "depth.json")
