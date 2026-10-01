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
    """Return a copy of a manifest with one dotted field set, or removed for ...."""
    changed = copy.deepcopy(manifest)
    *parents, leaf = path.split(".")
    node = changed
    for parent in parents:
        node = node[parent]
    if value is ...:
        del node[leaf]
    else:
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

    def test_decodes_identically_in_pillow(self) -> None:
        """A standard decoder reads back the exact codes, big-endian and unfiltered."""
        codes = np.random.default_rng(0).integers(0, 65536, (17, 23), dtype=np.uint16)

        decoded = read_png16(encode_png16(codes))

        np.testing.assert_array_equal(decoded, codes)

    @pytest.mark.parametrize(
        "codes",
        [
            pytest.param(np.zeros((2, 2), np.uint8), id="uint8"),
            pytest.param(np.zeros((2, 2, 1), np.uint16), id="three-dimensional"),
        ],
    )
    def test_rejects_anything_but_2d_uint16(self, codes: np.ndarray) -> None:
        """Only 2D uint16 arrays are PNG16 depth."""
        with pytest.raises(ValueError, match="2D uint16"):
            encode_png16(codes)


class TestEncodeCodes:
    def test_rounds_and_keeps_tiny_values_above_no_depth(self) -> None:
        """Valid values that would round to 0 are written as 1."""
        values = np.array([[0.0, 0.001, 1.0, 2.5]], dtype=np.float32)
        valid = np.array([[False, True, True, True]])

        codes = encode_codes(values, valid, scale=256)

        assert codes.tolist() == [[0, 1, 256, 640]]

    def test_refuses_overflow_and_suggests_a_scale(self) -> None:
        """A value above 65535 / scale is refused, naming a scale that fits."""
        values = np.array([[300.0]], dtype=np.float32)

        with pytest.raises(ValueError, match=r"largest storable value is 255\.996"):
            encode_codes(values, np.ones((1, 1), bool), scale=256)


class TestPowerOfTwoScale:
    @pytest.mark.parametrize(
        ("max_value", "expected"),
        [(63.0, 1024.0), (255.9, 256.0), (1.0, 32768.0), (65535.0, 1.0), (0.0, 1.0)],
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
            ("exact/{index:06}.png", 7, "exact/000007.png"),
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
    @pytest.mark.parametrize(
        "manifest",
        [
            pytest.param(STILL_MANIFEST, id="still"),
            pytest.param(CLIP_MANIFEST, id="clip"),
            pytest.param(
                _with(CLIP_MANIFEST, "preview.levels", ...), id="preview-full-range"
            ),
            pytest.param(_with(STILL_MANIFEST, "unknown", 1), id="unknown-field"),
        ],
    )
    def test_accepts_valid_manifests(self, manifest: dict[str, Any]) -> None:
        """Still and clip manifests, old full-range previews and new fields pass."""
        parsed = parse_manifest(manifest)

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
            pytest.param(
                _with(CLIP_MANIFEST, "frames.times_s", [0.0, 0.1]),
                "frames.times_s must be an array of frames.count",
                id="times-length",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "frames.times_s", [0.0, 0.2, 0.1]),
                "frames.times_s must strictly increase at index 2",
                id="times-order",
            ),
            pytest.param(
                _with(STILL_MANIFEST, "preview", CLIP_MANIFEST["preview"]),
                "preview is only valid next to frames",
                id="preview-on-still",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "kind", "depth_m"),
                "preview is only supported for kind disparity_px",
                id="preview-for-depth",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "preview.levels", "pc"),
                "preview.levels must be one of full, tv",
                id="preview-levels",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "preview.reserved_max", 15),
                "preview.reserved_max must be an integer from 16 to 233 at tv",
                id="reserved-below-tv-black",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "preview.range_px", ...),
                "preview.range_px is required",
                id="preview-range",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "preview.file", "../preview.mp4"),
                "preview.file must be a relative path inside",
                id="preview-file-traversal",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "preview.file", "/preview.mp4"),
                "preview.file must be a relative path inside",
                id="preview-file-absolute",
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

    def test_uint16_map_round_trips_byte_for_byte(self, tmp_path: Path) -> None:
        """Loading and saving again writes the same PNG and an equal map."""
        codes = np.random.default_rng(1).integers(0, 65536, (9, 11), dtype=np.uint16)
        sv.DepthMap(codes, kind="depth_m", scale=1000).save(tmp_path / "a.json")

        loaded = sv.DepthMap.load(tmp_path / "a.json")
        loaded.save(tmp_path / "b.json")

        assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
        assert sv.DepthMap.load(tmp_path / "b.json") == loaded

    def test_writes_snake_case_manifest_in_shared_order(self, tmp_path: Path) -> None:
        """The manifest has the fields and order supervision-js documents."""
        depth_map = sv.DepthMap(
            np.full((2, 4), 8.0, np.float32),
            kind="depth_m",
            display_range=(1.0, 9.0),
        )

        depth_map.save(tmp_path / "depth.json", scale=1000)

        manifest = json.loads((tmp_path / "depth.json").read_text())
        assert manifest == {
            "schema": "supervision.depth-manifest",
            "version": 1,
            "kind": "depth_m",
            "width": 4,
            "height": 2,
            "storage": {"format": "png16", "scale": 1000, "no_depth": 0},
            "display_range": [1.0, 9.0],
            "image": {"file": "depth.png"},
        }

    def test_save_refuses_values_that_overflow_the_scale(self, tmp_path: Path) -> None:
        """An explicit scale that cannot hold the map is refused."""
        depth_map = sv.DepthMap(np.full((2, 2), 300.0, np.float32), kind="disparity_px")

        with pytest.raises(ValueError, match="does not fit"):
            depth_map.save(tmp_path / "depth.json", scale=256)

    def test_loads_a_clip_frame(self, tmp_path: Path) -> None:
        """A clip manifest loads the frame its pattern names."""
        (tmp_path / "exact").mkdir()
        for index in range(3):
            codes = np.full((2, 4), index + 1, dtype=np.uint16)
            Image.fromarray(codes).save(tmp_path / f"exact/{index:06}.png")
        (tmp_path / "depth.json").write_text(json.dumps(CLIP_MANIFEST))

        frame = sv.DepthMap.load(tmp_path / "depth.json", frame_index=2)

        assert frame.values.tolist() == [[3, 3, 3, 3], [3, 3, 3, 3]]

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
                _with(STILL_MANIFEST, "image.file", "/depth.png"),
                None,
                "image.file '/depth.png' must name a file inside",
                id="image-absolute",
            ),
            pytest.param(
                _with(CLIP_MANIFEST, "frames.exact", "../{index:06}.png"),
                0,
                "frames.exact '../000000.png' must name a file inside",
                id="frames-traversal",
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
