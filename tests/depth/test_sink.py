from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import supervision as sv
from supervision.depth.sink import _frame_rate, _preview_codes

CLIP_RANGE = sv.DepthClipRange(display_range=(2.0, 60.0), max_value=63.0)
VIDEO_INFO = sv.VideoInfo(width=16, height=8, fps=24.0)
CAMERA = sv.DepthCamera(fx_px=500.0, baseline_m=0.1, cx_px=8.0, cy_px=4.0)


def _frames(count: int = 3) -> list[sv.DepthMap]:
    """Disparity frames with a no-depth corner and a ramp that moves each frame."""
    frames = []
    for index in range(count):
        values = np.tile(np.linspace(1, 60, 16, dtype=np.float32), (8, 1))
        values = np.roll(values, index, axis=1)
        values[:2, :2] = 0.0
        frames.append(
            sv.DepthMap(values, kind="disparity_px", camera=CAMERA, view="left")
        )
    return frames


def _decode_preview(
    path: Path,
) -> tuple[list[np.ndarray], list[float], list[bool], int]:
    """Decode a preview's luma codes, frame times, keyframes and range flag."""
    import av

    lumas, times, keyframes = [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        color_range = stream.codec_context.color_range
        height = stream.codec_context.height
        for frame in container.decode(stream):
            lumas.append(frame.to_ndarray()[:height])
            times.append(float(frame.time))
            keyframes.append(bool(frame.key_frame))
    return lumas, times, keyframes, color_range


class TestPreviewCodes:
    def test_maps_range_to_tv_codes_with_guard_band(self) -> None:
        """No depth is 16, the range runs from T + 1 to 235, outside values clamp."""
        values = np.array([[0.0, 1e-6, 31.5, 63.0, 80.0]], dtype=np.float32)
        depth_map = sv.DepthMap(values, kind="disparity_px")

        codes = _preview_codes(depth_map, (0.0, 63.0), reserved_max=31)

        assert codes.tolist() == [[16, 32, 134, 235, 235]]


class TestFrameRate:
    @pytest.mark.parametrize(
        ("fps", "expected"),
        [
            (24.0, Fraction(24)),
            (25, Fraction(25)),
            (29.97, Fraction(30000, 1001)),
            (30000 / 1001, Fraction(30000, 1001)),
            (23.976, Fraction(24000, 1001)),
            (59.94, Fraction(60000, 1001)),
            (12.5, Fraction(25, 2)),
        ],
    )
    def test_snaps_ntsc_rates(self, fps: float, expected: Fraction) -> None:
        """NTSC rates become x/1001 so preview times match the video's."""
        assert _frame_rate(fps) == expected


class TestDepthSink:
    def test_writes_exact_frames_preview_and_manifest(self, tmp_path: Path) -> None:
        """A clean exit leaves a clip manifest supervision-js accepts."""
        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE) as sink:
            for depth_map in _frames():
                sink.write_depth_map(depth_map)

        manifest = json.loads((tmp_path / "depth.json").read_text())
        assert manifest["frames"] == {"count": 3, "exact": "exact/{index:06}.png"}
        assert manifest["storage"] == {"format": "png16", "scale": 1024, "no_depth": 0}
        assert manifest["display_range_px"] == [2.0, 60.0]
        assert manifest["camera"] == {
            "fx_px": 500.0,
            "baseline_m": 0.1,
            "doffs_px": 0.0,
            "cx_px": 8.0,
            "cy_px": 4.0,
        }
        assert manifest["view"] == "left"
        assert manifest["preview"]["codec"].startswith("avc1.")
        assert {
            key: value for key, value in manifest["preview"].items() if key != "codec"
        } == {
            "file": "preview.mp4",
            "levels": "tv",
            "reserved_max": 31,
            "range_px": [0, 63.0],
        }

    def test_exact_frames_load_back_within_one_step(self, tmp_path: Path) -> None:
        """Every exact PNG holds its frame at the clip's power-of-two scale."""
        frames = _frames()
        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE) as sink:
            for depth_map in frames:
                sink.write_depth_map(depth_map)

        loaded = sv.DepthMap.load(tmp_path / "depth.json", frame_index=2)

        assert loaded.display_range == (2.0, 60.0)
        np.testing.assert_allclose(
            loaded.to_float(), frames[2].to_float(), atol=1 / 2048
        )

    def test_lossless_preview_decodes_to_the_code_formula(self, tmp_path: Path) -> None:
        """At CRF 0 the decoded luma is exactly the documented codes."""
        frames = _frames()
        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE, crf=0) as sink:
            for depth_map in frames:
                sink.write_depth_map(depth_map)

        lumas, _, _, _ = _decode_preview(tmp_path / "preview.mp4")

        for luma, depth_map in zip(lumas, frames):
            expected = _preview_codes(depth_map, (0.0, 63.0), reserved_max=31)
            np.testing.assert_array_equal(luma, expected)

    def test_preview_is_tv_range_timed_and_keyed_every_second(
        self, tmp_path: Path
    ) -> None:
        """Frame k is at k / fps, in flagged TV range, with a key frame each second."""
        video_info = sv.VideoInfo(width=16, height=8, fps=4.0)
        with sv.DepthSink(tmp_path, video_info, CLIP_RANGE) as sink:
            for depth_map in _frames(9):
                sink.write_depth_map(depth_map)

        lumas, times, keyframes, color_range = _decode_preview(tmp_path / "preview.mp4")

        assert len(lumas) == 9
        assert times == pytest.approx([index / 4 for index in range(9)])
        assert [index for index, key in enumerate(keyframes) if key] == [0, 4, 8]
        assert color_range == 1

    def test_default_crf_keeps_codes_near_exact(self, tmp_path: Path) -> None:
        """At CRF 18 decoded codes stay within a few steps and holes stay no depth."""
        frames = _frames()
        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE) as sink:
            for depth_map in frames:
                sink.write_depth_map(depth_map)

        lumas, _, _, _ = _decode_preview(tmp_path / "preview.mp4")

        expected = _preview_codes(frames[0], (0.0, 63.0), reserved_max=31)
        error = np.abs(lumas[0].astype(int) - expected.astype(int))
        assert error.max() <= 4
        assert (lumas[0][:2, :2] <= 31).all()

    def test_without_preview_writes_exact_frames_only(self, tmp_path: Path) -> None:
        """Preview=False skips the video, and metric maps are then allowed."""
        depth_map = sv.DepthMap(np.full((8, 16), 4.0, np.float32), kind="depth_m")
        clip_range = sv.DepthClipRange(display_range=(1.0, 9.0), max_value=10.0)

        with sv.DepthSink(tmp_path, VIDEO_INFO, clip_range, preview=False) as sink:
            sink.write_depth_map(depth_map)

        manifest = json.loads((tmp_path / "depth.json").read_text())
        assert "preview" not in manifest
        assert manifest["display_range"] == [1.0, 9.0]
        assert not (tmp_path / "preview.mp4").exists()

    def test_reuses_uint16_codes_at_the_clip_scale(self, tmp_path: Path) -> None:
        """Codes already at the clip's scale are written untouched."""
        codes = np.random.default_rng(2).integers(0, 64512, (8, 16), dtype=np.uint16)
        depth_map = sv.DepthMap(codes, kind="disparity_px", scale=1024)

        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE, preview=False) as sink:
            sink.write_depth_map(depth_map)

        loaded = sv.DepthMap.load(tmp_path / "depth.json", frame_index=0)
        np.testing.assert_array_equal(loaded.values, codes)

    def test_failed_block_writes_no_manifest(self, tmp_path: Path) -> None:
        """An exception inside the block leaves no depth.json behind."""

        def write_then_fail() -> None:
            """Write one frame, then fail inside the block as a model might."""
            with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE) as sink:
                sink.write_depth_map(_frames(1)[0])
                raise RuntimeError("model failed")

        with pytest.raises(RuntimeError, match="model failed"):
            write_then_fail()

        assert not (tmp_path / "depth.json").exists()

    def test_empty_clip_writes_no_manifest(self, tmp_path: Path) -> None:
        """A sink that received no frames has nothing to describe."""
        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE):
            pass

        assert not (tmp_path / "depth.json").exists()

    @pytest.mark.parametrize(
        ("depth_map", "preview", "match"),
        [
            pytest.param(
                sv.DepthMap(np.ones((8, 16), np.float32), kind="depth_m"),
                True,
                "disparity_px maps only",
                id="preview-for-depth",
            ),
            pytest.param(
                sv.DepthMap(np.ones((7, 16), np.float32), kind="disparity_px"),
                True,
                "even width and height",
                id="odd-height",
            ),
            pytest.param(
                sv.DepthMap(np.full((8, 16), 70.0, np.float32), kind="disparity_px"),
                False,
                "does not fit",
                id="above-max-value",
            ),
        ],
    )
    def test_rejects_frames_it_cannot_write(
        self, tmp_path: Path, depth_map: sv.DepthMap, preview: bool, match: str
    ) -> None:
        """Previews need even disparity maps; values must fit the clip scale."""
        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE, preview=preview) as sink:
            with pytest.raises(ValueError, match=match):
                sink.write_depth_map(depth_map)

    def test_rejects_a_frame_of_another_size(self, tmp_path: Path) -> None:
        """Every frame must match the first frame's kind and size."""
        other = sv.DepthMap(np.ones((4, 8), np.float32), kind="disparity_px")

        with sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE, preview=False) as sink:
            sink.write_depth_map(_frames(1)[0])
            with pytest.raises(ValueError, match="Frame 1"):
                sink.write_depth_map(other)

    def test_requires_an_open_context(self, tmp_path: Path) -> None:
        """Writing outside the with block is an error."""
        sink = sv.DepthSink(tmp_path, VIDEO_INFO, CLIP_RANGE)

        with pytest.raises(RuntimeError, match="open DepthSink context"):
            sink.write_depth_map(_frames(1)[0])

    @pytest.mark.parametrize(
        ("reserved_max", "crf", "match"),
        [(15, 18, "reserved_max"), (234, 18, "reserved_max"), (31, 52, "crf")],
    )
    def test_rejects_invalid_options(
        self, tmp_path: Path, reserved_max: int, crf: int, match: str
    ) -> None:
        """The guard band must fit TV range and CRF must be valid for x264."""
        with pytest.raises(ValueError, match=match):
            sv.DepthSink(
                tmp_path, VIDEO_INFO, CLIP_RANGE, crf=crf, reserved_max=reserved_max
            )
