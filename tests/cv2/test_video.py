"""Tests for the PyAV-backed video compatibility surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pytest

from supervision import _cv2
from supervision._cv2._video import (
    _mux_audio,
    _VideoCapture,
    _VideoWriter,
)


def _write_video(path: Path, values: list[int], fps: int = 5) -> None:
    """Write a small deterministic MPEG-4 video for fallback tests."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    try:
        for value in values:
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _write_video_with_audio(path: Path, frame_count: int = 5, fps: int = 5) -> None:
    """Write a short video with one AAC audio stream for remux tests."""
    container = av.open(str(path), mode="w")
    video_stream = container.add_stream("mpeg4", rate=fps)
    video_stream.width = 16
    video_stream.height = 16
    video_stream.pix_fmt = "yuv420p"
    audio_stream = container.add_stream("aac", rate=8_000)
    audio_stream.layout = "mono"
    try:
        for value in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value * 20, dtype=np.uint8), format="bgr24"
            )
            for packet in video_stream.encode(frame):
                container.mux(packet)

        samples = np.zeros((1, 8_000), dtype=np.int16)
        audio_frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        audio_frame.sample_rate = 8_000
        audio_frame.pts = 0
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _run_without_opencv(source: str) -> None:
    """Run a Python snippet with cv2 imports blocked."""
    env = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, env.get("PYTHONPATH")))
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_video_module_uses_required_pyav_dependency() -> None:
    """The video fallback imports required PyAV directly without a loader."""
    from supervision._cv2 import _video

    assert _video.av.__name__ == "av"
    assert not hasattr(_video, "_load_av")


def test_fallback_capture_reports_metadata_and_supports_exact_seek(
    tmp_path: Path,
) -> None:
    """Fallback capture exposes metadata and starts decoding at requested frames."""
    source_path = tmp_path / "source.mp4"
    _write_video(source_path, [0, 40, 80, 120, 160])

    capture = _VideoCapture(str(source_path))
    assert capture.isOpened()
    assert capture.get(_cv2.CAP_PROP_FRAME_WIDTH) == 16
    assert capture.get(_cv2.CAP_PROP_FRAME_HEIGHT) == 16
    assert capture.get(_cv2.CAP_PROP_FPS) == pytest.approx(5.0)
    assert capture.get(_cv2.CAP_PROP_FRAME_COUNT) == 5

    assert capture.set(_cv2.CAP_PROP_POS_FRAMES, 3)
    success, frame = capture.read()
    capture.release()

    assert success
    assert frame is not None
    assert frame.shape == (16, 16, 3)
    assert float(frame.mean()) == pytest.approx(120.0, abs=20.0)
    assert not capture.isOpened()


def test_fallback_writer_default_codec_round_trips(tmp_path: Path) -> None:
    """The guaranteed mp4v fallback writer creates a readable video."""
    target_path = tmp_path / "target.mp4"
    fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
    writer = _VideoWriter(str(target_path), fourcc, 5.0, (16, 16))

    assert writer.isOpened()
    for value in [0, 40, 80]:
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    capture = _VideoCapture(str(target_path))
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        assert frame is not None
        frames.append(frame)
    capture.release()

    assert len(frames) == 3
    assert target_path.stat().st_size > 0


def test_fallback_video_works_when_opencv_is_blocked(tmp_path: Path) -> None:
    """Production video APIs use PyAV when cv2 cannot be imported."""
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    _write_video(source_path, [0, 40, 80])
    source = f"""
import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for test")
        return None


sys.meta_path.insert(0, BlockCv2())
from supervision import _cv2
from supervision.utils.video import VideoInfo, VideoSink, get_video_frames_generator

assert _cv2._IS_CV2_AVAILABLE is False
info = VideoInfo.from_video_path({str(source_path)!r})
assert (info.width, info.height, info.total_frames) == (16, 16, 3)
frames = list(get_video_frames_generator({str(source_path)!r}, start=1, end=3))
assert len(frames) == 2
with VideoSink({str(target_path)!r}, info) as sink:
    for frame in frames:
        sink.write_frame(frame)
assert _cv2.VideoCapture({str(target_path)!r}).isOpened()
"""
    _run_without_opencv(source)


def test_mux_audio_remuxes_first_audio_stream_and_truncates_to_video(
    tmp_path: Path,
) -> None:
    """Audio remuxing uses PyAV and keeps the processed video duration."""
    source_path = tmp_path / "source_with_audio.mp4"
    target_path = tmp_path / "target.mp4"
    _write_video_with_audio(source_path, frame_count=5)
    _write_video(target_path, [0, 40, 80], fps=5)

    _mux_audio(str(source_path), str(target_path))

    output = av.open(str(target_path))
    try:
        assert len(output.streams.video) == 1
        assert len(output.streams.audio) == 1
        assert output.streams.video[0].frames == 3
    finally:
        output.close()


def test_mux_audio_leaves_target_unchanged_on_failure(tmp_path: Path) -> None:
    """A failed PyAV remux never replaces the existing target file."""
    target_path = tmp_path / "target.mp4"
    _write_video(target_path, [0, 40, 80])
    original = target_path.read_bytes()

    _mux_audio(str(tmp_path / "missing.mp4"), str(target_path))

    assert target_path.read_bytes() == original
