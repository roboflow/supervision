from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import supervision as sv


def _make_test_video(
    path: Path, *, width: int = 320, height: int = 240, fps: int = 10, frames: int = 20
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "src.mp4"
    _make_test_video(path)
    return path


def test_video_info_from_file(video_file: Path):
    v = sv.Video(str(video_file))
    info = v.info
    assert info.width == 320
    assert info.height == 240
    # FPS may be float; tolerate small differences
    assert 9.0 <= info.fps <= 61.0  # some systems re-encode at higher fps
    assert info.total_frames == 20


def test_simple_iteration_counts_frames(video_file: Path):
    v = sv.Video(str(video_file))
    count = sum(1 for _ in v)
    assert count == 20


def test_advanced_frames_stride_start_end_resize(video_file: Path):
    v = sv.Video(str(video_file))
    frames = list(v.frames(stride=5, start=2, end=18, resolution_wh=(160, 120)))
    # expected count: ceil((18-2)/5) -> ((16 + 5 - 1) // 5) = 4
    assert len(frames) == 4
    for f in frames:
        assert f.shape[1] == 160 and f.shape[0] == 120


def test_save_with_callback_and_fps_override(video_file: Path, tmp_path: Path):
    target = tmp_path / "out.mp4"

    def cb(frame: np.ndarray, i: int) -> np.ndarray:
        return cv2.GaussianBlur(frame, (9, 9), 0)

    sv.Video(str(video_file)).save(
        str(target), callback=cb, fps=60, show_progress=False
    )

    assert target.exists() and target.stat().st_size > 0
    cap = cv2.VideoCapture(str(target))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    assert 50.0 <= fps <= 70.0


def test_sink_manual_control(video_file: Path, tmp_path: Path):
    src = sv.Video(str(video_file))
    info = sv.VideoInfo(width=64, height=64, fps=24.0)
    target = tmp_path / "manual.mp4"
    with src.sink(str(target), info=info) as sink:  # type: ignore[assignment]
        for f in src.frames(end=5):
            f = cv2.resize(f, info.resolution_wh)
            sink.write(f)

    cap = cv2.VideoCapture(str(target))
    assert cap.isOpened()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (w, h) == (64, 64)


def test_deprecated_functions_still_work(video_file: Path, tmp_path: Path):
    # VideoInfo.from_video_path
    info = sv.VideoInfo.from_video_path(str(video_file))
    assert info.width == 320 and info.height == 240

    # get_video_frames_generator
    gen = sv.get_video_frames_generator(str(video_file), stride=2)
    assert sum(1 for _ in gen) == 10

    # process_video
    target = tmp_path / "legacy.mp4"

    def identity(frame: np.ndarray, i: int) -> np.ndarray:
        return frame

    sv.process_video(str(video_file), str(target), callback=identity, max_frames=5)
    assert target.exists() and target.stat().st_size > 0


def test_backend_selection_opencv(video_file: Path):
    v = sv.Video(str(video_file), backend="opencv")
    assert v.info.width == 320


@pytest.mark.skipif(
    "av"
    not in {m.split(".")[0] for m in list({*map(lambda x: x, os.sys.modules.keys())})},
    reason="PyAV not installed",
)
def test_backend_selection_pyav(video_file: Path):
    # Only run if PyAV is installed in the environment
    v = sv.Video(str(video_file), backend="pyav")
    assert v.info.width == 320
