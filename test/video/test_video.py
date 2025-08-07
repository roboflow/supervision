import numpy as np
import cv2
import os
import pytest

import supervision as sv


def _create_temp_video(path: str, width=320, height=240, fps=30, frames=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_info_and_iteration(tmp_path):
    vid_path = tmp_path / "test.mp4"
    _create_temp_video(str(vid_path))

    video = sv.Video(str(vid_path))
    info = video.info

    assert info.width == 320
    assert info.height == 240
    assert info.total_frames == 10

    frames = list(video.frames())
    assert len(frames) == 10


def test_frames_stride(tmp_path):
    vid_path = tmp_path / "test_stride.mp4"
    _create_temp_video(str(vid_path), frames=9)

    video = sv.Video(str(vid_path))
    frames = list(video.frames(stride=2))
    assert len(frames) == 5  # ceil(9/2)


def test_save_with_callback(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    _create_temp_video(str(src))

    def identity(frame, i):
        return frame

    sv.Video(str(src)).save(str(dst), callback=identity, show_progress=False)

    # confirm destination exists and metadata matches
    dst_video = sv.Video(str(dst))
    assert dst_video.info.total_frames == 10


def test_legacy_get_video_frames_generator(tmp_path):
    vid_path = tmp_path / "legacy.mp4"
    _create_temp_video(str(vid_path), frames=6)

    frames = list(sv.get_video_frames_generator(str(vid_path)))
    assert len(frames) == 6


def test_legacy_process_video(tmp_path):
    src = tmp_path / "legacy_src.mp4"
    dst = tmp_path / "legacy_dst.mp4"
    _create_temp_video(str(src), frames=4)

    sv.process_video(
        source_path=str(src),
        target_path=str(dst),
        callback=lambda f, i: f,
        show_progress=False,
    )

    assert os.path.exists(dst)
    assert sv.Video(str(dst)).info.total_frames == 4

