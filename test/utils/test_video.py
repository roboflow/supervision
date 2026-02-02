import os

import cv2
import numpy as np
import pytest

from supervision.utils.video import VideoInfo, get_video_frames_generator, process_video


@pytest.fixture
def dummy_video_path(tmp_path):
    path = str(tmp_path / "dummy_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25, (640, 480))
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_process_video_exception_handling(dummy_video_path, tmp_path):
    target_path = str(tmp_path / "target.mp4")

    def callback_with_exception(frame, index):
        if index == 5:
            raise ValueError("Test exception at frame 5")
        return frame

    with pytest.raises(ValueError, match="Test exception at frame 5"):
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=callback_with_exception,
        )


def test_process_video_success(dummy_video_path, tmp_path):
    target_path = str(tmp_path / "target_success.mp4")

    def callback_success(frame, index):
        return frame

    # This should complete without exception
    process_video(
        source_path=dummy_video_path, target_path=target_path, callback=callback_success
    )

    assert os.path.exists(target_path)


def test_process_video_exception_with_small_buffer(dummy_video_path, tmp_path):
    target_path = str(tmp_path / "target_exception_small_buffer.mp4")

    def callback_with_exception(frame, index):
        if index == 5:
            raise ValueError("Test exception at frame 5")
        return frame

    with pytest.raises(ValueError, match="Test exception at frame 5"):
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=callback_with_exception,
            prefetch=1,
            writer_buffer=1,
        )


def test_process_video_max_frames(dummy_video_path, tmp_path):
    target_path = str(tmp_path / "target_max_frames.mp4")
    processed_indices = []

    def callback(frame, index):
        processed_indices.append(index)
        return frame

    process_video(
        source_path=dummy_video_path,
        target_path=target_path,
        callback=callback,
        max_frames=5,
    )

    assert len(processed_indices) == 5
    assert processed_indices == [0, 1, 2, 3, 4]


def test_process_video_custom_params(dummy_video_path, tmp_path):
    target_path = str(tmp_path / "target_custom_params.mp4")

    def callback(frame, index):
        return frame

    # Test with very small prefetch and writer_buffer
    process_video(
        source_path=dummy_video_path,
        target_path=target_path,
        callback=callback,
        prefetch=1,
        writer_buffer=1,
    )

    assert os.path.exists(target_path)


def test_video_info(dummy_video_path):
    video_info = VideoInfo.from_video_path(dummy_video_path)
    assert video_info.width == 640
    assert video_info.height == 480
    assert video_info.fps == 25
    assert video_info.total_frames == 10
    assert video_info.resolution_wh == (640, 480)


def test_get_video_frames_generator(dummy_video_path):
    generator = get_video_frames_generator(dummy_video_path)
    frames = list(generator)
    assert len(frames) == 10
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(frame.shape == (480, 640, 3) for frame in frames)


def test_get_video_frames_generator_with_stride(dummy_video_path):
    generator = get_video_frames_generator(dummy_video_path, stride=2)
    frames = list(generator)
    assert len(frames) == 5


def test_get_video_frames_generator_with_start_end(dummy_video_path):
    generator = get_video_frames_generator(dummy_video_path, start=2, end=5)
    frames = list(generator)
    assert len(frames) == 3
