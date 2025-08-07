import os
import cv2
import numpy as np
import pytest

from supervision.video.core import Video
from supervision.video.dataclasses import VideoInfo


class TestVideo:
    def setup_method(self):
        self.video_path = "test.mp4"
        self.video_info = VideoInfo(width=10, height=10, fps=10, total_frames=10)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        for _ in range(self.video_info.total_frames):
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def teardown_method(self):
        os.remove(self.video_path)

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_info(self, backend):
        video = Video(self.video_path, backend=backend)
        info = video.info
        assert info.width == self.video_info.width
        assert info.height == self.video_info.height
        assert info.fps == self.video_info.fps
        assert info.total_frames == self.video_info.total_frames

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_iteration(self, backend):
        video = Video(self.video_path, backend=backend)
        frames = list(video)
        assert len(frames) == self.video_info.total_frames
        for frame in frames:
            assert frame.shape == (10, 10, 3)

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_frames_stride(self, backend):
        video = Video(self.video_path, backend=backend)
        frames = list(video.frames(stride=2))
        assert len(frames) == 5

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_frames_start_end(self, backend):
        video = Video(self.video_path, backend=backend)
        frames = list(video.frames(start=2, end=8))
        assert len(frames) == 6

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_frames_resolution(self, backend):
        video = Video(self.video_path, backend=backend)
        frames = list(video.frames(resolution_wh=(20, 20)))
        assert len(frames) == 10
        for frame in frames:
            assert frame.shape == (20, 20, 3)

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_save(self, backend):
        video = Video(self.video_path, backend=backend)
        target_path = "test_save.mp4"

        def callback(frame, i):
            return (frame * 0.5).astype(np.uint8)

        video.save(target_path, callback)
        saved_video = Video(target_path, backend=backend)
        assert saved_video.info.total_frames == self.video_info.total_frames
        os.remove(target_path)

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_save_fps(self, backend):
        video = Video(self.video_path, backend=backend)
        target_path = "test_save_fps.mp4"

        def callback(frame, i):
            return frame

        video.save(target_path, callback, fps=20)
        saved_video = Video(target_path, backend=backend)
        assert saved_video.info.fps == 20
        os.remove(target_path)

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_video_sink(self, backend):
        video = Video(self.video_path, backend=backend)
        target_path = "test_sink.mp4"
        target_info = VideoInfo(width=20, height=20, fps=10, total_frames=10)

        with video.sink(target_path, target_info) as sink:
            for frame in video.frames():
                frame = cv2.resize(frame, target_info.resolution_wh)
                sink.write(frame)

        saved_video = Video(target_path, backend=backend)
        assert saved_video.info.width == 20
        assert saved_video.info.height == 20
        os.remove(target_path)

