import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from supervision.video import Video, OpenCVBackend, VideoInfo

class DummyCapture:
    def __init__(self, frames, width=640, height=480, fps=30):
        self.frames = frames
        self.width = width
        self.height = height
        self.fps = fps
        self.idx = 0
        self.opened = True
    def isOpened(self):
        return self.opened
    def get(self, prop):
        if prop == 3:  # cv2.CAP_PROP_FRAME_WIDTH
            return self.width
        if prop == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
            return self.height
        if prop == 5:  # cv2.CAP_PROP_FPS
            return self.fps
        if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
            return len(self.frames)
        if prop == 1:  # cv2.CAP_PROP_POS_FRAMES
            return self.idx
        return 0
    def set(self, prop, value):
        if prop == 1:  # cv2.CAP_PROP_POS_FRAMES
            self.idx = int(value)
    def read(self):
        if self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame
        return False, None
    def grab(self):
        if self.idx < len(self.frames):
            self.idx += 1
            return True
        return False
    def release(self):
        self.opened = False

class DummyWriter:
    def __init__(self):
        self.frames = []
        self.closed = False
    def write(self, frame):
        self.frames.append(frame)
    def release(self):
        self.closed = True

@pytest.fixture
def dummy_frames():
    return [np.ones((480, 640, 3), dtype=np.uint8) * i for i in range(10)]

@patch('supervision.video.cv2')
def test_video_info_extraction(mock_cv2, dummy_frames):
    dummy = DummyCapture(dummy_frames)
    mock_cv2.VideoCapture.return_value = dummy
    video = Video('dummy.mp4', backend='opencv')
    info = video.info
    assert info.width == 640
    assert info.height == 480
    assert info.fps == 30
    assert info.total_frames == 10

@patch('supervision.video.cv2')
def test_video_frame_iteration(mock_cv2, dummy_frames):
    dummy = DummyCapture(dummy_frames)
    mock_cv2.VideoCapture.return_value = dummy
    video = Video('dummy.mp4', backend='opencv')
    frames = list(video)
    assert len(frames) == 10
    assert np.array_equal(frames[0], dummy_frames[0])
    assert np.array_equal(frames[-1], dummy_frames[-1])

@patch('supervision.video.cv2')
def test_video_advanced_frame_iteration(mock_cv2, dummy_frames):
    dummy = DummyCapture(dummy_frames)
    mock_cv2.VideoCapture.return_value = dummy
    video = Video('dummy.mp4', backend='opencv')
    frames = list(video.frames(stride=2, start=2, end=8))
    assert len(frames) == 3  # 2,4,6
    assert np.array_equal(frames[0], dummy_frames[2])
    assert np.array_equal(frames[1], dummy_frames[4])
    assert np.array_equal(frames[2], dummy_frames[6])

@patch('supervision.video.cv2')
def test_video_save_and_sink(mock_cv2, dummy_frames):
    dummy = DummyCapture(dummy_frames)
    mock_cv2.VideoCapture.return_value = dummy
    mock_cv2.VideoWriter_fourcc.return_value = 0
    dummy_writer = DummyWriter()
    mock_cv2.VideoWriter.return_value = dummy_writer
    video = Video('dummy.mp4', backend='opencv')
    video.save('target.mp4', callback=lambda f, i: f * 2)
    assert len(dummy_writer.frames) == 10
    assert np.array_equal(dummy_writer.frames[0], dummy_frames[0] * 2)
    # Test sink context
    dummy_writer2 = DummyWriter()
    mock_cv2.VideoWriter.return_value = dummy_writer2
    with video.sink('target2.mp4') as sink:
        for f in video.frames():
            sink.write(f)
    assert len(dummy_writer2.frames) == 10

@patch('supervision.video.cv2')
def test_video_backend_selection(mock_cv2, dummy_frames):
    dummy = DummyCapture(dummy_frames)
    mock_cv2.VideoCapture.return_value = dummy
    video = Video('dummy.mp4', backend='opencv')
    assert isinstance(video.backend, OpenCVBackend)
    with pytest.raises(ValueError):
        Video('dummy.mp4', backend='unknown')

@patch('supervision.video.cv2')
def test_video_error_handling(mock_cv2):
    dummy = DummyCapture([])
    dummy.opened = False
    mock_cv2.VideoCapture.return_value = dummy
    with pytest.raises(Exception):
        Video('dummy.mp4', backend='opencv')
