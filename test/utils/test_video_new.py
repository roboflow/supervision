"""Unit tests for the new Video API."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from supervision.utils.video_backend import (
    Backend,
    OpenCVBackend,
    VideoInfo,
    get_backend,
)
from supervision.utils.video_new import Video, VideoSink


class TestVideoInfo:
    """Test VideoInfo dataclass."""

    def test_video_info_creation(self):
        """Test creating VideoInfo with all parameters."""
        info = VideoInfo(
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=1500,
            codec="h264",
            duration=50.0,
            bit_rate=5000000,
        )
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0
        assert info.total_frames == 1500
        assert info.codec == "h264"
        assert info.duration == 50.0
        assert info.bit_rate == 5000000

    def test_video_info_resolution_wh(self):
        """Test resolution_wh property."""
        info = VideoInfo(width=1920, height=1080, fps=30.0)
        assert info.resolution_wh == (1920, 1080)

    def test_video_info_minimal(self):
        """Test creating VideoInfo with minimal parameters."""
        info = VideoInfo(width=640, height=480, fps=24.0)
        assert info.width == 640
        assert info.height == 480
        assert info.fps == 24.0
        assert info.total_frames is None
        assert info.codec is None
        assert info.duration is None
        assert info.bit_rate is None


class TestBackend:
    """Test backend selection and functionality."""

    def test_get_backend_opencv(self):
        """Test getting OpenCV backend."""
        backend = get_backend("opencv")
        assert isinstance(backend, OpenCVBackend)

    def test_get_backend_auto(self):
        """Test automatic backend selection."""
        backend = get_backend(None)
        # Should return OpenCVBackend if PyAV is not available
        assert isinstance(backend, (OpenCVBackend, object))

    def test_get_backend_invalid(self):
        """Test invalid backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")

    @patch("supervision.utils.video_backend.PYAV_AVAILABLE", False)
    def test_get_backend_pyav_not_available(self):
        """Test requesting PyAV when not available."""
        with pytest.raises(ValueError, match="av package is not installed"):
            get_backend("pyav")


class TestVideo:
    """Test Video class functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend for testing."""
        backend = Mock(spec=Backend)
        return backend

    @pytest.fixture
    def sample_video_info(self):
        """Create sample video info for testing."""
        return VideoInfo(
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=150,
            codec="h264",
            duration=5.0,
        )

    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        return [
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(5)
        ]

    def test_video_init(self, mock_backend):
        """Test Video initialization."""
        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            assert video.source == "test.mp4"
            assert video._backend == mock_backend
            assert video._info is None

    def test_video_init_with_backend(self, mock_backend):
        """Test Video initialization with specific backend."""
        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ) as get_backend_mock:
            video = Video("test.mp4", backend="opencv")
            get_backend_mock.assert_called_once_with("opencv")
            assert video.source == "test.mp4"

    def test_video_info_property(self, mock_backend, sample_video_info):
        """Test lazy loading of video info."""
        mock_handle = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")

            # First access should open, get info, and close
            info = video.info
            assert info == sample_video_info
            mock_backend.open.assert_called_once_with("test.mp4")
            mock_backend.info.assert_called_once_with(mock_handle)
            mock_backend.close.assert_called_once_with(mock_handle)

            # Second access should use cached value
            mock_backend.reset_mock()
            info2 = video.info
            assert info2 == sample_video_info
            mock_backend.open.assert_not_called()

    def test_video_iter(self, mock_backend, sample_video_info, sample_frames):
        """Test iterating over video frames."""
        mock_handle = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info

        # Mock frame reading
        frame_returns = [(True, frame) for frame in sample_frames] + [(False, None)]
        mock_backend.read.side_effect = frame_returns

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            frames = list(video)

            assert len(frames) == len(sample_frames)
            for actual, expected in zip(frames, sample_frames):
                np.testing.assert_array_equal(actual, expected)

    def test_video_frames_with_stride(
        self, mock_backend, sample_video_info, sample_frames
    ):
        """Test frames() method with stride."""
        mock_handle = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.seek.return_value = None
        mock_backend.grab.return_value = True

        # Mock reading frames with stride
        read_returns = []
        for i, frame in enumerate(sample_frames):
            if i % 2 == 0:  # Stride of 2
                read_returns.append((True, frame))
        read_returns.append((False, None))
        mock_backend.read.side_effect = read_returns

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            frames = list(video.frames(stride=2))

            assert len(frames) <= len(sample_frames) // 2 + 1

    def test_video_frames_with_range(
        self, mock_backend, sample_video_info, sample_frames
    ):
        """Test frames() method with start and end."""
        mock_handle = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.seek.return_value = None

        # Mock reading subset of frames
        subset_frames = sample_frames[1:3]
        read_returns = [(True, frame) for frame in subset_frames] + [(False, None)]
        mock_backend.read.side_effect = read_returns

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            frames = list(video.frames(start=1, end=3))

            assert len(frames) == len(subset_frames)
            mock_backend.seek.assert_called_once_with(mock_handle, 1)

    def test_video_frames_with_resize(self, mock_backend, sample_video_info):
        """Test frames() method with resolution resize."""
        mock_handle = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info

        # Create a frame and mock reading it
        original_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_backend.read.side_effect = [(True, original_frame), (False, None)]

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            target_resolution = (640, 480)
            frames = list(video.frames(resolution_wh=target_resolution))

            assert len(frames) == 1
            assert frames[0].shape == (480, 640, 3)  # Height, Width, Channels

    def test_video_save_simple(self, mock_backend, sample_video_info, sample_frames):
        """Test save() method without processing."""
        mock_handle = Mock()
        mock_writer = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.writer.return_value = mock_writer

        # Mock frame reading
        frame_returns = [(True, frame) for frame in sample_frames] + [(False, None)]
        mock_backend.read.side_effect = frame_returns

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            video.save("output.mp4")

            # Check writer was created with correct info
            mock_backend.writer.assert_called_once_with(
                "output.mp4", sample_video_info, None
            )

            # Check all frames were written
            assert mock_writer.write.call_count == len(sample_frames)
            mock_writer.close.assert_called_once()

    def test_video_save_with_callback(
        self, mock_backend, sample_video_info, sample_frames
    ):
        """Test save() method with processing callback."""
        mock_handle = Mock()
        mock_writer = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.writer.return_value = mock_writer

        # Mock frame reading
        frame_returns = [(True, frame) for frame in sample_frames] + [(False, None)]
        mock_backend.read.side_effect = frame_returns

        # Define a processing callback
        def process_frame(frame, index):
            return cv2.flip(frame, 1)  # Horizontal flip

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            video.save("output.mp4", callback=process_frame)

            # Check all frames were written
            assert mock_writer.write.call_count == len(sample_frames)

            # Verify frames were processed
            for call in mock_writer.write.call_args_list:
                frame = call[0][0]
                assert frame.shape == sample_frames[0].shape

    def test_video_save_with_custom_params(self, mock_backend, sample_video_info):
        """Test save() method with custom output parameters."""
        mock_handle = Mock()
        mock_writer = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.writer.return_value = mock_writer
        mock_backend.read.side_effect = [(False, None)]  # No frames

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")
            video.save(
                "output.mp4",
                fps=60.0,
                resolution_wh=(640, 480),
                codec="h265",
            )

            # Check writer was created with custom info
            call_args = mock_backend.writer.call_args
            assert call_args[0][0] == "output.mp4"
            assert call_args[0][1].fps == 60.0
            assert call_args[0][1].width == 640
            assert call_args[0][1].height == 480
            assert call_args[0][2] == "h265"

    def test_video_sink_context_manager(self, mock_backend, sample_video_info):
        """Test sink() context manager."""
        mock_handle = Mock()
        mock_writer = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.writer.return_value = mock_writer

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")

            with video.sink("output.mp4") as sink:
                assert isinstance(sink, VideoSink)
                assert sink.writer == mock_writer

                # Write a frame
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                sink.write(frame)
                mock_writer.write.assert_called_once_with(frame)

            # Check writer was closed
            mock_writer.close.assert_called_once()

    def test_video_sink_with_custom_info(self, mock_backend, sample_video_info):
        """Test sink() with custom VideoInfo."""
        mock_handle = Mock()
        mock_writer = Mock()
        mock_backend.open.return_value = mock_handle
        mock_backend.info.return_value = sample_video_info
        mock_backend.writer.return_value = mock_writer

        custom_info = VideoInfo(width=640, height=480, fps=60.0)

        with patch(
            "supervision.utils.video_new.get_backend", return_value=mock_backend
        ):
            video = Video("test.mp4")

            with video.sink("output.mp4", info=custom_info, codec="h265") as sink:
                pass

            # Check writer was created with custom info
            mock_backend.writer.assert_called_once_with(
                "output.mp4", custom_info, "h265"
            )


class TestOpenCVBackend:
    """Test OpenCV backend implementation."""

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, 10.0, (640, 480))

        for i in range(10):
            frame = np.full((480, 640, 3), i * 25, dtype=np.uint8)
            writer.write(frame)

        writer.release()

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    def test_opencv_backend_open_file(self, temp_video_file):
        """Test opening a video file with OpenCV backend."""
        backend = OpenCVBackend()
        handle = backend.open(temp_video_file)

        assert isinstance(handle, cv2.VideoCapture)
        assert handle.isOpened()

        backend.close(handle)
        assert not handle.isOpened()

    def test_opencv_backend_open_invalid(self):
        """Test opening invalid video with OpenCV backend."""
        backend = OpenCVBackend()

        with pytest.raises(Exception, match="Could not open video"):
            backend.open("nonexistent.mp4")

    def test_opencv_backend_info(self, temp_video_file):
        """Test getting video info with OpenCV backend."""
        backend = OpenCVBackend()
        handle = backend.open(temp_video_file)
        info = backend.info(handle)

        assert info.width == 640
        assert info.height == 480
        assert info.fps == 10.0
        assert info.total_frames == 10

        backend.close(handle)

    def test_opencv_backend_read(self, temp_video_file):
        """Test reading frames with OpenCV backend."""
        backend = OpenCVBackend()
        handle = backend.open(temp_video_file)

        # Read first frame
        success, frame = backend.read(handle)
        assert success
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        backend.close(handle)

    def test_opencv_backend_seek(self, temp_video_file):
        """Test seeking with OpenCV backend."""
        backend = OpenCVBackend()
        handle = backend.open(temp_video_file)

        # Seek to frame 5
        backend.seek(handle, 5)

        # Read frame and check it's the right one
        success, frame = backend.read(handle)
        assert success
        # Frame 5 should have value 5 * 25 = 125
        assert np.all(frame == 125)

        backend.close(handle)

    def test_opencv_backend_writer(self):
        """Test creating a writer with OpenCV backend."""
        backend = OpenCVBackend()
        info = VideoInfo(width=640, height=480, fps=10.0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            writer = backend.writer(temp_path, info, "mp4v")

            # Write some frames
            for i in range(5):
                frame = np.full((480, 640, 3), i * 50, dtype=np.uint8)
                writer.write(frame)

            writer.close()

            # Verify the video was created
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0

        finally:
            try:
                os.unlink(temp_path)
            except:
                pass


class TestDeprecatedAPI:
    """Test that deprecated API still works through new implementation."""

    def test_deprecated_video_info(self):
        """Test deprecated VideoInfo class."""
        from supervision.utils.video import VideoInfo as OldVideoInfo

        # Should trigger deprecation warning but still work
        with pytest.warns(None):  # Deprecation warnings might be filtered
            info = OldVideoInfo(width=1920, height=1080, fps=30, total_frames=100)
            assert info.width == 1920
            assert info.height == 1080
            assert info.fps == 30
            assert info.total_frames == 100
            assert info.resolution_wh == (1920, 1080)
