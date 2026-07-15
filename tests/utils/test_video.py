import os
import shutil
from pathlib import Path
from queue import Empty, Full
from queue import Queue as StdQueue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from supervision import _cv2 as cv2
from supervision.utils.video import (
    FPSMonitor,
    VideoInfo,
    _mux_audio,
    get_video_frames_generator,
    process_video,
)


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


def test_process_video_exception_handling(dummy_video_path, tmp_path) -> None:
    """
    Verify that process_video correctly propagates exceptions from the callback.

    Scenario: Processing a video where the callback raises an exception.
    Expected: `process_video` should propagate the exception, allowing users to
    handle errors during video processing.
    """
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


def test_process_video_success(dummy_video_path, tmp_path) -> None:
    """
    Verify successful video processing with a pass-through callback.

    Scenario: Successfully processing a video with a simple pass-through callback.
    Expected: The video is processed without error and the target file is created,
    verifying the core functionality of `process_video`.
    """
    target_path = str(tmp_path / "target_success.mp4")

    def callback_success(frame, index):
        return frame

    # This should complete without exception
    process_video(
        source_path=dummy_video_path, target_path=target_path, callback=callback_success
    )

    assert os.path.exists(target_path)


def test_process_video_exception_with_small_buffer(dummy_video_path, tmp_path) -> None:
    """
    Verify that process_video handles exceptions correctly even with small buffers.

    Scenario: Processing a video with minimal buffering where an exception occurs.
    Expected: The exception is still correctly propagated even with low memory settings.
    """
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


def test_process_video_enqueues_writer_sentinel_with_timeout(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """process_video enqueues the writer sentinel and bounded worker joins."""
    read_queue = StdQueue()
    read_queue.put((0, np.zeros((2, 2, 3), dtype=np.uint8)))
    read_queue.put((1, np.zeros((2, 2, 3), dtype=np.uint8)))
    read_queue.put(None)

    class RecordingWriteQueue:
        """Record writer queue puts so the shutdown path can be asserted."""

        def __init__(self) -> None:
            """Initialize the queue call log."""
            self.put_calls: list[tuple[object, object]] = []

        def put(self, item: object, timeout: object | None = None) -> None:
            """Record each put call and its timeout."""
            self.put_calls.append((item, timeout))

        def get(self, timeout: object | None = None) -> object:
            """The writer thread is disabled, so reads are not expected."""
            raise AssertionError("writer queue should not be read in this test")

    join_calls: list[object | None] = []

    class FakeThread:
        """Thread stand-in that keeps the test single-threaded."""

        def __init__(
            self,
            target: object,
            args: tuple[object, ...] = (),
            daemon: bool = False,
        ) -> None:
            """Store the thread target without starting it."""
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            """Do nothing; the test preloads the queues instead."""

        def join(self, timeout=None) -> None:
            """Do nothing; the worker targets are intentionally never started."""
            join_calls.append(timeout)

    class FakeVideoSink:
        """Minimal sink context manager used to verify shutdown ordering."""

        def __init__(self, target_path: str, video_info: object) -> None:
            """Store constructor arguments for completeness."""
            self.target_path = target_path
            self.video_info = video_info

        def __enter__(self) -> "FakeVideoSink":
            """Return the sink context manager."""
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            """Propagate any exception without side effects."""
            return None

        def write_frame(self, frame: object) -> None:
            """The writer thread is disabled in this test."""

    write_queue = RecordingWriteQueue()
    queue_factory_calls = iter([read_queue, write_queue])

    monkeypatch.setattr(
        "supervision.utils.video.Queue",
        lambda *args, **kwargs: next(queue_factory_calls),
    )
    monkeypatch.setattr("supervision.utils.video.threading.Thread", FakeThread)
    monkeypatch.setattr("supervision.utils.video.VideoSink", FakeVideoSink)
    monkeypatch.setattr(
        "supervision.utils.video.VideoInfo.from_video_path",
        lambda video_path: SimpleNamespace(total_frames=2),
    )

    target_path = str(tmp_path / "target_sentinel.mp4")

    def callback(frame, index):
        if index == 1:
            raise ValueError("Test exception at frame 1")
        return frame

    with pytest.raises(ValueError, match="Test exception at frame 1"):
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=callback,
            show_progress=False,
        )

    assert write_queue.put_calls[-1] == (None, 1)
    assert all(timeout is None for _item, timeout in write_queue.put_calls[:-1])
    assert join_calls == [10, 10]


def test_process_video_best_effort_sentinel_handles_full_queue(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """process_video should not hang if the writer queue is already full."""
    read_queue = StdQueue()
    read_queue.put((0, np.zeros((2, 2, 3), dtype=np.uint8)))
    read_queue.put(None)

    class FullWriteQueue:
        """Record writer queue puts and fail the shutdown sentinel."""

        def __init__(self) -> None:
            """Initialize the queue call log."""
            self.put_calls: list[tuple[object, object | None]] = []

        def put(self, item: object, timeout: object | None = None) -> None:
            """Record the put and raise Full for the shutdown sentinel."""
            self.put_calls.append((item, timeout))
            if item is None:
                raise Full

        def get(self, timeout: object | None = None) -> object:
            """The writer thread is disabled, so reads are not expected."""
            raise AssertionError("writer queue should not be read in this test")

    join_calls: list[object | None] = []

    class FakeThread:
        """Thread stand-in that keeps the test single-threaded."""

        def __init__(
            self,
            target: object,
            args: tuple[object, ...] = (),
            daemon: bool = False,
        ) -> None:
            """Store the thread target without starting it."""
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            """Do nothing; the test preloads the queues instead."""

        def join(self, timeout=None) -> None:
            """Record join timeouts for shutdown verification."""
            join_calls.append(timeout)

    class FakeVideoSink:
        """Minimal sink context manager used to verify shutdown ordering."""

        def __init__(self, target_path: str, video_info: object) -> None:
            """Store constructor arguments for completeness."""
            self.target_path = target_path
            self.video_info = video_info

        def __enter__(self) -> "FakeVideoSink":
            """Return the sink context manager."""
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            """Propagate any exception without side effects."""
            return None

        def write_frame(self, frame: object) -> None:
            """The writer thread is disabled in this test."""

    write_queue = FullWriteQueue()
    queue_factory_calls = iter([read_queue, write_queue])

    monkeypatch.setattr(
        "supervision.utils.video.Queue",
        lambda *args, **kwargs: next(queue_factory_calls),
    )
    monkeypatch.setattr("supervision.utils.video.threading.Thread", FakeThread)
    monkeypatch.setattr("supervision.utils.video.VideoSink", FakeVideoSink)
    monkeypatch.setattr(
        "supervision.utils.video.VideoInfo.from_video_path",
        lambda video_path: SimpleNamespace(total_frames=1),
    )

    target_path = str(tmp_path / "target_full_queue.mp4")

    def callback(frame, index):
        raise ValueError("Test exception at frame 0")

    with pytest.raises(ValueError, match="Test exception at frame 0"):
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=callback,
            show_progress=False,
        )

    assert write_queue.put_calls[-1] == (None, 1)
    assert join_calls == [10, 10]


def test_process_video_waits_for_reader_timeout_when_queue_is_empty(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """process_video should keep waiting briefly when the reader queue times out."""

    class TimeoutReadQueue:
        """Record the first frame read and then time out in shutdown."""

        def __init__(self) -> None:
            """Initialize the queue call log."""
            self.get_calls: list[object | None] = []

        def put(self, item: object, timeout: object | None = None) -> None:
            """The reader thread is disabled, so writes are not expected."""
            raise AssertionError("reader queue should not be written in this test")

        def get(self, timeout: object | None = None) -> object:
            """Yield one frame during processing, then time out during shutdown."""
            self.get_calls.append(timeout)
            if timeout is None:
                return (0, np.zeros((2, 2, 3), dtype=np.uint8))
            raise Empty

    class RecordingWriteQueue:
        """Record writer queue puts so the shutdown path can be asserted."""

        def __init__(self) -> None:
            """Initialize the queue call log."""
            self.put_calls: list[tuple[object, object | None]] = []

        def put(self, item: object, timeout: object | None = None) -> None:
            """Record each put call and its timeout."""
            self.put_calls.append((item, timeout))

        def get(self, timeout: object | None = None) -> object:
            """The writer thread is disabled, so reads are not expected."""
            raise AssertionError("writer queue should not be read in this test")

    join_calls: list[object | None] = []
    reader_alive_states = iter([True, False])

    class FakeThread:
        """Thread stand-in that keeps the test single-threaded."""

        def __init__(
            self,
            target: object,
            args: tuple[object, ...] = (),
            daemon: bool = False,
        ) -> None:
            """Store the thread target without starting it."""
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            """Do nothing; the test preloads the queues instead."""

        def join(self, timeout=None) -> None:
            """Record join timeouts for shutdown verification."""
            join_calls.append(timeout)

        def is_alive(self) -> bool:
            """Return a short-lived alive state so the timeout branch is hit."""
            return next(reader_alive_states, False)

    class FakeVideoSink:
        """Minimal sink context manager used to verify shutdown ordering."""

        def __init__(self, target_path: str, video_info: object) -> None:
            """Store constructor arguments for completeness."""
            self.target_path = target_path
            self.video_info = video_info

        def __enter__(self) -> "FakeVideoSink":
            """Return the sink context manager."""
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            """Propagate any exception without side effects."""
            return None

        def write_frame(self, frame: object) -> None:
            """The writer thread is disabled in this test."""

    read_queue = TimeoutReadQueue()
    write_queue = RecordingWriteQueue()
    queue_factory_calls = iter([read_queue, write_queue])

    monkeypatch.setattr(
        "supervision.utils.video.Queue",
        lambda *args, **kwargs: next(queue_factory_calls),
    )
    monkeypatch.setattr("supervision.utils.video.threading.Thread", FakeThread)
    monkeypatch.setattr("supervision.utils.video.VideoSink", FakeVideoSink)
    monkeypatch.setattr(
        "supervision.utils.video.VideoInfo.from_video_path",
        lambda video_path: SimpleNamespace(total_frames=1),
    )

    target_path = str(tmp_path / "target_timeout.mp4")

    def callback(frame, index):
        raise ValueError("Test exception at frame 0")

    with pytest.raises(ValueError, match="Test exception at frame 0"):
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=callback,
            show_progress=False,
        )

    assert read_queue.get_calls == [None, 1, 1]
    assert join_calls == [10, 10]


def test_process_video_max_frames(dummy_video_path, tmp_path) -> None:
    """
    Verify that process_video respects the max_frames parameter.

    Scenario: Processing only a limited number of frames using `max_frames`.
    Expected: Only the specified number of frames are processed, which is useful for
    quick testing or sampling.
    """
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


def test_process_video_custom_params(dummy_video_path, tmp_path) -> None:
    """
    Verify that process_video works correctly with custom performance parameters.

    Scenario: Processing video with custom prefetch and buffer parameters.
    Expected: Video is processed successfully, showing that these performance-tuning
    parameters are correctly handled.
    """
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


def test_video_info(dummy_video_path) -> None:
    """
    Verify that VideoInfo correctly retrieves metadata from a video file.

    Scenario: Retrieving metadata from a video file using `VideoInfo`.
    Expected: Correct width, height, fps, and frame count are returned, which is
    essential for initializing annotators or calculating statistics.
    """
    video_info = VideoInfo.from_video_path(dummy_video_path)
    assert video_info.width == 640
    assert video_info.height == 480
    assert video_info.fps == pytest.approx(25.0)
    assert isinstance(video_info.fps, float)
    assert video_info.total_frames == 10
    assert video_info.resolution_wh == (640, 480)


def test_video_info_float_fps(dummy_video_path, monkeypatch) -> None:
    """
    Verify that VideoInfo preserves non-integer FPS values as floats.

    Scenario: Retrieving metadata from a video while OpenCV reports 23.976 fps.
    Expected: fps is returned as the original float value, not truncated to an
    integer. This prevents frame-timing drift in long videos.
    """
    original_get = cv2.VideoCapture.get

    def mocked_get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 23.976
        return original_get(self, prop_id)

    monkeypatch.setattr(cv2.VideoCapture, "get", mocked_get)

    video_info = VideoInfo.from_video_path(dummy_video_path)
    assert isinstance(video_info.fps, float)
    assert video_info.fps == pytest.approx(23.976)
    assert video_info.fps != int(video_info.fps)


def test_get_video_frames_generator(dummy_video_path) -> None:
    """
    Verify that get_video_frames_generator yields frames with correct shapes.

    Scenario: Iterating over video frames using a generator.
    Expected: All frames are yielded in order as NumPy arrays with correct shapes,
    enabling frame-by-frame processing loops.
    """
    generator = get_video_frames_generator(dummy_video_path)
    frames = list(generator)
    assert len(frames) == 10
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(frame.shape == (480, 640, 3) for frame in frames)


def test_get_video_frames_generator_releases_on_early_break(monkeypatch) -> None:
    """
    Verify that the capture is released when a consumer breaks out early.

    Scenario: A consumer iterates one frame then abandons the generator, raising
    GeneratorExit at the yield point.
    Expected: The `try/finally` guard still calls `release()`, avoiding a decoder
    leak.
    """

    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def grab(self):
            return True

        def release(self) -> None:
            self.released = True

    fake_capture = FakeCapture()
    monkeypatch.setattr(
        "supervision.utils.video._validate_and_setup_video",
        lambda *args, **kwargs: (fake_capture, 0, 100),
    )

    generator = get_video_frames_generator("dummy")
    next(generator)
    generator.close()

    assert fake_capture.released


def test_get_video_frames_generator_with_stride(dummy_video_path) -> None:
    """
    Verify that get_video_frames_generator correctly handles the stride parameter.

    Scenario: Iterating over video frames with specified stride (e.g., every 2nd frame).
    Expected: The generator correctly skips frames according to the stride, allowing
    for faster processing of high-FPS videos.
    """
    generator = get_video_frames_generator(dummy_video_path, stride=2)
    frames = list(generator)
    assert len(frames) == 5


def test_fps_monitor_uses_frame_intervals(monkeypatch) -> None:
    """FPSMonitor must divide elapsed time by intervals, not sample count."""
    timestamps = iter([0.0, 0.5, 1.0])
    monkeypatch.setattr(
        "supervision.utils.video.time.monotonic", lambda: next(timestamps)
    )

    fps_monitor = FPSMonitor()
    fps_monitor.tick()
    fps_monitor.tick()
    fps_monitor.tick()

    assert fps_monitor.fps == pytest.approx(2.0)


def test_process_video_preserve_audio_calls_mux(dummy_video_path, tmp_path) -> None:
    """
    Verify that process_video calls _mux_audio when preserve_audio=True.

    Scenario: Processing a video with preserve_audio=True and ffmpeg available.
    Expected: _mux_audio is called exactly once with the correct source and target
    paths, confirming the audio muxing step is triggered after frame writing completes.
    """
    target_path = str(tmp_path / "target_audio.mp4")

    with patch("supervision.utils.video._mux_audio") as mock_mux:
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=lambda frame, idx: frame,
            preserve_audio=True,
        )
        mock_mux.assert_called_once_with(
            source_path=dummy_video_path, video_path=target_path
        )


def test_process_video_no_audio_by_default(dummy_video_path, tmp_path) -> None:
    """
    Verify that process_video does not call _mux_audio when preserve_audio=False.

    Scenario: Default process_video call without setting preserve_audio.
    Expected: _mux_audio is never called, preserving existing behavior for callers
    that do not need audio.
    """
    target_path = str(tmp_path / "target_no_audio.mp4")

    with patch("supervision.utils.video._mux_audio") as mock_mux:
        process_video(
            source_path=dummy_video_path,
            target_path=target_path,
            callback=lambda frame, idx: frame,
        )
        mock_mux.assert_not_called()


@pytest.mark.parametrize(
    ("which_rv", "run_kwargs"),
    [
        pytest.param(None, {}, id="ffmpeg_missing"),
        pytest.param(
            "/usr/bin/ffmpeg",
            {"return_value": MagicMock(returncode=1, stderr=b"")},
            id="ffmpeg_fails",
        ),
        pytest.param(
            "/usr/bin/ffmpeg",
            {"side_effect": OSError("mux failed")},
            id="subprocess_raises",
        ),
    ],
)
def test_mux_audio_file_unchanged_on_failure(
    dummy_video_path, tmp_path, which_rv, run_kwargs
) -> None:
    """_mux_audio leaves the output file unchanged when muxing cannot complete."""
    target_path = str(tmp_path / "video.mp4")
    shutil.copy(dummy_video_path, target_path)
    original_size = os.path.getsize(target_path)

    with (
        patch("supervision.utils.video.shutil.which", return_value=which_rv),
        patch("supervision.utils.video.subprocess.run", **run_kwargs),
    ):
        _mux_audio(source_path=dummy_video_path, video_path=target_path)

    assert os.path.getsize(target_path) == original_size


def test_mux_audio_replaces_file_on_success(dummy_video_path, tmp_path) -> None:
    """_mux_audio calls os.replace with video_path as destination on success."""
    target_path = str(tmp_path / "video.mp4")
    shutil.copy(dummy_video_path, target_path)

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stderr = b""

    with (
        patch("supervision.utils.video.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("supervision.utils.video.subprocess.run", return_value=success_result),
        patch("supervision.utils.video.os.replace") as mock_replace,
    ):
        _mux_audio(source_path=dummy_video_path, video_path=target_path)

    mock_replace.assert_called_once()
    assert mock_replace.call_args[0][1] == target_path


def test_get_video_frames_generator_with_start_end(dummy_video_path) -> None:
    """
    Verify that get_video_frames_generator respects start and end frame indices.

    Scenario: Iterating over a specific range of video frames using `start` and `end`.
    Expected: Only frames within the specified range are yielded, enabling targeted
    analysis of video segments.
    """
    generator = get_video_frames_generator(dummy_video_path, start=2, end=5)
    frames = list(generator)
    assert len(frames) == 3
