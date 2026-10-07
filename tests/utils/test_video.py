import os
import threading
import time
from pathlib import Path
from queue import Empty
from queue import Queue as StdQueue
from types import SimpleNamespace, TracebackType
from unittest.mock import patch

import av
import numpy as np
import pytest

from supervision import _cv2 as cv2
from supervision.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
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


def test_process_video_enqueues_writer_sentinel_and_waits_for_writer(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """process_video queues its sentinel and waits for the writer to exit."""
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

    assert write_queue.put_calls[-1] == (None, None)
    assert all(timeout is None for _item, timeout in write_queue.put_calls[:-1])
    assert join_calls == [10, None]


def test_process_video_retries_full_queue_sentinel_after_writer_recovers(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recovered writer drains a full queue and receives the shutdown sentinel."""
    target_path = str(tmp_path / "target_full_queue_recovered.mp4")
    write_started = threading.Event()
    release_write = threading.Event()
    callback_failed = threading.Event()
    queues: list[StdQueue[object]] = []
    writer_threads: list[threading.Thread] = []
    process_errors: list[BaseException] = []
    process_finished = threading.Event()

    def recording_queue(maxsize: int = 0) -> StdQueue[object]:
        """Retain the real queues so failure cleanup can be verified."""
        queue = StdQueue(maxsize=maxsize)
        queues.append(queue)
        return queue

    def blocked_write_frame(self: VideoSink, frame: np.ndarray) -> None:
        """Hold the first backend write while the bounded queue fills."""
        if not write_started.is_set():
            writer_threads.append(threading.current_thread())
            write_started.set()
            if not release_write.wait(timeout=10):
                raise TimeoutError("test backend was not released")

    def recover_backend() -> None:
        """Release the blocked backend after the former sentinel deadline."""
        if write_started.wait(timeout=5) and callback_failed.wait(timeout=5):
            release_write.wait(timeout=1.1)
            release_write.set()

    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Fill the one-frame queue, then preserve a callback failure."""
        if frame_index == 1:
            assert write_started.wait(timeout=5)
        if frame_index == 2:
            assert len(queues) == 2
            assert queues[1].full()
            callback_failed.set()
            raise ValueError("Test callback failure after queue filled")
        return frame

    def run_process_video() -> None:
        """Capture the pipeline exception without blocking the test thread."""
        try:
            process_video(
                source_path=dummy_video_path,
                target_path=target_path,
                callback=callback,
                writer_buffer=1,
            )
        except BaseException as exc:
            process_errors.append(exc)
        finally:
            process_finished.set()

    monkeypatch.setattr("supervision.utils.video.Queue", recording_queue)
    monkeypatch.setattr(
        "supervision.utils.video.VideoSink.write_frame", blocked_write_frame
    )
    recovery_thread = threading.Thread(target=recover_backend, daemon=True)
    recovery_thread.start()
    process_thread = threading.Thread(target=run_process_video, daemon=True)
    process_thread.start()

    try:
        assert callback_failed.wait(timeout=5)
        assert process_finished.wait(timeout=5)
    finally:
        release_write.set()
        if len(queues) == 2 and writer_threads and writer_threads[0].is_alive():
            queues[1].put(None, timeout=1)
        process_thread.join(timeout=5)
        recovery_thread.join(timeout=5)
        for writer_thread in writer_threads:
            writer_thread.join(timeout=5)

    assert len(process_errors) == 1
    assert isinstance(process_errors[0], ValueError)
    assert str(process_errors[0]) == "Test callback failure after queue filled"
    assert len(writer_threads) == 1
    assert not writer_threads[0].is_alive()


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
    assert join_calls == [10, None]


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


def _run_process_video_with_deadline(deadline_seconds: float, **kwargs) -> None:
    """Run process_video in a daemon thread; fail on timeout, re-raise its error.

    A hang must not block the whole test session, so the call runs in a daemon
    thread with a deadline. Exceptions stay confined to that thread, so they are
    captured and re-raised here to keep a crash from passing as a clean return.
    """
    errors: list[BaseException] = []
    finished = threading.Event()

    def run() -> None:
        """Call process_video, keeping any exception for the calling thread."""
        try:
            process_video(**kwargs)
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    threading.Thread(target=run, daemon=True).start()
    if not finished.wait(timeout=deadline_seconds):
        pytest.fail(f"process_video did not return within {deadline_seconds}s")
    if errors:
        raise errors[0]


def test_process_video_max_frames_larger_than_video_processes_whole_video(
    dummy_video_path: str, tmp_path: Path
) -> None:
    """A max_frames above the frame count processes every frame and returns."""
    target_path = str(tmp_path / "target_max_frames_overshoot.mp4")
    processed_indices: list[int] = []

    def callback(frame, index):
        """Record the index of every processed frame."""
        processed_indices.append(index)
        return frame

    _run_process_video_with_deadline(
        deadline_seconds=30,
        source_path=dummy_video_path,
        target_path=target_path,
        callback=callback,
        max_frames=10_000,
    )

    assert processed_indices == list(range(10))
    assert os.path.exists(target_path)


def test_process_video_max_frames_caps_read_when_frame_count_is_unknown(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-positive frame count leaves max_frames as the only cap on the read."""
    monkeypatch.setattr(
        "supervision.utils.video.VideoInfo.from_video_path",
        lambda video_path: VideoInfo(width=640, height=480, fps=25, total_frames=-1),
    )
    processed_indices: list[int] = []

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        """Record the index of every processed frame."""
        processed_indices.append(index)
        return frame

    process_video(
        source_path=dummy_video_path,
        target_path=str(tmp_path / "target_unknown_count.mp4"),
        callback=callback,
        max_frames=3,
    )

    assert processed_indices == [0, 1, 2]


def test_process_video_propagates_reader_thread_errors(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing reader thread raises RuntimeError instead of hanging forever."""
    target_path = str(tmp_path / "target_reader_error.mp4")

    def failing_generator(*args, **kwargs):
        """Stand in for a reader that cannot open or decode the source."""
        raise OSError("decode failed")

    monkeypatch.setattr(
        "supervision.utils.video.get_video_frames_generator", failing_generator
    )

    with pytest.raises(RuntimeError, match="Reader thread raised") as exc_info:
        _run_process_video_with_deadline(
            deadline_seconds=30,
            source_path=dummy_video_path,
            target_path=target_path,
            callback=lambda frame, index: frame,
        )

    assert isinstance(exc_info.value.__cause__, OSError)


def test_process_video_propagates_writer_thread_errors(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing frame write raises RuntimeError instead of hanging forever."""
    target_path = str(tmp_path / "target_writer_error.mp4")

    def failing_write_frame(self: VideoSink, frame: np.ndarray) -> None:
        """Stand in for a sink that cannot write, e.g. on a full disk."""
        raise OSError("write failed")

    monkeypatch.setattr(
        "supervision.utils.video.VideoSink.write_frame", failing_write_frame
    )

    with pytest.raises(RuntimeError, match="Writer thread raised") as exc_info:
        _run_process_video_with_deadline(
            deadline_seconds=30,
            source_path=dummy_video_path,
            target_path=target_path,
            callback=lambda frame, index: frame,
            writer_buffer=1,
        )

    assert isinstance(exc_info.value.__cause__, OSError)


def test_process_video_waits_for_delayed_writer_failure_before_releasing_sink(
    dummy_video_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A late write error is raised before the active sink is released."""
    target_path = str(tmp_path / "target_delayed_writer_error.mp4")
    write_started = threading.Event()
    release_write = threading.Event()
    write_active = threading.Event()
    process_finished = threading.Event()
    process_errors: list[BaseException] = []
    release_overlapped_write: list[bool] = []
    original_exit = VideoSink.__exit__

    def delayed_write_frame(self: VideoSink, frame: np.ndarray) -> None:
        """Hold the final write beyond the former ten-second join deadline."""
        write_active.set()
        write_started.set()
        try:
            if not release_write.wait(timeout=20):
                raise TimeoutError("test backend was not released")
            raise OSError("delayed write failed")
        finally:
            write_active.clear()

    def recording_exit(
        self: VideoSink,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Record whether the sink closes while a write remains active."""
        release_overlapped_write.append(write_active.is_set())
        original_exit(self, exc_type, exc_value, exc_traceback)

    def run_process_video() -> None:
        """Capture the pipeline error while the test controls the writer."""
        try:
            process_video(
                source_path=dummy_video_path,
                target_path=target_path,
                callback=lambda frame, index: frame,
                writer_buffer=32,
            )
        except BaseException as exc:
            process_errors.append(exc)
        finally:
            process_finished.set()

    monkeypatch.setattr(
        "supervision.utils.video.VideoSink.write_frame", delayed_write_frame
    )
    monkeypatch.setattr("supervision.utils.video.VideoSink.__exit__", recording_exit)
    process_thread = threading.Thread(target=run_process_video, daemon=True)
    process_thread.start()

    try:
        assert write_started.wait(timeout=5)
        assert not process_finished.wait(timeout=10.5)
    finally:
        release_write.set()
        process_thread.join(timeout=5)

    assert process_finished.is_set()
    assert len(process_errors) == 1
    assert isinstance(process_errors[0], RuntimeError)
    assert isinstance(process_errors[0].__cause__, OSError)
    assert not release_overlapped_write[0]


def test_process_video_rejects_callback_returning_none(
    dummy_video_path: str, tmp_path: Path
) -> None:
    """A callback returning None raises TypeError instead of hanging forever."""
    target_path = str(tmp_path / "target_callback_none.mp4")

    with pytest.raises(TypeError, match="returned None for frame 0"):
        _run_process_video_with_deadline(
            deadline_seconds=30,
            source_path=dummy_video_path,
            target_path=target_path,
            callback=lambda frame, index: None,
            writer_buffer=1,
        )


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


def test_video_info_and_frames_follow_display_rotation(tmp_path: Path) -> None:
    """Report and yield a phone-style portrait clip upright, not as it is stored."""
    video_path = str(tmp_path / "portrait.mp4")
    container = av.open(video_path, mode="w")
    stream = container.add_stream("mpeg4", rate=5)
    stream.width = 32
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    stream.set_display_rotation(90)
    stored_frame = np.zeros((16, 32, 3), dtype=np.uint8)
    for _ in range(3):
        for packet in stream.encode(av.VideoFrame.from_ndarray(stored_frame)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    video_info = VideoInfo.from_video_path(video_path)
    frames = list(get_video_frames_generator(video_path))

    assert video_info.resolution_wh == (16, 32)
    assert len(frames) == 3
    assert all(frame.shape == (32, 16, 3) for frame in frames)


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


def test_get_video_frames_generator_prefetch_matches_sync(dummy_video_path) -> None:
    """Verify that the prefetch path yields identical frames to the sync path.

    Scenario: Iterating over a video with prefetch=4 and again with prefetch=0
        (synchronous) on the same dummy video.
    Expected: Both generators yield the same number of frames in the same order,
        with each corresponding frame being pixel-for-pixel identical.
    """
    sync_frames = list(get_video_frames_generator(dummy_video_path))
    prefetched_frames = list(get_video_frames_generator(dummy_video_path, prefetch=4))
    assert len(prefetched_frames) == len(sync_frames) == 10
    for a, b in zip(prefetched_frames, sync_frames):
        assert np.array_equal(a, b)


def test_get_video_frames_generator_prefetch_propagates_decode_errors(tmp_path) -> None:
    """Verify that reader-thread exceptions reach the consumer, not get swallowed.

    Scenario: Passing a non-existent file path to the prefetch path so the reader
        thread fails immediately on video open.
    Expected: The exception propagates to the consumer and is raised as a
        RuntimeError wrapping the original error; the consumer does not hang.
    """
    missing_path = str(tmp_path / "does_not_exist.mp4")
    with pytest.raises(RuntimeError) as exc_info:
        list(get_video_frames_generator(missing_path, prefetch=4))
    assert exc_info.value.__cause__ is not None


def test_get_video_frames_generator_prefetch_early_termination(
    dummy_video_path,
) -> None:
    """Verify that breaking out of the prefetched generator does not block reuse.

    Scenario: Consuming only 3 frames from a 10-frame video with prefetch=4, then
        creating a fresh generator on the same file.
    Expected: The break exits cleanly without hanging; a new generator on the same
        file yields all 10 frames normally.
    """
    taken = []
    for frame in get_video_frames_generator(dummy_video_path, prefetch=4):
        taken.append(frame)
        if len(taken) >= 3:
            break
    assert len(taken) == 3
    # A fresh generator on the same file must still work normally.
    assert len(list(get_video_frames_generator(dummy_video_path, prefetch=4))) == 10


@pytest.mark.parametrize(
    ("stride", "start", "end"),
    [
        pytest.param(2, 0, None, id="stride2"),
        pytest.param(1, 2, 7, id="start2_end7"),
        pytest.param(2, 2, 8, id="stride2_start2_end8"),
    ],
)
def test_get_video_frames_generator_prefetch_param_forwarding(
    dummy_video_path, stride, start, end
) -> None:
    """Prefetch path must forward stride/start/end identically to the sync path.

    Scenario: Using the prefetch path with various stride, start, and end
        combinations to verify parameters are correctly forwarded.
    Expected: The prefetch output matches the sync path frame-for-frame for
        each combination; no frames skipped or duplicated.
    """
    sync_frames = list(
        get_video_frames_generator(
            dummy_video_path, stride=stride, start=start, end=end
        )
    )
    prefetched_frames = list(
        get_video_frames_generator(
            dummy_video_path, stride=stride, start=start, end=end, prefetch=4
        )
    )
    assert len(prefetched_frames) == len(sync_frames)
    for a, b in zip(prefetched_frames, sync_frames):
        assert np.array_equal(a, b)


def test_get_video_frames_generator_prefetch_minimum_queue(dummy_video_path) -> None:
    """prefetch=1 creates maximum backpressure; all frames must be returned in order.

    Scenario: Using prefetch=1 forces the reader to block after every decoded
        frame, maximising producer-consumer synchronisation pressure.
    Expected: All 10 frames are yielded in the same order as the sync path.
    """
    sync_frames = list(get_video_frames_generator(dummy_video_path))
    prefetched_frames = list(get_video_frames_generator(dummy_video_path, prefetch=1))
    assert len(prefetched_frames) == len(sync_frames) == 10
    for a, b in zip(prefetched_frames, sync_frames):
        assert np.array_equal(a, b)


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


def test_get_video_frames_generator_prefetch_negative_raises(dummy_video_path) -> None:
    """Negative prefetch raises ValueError when the generator is first consumed.

    Scenario: Creating get_video_frames_generator with prefetch=-1 and pulling the
        first item; because the function is a generator, the guard fires on the first
        `next()`, not at call time.
    Expected: A ValueError naming the invalid prefetch value is raised.
    """
    generator = get_video_frames_generator(dummy_video_path, prefetch=-1)
    with pytest.raises(ValueError, match="prefetch must be >= 0"):
        next(generator)


def test_get_video_frames_generator_prefetch_stride_early_termination(
    dummy_video_path,
) -> None:
    """Breaking early with stride>1 and prefetch>0 exits cleanly without hanging.

    Scenario: Consuming only 2 frames from a 10-frame video with stride=2 and
        prefetch=4, then creating a fresh strided prefetch generator on the same file.
    Expected: The break exits cleanly (no hang); exactly 2 valid frames are taken, and
        a fresh strided prefetch generator still yields all 5 strided frames.
    """
    taken = []
    for frame in get_video_frames_generator(dummy_video_path, stride=2, prefetch=4):
        taken.append(frame)
        if len(taken) >= 2:
            break
    assert len(taken) == 2
    assert all(isinstance(f, np.ndarray) and f.shape == (480, 640, 3) for f in taken)
    # A fresh strided prefetch generator on the same file must still work fully.
    fresh = get_video_frames_generator(dummy_video_path, stride=2, prefetch=4)
    assert len(list(fresh)) == 5


def test_get_video_frames_generator_prefetch_consumer_exception_cleans_up_thread(
    dummy_video_path,
) -> None:
    """A consumer exception propagates and the prefetch reader thread is cleaned up.

    Scenario: Raising inside the for-loop body while iterating a prefetch=4 generator,
        so the anonymous generator is closed as the exception unwinds.
    Expected: The RuntimeError propagates to the caller, and the reader thread is
        joined by the generator's finally (active thread count returns to baseline).
    """

    def consume_then_raise() -> None:
        """Consume the prefetch generator and raise from inside the loop body."""
        for _frame in get_video_frames_generator(dummy_video_path, prefetch=4):
            raise RuntimeError("consumer boom")

    baseline_threads = threading.active_count()
    with pytest.raises(RuntimeError, match="consumer boom"):
        consume_then_raise()
    assert threading.active_count() == baseline_threads


def test_get_video_frames_generator_prefetch_reader_outlives_join(monkeypatch) -> None:
    """A reader stuck in a slow read must not make the outer generator hang on close.

    Scenario: The background reader blocks in a slow `read()` that outlives the
        generator's `finally: thread.join(timeout=2.0)`; the consumer closes the
        generator after one frame.
    Expected: `close()` returns without hanging or raising — the bounded join gives up
        and the daemon reader is left to exit on its own.
    """

    class SlowCapture:
        """Fake capture that serves one frame then blocks on the next read."""

        def __init__(self) -> None:
            self.released = False
            self.calls = 0
            self.block = threading.Event()

        def read(self):
            """Return one frame, then block on subsequent reads to simulate a hang."""
            self.calls += 1
            if self.calls == 1:
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            self.block.wait(timeout=10.0)
            return False, None

        def grab(self):
            """Report a successful grab so stride handling proceeds."""
            return True

        def release(self) -> None:
            """Record that the capture was released."""
            self.released = True

    slow_capture = SlowCapture()
    monkeypatch.setattr(
        "supervision.utils.video._validate_and_setup_video",
        lambda *args, **kwargs: (slow_capture, 0, 100),
    )

    generator = get_video_frames_generator("dummy", prefetch=4)
    first_frame = next(generator)
    start = time.monotonic()
    generator.close()
    elapsed = time.monotonic() - start
    # Release the daemon reader so it can exit cleanly after the test.
    slow_capture.block.set()

    assert isinstance(first_frame, np.ndarray)
    assert elapsed < 5.0


def test_get_video_frames_generator_prefetch_yields_buffered_frames_before_error(
    monkeypatch,
) -> None:
    """Frames already decoded before a mid-stream failure are yielded, then raised.

    Scenario: A fake capture successfully reads 3 frames, then raises on the 4th
        `read()` call, with `prefetch=4` so all 3 good frames fit in the queue ahead
        of the sentinel.
    Expected: The consumer receives exactly the 3 good frames in order, then the
        wrapping `RuntimeError` — matching the documented "buffered frames before
        RuntimeError" ordering guarantee.
    """

    class FailAfterNCapture:
        """Fake capture that serves N frames then raises on the next read."""

        def __init__(self, good_reads: int) -> None:
            self.good_reads = good_reads
            self.calls = 0
            self.released = False

        def read(self):
            """Return a frame for the first `good_reads` calls, then raise."""
            self.calls += 1
            if self.calls <= self.good_reads:
                return True, np.full((2, 2, 3), self.calls, dtype=np.uint8)
            raise OSError("simulated mid-stream decode failure")

        def grab(self):
            """Report a successful grab so stride handling proceeds."""
            return True

        def release(self) -> None:
            """Record that the capture was released."""
            self.released = True

    fake_capture = FailAfterNCapture(good_reads=3)
    monkeypatch.setattr(
        "supervision.utils.video._validate_and_setup_video",
        lambda *args, **kwargs: (fake_capture, 0, 100),
    )

    collected = []

    def consume() -> None:
        """Drain the generator into `collected`, letting the error propagate."""
        for frame in get_video_frames_generator("dummy", prefetch=4):
            collected.append(frame)

    with pytest.raises(RuntimeError) as exc_info:
        consume()
    assert len(collected) == 3
    assert isinstance(exc_info.value.__cause__, OSError)


def test_get_video_frames_generator_prefetch_zero_frame_video(monkeypatch) -> None:
    """The prefetch path yields nothing and returns cleanly for a zero-frame video.

    Scenario: A fake capture reports failure on the very first `read()`, simulating
        an empty video, with `prefetch=4`.
    Expected: The generator yields no frames and returns without hanging on the
        initial `frame_queue.get`/`thread.is_alive()` poll.
    """

    class EmptyCapture:
        """Fake capture that yields zero frames."""

        def read(self):
            """Report immediate end-of-stream."""
            return False, None

        def grab(self):
            """Report a successful grab so stride handling proceeds."""
            return True

        def release(self) -> None:
            """No-op release for the empty-video fake."""

    monkeypatch.setattr(
        "supervision.utils.video._validate_and_setup_video",
        lambda *args, **kwargs: (EmptyCapture(), 0, 100),
    )

    frames = list(get_video_frames_generator("dummy", prefetch=4))

    assert frames == []


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


@pytest.fixture
def numbered_video_path(tmp_path: Path) -> str:
    """Write a 10-frame video whose frame `i` is filled with intensity `25 * i`."""
    path = str(tmp_path / "numbered_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25, (64, 48))
    for frame_index in range(10):
        out.write(np.full((48, 64, 3), 25 * frame_index, dtype=np.uint8))
    out.release()
    return path


@pytest.mark.parametrize("iterative_seek", [False, True])
@pytest.mark.parametrize(
    ("start", "end", "stride", "expected_frame_indices"),
    [
        pytest.param(2, 5, 1, [2, 3, 4], id="range-longer-than-start"),
        pytest.param(4, 6, 1, [4, 5], id="range-shorter-than-start"),
        pytest.param(2, 8, 2, [2, 4, 6], id="with-stride"),
    ],
)
def test_get_video_frames_generator_stops_at_end_after_seeking_to_start(
    numbered_video_path: str,
    start: int,
    end: int,
    stride: int,
    iterative_seek: bool,
    expected_frame_indices: list[int],
) -> None:
    """Frames from `start` up to `end` are yielded whichever way `start` is sought."""
    frames = get_video_frames_generator(
        numbered_video_path,
        stride=stride,
        start=start,
        end=end,
        iterative_seek=iterative_seek,
    )

    frame_indices = [round(float(frame.mean()) / 25) for frame in frames]

    assert frame_indices == expected_frame_indices


class _UnreliableCountCapture:
    """Fake capture that decodes `frame_count` frames but reports `reported_count`.

    OpenCV estimates `CAP_PROP_FRAME_COUNT` from container metadata. A WebM with
    no duration, as written by a browser's `MediaRecorder`, reports a huge negative
    count, and a variable frame rate MKV or WebM can report fewer frames than it
    holds.
    """

    def __init__(self, frame_count: int, reported_count: float) -> None:
        self.frame_count = frame_count
        self.reported_count = reported_count
        self.position = 0

    def isOpened(self) -> bool:
        """Report the capture as open."""
        return True

    def get(self, property_id: int) -> float:
        """Return the unreliable frame count estimate."""
        assert property_id == cv2.CAP_PROP_FRAME_COUNT
        return self.reported_count

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Decode the next frame, filled with its index, until the stream ends."""
        if self.position >= self.frame_count:
            return False, None
        frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame

    def grab(self) -> bool:
        """Skip the next frame."""
        success, _ = self.read()
        return success

    def release(self) -> None:
        """No-op release for the fake capture."""


@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize(
    ("reported_count", "end", "expected_frame_indices"),
    [
        pytest.param(-2.767e17, None, [0, 1, 2, 3, 4], id="negative-count"),
        pytest.param(0, None, [0, 1, 2, 3, 4], id="zero-count"),
        pytest.param(3, None, [0, 1, 2, 3, 4], id="underestimated-count"),
        pytest.param(-2.767e17, 2, [0, 1], id="negative-count-with-end"),
        pytest.param(0, 3, [0, 1, 2], id="zero-count-with-end"),
        pytest.param(5, 5, [0, 1, 2, 3, 4], id="end-at-positive-count"),
    ],
)
def test_get_video_frames_generator_reads_past_unreliable_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    reported_count: float,
    end: int | None,
    prefetch: int,
    expected_frame_indices: list[int],
) -> None:
    """Frames are read until the stream ends, not until the estimated frame count."""
    monkeypatch.setattr(
        "supervision.utils.video.cv2.VideoCapture",
        lambda source_path: _UnreliableCountCapture(
            frame_count=5, reported_count=reported_count
        ),
    )

    frames = get_video_frames_generator("recording.webm", end=end, prefetch=prefetch)

    assert [int(frame[0, 0, 0]) for frame in frames] == expected_frame_indices


def test_get_video_frames_generator_rejects_end_past_positive_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An `end` past a positive reported frame count raises on first iteration."""
    monkeypatch.setattr(
        "supervision.utils.video.cv2.VideoCapture",
        lambda source_path: _UnreliableCountCapture(frame_count=5, reported_count=5),
    )
    frames = get_video_frames_generator("recording.webm", end=6)

    with pytest.raises(Exception, match="outbound"):
        next(frames)


@pytest.mark.parametrize("iterative_seek", [False, True])
@pytest.mark.parametrize(
    ("reported_count", "start", "expected_frame_indices"),
    [
        pytest.param(-2.767e17, 2, [2, 3, 4], id="negative-count"),
        pytest.param(0, 2, [2, 3, 4], id="zero-count"),
        pytest.param(3, 4, [4], id="start-past-underestimated-count"),
        pytest.param(-2.767e17, 7, [], id="start-past-stream"),
        pytest.param(5, 7, [], id="start-past-positive-count"),
    ],
)
def test_get_video_frames_generator_seeks_to_start_past_unreliable_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    reported_count: float,
    start: int,
    iterative_seek: bool,
    expected_frame_indices: list[int],
) -> None:
    """The first frame yielded is `start` even when the count cannot bound it."""
    monkeypatch.setattr(
        "supervision.utils.video.cv2.VideoCapture",
        lambda source_path: _UnreliableCountCapture(
            frame_count=5, reported_count=reported_count
        ),
    )

    frames = get_video_frames_generator(
        "recording.webm", start=start, iterative_seek=iterative_seek
    )

    assert [int(frame[0, 0, 0]) for frame in frames] == expected_frame_indices


@pytest.mark.parametrize("iterative_seek", [False, True])
def test_get_video_frames_generator_yields_nothing_when_start_is_past_video(
    numbered_video_path: str, iterative_seek: bool
) -> None:
    """A `start` past the last frame of a real video yields no frames."""
    frames = get_video_frames_generator(
        numbered_video_path, start=12, iterative_seek=iterative_seek
    )

    assert list(frames) == []
