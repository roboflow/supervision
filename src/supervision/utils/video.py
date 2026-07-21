from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from types import TracebackType
from typing import cast

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from supervision import _cv2 as cv2
from supervision._cv2._video import _mux_audio
from supervision.utils.logger import _get_logger

logger = _get_logger(__name__)


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
        total number of frames.

    Attributes:
        width: width of the video in pixels
        height: height of the video in pixels
        fps: frames per second of the video as a float. Common values include
            23.976, 24.0, 25.0, 29.97, 30.0, 59.94, and 60.0.
        total_frames: total number of frames in the video,
            default is None

    Examples:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(video_path="<SOURCE_VIDEO_FILE>")

        video_info
        # VideoInfo(width=3840, height=2160, fps=25.0, total_frames=538)

        video_info.resolution_wh
        # (3840, 2160)
        ```
    """

    width: int
    height: int
    fps: float
    total_frames: int | None = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        """Read video metadata from a file path.

        Args:
            video_path: Path to the video file.

        Returns:
            A `VideoInfo` instance with width, height, fps, and total_frames.

        Examples:
            ```python
            import supervision as sv

            video_info = sv.VideoInfo.from_video_path(video_path="<SOURCE_VIDEO_FILE>")
            # VideoInfo(width=3840, height=2160, fps=25.0, total_frames=538)
            ```
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height


class VideoSink:
    """
    Context manager that saves video frames to a file using OpenCV.

    Attributes:
        target_path: The path to the output file where the video will be saved.
        video_info: Information about the video resolution, fps,
            and total frame count.
        codec: FOURCC code for video format

    Example:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path("<SOURCE_VIDEO_PATH>")
        frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

        with sv.VideoSink(target_path="<TARGET_VIDEO_PATH>", video_info=video_info) as sink:
            for frame in frames_generator:
                sink.write_frame(frame=frame)
        ```
    """  # noqa: E501 // docs

    def __init__(
        self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"
    ) -> None:
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__fourcc: int = 0
        self.__writer: cv2.VideoWriter | None = None

    def __enter__(self) -> VideoSink:
        """Open the underlying video writer for context-managed frame output."""
        fourcc_fn = cast(
            Callable[[str, str, str, str], int], getattr(cv2, "VideoWriter_fourcc")
        )
        try:
            self.__fourcc = int(fourcc_fn(*self.__codec))
        except TypeError as e:
            logger.warning("%s. Defaulting to mp4v...", str(e))
            self.__fourcc = int(fourcc_fn(*"mp4v"))
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        # OpenCV can construct a writer object that is not usable for the target path.
        if not self.__writer.isOpened():
            self.__writer.release()
            self.__writer = None
            raise RuntimeError(f"Could not open video writer for {self.target_path}")
        return self

    def write_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """
        Writes a single video frame to the target video file.

        Args:
            frame: The video frame to be written to the file. The frame
                must be in BGR color format.
        """
        # Preserve the context-manager invariant instead of silently dropping frames.
        if self.__writer is None:
            raise RuntimeError("write_frame requires an open VideoSink context.")
        self.__writer.write(frame)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Release the underlying video writer when leaving the context."""
        if self.__writer is not None:
            self.__writer.release()
            self.__writer = None


def _validate_and_setup_video(
    source_path: str, start: int, end: int | None, iterative_seek: bool = False
) -> tuple[cv2.VideoCapture, int, int]:
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is not None and end > total_frames:
        raise Exception("Requested frames are outbound")
    start = max(start, 0)
    end = min(end, total_frames) if end is not None else total_frames

    if iterative_seek:
        while start > 0:
            success = video.grab()
            if not success:
                break
            start -= 1
    elif start > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, start)

    return video, start, end


def get_video_frames_generator(
    source_path: str,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    iterative_seek: bool = False,
    prefetch: int = 0,
) -> Generator[npt.NDArray[np.uint8], None, None]:
    """
    Get a generator that yields the frames of the video.

    Args:
        source_path: The path of the video file.
        stride: Indicates the interval at which frames are returned,
            skipping stride - 1 frames between each.
        start: Indicates the starting position from which
            video should generate frames
        end: Indicates the ending position at which video
            should stop generating frames. If None, video will be read to the end.
        iterative_seek: If True, the generator will seek to the
            `start` frame by grabbing each frame, which is much slower. This is a
            workaround for videos that don't open at all when you set the `start` value.
        prefetch: If > 0, decode frames in a background thread and buffer up to
            this many frames in a bounded queue. Useful when the consumer (e.g.
            CPU inference) is the bottleneck and can overlap with decode I/O.
            Default 0 keeps the original synchronous behaviour unchanged. Note:
            each buffered frame occupies width x height x 3 bytes of uncompressed
            memory; use `sv.VideoInfo.from_video_path()` to size appropriately.

    Returns:
        A generator that yields the frames of the video.

    Raises:
        ValueError: If `prefetch` is negative.

    Note:
        For live camera streams, use `cv2.VideoCapture` with an integer device
        index directly. `get_video_frames_generator` is designed for file-based
        sources; `cv2.VideoCapture` must be released by the caller when done.
        This requires OpenCV to be installed — the PyAV-based fallback used
        when OpenCV is unavailable only supports file paths, not webcam device
        indexes; passing an integer source to it always leaves the capture
        closed (`isOpened()` returns `False`):

        ```python
        from supervision import _cv2 as cv2

        cap = cv2.VideoCapture(0)  # 0 = default webcam
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                ...  # process frame
        finally:
            cap.release()
        ```

    Examples:
        ```python
        import supervision as sv

        for frame in sv.get_video_frames_generator(source_path="<SOURCE_VIDEO_PATH>"):
            ...

        # Prefetch frames in a background thread to overlap I/O with CPU inference:
        for frame in sv.get_video_frames_generator(
            source_path="<SOURCE_VIDEO_PATH>", prefetch=8
        ):
            ...
        ```
    """
    if prefetch < 0:
        raise ValueError(f"prefetch must be >= 0, got {prefetch!r}")
    if prefetch > 0:
        yield from _prefetched_frames_generator(
            source_path=source_path,
            stride=stride,
            start=start,
            end=end,
            iterative_seek=iterative_seek,
            prefetch=prefetch,
        )
        return

    video, start, end = _validate_and_setup_video(
        source_path, start, end, iterative_seek
    )
    frame_position = start
    try:
        while True:
            success, frame = video.read()
            if not success or frame_position >= end:
                break
            if frame is not None:
                yield cast(npt.NDArray[np.uint8], frame)
            for _ in range(stride - 1):
                success = video.grab()
                if not success:
                    break
            frame_position += stride
    finally:
        video.release()


def _prefetched_frames_generator(
    source_path: str,
    stride: int,
    start: int,
    end: int | None,
    iterative_seek: bool,
    prefetch: int,
) -> Generator[npt.NDArray[np.uint8], None, None]:
    """Read frames into a bounded queue on a daemon thread.

    Sentinel protocol: None = normal EOF, Exception instance = reader error.
    """
    frame_queue: Queue[npt.NDArray[np.uint8] | BaseException | None] = Queue(
        maxsize=prefetch
    )
    stop_event = threading.Event()

    def reader() -> None:
        sentinel: BaseException | None = None
        try:
            for frame in get_video_frames_generator(
                source_path=source_path,
                stride=stride,
                start=start,
                end=end,
                iterative_seek=iterative_seek,
                prefetch=0,
            ):
                if stop_event.is_set():
                    return
                while True:
                    try:
                        frame_queue.put(frame, timeout=0.1)
                        break
                    except Full:
                        if stop_event.is_set():
                            return
        except BaseException as exc:
            if isinstance(exc, Exception):
                sentinel = exc
        finally:
            while not stop_event.is_set():
                try:
                    frame_queue.put(sentinel, timeout=0.1)
                    return
                except Full:
                    pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = frame_queue.get(timeout=0.5)
            except Empty:
                if not thread.is_alive():
                    break
                continue
            if isinstance(item, BaseException):
                raise RuntimeError(f"Reader thread raised: {item!r}") from item
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[npt.NDArray[np.uint8], int], npt.NDArray[np.uint8]],
    *,
    max_frames: int | None = None,
    prefetch: int = 32,
    writer_buffer: int = 32,
    show_progress: bool = False,
    progress_message: str = "Processing video",
    preserve_audio: bool = False,
) -> None:
    """
    Process video frames asynchronously using a threaded pipeline.

    This function orchestrates a three-stage pipeline to optimize video processing
    throughput:

    1. Reader thread: Continuously reads frames from the source video file and
       enqueues them into a bounded queue (`frame_read_queue`). The queue size is
       limited by the `prefetch` parameter to control memory usage.
    2. Main thread (Processor): Dequeues frames from `frame_read_queue`, applies the
       user-defined `callback` function to process each frame, then enqueues the
       processed frames into another bounded queue (`frame_write_queue`) for writing.
       The processing happens in the main thread, simplifying use of stateful objects
       without synchronization.
    3. Writer thread: Dequeues processed frames from `frame_write_queue` and writes
       them sequentially to the output video file.

    Args:
        source_path: Path to the input video file.
        target_path: Path where the processed video will be saved.
        callback: Function called for
            each frame, accepting the frame as a numpy array and its zero-based index,
            returning the processed frame.
        max_frames: Optional maximum number of frames to process.
            If None, the entire video is processed (default).
        prefetch: Maximum number of frames buffered by the reader thread.
            Controls memory use; default is 32.
        writer_buffer: Maximum number of frames buffered before writing.
            Controls output buffer size; default is 32.
        show_progress: Whether to display a tqdm progress bar during processing.
            Default is False.
        progress_message: Description shown in the progress bar.
        preserve_audio: If True, copy the audio stream from `source_path` into
            `target_path` after frame processing. Remuxing is done with PyAV;
            no external `ffmpeg` executable is required. If the mux step
            fails, a warning is logged and the output video is saved without
            audio — no exception is raised. Audio is truncated to match the
            processed video duration. Default is False.

    Returns:
        None

    Example:
        ```python
        import supervision as sv
        from rfdetr import RFDETRMedium

        model = RFDETRMedium()

        def callback(frame, frame_index):
            return model.predict(frame)

        sv.process_video(
            source_path="source.mp4",
            target_path="target.mp4",
            callback=callback,
            preserve_audio=True,
        )
        ```
    """
    video_info = VideoInfo.from_video_path(video_path=source_path)
    total_frames = (
        min(video_info.total_frames or 0, max_frames)
        if max_frames is not None
        else video_info.total_frames or 0
    )

    frame_read_queue: Queue[tuple[int, npt.NDArray[np.uint8]] | None] = Queue(
        maxsize=prefetch
    )
    frame_write_queue: Queue[npt.NDArray[np.uint8] | None] = Queue(
        maxsize=writer_buffer
    )

    def reader_thread() -> None:
        frame_generator = get_video_frames_generator(
            source_path=source_path,
            end=max_frames,
        )
        for frame_index, frame in enumerate(frame_generator):
            frame_read_queue.put((frame_index, frame))
        frame_read_queue.put(None)

    def writer_thread(video_sink: VideoSink) -> None:
        while True:
            frame = frame_write_queue.get()
            if frame is None:
                break
            video_sink.write_frame(frame=frame)

    reader_worker = threading.Thread(target=reader_thread, daemon=True)
    with VideoSink(target_path=target_path, video_info=video_info) as video_sink:
        writer_worker = threading.Thread(
            target=writer_thread,
            args=(video_sink,),
            daemon=True,
        )

        reader_worker.start()
        writer_worker.start()

        progress_bar = tqdm(
            total=total_frames,
            disable=not show_progress,
            desc=progress_message,
        )

        exception_in_worker: Exception | None = None
        read_finished = False

        try:
            while True:
                read_item = frame_read_queue.get()
                if read_item is None:
                    read_finished = True
                    break

                frame_index, frame = read_item
                try:
                    processed_frame = callback(frame, frame_index)
                    frame_write_queue.put(processed_frame)
                    progress_bar.update(1)
                except Exception as exc:
                    exception_in_worker = exc
                    break
        finally:
            try:
                frame_write_queue.put(None, timeout=1)
            except Full:
                # Best effort: if the writer is stuck and the queue never drains,
                # do not block shutdown forever trying to enqueue the sentinel.
                pass
            if not read_finished:
                while True:
                    # Use timeout to prevent indefinite blocking if reader thread fails
                    try:
                        read_item = frame_read_queue.get(timeout=1)
                        if read_item is None:
                            break
                    # If we timeout waiting for a frame, only assume failure if reader
                    # thread is no longer alive. Otherwise, keep waiting as the reader
                    # may simply be slow (for example, due to a slow source).
                    except Empty:
                        if not reader_worker.is_alive():
                            break
                        # Reader is still alive; continue waiting for frames.
                        continue
            reader_worker.join(timeout=10)
            writer_worker.join(timeout=10)
            progress_bar.close()
            if exception_in_worker is not None:
                raise exception_in_worker

    if preserve_audio:
        if writer_worker.is_alive():
            logger.warning(
                "Writer thread did not finish in time; skipping audio mux "
                "to avoid reading an incomplete output file."
            )
        else:
            _mux_audio(source_path=source_path, video_path=target_path)


class FPSMonitor:
    """
    A class for monitoring frames per second (FPS) to benchmark latency.
    """

    def __init__(self, sample_size: int = 30) -> None:
        """
        Args:
            sample_size: The maximum number of observations for latency
                benchmarking.

        Examples:
            ```python
            import supervision as sv

            frames_generator = sv.get_video_frames_generator(
                source_path="<SOURCE_FILE_PATH>")
            fps_monitor = sv.FPSMonitor()

            for frame in frames_generator:
                # your processing code here
                fps_monitor.tick()
                fps = fps_monitor.fps
            ```
        """
        self.all_timestamps: deque[float] = deque(maxlen=sample_size)

    @property
    def fps(self) -> float:
        """
        Computes and returns the average FPS based on the stored time stamps.

        Returns:
            The average FPS across the recorded intervals. Returns 0.0 if fewer
            than two time stamps are stored.
        """
        if len(self.all_timestamps) < 2:
            return 0.0
        taken_time = self.all_timestamps[-1] - self.all_timestamps[0]
        frame_intervals = len(self.all_timestamps) - 1
        return frame_intervals / taken_time if taken_time != 0 else 0.0

    def tick(self) -> None:
        """
        Adds a new time stamp to the deque for FPS calculation.
        """
        self.all_timestamps.append(time.monotonic())

    def reset(self) -> None:
        """
        Clears all the time stamps from the deque.
        """
        self.all_timestamps.clear()
