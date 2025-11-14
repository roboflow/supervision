from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from queue import Queue

import cv2
import numpy as np
from tqdm.auto import tqdm


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
        total number of frames.

    Attributes:
        width (int): width of the video in pixels
        height (int): height of the video in pixels
        fps (int): frames per second of the video
        total_frames (Optional[int]): total number of frames in the video,
            default is None

    Examples:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(video_path=<SOURCE_VIDEO_FILE>)

        video_info
        # VideoInfo(width=3840, height=2160, fps=25, total_frames=538)

        video_info.resolution_wh
        # (3840, 2160)
        ```
    """

    width: int
    height: int
    fps: int
    total_frames: int | None = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
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
        target_path (str): The path to the output file where the video will be saved.
        video_info (VideoInfo): Information about the video resolution, fps,
            and total frame count.
        codec (str): FOURCC code for video format

    Example:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(<SOURCE_VIDEO_PATH>)
        frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>)

        with sv.VideoSink(target_path=<TARGET_VIDEO_PATH>, video_info=video_info) as sink:
            for frame in frames_generator:
                sink.write_frame(frame=frame)
        ```
    """  # noqa: E501 // docs

    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__writer = None

    def __enter__(self):
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*self.__codec)
        except TypeError as e:
            print(str(e) + ". Defaulting to mp4v...")
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        return self

    def write_frame(self, frame: np.ndarray):
        """
        Writes a single video frame to the target video file.

        Args:
            frame (np.ndarray): The video frame to be written to the file. The frame
                must be in BGR color format.
        """
        self.__writer.write(frame)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__writer.release()


def _validate_and_setup_video(
    source_path: str, start: int, end: int | None, iterative_seek: bool = False
):
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
) -> Generator[np.ndarray]:
    """
    Get a generator that yields the frames of the video.

    Args:
        source_path (str): The path of the video file.
        stride (int): Indicates the interval at which frames are returned,
            skipping stride - 1 frames between each.
        start (int): Indicates the starting position from which
            video should generate frames
        end (Optional[int]): Indicates the ending position at which video
            should stop generating frames. If None, video will be read to the end.
        iterative_seek (bool): If True, the generator will seek to the
            `start` frame by grabbing each frame, which is much slower. This is a
            workaround for videos that don't open at all when you set the `start` value.

    Returns:
        (Generator[np.ndarray, None, None]): A generator that yields the
            frames of the video.

    Examples:
        ```python
        import supervision as sv

        for frame in sv.get_video_frames_generator(source_path=<SOURCE_VIDEO_PATH>):
            ...
        ```
    """
    video, start, end = _validate_and_setup_video(
        source_path, start, end, iterative_seek
    )
    frame_position = start
    while True:
        success, frame = video.read()
        if not success or frame_position >= end:
            break
        yield frame
        for _ in range(stride - 1):
            success = video.grab()
            if not success:
                break
        frame_position += stride
    video.release()


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    *,
    max_frames: int | None = None,
    prefetch: int = 32,
    writer_buffer: int = 32,
    show_progress: bool = False,
    progress_message: str = "Processing video",
) -> None:
    """
    Process a video using a threaded pipeline that asynchronously
    reads frames, applies a callback to each, and writes the results
    to an output file.

    Overview:
    This function implements a three-stage pipeline designed to maximize
    frame throughput.

        │   Reader   │ >> │  Processor   │ >> │   Writer   │
           (thread)           (main)             (thread)

    - Reader thread: reads frames from disk into a bounded queue ('read_q')
      until full, then blocks. This ensures we never load more than 'prefetch'
      frames into memory at once.

    - Main thread: dequeues frames, applies the 'callback(frame, idx)',
      and enqueues the processed result into 'write_q'.
      This is the compute stage. It's important to note that it's not threaded,
      so you can safely use any detectors, trackers, or other stateful objects
      without synchronization issues.

    - Writer thread: dequeues frames and writes them to disk.

    Both queues are bounded to enforce back-pressure:
      - The reader cannot outpace processing (avoids unbounded RAM usage).
      - The processor cannot outpace writing (avoids output buffer bloat).

    Summary:
    - It's thread-safe: because the callback runs only in the main thread,
    using a single stateful detector/tracker inside callback does not require
    synchronization with the reader/writer threads.

    - While the main thread processes frame N, the reader is already decoding frame N+1,
      and the writer is encoding frame N-1. They operate concurrently without blocking
      each other.

    - When is it fastest?
        - When there's heavy computation in the callback function that releases
          the Python GIL (for example, OpenCV filters, resizes, color conversions, ...)
        - When using CUDA or GPU-accelerated inference.

    - When is it better not to use it?
        - When the callback function is Python-heavy and GIL-bound. In that case,
          using a process-based approach is more effective.

    Examples:
        ```python
        import supervision as sv
        def callback(scene: np.ndarray, index: int) -> np.ndarray:
            ...
        process_video(
            source_path=<SOURCE_VIDEO_PATH>,
            target_path=<TARGET_VIDEO_PATH>,
            callback=callback
        )
        ```

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
            a numpy ndarray representation of a video frame and an
            int index of the frame and returns a processed numpy ndarray
            representation of the frame.
        max_frames (Optional[int]): The maximum number of frames to process.
        prefetch (int): The maximum number of frames buffered by the reader thread.
        writer_buffer (int): The maximum number of frames buffered before writing.
        show_progress (bool): Whether to show a progress bar.
        progress_message (str): The message to display in the progress bar.
    """

    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    total_frames = (
        min(source_video_info.total_frames, max_frames)
        if max_frames is not None
        else source_video_info.total_frames
    )

    # Each queue includes frames + sentinel
    read_q: Queue[tuple[int, np.ndarray] | None] = Queue(maxsize=prefetch)
    write_q: Queue[np.ndarray | None] = Queue(maxsize=writer_buffer)

    def reader_thread():
        gen = get_video_frames_generator(source_path=source_path, end=max_frames)
        for idx, frame in enumerate(gen):
            read_q.put((idx, frame))
        read_q.put(None)  # sentinel

    def writer_thread(video_sink: VideoSink):
        while True:
            frame = write_q.get()
            if frame is None:
                break
            video_sink.write_frame(frame=frame)

    # Heads up! We set 'daemon=True' so this thread won't block program exit
    # if the main thread finishes first.
    t_reader = threading.Thread(target=reader_thread, daemon=True)
    with VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        t_writer = threading.Thread(target=writer_thread, args=(sink,), daemon=True)
        t_reader.start()
        t_writer.start()

        process_bar = tqdm(
            total=total_frames, disable=not show_progress, desc=progress_message
        )

        # Main thread: we take a frame, apply function and update process bar.
        while True:
            item = read_q.get()
            if item is None:
                break
            idx, frame = item
            out = callback(frame, idx)
            write_q.put(out)
            if total_frames is not None:
                process_bar.update(1)

        write_q.put(None)
        t_reader.join()
        t_writer.join()
        process_bar.close()


class FPSMonitor:
    """
    A class for monitoring frames per second (FPS) to benchmark latency.
    """

    def __init__(self, sample_size: int = 30):
        """
        Args:
            sample_size (int): The maximum number of observations for latency
                benchmarking.

        Examples:
            ```python
            import supervision as sv

            frames_generator = sv.get_video_frames_generator(source_path=<SOURCE_FILE_PATH>)
            fps_monitor = sv.FPSMonitor()

            for frame in frames_generator:
                # your processing code here
                fps_monitor.tick()
                fps = fps_monitor.fps
            ```
        """  # noqa: E501 // docs
        self.all_timestamps = deque(maxlen=sample_size)

    @property
    def fps(self) -> float:
        """
        Computes and returns the average FPS based on the stored time stamps.

        Returns:
            float: The average FPS. Returns 0.0 if no time stamps are stored.
        """
        if not self.all_timestamps:
            return 0.0
        taken_time = self.all_timestamps[-1] - self.all_timestamps[0]
        return (len(self.all_timestamps)) / taken_time if taken_time != 0 else 0.0

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
