from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from queue import Queue, Full, Empty

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
    skip_on_error: bool = False,
) -> None:
    """Process video frames using a three-stage threaded pipeline with controlled memory usage.

    Reads frames in a background thread, processes them via user callback in the main thread,
    and writes results in another background thread. Uses bounded queues to limit memory.

    Args:
        source_path: Path to the input video file.
        target_path: Path where the processed video will be saved.
        callback: Function called for each frame. Receives frame (`numpy.ndarray`, shape `(H, W, 3)`)
            and zero-based frame index; must return processed frame of the same shape.
        max_frames: Maximum number of frames to process. If `None`, processes entire video.
        prefetch: Maximum number of raw frames kept in memory before processing.
        writer_buffer: Maximum number of processed frames kept in memory before writing.
        show_progress: Whether to display a tqdm progress bar.
        progress_message: Text shown in the progress bar when enabled.
        skip_on_error: If `True`, silently skip frames where callback raises an exception.
            If `False` (default), exception is logged and re-raised after cleanup.

    Raises:
        RuntimeError: When source video cannot be opened.
        Exception: Any unhandled exception raised by the callback (unless `skip_on_error=True`).
    """
    video_info = VideoInfo.from_video_path(video_path=source_path)
    total_frames = (
        min(video_info.total_frames, max_frames)
        if max_frames is not None and video_info.total_frames is not None
        else video_info.total_frames
    )

    frame_read_queue: Queue[tuple[int, np.ndarray] | None] = Queue(maxsize=prefetch)
    frame_write_queue: Queue[np.ndarray | None] = Queue(maxsize=writer_buffer)

    stop_event = threading.Event()

    def reader_thread() -> None:
        video = cv2.VideoCapture(source_path)
        try:
            if not video.isOpened():
                raise RuntimeError(f"Cannot open video: {source_path}")

            frame_generator = get_video_frames_generator(
                source_path=source_path,
                end=max_frames,
            )

            for frame_index, frame in enumerate(frame_generator):
                if stop_event.is_set():
                    break

                # non-blocking put with small timeout + backoff prevents tight CPU loop
                while not stop_event.is_set():
                    try:
                        frame_read_queue.put((frame_index, frame), timeout=0.1)
                        break
                    except Full:
                        time.sleep(0.01)  # light backoff
        finally:
            video.release()
            # best-effort sentinel – never block forever during shutdown
            try:
                frame_read_queue.put(None, timeout=0.1)
            except Full:
                pass

    def writer_thread(video_sink: VideoSink) -> None:
        while not stop_event.is_set():
            try:
                frame = frame_write_queue.get(timeout=0.1)
                if frame is None:
                    break
                video_sink.write_frame(frame=frame)
            except Empty:
                continue

    # Reader is non-daemon
    reader_worker = threading.Thread(target=reader_thread, daemon=False)

    with VideoSink(target_path=target_path, video_info=video_info) as video_sink:
        # Writer remains daemon
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

        raised: BaseException | None = None

        try:
            while True:
                try:
                    read_item = frame_read_queue.get(timeout=0.5)
                    if read_item is None:
                        break
                    frame_index, frame = read_item
                except Empty:
                    if stop_event.is_set():
                        break
                    continue

                try:
                    processed_frame = callback(frame, frame_index)
                    frame_write_queue.put(processed_frame)
                    progress_bar.update(1)
                except Exception as exc:
                    print(f"Error processing frame {frame_index}: {exc}")
                    if not skip_on_error:
                        raised = exc
                        break
                    # else: skip this frame silently

        finally:
            stop_event.set()

            # best-effort sentinel for writer – never block shutdown
            try:
                frame_write_queue.put(None, timeout=0.1)
            except Full:
                pass

            # Give threads reasonable time to finish cleanly
            reader_worker.join(timeout=10.0)
            if reader_worker.is_alive():
                print("Reader thread did not finish in time")

            writer_worker.join(timeout=5.0)
            if writer_worker.is_alive():
                print("Writer thread did not finish in time")

            progress_bar.close()

        if raised is not None:
            raise raised


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
