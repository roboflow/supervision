from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

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

    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__writer = None

    def __enter__(self) -> VideoSink:
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*self.__codec)
        except TypeError as e:
            logger.warning("%s. Defaulting to mp4v...", str(e))
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        return self

    def write_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """
        Writes a single video frame to the target video file.

        Args:
            frame: The video frame to be written to the file. The frame
                must be in BGR color format.
        """
        if self.__writer is not None:
            self.__writer.write(frame)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: Any,
    ) -> None:
        if self.__writer is not None:
            self.__writer.release()


def _mux_audio(source_path: str, video_path: str) -> None:
    """Mux audio from `source_path` into `video_path` in-place using ffmpeg.

    Args:
        source_path: Path to the original video file containing the audio stream.
        video_path: Path to the video-only file to be updated with audio.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning(
            "ffmpeg not found on PATH. Audio will not be preserved. "
            "Install ffmpeg to enable audio preservation."
        )
        return

    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=os.path.splitext(video_path)[1],
            dir=os.path.dirname(os.path.abspath(video_path)),
        )
        os.close(tmp_fd)
        result = subprocess.run(  # noqa: S603
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-nostats",
                "-i",
                video_path,
                "-i",
                source_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                tmp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace").strip()
            logger.warning(
                "ffmpeg failed to mux audio (return code %d)%s. "
                "The output video will not have audio.",
                result.returncode,
                f": {stderr_msg}" if stderr_msg else "",
            )
            return
        os.replace(tmp_path, video_path)
    except Exception as exc:
        logger.warning(
            "Audio muxing failed: %s. Output video will not have audio.", exc
        )
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


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

    Returns:
        A generator that yields the
            frames of the video.

    Examples:
        ```python
        import supervision as sv

        for frame in sv.get_video_frames_generator(source_path="<SOURCE_VIDEO_PATH>"):
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
        if frame is not None:
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
            `target_path` after frame processing. Requires `ffmpeg` on PATH
            (e.g. `apt install ffmpeg`, `brew install ffmpeg`). If ffmpeg is
            not found or the mux step fails, a warning is logged and the output
            video is saved without audio — no exception is raised. Audio is
            truncated to match the processed video duration. Default is False.

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
                # Queue is full; this is a best-effort attempt to enqueue the sentinel.
                # If we cannot enqueue it, the writer thread will still complete based
                # on previously queued frames or other shutdown conditions.
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

    def __init__(self, sample_size: int = 30):
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
            The average FPS. Returns 0.0 if no time stamps are stored.
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
