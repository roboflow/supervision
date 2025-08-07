from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Generator

import numpy as np
from tqdm.auto import tqdm

from supervision.utils.internal import deprecated
from supervision.video import Video
from supervision.video.dataclasses import VideoInfo




@deprecated(
    "VideoSink is deprecated and will be removed in supervision-0.23.0. "
    "Use `sv.Video(...).sink(...)` instead."
)
class VideoSink:
    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        self.video = Video(target_path)
        self.writer = self.video.sink(target_path, video_info)

    def __enter__(self):
        return self.writer

    def write_frame(self, frame: np.ndarray):
        self.writer.write(frame)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.writer.release()




@deprecated(
    "`get_video_frames_generator` is deprecated and will be removed in "
    "supervision-0.23.0. Use `sv.Video(...)` instead."
)
def get_video_frames_generator(
    source_path: str,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
) -> Generator[np.ndarray, None, None]:
    video = Video(source_path)
    yield from video.frames(stride=stride, start=start, end=end)


@deprecated(
    "`process_video` is deprecated and will be removed in supervision-0.23.0. "
    "Use `sv.Video(...).save(...)` instead."
)
def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    target_fps: Optional[int] = None,
    show_progress: bool = True,
):
    video = Video(source_path)
    video.save(
        target_path,
        callback,
        show_progress=show_progress,
        fps=target_fps,
    )


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
