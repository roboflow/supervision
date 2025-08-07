from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from supervision.utils.internal import deprecated, warn_deprecated
from supervision.video import Video


@deprecated(
    "VideoInfo is replaced by sv.Video.info. This class will be removed in 5 releases."
)
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: int
    total_frames: int | None = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        warn_deprecated(
            "VideoInfo.from_video_path is deprecated. Use sv.Video(video_path).info instead."
        )
        return Video(video_path).info

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height


@deprecated(
    "VideoSink is replaced by sv.Video.sink(). This class will be removed in 5 releases."
)
class VideoSink:
    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        warn_deprecated("VideoSink is deprecated. Use sv.Video.sink() instead.")
        self._video = None
        self._sink = None
        self.target_path = target_path
        self.video_info = video_info
        self.codec = codec

    def __enter__(self):
        self._video = Video(None)  # Dummy, just to access sink
        self._sink = Video.sink(
            self._video, self.target_path, self.video_info, self.codec
        )
        return self

    def write_frame(self, frame: np.ndarray):
        self._sink.write(frame)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._sink.close()


def _validate_and_setup_video(
    source_path: str, start: int, end: int | None, iterative_seek: bool = False
):
    warn_deprecated(
        "_validate_and_setup_video is deprecated and will be removed in 5 releases."
    )
    video = Video(source_path)
    total_frames = video.info.total_frames
    if end is not None and end > total_frames:
        raise Exception("Requested frames are outbound")
    start = max(start, 0)
    end = min(end, total_frames) if end is not None else total_frames
    return video, start, end


def get_video_frames_generator(
    source_path: str,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    iterative_seek: bool = False,
) -> Generator[np.ndarray]:
    warn_deprecated(
        "get_video_frames_generator is deprecated. Use sv.Video(...).frames() instead."
    )
    video = Video(source_path)
    for idx, frame in enumerate(video.frames(stride=stride, start=start, end=end)):
        yield frame


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    max_frames: int | None = None,
    show_progress: bool = False,
    progress_message: str = "Processing video",
) -> None:
    warn_deprecated("process_video is deprecated. Use sv.Video(...).save() instead.")
    video = Video(source_path)
    total_frames = video.info.total_frames
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    frames = video.frames(end=max_frames)
    if show_progress:
        frames = tqdm(frames, total=total_frames, desc=progress_message)
    for idx, frame in enumerate(frames):
        result_frame = callback(frame, idx)
        if idx == 0:
            sink = video.sink(target_path)
        sink.write(result_frame)
    sink.close()


@deprecated("FPSMonitor is deprecated and will be removed in 5 releases.")
class FPSMonitor:
    def __init__(self, sample_size: int = 30):
        warn_deprecated("FPSMonitor is deprecated and will be removed in 5 releases.")
        self.all_timestamps = deque(maxlen=sample_size)

    @property
    def fps(self) -> float:
        if not self.all_timestamps:
            return 0.0
        taken_time = self.all_timestamps[-1] - self.all_timestamps[0]
        return (len(self.all_timestamps)) / taken_time if taken_time != 0 else 0.0

    def tick(self) -> None:
        self.all_timestamps.append(time.monotonic())

    def reset(self) -> None:
        self.all_timestamps.clear()
