from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass

import cv2
import numpy as np

from supervision.utils.internal import deprecated, warn_deprecated
from supervision.video import Video as _NewVideo
from supervision.video import VideoInfo as _NewVideoInfo


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
    fps: float
    total_frames: int | None = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        warn_deprecated(
            "VideoInfo.from_video_path is deprecated and will be removed in at least 5 releases. "
            "Use sv.Video(video_path).info instead."
        )
        v = _NewVideo(video_path)
        info = v.info
        # Map new VideoInfo to legacy class
        return VideoInfo(
            width=info.width,
            height=info.height,
            fps=info.fps,
            total_frames=info.total_frames,
        )

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height


@deprecated(
    "Use sv.Video(...).sink(...) for manual writers. This alias will be supported for at least 5 releases."
)
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
        # Create writer via OpenCV backend by default
        from supervision.video import OpenCVBackend  # type: ignore

        backend = OpenCVBackend()
        writer = backend.writer(
            self.target_path,
            _NewVideoInfo(
                width=self.video_info.width,
                height=self.video_info.height,
                fps=float(self.video_info.fps),
                total_frames=self.video_info.total_frames,
            ),
            self.__codec,
        )
        self.__writer = writer
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
        self.__writer.close()


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


@deprecated(
    "Use sv.Video(source).frames(...) or iterate over sv.Video(source). This function will be supported for at least 5 releases."
)
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
    # Route to new API
    video = _NewVideo(source_path)
    yield from video.frames(
        stride=stride,
        start=start,
        end=end,
        iterative_seek=iterative_seek,
    )


@deprecated(
    "Use sv.Video(source).save(target, callback=...). This function will be supported for at least 5 releases."
)
def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    max_frames: int | None = None,
    show_progress: bool = False,
    progress_message: str = "Processing video",
) -> None:
    """
    Process a video file by applying a callback function on each frame
        and saving the result to a target video file.

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
            a numpy ndarray representation of a video frame and an
            int index of the frame and returns a processed numpy ndarray
            representation of the frame.
        max_frames (Optional[int]): The maximum number of frames to process.
        show_progress (bool): Whether to show a progress bar.
        progress_message (str): The message to display in the progress bar.

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
    """
    video = _NewVideo(source_path)
    video.save(
        target_path,
        callback=callback,
        show_progress=show_progress,
        progress_message=progress_message,
        end=max_frames,
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
