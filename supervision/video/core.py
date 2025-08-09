from collections.abc import Callable

import numpy as np

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.backend.openCV import OpenCVBackend
from supervision.video.utils import VideoInfo


class Video:
    """High-level interface for video operations.

    This class provides a convenient interface for video operations including
    reading frames, saving processed videos, and video information access.
    """

    info: VideoInfo
    source: str | int
    backend: BaseBackend

    def __init__(
        self, source: str | int, info: VideoInfo | None = None, backend: str = "opencv"
    ):
        if backend == "opencv":
            self.backend = OpenCVBackend()

        self.backend.open(source)
        self.info = self.backend.video_info
        self.source = source

    def __iter__(self):
        """Make the Video class iterable over frames.

        Returns:
            Generator: A generator yielding video frames.
        """
        return self.backend.frames()

    def sink(
        self, target_path: str, info: VideoInfo, codec: str = "mp4v"
    ) -> BaseWriter:
        """Create a video writer for saving frames.

        Args:
            target_path (str): Path where the video will be saved.
            info (VideoInfo): Video information containing resolution and FPS.
            codec (str, optional): FourCC code for video codec. Defaults to "mp4v".

        Returns:
            Writer: A video writer object for writing frames.
        """
        return self.backend.get_sink(target_path, info, codec)

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ):
        """Generate frames from the video.

        Args:
            stride (int, optional): Number of frames to skip. Defaults to 1.
            start (int, optional): Starting frame index. Defaults to 0.
            end (int | None, optional): Ending frame index. Defaults to None.
            resolution_wh (tuple[int, int] | None, optional): Target resolution
                (width, height). If provided, frames will be resized. Defaults to None.

        Returns:
            Generator: A generator yielding video frames.
        """
        return self.backend.frames(
            stride=stride, start=start, end=end, resolution_wh=resolution_wh
        )

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
        codec: str = "mp4v",
    ):
        """Save processed video frames to a file.

        Args:
            target_path (str): Path where the processed video will be saved.
            callback (Callable[[np.ndarray, int], np.ndarray]): Function that processes
                each frame. Takes frame and index as input, returns processed frame.
            fps (int | None, optional): Output video FPS.
            progress_message (str, optional): Message to show in progress bar.
                Defaults to "Processing video".
            show_progress (bool, optional): Whether to show progress bar.
                Defaults to False.
        """
        self.backend.save(
            target_path=target_path,
            callback=callback,
            fps=fps,
            progress_message=progress_message,
            show_progress=show_progress,
            codec=codec,
        )
