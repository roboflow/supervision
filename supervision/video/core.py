from typing import Optional, Tuple, Callable

import cv2
import numpy as np
from tqdm import tqdm

from supervision.video.backends import (
    OpenCVBackend,
    OpenCVWriter,
    PyAVBackend,
    PyAVWriter,
    VideoBackend,
    VideoWriter,
)
from supervision.video.dataclasses import VideoInfo


class Video:
    """
    A class for effortless video processing. It allows for iterating over video frames,
    accessing video information, and saving processed videos.

    Attributes:
        source_path (str): The path to the video file, video stream, or webcam.
        backend (VideoBackend): The video backend used for video processing.

    Examples:
        ```python
        import supervision as sv

        # Get video information
        video_info = sv.Video("my_video.mp4").info
        print(video_info)
        # VideoInfo(width=1920, height=1080, fps=30, total_frames=1000)

        # Iterate over video frames
        for frame in sv.Video("my_video.mp4"):
            # process frame
            pass

        # Iterate over a sub-clip of a video
        for frame in sv.Video("my_video.mp4").frames(start=100, end=200):
            # process frame
            pass
        ```
    """

    def __init__(self, source_path: str, backend: str = "opencv"):
        self.source_path = source_path
        if backend == "opencv":
            self.backend: VideoBackend = OpenCVBackend(source_path)
        elif backend == "pyav":
            self.backend: VideoBackend = PyAVBackend(source_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @property
    def info(self) -> VideoInfo:
        """
        Returns information about the video.

        Returns:
            VideoInfo: An object containing video information.
        """
        return self.backend.get_info()

    def __iter__(self):
        """
        Returns an iterator that yields video frames.
        """
        return self

    def __next__(self):
        """
        Returns the next frame of the video.
        """
        success, frame = self.backend.read()
        if not success:
            self.backend.release()
            raise StopIteration
        return frame

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        resolution_wh: Optional[Tuple[int, int]] = None,
    ):
        """
        A generator that yields frames from the video with specific settings.

        Args:
            stride (int, optional): The interval at which frames are returned.
                Defaults to 1.
            start (int, optional): The starting frame index. Defaults to 0.
            end (Optional[int], optional): The ending frame index.
                If None, the video is read to the end. Defaults to None.
            resolution_wh (Optional[Tuple[int, int]], optional): The resolution
                to which frames are resized. If None, frames are not resized.
                Defaults to None.

        Yields:
            np.ndarray: A video frame.
        """
        self.backend.seek(start)
        frame_idx = start
        while True:
            success, frame = self.backend.read()
            if not success or (end is not None and frame_idx >= end):
                self.backend.release()
                break

            if frame_idx % stride == 0:
                if resolution_wh:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame

            frame_idx += 1

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        show_progress: bool = True,
        fps: Optional[int] = None,
        codec: str = "mp4v",
    ):
        """
        Saves a video to a file after processing each frame with a callback.

        Args:
            target_path (str): The path to the output video file.
            callback (Callable[[np.ndarray, int], np.ndarray]): A function that
                takes a frame and its index and returns a processed frame.
            show_progress (bool, optional): If True, a progress bar is displayed.
                Defaults to True.
            fps (Optional[int], optional): The frames per second of the output video.
                If None, the FPS of the source video is used. Defaults to None.
            codec (str, optional): The codec to use for the output video.
                Defaults to "mp4v".
        """
        info = self.info
        if fps:
            info.fps = fps

        if isinstance(self.backend, PyAVBackend):
            codec = "libx264"

        writer_class = (
            OpenCVWriter if isinstance(self.backend, OpenCVBackend) else PyAVWriter
        )
        writer = writer_class(target_path, info, codec=codec)

        with writer as sink:
            for i, frame in enumerate(
                tqdm(
                    self,
                    total=info.total_frames,
                    disable=not show_progress,
                )
            ):
                processed_frame = callback(frame, i)
                sink.write(processed_frame)

    def sink(self, target_path: str, info: VideoInfo):
        """
        Returns a video writer for manual frame-by-frame processing.

        Args:
            target_path (str): The path to the output video file.
            info (VideoInfo): The video information for the output video.

        Returns:
            VideoWriter: A video writer object.
        """
        writer_class = (
            OpenCVWriter if isinstance(self.backend, OpenCVBackend) else PyAVWriter
        )
        return writer_class(target_path, info)

