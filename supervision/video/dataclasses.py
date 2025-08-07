from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from supervision.utils.internal import deprecated


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
    total number of frames.

    Attributes:
        width (int): The width of the video in pixels.
        height (int): The height of the video in pixels.
        fps (int): The frames per second of the video.
        total_frames (Optional[int]): The total number of frames in the video.
            This value can be `None` for video streams.

    Examples:
        ```python
        import supervision as sv

        # Get video info from a file
        video_info = sv.VideoInfo.from_video_path(video_path="my_video.mp4")
        print(video_info)
        # VideoInfo(width=1920, height=1080, fps=30, total_frames=1000)

        # Get video info from a stream
        # Note: total_frames will be None for streams
        video_info = sv.VideoInfo.from_video_path(video_path="rtsp://...")
        print(video_info)
        # VideoInfo(width=1280, height=720, fps=25, total_frames=None)
        ```
    """
    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @classmethod
    @deprecated(
        "VideoInfo.from_video_path is deprecated and will be removed in "
        "supervision-0.23.0. Use `sv.Video.info` instead."
    )
    def from_video_path(cls, video_path: str) -> "VideoInfo":
        from supervision.video.core import Video

        return Video(video_path).info

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """
        Returns the resolution of the video as a tuple `(width, height)`.
        """
        return self.width, self.height

