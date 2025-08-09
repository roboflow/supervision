from __future__ import annotations

from dataclasses import dataclass


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
    precise_fps: float | None = None
    total_frames: int | None = None

    @classmethod
    def from_video_path(cls, video_path: str, backend: str | None = None) -> VideoInfo:
        from .core import Video

        return Video(video_path, backend=backend).info

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height
