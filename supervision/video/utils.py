from __future__ import annotations

from dataclasses import dataclass
from typing import cast


@dataclass
class VideoInfo:
    """Static information about a video.

    Stores the width, height, frames-per-second values, and optionally the
    precise frames-per-second and total number of frames.

    Attributes:
        width: Width of the video in pixels.
        height: Height of the video in pixels.
        fps: Rounded frames per second of the video.
        precise_fps: Exact frames per second value when available.
        total_frames: Total number of frames in the video if known.

    Examples:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(video_path="/path/to/video.mp4")

        print(video_info)
        # VideoInfo(width=3840, height=2160, fps=25, precise_fps=25.0, total_frames=538)

        print(video_info.resolution_wh)
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
        """Construct a ``VideoInfo`` from a video file or stream.

        Args:
            video_path: Path or URL to the video file/stream.
            backend: Optional backend name to use (for example, ``"opencv"``).

        Returns:
            VideoInfo: Parsed static information from the source.
        """
        from supervision.video.core import Video  # Avoid circular import

        return cast("VideoInfo", Video(video_path, backend=backend).info)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Return the resolution as ``(width, height)``.

        Returns:
            tuple[int, int]: A tuple of width and height in pixels.
        """
        return self.width, self.height
