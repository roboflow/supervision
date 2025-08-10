from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2


class SOURCE_TYPE(Enum):
    """
    Enumeration of supported video source types.

    Attributes:
        VIDEO_FILE: A standard video file on disk.
        WEBCAM: A webcam or other direct camera device.
        RTSP: A network RTSP video stream.
    """

    VIDEO_FILE = "VIDEO_FILE"
    WEBCAM = "WEBCAM"
    RTSP = "RTSP"


@dataclass
class VideoInfo:
    """
    Stores metadata about a video, such as dimensions, frame rate, and source type.

    Attributes:
        width (int): Width of the video in pixels.
        height (int): Height of the video in pixels.
        fps (int): Frames per second of the video.
        total_frames (int | None): Total number of frames, or None if unknown.
        source_type (SOURCE_TYPE | None): The source type of the video (file, webcam, RTSP), or None.

    Examples:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path("video.mp4")

        print(video_info)
        # VideoInfo(width=3840, height=2160, fps=25, total_frames=538)

        print(video_info.resolution_wh)
        # (3840, 2160)
        ```
    """

    width: int
    height: int
    fps: int
    total_frames: int | None = None
    source_type: SOURCE_TYPE | None = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        """
        Create a VideoInfo instance from a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            VideoInfo: Metadata including width, height, FPS, and total frames.

        Raises:
            ValueError: If the video cannot be opened or has invalid properties.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video at {video_path}")

        try:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid video dimensions: {width}x{height}")

            fps = video.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default to 30fps if invalid
            fps = round(fps)

            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 0:
                total_frames = None  # Some formats may not report frame count
        finally:
            video.release()

        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """
        Get the video resolution as a (width, height) tuple.

        Returns:
            tuple[int, int]: The video dimensions in pixels.
        """
        return self.width, self.height
