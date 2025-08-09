from dataclasses import dataclass
from enum import Enum

import cv2


class SOURCE_TYPE(Enum):
    VIDEO_FILE = "VIDEO_FILE"
    WEBCAM = "WEBCAM"
    RTSP = "RTSP"


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
        source_type (Optional[SOURCE_TYPE]): source type of the video,
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
    total_frames: int = None
    source_type: SOURCE_TYPE = None

    @classmethod
    def from_video_path(cls, video_path: str) -> "VideoInfo":
        """Create VideoInfo from a video file path.

        Args:
            video_path (str): Path to the video file.

        Returns:
            VideoInfo: Video info containing width, height, fps, and total frames.

        Raises:
            ValueError: If video cannot be opened or has invalid properties.
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
                total_frames = None  # Some video formats may not report frame count
        finally:
            video.release()

        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Get the video resolution as (width, height).

        Returns:
            Tuple[int, int]: Video dimensions as (width, height).
        """
        return self.width, self.height
