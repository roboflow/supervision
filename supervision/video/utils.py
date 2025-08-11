from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2


class SourceType(Enum):
    """
    Enumeration of supported video source types.

    Attributes:
        VIDEO_FILE: A standard video file on disk.
        WEBCAM: A webcam or other direct camera device.
        RTSP: A network RTSP video stream.
    """

    VIDEO_FILE = "video_file"
    WEBCAM = "webcam"
    RTSP = "rtsp"

    @classmethod  
    def list(cls):  
        return list(map(lambda c: c.value, cls))  

    @classmethod  
    def from_value(cls, value: SourceType | str) -> SourceType:  
        if isinstance(value, cls):  
            return value  
        if isinstance(value, str):  
            value = value.lower()  
            try:  
                return cls(value)  
            except ValueError:  
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")  
        raise ValueError(  
            f"Invalid value type: {type(value)}. Must be an instance of "  
            f"{cls.__name__} or str."  
        )  


@dataclass
class VideoInfo:
    """
    Stores metadata about a video, such as dimensions, frame rate, and source type.

    Attributes:
        width (int): Width of the video in pixels.
        height (int): Height of the video in pixels.
        fps (int): Frames per second of the video.
        total_frames (int | None): Total number of frames, or None if unknown.
        SourceType (SourceType | None): Source type: VIDEO_FILE, WEBCAM, RTSP.

    Methods:
        from_video_path(video file, webcam, RTSP, or None).

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
    SourceType: SourceType | None = None

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """
        Get the video resolution as a (width, height) tuple.

        Returns:
            tuple[int, int]: The video dimensions in pixels.
        """
        return self.width, self.height
