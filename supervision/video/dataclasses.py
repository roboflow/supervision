from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass
class VideoInfo:
    """
    Data class containing information about the video resolution, fps, and total frame count.

    :param width: int : The width of the video frames in pixels.
    :param height: int : The height of the video frames in pixels.
    :param fps: int : The frames per second of the video.
    :param total_frames: int : The total number of frames in the video.
    """
    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        """
        Returns a VideoInfo data class containing information about the video resolution, fps, and total frame count.

        :param video_path: str : The path of the video file.
        :return: VideoInfo : A data class containing information about the video resolution, fps, and total frame count.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.width, self.height
