from __future__ import annotations

import cv2
import numpy as np

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.utils import SOURCE_TYPE, VideoInfo


class OpenCVBackend(BaseBackend):
    """
    OpenCV implementation of the Backend interface.
    Handles video capture, frame reading, seeking, and writing operations using OpenCV.
    """

    def __init__(self):
        """Initialize the OpenCV backend with empty video capture and writer objects."""
        super().__init__()
        self.cap = None
        self.video_info = None
        self.writer = OpenCVWriter
        self.path = None

    def open(self, path: str) -> None:
        """
        Open a video source and initialize the video capture object.

        Args:
            path (str): Path to the video file, RTSP URL, or camera index.

        Raises:
            RuntimeError: If unable to open the video source.
            ValueError: If the source type is not supported.
        """
        self.cap = cv2.VideoCapture(path)
        self.path = path

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path}")
        self.video_info = self._set_video_info()

        if isinstance(path, int):
            self.video_info.source_type = SOURCE_TYPE.WEBCAM
        elif isinstance(path, str):
            self.video_info.source_type = (
                SOURCE_TYPE.RTSP
                if path.lower().startswith("rtsp://")
                else SOURCE_TYPE.VIDEO_FILE
            )
        else:
            raise ValueError("Unsupported source type")

    def isOpened(self) -> bool:
        """Check if the video source is opened successfully.

        Returns:
            bool: True if the video source is opened, False otherwise.
        """
        return self.cap.isOpened()

    def _set_video_info(self) -> VideoInfo:
        """Set up video information from the opened video source.

        Returns:
            VideoInfo: Object containing video properties like width, height, fps, etc.

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(width, height, fps, total_frames)

    def info(self) -> VideoInfo:
        """Get video information.

        Returns:
            VideoInfo: Object containing video properties.

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        return self.video_info

    def read(self) -> tuple[bool, np.ndarray]:
        """Read a frame from the video source.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing:
                - bool: True if frame was successfully read
                - np.ndarray: The video frame in BGR format

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        ret, frame = self.cap.read()
        return ret, frame

    def grab(self) -> bool:
        """Grab a frame from video source without decoding.

        Returns:
            bool: True if frame was successfully grabbed.

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        return self.cap.grab()

    def seek(self, frame_idx: int) -> None:
        """Seek to a specific frame in the video.

        Args:
            frame_idx (int): Index of the frame to seek to (0-based).

        Raises:
            RuntimeError: If the video source is not opened yet.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self) -> None:
        """Release the video capture resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None


class OpenCVWriter(BaseWriter):
    """A class to handle video writing operations using OpenCV's VideoWriter.

    This class provides an interface to write frames to a video file using OpenCV,
    with support for different codecs and automatic fallback to mp4v if the specified
    codec fails.
    """

    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ):
        """Initialize the video writer.

        Args:
            filename (str): Path to the output video file.
            fps (int): Frames per second for the output video.
            frame_size (tuple[int, int]): Width and height of the output video frames.
            codec (str, optional): FourCC code for the video codec. Defaults to "mp4v".

        Raises:
            RuntimeError: If the video writer cannot be initialized.
        """
        try:
            fourcc_int = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(filename, fourcc_int, fps, frame_size)
        except Exception:
            fourcc_int = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(filename, fourcc_int, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for file: {filename}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video file.

        Args:
            frame (np.ndarray): The frame to write, in BGR format.
        """
        self.writer.write(frame)

    def close(self) -> None:
        """Release the video writer resources."""
        self.writer.release()
