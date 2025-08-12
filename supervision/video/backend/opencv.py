from __future__ import annotations

import cv2
import numpy as np

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.utils import SourceType, VideoInfo


class OpenCVBackend(BaseBackend):
    """
    OpenCV-based video backend implementation for video capture and processing.

    This backend provides video reading capabilities using OpenCV's VideoCapture.
    It supports:
    - Local video files
    - Webcam streams
    - RTSP network streams

    Attributes:
        cap (cv2.VideoCapture): OpenCV video capture instance.
        video_info (VideoInfo): Metadata about the video source.
        writer (class): Reference to the OpenCVWriter class for video writing.
        path (str | int): Path to the video source or webcam index.

    """

    def __init__(self):
        """Initialize the OpenCV backend with no active capture."""
        self.cap = None
        self.video_info = None
        self.writer = OpenCVWriter
        self.path = None

    def open(self, path: str | int) -> None:
        """
        Open a video source for reading.

        Args:
            path (str | int): Path to video file, RTSP URL, or webcam index.
                Webcam indices are typically 0 for default camera.

        Raises:
            RuntimeError: If the source cannot be opened.
            ValueError: If the source type is unsupported.
        """
        self.cap = cv2.VideoCapture(path)
        self.path = path

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path}")

        self.video_info = self._set_video_info()

        if isinstance(path, int):
            self.video_info.SourceType = SourceType.WEBCAM
        elif isinstance(path, str):
            self.video_info.SourceType = (
                SourceType.RTSP
                if path.lower().startswith("rtsp://")
                else SourceType.VIDEO_FILE
            )
        else:
            raise ValueError("Unsupported source type")

    def isOpened(self) -> bool:
        """
        Check if the video source is currently open and available.

        Returns:
            bool: True if source is open and ready for reading, False otherwise.
        """
        return self.cap.isOpened()

    def _set_video_info(self) -> VideoInfo:
        """
        Extract and store video metadata from the opened source.

        Returns:
            VideoInfo: Object containing:
                - width (int): Frame width in pixels
                - height (int): Frame height in pixels
                - fps (int): Frames per second
                - total_frames (int): Total frame count (0 for streams)

        Raises:
            RuntimeError: If no source is open.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return VideoInfo(width, height, fps, total_frames)

    def info(self) -> VideoInfo:
        """
        Retrieve stored video metadata.

        Returns:
            VideoInfo: Video properties including dimensions, FPS, and frame count.

        Raises:
            RuntimeError: If no source is open.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        return self.video_info

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read and decode the next frame from the video source.

        Returns:
            tuple[bool, np.ndarray]:
                - bool: True if frame was read successfully, False at end of stream
                - np.ndarray: Frame data in BGR format (height, width, 3)

        Raises:
            RuntimeError: If no source is open.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        return self.cap.read()

    def grab(self) -> bool:
        """
        Advance to the next frame without decoding.

        Useful for quickly skipping frames when pixel data isn't needed.

        Returns:
            bool: True if frame was advanced successfully, False otherwise

        Raises:
            RuntimeError: If no source is open.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        return self.cap.grab()

    def seek(self, frame_idx: int) -> None:
        """
        Seek to a specific frame index.

        Note: Seeking may be imprecise with compressed video formats.

        Args:
            frame_idx (int): Zero-based index of target frame.

        Raises:
            RuntimeError: If no source is open.
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
    """
    OpenCV-based video writer for creating video files.

    This writer provides basic video encoding capabilities using OpenCV's VideoWriter.
    Note: Does not support audio writing - use pyAVWriter for audio support.
    """

    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
        backend: OpenCVBackend | None = None,
        render_audio: bool | None = None,
    ):
        """
        Initialize the video writer.

        Args:
            filename (str): Output video file path (e.g., "output.mp4").
            fps (int): Target frames per second for output video.
            frame_size (tuple[int, int]): (width, height) of output frames.
            codec (str, optional): FourCC codec code (default "mp4v").
            backend (OpenCVBackend, optional): Unused (for API compatibility).
            render_audio (bool, optional): Must be None (OpenCV doesn't support audio).

        Raises:
            ValueError: If render_audio is specified (not supported).
            RuntimeError: If writer cannot be initialized.

        Note:
            Falls back to "mp4v" codec if specified codec fails.
        """
        if render_audio or render_audio is False:
            raise ValueError(
                "OpenCV backend does not support audio. "
                "Please use `pyav` backend instead or set `render_audio=None`"
            )

        self.backend = backend
        try:
            fourcc_int = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(filename, fourcc_int, fps, frame_size)
        except Exception:
            fourcc_int = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(filename, fourcc_int, fps, frame_size)

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for file: {filename}")

    def __enter__(self):
        """Enable context manager support (with statement)."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure proper cleanup when exiting context."""
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the output video.

        Args:
            frame (np.ndarray): Frame data in BGR format (height, width, 3).
        """
        self.writer.write(frame)

    def close(self) -> None:
        """Finalize and close the output video file."""
        self.writer.release()