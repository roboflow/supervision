from __future__ import annotations

import cv2
import numpy as np

from supervision.video.backend import BaseBackend, BaseWriter
from supervision.video.utils import SOURCE_TYPE, VideoInfo


class OpenCVBackend(BaseBackend):
    """
    OpenCV-based implementation of the video backend interface.

    Provides methods for opening video sources, reading frames, seeking,
    grabbing, and retrieving metadata using OpenCV.
    """

    def __init__(self):
        """Initialize with no active capture, writer, or path."""
        self.cap = None
        self.video_info = None
        self.writer = OpenCVWriter
        self.path = None

    def open(self, path: str | int) -> None:
        """
        Open a video source and initialize capture.

        Args:
            path (str | int): Path to a video file, RTSP URL, or webcam index.

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
        """
        Check if the video source is currently open.

        Returns:
            bool: True if the source is open, False otherwise.
        """
        return self.cap.isOpened()

    def _set_video_info(self) -> VideoInfo:
        """
        Extract and store video metadata from the open capture.

        Returns:
            VideoInfo: Video properties such as width, height, FPS, and frame count.

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
        Get the stored video metadata.

        Returns:
            VideoInfo: Metadata for the open source.

        Raises:
            RuntimeError: If no source is open.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")
        return self.video_info

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read the next frame from the source.

        Returns:
            tuple[bool, np.ndarray]:
                - bool: True if a frame was read successfully.
                - np.ndarray: The frame in BGR format.

        Raises:
            RuntimeError: If no source is open.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        return self.cap.read()

    def grab(self) -> bool:
        """
        Grab the next frame without decoding.

        Returns:
            bool: True if the frame pointer advanced successfully.

        Raises:
            RuntimeError: If no source is open.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        return self.cap.grab()

    def seek(self, frame_idx: int) -> None:
        """
        Jump to a specific frame.

        Args:
            frame_idx (int): Zero-based frame index to seek to.

        Raises:
            RuntimeError: If no source is open.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self) -> None:
        """Release capture resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None


class OpenCVWriter(BaseWriter):
    """
    Video writer implementation using OpenCV's VideoWriter.

    Supports configurable codecs, frame sizes, and FPS, with a fallback
    to "mp4v" if the specified codec fails.
    """

    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
        backend: OpenCVBackend | None = None,
        render_audio: bool = False,
    ):
        """
        Initialize the writer.

        Args:
            filename (str): Output video file path.
            fps (int): Output frames per second.
            frame_size (tuple[int, int]): Frame dimensions (width, height).
            codec (str, optional): FourCC codec code. Defaults to "mp4v".
            backend (OpenCVBackend | None, optional): Backend instance. Defaults to None

        Raises:
            RuntimeError: If the writer cannot be opened.
        """
        if render_audio:
            raise ValueError("OpenCV backend does not support audio. " \
            "Please use `pyav` backend instead or set `render_audio=False`")
        
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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the output.

        Args:
            frame (np.ndarray): Frame in BGR format.
        """
        self.writer.write(frame)

    def close(self) -> None:
        """Release writer resources."""
        self.writer.release()
