from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.utils import SOURCE_TYPE, VideoInfo

import cv2
import numpy as np
from tqdm.auto import tqdm
from typing import Callable

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
        self.writer = None
        self.path = None

    def get_sink(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        """Create a video writer for saving frames using OpenCV.

        Args:
            target_path (str): Path where the video will be saved.
            video_info (VideoInfo): Video information containing resolution and FPS.
            codec (str, optional): FourCC code for video codec. Defaults to "mp4v".

        Returns:
            OpenCVWriter: A video writer object for writing frames.
        """
        return OpenCVWriter(
            target_path, video_info.fps, video_info.resolution_wh, codec
        )

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


    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
    ):
        """Save processed video frames to a file with audio preservation.

        Args:
            target_path (str): Path where the processed video will be saved.
            callback (Callable[[np.ndarray, int], np.ndarray]): Function that processes
                each frame. Takes frame and index as input, returns processed frame.
            fps (int | None, optional): Output video FPS. If None, uses source FPS.
            progress_message (str, optional): Message to show in progress bar.
            show_progress (bool, optional): Whether to show progress bar.

        Raises:
            RuntimeError: If video source is not opened.
            ValueError: If source is not a video file.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened yet.")

        if self.video_info.source_type != SOURCE_TYPE.VIDEO_FILE:
            raise ValueError("Only video files can be saved.")

        if self.writer is not None:
            self.writer.close()
            self.writer = None

        source_codec = self.cap.get(cv2.CAP_PROP_FOURCC)

        if fps is None:
            fps = self.video_info.fps

        self.writer = OpenCVWriter(
            target_path, fps, self.video_info.resolution_wh, source_codec
        )
        total_frames = self.video_info.total_frames
        frames_generator = self.frames()
        for index, frame in enumerate(
            tqdm(
                frames_generator,
                total=total_frames,
                disable=not show_progress,
                desc=progress_message,
            )
        ):
            result_frame = callback(frame, index)
            self.writer.write(frame=result_frame)

        self.writer.close()


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
