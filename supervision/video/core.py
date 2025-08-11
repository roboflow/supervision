from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from tqdm.auto import tqdm

from supervision.video.backend import (
    BackendTypes,
    Backend,
    BackendDict,
    WriterTypes
)
from supervision.video.utils import SourceType, VideoInfo


class Video:
    """
    A high-level interface for reading, processing, and writing video files or streams.

    Attributes:
        info (VideoInfo): Metadata about the video.
        source (str | int): Path to the video file or index of the camera device.
        backend (BackendTypes): Video backend used for I/O operations.
    """

    info: VideoInfo
    source: str | int
    backend: BackendTypes

    def __init__(self, source: str | int, backend: Backend | str = Backend.OPENCV) -> None:
        """
        Initialize the Video object.

        Args:
            source (str | int): Path to a video file or index of a camera device.
            backend (BackendLiteral, optional): Backend type for video I/O.
                Defaults to "opencv".
        """
        self.backend = BackendDict.get(Backend.from_value(backend))
        if self.backend is None:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.backend.open(source)
        self.info = self.backend.info()
        self.source = source

    def __iter__(self):
        """
        Make the Video object iterable over frames.

        Yields:
            np.ndarray: The next frame in the video.
        """
        return self.backend.frames()

    def sink(
        self,
        target_path: str,
        info: VideoInfo,
        codec: str | None = None,
        render_audio: bool = False,
    ) -> WriterTypes:
        """
        Create a video writer for saving frames to a file.

        Args:
            target_path (str): Output file path for the video.
            info (VideoInfo): Video information including resolution and FPS.
            codec (str, optional): FourCC video codec code.
                If None, the backend's default codec is used.

        Returns:
            BaseWriter: Video writer instance for writing frames.
        """
        return self.backend.writer(
            target_path, info.fps, info.resolution_wh, codec, self.backend, render_audio
        )

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ):
        """
        Generate frames from the video with optional skipping, cropping, and resizing.

        Args:
            stride (int, optional): Number of frames to skip between each yield.
                Defaults to 1 (no skipping).
            start (int, optional): Index of the first frame to read. Defaults to 0.
            end (int | None, optional): Index after the last frame to read.
                If None, reads until the end of the video.
            resolution_wh (tuple[int, int] | None, optional): Target resolution
                (width, height) for resizing frames. If None, keeps original size.

        Yields:
            np.ndarray: The next frame in the video.
        """
        if self.backend.cap is None:
            raise RuntimeError("Video not opened yet.")

        total_frames = (
            self.backend.video_info.total_frames if self.backend.video_info else 0
        )
        is_live_stream = total_frames <= 0

        if is_live_stream:
            while True:
                for _ in range(stride - 1):
                    if not self.backend.grab():
                        return
                ret, frame = self.backend.read()
                if not ret:
                    return
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame
        else:
            if end is None or end > total_frames:
                end = total_frames

            frame_idx = start
            while frame_idx < end:
                self.backend.seek(frame_idx)
                ret, frame = self.backend.read()
                if not ret:
                    break
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame
                frame_idx += stride

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
        codec: str | None = None,
        render_audio: bool = False,
    ):
        """
        Process and save video frames to a file.

        Args:
            target_path (str): Output file path for the processed video.
            callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
                a numpy ndarray representation of a video frame and an
                int index of the frame and returns a processed numpy ndarray
                representation of the frame.
            fps (int | None, optional): Frames per second of the output video.
                If None, uses the original FPS.
            progress_message (str, optional): Message displayed in the progress bar.
                Defaults to "Processing video".
            show_progress (bool, optional): If True, displays a tqdm progress bar.
                Defaults to False.
            codec (str | None, optional): FourCC video codec code.
                If None, uses the backend's default codec.

        Raises:
            RuntimeError: If the video has not been opened.
            ValueError: If the video source is not a file.

        Returns:
            None
        """
        if self.backend.cap is None:
            raise RuntimeError("Video not opened yet.")

        if self.backend.video_info.SourceType != SourceType.VIDEO_FILE:
            raise ValueError("Only video files can be saved.")

        if fps is None:
            fps = self.backend.video_info.fps

        writer = self.backend.writer(
            target_path,
            fps,
            self.backend.video_info.resolution_wh,
            codec,
            self.backend,
            render_audio,
        )
        total_frames = self.backend.video_info.total_frames
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
            writer.write(frame=result_frame)

        writer.close()
