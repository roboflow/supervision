from __future__ import annotations

from collections.abc import Callable

import numpy as np
import cv2
from tqdm.auto import tqdm

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.backend.openCV import OpenCVBackend
from supervision.video.utils import VideoInfo, SOURCE_TYPE
from supervision.video.backend.pyAV import pyAVBackend


class Video:
    """High-level interface for video operations.

    This class provides a convenient interface for video operations including
    reading frames, saving processed videos, and video information access.
    """

    info: VideoInfo
    source: str | int
    backend: BaseBackend

    def __init__(
        self, source: str | int, info: VideoInfo | None = None, backend: str = "opencv"
    ):
        if backend == "opencv":
            self.backend = pyAVBackend()

        self.backend.open(source)
        self.info = self.backend.video_info
        self.source = source

    def __iter__(self):
        """Make the Video class iterable over frames.

        Returns:
            Generator: A generator yielding video frames.
        """
        return self.backend.frames()

    def sink(
        self, target_path: str, info: VideoInfo, codec: str | None = None
    ) -> BaseWriter:
        """Create a video writer for saving frames.

        Args:
            target_path (str): Path where the video will be saved.
            info (VideoInfo): Video information containing resolution and FPS.
            codec (str, optional): FourCC code for video codec. Defaults to "None".

        Returns:
            Writer: A video writer object for writing frames.
        """
        return self.backend.writer(
             target_path, info.fps, info.resolution_wh, codec
        )

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ):
        """Generate frames from the video.

        Args:
            stride (int, optional): Number of frames to skip. Defaults to 1.
            start (int, optional): Starting frame index. Defaults to 0.
            end (int | None, optional): Ending frame index. Defaults to None.
            resolution_wh (tuple[int, int] | None, optional): Target resolution
                (width, height). If provided, frames will be resized. Defaults to None.

        Returns:
            Generator: A generator yielding video frames.
        """
        if self.backend.cap is None:
            raise RuntimeError("Video not opened yet.")

        total_frames = self.backend.video_info.total_frames if self.backend.video_info else 0
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
    ):
        """Save processed video frames to a file.

        Args:
            target_path (str): Path where the processed video will be saved.
            callback (Callable[[np.ndarray, int], np.ndarray]): Function that processes
                each frame. Takes frame and index as input, returns processed frame.
            fps (int | None, optional): Output video FPS.
            progress_message (str, optional): Message to show in progress bar.
                Defaults to "Processing video".
            show_progress (bool, optional): Whether to show progress bar.
                Defaults to False.
        """
        if self.backend.cap is None:
            raise RuntimeError("Video not opened yet.")

        if self.backend.video_info.source_type != SOURCE_TYPE.VIDEO_FILE:
            raise ValueError("Only video files can be saved.")

        if fps is None:
            fps = self.backend.video_info.fps

        writer = self.backend.writer(
            target_path, fps, self.backend.video_info.resolution_wh, codec, self.backend
        )
        total_frames = self.backend.video_info.total_frames
        print(self.backend.video_info)
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
