"""New Video API implementation."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Union

import cv2
import numpy as np
from tqdm.auto import tqdm

from supervision.utils.video_backend import VideoInfo, Writer, get_backend


class Video:
    """
    A unified interface for video processing supporting files, streams, and webcams.

    This class provides a comprehensive API for video manipulation including:
    - Reading video information
    - Frame iteration with advanced options
    - Video processing and saving
    - Multi-backend support (OpenCV and PyAV)

    Attributes:
        source: Path to video file, RTSP URL, or webcam index
        backend: Video backend to use for processing
        info: Video information (lazy-loaded)

    Examples:
        Basic usage with video file:
        ```python
        import supervision as sv

        # Get video information
        video = sv.Video("source.mp4")
        print(video.info)
        # VideoInfo(width=1920, height=1080, fps=30.0, total_frames=1500)

        # Simple frame iteration
        for frame in video:
            # Process frame
            pass
        ```

        Advanced frame iteration:
        ```python
        import supervision as sv

        # Iterate with stride, sub-clip, and resize
        video = sv.Video("source.mp4")
        for frame in video.frames(
            stride=2,
            start=100,
            end=500,
            resolution_wh=(640, 480)
        ):
            # Process every 2nd frame from 100 to 500, resized
            pass
        ```

        Process and save video:
        ```python
        import cv2
        import supervision as sv

        def blur_callback(frame: np.ndarray, index: int) -> np.ndarray:
            return cv2.GaussianBlur(frame, (11, 11), 0)

        video = sv.Video("source.mp4")
        video.save(
            "output.mp4",
            callback=blur_callback,
            show_progress=True
        )
        ```

        Using different backends:
        ```python
        import supervision as sv

        # Use PyAV backend for better codec support
        video = sv.Video("source.mkv", backend="pyav")

        # Use OpenCV backend for compatibility
        video = sv.Video("source.mp4", backend="opencv")
        ```

        Manual control with VideoSink:
        ```python
        import supervision as sv

        source = sv.Video("source.mp4")
        target_info = sv.VideoInfo(width=800, height=800, fps=24)

        with source.sink("output.mp4", info=target_info) as sink:
            for frame in source.frames():
                # Custom processing
                frame = cv2.resize(frame, target_info.resolution_wh)
                sink.write(frame)
        ```
    """

    def __init__(
        self,
        source: str | int,
        backend: str | None = None,
    ):
        """
        Initialize a Video object.

        Args:
            source: Path to video file, RTSP URL, or webcam index (0, 1, etc.)
            backend: Backend to use ('opencv', 'pyav', or None for auto-selection)
        """
        self.source = source
        self._backend = get_backend(backend)
        self._info = None
        self._handle = None

    @property
    def info(self) -> VideoInfo:
        """
        Get video information.

        Returns:
            VideoInfo object containing video metadata

        Note:
            This property is lazy-loaded and cached. The video is opened
            temporarily to read metadata if not already cached.
        """
        if self._info is None:
            handle = self._backend.open(self.source)
            try:
                self._info = self._backend.info(handle)
            finally:
                self._backend.close(handle)
        return self._info

    def __iter__(self) -> Generator[np.ndarray]:
        """
        Iterate over all frames in the video.

        Yields:
            Video frames as numpy arrays in BGR format
        """
        yield from self.frames()

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
        iterative_seek: bool = False,
    ) -> Generator[np.ndarray]:
        """
        Generate frames from the video with advanced options.

        Args:
            stride: Return every nth frame (1 = every frame, 2 = every other frame)
            start: Starting frame index (0-based)
            end: Ending frame index (exclusive), None for end of video
            resolution_wh: Resize frames to (width, height), None to keep original
            iterative_seek: If True, seek by grabbing frames (slower but more compatible)

        Yields:
            Video frames as numpy arrays in BGR format

        Examples:
            ```python
            import supervision as sv

            video = sv.Video("source.mp4")

            # Get every 5th frame
            for frame in video.frames(stride=5):
                pass

            # Get frames 100-500
            for frame in video.frames(start=100, end=500):
                pass

            # Get resized frames
            for frame in video.frames(resolution_wh=(640, 480)):
                pass
            ```
        """
        handle = self._backend.open(self.source)
        try:
            info = self._backend.info(handle)

            # Validate and adjust bounds
            if end is not None and info.total_frames is not None:
                if end > info.total_frames:
                    warnings.warn(
                        f"Requested end frame {end} exceeds total frames {info.total_frames}"
                    )
                    end = info.total_frames
            elif end is None and info.total_frames is not None:
                end = info.total_frames

            start = max(start, 0)

            # Seek to start position
            if iterative_seek:
                # Grab frames iteratively (slower but more compatible)
                for _ in range(start):
                    if not self._backend.grab(handle):
                        return
            else:
                # Direct seek (faster but may not work with all formats)
                if start > 0:
                    self._backend.seek(handle, start)

            frame_idx = start
            while True:
                # Check if we've reached the end
                if end is not None and frame_idx >= end:
                    break

                # Read the frame
                success, frame = self._backend.read(handle)
                if not success or frame is None:
                    break

                # Resize if requested
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)

                yield frame

                # Skip frames according to stride
                for _ in range(stride - 1):
                    if end is not None and frame_idx + 1 >= end:
                        break
                    if not self._backend.grab(handle):
                        return
                    frame_idx += 1

                frame_idx += 1

        finally:
            self._backend.close(handle)

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray] | None = None,
        codec: str | None = None,
        fps: float | None = None,
        resolution_wh: tuple[int, int] | None = None,
        max_frames: int | None = None,
        show_progress: bool = False,
        progress_message: str = "Processing video",
        **kwargs,
    ) -> None:
        """
        Process and save the video to a file.

        Args:
            target_path: Path to save the output video
            callback: Function to process each frame, receives (frame, index) and returns processed frame
            codec: Video codec to use (e.g., 'mp4v', 'h264'), None for default
            fps: Output video FPS, None to use source FPS
            resolution_wh: Output resolution as (width, height), None to use source resolution
            max_frames: Maximum number of frames to process, None for all frames
            show_progress: Show progress bar during processing
            progress_message: Message to display in progress bar
            **kwargs: Additional arguments passed to frames() method

        Examples:
            Simple copy:
            ```python
            video = sv.Video("source.mp4")
            video.save("copy.mp4")
            ```

            With processing:
            ```python
            def enhance(frame, index):
                return cv2.convertScaleAbs(frame, alpha=1.5, beta=10)

            video.save("enhanced.mp4", callback=enhance, show_progress=True)
            ```

            Change parameters:
            ```python
            video.save(
                "output.mp4",
                fps=60,
                resolution_wh=(1280, 720),
                codec="h264"
            )
            ```
        """
        # Prepare output video info
        source_info = self.info
        output_info = VideoInfo(
            width=resolution_wh[0] if resolution_wh else source_info.width,
            height=resolution_wh[1] if resolution_wh else source_info.height,
            fps=fps if fps is not None else source_info.fps,
            total_frames=min(max_frames, source_info.total_frames)
            if max_frames and source_info.total_frames
            else max_frames,
            codec=codec,
        )

        # Create writer
        writer = self._backend.writer(target_path, output_info, codec)

        try:
            # Calculate total frames for progress bar
            total_frames = output_info.total_frames
            if total_frames is None and source_info.total_frames is not None:
                end = kwargs.get("end", source_info.total_frames)
                start = kwargs.get("start", 0)
                stride = kwargs.get("stride", 1)
                total_frames = (min(end, source_info.total_frames) - start) // stride
                if max_frames:
                    total_frames = min(total_frames, max_frames)

            # Create frame generator
            frame_generator = self.frames(resolution_wh=resolution_wh, **kwargs)

            # Add progress bar if requested
            if show_progress and total_frames:
                frame_generator = tqdm(
                    frame_generator,
                    total=total_frames,
                    desc=progress_message,
                )

            # Process and write frames
            for index, frame in enumerate(frame_generator):
                if max_frames and index >= max_frames:
                    break

                # Apply callback if provided
                if callback:
                    frame = callback(frame, index)

                writer.write(frame)

        finally:
            writer.close()

    @contextmanager
    def sink(
        self,
        target_path: str,
        info: VideoInfo | None = None,
        codec: str | None = None,
    ):
        """
        Context manager for writing video frames with manual control.

        Args:
            target_path: Path to save the output video
            info: Video information for output, None to use source info
            codec: Video codec to use, None for default

        Yields:
            VideoSink object for writing frames

        Examples:
            ```python
            import supervision as sv
            import cv2

            source = sv.Video("source.mp4")

            # Custom resolution and FPS
            output_info = sv.VideoInfo(width=640, height=480, fps=60)

            with source.sink("output.mp4", info=output_info) as sink:
                for frame in source.frames():
                    # Custom processing
                    frame = cv2.resize(frame, (640, 480))
                    frame = cv2.flip(frame, 1)  # Horizontal flip
                    sink.write(frame)
            ```
        """
        if info is None:
            info = self.info

        writer = self._backend.writer(target_path, info, codec)

        try:
            # Create a sink wrapper
            sink = VideoSink(writer)
            yield sink
        finally:
            writer.close()


class VideoSink:
    """
    A wrapper for video writers providing a consistent interface.

    This class is typically used through the Video.sink() context manager.

    Attributes:
        writer: The underlying video writer
    """

    def __init__(self, writer: Writer):
        """
        Initialize the VideoSink.

        Args:
            writer: Backend-specific writer implementation
        """
        self.writer = writer

    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.

        Args:
            frame: Video frame in BGR format
        """
        self.writer.write(frame)

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video (alias for compatibility).

        Args:
            frame: Video frame in BGR format
        """
        self.write(frame)
