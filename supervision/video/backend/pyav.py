from __future__ import annotations

import platform
import re
from fractions import Fraction

try:
    import av
except ImportError:
    av = None
import numpy as np

from supervision.video.backend.base import BaseBackend, BaseWriter
from supervision.video.utils import SourceType, VideoInfo


class pyAVBackend(BaseBackend):
    """
    PyAV-based implementation of the `BaseBackend` interface for video processing.

    This backend provides video capture and frame reading capabilities using the PyAV
    library, which is a Pythonic binding for FFmpeg. It supports:
    - Local video files
    - Webcam streams (platform-specific)
    - RTSP network streams
    """

    def __init__(self):
        """
        Initialize the pyAVBackend instance.

        Raises:
            RuntimeError: If PyAV (`av` module) is not installed.
        """
        super().__init__()

        if av is None:
            raise RuntimeError(
                "PyAV (`av` module) is not installed. Run `pip install av`."
            )

        self.container = None
        self.stream = None
        self.writer = pyAVWriter
        self.frame_generator = None
        self.video_info = None
        self.current_frame_idx = 0

    def open(self, path: str | int) -> None:
        """
        Open and initialize a video source.

        This method opens a video file, RTSP stream, or webcam, and sets up
        the necessary components for decoding and reading frames.

        Args:
            path (str | int): Path to the video file, RTSP URL, or webcam path.

        Raises:
            RuntimeError: If the video source cannot be opened.
            ValueError: If the source type is unsupported.
        """
        _source_type = None
        _format = None

        def is_webcam_path(path: str) -> tuple[bool, str]:
            """
            Determine if the path refers to a webcam and get platform-specific format.

            Args:
                path (str): The path to check.

            Returns:
                tuple[bool, str]: (True if webcam, FFmpeg format string)
            """
            if not isinstance(path, str):
                return False, None

            system = platform.system()
            path_lower = path.lower()

            if system == "Windows":
                return path_lower.startswith("video="), "dshow"
            elif system == "Linux":
                return bool(re.match(r"^/dev/video\d+$", path_lower)), "v4l2"
            elif system == "Darwin":
                return path_lower.isdigit(), "avfoundation"
            else:
                return False, None

        isWebcam, ffmpeg_os_format = is_webcam_path(path=path)
        if isWebcam:
            _source_type = SourceType.WEBCAM
            _format = ffmpeg_os_format
        elif isinstance(path, str):
            _source_type = (
                SourceType.RTSP
                if path.lower().startswith("rtsp://")
                else SourceType.VIDEO_FILE
            )
        else:
            raise ValueError("Unsupported source type")

        try:
            self.container = av.open(path, format=_format)
            self.audio_src_container = self.container
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "AUTO"
            self.cap = self.container

            self.frame_generator = self.container.decode(video=0)
            self.video_info = self._set_video_info()
            self.current_frame_idx = 0

            # If audio exists
            if len(self.container.streams.audio) > 0:
                self.audio_stream = self.container.streams.audio[0]
            else:
                self.audio_stream = None

            self.video_info.SourceType = _source_type

        except Exception as e:
            raise RuntimeError(f"Cannot open video source: {path}") from e

    def isOpened(self) -> bool:
        """
        Check if the video source has been successfully opened.

        Returns:
            bool: True if video source is opened and ready, False otherwise.
        """
        return self.container is not None and self.stream is not None

    def _set_video_info(self) -> VideoInfo:
        """
        Extract and calculate video information from the opened source.

        Returns:
            VideoInfo: Object containing:
                - width (int): Frame width in pixels
                - height (int): Frame height in pixels
                - fps (int): Frames per second (estimated if not available)
                - total_frames (int | None): Total frame count if available

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        width = self.stream.width
        height = self.stream.height
        fps = float(self.stream.average_rate or self.stream.guessed_rate)
        if fps <= 0:
            fps = 30  # Default FPS if cannot be determined

        total_frames = self.stream.frames
        if total_frames == 0:
            total_frames = None  # Unknown frame count

        return VideoInfo(width, height, round(fps), total_frames)

    def info(self) -> VideoInfo:
        """
        Retrieve video information for the opened source.

        Returns:
            VideoInfo: Video properties including dimensions, FPS, and frame count.

        Raises:
            RuntimeError: If the video source is not opened.
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
                - np.ndarray: Frame data in BGR format with shape (height, width, 3)

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        try:
            frame = next(self.frame_generator)
            self.current_frame_idx += 1
            frame_bgr = frame.to_ndarray(format="bgr24")
            return True, frame_bgr
        except (StopIteration, av.error.EOFError):
            return False, np.array([])

    def grab(self) -> bool:
        """
        Advance to the next frame packet without decoding it.

        This is useful for quickly skipping frames when decoding isn't needed.

        Returns:
            bool: True if frame packet was advanced, False at end of stream

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        try:
            for packet in self.container.demux(video=0):
                if packet.stream.type == "video":
                    return True
            return False
        except (StopIteration, av.error.EOFError):
            return False

    def seek(self, frame_idx: int) -> None:
        """
        Seek to a specific frame index in the video.

        Uses keyframe-based seeking followed by sequential decoding to reach
        the exact requested frame. This is more efficient than sequential seeking
        but may be slower for very large jumps.

        Args:
            frame_idx (int): Zero-based index of the target frame.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        framerate = float(self.stream.average_rate or self.stream.guessed_rate or 30.0)
        if framerate <= 0:
            framerate = 30.0

        time_base = float(self.stream.time_base)
        timestamp = int((frame_idx / framerate) / time_base)

        self.container.seek(
            timestamp, stream=self.stream, any_frame=False, backward=True
        )
        self.frame_generator = self.container.decode(video=0)

        self.current_frame_idx = 0
        while True:
            try:
                frame = next(self.frame_generator)
            except (StopIteration, av.error.EOFError):
                break

            if getattr(frame, "time", None) is not None:
                self.current_frame_idx = round(frame.time * framerate)
            elif getattr(frame, "pts", None) is not None:
                self.current_frame_idx = round((frame.pts * time_base) * framerate)
            else:
                self.current_frame_idx += 1

            if self.current_frame_idx >= frame_idx:

                def _prepend_frame(first_frame, gen):
                    yield first_frame
                    yield from gen

                self.frame_generator = _prepend_frame(frame, self.frame_generator)
                break

    def release(self) -> None:
        """
        Release the video source and free all associated resources.

        This closes the video container and resets all internal state.
        Should be called when finished with the video source.
        """
        if self.container:
            self.container.close()
            self.container = None
            self.stream = None
            self.frame_generator = None


class pyAVWriter(BaseWriter):
    """
    PyAV-based video writer for creating video files with optional audio.

    This writer provides high-quality video encoding with precise frame timing
    (millisecond accuracy) and supports audio muxing from a source video.

    Methods:
        write(frame): Write a video frame.
        close(): Finalize and close the video file.
    """

    def __init__(
        self,
        filename: str,
        fps: int,
        frame_size: tuple[int, int],
        codec: str = "h264",
        backend: pyAVBackend | None = None,
        render_audio: bool | None = None,
    ):
        """
        Initialize the video writer.

        Args:
            filename (str): Path to the output video file.
            fps (int): Target frames per second for the output video.
            frame_size (tuple[int, int]): (width, height) of output frames.
            codec (str, optional): Video codec name (default "h264").
            backend (pyAVBackend, optional): Source backend for audio muxing.
            render_audio (bool, optional): Whether to include audio (default True if available).

        Raises:
            RuntimeError: If the output file cannot be created.
        """
        try:
            self.container = av.open(filename, mode="w")
            self.backend = backend

            if render_audio is None:
                render_audio = True

            if codec is None:
                codec = "h264"
            self.stream = self.container.add_stream(codec, rate=fps)
            self.stream.width = frame_size[0]
            self.stream.height = frame_size[1]
            self.stream.pix_fmt = "yuv420p"

            # Use finer time_base (1/1000) for millisecond precision timestamps
            self.stream.codec_context.time_base = Fraction(1, 1000)

            self.frame_idx = 0
            self.fps = fps  # Store FPS for timestamp calculations

            self.audio_stream_out = None
            self.audio_packets = []

            if (
                render_audio
                and backend
                and backend.audio_stream
                and backend.audio_src_container
            ):
                audio_codec_name = backend.audio_stream.codec_context.name
                audio_rate = backend.audio_stream.codec_context.rate
                self.audio_stream_out = self.container.add_stream(
                    audio_codec_name, rate=audio_rate
                )

                # Buffer all audio packets from backend for muxing later
                for packet in backend.audio_src_container.demux(backend.audio_stream):
                    if packet.dts is not None:
                        self.audio_packets.append(packet)

        except Exception as e:
            raise RuntimeError(f"Cannot open video writer for file: {filename}") from e

    def __enter__(self):
        """Enable use as a context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure proper cleanup when exiting context."""
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single video frame to the output file.

        Args:
            frame (np.ndarray): Frame data in BGR format (height, width, 3).
        """
        # Calculate PTS as milliseconds: frame_index * (1000 ms / fps)
        pts = int(self.frame_idx * (1000 / self.fps))

        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        av_frame.pts = pts
        av_frame.time_base = self.stream.codec_context.time_base
        self.frame_idx += 1

        packets = self.stream.encode(av_frame)
        for packet in packets:
            self.container.mux(packet)

    def close(self) -> None:
        """
        Finalize the video file, mux audio with adjusted timestamps to sync with video,
        and close the container.
        """
        def rescale_timestamp(value, src_tb, dst_tb):
            """
            Rescale timestamp between timebases.

            Args:
                value (int): Original timestamp value
                src_tb (Fraction): Source timebase
                dst_tb (Fraction): Destination timebase

            Returns:
                int: Rescaled timestamp
            """
            return int(value * src_tb / dst_tb)

        # Flush any remaining video packets
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

        # Calculate audio speed adjustment factor if needed
        speed_factor = 1.0

        try:
            if (
                self.backend
                and self.backend.audio_stream
                and self.backend.audio_stream.duration
            ):
                orig_audio_duration = float(
                    self.backend.audio_stream.duration
                    * self.backend.audio_stream.time_base
                )
            elif (
                self.backend
                and self.backend.audio_src_container
                and self.backend.audio_src_container.duration
            ):
                orig_audio_duration = self.backend.audio_src_container.duration / 1000
            else:
                orig_audio_duration = None

            new_video_duration = self.frame_idx * (1 / self.fps)

            if orig_audio_duration and new_video_duration > 0:
                speed_factor = orig_audio_duration / new_video_duration
        except Exception:
            speed_factor = 1.0

        # Process and mux audio packets with timestamp adjustments
        if self.audio_stream_out and speed_factor != 1.0:
            for packet in self.audio_packets:
                if packet.pts is not None:
                    packet.pts = rescale_timestamp(
                        packet.pts, packet.time_base, self.audio_stream_out.time_base
                    )
                    packet.pts = int(packet.pts / speed_factor)
                if packet.dts is not None:
                    packet.dts = rescale_timestamp(
                        packet.dts, packet.time_base, self.audio_stream_out.time_base
                    )
                    packet.dts = int(packet.dts / speed_factor)
                packet.stream = self.audio_stream_out
                self.container.mux(packet)
        elif self.audio_stream_out:
            for packet in self.audio_packets:
                if packet.pts is not None:
                    packet.pts = rescale_timestamp(
                        packet.pts, packet.time_base, self.audio_stream_out.time_base
                    )
                if packet.dts is not None:
                    packet.dts = rescale_timestamp(
                        packet.dts, packet.time_base, self.audio_stream_out.time_base
                    )
                packet.stream = self.audio_stream_out
                self.container.mux(packet)

        self.container.close()