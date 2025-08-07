"""Video backend implementations for the new Video API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Union

import cv2
import numpy as np

try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
    total number of frames.

    Attributes:
        width: Width of the video in pixels
        height: Height of the video in pixels
        fps: Frames per second of the video (as float for precision)
        total_frames: Total number of frames in the video, None for streams
        codec: Video codec used (e.g., 'h264', 'mjpeg')
        duration: Duration of the video in seconds, None for streams
        bit_rate: Bit rate of the video, None if unavailable
    """

    width: int
    height: int
    fps: float
    total_frames: int | None = None
    codec: str | None = None
    duration: float | None = None
    bit_rate: int | None = None

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Returns the resolution as (width, height) tuple."""
        return self.width, self.height

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        """
        Create VideoInfo from a video file path.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo object with video metadata

        Raises:
            Exception: If video cannot be opened
        """
        # Use OpenCV backend by default for backward compatibility
        backend = OpenCVBackend()
        handle = backend.open(video_path)
        info = backend.info(handle)
        backend.close(handle)
        return info


class Writer(Protocol):
    """Protocol for video writers."""

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        ...

    def close(self) -> None:
        """Close the writer and finalize the video file."""
        ...


class Backend(Protocol):
    """Protocol defining the interface for video backends."""

    def open(self, path: str | int) -> Any:
        """
        Open a video source.

        Args:
            path: Path to video file, RTSP URL, or device index for webcam

        Returns:
            Handle to the opened video source
        """
        ...

    def info(self, handle: Any) -> VideoInfo:
        """
        Get video information from an opened handle.

        Args:
            handle: Video handle from open()

        Returns:
            VideoInfo object with video metadata
        """
        ...

    def read(self, handle: Any) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame from the video.

        Args:
            handle: Video handle from open()

        Returns:
            Tuple of (success, frame) where frame is None if unsuccessful
        """
        ...

    def grab(self, handle: Any) -> bool:
        """
        Grab the next frame without decoding it.

        Args:
            handle: Video handle from open()

        Returns:
            True if frame was grabbed successfully
        """
        ...

    def seek(self, handle: Any, frame_idx: int) -> None:
        """
        Seek to a specific frame index.

        Args:
            handle: Video handle from open()
            frame_idx: Frame index to seek to
        """
        ...

    def writer(self, path: str, info: VideoInfo, codec: str | None = None) -> Writer:
        """
        Create a video writer.

        Args:
            path: Output file path
            info: Video information for output
            codec: Video codec to use (backend-specific format)

        Returns:
            Writer object for writing frames
        """
        ...

    def close(self, handle: Any) -> None:
        """
        Close the video handle.

        Args:
            handle: Video handle from open()
        """
        ...


class OpenCVWriter:
    """OpenCV-based video writer."""

    def __init__(self, path: str, info: VideoInfo, codec: str = "mp4v"):
        self.path = path
        self.info = info
        self.codec = codec
        self._writer = None
        self._initialize()

    def _initialize(self):
        """Initialize the OpenCV VideoWriter."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
        except TypeError:
            # Fallback to mp4v if codec is invalid
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._writer = cv2.VideoWriter(
            self.path,
            fourcc,
            int(self.info.fps),  # OpenCV requires integer FPS
            self.info.resolution_wh,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        if self._writer is not None:
            self._writer.write(frame)

    def close(self) -> None:
        """Close the writer and finalize the video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class OpenCVBackend:
    """OpenCV-based video backend implementation."""

    def open(self, path: str | int) -> cv2.VideoCapture:
        """Open a video source using OpenCV."""
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {path}")
        return video

    def info(self, handle: cv2.VideoCapture) -> VideoInfo:
        """Get video information from OpenCV VideoCapture."""
        width = int(handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = handle.get(cv2.CAP_PROP_FPS)  # Keep as float for precision
        total_frames = int(handle.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle invalid values for streams
        if total_frames <= 0:
            total_frames = None

        # Calculate duration if we have total frames and FPS
        duration = None
        if total_frames is not None and fps > 0:
            duration = total_frames / fps

        # Get codec information
        codec_int = int(handle.get(cv2.CAP_PROP_FOURCC))
        codec = None
        if codec_int > 0:
            codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

        # Get bit rate if available
        bit_rate = None
        try:
            bit_rate_val = handle.get(cv2.CAP_PROP_BITRATE)
            if bit_rate_val > 0:
                bit_rate = int(bit_rate_val)
        except:
            pass

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            codec=codec,
            duration=duration,
            bit_rate=bit_rate,
        )

    def read(self, handle: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from the video."""
        success, frame = handle.read()
        return success, frame if success else None

    def grab(self, handle: cv2.VideoCapture) -> bool:
        """Grab the next frame without decoding it."""
        return handle.grab()

    def seek(self, handle: cv2.VideoCapture, frame_idx: int) -> None:
        """Seek to a specific frame index."""
        handle.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def writer(self, path: str, info: VideoInfo, codec: str | None = None) -> Writer:
        """Create an OpenCV video writer."""
        if codec is None:
            codec = "mp4v"
        return OpenCVWriter(path, info, codec)

    def close(self, handle: cv2.VideoCapture) -> None:
        """Close the OpenCV VideoCapture."""
        handle.release()


if PYAV_AVAILABLE:

    class PyAVWriter:
        """PyAV-based video writer."""

        def __init__(self, path: str, info: VideoInfo, codec: str = "h264"):
            self.path = path
            self.info = info
            self.codec = codec
            self._container = None
            self._stream = None
            self._initialize()

        def _initialize(self):
            """Initialize the PyAV container and stream."""
            self._container = av.open(self.path, mode="w")
            self._stream = self._container.add_stream(self.codec, rate=self.info.fps)
            self._stream.width = self.info.width
            self._stream.height = self.info.height
            self._stream.pix_fmt = "yuv420p"

        def write(self, frame: np.ndarray) -> None:
            """Write a frame to the video."""
            if self._stream is not None:
                # Convert BGR to RGB (PyAV expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                for packet in self._stream.encode(av_frame):
                    self._container.mux(packet)

        def close(self) -> None:
            """Close the writer and finalize the video file."""
            if self._stream is not None:
                # Flush remaining packets
                for packet in self._stream.encode():
                    self._container.mux(packet)
            if self._container is not None:
                self._container.close()
                self._container = None
                self._stream = None

    class PyAVBackend:
        """PyAV-based video backend implementation."""

        def open(
            self, path: str | int
        ) -> tuple[av.container.InputContainer, av.video.stream.VideoStream]:
            """Open a video source using PyAV."""
            if isinstance(path, int):
                # Convert webcam index to device path
                # This is platform-specific and may need adjustment
                import platform

                system = platform.system()
                if system == "Linux":
                    path = f"/dev/video{path}"
                elif system == "Darwin":  # macOS
                    path = f"avfoundation:{path}"
                elif system == "Windows":
                    path = f"video={path}"
                else:
                    raise NotImplementedError(
                        f"Webcam support not implemented for {system}"
                    )

            container = av.open(path)
            video_stream = container.streams.video[0]
            return container, video_stream

        def info(
            self,
            handle: tuple[av.container.InputContainer, av.video.stream.VideoStream],
        ) -> VideoInfo:
            """Get video information from PyAV container."""
            container, stream = handle

            width = stream.width
            height = stream.height

            # Get FPS as float for precision
            if stream.average_rate:
                fps = float(stream.average_rate)
            elif stream.guessed_rate:
                fps = float(stream.guessed_rate)
            else:
                fps = 30.0  # Default fallback

            # Get total frames
            total_frames = stream.frames
            if total_frames == 0:
                # Try to calculate from duration
                if stream.duration and stream.time_base:
                    duration_sec = float(stream.duration * stream.time_base)
                    total_frames = int(duration_sec * fps)
                else:
                    total_frames = None

            # Get duration
            duration = None
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif total_frames and fps > 0:
                duration = total_frames / fps

            # Get codec name
            codec = stream.codec_context.name if stream.codec_context else None

            # Get bit rate
            bit_rate = stream.bit_rate if stream.bit_rate else None

            return VideoInfo(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                codec=codec,
                duration=duration,
                bit_rate=bit_rate,
            )

        def read(
            self,
            handle: tuple[av.container.InputContainer, av.video.stream.VideoStream],
        ) -> tuple[bool, np.ndarray | None]:
            """Read the next frame from the video."""
            container, stream = handle
            try:
                for frame in container.decode(stream):
                    # Convert to numpy array in BGR format (OpenCV convention)
                    img = frame.to_ndarray(format="rgb24")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    return True, img
                return False, None
            except av.error.EOFError:
                return False, None
            except StopIteration:
                return False, None

        def grab(
            self,
            handle: tuple[av.container.InputContainer, av.video.stream.VideoStream],
        ) -> bool:
            """Grab the next frame without fully decoding it."""
            container, stream = handle
            try:
                # PyAV doesn't have a direct grab equivalent, so we decode but don't convert
                for _ in container.decode(stream):
                    return True
                return False
            except (av.error.EOFError, StopIteration):
                return False

        def seek(
            self,
            handle: tuple[av.container.InputContainer, av.video.stream.VideoStream],
            frame_idx: int,
        ) -> None:
            """Seek to a specific frame index."""
            container, stream = handle
            # Convert frame index to timestamp
            timestamp = int(frame_idx / stream.average_rate * av.time_base)
            container.seek(timestamp, stream=stream)

        def writer(
            self, path: str, info: VideoInfo, codec: str | None = None
        ) -> Writer:
            """Create a PyAV video writer."""
            if codec is None:
                codec = "h264"
            return PyAVWriter(path, info, codec)

        def close(
            self,
            handle: tuple[av.container.InputContainer, av.video.stream.VideoStream],
        ) -> None:
            """Close the PyAV container."""
            container, _ = handle
            container.close()


def get_backend(backend_name: str | None = None) -> Backend:
    """
    Get a video backend by name.

    Args:
        backend_name: Name of the backend ('opencv', 'pyav', or None for auto-selection)

    Returns:
        Backend instance

    Raises:
        ValueError: If requested backend is not available
    """
    if backend_name is None:
        # Auto-select: prefer PyAV if available, otherwise OpenCV
        if PYAV_AVAILABLE:
            return PyAVBackend()
        else:
            return OpenCVBackend()
    elif backend_name.lower() == "opencv":
        return OpenCVBackend()
    elif backend_name.lower() == "pyav":
        if not PYAV_AVAILABLE:
            raise ValueError(
                "PyAV backend requested but av package is not installed. Install with: pip install av"
            )
        return PyAVBackend()
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: 'opencv', 'pyav'"
        )
