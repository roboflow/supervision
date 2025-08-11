from __future__ import annotations

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
    PyAV-based implementation of the `BaseBackend` interface.

    This backend handles video capture, frame reading, seeking, and writing
    operations using the PyAV library. Supports local video files, webcams,
    and RTSP streams.
    """

    def __init__(self):
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

    def open(self, path: str) -> None:
        """
        Open and initialize a video source.

        This method opens a video file, RTSP stream, or webcam, and sets up
        the necessary components for decoding and reading frames.

        Args:
            path (str | int): Path to the video file, RTSP URL, or webcam index.

        Raises:
            RuntimeError: If the video source cannot be opened.
            ValueError: If the source type is unsupported.
        """
        try:
            self.container = av.open(path)
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

        except Exception as e:
            raise RuntimeError(f"Cannot open video source: {path}") from e

    def isOpened(self) -> bool:
        """Check if the video source has been successfully opened."""
        return self.container is not None and self.stream is not None

    def _set_video_info(self) -> VideoInfo:
        """
        Extract video information from the opened source.

        Returns:
            VideoInfo: Object containing width, height, fps, and frame count.

        Raises:
            RuntimeError: If the video source is not opened.
        """
        if not self.isOpened():
            raise RuntimeError("Video not opened yet.")

        width = self.stream.width
        height = self.stream.height
        fps = float(self.stream.average_rate or self.stream.guessed_rate)
        if fps <= 0:
            fps = 30

        total_frames = self.stream.frames
        if total_frames == 0:
            total_frames = None

        return VideoInfo(width, height, round(fps), total_frames)

    def info(self) -> VideoInfo:
        """
        Retrieve video information.

        Returns:
            VideoInfo: Video properties for the opened source.

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
                - `bool`: True if a frame was read successfully, False if end of stream.
                - `np.ndarray`: Frame data in BGR format (H, W, 3).

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
        Grab the next frame packet without decoding it.

        Useful for skipping frames quickly without the overhead of decoding.

        Returns:
            bool: True if a frame packet was grabbed successfully, False otherwise.

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

        This uses keyframe-based seeking, then decodes forward to the exact
        requested frame.

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
        """
        if self.container:
            self.container.close()
            self.container = None
            self.stream = None
            self.frame_generator = None


class pyAVWriter(BaseWriter):
    """
    PyAV-based video writer.

    Writes frames to a video file with optional audio from a backend source.
    Uses finer timestamp granularity (milliseconds) for smoother video playback.
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
            fps (int): Frames per second for the output video.
            frame_size (tuple[int, int]): Width and height of the video frames.
            codec (str, optional): Video codec name (default "h264").
            backend (pyAVBackend, optional): Backend providing audio stream.

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
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the writer on context exit."""
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single video frame.

        Args:
            frame (np.ndarray): Frame data in BGR format (H, W, 3).
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
            Rescale timestamp value from source timebase to destination timebase.

            Args:
                value (int): Timestamp value (PTS or DTS).
                src_tb (Fraction): Source time base.
                dst_tb (Fraction): Destination time base.

            Returns:
                int: Rescaled timestamp.
            """
            return int(value * src_tb / dst_tb)

        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

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
